"""The request engine: sessions -> journeys -> requests -> hops, per shard.

A shard is a contiguous slice of simulated ticks. Sessions arriving in the
slice are generated atomically: their journeys may spill past the slice, so
the worker holds the latent trajectory for [t0, t1 + spill). Sessions advance
in lockstep over journey steps and client retry attempts; at every
(step, attempt) the due requests are grouped by BFF endpoint and each group's
request trees are evaluated vectorised — invocation top-down (a caller calls
a callee on the edge's share of requests, and a WARM cache skips a cached
call), outcomes and durations bottom-up (callee finals reach
the caller through the mechanism's worst-of aggregator; a hop's total time is
its own time plus the totals of the callees it invoked; SLOW is total > the
op's threshold). Every draw is a counter-keyed uniform (hashing.py), so the
same identifiers always draw the same numbers whatever was forced.

Outputs are flat tables (numpy arrays) consumed by emit.py and the oracle.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import KIND_BFF, KIND_CLIENT, KIND_EXTERNAL, KIND_SERVICE, OUTCOME_IDS, T_VALUES, Stream
from .hashing import D_CLIENT, D_HOP, D_REQUEST, D_SESSION, categorical, uniforms
from .intensity import tick_lambdas
from .latency import QUANTILE_PROBS, TIER_OF_KIND
from .latents import LatentSlice, Slots
from .mechanism import (
    CLASS_VALUES, HEALTH_VALUES, P_BFF_4XX_GIVEN_EXPIRED, P_CLIENT_ERR_GIVEN_BFF_FAILURE,
    P_CLIENT_ERR_GIVEN_FLAKY, POOL_VALUES, PROPAGATION, SEVERITY,
)
from .rng import shard_generator

# hop draw kinds
H_CLASS, H_OWN, H_HIT, H_RHO, H_RETRY, H_POD, H_STATUS, H_CALL = 0, 1, 2, 3, 4, 5, 6, 7
# session / request draw kinds
S_SCEN, S_NET, S_AUTH, S_LOGIN, S_MS, S_GAP, S_BACKOFF = 0, 1, 2, 3, 4, 5, 6
C_ERR, C_TAG, C_TXN = 0, 1, 2
from .latency import RETRY_BACKOFF_S
CLIENT_RETRY_BACKOFF_S = 1.0
MAX_STEP_GAP_S = 3600.0
OUTCOME_OK, OUTCOME_4XX, OUTCOME_5XX, OUTCOME_ERR, OUTCOME_SLOW, OUTCOME_ABSENT = 0, 1, 2, 3, 4, 5
STATUS_CHOICES = {OUTCOME_OK: (200, 200, 200, 201, 204), OUTCOME_4XX: (400, 400, 404, 403, 409),
                  OUTCOME_5XX: (500, 500, 503, 502, 500), OUTCOME_ERR: (0, 0, 0, 0, 0)}


@dataclass
class Table:
    """Column store built by appending equal-length column dicts."""
    cols: dict = field(default_factory=dict)
    _parts: list = field(default_factory=list)

    def append(self, part: dict):
        self._parts.append(part)

    def finalize(self):
        if not self._parts:
            return self
        keys = list(self._parts[0].keys())
        self.cols = {k: np.concatenate([np.asarray(p[k]) for p in self._parts]) for k in keys}
        self._parts = []
        return self

    def __len__(self):
        return len(next(iter(self.cols.values()))) if self.cols else sum(len(next(iter(p.values()))) for p in self._parts)


@dataclass
class ShardResult:
    shard: int
    t0: int
    t1: int
    sessions: dict            # column arrays, one row per session
    requests: Table           # one row per BFF attempt (= one edge request)
    hops: Table               # one row per op attempt within a request (BFF hop included)
    clients: Table            # one row per client attempt
    latents: LatentSlice
    spill_ticks: int


class Engine:
    def __init__(self, inst, seed, slots: Slots, forcings=()):
        self.inst = inst
        self.seed = int(seed)
        self.slots = slots
        self.topo = inst.topo
        self.mech = inst.mechanism
        self.cfg = inst.cfg
        self.constants = inst.constants
        self.lat = inst.latency
        self.forcings = list(forcings)
        self.tick_s = self.cfg.run.tick_s
        # event-level forcings (PRD scenario 21): op -> outcome index for the op's
        # final; op -> True/False for its invocation; scenario latents by name
        self.force_final = {}          # op -> outcome index (F:<op>)
        self.force_invoke = {}         # op -> True/False (I:<op>)
        self.force_attempt = {}        # (op, k) -> outcome index (A:<op>:<k>)
        self.force_session = {}        # "net" | "auth" -> value index
        self.force_bff = {}            # (scenario, step, attempt) -> outcome index (T:bff:...)
        self.force_bff_invoke = {}     # (scenario, step, attempt) -> True/False (I:bff:...)
        self.force_client = {}         # (scenario, step, attempt) -> 0 ok / 1 err / 2 absent (C:...)
        self.force_client_final = {}   # (scenario, step) -> 0 ok / 1 err (C:...:F)
        self.base_rps_override = None
        self._prepare()

    # --- static tables -------------------------------------------------------------------
    def _prepare(self):
        topo, mech, cfg = self.topo, self.mech, self.cfg
        n_ops = len(topo.ops)
        self.tier = np.array([TIER_OF_KIND.get(op.kind, "service") for op in topo.ops], dtype=object)
        self.knots = np.zeros((n_ops, len(QUANTILE_PROBS)))
        self.thr = np.full(n_ops, np.inf)
        for op in topo.ops:
            if op.id in self.lat.knots:
                self.knots[op.id] = self.lat.knots[op.id]
                self.thr[op.id] = self.lat.thresholds[op.id]
        # class CDFs per tier: [health][pool] -> cdf(4), and per propagation class
        self.class_cdf = {t: np.cumsum(mech.class_table[t], axis=-1) for t in mech.class_table}
        for t in self.class_cdf:
            self.class_cdf[t][..., -1] = 1.0
        self.prop_cdf = {}
        for w, pmf in PROPAGATION.items():
            c = np.cumsum([pmf["ok"], pmf["4xx"], pmf["5xx"], pmf["err"]])
            c[-1] = 1.0
            self.prop_cdf[w] = c
        self.backend_mask = np.array([o in cfg.endpoints.repertoire for o in T_VALUES[:5]])
        self.retry_on_backend = np.zeros(6, bool)
        for o in cfg.endpoints.retry.retry_on:
            self.retry_on_backend[T_VALUES.index(o)] = True
        self.R_backend = cfg.endpoints.retry.max_retries
        self.p_retry = self.constants["retry.p_retry_5xx"]
        self.has_cached_callee = np.zeros(n_ops, bool)
        for e in topo.edges:
            if e.cached:
                self.has_cached_callee[e.caller] = True
        self.reach = {b: topo.reachable_from(b) for b in topo.bff_ops}
        self.pods = {s.index: s.pods for s in topo.services}
        self.n_pods = np.array([max(1, len(topo.services[op.service].pods)) for op in topo.ops])
        # scenario tables
        self.scenarios = self.inst.sset.scenarios
        self.scen_cdf = np.cumsum([s.weight for s in self.scenarios])
        self.scen_cdf[-1] = 1.0
        self.max_steps = max(len(s.steps) for s in self.scenarios)
        self.max_client_retries = max(s.max_retries for s in self.scenarios)
        gap_q = [self.constants[f"session.step_gap_quantiles.{q}"] for q in ("p50", "p90", "p95", "p99")]
        self.gap_knots = np.array([0.5, gap_q[0], gap_q[1], gap_q[2], gap_q[3], min(3 * gap_q[3], MAX_STEP_GAP_S)])
        self.mult_deg, self.mult_tight, self.mult_exh, self.mult_cold = (
            self.lat.mult_degraded, self.lat.mult_pool_tight, self.lat.mult_pool_exhausted, self.lat.mult_cache_cold)

    # --- sessions -----------------------------------------------------------------------------
    def arrivals(self, t0, t1):
        """Poisson arrivals per tick from the shard's ARRIVALS stream (one call)."""
        lam = tick_lambdas(self.cfg, self.constants, t0, t1 - t0) * self.tick_s
        if self.base_rps_override is not None:
            lam = np.full(t1 - t0, float(self.base_rps_override) * self.tick_s)
        rng = shard_generator(self.seed, self._shard, Stream.ARRIVALS)
        return rng.poisson(lam)

    def run_shard(self, shard, t0, t1, latents: LatentSlice) -> ShardResult:
        self._shard = int(shard)
        counts = self.arrivals(t0, t1)
        n = int(counts.sum())
        ticks = np.repeat(np.arange(t0, t1), counts)
        local = np.arange(n)
        gid = (np.int64(shard) << np.int64(32)) | local.astype(np.int64)
        seed = self.seed
        u_scen = uniforms(seed, D_SESSION, gid, S_SCEN)
        scen = categorical(u_scen, np.tile(self.scen_cdf, (n, 1))) if n else np.zeros(0, int)
        net = (uniforms(seed, D_SESSION, gid, S_NET) < self.constants["client.net_flaky_share"]).astype(np.int8)
        auth = (uniforms(seed, D_SESSION, gid, S_AUTH) < self.constants["client.auth_expired_share"]).astype(np.int8)
        login = (uniforms(seed, D_SESSION, gid, S_LOGIN) < self.constants["client.logged_in_share"])
        if "net" in self.force_session:
            net[:] = self.force_session["net"]
        if "auth" in self.force_session:
            auth[:] = self.force_session["auth"]
        arrival_ms = ticks.astype(np.int64) * (self.tick_s * 1000) + (uniforms(seed, D_SESSION, gid, S_MS) * self.tick_s * 1000).astype(np.int64)
        sessions = {"gid": gid, "arrival_tick": ticks, "arrival_ms": arrival_ms, "scenario": scen,
                    "net": net, "auth": auth, "logged_in": login,
                    "steps_done": np.zeros(n, np.int16), "final_ok": np.zeros(n, bool)}
        requests, hops, clients = Table(), Table(), Table()
        self._req_counter = 0
        self._hop_counter = 0
        active = np.ones(n, bool)
        clock_ms = arrival_ms.copy()             # when the next request of each session may start
        n_steps = np.array([len(self.scenarios[s].steps) for s in scen], dtype=np.int64) if n else np.zeros(0, np.int64)
        max_retries = np.array([self.scenarios[s].max_retries for s in scen], dtype=np.int64) if n else np.zeros(0, np.int64)
        last_tick = t0
        for j in range(self.max_steps):
            if self.force_bff_invoke:
                forced_on = np.array([any(self.force_bff_invoke.get((int(s_), j, kk), None) is True
                                          for kk in range(self.max_client_retries + 1)) for s_ in scen], dtype=bool)
                active = active | (forced_on & (n_steps > j))
            due = active & (n_steps > j)
            if not due.any():
                if self.force_bff_invoke and any(v is True and key[1] > j for key, v in self.force_bff_invoke.items()):
                    continue
                break
            idx = np.nonzero(due)[0]
            step_ok = np.zeros(n, bool)
            for k in range(self.max_client_retries + 1):
                idx_k = idx[max_retries[idx] >= k] if k > 0 else idx
                if self.force_bff_invoke:
                    if k > 0:
                        forced_on = np.array([self.force_bff_invoke.get((int(s_), j, k), None) is True for s_ in scen], dtype=bool)
                        extra = np.nonzero(forced_on & (n_steps > j) & (max_retries >= k))[0]
                        idx_k = np.unique(np.concatenate([idx_k, extra])).astype(np.int64)
                    keep = np.array([self.force_bff_invoke.get((int(sessions["scenario"][i]), j, k), True) for i in idx_k], dtype=bool)
                    idx_k = idx_k[keep]
                if len(idx_k) == 0:
                    if any(v is True and key[1] == j and key[2] > k for key, v in self.force_bff_invoke.items()):
                        idx = np.zeros(0, np.int64)
                        continue
                    break
                start_ms = clock_ms[idx_k]
                tick = (start_ms // (self.tick_s * 1000)).astype(np.int64)
                last_tick = max(last_tick, int(tick.max()))
                out = self._evaluate_requests(idx_k, sessions, j, k, start_ms, tick, latents, requests, hops)
                bff_out, end_ms = out["outcome"], out["end_ms"]
                # client outcome
                u_err = uniforms(seed, D_CLIENT, sessions["gid"][idx_k], j, k, C_ERR)
                fail = np.isin(bff_out, [OUTCOME_4XX, OUTCOME_5XX, OUTCOME_ERR])
                p_err = np.where(fail, P_CLIENT_ERR_GIVEN_BFF_FAILURE, 0.0)
                flaky = sessions["net"][idx_k] == 1
                p_err = p_err + (1 - p_err) * np.where(flaky, P_CLIENT_ERR_GIVEN_FLAKY, 0.0)
                mask_err = np.array([("err" in self.scenarios[s].client_mask) for s in sessions["scenario"][idx_k]])
                client_err = (u_err < p_err) & mask_err
                if self.force_client:
                    fc = np.array([self.force_client.get((int(s_), j, k), -1) for s_ in sessions["scenario"][idx_k]])
                    client_err = np.where(fc >= 0, fc == 1, client_err)
                clients.append({
                    "session_row": idx_k, "gid": sessions["gid"][idx_k], "step": np.full(len(idx_k), j, np.int16),
                    "attempt": np.full(len(idx_k), k, np.int16), "client_op": out["client_op"], "bff_op": out["bff_op"],
                    "request_row": out["request_row"], "ts_ms": end_ms + 40, "outcome": np.where(client_err, 1, 0).astype(np.int8),
                    "bff_outcome": bff_out, "tagged": uniforms(seed, D_CLIENT, sessions["gid"][idx_k], j, k, C_TAG) >= self.constants["client.auto_captured_share"],
                    "sampled_txn": uniforms(seed, D_CLIENT, sessions["gid"][idx_k], j, k, C_TXN) < self.constants["client.txn_sampling"],
                })
                clock_ms[idx_k] = end_ms + int(CLIENT_RETRY_BACKOFF_S * 1000)
                ok_now = ~client_err
                step_ok[idx_k[ok_now]] = True
                # retry: only sessions whose BFF outcome is in the scenario's retry set, with prob p_retry
                retry_set = np.array([bff_out[i] != OUTCOME_ABSENT and T_VALUES[bff_out[i]] in self.scenarios[sessions["scenario"][idx_k[i]]].retry_on
                                      for i in range(len(idx_k))], dtype=bool)
                u_r = uniforms(seed, D_CLIENT, sessions["gid"][idx_k], j, k, 7)
                will_retry = (~ok_now) & retry_set & (u_r < self.p_retry) & (max_retries[idx_k] > k)
                idx = idx_k[will_retry]
                if len(idx) == 0 and not any(v is True and key[1] == j and key[2] > k for key, v in self.force_bff_invoke.items()):
                    break
            done = np.nonzero(due)[0]
            sessions["steps_done"][done] += 1
            if self.force_client_final:
                ff = np.array([self.force_client_final.get((int(s_), j), -1) for s_ in sessions["scenario"][done]])
                step_ok[done] = np.where(ff >= 0, ff == 0, step_ok[done])
            active[done] = step_ok[done]
            # gap before the next step
            gap_u = uniforms(seed, D_SESSION, sessions["gid"][done], S_GAP, j)
            gap_s = np.interp(gap_u, QUANTILE_PROBS, self.gap_knots)
            clock_ms[done] = clock_ms[done] + (gap_s * 1000).astype(np.int64)
        sessions["final_ok"] = active & (sessions["steps_done"] >= n_steps)
        spill = max(0, last_tick + 1 - t1)
        return ShardResult(shard, t0, t1, sessions, requests.finalize(), hops.finalize(), clients.finalize(), latents, spill)

    # --- request trees ----------------------------------------------------------------------------
    def _evaluate_requests(self, idx, sessions, step, attempt, start_ms, tick, latents, requests, hops):
        """Evaluate the BFF attempts of the given sessions; returns per-session outcome arrays."""
        n = len(idx)
        scen = sessions["scenario"][idx]
        bff_ops = np.array([self.scenarios[s].steps[step].bff_op for s in scen])
        client_ops = np.array([self.scenarios[s].steps[step].client_op for s in scen])
        outcome = np.full(n, OUTCOME_ABSENT, np.int8)
        end_ms = start_ms.copy()
        req_rows = np.zeros(n, np.int64)
        for b in np.unique(bff_ops):
            sel = np.nonzero(bff_ops == b)[0]
            res = self._evaluate_tree(int(b), idx[sel], sessions, step, attempt, start_ms[sel], tick[sel], latents, requests, hops)
            outcome[sel] = res["outcome"]
            end_ms[sel] = res["end_ms"]
            req_rows[sel] = res["request_row"]
        return {"outcome": outcome, "end_ms": end_ms, "request_row": req_rows, "bff_op": bff_ops, "client_op": client_ops}

    def _state_at(self, latents: LatentSlice, tick, op):
        i = tick - latents.t0
        svc_slot = self.slots.svc_slot_of_op[op]
        return (latents.health[i, self.slots.slot_of_op[op]], latents.pool[i, svc_slot], latents.cache[i, svc_slot],
                latents.load[i, svc_slot], latents.intensity[i])

    def _evaluate_tree(self, b, idx, sessions, step, attempt, start_ms, tick, latents, requests, hops):
        """All requests in this group share the BFF op `b`."""
        seed = self.seed
        n = len(idx)
        gid = sessions["gid"][idx]
        req_gid = (gid << np.int64(8)) | np.int64(step * 16 + attempt)    # unique per (session, step, attempt)
        order = self.reach[b]                                              # BFS (top-down)
        pos = {op: i for i, op in enumerate(order)}
        # --- invocation top-down ---
        invoked = {b: np.ones(n, bool)}
        parent_of = {b: None}
        for v in order[1:]:
            inv = np.zeros(n, bool)
            parent = np.full(n, -1, np.int64)
            for e in self.topo.caller_edges(v):
                u = e.caller
                if u not in invoked:
                    continue
                # v is invoked through e: u is invoked, u calls v on this request,
                # and a cache does not answer the call
                calls = np.ones(n, bool)
                if e.p_call < 1.0:
                    calls = uniforms(seed, D_HOP, req_gid, e.index, H_CALL) < e.p_call
                if e.cached:
                    cache_state = self._state_at(latents, tick, u)[2]
                    u_hit = uniforms(seed, D_HOP, req_gid, e.index, H_HIT)
                    calls &= ~((cache_state == 0) & (u_hit < e.p_hit))
                via = invoked[u] & calls & ~inv
                parent[via] = u
                inv |= invoked[u] & calls
            if v in self.force_invoke:
                if self.force_invoke[v]:
                    # present: invoked in every request of this tree (the held
                    # context says so); the record's caller is the first caller
                    # present, else the first caller in edge order
                    inv = np.ones(n, bool)
                    first = next((e.caller for e in self.topo.caller_edges(v) if e.caller in invoked), -1)
                    parent = np.full(n, first, np.int64)
                    for e in self.topo.caller_edges(v):
                        if e.caller in invoked:
                            parent = np.where(invoked[e.caller] & (parent == first), e.caller, parent)
                else:
                    inv = np.zeros(n, bool)
            invoked[v] = inv
            parent_of[v] = parent
        # --- outcomes bottom-up ---
        final = {}          # op -> outcome per request (OUTCOME_ABSENT if not invoked)
        total_ms = {}       # op -> total duration incl. all attempts and callees
        attempts_of = {}    # op -> list of per-attempt dicts
        scen_arr = sessions["scenario"][idx]
        for v in reversed(order):
            op = self.topo.ops[v]
            inv = invoked[v]
            health, pool, cache, load, inten = self._state_at(latents, tick, v)
            svc_slot = self.slots.svc_slot_of_op[v]
            # worst callee class (post-mask) per request
            worst = np.zeros(n, np.int8)       # severity 0..3
            callee_time = np.zeros(n)
            for e in self.topo.callee_edges(v):
                w = e.callee
                if w not in final:
                    continue
                f = final[w]
                sev = np.array([SEVERITY[T_VALUES[x]] for x in f], dtype=np.int8) * (f != OUTCOME_ABSENT)
                if e.critical:
                    u_rho = uniforms(seed, D_HOP, req_gid, e.index, H_RHO)
                    reach = (u_rho < e.rho) & (sev > 0)
                    worst = np.maximum(worst, np.where(reach, sev, 0))
                callee_time += total_ms[w]
            cold = self.has_cached_callee[v] & (cache == 1)
            mult = np.ones(n)
            mult *= np.where(health == 1, self.mult_deg, 1.0)
            mult *= np.where(pool == 1, self.mult_tight, 1.0)
            mult *= np.where(pool == 2, self.mult_exh, 1.0)
            mult *= np.where(cold, self.mult_cold, 1.0)
            tier = self.tier[v]
            R = 0 if op.kind == KIND_BFF else self.R_backend
            is_bff = op.kind == KIND_BFF
            mask = self.backend_mask
            if is_bff:
                allowed = self.scenarios[int(scen_arr[0])].bff_mask if n else []
                mask = np.array([o in allowed for o in T_VALUES[:5]])
            present = inv.copy()
            prev_out = np.full(n, OUTCOME_ABSENT, np.int8)
            fin = np.full(n, OUTCOME_ABSENT, np.int8)
            t_total = np.zeros(n)
            att_list = []
            for k in range(R + 1):
                if k > 0:
                    u_r = uniforms(seed, D_HOP, req_gid, v, H_RETRY, k)
                    present = self.retry_on_backend[prev_out] & (u_r < self.p_retry) & (prev_out != OUTCOME_ABSENT)
                if not present.any() and not any(key[0] == v and key[1] >= k for key in self.force_attempt):
                    break
                # class
                base_cdf = self.class_cdf[tier][health, pool]                 # [n, 4]
                u_c = uniforms(seed, D_HOP, req_gid, v, H_CLASS, k)
                cls = categorical(u_c, base_cdf)
                for sev, wname in ((1, "4xx"), (2, "5xx"), (3, "err")):
                    m = worst == sev
                    if m.any():
                        cls[m] = categorical(u_c[m], np.tile(self.prop_cdf[wname], (int(m.sum()), 1)))
                if is_bff:
                    expired = sessions["auth"][idx] == 1
                    u_a = uniforms(seed, D_HOP, req_gid, v, H_STATUS, k)
                    cls = np.where(expired & (u_a < P_BFF_4XX_GIVEN_EXPIRED), 1, cls)
                # repertoire mask: a masked class falls back to ok
                cls = np.where(mask[cls], cls, 0)
                # own time and total
                u_o = uniforms(seed, D_HOP, req_gid, v, H_OWN, k)
                own = np.interp(u_o, QUANTILE_PROBS, self.knots[v]) * mult
                total = own + (callee_time / 1000.0 if k == 0 else RETRY_BACKOFF_S)
                if v in self.force_final and k == 0:
                    # a forced final class places the hop's total in that class's duration band
                    fv = int(self.force_final[v])
                    thr = self.thr[v]
                    if fv == OUTCOME_SLOW:
                        total = np.maximum(total, thr * 1.0001)
                    elif fv != OUTCOME_ABSENT:
                        total = np.minimum(total, thr)
                    own = np.maximum(total - (callee_time / 1000.0 if k == 0 else RETRY_BACKOFF_S), 0.0)
                out_k = cls.astype(np.int8)
                slow = (out_k == 0) & mask[OUTCOME_SLOW] & (total > self.thr[v])
                out_k = np.where(slow, OUTCOME_SLOW, out_k).astype(np.int8)
                out_k = np.where(present, out_k, OUTCOME_ABSENT).astype(np.int8)
                if (v, k) in self.force_attempt:
                    fv = int(self.force_attempt[(v, k)])
                    present = np.zeros(n, bool) if fv == OUTCOME_ABSENT else inv.copy()
                    out_k = np.where(present, fv, OUTCOME_ABSENT).astype(np.int8)
                if is_bff and self.force_bff:
                    fb = np.array([self.force_bff.get((int(sc), step, attempt), -1) for sc in scen_arr])
                    present = present & (fb != OUTCOME_ABSENT)
                    out_k = np.where(present & (fb >= 0), fb, np.where(present, out_k, OUTCOME_ABSENT)).astype(np.int8)
                if not present.any():
                    break
                total_k = np.where(present, total, 0.0)
                att_list.append({"present": present.copy(), "outcome": out_k, "own_s": np.where(present, own, 0.0), "total_s": total_k})
                fin = np.where(present, out_k, fin).astype(np.int8)
                t_total += total_k
                prev_out = out_k
            if v in self.force_final:
                fin = np.where(inv, self.force_final[v], OUTCOME_ABSENT).astype(np.int8)
            final[v] = fin
            total_ms[v] = t_total * 1000.0
            attempts_of[v] = att_list
        # --- timing top-down, then rows ---
        start_of = {b: start_ms.astype(np.float64)}
        # siblings are called sequentially in edge order after half of the caller's own time
        for v in order:
            if not attempts_of[v]:
                continue
            if v not in start_of:
                start_of[v] = start_ms.astype(np.float64)      # invoking caller had no attempt of its own (forced)
            t_cursor = start_of[v] + 0.5 * attempts_of[v][0]["own_s"] * 1000.0
            for e in self.topo.callee_edges(v):
                w = e.callee
                if w not in start_of:
                    start_of[w] = t_cursor.copy()
                else:
                    start_of[w] = np.where(parent_of[w] == v, t_cursor, start_of[w])
                t_cursor = t_cursor + np.where(parent_of[w] == v, total_ms[w], 0.0)
        request_row0 = self._req_counter
        req_rows = np.arange(request_row0, request_row0 + n)
        self._req_counter += n
        bff_final = final[b]
        bff_end = start_ms + np.rint(total_ms[b]).astype(np.int64)
        requests.append({
            "request_row": req_rows, "req_gid": req_gid, "session_row": idx, "gid": gid,
            "step": np.full(n, step, np.int16), "attempt": np.full(n, attempt, np.int16), "bff_op": np.full(n, b, np.int64),
            "start_ms": start_ms, "end_ms": bff_end, "outcome": bff_final, "tick": tick,
            "n_hops": np.zeros(n, np.int32),
        })
        hop_rows_of = {}
        n_hops = np.zeros(n, np.int32)
        for v in order:
            if not attempts_of[v] or v not in start_of:
                continue
            op = self.topo.ops[v]
            health, pool, cache, load, inten = self._state_at(latents, tick, v)
            t_start = start_of[v]
            for k, att in enumerate(attempts_of[v]):
                pres = att["present"]
                m = np.nonzero(pres)[0]
                if len(m) == 0:
                    continue
                rows = np.arange(self._hop_counter, self._hop_counter + len(m))
                self._hop_counter += len(m)
                if k == 0:
                    hop_rows_of[v] = np.full(n, -1, np.int64)
                hop_rows_of[v][m] = rows if k == 0 else hop_rows_of[v][m]      # the first attempt is the tree node
                par = parent_of[v]
                parent_hop = np.full(len(m), -1, np.int64)
                if par is not None:
                    for i, r in enumerate(m):
                        pu = int(par[r])
                        if pu >= 0 and pu in hop_rows_of:
                            parent_hop[i] = hop_rows_of[pu][r]
                u_pod = uniforms(seed, D_HOP, req_gid[m], v, H_POD, k)
                u_st = uniforms(seed, D_HOP, req_gid[m], v, H_STATUS, k + 8)
                outc = att["outcome"][m]
                status = np.array([STATUS_CHOICES[o if o != OUTCOME_SLOW else OUTCOME_OK][int(u * 5)] for o, u in zip(outc, u_st)], dtype=np.int32)
                s_ms = (t_start[m] + (0 if k == 0 else 0)).astype(np.int64)
                if k > 0:
                    prev_total = sum(a["total_s"][m] for a in attempts_of[v][:k]) * 1000.0
                    s_ms = s_ms + np.rint(prev_total).astype(np.int64)
                d_ms = np.rint(att["total_s"][m] * 1000.0).astype(np.int64)
                hops.append({
                    "hop_row": rows, "request_row": req_rows[m], "req_gid": req_gid[m], "gid": gid[m],
                    "op": np.full(len(m), v, np.int64), "attempt": np.full(len(m), k, np.int16),
                    "parent_hop": parent_hop, "parent_op": (par[m] if par is not None else np.full(len(m), -1, np.int64)),
                    "start_ms": s_ms, "end_ms": s_ms + d_ms, "duration_ms": d_ms, "own_ms": np.rint(att["own_s"][m] * 1000.0).astype(np.int64),
                    "own_us": np.rint(att["own_s"][m] * 1e6).astype(np.int64), "duration_us": np.rint(att["total_s"][m] * 1e6).astype(np.int64),
                    "outcome": outc, "status": status, "pod": (u_pod * self.n_pods[v]).astype(np.int16),
                    "tick": tick[m], "state_health": health[m], "state_pool": pool[m], "state_cache": cache[m],
                    "state_load": load[m], "state_intensity": inten[m], "is_edge": np.full(len(m), op.kind == KIND_BFF),
                })
                n_hops[m] += 1
        requests._parts[-1]["n_hops"] = n_hops
        return {"outcome": bff_final, "end_ms": bff_end, "request_row": req_rows}
