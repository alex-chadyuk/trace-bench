"""The mechanism: state variables, conditional distributions and by-construction strengths.

A two-timescale dynamic Bayesian network. Tick-level latents (intensity, load,
pool, cache, endpoint health) evolve once per simulated tick with lag-1
self-loops; session-level latents (client network / credential state) are
drawn per journey; event-level variables (invocation, attempt outcome, final
outcome, client outcome) are drawn per request. Observables never point into
tick-level latents, so the projected directed graph is acyclic.

Every node exposes `dist(ctx)` — its distribution given a mapping parent id ->
value name — built from the same tables and samplers the engine uses to
generate data. The strength of an edge P -> C is the maximum over pairs of P's
values of the total-variation distance between C's distributions, with C's
other parents held at the node's nominal context (D-TB-3); a retry attempt is
held ABSENT in the nominal context of the operation's final and of the attempt
that follows it, so a first-attempt strength is the controlled direct effect
with no retry (D-TB-20). Duration-mediated
probabilities (the SLOW class) are Monte Carlo estimates from a dedicated,
order-independent random stream (`Stream.LATENCY`, keyed by op and context).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .constants import (
    CLIENT_T_VALUES, INSTANTIATE_SHARD, KIND_BFF, KIND_CLIENT, KIND_EXTERNAL,
    KIND_SERVICE, SCOPE_INSTANTIATE, T_VALUES, Stream,
)
from .latency import TIER_OF_KIND, LatencyModel
from .rng import stable_id
from .tables import build_class_tables

# --- value sets (nominal = index 0) ------------------------------------------------
INTENSITY_VALUES = ("day", "night", "peak")
LOAD_VALUES = ("mid", "low", "high")
POOL_VALUES = ("free", "tight", "exhausted")
CACHE_VALUES = ("warm", "cold")
HEALTH_VALUES = ("healthy", "degraded", "failed")
NET_VALUES = ("good", "flaky")
AUTH_VALUES = ("valid", "expired")
PRESENCE_VALUES = ("present", "absent")
CLASS_VALUES = ("ok", "4xx", "5xx", "err")          # the categorical part of an attempt
SEVERITY = {"ok": 0, "slow": 0, "4xx": 1, "5xx": 2, "err": 3, "absent": 0}

# Structural propagation tables: a critical callee failure that reaches the
# caller maps onto the caller's own class like this.
PROPAGATION = {
    "4xx": {"ok": 0.10, "4xx": 0.80, "5xx": 0.10, "err": 0.00},
    "5xx": {"ok": 0.05, "4xx": 0.00, "5xx": 0.85, "err": 0.10},
    "err": {"ok": 0.05, "4xx": 0.00, "5xx": 0.35, "err": 0.60},
}
P_CLIENT_ERR_GIVEN_BFF_FAILURE = 0.98
P_CLIENT_ERR_GIVEN_FLAKY = 0.05
P_BFF_4XX_GIVEN_EXPIRED = 0.90
LOAD_TARGET = {  # intensity -> pmf over LOAD_VALUES (mid, low, high)
    "day": (0.70, 0.15, 0.15), "night": (0.25, 0.70, 0.05), "peak": (0.30, 0.00, 0.70),
}


@dataclass
class StateVar:
    id: str
    group: str                 # intensity | load | pool | cache | health | net | auth | invoke | attempt | final | client | client_final
    kind: str                  # tick | session | event
    values: tuple
    latent: bool
    derived: bool = False      # a deterministic function of observables; carries no token of its own
    token_op: int | None = None
    lag_self: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def nominal(self):
        return self.values[0]

    def to_dict(self):
        d = dict(vars(self))
        d["values"] = list(self.values)
        return d


class Node:
    """A variable with parents and a conditional distribution."""

    def __init__(self, var: StateVar, parents: list[str], context_override=None):
        self.var = var
        self.parents = list(parents)
        self.context_override = dict(context_override or {})

    def dist(self, ctx) -> np.ndarray:
        raise NotImplementedError

    def nominal_context(self, mech, exclude=None):
        """The context in which the edge `exclude -> self` is measured: every
        other parent at its nominal value, subject to the node's overrides and
        to `context_for(exclude)` (a parent-specific adjustment, e.g. a cache
        latent is only testable while one of its callers is present)."""
        ctx = {}
        for p in self.parents:
            if p == exclude:
                continue
            ctx[p] = self.context_override.get(p, mech.nodes[p].var.nominal)
        if exclude is not None:
            ctx.update(self.context_for(exclude))
        return ctx

    def context_for(self, parent_id):
        return {}

    def dist_held(self, mech, ctx):
        """Distribution of this variable when its parents are HELD at `ctx`:
        the stationary distribution of the lag-1 chain for tick-level nodes
        with a self-loop (a parent held for hours, not a one-tick impulse),
        the plain conditional otherwise."""
        if not self.var.lag_self:
            return self.dist(ctx)
        # Memoised per held context (D-TB-17). The projection asks for the same
        # (node, context) once per particle of every effect that passes through
        # the node — 11 M requests for 255 k distinct contexts at m — and each
        # cold evaluation is a power iteration to 1e-13. The same function on
        # the same inputs, so the cache changes no byte; the cached array is
        # read-only so no caller can alter it in place.
        try:
            key = tuple(sorted(ctx.items()))
        except TypeError:
            return self._held(ctx)
        cache = self.__dict__.setdefault("_held_cache", {})
        out = cache.get(key)
        if out is None:
            out = self._held(ctx)
            out.flags.writeable = False
            cache[key] = out
        return out

    def _held(self, ctx):
        values = self.var.values
        n = len(values)
        K = np.zeros((n, n))
        c = dict(ctx)
        for i, v in enumerate(values):
            c[self.var.id] = v
            K[i] = self.dist(c)
        return stationary(K)


def stationary(K, iters=100000, tol=1e-13):
    """Stationary distribution of a row-stochastic matrix by power iteration
    (deterministic; small matrices)."""
    n = K.shape[0]
    pi = np.full(n, 1.0 / n)
    for _ in range(iters):
        nxt = pi @ K
        if np.abs(nxt - pi).max() < tol:
            pi = nxt
            break
        pi = nxt
    pi = np.clip(pi, 0.0, None)
    return pi / pi.sum()


def cache_table(constants):
    """[load][cache] -> pmf(cache'): the per-tick cache transition table (a WARM
    cache turns COLD at the high-load hazard under HIGH load, a tenth of it
    otherwise; a COLD cache stays COLD with the persistence probability)."""
    hc, pc = constants["state_dynamics.cache_cold_given_high"], constants["state_dynamics.cache_persist_cold"]
    table = np.zeros((3, 2, 2))
    for li, load in enumerate(LOAD_VALUES):
        high = load == "high"
        table[li, 0] = (1 - (hc if high else hc * 0.1), hc if high else hc * 0.1)
        table[li, 1] = (1 - pc, pc)
    return table


class TableNode(Node):
    """Explicit table: `table[idx(parent_1), ..., idx(parent_m)] -> pmf`."""

    def __init__(self, var, parents, table, context_override=None):
        super().__init__(var, parents, context_override)
        self.table = np.asarray(table, dtype=np.float64)

    def dist(self, ctx):
        idx = tuple(_value_index(_PARENT_VALUES_CACHE[p], ctx[p]) for p in self.parents)
        return self.table[idx]


class FunctionNode(Node):
    def __init__(self, var, parents, fn, context_override=None, context_for=None):
        super().__init__(var, parents, context_override)
        self.fn = fn
        self._context_for = context_for or {}

    def dist(self, ctx):
        return np.asarray(self.fn(ctx), dtype=np.float64)

    def context_for(self, parent_id):
        return dict(self._context_for.get(parent_id, {}))


_PARENT_VALUES_CACHE: dict[str, tuple] = {}


def _value_index(values, name):
    return values.index(name)


def tv(p, q):
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


# =====================================================================================
class Mechanism:
    def __init__(self, cfg, constants, topo, sset, latency: LatencyModel, seed):
        self.cfg = cfg
        self.constants = constants
        self.topo = topo
        self.sset = sset
        self.latency = latency
        self.seed = int(seed)
        self.nodes: dict[str, Node] = {}
        self.order: list[str] = []
        self._pslow_cache: dict = {}
        self.backend_retries = cfg.endpoints.retry.max_retries
        self.backend_retry_on = tuple(cfg.endpoints.retry.retry_on)
        self.backend_mask = frozenset(cfg.endpoints.repertoire)
        self.p_retry = constants["retry.p_retry_5xx"]
        self._build_tables()
        self._build_nodes()
        _PARENT_VALUES_CACHE.clear()
        for nid, n in self.nodes.items():
            _PARENT_VALUES_CACHE[nid] = n.var.values

    # --- tables ---------------------------------------------------------------------
    def _build_tables(self):
        c = self.constants
        d = c["state_dynamics.load_persist"]
        self.load_table = np.zeros((3, 3, 3))
        for i, inten in enumerate(INTENSITY_VALUES):
            for j in range(3):
                target = np.array(LOAD_TARGET[inten])
                row = (1 - d) * target
                row[j] += d
                self.load_table[i, j] = row / row.sum()
        ht, he, rp = c["state_dynamics.pool_tight_given_high"], c["state_dynamics.pool_exhausted_given_high"], c["state_dynamics.pool_recover"]
        self.pool_table = np.zeros((3, 3, 3))          # [load][pool] -> pmf(pool')
        for li, load in enumerate(LOAD_VALUES):
            high = load == "high"
            low = load == "low"
            self.pool_table[li, 0] = (1 - (ht if high else ht * 0.1), ht if high else ht * 0.1, 0.0)
            self.pool_table[li, 1] = (rp * (2.0 if low else 1.0), 1 - rp * (2.0 if low else 1.0) - (he if high else 0.0), he if high else 0.0)
            self.pool_table[li, 2] = (0.0, rp if not high else rp * 0.3, 1 - (rp if not high else rp * 0.3))
        self.cache_table = cache_table(c)              # [load][cache] -> pmf(cache')
        hd_e, hd_b, hf, hr = (c["state_dynamics.health_degraded_given_exhausted"], c["state_dynamics.health_degraded_base"],
                              c["state_dynamics.health_failed_given_degraded"], c["state_dynamics.health_recover"])
        self.health_table = np.zeros((3, 3, 3))        # [pool][health] -> pmf(health')
        for pi, pool in enumerate(POOL_VALUES):
            exh = pool == "exhausted"
            hd = hd_e if exh else hd_b
            self.health_table[pi, 0] = (1 - hd, hd, 0.0)
            rec = hr * (0.0 if exh else 1.0)
            self.health_table[pi, 1] = (rec, 1 - rec - hf, hf)
            self.health_table[pi, 2] = (0.0, hr, 1 - hr)
        self.class_table = build_class_tables(self.cfg, self.constants)

    # --- helpers shared with the engine ----------------------------------------------
    def tier_of(self, op_id):
        return TIER_OF_KIND[self.topo.ops[op_id].kind]

    def class_pmf(self, op_id, health, pool, worst):
        """pmf over CLASS_VALUES given own state and the post-mask worst callee class."""
        base = self.class_table[self.tier_of(op_id)][health, pool]
        if worst in ("ok", "slow", "absent"):
            return base
        p = PROPAGATION[worst]
        return np.array([p["ok"], p["4xx"], p["5xx"], p["err"]])

    def worst_pmf(self, op_id, callee_classes):
        """Distribution of the post-mask worst class over ('ok','4xx','5xx','err')
        given callee final classes {callee_op: class}. Optional callees never
        reach; a critical callee failure reaches with probability rho."""
        reach = {1: [], 2: [], 3: []}
        for e in self.topo.callee_edges(op_id):
            cls = callee_classes.get(e.callee, "ok")
            sev = SEVERITY[cls]
            if sev > 0 and e.critical:
                reach[sev].append(e.rho)
        out = np.zeros(4)
        p_none_higher = 1.0
        for sev in (3, 2, 1):
            p_none_this = float(np.prod([1 - r for r in reach[sev]])) if reach[sev] else 1.0
            out[sev] = p_none_higher * (1 - p_none_this)
            p_none_higher *= p_none_this
        out[0] = p_none_higher
        return out

    def p_slow(self, op_id, health, pool, cache_cold, callee_classes, callee_retry=None):
        key = (op_id, health, pool, bool(cache_cold), tuple(sorted(callee_classes.items())),
               tuple(sorted((callee_retry or {}).items())))
        if key not in self._pslow_cache:
            ss = np.random.SeedSequence([self.seed, SCOPE_INSTANTIATE, INSTANTIATE_SHARD & 0xFFFFFFFF,
                                         int(Stream.LATENCY), int(op_id), int(stable_id(key)[:8], 16)])
            rng = np.random.Generator(np.random.PCG64(ss))
            self._pslow_cache[key] = self.latency.p_slow(
                rng, op_id, health=health, pool=pool, cache_miss=cache_cold,
                callee_classes=callee_classes, n=self.cfg.mechanism.mc_samples, callee_retry=callee_retry)
        return self._pslow_cache[key]

    def callee_retry_probs(self, first_attempts):
        """{callee: P(retry)} from the callees' first-attempt classes in the context."""
        out = {}
        for w, cls in first_attempts.items():
            out[w] = self.p_retry if (cls in self.backend_retry_on and self.backend_retries > 0) else 0.0
        return out

    def attempt_dist(self, op_id, present_p, health, pool, cache_cold, callee_classes, force_class=None, mask=None,
                     callee_retry=None):
        """Distribution over T_VALUES of an attempt: present with prob `present_p`;
        class from the table (or a forced pmf), SLOW splits the OK mass. `mask`
        is the outcome repertoire (names allowed); masked classes get zero
        probability and the rest renormalises."""
        out = np.zeros(len(T_VALUES))
        if present_p <= 0:
            out[5] = 1.0
            return out
        mask = self.backend_mask if mask is None else mask
        worst = self.worst_pmf(op_id, callee_classes)
        pmf = np.zeros(4)
        for wi, wname in enumerate(CLASS_VALUES):
            if worst[wi] > 0:
                pmf += worst[wi] * self.class_pmf(op_id, health, pool, wname)
        if force_class is not None:
            pmf = force_class
        pmf = pmf * np.array([1.0, "4xx" in mask, "5xx" in mask, "err" in mask], dtype=np.float64)
        pmf = pmf / pmf.sum()
        ps = self.p_slow(op_id, health, pool, cache_cold, callee_classes, callee_retry) if "slow" in mask else 0.0
        out[0] = pmf[0] * (1 - ps)
        out[1:4] = pmf[1:4]
        out[4] = pmf[0] * ps
        out *= present_p
        out[5] = 1 - present_p
        return out

    # --- node construction ---------------------------------------------------------------
    def _add(self, node: Node):
        if node.var.id in self.nodes:
            raise ValueError(f"duplicate node {node.var.id}")
        self.nodes[node.var.id] = node
        self.order.append(node.var.id)
        return node

    def _build_nodes(self):
        topo, sset, c = self.topo, self.sset, self.constants
        # intensity: exogenous root (deterministic in the daily profile)
        self._add(FunctionNode(StateVar("intensity", "intensity", "tick", INTENSITY_VALUES, latent=True),
                               [], lambda ctx: (1.0, 0.0, 0.0)))
        backend_services = [s for s in topo.services if s.kind in (KIND_SERVICE, KIND_BFF, KIND_EXTERNAL)]
        for s in backend_services:
            self._add(TableNode(StateVar(f"load:{s.index}", "load", "tick", LOAD_VALUES, latent=True, lag_self=True,
                                         meta={"service": s.name}),
                                ["intensity", f"load:{s.index}"], self.load_table))
            self._add(TableNode(StateVar(f"pool:{s.index}", "pool", "tick", POOL_VALUES, latent=True, lag_self=True,
                                         meta={"service": s.name}),
                                [f"load:{s.index}", f"pool:{s.index}"], self.pool_table))
            self._add(TableNode(StateVar(f"cache:{s.index}", "cache", "tick", CACHE_VALUES, latent=True, lag_self=True,
                                         meta={"service": s.name}),
                                [f"load:{s.index}", f"cache:{s.index}"], self.cache_table))
        for op in topo.ops:
            if op.kind == KIND_CLIENT:
                continue
            self._add(TableNode(StateVar(f"health:{op.id}", "health", "tick", HEALTH_VALUES, latent=True, lag_self=True,
                                         meta={"op": op.name, "service": topo.services[op.service].name}),
                                [f"pool:{op.service}", f"health:{op.id}"], self.health_table))
        # session latents per scenario
        pf, pa = c["client.net_flaky_share"], c["client.auth_expired_share"]
        for sc in sset.scenarios:
            self._add(FunctionNode(StateVar(f"net:{sc.index}", "net", "session", NET_VALUES, latent=True,
                                            meta={"scenario": sc.name}), [], lambda ctx, pf=pf: (1 - pf, pf)))
            self._add(FunctionNode(StateVar(f"auth:{sc.index}", "auth", "session", AUTH_VALUES, latent=True,
                                            meta={"scenario": sc.name}), [], lambda ctx, pa=pa: (1 - pa, pa)))
        # journeys: BFF attempt contexts
        self.bff_contexts = {}          # bff op id -> list of (scenario, step, attempt)
        for sc in sset.scenarios:
            for st in sc.steps:
                for k in range(sc.max_retries + 1):
                    self.bff_contexts.setdefault(st.bff_op, []).append((sc.index, st.index, k))
        # backend invocation + attempts + finals, deepest first so finals exist before callers
        backend = sorted((op for op in topo.ops if op.kind in (KIND_SERVICE, KIND_EXTERNAL)), key=lambda o: -o.depth)
        for op in backend:
            self._build_backend_op(op)
        # BFF contexts and client outcomes, in journey order
        for sc in sset.scenarios:
            for st in sc.steps:
                self._build_bff_step(sc, st)

    def _invoke_parents(self, op):
        """Callers' presence variables and the cache latents of cached incoming
        edges. Other callers are absent in the nominal context (an OR aggregator
        is otherwise saturated); a cache latent is measured with the first
        caller that uses it present. A present caller calls on its edge's share
        of requests (exogenous per-request noise, no node of its own)."""
        parents, edge_info, override, ctx_for = [], [], {}, {}
        for e in self.topo.caller_edges(op.id):
            caller = self.topo.ops[e.caller]
            if caller.kind == KIND_BFF:
                ids = [f"I:bff:{s}:{j}:{k}" for (s, j, k) in self.bff_contexts.get(caller.id, [])]
            else:
                ids = [f"I:{caller.id}"]
            if not ids:
                continue          # a BFF endpoint no scenario visits: the edge is never exercised
            cache_id = f"cache:{caller.service}" if e.cached else None
            for pid in ids:
                parents.append(pid)
                override[pid] = "absent"
                edge_info.append((pid, cache_id, e.p_hit, e.p_call))
            if cache_id:
                if cache_id not in parents:
                    parents.append(cache_id)
                ctx_for.setdefault(cache_id, {ids[0]: "present"})
        return parents, edge_info, override, ctx_for

    def _build_backend_op(self, op):
        topo = self.topo
        parents, edge_info, override, ctx_for = self._invoke_parents(op)
        var = StateVar(f"I:{op.id}", "invoke", "event", PRESENCE_VALUES, latent=False, derived=True, token_op=op.id,
                       meta={"op": op.name, "service": topo.services[op.service].name})

        def invoke_fn(ctx, edge_info=edge_info):
            p_absent = 1.0
            for pid, cache_id, p_hit, p_call in edge_info:
                if ctx.get(pid, "absent") != "present":
                    continue
                miss = 1.0 if cache_id is None else (1 - p_hit if ctx.get(cache_id, "warm") == "warm" else 1.0)
                p_absent *= (1 - p_call * miss)
            return (1 - p_absent, p_absent)

        inv = self._add(FunctionNode(var, parents, invoke_fn, override, ctx_for))
        # (caller presence id, cache latent id, p_hit, p_call) per incoming call
        # edge: the projection's vectorised sampler evaluates the noisy-OR from it.
        inv.edge_info = list(edge_info)
        svc = op.service
        callee_finals = [f"F:{e.callee}" for e in topo.callee_edges(op.id)]
        callee_firsts = [f"A:{e.callee}:0" for e in topo.callee_edges(op.id)] if self.backend_retries > 0 else []
        has_cache = any(e.cached for e in topo.callee_edges(op.id))
        state_parents = [f"health:{op.id}", f"pool:{svc}"] + ([f"cache:{svc}"] if has_cache else [])
        R = self.backend_retries
        for k in range(R + 1):
            first = f"I:{op.id}" if k == 0 else f"A:{op.id}:{k - 1}"
            parents_k = [first] + state_parents + callee_finals + callee_firsts
            var_k = StateVar(f"A:{op.id}:{k}", "attempt", "event", T_VALUES, latent=False, token_op=op.id,
                             meta={"op": op.name, "service": topo.services[op.service].name, "attempt": k})

            def attempt_fn(ctx, op_id=op.id, k=k, first=first, svc=svc, has_cache=has_cache, callee_finals=callee_finals, callee_firsts=callee_firsts):
                if k == 0:
                    present_p = 1.0 if ctx[first] == "present" else 0.0
                else:
                    prev = ctx[first]
                    present_p = self.p_retry if prev in self.backend_retry_on else 0.0
                health = HEALTH_VALUES.index(ctx[f"health:{op_id}"])
                pool = POOL_VALUES.index(ctx[f"pool:{svc}"])
                cold = has_cache and ctx[f"cache:{svc}"] == "cold"
                classes = {int(f.split(":")[1]): ctx[f] for f in callee_finals}
                retry = self.callee_retry_probs({int(f.split(":")[1]): ctx[f] for f in callee_firsts})
                return self.attempt_dist(op_id, present_p, health, pool, cold, classes, callee_retry=retry)

            # D-TB-20: the attempt that follows a retry holds that retry at absent.
            self._add(FunctionNode(var_k, parents_k, attempt_fn, {first: "absent"} if k >= 2 else None))
        self._add_final(f"F:{op.id}", [f"A:{op.id}:{k}" for k in range(R + 1)], T_VALUES, op.id,
                        {"op": op.name, "service": topo.services[op.service].name})

    def _add_final(self, fid, attempts, values, token_op, meta):
        var = StateVar(fid, "final", "event", values, latent=False, derived=True, token_op=token_op, meta=meta)
        absent = values[-1]

        def final_fn(ctx, attempts=attempts, values=values, absent=absent):
            last = absent
            for a in attempts:
                v = ctx[a]
                if v != absent:
                    last = v
            out = np.zeros(len(values))
            out[values.index(last)] = 1.0
            return out

        # D-TB-20: a final holds every retry attempt at absent, so the first
        # attempt's effect on it (and through it) is the controlled direct
        # effect with no retry; before, the last attempt's nominal `ok` made the
        # earlier attempts inert.
        self._add(FunctionNode(var, attempts, final_fn, {a: "absent" for a in attempts[1:]}))

    def _build_bff_step(self, sc, st):
        topo = self.topo
        b = st.bff_op
        bff_svc = topo.ops[b].service
        callee_finals = [f"F:{e.callee}" for e in topo.callee_edges(b)]
        callee_firsts = [f"A:{e.callee}:0" for e in topo.callee_edges(b)] if self.backend_retries > 0 else []
        has_cache = any(e.cached for e in topo.callee_edges(b))
        R = sc.max_retries
        retry_on = tuple(sc.retry_on)
        bff_mask = frozenset(sc.bff_mask)
        client_err_allowed = "err" in sc.client_mask
        for k in range(R + 1):
            iid = f"I:bff:{sc.index}:{st.index}:{k}"
            if k == 0:
                inv_parents = [] if st.index == 0 else [f"C:{sc.index}:{st.index - 1}:F"]
            else:
                inv_parents = [f"T:bff:{sc.index}:{st.index}:{k - 1}"]

            def inv_fn(ctx, inv_parents=inv_parents, k=k, retry_on=retry_on):
                if k == 0:
                    if not inv_parents:
                        return (1.0, 0.0)
                    return (1.0, 0.0) if ctx[inv_parents[0]] == "ok" else (0.0, 1.0)
                prev = ctx[inv_parents[0]]
                p = self.p_retry if prev in retry_on else 0.0
                return (p, 1 - p)

            self._add(FunctionNode(StateVar(iid, "invoke", "event", PRESENCE_VALUES, latent=False, derived=True,
                                            token_op=b, meta={"scenario": sc.name, "step": st.index, "attempt": k, "op": topo.ops[b].name}),
                                   inv_parents, inv_fn, {inv_parents[0]: "absent"} if k >= 2 else None))
            tid = f"T:bff:{sc.index}:{st.index}:{k}"
            t_parents = [iid, f"auth:{sc.index}", f"health:{b}", f"pool:{bff_svc}"] + ([f"cache:{bff_svc}"] if has_cache else []) + callee_finals + callee_firsts

            def t_fn(ctx, iid=iid, sc_i=sc.index, b=b, bff_svc=bff_svc, has_cache=has_cache, callee_finals=callee_finals, bff_mask=bff_mask, callee_firsts=callee_firsts):
                present_p = 1.0 if ctx[iid] == "present" else 0.0
                health = HEALTH_VALUES.index(ctx[f"health:{b}"])
                pool = POOL_VALUES.index(ctx[f"pool:{bff_svc}"])
                cold = has_cache and ctx[f"cache:{bff_svc}"] == "cold"
                classes = {int(f.split(":")[1]): ctx[f] for f in callee_finals}
                retry = self.callee_retry_probs({int(f.split(":")[1]): ctx[f] for f in callee_firsts})
                force = None
                if ctx[f"auth:{sc_i}"] == "expired":
                    base = self.attempt_dist(b, 1.0, health, pool, cold, classes, mask=bff_mask, callee_retry=retry)
                    cls = base[:4].copy()
                    cls[0] += base[4]           # fold slow back into ok for the mixing step
                    force = (1 - P_BFF_4XX_GIVEN_EXPIRED) * cls / max(cls.sum(), 1e-12)
                    force[1] += P_BFF_4XX_GIVEN_EXPIRED
                return self.attempt_dist(b, present_p, health, pool, cold, classes, force_class=force, mask=bff_mask, callee_retry=retry)

            self._add(FunctionNode(StateVar(tid, "attempt", "event", T_VALUES, latent=False, token_op=b,
                                            meta={"scenario": sc.name, "step": st.index, "attempt": k, "op": topo.ops[b].name}),
                                   t_parents, t_fn))
            cid = f"C:{sc.index}:{st.index}:{k}"

            def c_fn(ctx, tid=tid, sc_i=sc.index, client_err_allowed=client_err_allowed):
                t = ctx[tid]
                if t == "absent":
                    return (0.0, 0.0, 1.0)
                p_err = P_CLIENT_ERR_GIVEN_BFF_FAILURE if t in ("4xx", "5xx", "err") else 0.0
                if ctx[f"net:{sc_i}"] == "flaky":
                    p_err = p_err + (1 - p_err) * P_CLIENT_ERR_GIVEN_FLAKY
                if not client_err_allowed:
                    p_err = 0.0
                return (1 - p_err, p_err, 0.0)

            self._add(FunctionNode(StateVar(cid, "client", "event", CLIENT_T_VALUES, latent=False, token_op=st.client_op,
                                            meta={"scenario": sc.name, "step": st.index, "attempt": k}),
                                   [tid, f"net:{sc.index}"], c_fn))
        self._add_final(f"C:{sc.index}:{st.index}:F", [f"C:{sc.index}:{st.index}:{k}" for k in range(R + 1)],
                        CLIENT_T_VALUES, st.client_op, {"scenario": sc.name, "step": st.index})

    # --- strengths ---------------------------------------------------------------------------
    def compatible(self, parent_id, value, ctx):
        """False when `value` for the parent contradicts a held derived variable
        of the same operation (an attempt cannot be absent while the op's final
        or a later attempt is held non-absent; a final cannot be absent while an
        attempt is held present; an invocation cannot be absent while an
        attempt or final is held non-absent). Only such values are excluded from
        a strength: "holding the other parents" is what makes them impossible."""
        g, *rest = parent_id.split(":")
        if g not in ("A", "F", "I") or (rest and rest[0] == "bff"):
            return True
        op = rest[0]
        same_op = [(k, v) for k, v in ctx.items() if k.split(":")[0] in ("A", "F", "I") and len(k.split(":")) > 1
                   and k.split(":")[1] == op and k != parent_id]
        held_nonabsent = any(v not in ("absent",) and not (k.startswith("I:") and v == "absent") for k, v in same_op
                             if not k.startswith("I:"))
        if value == "absent" and g in ("A", "F") and held_nonabsent:
            return False
        if value == "absent" and g == "I" and held_nonabsent:
            return False
        if g == "A" and value != "absent":
            k_self = int(rest[1])
            for k, v in same_op:
                if k.startswith("A:") and int(k.split(":")[2]) < k_self and v == "absent":
                    return False    # a later attempt cannot be present when an earlier one is absent
                if k.startswith("I:") and v == "absent":
                    return False
        if g == "F" and value != "absent":
            for k, v in same_op:
                if k.startswith("I:") and v == "absent":
                    return False
        return True

    def strength(self, child_id, parent_id, ctx=None):
        node = self.nodes[child_id]
        parent = self.nodes[parent_id]
        base = dict(node.nominal_context(self, exclude=parent_id) if ctx is None else ctx)
        dists = []
        for v in parent.var.values:
            if not self.compatible(parent_id, v, base):
                continue
            base[parent_id] = v
            dists.append(node.dist_held(self, base))
        best = 0.0
        for a, b in itertools.combinations(range(len(dists)), 2):
            best = max(best, tv(dists[a], dists[b]))
        return best

    def strength_ctxmax(self, child_id, parent_id, max_parents=4):
        node = self.nodes[child_id]
        others = [p for p in node.parents if p != parent_id]
        if len(others) > max_parents:
            return None
        best = 0.0
        for combo in itertools.product(*[self.nodes[p].var.values for p in others]):
            ctx = dict(zip(others, combo))
            best = max(best, self.strength(child_id, parent_id, ctx))
        return best

    def edges(self, ctxmax=False):
        """All parent -> child dependences with strengths; self-loops listed apart."""
        edges, loops = [], []
        for cid in self.order:
            node = self.nodes[cid]
            for pid in node.parents:
                s = self.strength(cid, pid)
                rec = {"src": pid, "dst": cid, "strength": round(s, 6),
                       "context": {k: v for k, v in node.nominal_context(self, exclude=pid).items()}}
                if ctxmax:
                    m = self.strength_ctxmax(cid, pid)
                    rec["strength_ctxmax"] = None if m is None else round(m, 6)
                if pid == cid:
                    loops.append(rec)
                else:
                    edges.append(rec)
        return edges, loops

    def graph_json(self, ctxmax=None):
        if ctxmax is None:
            ctxmax = self.cfg.mechanism.ctxmax
        edges, loops = self.edges(ctxmax=ctxmax)
        nodes = [self.nodes[i].var.to_dict() for i in self.order]
        return {
            "description": "trace-bench mechanism graph: every state variable the simulation carries, latent ones flagged, every dependence with the strength that produced it (D-TB-3: nominal-context total variation)",
            "strength_definition": "max over parent-value pairs of TV(P(child | parent=a, others=nominal), P(child | parent=b, others=nominal)); strength_ctxmax = max over all contexts when the other-parent set has <= 4 members",
            "n_nodes": len(nodes), "n_edges": len(edges), "n_latent": sum(1 for n in nodes if n["latent"]),
            "nodes": nodes, "edges": edges, "self_loops": loops,
        }

    # --- summaries for the engine -------------------------------------------------------------
    def latent_ids(self):
        return [i for i in self.order if self.nodes[i].var.latent]

    def observable_ids(self):
        return [i for i in self.order if not self.nodes[i].var.latent]
