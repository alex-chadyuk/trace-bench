"""Latent projection, token expansion, floor sweep and coarsened views.

Vocabulary. Mechanism nodes fall into three classes: token-bearing observables
(attempt and client outcomes — their values are `(operation, outcome)` event
types), derived observables (invocation and final variables — deterministic
functions of observables carrying no token of their own) and latents. On the
observable twin every latent is observed and its values are state tokens.

Projection (Verma 1993 / Richardson ADMG, over the summary graph). For
token-bearing X, Y: a directed edge X -> Y exists iff a directed path X ... Y
has all intermediates among "through" nodes; a bidirected edge X <-> Y exists
iff some latent l reaches both X and Y along such paths. The through set is
the derived and latent nodes, per grain (D-TB-19): at the request grain the
BFF invocation variables `I:bff:*` are excluded, because every path through
one of them (a client retry re-invoking the BFF, a step's client outcome
gating the next step) crosses a request boundary; at the session grain they
are through nodes, which adds the journey and client-retry edges and may make
the type-level target cyclic. Bidirected groups are grain-independent (no
latent path crosses a token node) and are computed once.

Strengths are computed by marginalising the intermediates of each (source,
destination) pair exactly, with every other parent held at its nominal context
and tick-level latents held at a value contributing their stationary
distribution (`Node.dist_held`). The chain of intermediates is enumerated in a
per-pair topological order (the mechanism is cyclic at the type level through
journeys, but every per-pair chain is acyclic): a greedy minimum-frontier order
with construction position as the tie-break, so that every node sees its
assigned parents. Enumeration is frontier-merged and prunes mass below
`PRUNE`. Where a frontier would exceed `PROJECTION_CAP_PARTICLES` the effect
is instead a counter-keyed Monte-Carlo estimate over `PROJECTION_MC_N` forward
samples of the chain (`hashing.uniforms`, domain `D_PROJECTION`, the same draws
for every value of the source — common random numbers), the last node exact
given its sampled parents, rounded at `MC_DECIMALS`; every edge such an
estimate feeds carries `mc: {n, se}`.

Token expansion. For X with values V and Y with values W:
    s((X,x) -> (Y,y)) = | P(Y=y | do(X=x)) - mean_{x' != x} P(Y=y | do(X=x')) |
(one-vs-rest Bernoulli total variation under uniform replacement). Bidirected:
    s((X,x) <-> (Y,y)) = max_l max_a min(delta_X(l,a,x), delta_Y(l,a,y)),
    delta_X(l,a,x) = | P(X=x | do(l=a)) - mean_{a' != a} P(X=x | do(l=a')) |.
Several variables map onto the same token pair (retry attempts, journey
contexts): the pair takes the maximum.

Support. The request-grain target keeps token pairs whose operations can
co-occur in one request tree; the session-grain target keeps pairs that can
co-occur in one journey (it adds journey edges and cross-request confounding
and may be cyclic at the type level). Within-operation token pairs (retries)
are recorded apart and never scored. The floor is applied LAST (project, then
floor); every edge is stored with its strength.
"""
from __future__ import annotations

import itertools
import math
from collections import defaultdict
from fractions import Fraction

import numpy as np

from .constants import (
    CLIENT_T_VALUES, GRAINS, KIND_BFF, KIND_CLIENT, KIND_EXTERNAL, KIND_SERVICE, MC_DECIMALS,
    PROJECTION_CAP_PARTICLES, PROJECTION_MC_N, T_VALUES,
)
from .hashing import D_PROJECTION, uniforms
from .mechanism import Mechanism
from .rng import stable_id

TOKEN_GROUPS = ("attempt", "client")
PRUNE = 1e-10
# Derived nodes whose every incoming path crosses a request boundary; not
# through nodes at the request grain (D-TB-19).
CROSS_REQUEST_PREFIX = "I:bff:"


def token_key(op_id, outcome):
    return f"{op_id}:{outcome}"


def state_token_key(var_id, value):
    return f"state:{var_id}={value}"


class Projector:
    def __init__(self, mech: Mechanism, grain="request", twin=False, prune=PRUNE,
                 cap=PROJECTION_CAP_PARTICLES, mc_n=PROJECTION_MC_N):
        if grain not in GRAINS:
            raise ValueError(f"grain must be one of {GRAINS}, got {grain!r}")
        self.mech = mech
        self.grain = grain
        self.twin = bool(twin)
        self.prune = prune
        self.cap = None if cap is None else int(cap)
        self.mc_n = int(mc_n)
        self.children = defaultdict(list)
        for cid, node in mech.nodes.items():
            for pid in node.parents:
                if pid != cid:
                    self.children[pid].append(cid)
        self.pos = {nid: i for i, nid in enumerate(mech.order)}
        self.token_nodes = [nid for nid in mech.order if mech.nodes[nid].var.group in TOKEN_GROUPS]
        self.latent = set() if self.twin else set(mech.latent_ids())
        self.derived = {nid for nid in mech.order if mech.nodes[nid].var.derived}
        self.through = self.latent | self.derived
        if grain == "request":
            self.through = {n for n in self.through if not n.startswith(CROSS_REQUEST_PREFIX)}
        self.state_nodes = list(mech.latent_ids()) if self.twin else []
        self._vals_index = {nid: {v: i for i, v in enumerate(n.var.values)} for nid, n in mech.nodes.items()}
        self._effect_cache = {}     # (src, value, dst) -> (pmf, meta | None)
        self._chain_cache = {}      # (src, dst) -> (ordered intermediates, later-parent sets)
        self.reset_stats()

    # --- bookkeeping --------------------------------------------------------------------
    def reset_stats(self):
        self.n_effects_exact = 0
        self.n_effects_mc = 0
        self.mc_se_max = 0.0
        self.peak_particles = 0
        self.mc_pairs = set()

    def stats(self):
        return {"n_effects_exact": self.n_effects_exact, "n_effects_mc": self.n_effects_mc,
                "mc_se_max": self.mc_se_max, "peak_particles": self.peak_particles,
                "n_mc_pairs": len(self.mc_pairs)}

    # --- reachability through latent/derived intermediates -----------------------------
    def forward(self, src):
        """(endpoints, intermediates): non-through nodes reached from src along
        through-only paths, and the through nodes visited."""
        endpoints, inter = set(), set()
        stack = [src]
        while stack:
            u = stack.pop()
            for c in self.children[u]:
                if c in self.through:
                    if c not in inter:
                        inter.add(c)
                        stack.append(c)
                else:
                    endpoints.add(c)
        return endpoints, inter

    def backward(self, dst):
        inter = set()
        stack = [dst]
        while stack:
            u = stack.pop()
            for p in self.mech.nodes[u].parents:
                if p == u:
                    continue
                if p in self.through and p not in inter:
                    inter.add(p)
                    stack.append(p)
        return inter

    # --- per-pair chain order -----------------------------------------------------------------
    def chain_order(self, src, dst):
        """The intermediates of (src, dst) in a topological order of the induced
        through-subgraph: at each step the ready node whose processing grows the
        held frontier least (the ratio of the value count it adds to the value
        counts it releases, compared exactly), ties by construction position.
        The mechanism is cyclic at the type level, but every cycle passes a
        token node and token nodes are never through nodes, so each chain is
        acyclic; a cyclic chain is an error, never a silent partial order."""
        mech = self.mech
        M = set(self.forward(src)[1]) & self.backward(dst)
        M.discard(src)
        if not M:
            return []
        allset = M | {dst}
        par = {m: [p for p in mech.nodes[m].parents if p != m and p in allset] for m in allset}
        rem = defaultdict(int)
        for m in allset:
            for p in par[m]:
                rem[p] += 1
        nv = {m: len(mech.nodes[m].var.values) for m in allset}
        done, order, todo = set(), [], set(M)
        while todo:
            best = None
            for c in todo:
                ps = par[c]
                if any(p not in done for p in ps):
                    continue
                added = nv[c] if rem[c] > 0 else 1
                freed = math.prod(nv[p] for p in ps if rem[p] == 1)
                key = (Fraction(added, freed), self.pos[c])
                if best is None or key < best[0]:
                    best = (key, c)
            if best is None:
                raise ValueError(f"cyclic chain of intermediates for {src} -> {dst}")
            c = best[1]
            for p in par[c]:
                rem[p] -= 1
            done.add(c)
            todo.discard(c)
            order.append(c)
        return order

    def chain(self, src, dst):
        """(ordered intermediates, later-parent set per position of the chain + dst)."""
        key = (src, dst)
        hit = self._chain_cache.get(key)
        if hit is None:
            order = self.chain_order(src, dst)
            full = order + [dst]
            later = [set() for _ in full]
            acc = set()
            for idx in range(len(full) - 1, -1, -1):
                later[idx] = set(acc)
                acc.update(self.mech.nodes[full[idx]].parents)
            hit = (order, later)
            self._chain_cache[key] = hit
        return hit

    def intermediates(self, src, dst):
        return list(self.chain(src, dst)[0])

    # --- effects --------------------------------------------------------------------------------
    @staticmethod
    def _context(mech, node, assigned):
        ctx = node.nominal_context(mech)
        for k in assigned:
            if k in node.parents:
                ctx.update(node.context_for(k))
        for k, v in assigned.items():
            if k in node.parents:
                ctx[k] = v
        return ctx

    def effect(self, src, src_value, dst):
        """P(dst | do(src = src_value)), other parents at nominal context,
        intermediates marginalised exactly (up to `prune` mass) under the
        particle cap, else estimated by counter-keyed Monte Carlo."""
        return self._effect(src, src_value, dst)[0]

    def effect_meta(self, src, src_value, dst):
        """None for an exact effect; {"n", "se"} for a Monte-Carlo estimate."""
        return self._effect(src, src_value, dst)[1]

    def _effect(self, src, src_value, dst):
        key = (src, src_value, dst)
        hit = self._effect_cache.get(key)
        if hit is not None:
            return hit
        chain, later = self.chain(src, dst)
        out, peak = self._effect_exact(src, src_value, dst, chain, later)
        self.peak_particles = max(self.peak_particles, peak)
        if out is None:
            out, se = self._effect_mc(src, src_value, dst, chain, self.mc_n)
            meta = {"n": self.mc_n, "se": se}
            self.n_effects_mc += 1
            self.mc_se_max = max(self.mc_se_max, se)
            self.mc_pairs.add((src, dst))
        else:
            meta = None
            self.n_effects_exact += 1
        hit = (out, meta)
        self._effect_cache[key] = hit
        return hit

    # --- vectorised evaluation of one node over many assignments ---------------------------
    def _node_rows(self, node, keys, sub, n):
        """Distribution rows of `node` under `n` assignments of its in-chain parents
        `keys` (`sub` is the (n, len(keys)) matrix of their value indices; every
        other parent at the node's nominal context, `context_for` applied for the
        assigned ones). Returns (D, inv): the distinct rows and the row index of
        every assignment. An invocation node is evaluated per row in closed form —
        the same noisy-OR product, in the same order, as `invoke_fn`."""
        mech = self.mech
        base = node.nominal_context(mech)
        for k in keys:
            base.update(node.context_for(k))
        edge_info = getattr(node, "edge_info", None)
        if edge_info is not None:
            col = {k: i for i, k in enumerate(keys)}
            p_absent = np.ones(n)
            for pid, cache_id, p_hit, p_call in edge_info:
                if pid in col:
                    present = sub[:, col[pid]] == 0
                elif base.get(pid, "absent") != "present":
                    continue
                else:
                    present = True
                if cache_id is None:
                    miss = 1.0
                elif cache_id in col:
                    miss = np.where(sub[:, col[cache_id]] == 0, 1 - p_hit, 1.0)
                else:
                    miss = (1 - p_hit) if base.get(cache_id, "warm") == "warm" else 1.0
                p_absent = np.where(present, p_absent * (1 - p_call * miss), p_absent)
            return np.stack([1 - p_absent, p_absent], axis=1), np.arange(n)
        if keys:
            uniq, inv = np.unique(sub, axis=0, return_inverse=True)
            inv = inv.reshape(-1)
        else:
            uniq, inv = np.zeros((1, 0), dtype=np.int64), np.zeros(n, dtype=np.int64)
        rows = []
        for r in uniq:
            ctx = dict(base)
            for k, vi in zip(keys, r):
                ctx[k] = mech.nodes[k].var.values[int(vi)]
            rows.append(np.asarray(node.dist_held(mech, ctx), dtype=np.float64))
        return np.stack(rows, axis=0), inv

    def _merge(self, A, w, cols):
        """Merge equal assignment rows, summing their weights (sequentially, in row
        order). Rows are keyed by a mixed-radix integer when the joint value count
        fits 62 bits, by row-unique otherwise; both orders are functions of the
        integer rows alone."""
        n, k = A.shape
        if k == 0:
            return A[:1], np.bincount(np.zeros(n, dtype=np.int64), weights=w, minlength=1)
        radix = [len(self.mech.nodes[c].var.values) for c in cols]
        if sum(math.log2(r) for r in radix) < 62:
            key = np.zeros(n, dtype=np.int64)
            mult = 1
            for j in range(k):
                key += A[:, j].astype(np.int64) * mult
                mult *= radix[j]
            _u, first, inv = np.unique(key, return_index=True, return_inverse=True)
            uniq = A[first]
        else:
            uniq, inv = np.unique(A, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        return uniq, np.bincount(inv, weights=w, minlength=len(uniq))

    def _effect_exact(self, src, src_value, dst, chain, later_sets):
        """Frontier-merged enumeration with the frontier as an int8 matrix of
        assignments (one column per held variable) and a weight vector: each
        step evaluates the node per distinct context, expands every particle by
        the node's values, prunes mass at or below `prune`, projects onto the
        variables later nodes still need and merges equal rows. Returns (pmf,
        peak particles), or (None, peak) once a merged frontier exceeds the cap
        (particle counts are integer functions of the tables and the prune, so
        the decision is the same everywhere). Memory is a few dozen bytes per
        particle plus the |values|-fold expansion of one step."""
        mech = self.mech
        full = list(chain) + [dst]
        cols = [src]
        A = np.array([[self._vals_index[src][src_value]]], dtype=np.int8)
        w = np.array([1.0])
        peak = 1
        cap = self.cap
        for idx, m in enumerate(full):
            node = mech.nodes[m]
            parents = set(node.parents)
            key_idx = [i for i, c in enumerate(cols) if c in parents]
            D, inv = self._node_rows(node, [cols[i] for i in key_idx], A[:, key_idx], len(w))
            if m == dst:
                zero = np.zeros(len(w), dtype=np.int64)
                out = np.array([np.bincount(zero, weights=w * D[inv, v], minlength=1)[0] for v in range(D.shape[1])])
                total = float(out.sum())
                if total <= 0:
                    out[:] = 0.0
                else:
                    out = out / total
                return out, peak
            later = later_sets[idx]
            Q = w[:, None] * D[inv]
            ii, vv = np.nonzero(Q > self.prune)
            keep = [i for i, c in enumerate(cols) if c in later]
            new_cols = [cols[i] for i in keep]
            parts = [A[ii][:, keep]]
            if m in later:
                parts.append(vv.astype(np.int8)[:, None])
                new_cols.append(m)
            A, w = self._merge(np.concatenate(parts, axis=1), Q[ii, vv], new_cols)
            cols = new_cols
            peak = max(peak, len(w))
            if cap is not None and len(w) > cap:
                return None, peak
        raise AssertionError("the destination is always the last node")

    def _effect_mc(self, src, src_value, dst, chain, n):
        """Forward sampling of the chain, vectorised over samples. Draw j of
        sample i is `uniforms(seed, D_PROJECTION, pair, i, j)`: a pure function
        of the pair and the coordinates, so every value of the source sees the
        same draws (common random numbers) and the estimate never depends on
        call order, cap or platform. Nodes are evaluated by `_node_rows`; the
        last node is exact given its sampled parents. Returns (mean pmf, se)
        rounded at MC_DECIMALS, `se` the largest standard error over the
        destination's values."""
        mech = self.mech
        pair_key = int(stable_id(src, dst)[:15], 16)
        sample = np.arange(n)
        assigned = {src: np.full(n, self._vals_index[src][src_value], dtype=np.int64)}
        full = list(chain) + [dst]
        for j, m in enumerate(full):
            node = mech.nodes[m]
            keys = [p for p in node.parents if p in assigned]
            sub = np.stack([assigned[k] for k in keys], axis=1) if keys else np.zeros((n, 0), dtype=np.int64)
            D, inv = self._node_rows(node, keys, sub, n)
            if m == dst:
                R = D[inv]
                mean = R.mean(axis=0)
                se = R.std(axis=0, ddof=1) / np.sqrt(n)
                return np.round(mean, MC_DECIMALS), float(np.round(se.max(), MC_DECIMALS))
            C = np.cumsum(D, axis=1)[inv]
            u = uniforms(mech.seed, D_PROJECTION, pair_key, sample, j)
            idx = (u[:, None] >= C).sum(axis=1)
            assigned[m] = np.minimum(idx, C.shape[1] - 1).astype(np.int64)
        raise AssertionError("the destination is always the last node")

    def _pair_mc(self, src, vals, dst):
        """The `mc` record of a (src, dst) pair: None if every effect behind it
        is exact, else the sample count and the largest standard error."""
        metas = [self.effect_meta(src, v, dst) for v in vals]
        used = [m for m in metas if m is not None]
        if not used:
            return None
        return {"n": max(m["n"] for m in used), "se": max(m["se"] for m in used)}

    # --- tokens -------------------------------------------------------------------------
    def token_of(self, nid, value):
        node = self.mech.nodes[nid]
        if node.var.group in TOKEN_GROUPS:
            if value == "absent":
                return None
            return token_key(node.var.token_op, value)
        if self.twin and nid in self.latent_all():
            return state_token_key(nid, value)
        return None

    def latent_all(self):
        return set(self.mech.latent_ids())

    def sources(self):
        return list(self.token_nodes) + list(self.state_nodes)

    def token_op(self, nid):
        return self.mech.nodes[nid].var.token_op

    # --- directed token edges -------------------------------------------------------------
    def directed_token_edges(self):
        """{(tok_src, tok_dst): {"strength", "via": (src_node, dst_node), "mc"}};
        within-operation pairs are returned separately as retry pairs. `mc` is
        None when the pair's effects are exact."""
        edges = {}
        within = {}
        mech = self.mech
        for src in self.sources():
            endpoints, _ = self.forward(src)
            vals = mech.nodes[src].var.values
            for dst in sorted(endpoints, key=self.pos.__getitem__):
                if dst == src:
                    continue
                dnode = mech.nodes[dst]
                if dnode.var.group not in TOKEN_GROUPS and not (self.twin and dst in self.latent):
                    continue
                P = {v: self.effect(src, v, dst) for v in vals}
                mc = self._pair_mc(src, vals, dst)
                same_op = (self.token_op(src) is not None and self.token_op(src) == self.token_op(dst))
                for x in vals:
                    tx = self.token_of(src, x)
                    if tx is None:
                        continue
                    rest = [P[v] for v in vals if v != x]
                    mean_rest = np.mean(rest, axis=0) if rest else np.zeros_like(P[x])
                    for i, y in enumerate(dnode.var.values):
                        ty = self.token_of(dst, y)
                        if ty is None:
                            continue
                        s = float(abs(P[x][i] - mean_rest[i]))
                        if s <= 0:
                            continue
                        target = within if same_op else edges
                        rec = target.get((tx, ty))
                        if rec is None or s > rec["strength"]:
                            target[(tx, ty)] = {"strength": s, "via": (src, dst), "mc": mc}
        return edges, within

    # --- bidirected token edges (latent instance only) -----------------------------------------
    def bidirected_groups(self):
        """Per latent l: {token: [delta for each value a of l]} over the token
        nodes l reaches through latent/derived paths. Pair strengths derive
        from these as max_a min(delta_i[a], delta_j[a]). `mc` maps the tokens
        whose deltas rest on a Monte-Carlo estimate to {"n", "se"}."""
        groups = []
        mech = self.mech
        for l in sorted(self.latent, key=self.pos.__getitem__):
            endpoints, _ = self.forward(l)
            endpoints = [e for e in endpoints if mech.nodes[e].var.group in TOKEN_GROUPS]
            if len(endpoints) < 2:
                continue
            vals = mech.nodes[l].var.values
            deltas = {}      # token -> np.array over vals (max over nodes mapping to the token)
            mc_tokens = {}
            for dst in sorted(endpoints, key=self.pos.__getitem__):
                dnode = mech.nodes[dst]
                P = {a: self.effect(l, a, dst) for a in vals}
                mc = self._pair_mc(l, vals, dst)
                for i, y in enumerate(dnode.var.values):
                    ty = self.token_of(dst, y)
                    if ty is None:
                        continue
                    row = np.zeros(len(vals))
                    for ai, a in enumerate(vals):
                        rest = [P[b][i] for b in vals if b != a]
                        row[ai] = abs(P[a][i] - (np.mean(rest) if rest else 0.0))
                    if ty in deltas:
                        deltas[ty] = np.maximum(deltas[ty], row)
                    else:
                        deltas[ty] = row
                    if mc is not None:
                        cur = mc_tokens.get(ty)
                        mc_tokens[ty] = mc if cur is None else {"n": max(cur["n"], mc["n"]), "se": max(cur["se"], mc["se"])}
            groups.append({"latent": l, "values": list(vals),
                           "members": {t: [float(x) for x in row] for t, row in sorted(deltas.items())},
                           "mc": {t: mc_tokens[t] for t in sorted(mc_tokens)}})
        return groups


# =====================================================================================
def co_occurrence_support(inst):
    """Sets of op pairs that can share a request tree / a journey."""
    topo, sset = inst.topo, inst.sset
    request_ops = []
    for b in topo.bff_ops:
        request_ops.append(set(topo.reachable_from(b)))
    request_pairs = set()
    for ops in request_ops:
        for a, b in itertools.combinations(sorted(ops), 2):
            request_pairs.add((a, b))
    journey_pairs = set(request_pairs)
    for sc in sset.scenarios:
        ops = set()
        for st in sc.steps:
            ops.add(st.client_op)
            ops.update(topo.reachable_from(st.bff_op))
        for a, b in itertools.combinations(sorted(ops), 2):
            journey_pairs.add((a, b))
    return request_pairs, journey_pairs


def _op_of_token(tok):
    if tok.startswith("state:"):
        return None
    return int(tok.split(":")[0])


def _pair_in_support(ta, tb, pairs):
    a, b = _op_of_token(ta), _op_of_token(tb)
    if a is None or b is None:
        return True          # state tokens: supported everywhere they reach
    if a == b:
        return False
    return (min(a, b), max(a, b)) in pairs


def build_alphabet(inst, twin=False):
    """Token universe: every (op, outcome) an op can emit under its repertoire
    masks; on the twin, every (latent variable, value) as well."""
    cfg, topo, sset = inst.cfg, inst.topo, inst.sset
    tokens = []
    bff_masks = defaultdict(set)
    client_masks = {}
    for sc in sset.scenarios:
        for st in sc.steps:
            bff_masks[st.bff_op].update(sc.bff_mask)
            client_masks[st.client_op] = list(sc.client_mask)
    for op in topo.ops:
        svc = topo.services[op.service].name
        if op.kind == KIND_CLIENT:
            outs = client_masks.get(op.id, [])
        elif op.kind == KIND_BFF:
            outs = sorted(bff_masks.get(op.id, set()), key=T_VALUES.index)
        else:
            outs = [o for o in T_VALUES[:5] if o in set(cfg.endpoints.repertoire)]
        for o in outs:
            tokens.append({"token": token_key(op.id, o), "op_id": op.id, "service": svc, "name": op.name,
                           "kind": op.kind, "outcome": o})
    if twin:
        for nid in inst.mechanism.latent_ids():
            var = inst.mechanism.nodes[nid].var
            for v in var.values:
                tokens.append({"token": state_token_key(nid, v), "op_id": None, "service": None,
                               "name": nid, "kind": -1, "outcome": v, "state_var": nid, "group": var.group})
    return {"description": "trace-bench token alphabet: (operation, outcome) event types" + (" plus (state variable, value) tokens of the observable twin" if twin else ""),
            "n_tokens": len(tokens), "tokens": tokens}


def _mc_merge(*recs):
    used = [r for r in recs if r]
    if not used:
        return None
    return {"n": max(r["n"] for r in used), "se": max(r["se"] for r in used)}


def build_targets(inst, twin=False, cap=PROJECTION_CAP_PARTICLES, mc_n=PROJECTION_MC_N):
    """Returns (request_target, session_target, {"request": projector, "session": projector})
    as JSON-ready dicts. Directed edges are projected per grain; the bidirected
    groups once (they are grain-independent)."""
    mech = inst.mechanism
    projs = {g: Projector(mech, grain=g, twin=twin, cap=cap, mc_n=mc_n) for g in GRAINS}
    directed, within, dir_stats = {}, {}, {}
    for g, proj in projs.items():
        directed[g], within[g] = proj.directed_token_edges()
        dir_stats[g] = proj.stats()
        proj.reset_stats()
    groups = projs["request"].bidirected_groups() if not twin else []
    group_stats = projs["request"].stats()
    req_pairs, jour_pairs = co_occurrence_support(inst)

    def materialise(grain, pairs_support):
        d_edges = []
        for (a, b), r in sorted(directed[grain].items()):
            if not _pair_in_support(a, b, pairs_support) or round(r["strength"], 6) <= 0:
                continue
            rec = {"src": a, "dst": b, "strength": round(r["strength"], 6), "via": list(r["via"])}
            if r["mc"]:
                rec["mc"] = dict(r["mc"])
            d_edges.append(rec)
        b_edges = {}
        for g in groups:
            members = list(g["members"].items())
            for (ta, ra), (tb, rb) in itertools.combinations(members, 2):
                if not _pair_in_support(ta, tb, pairs_support):
                    continue
                s = max(min(x, y) for x, y in zip(ra, rb))
                if round(float(s), 6) <= 0:
                    continue
                key = (ta, tb) if ta < tb else (tb, ta)
                cur = b_edges.get(key)
                if cur is None or s > cur["strength"]:
                    rec = {"a": key[0], "b": key[1], "strength": round(float(s), 6), "via": g["latent"]}
                    mc = _mc_merge(g["mc"].get(ta), g["mc"].get(tb))
                    if mc:
                        rec["mc"] = mc
                    b_edges[key] = rec
        return d_edges, [b_edges[k] for k in sorted(b_edges)]

    req_d, req_b = materialise("request", req_pairs)
    ses_d, ses_b = materialise("session", jour_pairs)
    floor = inst.cfg.mechanism.floor

    def pack(grain, d_edges, b_edges, pairs_support):
        retry_pairs = [{"src": a, "dst": b, "strength": round(r["strength"], 6), "via": list(r["via"])}
                       for (a, b), r in sorted(within[grain].items()) if round(r["strength"], 6) > 0]
        ds, gs = dir_stats[grain], group_stats
        return {
            "description": f"trace-bench scoring target ({grain} grain): latent projection of the mechanism over (operation, outcome) tokens, projected first and floored second; every edge stored with its strength; an edge whose strength rests on a Monte-Carlo estimate above the projection's particle cap carries `mc` (D-TB-19)",
            "grain": grain, "variant": "twin" if twin else "latent", "default_floor": floor,
            "n_directed": len(d_edges), "n_bidirected": len(b_edges),
            "n_directed_at_floor": sum(1 for e in d_edges if e["strength"] >= floor),
            "n_bidirected_at_floor": sum(1 for e in b_edges if e["strength"] >= floor),
            "directed_acyclic_at_floor": is_acyclic([(e["src"], e["dst"]) for e in d_edges if e["strength"] >= floor]),
            "support_pairs": len(pairs_support),
            "support_op_pairs": [list(x) for x in sorted(pairs_support)],
            "projection_cap": cap, "mc_n": mc_n,
            "n_effects_exact": ds["n_effects_exact"] + gs["n_effects_exact"],
            "n_effects_mc": ds["n_effects_mc"] + gs["n_effects_mc"],
            "mc_se_max": max(ds["mc_se_max"], gs["mc_se_max"]),
            "n_directed_mc": sum(1 for e in d_edges if "mc" in e),
            "n_bidirected_mc": sum(1 for e in b_edges if "mc" in e),
            "directed": d_edges, "bidirected": b_edges,
            "bidirected_groups": groups,
            "retry_pairs": retry_pairs,
        }

    return pack("request", req_d, req_b, req_pairs), pack("session", ses_d, ses_b, jour_pairs), projs


def is_acyclic(edges):
    children = defaultdict(list)
    nodes = set()
    for a, b in edges:
        children[a].append(b)
        nodes.add(a)
        nodes.add(b)
    state = {}
    for n in sorted(nodes):
        if n in state:
            continue
        stack = [(n, iter(children[n]))]
        state[n] = 1
        while stack:
            u, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                state[u] = 2
                stack.pop()
                continue
            s = state.get(nxt)
            if s == 1:
                return False
            if s is None:
                state[nxt] = 1
                stack.append((nxt, iter(children[nxt])))
    return True


def floor_sensitivity(target, floors):
    out = []
    d = target["directed"]
    b = target["bidirected"]
    nodes_all = set()
    for e in d:
        nodes_all.update((e["src"], e["dst"]))
    for e in b:
        nodes_all.update((e["a"], e["b"]))
    for f in floors:
        dd = [e for e in d if e["strength"] >= f]
        bb = [e for e in b if e["strength"] >= f]
        touched = set()
        for e in dd:
            touched.update((e["src"], e["dst"]))
        for e in bb:
            touched.update((e["a"], e["b"]))
        out.append({"floor": f, "n_directed": len(dd), "n_bidirected": len(bb),
                    "nodes_with_degree": len(touched),
                    "density_directed": (len(dd) / max(1, len(nodes_all) * (len(nodes_all) - 1))),
                    "acyclic": is_acyclic([(e["src"], e["dst"]) for e in dd])})
    return {"default_floor": target["default_floor"], "sweep": out}


def coarsen(target, level, topo):
    """Endpoint- or service-level view: a function of the target alone
    (max over member token pairs)."""
    def key_of(tok):
        op = _op_of_token(tok)
        if op is None:
            return tok
        if level == "endpoint":
            return f"op:{op}"
        return f"svc:{topo.ops[op].service}"

    d = {}
    for e in target["directed"]:
        a, b = key_of(e["src"]), key_of(e["dst"])
        if a == b:
            continue
        if e["strength"] > d.get((a, b), 0.0):
            d[(a, b)] = e["strength"]
    bd = {}
    for e in target["bidirected"]:
        a, b = key_of(e["a"]), key_of(e["b"])
        if a == b:
            continue
        k = (a, b) if a < b else (b, a)
        if e["strength"] > bd.get(k, 0.0):
            bd[k] = e["strength"]
    return {"level": level, "grain": target["grain"], "default_floor": target["default_floor"],
            "directed": [{"src": a, "dst": b, "strength": s} for (a, b), s in sorted(d.items())],
            "bidirected": [{"a": a, "b": b, "strength": s} for (a, b), s in sorted(bd.items())]}
