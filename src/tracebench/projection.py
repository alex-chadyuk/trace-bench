"""Latent projection, token expansion, floor sweep and coarsened views.

Vocabulary. Mechanism nodes fall into three classes: token-bearing observables
(attempt and client outcomes — their values are `(operation, outcome)` event
types), derived observables (invocation and final variables — deterministic
functions of observables carrying no token of their own) and latents. On the
observable twin every latent is observed and its values are state tokens.

Projection (Verma 1993 / Richardson ADMG, over the summary graph). For
token-bearing X, Y: a directed edge X -> Y exists iff a directed path X ... Y
has all intermediates among derived or latent nodes; a bidirected edge X <-> Y
exists iff some latent l reaches both X and Y along such paths. Strengths are
computed by exact marginalisation of the intermediates (frontier-merged
enumeration in topological order), with every other parent held at its
nominal context; tick-level latents held at a value contribute their
stationary distribution (`Node.dist_held`).

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
from collections import defaultdict

import numpy as np

from .constants import CLIENT_T_VALUES, KIND_BFF, KIND_CLIENT, KIND_EXTERNAL, KIND_SERVICE, T_VALUES
from .mechanism import Mechanism

TOKEN_GROUPS = ("attempt", "client")
PRUNE = 1e-10


def token_key(op_id, outcome):
    return f"{op_id}:{outcome}"


def state_token_key(var_id, value):
    return f"state:{var_id}={value}"


class Projector:
    def __init__(self, mech: Mechanism, twin=False, prune=PRUNE):
        self.mech = mech
        self.twin = bool(twin)
        self.prune = prune
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
        self.state_nodes = list(mech.latent_ids()) if self.twin else []
        self._effect_cache = {}

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

    def intermediates(self, src, dst):
        _, fw = self.forward(src)
        bw = self.backward(dst)
        return sorted(fw & bw, key=self.pos.__getitem__)

    # --- exact effect by frontier-merged enumeration -----------------------------------
    def effect(self, src, src_value, dst):
        """P(dst | do(src = src_value)), other parents at nominal context,
        intermediates marginalised exactly (up to `prune` mass)."""
        key = (src, src_value, dst)
        if key in self._effect_cache:
            return self._effect_cache[key]
        mech = self.mech
        M = self.intermediates(src, dst)
        chain = M + [dst]
        particles = {frozenset({(src, src_value)}): 1.0}
        for idx, m in enumerate(chain):
            node = mech.nodes[m]
            later_parents = set()
            for r in chain[idx + 1:]:
                later_parents.update(mech.nodes[r].parents)
            new = defaultdict(float)
            is_last = m == dst
            for assign, p in particles.items():
                a = dict(assign)
                ctx = node.nominal_context(mech)
                for k in a:
                    if k in node.parents:
                        ctx.update(node.context_for(k))
                for k, v in a.items():
                    if k in node.parents:
                        ctx[k] = v
                d = node.dist_held(mech, ctx)
                if is_last:
                    new[("__out__",)] += 0.0
                    for i, pv in enumerate(d):
                        new[("__out__", i)] += p * pv
                    continue
                for i, pv in enumerate(d):
                    q = p * pv
                    if q <= self.prune:
                        continue
                    a2 = {k: v for k, v in a.items() if k in later_parents}
                    if m in later_parents:
                        a2[m] = node.var.values[i]
                    new[frozenset(a2.items())] += q
            particles = new
        out = np.zeros(len(mech.nodes[dst].var.values))
        for k, p in particles.items():
            if len(k) == 2 and k[0] == "__out__":
                out[k[1]] = p
        total = float(out.sum())
        if total <= 0:
            out[:] = 0.0
        else:
            out = out / total
        self._effect_cache[key] = out
        return out

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
        """{(tok_src, tok_dst): {"strength", "via": [(src_node, dst_node)]}};
        within-operation pairs are returned separately as retry pairs."""
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
                            target[(tx, ty)] = {"strength": s, "via": (src, dst)}
        return edges, within

    # --- bidirected token edges (latent instance only) -----------------------------------------
    def bidirected_groups(self):
        """Per latent l: {token: [delta for each value a of l]} over the token
        nodes l reaches through latent/derived paths. Pair strengths derive
        from these as max_a min(delta_i[a], delta_j[a])."""
        groups = []
        mech = self.mech
        for l in sorted(self.latent, key=self.pos.__getitem__):
            endpoints, _ = self.forward(l)
            endpoints = [e for e in endpoints if mech.nodes[e].var.group in TOKEN_GROUPS]
            if len(endpoints) < 2:
                continue
            vals = mech.nodes[l].var.values
            deltas = {}      # token -> np.array over vals (max over nodes mapping to the token)
            for dst in sorted(endpoints, key=self.pos.__getitem__):
                dnode = mech.nodes[dst]
                P = {a: self.effect(l, a, dst) for a in vals}
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
            groups.append({"latent": l, "values": list(vals),
                           "members": {t: [float(x) for x in row] for t, row in sorted(deltas.items())}})
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


def build_targets(inst, twin=False):
    """Returns (request_target, session_target, projector) as JSON-ready dicts."""
    mech = inst.mechanism
    proj = Projector(mech, twin=twin)
    directed, within = proj.directed_token_edges()
    groups = proj.bidirected_groups() if not twin else []
    req_pairs, jour_pairs = co_occurrence_support(inst)

    def materialise(pairs_support):
        d_edges = [{"src": a, "dst": b, "strength": round(r["strength"], 6), "via": list(r["via"])}
                   for (a, b), r in sorted(directed.items())
                   if _pair_in_support(a, b, pairs_support) and round(r["strength"], 6) > 0]
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
                    b_edges[key] = {"a": key[0], "b": key[1], "strength": round(float(s), 6), "via": g["latent"]}
        return d_edges, [b_edges[k] for k in sorted(b_edges)]

    req_d, req_b = materialise(req_pairs)
    ses_d, ses_b = materialise(jour_pairs)
    floor = inst.cfg.mechanism.floor
    retry_pairs = [{"src": a, "dst": b, "strength": round(r["strength"], 6), "via": list(r["via"])}
                   for (a, b), r in sorted(within.items()) if round(r["strength"], 6) > 0]

    def pack(grain, d_edges, b_edges, pairs_support):
        return {
            "description": f"trace-bench scoring target ({grain} grain): latent projection of the mechanism over (operation, outcome) tokens, projected first and floored second; every edge stored with its strength",
            "grain": grain, "variant": "twin" if twin else "latent", "default_floor": floor,
            "n_directed": len(d_edges), "n_bidirected": len(b_edges),
            "n_directed_at_floor": sum(1 for e in d_edges if e["strength"] >= floor),
            "n_bidirected_at_floor": sum(1 for e in b_edges if e["strength"] >= floor),
            "directed_acyclic_at_floor": is_acyclic([(e["src"], e["dst"]) for e in d_edges if e["strength"] >= floor]),
            "support_pairs": len(pairs_support),
            "support_op_pairs": [list(x) for x in sorted(pairs_support)],
            "directed": d_edges, "bidirected": b_edges,
            "bidirected_groups": groups,
            "retry_pairs": retry_pairs,
        }

    return pack("request", req_d, req_b, req_pairs), pack("session", ses_d, ses_b, jour_pairs), proj


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
