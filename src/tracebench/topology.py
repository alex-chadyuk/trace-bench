"""Deployment topology: services, endpoints, the BFF, externals and call edges.

Sampled once at instantiation from the INSTANTIATE stream and recorded as
`topology/callgraph.json` (the lab's CallGraphTruth shape, caller -> callee)
and `topology/prior.json` (an edge-confidence prior, callee -> caller: the
direction in which outcomes propagate). The topology is a layered DAG at the
endpoint level — calls go strictly deeper — so every request tree is acyclic
and every backend endpoint sits at one depth.

Operation ids are dense from 0 in the order BFF endpoints, backend endpoints,
client operations, external endpoints; `kind` follows constants.KIND_*.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import KIND_BFF, KIND_CLIENT, KIND_EXTERNAL, KIND_SERVICE
from .naming import BFF_SERVICE, Namer


@dataclass
class Service:
    index: int
    name: str
    kind: int              # KIND_BFF | KIND_SERVICE | KIND_EXTERNAL
    depth: int             # 0 = BFF, 1.. = backend layer, externals = max depth + 1
    pods: list[str]
    host: str
    ip: str
    endpoints: list[int] = field(default_factory=list)   # op ids


@dataclass
class Op:
    id: int
    service: int           # Service.index
    name: str              # endpoint path (or client page)
    kind: int
    depth: int
    callees: list[int] = field(default_factory=list)      # edge indices
    callers: list[int] = field(default_factory=list)      # edge indices


@dataclass
class CallEdge:
    index: int
    caller: int            # op id
    callee: int            # op id
    critical: bool         # a failure of the callee is critical to the caller
    rho: float             # propagation probability of a callee failure into the caller's outcome
    cached: bool           # the call is fronted by a cache (a WARM cache skips it)
    p_hit: float           # cache hit probability when WARM


@dataclass
class Topology:
    services: list[Service]
    ops: list[Op]
    edges: list[CallEdge]
    bff_ops: list[int]
    client_ops: list[int]
    external_ops: list[int]
    depth_levels: int

    # --- derived accessors --------------------------------------------------------
    def op_by_id(self, op_id):
        return self.ops[op_id]

    def service_of(self, op_id):
        return self.services[self.ops[op_id].service]

    def callee_edges(self, op_id):
        return [self.edges[i] for i in self.ops[op_id].callees]

    def caller_edges(self, op_id):
        return [self.edges[i] for i in self.ops[op_id].callers]

    def reachable_from(self, root_op):
        """Ops reachable from `root_op` along call edges in topological order —
        calls go strictly deeper, so (depth, id) order is topological — each op
        once. Deterministic."""
        seen = {root_op}
        stack = [root_op]
        while stack:
            u = stack.pop()
            for e in self.callee_edges(u):
                if e.callee not in seen:
                    seen.add(e.callee)
                    stack.append(e.callee)
        return sorted(seen, key=lambda o: (self.ops[o].depth, o))

    def backend_ops(self):
        return [op.id for op in self.ops if op.kind == KIND_SERVICE]

    def column_name(self, op_id):
        """`service:endpoint` in the lab's call-graph column convention."""
        op = self.ops[op_id]
        return f"{self.services[op.service].name}:{endpoint_column(op.name)}"

    def to_dict(self):
        return {
            "services": [vars(s) for s in self.services],
            "ops": [vars(o) for o in self.ops],
            "edges": [vars(e) for e in self.edges],
            "bff_ops": self.bff_ops,
            "client_ops": self.client_ops,
            "external_ops": self.external_ops,
            "depth_levels": self.depth_levels,
        }

    @classmethod
    def from_dict(cls, d):
        services = [Service(**s) for s in d["services"]]
        ops = [Op(**o) for o in d["ops"]]
        edges = [CallEdge(**e) for e in d["edges"]]
        return cls(services, ops, edges, list(d["bff_ops"]), list(d["client_ops"]),
                   list(d["external_ops"]), int(d["depth_levels"]))


def endpoint_column(path):
    """`/v1/noun/verb` -> `_v1_noun_verb` (the convention `service:endpoint`
    column names follow in the lab's call-graph artifacts)."""
    return "".join("_" if c in "/{}." else c for c in path)


def _sample_pmf(rng, pmf):
    return int(np.searchsorted(np.cumsum(pmf), rng.random(), side="right"))


def build_topology(cfg, constants, rng, namer: Namer, scenario_steps):
    """Sample the deployment topology.

    cfg: InstanceConfig; constants: RealismConstants; rng: INSTANTIATE stream;
    scenario_steps: total number of client operations to mint (one per
    scenario step)."""
    counts = cfg.counts
    depth_pmf = list(cfg.topology.depth_pmf)
    levels = len(depth_pmf)
    # Fan-out is an instance TARGET (PRD Interface: "fan-out and depth targets"), not an
    # inherited constant: the fitted reference system is star-shaped and the named
    # instances must not be. A calling endpoint draws its callee count from a
    # Poisson(fanout_mean) truncated to 1..4; the realism report compares the
    # realised fan-out with the fitted constant and states the deviation.
    lam = float(cfg.topology.fanout_mean)
    import math
    w = np.array([0.0] + [lam ** k * math.exp(-lam) / math.factorial(k) for k in range(1, 5)])
    w = w / w.sum()
    fanout_by_depth = [w.tolist() for _ in range(max(1, levels))]
    pods_pmf = constants["pods_per_service"]

    services: list[Service] = []
    ops: list[Op] = []
    edges: list[CallEdge] = []

    # --- BFF ------------------------------------------------------------------------
    bff = Service(index=0, name=BFF_SERVICE, kind=KIND_BFF, depth=0,
                  pods=namer.pods(BFF_SERVICE, 1 + _sample_pmf(rng, pods_pmf)),
                  host=Namer.host(0), ip=Namer.internal_ip(0))
    services.append(bff)
    bff_ops = []
    for path in namer.endpoints(counts.bff_endpoints):
        op = Op(id=len(ops), service=0, name=path, kind=KIND_BFF, depth=0)
        ops.append(op)
        bff.endpoints.append(op.id)
        bff_ops.append(op.id)

    # --- backend services assigned to layers 1..levels ------------------------------
    # Layer sizes follow the survival function of the depth pmf so deeper layers
    # are smaller; every layer keeps at least one service.
    surv = np.cumsum(depth_pmf[::-1])[::-1]          # P(depth >= i+1)
    weights = surv / surv.sum()
    layer_of = []
    for i in range(counts.services):
        if i < levels:
            layer_of.append(i + 1)                     # guarantee one service per layer
        else:
            layer_of.append(1 + _sample_pmf(rng, weights))
    service_names = namer.services(counts.services)
    for i in range(counts.services):
        s = Service(index=len(services), name=service_names[i], kind=KIND_SERVICE, depth=layer_of[i],
                    pods=namer.pods(service_names[i], 1 + _sample_pmf(rng, pods_pmf)),
                    host=Namer.host(len(services)), ip=Namer.internal_ip(len(services)))
        services.append(s)
        for path in namer.endpoints(counts.endpoints_per_service):
            op = Op(id=len(ops), service=s.index, name=path, kind=KIND_SERVICE, depth=s.depth)
            ops.append(op)
            s.endpoints.append(op.id)

    # --- client operations (one per scenario step; wired by scenarios.py) ----------
    client_svc = Service(index=len(services), name="client", kind=KIND_CLIENT, depth=-1,
                         pods=[], host="", ip="")
    services.append(client_svc)
    client_ops = []
    for path in namer.client_ops(scenario_steps):
        op = Op(id=len(ops), service=client_svc.index, name=path, kind=KIND_CLIENT, depth=-1)
        ops.append(op)
        client_svc.endpoints.append(op.id)
        client_ops.append(op.id)

    # --- external services (one endpoint each, deepest layer) ------------------------
    external_ops = []
    for name in namer.externals(cfg.topology.external_services):
        s = Service(index=len(services), name=name, kind=KIND_EXTERNAL, depth=levels + 1,
                    pods=namer.pods(name, 1), host=Namer.host(len(services)), ip=Namer.internal_ip(len(services)))
        services.append(s)
        path = namer.endpoints(1)[0]
        op = Op(id=len(ops), service=s.index, name=path, kind=KIND_EXTERNAL, depth=levels + 1)
        ops.append(op)
        s.endpoints.append(op.id)
        external_ops.append(op.id)

    # --- call edges: each calling endpoint picks callees from strictly deeper layers -
    ops_at_depth = {d: [o.id for o in ops if o.kind == KIND_SERVICE and o.depth == d] for d in range(1, levels + 1)}
    # P(a depth-d endpoint calls on | it exists), derived from the TREE-level
    # depth pmf: a request tree that has reached depth d stops there iff none
    # of its ~f^d depth-d endpoints continues, so
    #     (1 - q_d)^(f^d) = P(max depth = d | max depth >= d),
    # with f the mean fan-out of a calling endpoint.
    fan_means = [sum(i * p for i, p in enumerate(row)) / max(1e-9, 1 - row[0]) for row in fanout_by_depth]
    f_mean = max(1.0, float(np.mean(fan_means)))

    def continue_prob(d):
        s = surv[d - 1] if d - 1 < len(surv) else 0.0
        if s <= 0:
            return 0.0
        stop_here = min(1.0, max(0.0, depth_pmf[d - 1] / s))
        k = f_mean ** d
        return float(max(0.0, min(1.0, 1.0 - stop_here ** (1.0 / k))))

    def add_edge(caller, callee):
        e = CallEdge(
            index=len(edges), caller=caller, callee=callee,
            critical=bool(rng.random() < cfg.topology.criticality_share),
            rho=float(0.5 + 0.5 * rng.random()),
            cached=bool(rng.random() < cfg.topology.cache_hit_share),
            p_hit=float(0.3 + 0.6 * rng.random()),
        )
        edges.append(e)
        ops[caller].callees.append(e.index)
        ops[callee].callers.append(e.index)

    def pick_callees(caller_id, depth_index, candidates, k):
        """k distinct callees, preferring distinct services (fan-out across services)."""
        if not candidates or k <= 0:
            return
        chosen = []
        pool = list(candidates)
        for _ in range(min(k, len(pool))):
            weights = np.ones(len(pool))
            used_services = {ops[c].service for c in chosen}
            for i, c in enumerate(pool):
                if ops[c].service in used_services:
                    weights[i] = 0.2
            weights /= weights.sum()
            j = _sample_pmf(rng, weights)
            chosen.append(pool.pop(j))
        for c in chosen:
            add_edge(caller_id, c)

    # BFF endpoints always call at least one layer-1 endpoint.
    for b in bff_ops:
        pmf = fanout_by_depth[0]
        k = max(1, _sample_pmf(rng, pmf))
        pick_callees(b, 0, ops_at_depth.get(1, []), k)
        if external_ops and rng.random() < 0.5:
            add_edge(b, external_ops[_sample_pmf(rng, np.ones(len(external_ops)) / len(external_ops))])
    # Backend endpoints continue deeper with the depth-conditioned probability.
    for d in range(1, levels + 1):
        pmf = fanout_by_depth[min(d, len(fanout_by_depth) - 1)]
        deeper = [o for dd in range(d + 1, levels + 1) for o in ops_at_depth.get(dd, [])]
        for u in ops_at_depth.get(d, []):
            if not deeper:
                break
            if rng.random() < continue_prob(d):
                k = max(1, _sample_pmf(rng, pmf))
                # prefer the next layer, fall back to any deeper op
                next_layer = ops_at_depth.get(d + 1, []) or deeper
                pick_callees(u, d, next_layer, k)
            elif external_ops and rng.random() < 0.15:
                add_edge(u, external_ops[_sample_pmf(rng, np.ones(len(external_ops)) / len(external_ops))])
    # Every backend endpoint must be reachable from some BFF endpoint: attach orphans.
    reachable = set()
    for b in bff_ops:
        reachable.update(_reach(ops, edges, b))
    for o in ops:
        if o.kind == KIND_SERVICE and o.id not in reachable:
            # attach to a random shallower caller (BFF for layer 1)
            # Candidate callers are shallower ops that are themselves reachable
            # (the BFF ops always are): attaching to a dead caller would mark the
            # subtree reachable without a path from any BFF endpoint.
            shallower = list(bff_ops) if o.depth == 1 else [
                c.id for c in ops if c.kind in (KIND_BFF, KIND_SERVICE) and 0 <= c.depth < o.depth and c.id in reachable]
            caller = shallower[_sample_pmf(rng, np.ones(len(shallower)) / len(shallower))]
            add_edge(caller, o.id)
            reachable.update(_reach(ops, edges, caller))

    return Topology(services, ops, edges, bff_ops, client_ops, external_ops, levels)


def _reach(ops, edges, root):
    seen = {root}
    stack = [root]
    while stack:
        u = stack.pop()
        for ei in ops[u].callees:
            v = edges[ei].callee
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


def callgraph_json(topo: Topology, derived, env_name):
    """The lab's CallGraphTruth artifact shape: caller -> callee as stored."""
    analyzed = []
    edges_out = []
    for op in topo.ops:
        if op.kind not in (KIND_BFF, KIND_SERVICE):
            continue
        svc = topo.services[op.service]
        analyzed.append({
            "caller_service": svc.name, "caller_endpoint": endpoint_column(op.name),
            "repo": "synthetic", "handler": "synthetic", "auth_mode": "synthetic", "nginx": "synthetic",
            "depth": 1, "evidence_class": "instantiation",
        })
    for e in topo.edges:
        cs, ce = topo.services[topo.ops[e.caller].service], topo.services[topo.ops[e.callee].service]
        edges_out.append({
            "caller_service": cs.name, "caller_endpoint": endpoint_column(topo.ops[e.caller].name),
            "callee_service": ce.name, "callee_endpoint": endpoint_column(topo.ops[e.callee].name),
            "confidence": 1.0,
            "why": "recorded at instantiation (deployment topology, not derived from any graph)",
            "evidence": f"topology.edges[{e.index}]",
        })
    return {
        "description": "trace-bench deployment call topology, recorded at instantiation; caller -> callee as stored, not pre-flipped",
        "env": env_name, "derived": derived,
        "analyzed_endpoints": analyzed, "edges": edges_out, "candidates": [],
        "no_edge_endpoints": [], "unresolved_endpoints": [], "opaque_caller_endpoints": [],
    }


def prior_json(topo: Topology, default_prob=0.01, reverse_prob=0.005):
    """Edge-confidence prior over `service:endpoint` columns. `from -> to` is
    the outcome-propagation direction, callee -> caller."""
    columns = [topo.column_name(op.id) for op in topo.ops if op.kind in (KIND_BFF, KIND_SERVICE, KIND_EXTERNAL)]
    edges = [{"from": topo.column_name(e.callee), "to": topo.column_name(e.caller), "prob": 1.0,
              "why": "deployment call edge; outcomes propagate callee -> caller",
              "evidence": f"topology.edges[{e.index}]"} for e in topo.edges]
    return {
        "description": "trace-bench deployment topology as a candidate structural prior (callee -> caller)",
        "dataset": "trace-bench", "columns": columns, "default_prob": default_prob,
        "reverse_prob": reverse_prob, "edges": edges,
    }
