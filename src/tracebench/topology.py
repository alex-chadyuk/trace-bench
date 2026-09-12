"""Deployment topology: services, endpoints, the BFF, externals and call edges.

Sampled once at instantiation from the INSTANTIATE stream and recorded as
`topology/callgraph.json` (the lab's CallGraphTruth shape, caller -> callee)
and `topology/prior.json` (an edge-confidence prior, callee -> caller: the
direction in which outcomes propagate). The topology is a layered DAG at the
endpoint level — calls go strictly deeper — so every request tree is acyclic
and every backend endpoint sits at one depth.

Every call edge carries a call probability: the share of the caller's
requests that invoke the callee, drawn independently per request. The edges
say which calls can happen; the probabilities set how deep a request goes.
They are calibrated after the scenarios exist (`assign_call_probabilities`),
because the traffic each BFF endpoint carries weights the request-depth
distribution.

Request depth (pinned): the deepest backend-service layer a request's
invoked hops reach, with the BFF hop at layer 0; external endpoints are not a
service layer and set no depth. `depth_pmf[d-1]` is the share of requests at
depth d among requests that reach layer 1 at least (requests served without
any backend call are counted apart). This is the fitted constant's own
definition (service-chain depth below the root, root-only traces excluded).

Operation ids are dense from 0 in the order BFF endpoints, backend endpoints,
client operations, external endpoints; `kind` follows constants.KIND_*.
"""
from __future__ import annotations

import math
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
    p_call: float = 1.0    # share of the caller's requests that invoke this callee (independent per request)
    attached: bool = False # added by the reachability rule (no sampled call reached the callee)


@dataclass
class Topology:
    services: list[Service]
    ops: list[Op]
    edges: list[CallEdge]
    bff_ops: list[int]
    client_ops: list[int]
    external_ops: list[int]
    depth_levels: int
    calibration: dict = field(default_factory=dict)    # how the call probabilities were set (assign_call_probabilities)

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
            "calibration": self.calibration,
        }

    @classmethod
    def from_dict(cls, d):
        services = [Service(**s) for s in d["services"]]
        ops = [Op(**o) for o in d["ops"]]
        edges = [CallEdge(**{"p_call": 1.0, **e}) for e in d["edges"]]     # records older than call probabilities
        return cls(services, ops, edges, list(d["bff_ops"]), list(d["client_ops"]),
                   list(d["external_ops"]), int(d["depth_levels"]), dict(d.get("calibration", {})))


def endpoint_column(path):
    """`/v1/noun/verb` -> `_v1_noun_verb` (the convention `service:endpoint`
    column names follow in the lab's call-graph artifacts)."""
    return "".join("_" if c in "/{}." else c for c in path)


def _sample_pmf(rng, pmf):
    return int(np.searchsorted(np.cumsum(pmf), rng.random(), side="right"))


def _fanout_pmf(lam, max_k=4):
    """Callee count of a calling endpoint: Poisson(lam) truncated to 1..max_k."""
    w = np.array([0.0] + [lam ** k * math.exp(-lam) / math.factorial(k) for k in range(1, max_k + 1)])
    return w / w.sum()


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
    fanout_by_depth = [_fanout_pmf(float(cfg.topology.fanout_mean)).tolist() for _ in range(max(1, levels))]
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

    def add_edge(caller, callee, attached=False):
        e = CallEdge(
            index=len(edges), caller=caller, callee=callee,
            critical=bool(rng.random() < cfg.topology.criticality_share),
            rho=float(0.5 + 0.5 * rng.random()),
            cached=bool(rng.random() < cfg.topology.cache_hit_share),
            p_hit=float(0.3 + 0.6 * rng.random()),
            attached=attached,
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
    # Every endpoint above the deepest layer calls into the next layer; how often
    # a request continues is the call probability (assign_call_probabilities).
    for d in range(1, levels + 1):
        pmf = fanout_by_depth[min(d, len(fanout_by_depth) - 1)]
        deeper = [o for dd in range(d + 1, levels + 1) for o in ops_at_depth.get(dd, [])]
        for u in ops_at_depth.get(d, []):
            if not deeper:
                break
            k = max(1, _sample_pmf(rng, pmf))
            # prefer the next layer, fall back to any deeper op
            next_layer = ops_at_depth.get(d + 1, []) or deeper
            pick_callees(u, d, next_layer, k)
            if external_ops and rng.random() < 0.15:
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
            add_edge(caller, o.id, attached=True)
            reachable.update(_reach(ops, edges, caller))

    return Topology(services, ops, edges, bff_ops, client_ops, external_ops, levels)


# --- call probabilities and request depth ---------------------------------------------
CALL_P_MIN = 0.02              # no call edge is exercised on fewer than 2 % of its caller's requests
CALIBRATION_REQUESTS = 20000   # topology-only Monte Carlo requests per calibration step
CALIBRATION_MAX_STEPS = 6
CALIBRATION_TV = 0.02          # stop once the realised depth pmf is this close to the target


def service_layer(op):
    """The layer an invoked op contributes to request depth: its layer for a
    backend service, 0 for the BFF and for externals (not a service layer)."""
    return op.depth if op.kind == KIND_SERVICE else 0


def depth_distribution(depths, levels):
    """(pmf over layers 1..levels among requests that reach layer 1, share of
    requests that reach no backend layer) from per-request depths."""
    depths = np.asarray(depths, dtype=np.int64)
    n = len(depths)
    chain = depths[depths >= 1]
    if len(chain) == 0:
        return [0.0] * levels, (1.0 if n else 0.0)
    chain = np.minimum(chain, levels)
    pmf = [float(np.count_nonzero(chain == d) / len(chain)) for d in range(1, levels + 1)]
    return pmf, float(1.0 - len(chain) / n)


def bff_traffic_weights(topo, sset):
    """Relative request volume of each BFF endpoint: the weight of every
    scenario step that visits it (step attrition and retries ignored)."""
    w = {b: 0.0 for b in topo.bff_ops}
    for sc in sset.scenarios:
        for st in sc.steps:
            w[st.bff_op] += sc.weight
    return w


def nominal_warm_share(constants):
    """Stationary WARM share of a cache at nominal load, from the mechanism's
    own cache transition table."""
    from .mechanism import LOAD_VALUES, cache_table, stationary
    return float(stationary(cache_table(constants)[LOAD_VALUES.index("mid")])[0])


def simulate_request_depths(topo, weights, n_requests, seeds, warm_share):
    """Topology-only Monte Carlo of request depth. `n_requests` requests are
    split over BFF endpoints in proportion to `weights` (largest remainder);
    a callee is invoked when a caller is, the call draw falls under the edge's
    call probability and the call misses its cache (hit probability p_hit x
    warm_share on cached edges). `seeds[b]` seeds BFF b's draws, drawn in a
    fixed order, so repeated runs with the same seeds reuse the same numbers.
    Returns the per-request depths (0 = no backend layer reached)."""
    bffs = [b for b in topo.bff_ops if weights.get(b, 0.0) > 0]
    total = sum(weights[b] for b in bffs)
    if not bffs or total <= 0:
        return np.zeros(0, np.int64)
    raw = np.array([weights[b] / total * n_requests for b in bffs])
    alloc = np.floor(raw).astype(np.int64)
    alloc[np.argsort(-(raw - alloc), kind="stable")[: n_requests - int(alloc.sum())]] += 1
    out = []
    for b, n in zip(bffs, alloc.tolist()):
        if n == 0:
            continue
        rng = np.random.Generator(np.random.PCG64(seeds[b]))
        invoked = {b: np.ones(n, bool)}
        depth = np.zeros(n, np.int64)
        for v in topo.reachable_from(b)[1:]:
            inv = np.zeros(n, bool)
            for e in topo.caller_edges(v):
                if e.caller not in invoked:
                    continue
                u_call, u_hit = rng.random(n), rng.random(n)
                called = u_call < e.p_call
                if e.cached:
                    called &= u_hit >= e.p_hit * warm_share
                inv |= invoked[e.caller] & called
            invoked[v] = inv
            layer = service_layer(topo.ops[v])
            if layer:
                depth = np.where(inv, np.maximum(depth, layer), depth)
        out.append(depth)
    return np.concatenate(out)


def _clamp_call_p(p):
    return float(min(1.0, max(CALL_P_MIN, p)))


def _set_call_probabilities(topo, stop_by_layer):
    """Edge call probabilities from per-layer stop probabilities: an invoked
    layer-d endpoint with k backend callees invokes none of them (before
    caches) with probability stop_by_layer[d], so each of its edges carries
    1 - stop^(1/k); its external calls carry the same probability.

    A BFF endpoint makes its sampled layer-1 calls and its external calls on
    every request, so every request reaches layer 1. The m layer-1 endpoints
    attached to it only for reachability share one call per request between
    them (1/m each), and a deeper endpoint attached to it is called as often as
    one layer-1 endpoint continues."""
    ops = topo.ops
    for u in ops:
        if u.kind not in (KIND_BFF, KIND_SERVICE) or not u.callees:
            continue
        edges = topo.callee_edges(u.id)
        if u.kind == KIND_BFF:
            n_attached = sum(1 for e in edges if e.attached and ops[e.callee].depth == 1)
            for e in edges:
                callee = ops[e.callee]
                if e.attached and callee.depth == 1:
                    e.p_call = _clamp_call_p(1.0 / n_attached)
                elif callee.kind == KIND_EXTERNAL or callee.depth == 1:
                    e.p_call = 1.0
                else:
                    e.p_call = _clamp_call_p(1.0 - stop_by_layer.get(1, 1.0))
            continue
        k = sum(1 for e in edges if ops[e.callee].kind == KIND_SERVICE)
        stop = stop_by_layer.get(u.depth)
        p = 1.0 if stop is None or k == 0 else _clamp_call_p(1.0 - stop ** (1.0 / k))
        for e in edges:
            e.p_call = p


def assign_call_probabilities(topo, cfg, constants, sset, rng):
    """Set every edge's call probability so that request depth follows the
    configured `depth_pmf`, and record how (`topo.calibration`).

    The parameters are per-layer stop probabilities s_d (an invoked layer-d
    endpoint calls no deeper endpoint). A request that has reached layer d
    stops there when every one of its n_d invoked layer-d endpoints stops, so
    the closed form s_d = stop_d^(1/n_d), stop_d = pmf[d-1] / P(depth >= d),
    starts from n_d = f^d (f the mean sampled fan-out). Fan-out, attachments
    and caches make n_d heterogeneous, so a topology-only Monte Carlo
    (common random numbers across steps) measures the realised stop hazard h_d
    and refits the effective exponent n_d = log h_d / log s_d, up to
    CALIBRATION_MAX_STEPS steps or until the total variation to the target is
    within CALIBRATION_TV; the best step is kept. Uses the INSTANTIATE stream
    (after the scenarios) for the Monte Carlo seeds."""
    levels = topo.depth_levels
    pmf = [float(x) for x in cfg.topology.depth_pmf]
    surv = [sum(pmf[i:]) for i in range(levels)]
    target_stop = {d: (min(1.0, pmf[d - 1] / surv[d - 1]) if surv[d - 1] > 0 else 1.0) for d in range(1, levels)}
    fan = _fanout_pmf(float(cfg.topology.fanout_mean))
    f_mean = max(1.0, float(np.dot(np.arange(len(fan)), fan)))
    stop = {d: target_stop[d] ** (1.0 / f_mean ** d) for d in range(1, levels)}
    weights = bff_traffic_weights(topo, sset)
    seeds = {b: int(s) for b, s in zip(topo.bff_ops, rng.integers(0, 2 ** 63 - 1, size=len(topo.bff_ops)))}
    warm = nominal_warm_share(constants)
    best = None
    for step in range(1, CALIBRATION_MAX_STEPS + 1):
        _set_call_probabilities(topo, stop)
        depths = simulate_request_depths(topo, weights, CALIBRATION_REQUESTS, seeds, warm)
        realised, root_only = depth_distribution(depths, levels)
        tv = 0.5 * sum(abs(a - b) for a, b in zip(realised, pmf))
        if best is None or tv < best[0]:
            best = (tv, dict(stop), realised, root_only, step)
        if tv <= CALIBRATION_TV:
            break
        for d in range(1, levels):
            reached = int(np.count_nonzero(depths >= d))
            hazard = int(np.count_nonzero(depths == d)) / reached if reached else target_stop[d]
            s = stop[d]
            if 0.0 < hazard < 1.0 and 0.0 < s < 1.0:
                stop[d] = target_stop[d] ** (math.log(s) / math.log(hazard))
            elif hazard >= 1.0 and 0.0 < s < 1.0:
                stop[d] = s * s          # nothing continued: continue more often
            elif hazard <= 0.0 and 0.0 < s < 1.0:
                stop[d] = math.sqrt(s)   # nothing stopped: continue less often
    tv, stop, realised, root_only, steps = best
    _set_call_probabilities(topo, stop)
    p = np.array([e.p_call for e in topo.edges]) if topo.edges else np.zeros(0)
    topo.calibration = {
        "method": "per-layer stop probabilities fitted by topology-only Monte Carlo over request depth",
        "requests": CALIBRATION_REQUESTS, "steps": steps, "tv": round(float(tv), 6),
        "target_pmf": pmf, "realised_pmf": [round(x, 6) for x in realised], "root_only_share": round(root_only, 6),
        "layer_stop_probability": [round(float(stop[d]), 6) for d in range(1, levels)],
        "warm_share": round(warm, 6), "p_call_min": CALL_P_MIN,
        "share_at_min": round(float(np.mean(np.isclose(p, CALL_P_MIN))) if len(p) else 0.0, 6),
    }
    return topo.calibration


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
            "evidence": f"topology.edges[{e.index}]; p_call={e.p_call:.4f}",
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
              "evidence": f"topology.edges[{e.index}]; p_call={e.p_call:.4f}"} for e in topo.edges]
    return {
        "description": "trace-bench deployment topology as a candidate structural prior (callee -> caller)",
        "dataset": "trace-bench", "columns": columns, "default_prob": default_prob,
        "reverse_prob": reverse_prob, "edges": edges,
    }
