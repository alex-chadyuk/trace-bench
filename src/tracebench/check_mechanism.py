"""Ground truth by construction, made checkable (PRD scenario 21).

For every recorded edge P -> C the simulation is re-run with P forced to each
of its values and C's other parents forced to the recorded nominal context;
the child's empirical distribution under each forcing is compared and the
largest total-variation distance must match the recorded strength within
tolerance. For a sample of non-edges the same procedure must show a strength
of zero: with C's parents held, no other variable moves C.

Tick-level children are measured over a long latent-only run (the recorded
strength is the stationary one); event-level children over a request run at
a high arrival rate. Every forcing goes through the same hooks that faults
and the twin use, so the check exercises the generator itself.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .constants import CLIENT_T_VALUES, GRAPHS_DIR, MECHANISM_CHECK_JSON, MECHANISM_GRAPH_JSON, REPORTS_DIR, T_VALUES
from .engine import Engine, OUTCOME_ABSENT
from .instantiate import load_instantiation
from .latents import Forcing, Slots, initial_state, simulate_latents
from .log import log
from .mechanism import (
    AUTH_VALUES, CACHE_VALUES, HEALTH_VALUES, INTENSITY_VALUES, LOAD_VALUES, NET_VALUES, POOL_VALUES, PRESENCE_VALUES, tv,
)
from .record import RunRecord, read_json, write_json

TICK_HORIZON = 20000
TICK_BURNIN = 2000
EVENT_TICKS = 240
EVENT_SPILL = 4000          # journeys spill past the arrival horizon; forcings cover the whole slice
EVENT_RPS = 25.0
TOLERANCE = 0.03
# A callee's duration varies within its class band and reaches its caller's total;
# a categorical graph cannot carry that residual channel, so non-edges are held to
# a looser tolerance and the largest residual is reported (D-TB-9).
NON_EDGE_TOLERANCE = 0.05


class Forcer:
    """Translates mechanism node ids + values into engine/latent forcings."""

    def __init__(self, inst):
        self.inst = inst
        self.slots = Slots.build(inst.topo)
        self.mech = inst.mechanism

    def supports(self, node_id):
        g = node_id.split(":")[0]
        return g in ("intensity", "load", "pool", "cache", "health", "net", "auth", "I", "A", "F", "T", "C")

    def apply(self, engine: Engine, tick_forcings: list, node_id, value, horizon):
        parts = node_id.split(":")
        g = parts[0]
        if g == "intensity":
            tick_forcings.append(Forcing("intensity", np.zeros(0, np.int64), INTENSITY_VALUES.index(value), 0, horizon))
        elif g in ("load", "pool", "cache"):
            vals = {"load": LOAD_VALUES, "pool": POOL_VALUES, "cache": CACHE_VALUES}[g]
            tick_forcings.append(Forcing(g, np.array([self.slots.slot_of_svc[int(parts[1])]]), vals.index(value), 0, horizon))
        elif g == "health":
            tick_forcings.append(Forcing("health", np.array([self.slots.slot_of_op[int(parts[1])]]), HEALTH_VALUES.index(value), 0, horizon))
        elif g == "net":
            engine.force_session["net"] = NET_VALUES.index(value)
        elif g == "auth":
            engine.force_session["auth"] = AUTH_VALUES.index(value)
        elif g == "I" and parts[1] == "bff":
            engine.force_bff_invoke[(int(parts[2]), int(parts[3]), int(parts[4]))] = value == "present"
        elif g == "I":
            engine.force_invoke[int(parts[1])] = value == "present"
        elif g == "A":
            engine.force_attempt[(int(parts[1]), int(parts[2]))] = T_VALUES.index(value)
            if value != "absent":
                engine.force_invoke.setdefault(int(parts[1]), True)
        elif g == "F":
            engine.force_final[int(parts[1])] = T_VALUES.index(value)
            if value != "absent":
                engine.force_invoke.setdefault(int(parts[1]), True)
        elif g == "T":
            key = (int(parts[2]), int(parts[3]), int(parts[4]))
            engine.force_bff[key] = T_VALUES.index(value)
            if value != "absent":
                engine.force_bff_invoke.setdefault(key, True)
        elif g == "C" and parts[3] == "F":
            engine.force_client_final[(int(parts[1]), int(parts[2]))] = CLIENT_T_VALUES.index(value)
        elif g == "C":
            key = (int(parts[1]), int(parts[2]), int(parts[3]))
            engine.force_client[key] = CLIENT_T_VALUES.index(value)
            if value != "absent":
                engine.force_bff_invoke.setdefault(key, True)
        else:
            raise ValueError(node_id)


def _is_tick(node_id):
    return node_id.split(":")[0] in ("intensity", "load", "pool", "cache", "health")


def empirical_distribution(inst, forcer: Forcer, child_id, forcings_spec, seed, parent_id=None):
    """Run with the given {node: value} forcings and return the child's
    empirical distribution over its values (and the sample count)."""
    mech = inst.mechanism
    node = mech.nodes[child_id]
    values = node.var.values
    engine = Engine(inst, seed, forcer.slots, [])
    tick_forcings = []
    horizon = TICK_HORIZON if _is_tick(child_id) else EVENT_TICKS
    force_until = horizon if _is_tick(child_id) else horizon + EVENT_SPILL
    for nid, val in forcings_spec.items():
        forcer.apply(engine, tick_forcings, nid, val, force_until)
    if _is_tick(child_id):
        # intensity is exogenous: under no forcing it follows the profile; hold it at nominal
        if "intensity" not in forcings_spec:
            tick_forcings.append(Forcing("intensity", np.zeros(0, np.int64), 0, 0, horizon))
        sl, _ = simulate_latents(inst, seed, forcer.slots, initial_state(forcer.slots), 0, horizon, tick_forcings)
        g, rest = child_id.split(":")[0], child_id.split(":")[1:]
        if g == "intensity":
            arr = sl.intensity
        elif g == "health":
            arr = sl.health[:, forcer.slots.slot_of_op[int(rest[0])]]
        else:
            arr = getattr(sl, g)[:, forcer.slots.slot_of_svc[int(rest[0])]]
        arr = arr[TICK_BURNIN:]
        counts = np.bincount(arr, minlength=len(values)).astype(np.float64)
        return counts / counts.sum(), int(counts.sum())
    # event-level child: hold every tick latent at nominal unless forced, run requests
    if "intensity" not in forcings_spec:
        tick_forcings.append(Forcing("intensity", np.zeros(0, np.int64), 0, 0, force_until))
    S, O = len(forcer.slots.svc_of_slot), len(forcer.slots.op_of_slot)
    for arr, n in (("load", S), ("pool", S), ("cache", S), ("health", O)):
        tick_forcings.append(Forcing(arr, np.arange(n), 0, 0, force_until, "nominal"))
    # forced latents must win over the nominal hold: apply nominal first, then specific
    tick_forcings.sort(key=lambda f: 0 if f.label == "nominal" else 1)
    engine.base_rps_override = EVENT_RPS
    sl, _ = simulate_latents(inst, seed, forcer.slots, initial_state(forcer.slots), 0, force_until, tick_forcings)
    res = engine.run_shard(0, 0, horizon, sl)
    required = [int(k.split(":")[1]) for k, v in forcings_spec.items()
                if k.startswith("I:") and k.split(":")[1] != "bff" and v == "present"]
    d, n_pop = _child_distribution(inst, res, child_id, parent_id, required_ops=required)
    return d, n_pop


def _realised_filter(inst, res, parent_id):
    """(request_rows, session_rows) where an outcome parent was realised: an
    invoked op for A/F parents, an existing attempt for T/C parents; None
    means no restriction (invocation parents, latents)."""
    if parent_id is None:
        return None, None
    parts = parent_id.split(":")
    g = parts[0]
    reqs, hops, cl = res.requests.cols, res.hops.cols, res.clients.cols
    empty = np.zeros(0, np.int64)
    if g in ("A", "F", "T", "C"):
        if not reqs or not hops:
            return empty, empty
    if g in ("A", "F") and parts[1] != "bff":
        op = int(parts[1])
        m = (hops["op"] == op) & (hops["attempt"] == 0) if len(hops.get("op", [])) else np.zeros(0, bool)
        return np.unique(hops["request_row"][m]), None
    if g == "I":
        return None, None
    if g == "T":
        sc, j, k = int(parts[2]), int(parts[3]), int(parts[4])
        m = (reqs["step"] == j) & (reqs["attempt"] == k) if len(reqs.get("step", [])) else np.zeros(0, bool)
        srows = reqs["session_row"][m]
        srows = srows[res.sessions["scenario"][srows] == sc]
        return None, np.unique(srows)
    if g == "C":
        sc, j = int(parts[1]), int(parts[2])
        m = (cl["step"] == j) if len(cl.get("step", [])) else np.zeros(0, bool)
        if parts[3] != "F":
            m = m & (cl["attempt"] == int(parts[3]))
        srows = cl["session_row"][m]
        srows = srows[res.sessions["scenario"][srows] == sc]
        return None, np.unique(srows)
    return None, None


def _child_distribution(inst, res, child_id, parent_id=None, required_ops=()):
    mech = inst.mechanism
    node = mech.nodes[child_id]
    values = node.var.values
    parts = child_id.split(":")
    g = parts[0]
    reqs, hops, cl = res.requests.cols, res.hops.cols, res.clients.cols
    counts = np.zeros(len(values))
    if not reqs:
        counts[-1] = 1.0
        return counts, 0
    req_filter, sess_filter = _realised_filter(inst, res, parent_id)
    if g in ("I", "A", "F") and parts[1] != "bff":
        op = int(parts[1])
        reach_ops = {b for b in inst.topo.bff_ops
                     if op in inst.topo.reachable_from(b) and all(r in inst.topo.reachable_from(b) for r in required_ops)}
        pop = np.isin(reqs["bff_op"], list(reach_ops)) if len(reqs.get("bff_op", [])) else np.zeros(0, bool)
        if req_filter is not None:
            pop = pop & np.isin(reqs["request_row"], req_filter)
        if sess_filter is not None:
            pop = pop & np.isin(reqs["session_row"], sess_filter)
        n_pop = int(pop.sum())
        if n_pop == 0:
            counts[-1] = 1.0          # no request could invoke the op: absent
            return counts, 0
        req_rows = reqs["request_row"][pop]
        sel = (hops["op"] == op) & np.isin(hops["request_row"], req_rows) if len(hops.get("op", [])) else np.zeros(0, bool)
        if g == "I":
            present = len(np.unique(hops["request_row"][sel])) if sel.any() else 0
            counts[0], counts[1] = present, n_pop - present
        elif g == "A":
            k = int(parts[2])
            m = sel & (hops["attempt"] == k)
            out = hops["outcome"][m]
            for o in out:
                counts[int(o)] += 1
            counts[len(values) - 1] += n_pop - len(out)
        else:  # F: final = last attempt
            by_req = {}
            for r, k, o in zip(hops["request_row"][sel], hops["attempt"][sel], hops["outcome"][sel]):
                if r not in by_req or k > by_req[r][0]:
                    by_req[r] = (k, o)
            for _, o in by_req.values():
                counts[int(o)] += 1
            counts[len(values) - 1] += n_pop - len(by_req)
        return counts / max(counts.sum(), 1), n_pop
    if g in ("I", "T") and parts[1] == "bff":
        sc, j, k = int(parts[2]), int(parts[3]), int(parts[4])
        sess = res.sessions
        pop = sess["scenario"] == sc
        if sess_filter is not None:
            pop = pop & np.isin(np.arange(len(pop)), sess_filter)
        n_pop = int(pop.sum())
        srows = np.nonzero(pop)[0]
        m = np.isin(reqs["session_row"], srows) & (reqs["step"] == j) & (reqs["attempt"] == k) if len(reqs.get("step", [])) else np.zeros(0, bool)
        if g == "I":
            counts[0], counts[1] = int(m.sum()), n_pop - int(m.sum())
        else:
            for o in reqs["outcome"][m]:
                counts[int(o)] += 1
            counts[len(values) - 1] += n_pop - int(m.sum())
        return counts / max(counts.sum(), 1), n_pop
    if g == "C":
        sc, j = int(parts[1]), int(parts[2])
        sess = res.sessions
        pop = sess["scenario"] == sc
        if sess_filter is not None:
            pop = pop & np.isin(np.arange(len(pop)), sess_filter)
        n_pop = int(pop.sum())
        srows = np.nonzero(pop)[0]
        if parts[3] == "F":
            m = np.isin(cl["session_row"], srows) & (cl["step"] == j) if len(cl.get("step", [])) else np.zeros(0, bool)
            by = {}
            for r, k, o in zip(cl["session_row"][m], cl["attempt"][m], cl["outcome"][m]):
                if r not in by or k > by[r][0]:
                    by[r] = (k, o)
            for _, o in by.values():
                counts[int(o)] += 1
            counts[2] += n_pop - len(by)
        else:
            k = int(parts[3])
            m = np.isin(cl["session_row"], srows) & (cl["step"] == j) & (cl["attempt"] == k) if len(cl.get("step", [])) else np.zeros(0, bool)
            for o in cl["outcome"][m]:
                counts[int(o)] += 1
            counts[2] += n_pop - int(m.sum())
        return counts / max(counts.sum(), 1), n_pop
    raise ValueError(child_id)


def check_edge(inst, forcer, edge, seed):
    parent, child = edge["src"], edge["dst"]
    pnode = inst.mechanism.nodes[parent]
    dists = []
    for v in pnode.var.values:
        if not inst.mechanism.compatible(parent, v, edge["context"]):
            continue
        spec = dict(edge["context"])
        spec[parent] = v
        d, _ = empirical_distribution(inst, forcer, child, spec, seed, parent_id=parent if v != "absent" else None)
        dists.append(d)
    best = max(tv(a, b) for a, b in itertools.combinations(dists, 2)) if len(dists) > 1 else 0.0
    return {"src": parent, "dst": child, "recorded": edge["strength"], "empirical": round(best, 4),
            "abs_diff": round(abs(best - edge["strength"]), 4), "pass": abs(best - edge["strength"]) <= TOLERANCE,
            "dists": [[round(float(x), 4) for x in d] for d in dists]}


def check_non_edge(inst, forcer, parent, child, seed):
    node = inst.mechanism.nodes[child]
    ctx = node.nominal_context(inst.mechanism)
    pnode = inst.mechanism.nodes[parent]
    dists = []
    for v in pnode.var.values:
        if not inst.mechanism.compatible(parent, v, ctx):
            continue
        spec = dict(ctx)
        spec[parent] = v
        d, n_pop = empirical_distribution(inst, forcer, child, spec, seed, parent_id=parent if v != "absent" else None)
        if n_pop == 0:
            return None
        dists.append(d)
    best = max(tv(a, b) for a, b in itertools.combinations(dists, 2)) if len(dists) > 1 else 0.0
    return {"src": parent, "dst": child, "recorded": 0.0, "empirical": round(best, 4), "abs_diff": round(best, 4),
            "pass": best <= NON_EDGE_TOLERANCE, "non_edge": True}


def run_check(corpus_dir, max_edges=None, n_non_edges=8, seed=12345, edge_filter=None):
    corpus_dir = Path(corpus_dir)
    inst = load_instantiation(corpus_dir)
    forcer = Forcer(inst)
    graph = read_json(corpus_dir / GRAPHS_DIR / MECHANISM_GRAPH_JSON)
    edges = [e for e in graph["edges"] if forcer.supports(e["src"]) and forcer.supports(e["dst"])]
    if edge_filter is not None:
        edges = [e for e in edges if edge_filter(e)]
    if max_edges is not None:
        rng = np.random.Generator(np.random.PCG64(seed))
        edges = [edges[i] for i in sorted(rng.choice(len(edges), size=min(max_edges, len(edges)), replace=False))]
    results = [check_edge(inst, forcer, e, seed) for e in edges]
    # non-edges: pairs (P, C) where P is not a parent of C, P supported, C event- or tick-level
    rng = np.random.Generator(np.random.PCG64(seed + 1))
    ids = [i for i in inst.mechanism.order if forcer.supports(i)]
    non = []
    tries = 0
    while len(non) < n_non_edges and tries < 500:
        tries += 1
        p, c = ids[int(rng.integers(len(ids)))], ids[int(rng.integers(len(ids)))]
        if p == c or p in inst.mechanism.nodes[c].parents or c in inst.mechanism.nodes[p].parents:
            continue
        if not inst.mechanism.nodes[c].parents:
            continue          # roots (intensity, session latents) have nothing to hold
        if _is_tick(c) and not _is_tick(p):
            continue
        r = check_non_edge(inst, forcer, p, c, seed)
        if r is None:
            continue          # no shared population (another scenario / an op outside the child's trees)
        non.append(r)
    report = {"tolerance": TOLERANCE, "non_edge_tolerance": NON_EDGE_TOLERANCE,
              "max_non_edge_residual": max([r["empirical"] for r in non], default=0.0),
              "n_edges_checked": len(results), "n_edges_supported": len(edges),
              "n_edges_total": len(graph["edges"]), "n_pass": sum(r["pass"] for r in results),
              "n_non_edges": len(non), "n_non_edge_pass": sum(r["pass"] for r in non),
              "edges": results, "non_edges": non}
    report["all_pass"] = report["n_pass"] == len(results) and report["n_non_edge_pass"] == len(non)
    return report


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", required=True)
    p.add_argument("--max-edges", type=int, default=None, help="check a random sample of edges (default: all)")
    p.add_argument("--non-edges", type=int, default=8)
    p.add_argument("--seed", type=int, default=12345)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    rec = RunRecord(args.corpus, "check_mechanism", vars(args))
    report = run_check(args.corpus, args.max_edges, args.non_edges, args.seed)
    write_json(Path(args.corpus) / REPORTS_DIR / MECHANISM_CHECK_JSON, report)
    from .manifest import refresh_manifest
    refresh_manifest(args.corpus)  # the report is a new file of the corpus
    rec.finish({k: v for k, v in report.items() if k not in ("edges", "non_edges")})
    log({"event": "check_mechanism", "all_pass": report["all_pass"], "n_pass": report["n_pass"], "n_checked": report["n_edges_checked"]})
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
