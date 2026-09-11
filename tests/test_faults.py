"""PRD scenarios 7 (fault record and case labels) and 8 (a fault at least
doubles the named component's error rate over its interval while every
non-descendant component stays within one percentage point)."""
import numpy as np
import pyarrow.parquet as pq

from tracebench.record import read_json
from corpus_fixture import xs_corpus


def test_fault_record_and_case_labels():
    corpus = xs_corpus()
    faults = read_json(corpus / "labels" / "faults.json")["faults"]
    cases = read_json(corpus / "labels" / "cases.json")["cases"]
    assert len(faults) == 2 and len(cases) == 2
    f0 = faults[0]
    assert f0["component"] == "service:1" and f0["kind"] == "degrade" and (f0["start_s"], f0["end_s"]) == (1200, 1800)
    assert f0["forced_nodes"] and f0["forced_value"] == "degraded" and f0["indicator"] == "health"
    c0 = cases[0]
    assert c0["root_cause_component"] == f0["service"] and c0["indicator"] == "health" and c0["inject_time_s"] == 1200


def _spans(corpus):
    """span_id -> (op, outcome, tick, start_ms) of every oracle span."""
    out = {}
    for shard in sorted((corpus / "oracle").glob("shard=*/")):
        t = pq.read_table(shard / "spans.parquet", columns=["span_id", "op", "start_ms", "outcome", "tick"]).to_pydict()
        for sid, op, s, o, k in zip(t["span_id"], t["op"], t["start_ms"], t["outcome"], t["tick"]):
            out[sid] = (op, o, k, s)
    return out


def _error_rates(spans, t_lo_ms, t_hi_ms):
    rates = {}
    for op, o, _, s in spans.values():
        if t_lo_ms <= s < t_hi_ms:
            r = rates.setdefault(op, [0, 0])
            r[0] += 1
            r[1] += int(o in (1, 2, 3))
    return {op: (e / n if n else 0.0, n) for op, (n, e) in rates.items()}


def test_fault_doubles_error_rate_at_the_component_only():
    with_f = xs_corpus("latent", True, "base")
    no_f = xs_corpus("latent", False, "base")
    faults = read_json(with_f / "labels" / "faults.json")["faults"]
    inst = read_json(with_f / "instantiation.json")
    topo = inst["topology"]
    f = faults[0]
    lo, hi = int(f["start_s"] * 1000), int(f["end_s"] * 1000)
    sw, sn = _spans(with_f), _spans(no_f)
    rw, rn = _error_rates(sw, lo, hi), _error_rates(sn, lo, hi)
    forced_ops = {int(n.split(":")[1]) for n in f["forced_nodes"]}
    # ancestors along the call graph are the fault's causal descendants (callee -> caller)
    callers = {}
    for e in topo["edges"]:
        callers.setdefault(e["callee"], set()).add(e["caller"])
    downstream = set()
    stack = list(forced_ops)
    while stack:
        v = stack.pop()
        for u in callers.get(v, ()):
            if u not in downstream:
                downstream.add(u)
                stack.append(u)
    # journey descendants: later steps of every scenario that passes through an
    # affected BFF endpoint (the client retries or abandons), and their callees
    callees = {}
    for e in topo["edges"]:
        callees.setdefault(e["caller"], set()).add(e["callee"])
    affected = forced_ops | downstream
    journey = set()
    for sc in inst["scenarios"]["scenarios"]:
        hit = False
        for st in sc["steps"]:
            if hit:
                stack = [st["bff_op"], st["client_op"]]
                while stack:
                    v = stack.pop()
                    if v not in journey:
                        journey.add(v)
                        stack.extend(callees.get(v, ()))
            if st["bff_op"] in affected or st["client_op"] in affected:
                hit = True
    for op in forced_ops:
        assert rw[op][1] > 20, "too few invocations to test"
        assert rw[op][0] >= max(2 * rn[op][0], 0.02), (op, rw[op], rn[op])
    # Common random numbers: a hop present in both corpora (same session, step,
    # attempt and position, hence the same span id) at the same tick, whose
    # operation is not a call-graph descendant of the fault, has the identical
    # outcome. The fault changes which hops exist and when (a retried step
    # delays every later step of its journey, which then meets other latent
    # states), never the outcome of a hop it does not reach.
    matched = [sid for sid, v in sw.items() if sid in sn and v[0] not in affected and v[2] == sn[sid][2] and lo <= v[3] < hi]
    assert len(matched) > 100, "too few shared unaffected hops in the fault window to compare"
    changed = [sid for sid in matched if sw[sid][1] != sn[sid][1]]
    assert not changed, (len(changed), len(matched), [(sw[s], sn[s]) for s in changed[:5]])
    # Population-level rates of operations outside the fault's request and
    # journey descendants differ only by the sampling noise of the hops one
    # corpus has and the other has not (none may exist on a tiny instance).
    for op, (rate, n) in rn.items():
        if op in affected or op in journey or n < 50:
            continue
        n_min = min(n, rw[op][1])
        tol = max(0.01, 3.0 * (max(rate, 0.01) * (1 - max(rate, 0.01)) / n_min) ** 0.5)
        assert abs(rw[op][0] - rate) <= tol, (op, rw[op], (rate, n), tol)
