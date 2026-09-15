"""PRD scenarios 1, 2 and 18: the mechanism graph is the mechanism the
generator applies — node set = instantiated state variables, latent flag =
"no emitter", every edge carries a strength; one graph per regime with
changepoints and changed-edge sets."""
import copy

import numpy as np
import yaml

from tracebench.constants import EMITTED_GROUPS, KIND_CLIENT, KIND_SERVICE, T_VALUES
from tracebench.graphs import write_mechanism_graphs
from tracebench.instantiate import instantiate
from tracebench.mechanism import tv
from tracebench.record import read_json
from tracebench.topology import CALL_P_MIN
from xs_fixture import XS, xs_instantiation


def test_node_set_is_the_instantiated_state_variables():
    inst = xs_instantiation()
    g = inst.mechanism.graph_json(ctxmax=False)
    ids = {n["id"] for n in g["nodes"]}
    topo, cfg = inst.topo, inst.cfg
    backend = [op for op in topo.ops if op.kind != KIND_CLIENT]
    for op in backend:
        assert f"health:{op.id}" in ids
    for s in topo.services:
        if s.kind != 3 and s.kind != KIND_CLIENT and s.depth >= 0:
            assert f"load:{s.index}" in ids and f"pool:{s.index}" in ids and f"cache:{s.index}" in ids
    R = cfg.endpoints.retry.max_retries
    for op in topo.ops:
        if op.kind == KIND_SERVICE:
            assert all(f"A:{op.id}:{k}" in ids for k in range(R + 1))
            assert f"F:{op.id}" in ids and f"I:{op.id}" in ids
    for sc in inst.sset.scenarios:
        for st in sc.steps:
            for k in range(sc.max_retries + 1):
                assert f"T:bff:{sc.index}:{st.index}:{k}" in ids
                assert f"C:{sc.index}:{st.index}:{k}" in ids
    assert g["n_nodes"] == len(inst.mechanism.nodes)


def test_latent_flag_follows_the_emission_spec():
    inst = xs_instantiation()
    g = inst.mechanism.graph_json(ctxmax=False)
    for n in g["nodes"]:
        assert n["latent"] == (n["group"] not in EMITTED_GROUPS), n
    assert g["n_latent"] > 0
    assert any(n["latent"] and n["group"] == "health" for n in g["nodes"])


def test_every_edge_is_a_parent_dependence_with_a_strength():
    inst = xs_instantiation()
    mech = inst.mechanism
    g = inst.mechanism.graph_json(ctxmax=True)
    edges = {(e["src"], e["dst"]) for e in g["edges"]}
    loops = {(e["src"], e["dst"]) for e in g["self_loops"]}
    expected, expected_loops = set(), set()
    for cid, node in mech.nodes.items():
        for pid in node.parents:
            (expected_loops if pid == cid else expected).add((pid, cid))
    assert edges == expected and loops == expected_loops
    for e in g["edges"]:
        assert 0.0 <= e["strength"] <= 1.0
        assert e["strength_ctxmax"] is None or e["strength_ctxmax"] >= e["strength"] - 1e-9
        assert isinstance(e["context"], dict)
    # a strength is what the definition says: max TV over parent-value pairs at nominal context
    e = next(x for x in g["edges"] if x["src"].startswith("health:") and x["dst"].startswith("A:"))
    node, parent = mech.nodes[e["dst"]], mech.nodes[e["src"]]
    ctx = node.nominal_context(mech, exclude=e["src"])
    dists = []
    for v in parent.var.values:
        ctx[e["src"]] = v
        dists.append(node.dist(ctx))
    best = max(tv(a, b) for i, a in enumerate(dists) for b in dists[i + 1:])
    assert abs(best - e["strength"]) < 1e-6


def test_distributions_are_normalised_and_cache_edges_measured():
    inst = xs_instantiation()
    mech = inst.mechanism
    for cid, node in mech.nodes.items():
        d = node.dist(node.nominal_context(mech))
        assert abs(float(np.sum(d)) - 1.0) < 1e-9, cid
        assert (d >= -1e-12).all(), cid
    g = inst.mechanism.graph_json(ctxmax=False)
    cache_edges = [e for e in g["edges"] if e["src"].startswith("cache:") and e["dst"].startswith("I:")]
    assert cache_edges and all(e["strength"] > 0 for e in cache_edges)


def test_invoke_edges_carry_the_call_probability_times_the_cache_miss():
    """At the nominal context (caches WARM, other callers absent) an invoke
    edge caller -> callee has strength p_call x (1 - p_hit) on a cached call
    and p_call otherwise; a cache -> invoke edge has p_call x p_hit of the
    first cached call it fronts."""
    inst = xs_instantiation()
    topo, sset = inst.topo, inst.sset
    g = inst.mechanism.graph_json(ctxmax=False)
    call = {(e.caller, e.callee): e for e in topo.edges}

    def op_of(invoke_id):
        parts = invoke_id.split(":")
        return sset.scenarios[int(parts[2])].steps[int(parts[3])].bff_op if parts[1] == "bff" else int(parts[1])

    n_invoke = n_cache = 0
    for rec in g["edges"]:
        if not rec["dst"].startswith("I:") or rec["dst"].startswith("I:bff"):
            continue
        callee = op_of(rec["dst"])
        if rec["src"].startswith("I:"):
            e = call[(op_of(rec["src"]), callee)]
            assert e.p_call >= CALL_P_MIN
            expected = e.p_call * ((1 - e.p_hit) if e.cached else 1.0)
            n_invoke += 1
        elif rec["src"].startswith("cache:"):
            svc = int(rec["src"].split(":")[1])
            e = next(x for x in topo.caller_edges(callee) if x.cached and topo.ops[x.caller].service == svc)
            expected = e.p_call * e.p_hit
            n_cache += 1
        else:
            continue
        assert abs(rec["strength"] - expected) < 1e-6, (rec["src"], rec["dst"], rec["strength"], expected)
    assert n_invoke and n_cache


def test_graph_is_deterministic_across_builds():
    a = xs_instantiation().mechanism.graph_json(ctxmax=False)
    inst2 = instantiate(xs_instantiation().cfg, xs_instantiation().constants, 0)
    b = inst2.mechanism.graph_json(ctxmax=False)
    assert a == b


def test_regimes_emit_one_graph_each_with_changepoints(tmp_path):
    base = xs_instantiation()
    # no regime schedule: exactly one mechanism graph
    graphs = write_mechanism_graphs(base, tmp_path / "a")
    assert len(graphs) == 1
    cps = read_json(tmp_path / "a" / "graphs" / "changepoints.json")
    assert cps["n_regimes"] == 1 and cps["changepoints"] == []
    # a regime schedule: one graph per regime, changepoint times, changed edges
    data = yaml.safe_load(XS.read_text())
    data = copy.deepcopy(data)
    data["schedules"]["regimes"] = [{"name": "deploy-a", "at_s": 1800,
                                     "overlay": {"mechanism": {"error_rate_multiplier": 60.0}}}]
    from tracebench.config import parse_instance_config
    cfg = parse_instance_config(data)
    inst = instantiate(cfg, base.constants, 0)
    graphs = write_mechanism_graphs(inst, tmp_path / "b")
    assert [name for name, _ in graphs] == ["base", "deploy-a"]
    assert (tmp_path / "b" / "graphs" / "mechanism-graph.json").exists()
    assert (tmp_path / "b" / "graphs" / "mechanism-graph.r1.json").exists()
    cps = read_json(tmp_path / "b" / "graphs" / "changepoints.json")
    assert cps["n_regimes"] == 2
    assert cps["changepoints"][0]["at_s"] == 1800 and cps["changepoints"][0]["name"] == "deploy-a"
    changed = cps["changepoints"][0]["changed_edges"]
    assert changed and all({"src", "dst", "before", "after"} <= set(c) for c in changed)
    # the base regime graph equals the unscheduled graph (same topology, same seed)
    assert graphs[0][1]["edges"] == write_mechanism_graphs(base, tmp_path / "c")[0][1]["edges"]


def test_held_distributions_are_memoised_and_identical_to_a_cold_evaluation(monkeypatch):
    """The guard behind D-TB-17: the projection asks `dist_held` for the same
    (node, held context) once per particle, and a cold evaluation is a power
    iteration to 1e-13 — 6 h of the 10 h m job on 2026-09-13, and the whole 12 h
    cap on the other four. The cache must serve repeats without recomputing and
    must return exactly what a cold evaluation returns, or a corpus byte moves."""
    import tracebench.mechanism as M

    inst = xs_instantiation()
    mech = inst.mechanism
    lagged = [n for n in mech.nodes.values() if n.var.lag_self]
    assert lagged, "xs has no lag-1 state variables"
    calls = {"n": 0}
    cold = M.stationary

    def counting(K, *a, **k):
        calls["n"] += 1
        return cold(K, *a, **k)

    monkeypatch.setattr(M, "stationary", counting)
    for node in lagged[:24]:
        node.__dict__.pop("_held_cache", None)
        ctx = node.nominal_context(mech)
        first = node.dist_held(mech, ctx)
        after_first = calls["n"]
        assert after_first >= 1
        again = node.dist_held(mech, dict(ctx))          # an equal context, not the same dict
        assert calls["n"] == after_first, node.var.id      # served from the cache
        assert again is first and not again.flags.writeable
        node.__dict__.pop("_held_cache", None)
        fresh = node.dist_held(mech, ctx)                  # cold again
        assert calls["n"] == after_first + 1
        assert np.array_equal(fresh, first) and fresh.dtype == first.dtype, node.var.id
        assert abs(float(first.sum()) - 1.0) < 1e-12


def test_retry_attempts_are_held_absent_in_finals_and_following_attempts():
    """D-TB-20: a retry attempt is held ABSENT wherever it is a held parent —
    in the operation's final and in the attempt (or BFF invocation) that follows
    it — so a first attempt's strength is the controlled direct effect with no
    retry. Before, the last attempt's nominal `ok` made every earlier attempt
    inert on the final (A:v:0 -> F:v had strength 0)."""
    inst = xs_instantiation()
    mech = inst.mechanism
    n_finals = n_following = 0
    for nid, node in mech.nodes.items():
        parts = nid.split(":")
        if node.var.group == "final":
            attempts = list(node.parents)
            assert all(node.context_override.get(a) == "absent" for a in attempts[1:]), nid
            assert attempts[0] not in node.context_override
            n_finals += 1
        elif node.var.group == "attempt" and parts[0] == "A" and int(parts[2]) >= 2:
            assert node.context_override == {f"A:{parts[1]}:{int(parts[2]) - 1}": "absent"}, nid
            n_following += 1
        elif node.var.group == "invoke" and parts[1] == "bff" and int(parts[4]) >= 2:
            assert node.context_override == {f"T:bff:{parts[2]}:{parts[3]}:{int(parts[4]) - 1}": "absent"}, nid
            n_following += 1
    assert n_finals > 0 and n_following > 0
    g = mech.graph_json(ctxmax=False)
    first_to_final = [e for e in g["edges"] if e["src"].split(":")[0] in ("A", "C") and e["src"].endswith(":0") and e["dst"].startswith(("F:", "C:")) and e["dst"].endswith(("F:", ":F")) or (e["src"].startswith("A:") and e["src"].endswith(":0") and e["dst"].startswith("F:"))]
    assert first_to_final and all(e["strength"] == 1.0 for e in first_to_final), [e for e in first_to_final if e["strength"] != 1.0][:3]
    for e in first_to_final:
        assert all(v == "absent" for k, v in e["context"].items()), e
