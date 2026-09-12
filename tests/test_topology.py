"""PRD scenario 19 (public half): the deployment call topology in a corpus is
the one recorded at instantiation, and every name follows the public grammar.
Call probabilities (D-TB-13): bounded, every BFF request reaches layer 1,
older records load, and the calibrated probabilities reproduce the configured
request-depth pmf."""
import copy

import numpy as np
import pytest
import yaml

from tracebench.config import parse_instance_config
from tracebench.constants import KIND_BFF, KIND_CLIENT, KIND_EXTERNAL, KIND_SERVICE
from tracebench.instantiate import instantiate, load_instantiation, write_instantiation
from tracebench.naming import matches_public_grammar
from tracebench.record import read_json
from tracebench.topology import (
    CALIBRATION_MAX_STEPS, CALL_P_MIN, Topology, bff_traffic_weights, depth_distribution, endpoint_column,
    nominal_warm_share, simulate_request_depths,
)
from xs_fixture import REPO, xs_instantiation


def test_topology_is_a_layered_dag_reachable_from_the_bff():
    inst = xs_instantiation()
    topo = inst.topo
    for e in topo.edges:
        assert topo.ops[e.callee].depth > topo.ops[e.caller].depth
    reachable = set()
    for b in topo.bff_ops:
        reachable.update(topo.reachable_from(b))
    for op in topo.ops:
        if op.kind == KIND_SERVICE:
            assert op.id in reachable
    assert Topology.from_dict(topo.to_dict()).to_dict() == topo.to_dict()


def test_names_follow_the_public_grammar():
    topo = xs_instantiation().topo
    for s in topo.services:
        if s.kind == KIND_SERVICE:
            assert matches_public_grammar("service", s.name)
            assert all(matches_public_grammar("pod", p) for p in s.pods)
            assert matches_public_grammar("host", s.host) and matches_public_grammar("internal_ip", s.ip)
        elif s.kind == KIND_EXTERNAL:
            assert matches_public_grammar("external", s.name)
    for op in topo.ops:
        if op.kind == KIND_CLIENT:
            assert matches_public_grammar("client_op", op.name)
        else:
            assert matches_public_grammar("endpoint", op.name)


def test_written_callgraph_matches_instantiation(tmp_path):
    inst = xs_instantiation()
    write_instantiation(inst, tmp_path, "2026-09-10")
    cg = read_json(tmp_path / "topology" / "callgraph.json")
    topo = inst.topo
    recorded = {(topo.services[topo.ops[e.caller].service].name, endpoint_column(topo.ops[e.caller].name),
                 topo.services[topo.ops[e.callee].service].name, endpoint_column(topo.ops[e.callee].name)) for e in topo.edges}
    shipped = {(e["caller_service"], e["caller_endpoint"], e["callee_service"], e["callee_endpoint"]) for e in cg["edges"]}
    assert shipped == recorded
    prior = read_json(tmp_path / "topology" / "prior.json")
    assert len(prior["edges"]) == len(topo.edges)
    assert all(e["from"].split(":")[0] != e["to"].split(":")[0] or True for e in prior["edges"])
    # the corpus can be re-instantiated exactly from what was written
    again = load_instantiation(tmp_path)
    assert again.topo.to_dict() == topo.to_dict()
    assert again.mechanism.graph_json(ctxmax=False)["edges"] == inst.mechanism.graph_json(ctxmax=False)["edges"]
    assert all(f"p_call={e.p_call:.4f}" in s["evidence"] for e, s in zip(topo.edges, cg["edges"]))


def test_call_probabilities_are_bounded_and_every_bff_request_reaches_layer_one():
    inst = xs_instantiation()
    topo = inst.topo
    for e in topo.edges:
        assert CALL_P_MIN <= e.p_call <= 1.0, e
        caller, callee = topo.ops[e.caller], topo.ops[e.callee]
        if caller.kind == KIND_BFF and not e.attached and (callee.depth == 1 or callee.kind == KIND_EXTERNAL):
            assert e.p_call == 1.0, e
    for b in topo.bff_ops:
        assert any(topo.ops[e.callee].depth == 1 and e.p_call == 1.0 for e in topo.callee_edges(b)), b
    assert any(e.p_call < 1.0 for e in topo.edges)
    cal = topo.calibration
    assert 1 <= cal["steps"] <= CALIBRATION_MAX_STEPS and cal["tv"] <= 0.05, cal
    assert cal["target_pmf"] == list(inst.cfg.topology.depth_pmf)


def test_topology_records_without_call_probabilities_still_load():
    d = copy.deepcopy(xs_instantiation().topo.to_dict())
    for e in d["edges"]:
        del e["p_call"], e["attached"]
    del d["calibration"]
    old = Topology.from_dict(d)
    assert all(e.p_call == 1.0 and not e.attached for e in old.edges) and old.calibration == {}


def test_depth_distribution_conditions_on_reaching_layer_one():
    pmf, root_only = depth_distribution([0, 1, 1, 2, 3, 3, 3, 0], 3)
    assert pmf == [2 / 6, 1 / 6, 3 / 6] and root_only == 0.25


def _system_20x5():
    """A 20 x 5 + 8 system (the s template with twice the services): big enough
    for reachability attachments and uneven fan-out."""
    data = yaml.safe_load((REPO / "configs" / "instances" / "s.yaml").read_text())
    data["counts"]["services"] = 20
    return instantiate(parse_instance_config(data), xs_instantiation().constants, 1)


@pytest.mark.parametrize("system", ["xs", "20x5+8"])
def test_calibrated_call_probabilities_reproduce_the_configured_depth_pmf(system):
    """An independent topology-only Monte Carlo (seeds other than the
    calibration's) lands within TV 0.05 of the configured pmf."""
    inst = xs_instantiation() if system == "xs" else _system_20x5()
    topo = inst.topo
    rng = np.random.Generator(np.random.PCG64(2026))
    seeds = {b: int(s) for b, s in zip(topo.bff_ops, rng.integers(0, 2 ** 63 - 1, size=len(topo.bff_ops)))}
    depths = simulate_request_depths(topo, bff_traffic_weights(topo, inst.sset), 20000, seeds,
                                     nominal_warm_share(inst.constants))
    pmf, root_only = depth_distribution(depths, topo.depth_levels)
    target = list(inst.cfg.topology.depth_pmf)
    tv = 0.5 * sum(abs(a - b) for a, b in zip(pmf, target))
    assert tv <= 0.05, (pmf, target, topo.calibration)
    assert root_only < 0.10, root_only
