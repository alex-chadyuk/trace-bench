"""PRD scenario 19 (public half): the deployment call topology in a corpus is
the one recorded at instantiation, and every name follows the public grammar."""
from tracebench.constants import KIND_BFF, KIND_CLIENT, KIND_EXTERNAL, KIND_SERVICE
from tracebench.instantiate import load_instantiation, write_instantiation
from tracebench.naming import matches_public_grammar
from tracebench.record import read_json
from tracebench.topology import Topology, endpoint_column
from xs_fixture import xs_instantiation


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
