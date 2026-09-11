"""PRD scenarios 3, 3a (target half), 5 and 6: the latent projection, the
twin's acyclic bidirected-free target, the floor sweep and the coarsened views."""
import numpy as np

from tracebench.constants import FLOOR_SWEEP
from tracebench.graphs import write_graph_artifacts
from tracebench.mechanism import FunctionNode, StateVar
from tracebench.projection import Projector, build_targets, coarsen, floor_sensitivity, is_acyclic
from tracebench.record import read_json
from xs_fixture import xs_instantiation


class TinyMechanism:
    """A hand-built mechanism exposing the interface the projector needs."""

    def __init__(self):
        self.nodes = {}
        self.order = []

    def add(self, node):
        self.nodes[node.var.id] = node
        self.order.append(node.var.id)
        return node

    def latent_ids(self):
        return [i for i in self.order if self.nodes[i].var.latent]


def _tok(vid, group, op, values, latent=False, derived=False):
    return StateVar(vid, group, "event", values, latent=latent, derived=derived, token_op=op)


def _binary_child(parent, p_if_a, p_if_b):
    def fn(ctx):
        p = p_if_a if ctx[parent] == "ok" else p_if_b
        return (p, 1 - p)
    return fn


def test_latent_mediator_gives_directed_edge_and_latent_confounder_gives_bidirected():
    m = TinyMechanism()
    m.add(FunctionNode(_tok("X", "attempt", 0, ("ok", "err")), [], lambda ctx: (0.9, 0.1)))
    # L is a latent mediator X -> L -> Y
    m.add(FunctionNode(_tok("L", "health", None, ("ok", "err"), latent=True), ["X"], _binary_child("X", 0.95, 0.2)))
    m.add(FunctionNode(_tok("Y", "attempt", 1, ("ok", "err")), ["L"], _binary_child("L", 0.9, 0.3)))
    # H is a latent common cause of U and V, and a derived node D mediates U -> D -> W
    m.add(FunctionNode(_tok("H", "pool", None, ("ok", "err"), latent=True), [], lambda ctx: (0.8, 0.2)))
    m.add(FunctionNode(_tok("U", "attempt", 2, ("ok", "err")), ["H"], _binary_child("H", 0.95, 0.4)))
    m.add(FunctionNode(_tok("V", "attempt", 3, ("ok", "err")), ["H"], _binary_child("H", 0.9, 0.5)))
    m.add(FunctionNode(_tok("D", "final", 2, ("ok", "err"), derived=True), ["U"], _binary_child("U", 1.0, 0.0)))
    m.add(FunctionNode(_tok("W", "attempt", 4, ("ok", "err")), ["D"], _binary_child("D", 0.9, 0.2)))
    # Z has no path to anything
    m.add(FunctionNode(_tok("Z", "attempt", 5, ("ok", "err")), [], lambda ctx: (0.5, 0.5)))
    proj = Projector(m, twin=False)
    directed, within = proj.directed_token_edges()
    groups = proj.bidirected_groups()
    d = {k: v["strength"] for k, v in directed.items()}
    # mediator collapses: X -> Y with the composed strength |P(Y=err|X=err) - P(Y=err|X=ok)|
    y_err_x_ok = 0.95 * 0.1 + 0.05 * 0.7
    y_err_x_err = 0.2 * 0.1 + 0.8 * 0.7
    assert abs(d[("0:err", "1:err")] - abs(y_err_x_err - y_err_x_ok)) < 1e-9
    assert ("1:err", "0:err") not in d                       # direction preserved
    assert d[("2:err", "4:err")] > 0.5                          # derived mediator collapses too
    assert not any(k[0].startswith("5:") or k[1].startswith("5:") for k in d)   # no path, no edge
    assert not any(k[0].startswith("0:") and k[1].startswith("2:") for k in d)
    assert not within
    # confounder: exactly one group, via H, over U's and V's tokens
    assert len(groups) == 1 and groups[0]["latent"] == "H"
    members = groups[0]["members"]
    assert set(members) == {"2:ok", "2:err", "3:ok", "3:err"}
    delta_u = abs((1 - 0.4) - (1 - 0.95))      # P(U=err | H=err) - P(U=err | H=ok)
    delta_v = abs((1 - 0.5) - (1 - 0.9))
    assert abs(max(members["2:err"]) - delta_u) < 1e-9 and abs(max(members["3:err"]) - delta_v) < 1e-9
    # nothing bidirected via the mediator L (it has a single token endpoint)
    assert all(g["latent"] != "L" for g in groups)


def test_xs_targets_have_expected_shape():
    inst = xs_instantiation()
    req, ses, proj = build_targets(inst)
    assert req["n_directed"] > 0 and req["n_bidirected"] > 0
    assert req["directed_acyclic_at_floor"]
    for e in req["directed"]:
        assert 0 < e["strength"] <= 1 and e["src"].split(":")[0] != e["dst"].split(":")[0]
    for e in req["bidirected"]:
        assert 0 < e["strength"] <= 1 and e["a"] < e["b"]
    # no latent node appears in the target; every via is a latent for bidirected edges
    latent = set(inst.mechanism.latent_ids())
    assert all(e["via"] in latent for e in req["bidirected"])
    assert all(not e["src"].startswith("state:") for e in req["directed"])
    # session grain is a superset in support
    assert ses["support_pairs"] >= req["support_pairs"] and ses["n_directed"] >= req["n_directed"]


def test_twin_target_is_acyclic_without_bidirected_edges():
    inst = xs_instantiation()
    req, ses, proj = build_targets(inst, twin=True)
    assert req["n_bidirected"] == 0 and req["bidirected_groups"] == []
    assert req["directed_acyclic_at_floor"]
    assert is_acyclic([(e["src"], e["dst"]) for e in req["directed"]])
    assert any(e["src"].startswith("state:") for e in req["directed"])


def test_floor_sweep_and_default_floor(tmp_path):
    inst = xs_instantiation()
    summary = write_graph_artifacts(inst, tmp_path, "latent")
    fs = read_json(tmp_path / "graphs" / "floor-sensitivity.json")
    assert fs["default_floor"] == inst.cfg.mechanism.floor
    floors = [r["floor"] for r in fs["sweep"]]
    assert floors == list(FLOOR_SWEEP)
    counts = [r["n_directed"] + r["n_bidirected"] for r in fs["sweep"]]
    assert counts == sorted(counts, reverse=True)
    assert summary["request"]["n_directed"] > 0


def test_views_are_functions_of_the_target_alone(tmp_path):
    inst = xs_instantiation()
    write_graph_artifacts(inst, tmp_path, "latent")
    target = read_json(tmp_path / "graphs" / "scoring-target.json")
    endpoint = read_json(tmp_path / "graphs" / "views" / "endpoint.json")
    service = read_json(tmp_path / "graphs" / "views" / "service.json")
    assert coarsen(target, "endpoint", inst.topo) == endpoint
    assert coarsen(target, "service", inst.topo) == service
    assert len(service["directed"]) <= len(endpoint["directed"]) <= len(target["directed"])
