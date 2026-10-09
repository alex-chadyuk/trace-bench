"""The service-level graph axis (docs/scorers-service-rca.md, 2026-10-09): the shipped service view
scored over an explicit universe of every ordered pair of deployed services, the topology reference,
and the CLI."""
import json

import pytest

from tracebench.constants import KIND_CLIENT
from tracebench.graphs import write_graph_artifacts
from tracebench.instantiate import write_instantiation
from tracebench.record import read_json
from tracebench.score import target_as_prediction
from tracebench.score_service import (deployed_services, load_view, main, score_service, service_key,
                                      service_universe, topology_reference)
from xs_fixture import xs_instantiation


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    inst = xs_instantiation()
    d = tmp_path_factory.mktemp("svc")
    write_instantiation(inst, d, "2026-10-09")
    write_graph_artifacts(inst, d, "latent")
    return d


def test_universe_is_every_ordered_pair_of_deployed_services(corpus):
    services = deployed_services(corpus)
    inst = read_json(corpus / "instantiation.json")["topology"]
    assert len(services) == sum(1 for s in inst["services"] if s["kind"] != KIND_CLIENT) and len(services) < len(inst["services"])
    ordered, unordered = service_universe(services)
    n = len(services)
    assert len(ordered) == n * (n - 1) and len(unordered) == n * (n - 1) // 2
    assert ordered == sorted(ordered) and all(a != b for a, b in ordered)
    client = [service_key(s["index"]) for s in inst["services"] if s["kind"] == KIND_CLIENT]
    assert client and all(c not in {x for p in ordered for x in p} for c in client)


def test_view_scores_itself_perfectly_and_outside_pairs_are_counted(corpus):
    view = load_view(corpus)
    res = score_service(corpus, target_as_prediction(view))
    assert res["level"] == "service" and res["grain"] == "request"
    assert res["directed"]["f1"] == 1.0 and res["bidirected"]["f1"] == 1.0 and res["shd_mixed"] == 0
    assert res["universe"]["n_services"] == len(res["universe"]["services"]) == len(deployed_services(corpus))
    assert res["universe"]["ordered_pairs"] == res["universe"]["n_services"] * (res["universe"]["n_services"] - 1)
    inst = read_json(corpus / "instantiation.json")["topology"]
    client = service_key(next(s["index"] for s in inst["services"] if s["kind"] == KIND_CLIENT))
    pred = {"directed": [{"src": client, "dst": "svc:0", "score": 1.0}], "bidirected": []}
    res = score_service(corpus, pred)
    assert res["universe"]["predictions_outside_universe"] == 1 and res["directed"]["tp"] == 0


def test_topology_reference_is_the_coarsened_call_graph(corpus):
    ref = topology_reference(corpus)
    inst = read_json(corpus / "instantiation.json")["topology"]
    svc_of = {o["id"]: o["service"] for o in inst["ops"]}
    want = {}
    for e in inst["edges"]:
        a, b = svc_of[e["callee"]], svc_of[e["caller"]]
        if a != b:
            want[(service_key(a), service_key(b))] = max(want.get((service_key(a), service_key(b)), 0.0), e["p_call"])
    assert {(e["src"], e["dst"]): e["score"] for e in ref["directed"]} == want and ref["bidirected"] == []
    res = score_service(corpus, ref)
    assert res["reference"] == "topology" and res["universe"]["predictions_outside_universe"] == 0
    assert 0.0 < res["directed"]["recall"] <= 1.0 and 0.0 < res["directed"]["precision"] <= 1.0
    assert res["auroc"]["directed"] is None or 0.0 <= res["auroc"]["directed"] <= 1.0


def test_cli_scores_a_file_and_the_reference(corpus, tmp_path):
    pred = tmp_path / "pred.json"
    pred.write_text(json.dumps(target_as_prediction(load_view(corpus))))
    assert main(["--corpus", str(corpus), "--prediction", str(pred), "--out", str(tmp_path / "a.json")]) == 0
    assert read_json(tmp_path / "a.json")["directed"]["f1"] == 1.0
    assert main(["--corpus", str(corpus), "--prediction", "topology", "--out", str(tmp_path / "b.json")]) == 0
    assert read_json(tmp_path / "b.json")["reference"] == "topology"
