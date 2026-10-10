"""The ranking axis (docs/scorers-service-rca.md, 2026-10-09): AC@k and Avg@5 by hand, the seeded
bootstrap, the random-ranking expectation, pooling, and the CLI."""
import json

import pytest

from tracebench.instantiate import write_instantiation
from tracebench.record import read_json, write_json
from tracebench.score_rca import (bootstrap, case_ranks, main, metrics_of, pool, random_expectation,
                                  score_rca)
from tracebench.score_service import deployed_services
from xs_fixture import xs_instantiation

CASES = [
    {"case_id": "c1", "system": "xs", "root_cause_component": "svc-a", "fault_kind": "degrade", "inject_time_s": 10, "end_time_s": 20},
    {"case_id": "c2", "system": "xs", "root_cause_component": "svc-b", "fault_kind": "crash", "inject_time_s": 30, "end_time_s": 40},
    {"case_id": "c3", "system": "xs", "root_cause_component": "svc-c", "fault_kind": "degrade", "inject_time_s": 50, "end_time_s": 60},
    {"case_id": "c4", "system": "xs", "root_cause_component": "svc-d", "fault_kind": "crash", "inject_time_s": 70, "end_time_s": 80},
]
PRED = {"cases": {
    "c1": ["svc-a", "svc-b"],  # rank 1
    "c2": ["svc-x", "svc-y", "svc-b"],  # rank 3
    "c4": ["svc-1", "svc-2", "svc-3", "svc-4", "svc-5", "svc-d"],  # rank 6: beyond k = 5
}}  # c3 has no ranking


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("rca")
    write_instantiation(xs_instantiation(), d, "2026-10-09")
    write_json(d / "labels" / "cases.json", {"cases": CASES})
    return d


def test_metrics_by_hand():
    rows = case_ranks(CASES, PRED)
    assert [r["rank"] for r in rows] == [1, 3, None, 6] and [r["missing"] for r in rows] == [False, False, True, False]
    m = metrics_of([r["rank"] for r in rows])
    assert m["ac@1"] == 0.25 and m["ac@3"] == 0.5 and m["ac@5"] == 0.5 and m["n_cases"] == 4
    assert m["avg@5"] == pytest.approx((0.25 + 0.25 + 0.5 + 0.5 + 0.5) / 5)
    assert metrics_of([])["ac@1"] is None and metrics_of([])["n_cases"] == 0


def test_bootstrap_is_seeded_and_brackets_the_point():
    ranks = [1, 3, None, 6]
    a, b = bootstrap(ranks, 500, 7), bootstrap(ranks, 500, 7)
    assert a == b and a["level"] == 0.95 and a["b"] == 500 and a["seed"] == 7
    m = metrics_of(ranks)
    for k in ("ac@1", "ac@3", "ac@5", "avg@5"):
        iv = a["intervals"][k]
        assert 0.0 <= iv["low"] <= m[k] <= iv["high"] <= 1.0
    assert bootstrap(ranks, 500, 8) != a
    assert bootstrap([], 500, 7)["intervals"] == {} and bootstrap(ranks, 0, 7)["intervals"] == {}


def test_random_expectation_is_analytic():
    r = random_expectation(5)
    assert r["ac@1"] == pytest.approx(0.2) and r["ac@3"] == pytest.approx(0.6) and r["ac@5"] == pytest.approx(1.0)
    assert r["avg@5"] == pytest.approx((0.2 + 0.4 + 0.6 + 0.8 + 1.0) / 5) and r["n_candidates"] == 5
    assert random_expectation(2)["ac@5"] == 1.0 and random_expectation(0) is None


def test_score_rca_and_pool(corpus):
    res = score_rca(corpus, PRED, 200, 0)
    assert res["metrics"]["ac@1"] == 0.25 and res["n_missing"] == 1 and len(res["cases"]) == 4
    assert res["random"]["n_candidates"] == len(deployed_services(corpus))
    assert set(res["by_fault_kind"]) == {"crash", "degrade"} and res["by_fault_kind"]["degrade"]["ac@1"] == 0.5
    pooled = pool([res, res], 200, 0)
    assert pooled["metrics"]["n_cases"] == 8 and pooled["metrics"]["ac@1"] == 0.25 and pooled["n_missing"] == 2
    assert pooled["random"] == res["random"] and pooled["n_cases_per_result"] == [4, 4]


def test_cli_scores_and_pools(corpus, tmp_path):
    pred = tmp_path / "pred.json"
    pred.write_text(json.dumps(PRED))
    out = tmp_path / "r.json"
    assert main(["--corpus", str(corpus), "--prediction", str(pred), "--bootstrap-b", "100", "--bootstrap-seed", "1", "--out", str(out)]) == 0
    r = read_json(out)
    assert r["metrics"]["avg@5"] == pytest.approx(0.4) and r["bootstrap"]["b"] == 100
    pooled = tmp_path / "p.json"
    assert main(["--pool", str(out), str(out), "--bootstrap-b", "100", "--bootstrap-seed", "1", "--out", str(pooled)]) == 0
    assert read_json(pooled)["metrics"]["n_cases"] == 8
    with pytest.raises(SystemExit):
        main(["--bootstrap-b", "1", "--bootstrap-seed", "1", "--out", str(tmp_path / "x.json")])
