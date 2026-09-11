"""PRD scenarios 4 (a method whose output equals the projection scores
perfectly), 25 (what a method may read) and 3a (causal-validity axis on the
twin, not-applicable with reason on the latent instance)."""
import pytest

from tracebench.allowlist import NotMethodReadable, is_method_readable, method_readable_files, open_for_method
from tracebench.graphs import write_graph_artifacts
from tracebench.instantiate import write_instantiation
from tracebench.record import read_json
from tracebench.score import score_corpus, self_check, target_as_prediction
from xs_fixture import xs_instantiation


@pytest.fixture(scope="module")
def corpora(tmp_path_factory):
    inst = xs_instantiation()
    out = {}
    for variant in ("latent", "twin"):
        d = tmp_path_factory.mktemp(variant)
        write_instantiation(inst, d, "2026-09-10")
        write_graph_artifacts(inst, d, variant)
        out[variant] = d
    return out


def test_target_scores_itself_perfectly(corpora):
    for variant, d in corpora.items():
        res = self_check(d)
        assert res["self_check_passed"], (variant, res["self_check_problems"])
        assert res["directed"]["f1"] == 1.0 and res["skeleton"]["shd"] == 0 and res["shd_mixed"] == 0
        assert res["orientation"]["accuracy"] == 1.0
        assert all(v == 1.0 for v in res["auroc"].values() if v is not None)
        assert res["baselines"]["shd_empty_directed"] == res["universe"]["truth_directed"]


def test_causal_validity_on_twin_and_not_applicable_on_latent(corpora):
    latent = self_check(corpora["latent"])
    assert latent["causal_validity"]["value"] is None
    assert "bidirected" in latent["causal_validity"]["reason"]
    twin = self_check(corpora["twin"])
    cv = twin["causal_validity"]["value"]
    assert cv is not None and cv["sid"] == 0.0 and cv["parent_aid"] == 0.0 and cv["ancestor_aid"] == 0.0
    assert twin["universe"]["truth_bidirected"] == 0


def test_degraded_prediction_is_scored_and_ranked(corpora):
    d = corpora["latent"]
    target = read_json(d / "graphs" / "scoring-target.json")
    pred = target_as_prediction(target)
    pred["directed"] = pred["directed"][::2]
    pred["directed"].append({"src": "999:ok", "dst": "998:ok", "score": 1.0})   # outside the universe
    res = score_corpus(d, pred)
    assert res["directed"]["precision"] == 1.0 and 0.4 < res["directed"]["recall"] < 0.6
    assert res["universe"]["predictions_outside_universe"] == 1
    assert res["directed"]["shd"] == res["directed"]["fn"]
    assert 0.5 < res["auroc"]["directed"] < 1.0
    assert res["sweep"] and [r["floor"] for r in res["sweep"]] == [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]


def test_allowlist_hides_labels_graphs_and_oracle(corpora, tmp_path):
    d = corpora["latent"]
    (d / "raw" / "shard=0000").mkdir(parents=True)
    (d / "raw" / "shard=0000" / "vl.jsonl").write_text("{}\n")
    (d / "labels").mkdir()
    (d / "labels" / "faults.json").write_text("{}")
    (d / "oracle").mkdir()
    (d / "oracle" / "linkage.parquet").write_bytes(b"")
    readable = method_readable_files(d)
    assert readable == ["raw/shard=0000/vl.jsonl"]
    with open_for_method(d, "raw/shard=0000/vl.jsonl") as f:
        assert f.read()
    for forbidden in ("labels/faults.json", "graphs/scoring-target.json", "graphs/mechanism-graph.json",
                      "oracle/linkage.parquet", "instantiation.json", "../outside"):
        with pytest.raises(NotMethodReadable):
            open_for_method(d, forbidden)
    assert is_method_readable("views/end-request/sequences/split=train/date=2026-01-05/part-0000.parquet")
    assert not is_method_readable("graphs/alphabet.json")
