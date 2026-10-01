"""PRD scenarios 4 (a method whose output equals the projection scores
perfectly), 25 (what a method may read) and 3a (causal-validity axis on the
twin, not-applicable with reason on the latent instance)."""
import json
import random

import pytest

from tracebench.allowlist import NotMethodReadable, is_method_readable, method_readable_files, open_for_method
from tracebench.graphs import write_graph_artifacts
from tracebench.instantiate import write_instantiation
from tracebench.record import read_json
from tracebench.score import (ScoreContext, _score_at_floor_dense, load_context, score_at_floor, score_corpus, self_check,
                              target_as_prediction)
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


# --- the sparse path equals the dense reference bit for bit ---------------------------------------
def _bytes(result):
    return json.dumps(result, sort_keys=True)


def _random_case(rng, n_ops, outcomes, acyclic, state):
    """A target over a random support, with truth edges inside and outside the universe."""
    tokens = [f"{op}:{o}" for op in range(n_ops) for o in outcomes]
    states = [f"state:v{i}={j}" for i in range(2) for j in range(2)] if state else []
    alphabet = {"tokens": [{"token": t} for t in tokens + states]}
    op_pairs = [(a, b) for a in range(n_ops) for b in range(a + 1, n_ops)]
    support = rng.sample(op_pairs, max(1, len(op_pairs) // 2))
    pool = [(f"{a}:{x}", f"{b}:{y}") for a, b in op_pairs for x in outcomes for y in outcomes]   # op a < op b
    if not acyclic:
        pool += [(b, a) for a, b in pool]
    directed = [{"src": a, "dst": b, "strength": rng.choice([0.01, 0.05, 0.2, 0.6])}
                for a, b in rng.sample(pool, min(len(pool), 4 * n_ops))]
    directed += [{"src": s, "dst": rng.choice(tokens), "strength": 0.3} for s in states]
    bidirected = []
    for a, b in rng.sample(pool, min(len(pool), n_ops)):
        if rng.random() < 0.8:                       # the scorer does not normalise the target's order
            a, b = min(a, b), max(a, b)
        bidirected.append({"a": a, "b": b, "strength": rng.choice([0.01, 0.05, 0.2, 0.6])})
    target = {"support_op_pairs": [list(p) for p in support], "directed": directed, "bidirected": bidirected,
              "default_floor": 0.05}
    return target, alphabet, tokens + states


def _random_prediction(rng, toks, n_edges):
    score = lambda: rng.choice([1.0, 0.5, 0.5, 0.25, 0.0, -0.0, -0.5, -1.0, rng.random(), -rng.random()])  # noqa: E731
    directed = []
    for _ in range(n_edges):
        a, b = rng.sample(toks, 2)
        directed.append({"src": a, "dst": b, "score": score()})
        if rng.random() < 0.2:
            directed.append({"src": b, "dst": a, "score": score()})
    directed.append({"src": "999:ok", "dst": "998:ok", "score": 1.0})           # outside the universe
    directed.append({"src": toks[0], "dst": toks[1]})                            # default score
    bidirected = [dict(zip("ab", rng.sample(toks, 2)), score=score()) for _ in range(n_edges // 3)]
    return {"directed": directed, "bidirected": bidirected}


@pytest.mark.parametrize("n_ops,outcomes,acyclic,state", [
    (4, ["ok"], True, False),
    (6, ["ok", "5xx"], False, False),
    (7, ["ok", "4xx", "5xx"], True, True),
    (30, ["ok", "4xx", "5xx", "err", "slow"], False, False),      # a universe past numpy's pairwise-sum block
])
def test_sparse_scorer_equals_the_dense_reference(n_ops, outcomes, acyclic, state):
    rng = random.Random(f"{n_ops}-{acyclic}-{state}")
    for trial in range(40 if n_ops < 30 else 6):
        target, alphabet, toks = _random_case(rng, n_ops, outcomes, acyclic, state)
        context = ScoreContext(target, alphabet)
        assert context.sparse
        predictions = [_random_prediction(rng, toks, n) for n in (0, 3, 5 * n_ops, 40 * n_ops)]
        predictions += [target_as_prediction(target), {"directed": []}]
        for prediction in predictions:
            for floor in (0.0, 0.05, 0.2, 0.9):
                dense = _score_at_floor_dense(target, alphabet, prediction, floor)
                assert _bytes(score_at_floor(target, alphabet, prediction, floor, context=context)) == _bytes(dense), \
                    (trial, floor)
                assert _bytes(score_at_floor(target, alphabet, prediction, floor)) == _bytes(dense)
        assert context.truth(0.05)["acyclic"] == acyclic or state or not acyclic


def test_sparse_scorer_equals_the_dense_reference_on_the_corpora(corpora):
    rng = random.Random(7)
    for variant, d in corpora.items():
        context = load_context(d)
        target, alphabet = context.target, context.alphabet
        toks = [t["token"] for t in alphabet["tokens"]]
        full = target_as_prediction(target, floor=0.0)
        degraded = {"directed": full["directed"][::2], "bidirected": full["bidirected"][1::2]}
        for prediction in (full, degraded, _random_prediction(rng, toks, 400)):
            for floor in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
                assert _bytes(score_at_floor(target, alphabet, prediction, floor, context=context)) == \
                    _bytes(_score_at_floor_dense(target, alphabet, prediction, floor)), (variant, floor)
            assert _bytes(score_corpus(d, prediction, context=context)) == _bytes(score_corpus(d, prediction))


def test_scorer_falls_back_to_the_dense_path():
    rng = random.Random(3)
    target, alphabet, toks = _random_case(rng, 6, ["ok", "5xx"], False, False)
    context = ScoreContext(target, alphabet)
    prediction = _random_prediction(rng, toks, 30)
    prediction["directed"][0]["score"] = float("nan")            # a non-finite score
    prediction["directed"][1]["score"] = float("inf")
    assert _bytes(score_at_floor(target, alphabet, prediction, 0.05, context=context)) == \
        _bytes(_score_at_floor_dense(target, alphabet, prediction, 0.05))
    shuffled = ScoreContext(target, alphabet, context.ordered[::-1], context.unordered[::-1])   # an unsorted universe
    assert not shuffled.sparse
    prediction = _random_prediction(rng, toks, 30)
    assert _bytes(score_at_floor(target, alphabet, prediction, 0.05, context=shuffled)) == \
        _bytes(_score_at_floor_dense(target, alphabet, prediction, 0.05, context.ordered[::-1], context.unordered[::-1]))
    with pytest.raises(ValueError):
        score_at_floor(dict(target), alphabet, prediction, 0.05, context=context)
