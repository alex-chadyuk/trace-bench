"""PRD scenario 21: re-running the simulation with a parent forced to each of
its values (and the child's other parents held at the recorded context)
reproduces the recorded strength within tolerance, and forcing a non-parent
moves nothing. The full check is `python -m tracebench.check_mechanism`; the
test samples edges to stay fast. Slow-class edges rest on two Monte Carlo
estimates (the mechanism's and the run's), so a small share of sampled edges
may sit outside the 0.03 tolerance; D-TB-9 records the residual channel."""
from tracebench.check_mechanism import run_check
from corpus_fixture import xs_corpus


def test_forced_reruns_reproduce_recorded_strengths():
    corpus = xs_corpus()
    rep = run_check(corpus, max_edges=14, n_non_edges=5, seed=7)
    assert rep["n_edges_checked"] == 14
    failed = [r for r in rep["edges"] if not r["pass"]]
    assert rep["n_pass"] >= 12, failed
    assert rep["n_non_edge_pass"] == rep["n_non_edges"], [r for r in rep["non_edges"] if not r["pass"]]
    assert rep["max_non_edge_residual"] <= rep["non_edge_tolerance"]
    for r in rep["edges"]:
        assert 0.0 <= r["recorded"] <= 1.0 and 0.0 <= r["empirical"] <= 1.0
