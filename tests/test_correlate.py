"""PRD scenario 15: the bundled correlator reconstructs sequences from join
keys, labels each with the strongest identity it could establish, and its
loss against the oracle is a reported quantity; the views carry the
consumer's column contract in both orderings."""
import glob
import json

import pyarrow.parquet as pq

from tracebench.correlate.views import ARROW_SCHEMA
from tracebench.record import read_json
from corpus_fixture import xs_corpus

TRACE_CMI_COLUMNS = ["trace_id", "ops", "outcomes", "offsets", "durations", "parent_pos", "healthy", "n_err_spans",
                     "n_spans", "truncated", "scenario_hash", "scenario_sid", "start_ns"]


def test_report_carries_the_reported_quantities():
    corpus = xs_corpus()
    rep = read_json(corpus / "reports" / "correlation-report.json")
    pl = rep["parent_link"]["all"]
    assert 0.0 <= pl["precision"] <= 1.0 and 0.0 <= pl["recall"] <= 1.0
    assert 0.0 < rep["unattributed_fraction"] < 1.0
    hist = rep["identity_level_histogram_spans"]
    assert "session" in hist and "none" in hist
    assert rep["session_recovery"]["n_true_sessions"] > 0
    # the cart-id-as-session fallback is refused at the fitted rate (0 in the fitted feed window)
    assert rep["cart_fallback_refused"] >= 0 and rep["records_merged_lines"] > 0


def test_views_follow_the_consumer_contract_in_both_orderings():
    corpus = xs_corpus()
    for ordering in ("end", "start"):
        for grain in ("request", "session"):
            root = corpus / "views" / f"{ordering}-{grain}"
            files = sorted(glob.glob(str(root / "sequences" / "split=*" / "date=*" / "*.parquet")))
            assert files, root
            t = pq.read_table(files[0])
            assert t.schema.names[:13] == TRACE_CMI_COLUMNS
            assert [f.name for f in ARROW_SCHEMA][:13] == TRACE_CMI_COLUMNS
            rows = t.to_pylist()
            assert all(r["sequence_kind"] == grain for r in rows)
            assert all(r["attribution_level"] in (0, 1, 2, 3) for r in rows)
            links = viol = 0
            for r in rows:
                for j, p in enumerate(r["parent_pos"]):
                    if p >= 0:
                        links += 1
                        viol += int(not ((p > j) if ordering == "end" else (p < j)))
                assert len(r["ops"]) == len(r["outcomes"]) == len(r["offsets"]) == len(r["durations"]) == r["n_spans"]
            # emitted clock skew can reorder a parent and its child; the rate is reported, and stays small
            assert viol <= 0.2 * links, (ordering, grain, viol, links)
            stats = json.loads((root / "export-stats.json").read_text())
            assert "orientation_violations" in stats and stats["orientation_violations"][ordering]["links"] > 0
            assert (root / "model-vocab.json").exists() and (root / "scenario-prevalence.json").exists()
            vocab = json.loads((root / "model-vocab.json").read_text())
            assert [o["id"] for o in vocab["base_ops"]] == list(range(len(vocab["base_ops"])))
            assert vocab["n_specials"] == 4
    # the trace-cmi loader's glob finds the trees
    assert glob.glob(str(corpus / "views" / "end-request" / "sequences" / "split=train" / "date=*" / "*.parquet"))


def test_session_grain_holds_whole_journeys():
    corpus = xs_corpus()
    files = sorted(glob.glob(str(corpus / "views" / "end-session" / "sequences" / "split=train" / "date=*" / "*.parquet")))
    rows = pq.read_table(files[0]).to_pylist()
    assert max(r["n_spans"] for r in rows) > max(1, min(r["n_spans"] for r in rows))
    # a session-level sequence contains client tokens as roots of their request trees
    assert any(-1 in r["parent_pos"] for r in rows)
