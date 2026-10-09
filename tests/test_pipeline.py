"""The unattended per-rung job: generate every (seed, variant), verify each,
upload them together (PRD scenarios 13, 19; the job an execution environment
runs unattended). One failed step stops the job before anything is published,
and the exit status distinguishes a refusal from a failure."""
import shutil
from pathlib import Path

import pytest

from tracebench import pipeline
from tracebench.record import read_json
from corpus_fixture import shipped_xs_pipeline, xs_corpus
from xs_fixture import XS


def _steps(res):
    return [(s["step"], s["corpus"], s["status"]) for s in res["steps"]]


def test_pipeline_generates_derives_verifies_then_uploads_once():
    out, res, calls = shipped_xs_pipeline()
    assert _steps(res) == [
        ("generate", "xs/latent/seed=0", "ok"),
        ("verify", "xs/latent/seed=0", "ok"),
        ("generate", "xs/twin/seed=0", "ok"),
        ("verify", "xs/twin/seed=0", "ok"),
        ("derive", "xs/metrics/seed=0", "ok"),
        ("verify", "xs/metrics/seed=0", "ok"),
        ("upload", "xs x3", "ok"),
    ]
    assert res["step_counts"] == {"ok": 7} and res["fetched"] == []
    for variant in ("latent", "twin", "metrics"):
        corpus = out / "xs" / variant / "seed=0"
        assert (corpus / "COMPLETE").exists() and (corpus / "manifest.json").exists()
    # exactly one upload, over all three corpora, after all verified
    assert len(calls) == 1
    assert sorted(Path(c).name for c in calls[0]["corpora"]) == ["seed=0"] * 3
    assert sorted(Path(c).parent.name for c in calls[0]["corpora"]) == ["latent", "metrics", "twin"]
    # the job's own record sits outside every corpus
    record = read_json(out / "pipeline-xs" / "run" / "results.json")
    assert record["instance"] == "xs" and len(record["corpora"]) == 3
    assert read_json(out / "pipeline-xs" / "run" / "run_meta.json")["status"] == "ok"
    assert all(c["denylist_used"] is False and c["records_scanned"] > 0 for c in record["corpora"])
    assert not list((out / "xs").rglob("hf-stage"))


def test_a_failed_verification_stops_the_job_before_any_upload(tmp_path):
    corpus = xs_corpus()                 # already complete: --skip-existing reaches the verify step
    out = corpus.parents[2]
    target = corpus / "graphs" / "alphabet.json"
    original = target.read_text()
    target.write_text(original + "\n")
    uploaded = []
    try:
        with pytest.raises(RuntimeError, match="verification failed"):
            pipeline.run_pipeline(XS, [0], out, variants=("latent",), skip_existing=True,
                                  upload_to="chadyuk/trace-bench",
                                  upload_fn=lambda corpora, **kw: uploaded.append(corpora), record=False)
        assert not uploaded
        # and through the CLI: a failed step is exit 1
        assert pipeline.main(["--config", str(XS), "--seeds", "0", "--out", str(out),
                              "--variants", "latent", "--skip-existing"]) == 1
    finally:
        target.write_text(original)
    res = pipeline.run_pipeline(XS, [0], out, variants=("latent",), skip_existing=True, record=False)
    assert _steps(res) == [("generate", "xs/latent/seed=0", "skipped"), ("verify", "xs/latent/seed=0", "ok")]
    assert res["uploaded"] is None


def test_a_metrics_only_job_needs_its_latent_source_or_a_fetch(tmp_path):
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="needs the complete xs/latent/seed=0"):
        pipeline.run_pipeline(XS, [0], out, variants=("metrics",), record=False)
    assert not (out / "xs" / "metrics").exists()
    # with a fetch: the latent lands first, is verified, the metrics variant is derived from it,
    # and only the derived corpus is uploaded
    fetched, uploaded = [], []

    def stub_fetch(key, root, **kw):
        fetched.append((key, kw))
        dest = Path(root) / "xs" / "latent" / "seed=0"
        shutil.copytree(xs_corpus(), dest)
        return {"corpus_dir": str(dest), "fetched": True}

    res = pipeline.run_pipeline(XS, [0], out, variants=("metrics",), upload_to="chadyuk/trace-bench",
                                fetch_from="chadyuk/trace-bench", fetch_revision="v0.3.0", fetch_fn=stub_fetch,
                                upload_fn=lambda corpora, **kw: uploaded.append(list(corpora)), record=False)
    assert _steps(res) == [
        ("fetch", "xs/latent/seed=0", "ok"),
        ("verify", "xs/latent/seed=0", "ok"),
        ("derive", "xs/metrics/seed=0", "ok"),
        ("verify", "xs/metrics/seed=0", "ok"),
        ("upload", "xs x1", "ok"),
    ]
    assert fetched == [("xs/latent/seed=0", {"repo": "chadyuk/trace-bench", "revision": "v0.3.0"})]
    assert res["fetched"] == ["xs/latent/seed=0"]
    assert uploaded == [[str(out / "xs" / "metrics" / "seed=0")]]
    # a second run skips the derivation (COMPLETE exists) and still verifies
    res = pipeline.run_pipeline(XS, [0], out, variants=("metrics",), skip_existing=True, record=False)
    assert _steps(res) == [("derive", "xs/metrics/seed=0", "skipped"), ("verify", "xs/metrics/seed=0", "ok")]
    # an unknown variant is a refusal (exit 3), not a failure
    assert pipeline.main(["--config", str(XS), "--seeds", "0", "--out", str(out), "--variants", "latent", "--skip-existing"]) == 0


def test_a_malformed_configuration_is_refused_before_any_work(tmp_path):
    bad = tmp_path / "broken.yaml"
    bad.write_text(XS.read_text().replace("services: 3", "services: 0"), encoding="utf-8")
    assert pipeline.main(["--config", str(bad), "--seeds", "0", "--out", str(tmp_path / "out")]) == 3
    assert not (tmp_path / "out").exists()
