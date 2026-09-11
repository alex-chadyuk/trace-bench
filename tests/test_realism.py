"""PRD scenarios 9 (realism as a measurement), 10 (generation needs no
network) and 11 (the constants loader fails closed; it never invents values)."""
import json
import socket

import pytest

from tracebench.realism import RealismError, load_realism
from tracebench.realism_check import realism_report
from tracebench.record import read_json
from corpus_fixture import scratch_dir, write_variant_config, xs_corpus
from xs_fixture import REPO


def test_realism_report_measures_every_quantity_and_names_the_constants_version():
    corpus = xs_corpus()
    rep = realism_report(corpus)
    quantities = {i["quantity"] for i in rep["items"]}
    assert rep["constants_version"] == json.loads((corpus / "constants.json").read_text())["version"]
    for q in ("latency_quantiles.service.p50", "error_rate.service", "retry.mean_retries", "fanout_mean", "depth_pmf", "vocab_size"):
        assert q in quantities, q
    latency = [i for i in rep["items"] if i["quantity"].startswith("latency_quantiles.service.p5")]
    assert latency and all(i["pass"] for i in latency)
    for i in rep["items"]:
        assert set(i) >= {"quantity", "constant", "realised", "tolerance", "pass"}
    meta = read_json(corpus / "run" / "run_meta.json")
    assert meta["command"] == "generate"


@pytest.mark.slow
def test_named_instance_s_matches_its_constants():
    """Rung s against fitted constants; runs once realism-v1.json exists."""
    pytest.skip("requires the fitted realism-v1.json (private fitter, M6)")


def test_generation_makes_no_network_call(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network access attempted during generation")
    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    from tracebench.generate import generate
    def shrink(d):
        d["run"]["window"] = {"start": "2026-01-05T09:00:00Z", "duration_hours": 0.25}
        d["schedules"]["faults"] = []  # the xs faults lie outside a quarter-hour window
    cfg_path = write_variant_config(shrink, "xs-nonet")
    res = generate(cfg_path, 1, scratch_dir("xs-nonet-out"), stop_after_shard=0, correlate=False)
    assert res["shards_written"] == [0]


def test_constants_loader_fails_closed(tmp_path):
    data = json.loads((REPO / "constants" / "realism-dev.json").read_text())
    del data["leaves"]["latency_quantiles"]["service"]["p95"]
    p = tmp_path / "broken.json"
    p.write_text(json.dumps(data))
    with pytest.raises(RealismError) as ei:
        load_realism(p)
    assert "latency_quantiles.service.p95" in str(ei.value) and "missing" in str(ei.value)
    data = json.loads((REPO / "constants" / "realism-dev.json").read_text())
    data["leaves"]["cart_fallback_rate"]["value"] = 1.7
    p.write_text(json.dumps(data))
    with pytest.raises(RealismError) as ei:
        load_realism(p)
    assert "cart_fallback_rate" in str(ei.value)
    with pytest.raises(RealismError):
        load_realism(tmp_path / "absent.json")
