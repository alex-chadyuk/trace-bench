"""The root-cause export (`tracebench.export_rca`, 2026-10-15): the standard suite's case shape from the raw
access-log lines, the window contract (two label fields, nothing else), the registered bin-width rule against
the shipped schedules, and an independent recount of one case's series."""
import csv
import datetime as dt
import gzip
import json
import numpy as np
import pytest
import yaml

from corpus_fixture import xs_corpus
from tracebench.export_rca import (BIN_WIDTH_S, GRID_WIDTHS_S, LABEL_FIELDS_READ, METRICS, MIN_POST_BINS, MIN_PRE_BINS,
                                   SHARD_S, WINDOW_KEYS, case_dirs, export, main, window_epoch)
from tracebench.record import read_json
from xs_fixture import REPO


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    corpus = xs_corpus()
    out = tmp_path_factory.mktemp("rca-export") / "xs-s0"
    results = export(corpus, "xs", out, workers=1)
    return corpus, out, results


def _read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def test_case_shape_and_window_contract(exported):
    corpus, out, res = exported
    cases = read_json(corpus / "labels" / "cases.json")["cases"]
    labels = {d: c for d, c in case_dirs(cases)}
    dirs = sorted(p.name for p in out.glob("case-*"))
    assert dirs == ["case-00", "case-01"] == [d for d, _ in case_dirs(cases)]
    epoch, window_s = window_epoch(yaml.safe_load((corpus / "config.yaml").read_text()))
    services = [s["name"] for s in read_json(out / "case-00" / "services.json")["services"]]
    inst = read_json(corpus / "instantiation.json")["topology"]["services"]
    assert services == sorted(s["name"] for s in inst if s["kind"] != 2) and "client" not in services
    prev_end = 0
    for d in dirs:
        c = labels[d]
        window = read_json(out / d / "window.json")
        assert set(window) == set(WINDOW_KEYS) and len(window) == 2  # exactly the two registered fields
        assert window["inject_time"] == epoch + int(c["inject_time_s"])
        assert window["end_time"] == epoch + int(c["end_time_s"])
        header, rows = _read_csv(out / d / "data.csv")
        assert header == ["time"] + [f"{s}_{m}" for s in services for m in METRICS]
        times = [int(r[0]) for r in rows]
        assert times[0] == epoch + prev_end and times[-1] == window["end_time"] - 30
        assert all(b - a == 30 for a, b in zip(times, times[1:]))
        n_pre = sum(1 for t in times if t < window["inject_time"])
        assert n_pre >= MIN_PRE_BINS and len(times) - n_pre >= MIN_POST_BINS
        prev_end = int(c["end_time_s"])
    assert res["width_s"] == 30 and res["n_bins"] == window_s // 30
    assert [c["n_bins_pre"] for c in res["cases"]] == [40, 30] and [c["n_bins_post"] for c in res["cases"]] == [20, 10]


def test_root_cause_never_appears_in_the_export(exported):
    corpus, out, res = exported
    cases = read_json(corpus / "labels" / "cases.json")["cases"]
    forbidden = {"root_cause", "fault_kind", "forced_nodes", "forced_value", "indicator"}
    forbidden |= {c["case_id"] for c in cases} | {c["fault_kind"] for c in cases}
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        text = path.read_text(encoding="utf-8")
        hits = sorted(f for f in forbidden if f in text)
        assert not hits, f"{path.relative_to(out)} carries {hits}"
    # every deployed service gets the same four columns: the root cause is one of them, indistinguishable
    header, _ = _read_csv(out / "case-00" / "data.csv")
    per_service = {}
    for col in header[1:]:
        svc, metric = col.rsplit("_", 1)
        per_service.setdefault(svc, []).append(metric)
    assert all(m == list(METRICS) for m in per_service.values())
    assert {c["root_cause_component"] for c in cases} <= set(per_service)
    # the record names the two label fields it read and the label files it did not
    assert res["labels_fields_read"] == list(LABEL_FIELDS_READ)
    assert "labels/faults.json" in res["labels_not_read"]
    assert sorted(i["path"] for i in res["inputs"] if i["path"].startswith("labels/")) == ["labels/cases.json"]


def test_record_hashes_every_input(exported):
    corpus, out, res = exported
    paths = [i["path"] for i in res["inputs"]]
    shards = sorted(p.relative_to(corpus).as_posix() for p in corpus.glob("raw/shard=*/vl.jsonl.gz"))
    assert paths == ["config.yaml", "instantiation.json", "labels/cases.json", *shards]
    assert all(len(i["sha256"]) == 64 and i["bytes"] > 0 for i in res["inputs"])
    assert res["n_access_lines"] == res["n_healthz_dropped"] + res["n_rows_binned"] + res["n_rows_outside_window"] and res["n_healthz_dropped"] > 0
    for c in res["cases"]:
        assert set(c["sha256"]) == {"data.csv", "window.json", "services.json"}


def test_series_equal_an_independent_recount(exported):
    """One case's table against a plain per-line recount with the Python stdlib and numpy quantiles."""
    corpus, out, res = exported
    epoch = res["corpus_start_unix"]
    width = res["width_s"]
    header, rows = _read_csv(out / "case-01" / "data.csv")
    col = {name: i for i, name in enumerate(header)}
    t_lo, t_hi = int(rows[0][0]) - epoch, int(rows[-1][0]) - epoch + width
    bins = {}
    for shard in sorted(corpus.glob("raw/shard=*/vl.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["kind"] != "vl.access" or r["path"] == "/healthz":
                    continue
                t = (dt.datetime.fromisoformat(r["ts"]) - dt.datetime.fromtimestamp(epoch, dt.timezone.utc)).total_seconds()
                if not (t_lo <= t < t_hi):
                    continue
                b = int(t // width) * width
                bins.setdefault((b, r["service"]), []).append((int(r["status"]), float(r["request_time"])))
    n_checked = 0
    for row in rows:
        b = int(row[0]) - epoch
        for svc in {k[1] for k in bins}:
            recs = bins.get((b, svc), [])
            assert int(row[col[f"{svc}_count"]]) == len(recs)
            if recs:
                rt = np.array([x[1] for x in recs])
                assert float(row[col[f"{svc}_error"]]) == pytest.approx(sum(1 for s, _ in recs if 500 <= s < 600) / len(recs))
                assert float(row[col[f"{svc}_latency-50"]]) == pytest.approx(np.quantile(rt, 0.5), abs=1e-6)
                assert float(row[col[f"{svc}_latency-95"]]) == pytest.approx(np.quantile(rt, 0.95), abs=1e-6)
                n_checked += 1
            else:
                assert row[col[f"{svc}_error"]] == "0.0" and row[col[f"{svc}_latency-50"]] == "" and row[col[f"{svc}_latency-95"]] == ""
    assert n_checked > 50


@pytest.mark.parametrize("rung", sorted(BIN_WIDTH_S))
def test_registered_width_divides_the_shard_and_every_shipped_fault_boundary(rung):
    width = BIN_WIDTH_S[rung]
    assert width in GRID_WIDTHS_S and SHARD_S % width == 0
    cfg = yaml.safe_load((REPO / "configs" / "instances" / f"{rung}.yaml").read_text())
    faults = cfg["schedules"]["faults"]
    assert faults
    for f in faults:
        assert f["start_s"] % width == 0 and f["end_s"] % width == 0, (rung, f)
        assert (f["end_s"] - f["start_s"]) // width >= MIN_POST_BINS
    assert cfg["run"]["shard_ticks"] * cfg["run"]["tick_s"] == SHARD_S
    assert (cfg["run"]["window"]["duration_hours"] * 3600) % width == 0


def test_refusals(exported, tmp_path):
    corpus, out, _ = exported
    with pytest.raises(ValueError, match="instance"):
        export(corpus, "s", tmp_path / "wrong-rung")
    with pytest.raises(ValueError, match="registered"):
        export(corpus, "l", tmp_path / "no-width")
    with pytest.raises(FileExistsError):
        export(corpus, "xs", out)  # a replay goes to a new directory
    with pytest.raises(SystemExit):
        main(["--corpus", str(corpus), "--rung", "xl", "--out", str(tmp_path / "x")])


def test_cli_writes_the_record(exported, tmp_path):
    corpus, _, res = exported
    out = tmp_path / "cli"
    assert main(["--corpus", str(corpus), "--rung", "xs", "--out", str(out), "--workers", "2"]) == 0
    again = read_json(out / "run" / "results.json")
    assert again["cases"] == res["cases"]  # two workers, pooled before binning: the same bytes
    assert read_json(out / "run" / "arguments.json")["arguments"]["rung"] == "xs"
    meta = read_json(out / "run" / "run_meta.json")
    assert meta["command"] == "export_rca" and meta["status"] == "ok"
    assert (out / "case-00" / "data.csv").read_bytes() == (exported[1] / "case-00" / "data.csv").read_bytes()
