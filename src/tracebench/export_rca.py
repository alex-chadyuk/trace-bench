"""Export one corpus's fault cases in the standard root-cause suite's shape (score side).

    python -m tracebench.export_rca --corpus <dir> --rung {xs,s,m} --out <dir> [--workers N]

For every injected fault of the corpus a case directory `<out>/case-<k>/` holding

- `data.csv` — one `time` column (UNIX seconds, the start of each bin) and four log-derived
  series per deployed service, named `<service>_count`, `<service>_error`,
  `<service>_latency-50`, `<service>_latency-95`: the request count, the share of responses
  with a 5xx status, and the p50 / p95 of `request_time` per fixed-width bin, computed from the
  raw feed's access-log lines (`raw/shard=*/vl.jsonl.gz`, `kind: vl.access`, health-check probes
  dropped), pooled over the whole corpus before binning because per-pod clock skew moves lines
  across shard boundaries. The table runs from the end of the previous fault (or the corpus
  start) to the labelled end of this fault. An empty bin has count 0, error 0 and empty
  latency cells; nothing is imputed here (the consumer applies the suite loader's own rule).
  Quantiles are written at six decimals and the share at nine, so the file is a pure function
  of the corpus on every platform.
- `window.json` — exactly two keys, `inject_time` and `end_time` (UNIX seconds), the only two
  label fields a method receives. The root-cause service, the fault kind and everything else
  in `labels/cases.json` never leave the labels, and the case directory is named by position
  (`case-00`, `case-01`, …, in injection order), never by the labelled case id.
- `services.json` — the deployed-service universe of `instantiation.json` (name and kind;
  the synthetic client excluded), the candidate set every ranking is scored over.

The bin width is the registered rule's value for the rung (publication plan question 9,
decided 2026-10-14, N = 10): the smallest grid width in {5, 10, 15, 30, 60, 90, 180, 300} s at
which, on every seed, the sparsest candidate service's pre-fault median request count per bin
reaches N and its 10th percentile reaches 1, with at least five bins inside the shortest fault
— xs 30 s, s 15 s, m 60 s. It is a constant of this module, selected by `--rung`, never a
command-line knob; a rung without a registered width is refused.

The run record (`<out>/run/`) lists every input file read with its sha256 — the configuration,
the instantiation, `labels/cases.json` and every access-log shard — and names the two label
fields read, which is the evidence that the exporter read the fault window and nothing else.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from .constants import (CASES_JSON, CONFIG_YAML, INSTANTIATION_JSON, KIND_ACCESS, KIND_CLIENT, KIND_NAMES,
                        LABELS_DIR, RAW_DIR)
from .log import log
from .record import RunRecord, read_json, sha256_file, write_json

# --- the registered bin-width rule (plan Q9, owner decision 2026-10-14) ----------------------
GRID_WIDTHS_S = (5, 10, 15, 30, 60, 90, 180, 300)
RULE_N = 10
BIN_WIDTH_S = {"xs": 30, "s": 15, "m": 60}
RULE = {
    "statement": ("the smallest grid width at which, on every seed, the sparsest candidate service's pre-fault "
                  "median request count per bin reaches N and its 10th percentile reaches 1, with at least five "
                  "bins inside the shortest fault"),
    "N": RULE_N,
    "grid_widths_s": list(GRID_WIDTHS_S),
    "min_bins_in_shortest_fault": 5,
    "widths_s": dict(BIN_WIDTH_S),
    "decided": "2026-10-14",
}
SHARD_S = 900  # every registered width divides the shard clock, so bins never straddle a shard boundary

# --- the export's shape ----------------------------------------------------------------------
METRICS = ("count", "error", "latency-50", "latency-95")
CASE_DIR_FMT = "case-{k:02d}"
DATA_CSV = "data.csv"
WINDOW_JSON = "window.json"
SERVICES_JSON = "services.json"
WINDOW_KEYS = ("inject_time", "end_time")
LABEL_FIELDS_READ = ("inject_time_s", "end_time_s")
MIN_PRE_BINS = 4  # RobustScorer's minimum pre-fault length
MIN_POST_BINS = 5  # the rule's fault-resolution bound
LATENCY_DECIMALS = 6  # quantiles of a millisecond-valued request_time, written at microsecond resolution
SHARE_DECIMALS = 9  # the 5xx share, rounded like every stored Monte-Carlo quantity (constants.MC_DECIMALS)
HEALTHZ = "/healthz"
ACCESS_TAG = f'"kind":"{KIND_ACCESS}"'  # the raw JSON is compact with sorted keys


def window_epoch(cfg):
    """The corpus clock: UNIX seconds of the simulated window start, and the window length."""
    start = dt.datetime.strptime(cfg["run"]["window"]["start"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    return int(start.timestamp()), float(cfg["run"]["window"]["duration_hours"]) * 3600.0


def deployed_services(inst):
    """`[(name, kind_name)]` of the deployed services in name order — every kind but the client."""
    svcs = [(str(s["name"]), KIND_NAMES[int(s["kind"])]) for s in inst["topology"]["services"] if int(s["kind"]) != KIND_CLIENT]
    return sorted(svcs)


def ordered_cases(cases):
    """The labelled cases in injection order, checked non-overlapping; the position is the case directory's name."""
    cases = sorted(cases, key=lambda c: float(c["inject_time_s"]))
    prev_end = 0.0
    out = []
    for k, c in enumerate(cases):
        t_i, t_e = float(c["inject_time_s"]), float(c["end_time_s"])
        if t_i < prev_end:
            raise ValueError(f"fault windows overlap: injection at {t_i:g} s before the previous end {prev_end:g} s")
        if t_e <= t_i:
            raise ValueError(f"fault end {t_e:g} s is not after its injection {t_i:g} s")
        out.append({"dir": CASE_DIR_FMT.format(k=k), "window_start_s": prev_end, "inject_time_s": t_i, "end_time_s": t_e})
        prev_end = t_e
    return out


def case_dirs(cases):
    """`[(case directory name, labelled case)]` — the one rule both sides use to pair an export with its labels."""
    ordered = sorted(cases, key=lambda c: float(c["inject_time_s"]))
    return [(CASE_DIR_FMT.format(k=k), c) for k, c in enumerate(ordered)]


def check_alignment(width, cases):
    for c in cases:
        for key in ("inject_time_s", "end_time_s"):
            v = float(c[key])
            if v != int(v) or int(v) % width:
                raise ValueError(f"{key} = {v:g} s is not on the {width} s grid; the registered widths were chosen "
                                 "to align with every shipped fault boundary")


def access_shards(corpus_dir):
    return sorted(Path(corpus_dir).glob(f"{RAW_DIR}/shard=*/vl.jsonl.gz"))


def parse_shard(args):
    """One shard → (counts, t seconds since the corpus start, service index, status, request_time) of its
    non-probe access lines; services outside the universe are collected and refused by the caller."""
    path, epoch_unix, svc_index = args
    epoch = dt.datetime.fromtimestamp(epoch_unix, dt.timezone.utc)
    t, s, st, rt = [], [], [], []
    n_lines = n_access = n_healthz = 0
    unknown = set()
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            n_lines += 1
            if ACCESS_TAG not in line:
                continue
            r = json.loads(line)
            if r.get("kind") != KIND_ACCESS:
                continue
            n_access += 1
            if r["path"] == HEALTHZ:
                n_healthz += 1
                continue
            idx = svc_index.get(r["service"])
            if idx is None:
                unknown.add(r["service"])
                continue
            t.append((dt.datetime.fromisoformat(r["ts"]) - epoch).total_seconds())
            s.append(idx)
            st.append(int(r["status"]))
            rt.append(float(r["request_time"]))
    return {"path": str(path), "n_lines": n_lines, "n_access": n_access, "n_healthz": n_healthz, "unknown": sorted(unknown),
            "t": np.asarray(t, dtype=np.float64), "svc": np.asarray(s, dtype=np.int32),
            "status": np.asarray(st, dtype=np.int32), "rt": np.asarray(rt, dtype=np.float64)}


def bin_series(t, svc, status, rt, width, n_bins, n_svc):
    """The complete (bin × service) grid at one width: count, 5xx count, p50 and p95 of request_time
    (NaN where the bin is empty), plus the number of rows outside the corpus window."""
    b = np.floor(t / width).astype(np.int64)
    keep = (t >= 0) & (b < n_bins)
    n_outside = int((~keep).sum())
    b, svc, status, rt = b[keep], svc[keep], status[keep], rt[keep]
    key = b * n_svc + svc.astype(np.int64)
    order = np.argsort(key, kind="stable")
    key, status, rt = key[order], status[order], rt[order]
    uniq, start, counts = np.unique(key, return_index=True, return_counts=True)
    size = n_bins * n_svc
    count = np.zeros(size, dtype=np.int64)
    count[uniq] = counts
    n5xx = np.zeros(size, dtype=np.int64)
    is5xx = ((status >= 500) & (status < 600)).astype(np.int64)
    if len(uniq):
        n5xx[uniq] = np.add.reduceat(is5xx, start)
    p50 = np.full(size, np.nan)
    p95 = np.full(size, np.nan)
    for u, i0, c in zip(uniq.tolist(), start.tolist(), counts.tolist()):
        seg = rt[i0:i0 + c]
        p50[u] = np.quantile(seg, 0.5)
        p95[u] = np.quantile(seg, 0.95)
    shape = (n_bins, n_svc)
    return count.reshape(shape), n5xx.reshape(shape), p50.reshape(shape), p95.reshape(shape), n_outside


def _cell(x):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)


def write_case(case_dir, names, kinds, epoch_unix, width, lo, hi, count, n5xx, p50, p95):
    """One case directory from the grid rows `[lo, hi)`; returns the written files' sha256 and the empty-cell count."""
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=False)
    header = ["time"] + [f"{n}_{m}" for n in names for m in METRICS]
    n_empty = 0
    with open(case_dir / DATA_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for b in range(lo, hi):
            row = [epoch_unix + b * width]
            for j in range(len(names)):
                c = int(count[b, j])
                err = round(float(n5xx[b, j]) / c, SHARE_DECIMALS) if c else 0.0
                if c == 0:
                    n_empty += 1
                row += [c, _cell(err), _cell(round(float(p50[b, j]), LATENCY_DECIMALS)), _cell(round(float(p95[b, j]), LATENCY_DECIMALS))]
            w.writerow(row)
    return n_empty


def export(corpus_dir, rung, out_dir, workers=1):
    corpus_dir, out_dir = Path(corpus_dir), Path(out_dir)
    if rung not in BIN_WIDTH_S:
        raise ValueError(f"no registered bin width for rung {rung!r}; registered: {sorted(BIN_WIDTH_S)}")
    width = BIN_WIDTH_S[rung]
    if any(out_dir.glob("case-*")):
        raise FileExistsError(f"{out_dir} already holds an export; a replay writes to a new directory")
    cfg_path, inst_path, cases_path = corpus_dir / CONFIG_YAML, corpus_dir / INSTANTIATION_JSON, corpus_dir / LABELS_DIR / CASES_JSON
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    inst = read_json(inst_path)
    if inst.get("instance") != rung:
        raise ValueError(f"--rung {rung!r} but the corpus is a {inst.get('instance')!r} instance")
    cases = read_json(cases_path)["cases"]
    check_alignment(width, cases)
    plan = ordered_cases(cases)
    epoch_unix, window_s = window_epoch(cfg)
    if window_s != int(window_s) or int(window_s) % width:
        raise ValueError(f"the corpus window {window_s:g} s is not a multiple of the {width} s bin")
    n_bins = int(window_s) // width
    services = deployed_services(inst)
    names, kinds = [n for n, _ in services], [k for _, k in services]
    svc_index = {n: i for i, n in enumerate(names)}
    shards = access_shards(corpus_dir)
    if not shards:
        raise FileNotFoundError(f"no access-log shards under {corpus_dir / RAW_DIR}")
    log({"event": "export_rca_start", "rung": rung, "width_s": width, "n_shards": len(shards), "n_services": len(names),
         "n_cases": len(plan)})
    jobs = [(p, epoch_unix, svc_index) for p in shards]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            parts = list(ex.map(parse_shard, jobs))
    else:
        parts = [parse_shard(j) for j in jobs]
    unknown = sorted({u for p in parts for u in p["unknown"]})
    if unknown:
        raise ValueError(f"access lines name services outside the deployed universe: {unknown}")
    t = np.concatenate([p["t"] for p in parts])
    svc = np.concatenate([p["svc"] for p in parts])
    status = np.concatenate([p["status"] for p in parts])
    rt = np.concatenate([p["rt"] for p in parts])
    count, n5xx, p50, p95, n_outside = bin_series(t, svc, status, rt, width, n_bins, len(names))
    services_doc = {"services": [{"name": n, "kind": k} for n, k in services],
                    "source": f"{INSTANTIATION_JSON}: the deployed services (the synthetic client excluded)"}
    case_records = []
    for c in plan:
        lo, inj, hi = int(c["window_start_s"]) // width, int(c["inject_time_s"]) // width, int(c["end_time_s"]) // width
        if inj - lo < MIN_PRE_BINS or hi - inj < MIN_POST_BINS:
            raise ValueError(f"{c['dir']}: {inj - lo} pre-fault and {hi - inj} fault bins at {width} s; the export needs "
                             f"at least {MIN_PRE_BINS} and {MIN_POST_BINS}")
        case_dir = out_dir / c["dir"]
        n_empty = write_case(case_dir, names, kinds, epoch_unix, width, lo, hi, count, n5xx, p50, p95)
        window = {"inject_time": epoch_unix + int(c["inject_time_s"]), "end_time": epoch_unix + int(c["end_time_s"])}
        write_json(case_dir / WINDOW_JSON, window)
        write_json(case_dir / SERVICES_JSON, services_doc)
        case_records.append({"dir": c["dir"], "window_start": epoch_unix + int(c["window_start_s"]), **window,
                             "n_bins_pre": inj - lo, "n_bins_post": hi - inj, "n_rows": hi - lo, "n_empty_cells": n_empty,
                             "sha256": {f: sha256_file(case_dir / f) for f in (DATA_CSV, WINDOW_JSON, SERVICES_JSON)}})
    inputs = [{"path": p.relative_to(corpus_dir).as_posix(), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
              for p in [cfg_path, inst_path, cases_path, *shards]]
    results = {
        "rung": rung, "width_s": width, "rule": RULE, "corpus_dir_name": corpus_dir.name, "instance": inst.get("instance"),
        "config_hash": inst.get("config_hash"), "corpus_tool_version": inst.get("tool_version"),
        "corpus_start_unix": epoch_unix, "window_s": int(window_s), "n_bins": n_bins, "shard_s": SHARD_S,
        "metrics": list(METRICS), "services": {"n": len(names), "names": names, "kinds": kinds},
        "inputs": inputs, "labels_fields_read": list(LABEL_FIELDS_READ),
        "labels_not_read": [f"{LABELS_DIR}/faults.json", f"{LABELS_DIR}/sequence-labels.parquet"],
        "n_lines": sum(p["n_lines"] for p in parts), "n_access_lines": sum(p["n_access"] for p in parts),
        "n_healthz_dropped": sum(p["n_healthz"] for p in parts), "n_rows_binned": int(len(t)) - n_outside,
        "n_rows_outside_window": n_outside, "n_status_0": int((status == 0).sum()),
        "n_5xx": int(((status >= 500) & (status < 600)).sum()), "cases": case_records,
    }
    log({"event": "export_rca_done", "rung": rung, "n_cases": len(case_records), "n_rows_binned": results["n_rows_binned"],
         "n_rows_outside_window": n_outside})
    return results


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", required=True, help="a corpus directory holding config.yaml, instantiation.json, labels/ and raw/")
    p.add_argument("--rung", required=True, choices=sorted(BIN_WIDTH_S, key=list(BIN_WIDTH_S).index),
                   help="selects the registered bin width (xs 30 s, s 15 s, m 60 s); must match the corpus instance")
    p.add_argument("--out", required=True, help="the export root; one case-<k>/ per fault plus run/")
    p.add_argument("--workers", type=int, default=1, help="shard parsers run in parallel (rows are pooled before binning)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    rec = RunRecord(args.out, "export_rca", vars(args))
    results = export(args.corpus, args.rung, args.out, workers=args.workers)
    rec.finish(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
