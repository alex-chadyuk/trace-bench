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

Two optional extras (`--extras topology spans`, 2026-10-26; the stretch arms' inputs) add a file per
case beside the three above and change nothing in them:

- `topology.json` — the deployment call topology coarsened to deployed services, caller → callee
  as stored in `instantiation.json` (`topology.edges`, op-level, mapped to services through
  `topology.ops`; the client excluded; within-service calls dropped; one edge per ordered pair
  with the largest `p_call`). The same content in every case directory. A method that declares
  it reads a topology-informed input.
- `spans.csv` — one row per span of every request trace of the correlated end-ordered request
  view (`views/end-request/sequences/**`) whose last span ends inside the case's window
  `[window start, end_time)`: `trace_id, service, operation, start_us, duration_us, parent_pos`
  (microseconds; the view's `start_ns + offsets[i]` and `durations[i]`; service and operation
  from the view's `model-vocab.json` base ops). Every split's rows are read: the root-cause
  read is not a test-split read. The reconstruction's measured loss is part of this input.

The run record (`<out>/run/`) lists every input file read with its sha256 — the configuration,
the instantiation, `labels/cases.json` and every access-log shard — and names the two label
fields read, which is the evidence that the exporter read the fault window and nothing else.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import datetime as dt
import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from .constants import (CASES_JSON, CONFIG_YAML, INSTANTIATION_JSON, KIND_ACCESS, KIND_CLIENT, KIND_NAMES,
                        LABELS_DIR, RAW_DIR, SEQUENCES_DIR, VIEWS_DIR)
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
# --- extras (2026-10-26): the stretch arms' inputs, beside the three case files ---------------
EXTRA_TOPOLOGY = "topology"
EXTRA_SPANS = "spans"
EXTRAS = (EXTRA_TOPOLOGY, EXTRA_SPANS)
TOPOLOGY_JSON = "topology.json"
SPANS_CSV = "spans.csv"
SPANS_COLUMNS = ("trace_id", "service", "operation", "start_us", "duration_us", "parent_pos")
SPANS_VIEW = "end-request"  # the correlated request view in end ordering (the arms' latent view)
MODEL_VOCAB_JSON = "model-vocab.json"
NS_PER_S = 1_000_000_000
NS_PER_US = 1_000
SPANS_BATCH_ROWS = 20_000
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


def topology_edges(inst):
    """`[{"caller", "callee", "p_call"}]` — the deployment call topology coarsened to deployed services,
    caller → callee as stored (not pre-flipped), the client excluded, within-service calls dropped, the
    largest `p_call` per ordered pair. The service pairs are `score_service.topology_reference`'s, in the
    stored direction."""
    topo = inst["topology"]
    names = {int(s["index"]): str(s["name"]) for s in topo["services"] if int(s["kind"]) != KIND_CLIENT}
    svc_of = {int(o["id"]): int(o["service"]) for o in topo["ops"]}
    best = {}
    for e in topo["edges"]:
        a, b = svc_of[int(e["caller"])], svc_of[int(e["callee"])]
        if a == b or a not in names or b not in names:
            continue
        key = (names[a], names[b])
        best[key] = max(best.get(key, 0.0), float(e.get("p_call", 1.0)))
    return [{"caller": a, "callee": b, "p_call": p} for (a, b), p in sorted(best.items())]


def view_inputs(corpus_dir, view=SPANS_VIEW):
    """The view's vocabulary file and its sequence parquet files, sorted."""
    root = Path(corpus_dir) / VIEWS_DIR / view
    files = sorted(root.glob(f"{SEQUENCES_DIR}/split=*/date=*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no sequence files under {root / SEQUENCES_DIR}")
    return root / MODEL_VOCAB_JSON, files


def base_ops(vocab):
    """op id → (service, operation name) from the view's `model-vocab.json`."""
    return {int(o["id"]): (str(o["service"]), str(o["name"])) for o in vocab["base_ops"]}


def view_spans(files, ops, windows_ns):
    """Per case window (sorted, non-overlapping `[lo_ns, hi_ns)` on the trace's last span end), the span rows
    of every request trace of the view whose last span ends inside it; returns `(rows per case, counts)`."""
    import pyarrow.parquet as pq

    los = [lo for lo, _ in windows_ns]
    rows = [[] for _ in windows_ns]
    n_traces = [0] * len(windows_ns)
    n_read = n_outside = 0
    cols = ["trace_id", "ops", "offsets", "durations", "parent_pos", "start_ns"]
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=SPANS_BATCH_ROWS, columns=cols):
            for r in batch.to_pylist():
                n_read += 1
                start, offs, durs = int(r["start_ns"]), r["offsets"], r["durations"]
                end = start + max(o + d for o, d in zip(offs, durs))
                k = bisect.bisect_right(los, end) - 1
                if k < 0 or end >= windows_ns[k][1]:
                    n_outside += 1
                    continue
                n_traces[k] += 1
                for i, (op, o, d, pp) in enumerate(zip(r["ops"], offs, durs, r["parent_pos"])):
                    if int(op) not in ops:
                        raise ValueError(f"trace {r['trace_id']} position {i}: token {op} is not a base op of the view's vocabulary")
                    svc, name = ops[int(op)]
                    rows[k].append((r["trace_id"], svc, name, (start + int(o)) // NS_PER_US, int(d) // NS_PER_US, int(pp)))
    return rows, {"n_traces_read": n_read, "n_traces_outside_cases": n_outside, "n_traces_per_case": n_traces}


def write_spans(case_dir, rows):
    with open(Path(case_dir) / SPANS_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(SPANS_COLUMNS)
        w.writerows(rows)
    return len(rows)



def export(corpus_dir, rung, out_dir, workers=1, extras=()):
    corpus_dir, out_dir = Path(corpus_dir), Path(out_dir)
    extras = tuple(extras)
    unknown_extras = [x for x in extras if x not in EXTRAS]
    if unknown_extras or len(set(extras)) != len(extras):
        raise ValueError(f"unknown or repeated extras {extras}; registered: {EXTRAS}")
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
    extra_inputs, extra_results = [], {}
    if EXTRA_TOPOLOGY in extras:
        edges = topology_edges(inst)
        topology_doc = {"edges": edges, "source": f"{INSTANTIATION_JSON}: topology.edges coarsened to the deployed services, "
                        "caller -> callee as stored, the client excluded, within-service calls dropped, largest p_call per pair"}
        for c, rec in zip(plan, case_records):
            write_json(out_dir / c["dir"] / TOPOLOGY_JSON, topology_doc)
            rec["sha256"][TOPOLOGY_JSON] = sha256_file(out_dir / c["dir"] / TOPOLOGY_JSON)
        extra_results[EXTRA_TOPOLOGY] = {"n_edges": len(edges), "fields_read": ["topology.services", "topology.ops", "topology.edges"]}
    if EXTRA_SPANS in extras:
        vocab_path, view_files = view_inputs(corpus_dir)
        ops = base_ops(read_json(vocab_path))
        windows_ns = [((epoch_unix + int(c["window_start_s"])) * NS_PER_S, (epoch_unix + int(c["end_time_s"])) * NS_PER_S) for c in plan]
        span_rows, counts = view_spans(view_files, ops, windows_ns)
        for c, rec, rows in zip(plan, case_records, span_rows):
            rec["n_spans"] = write_spans(out_dir / c["dir"], rows)
            rec["sha256"][SPANS_CSV] = sha256_file(out_dir / c["dir"] / SPANS_CSV)
        extra_inputs = [vocab_path, *view_files]
        extra_results[EXTRA_SPANS] = {"view": SPANS_VIEW, "n_files": len(view_files), "columns": list(SPANS_COLUMNS),
                                      "unit": "microseconds", "selection": "request traces whose last span ends in [window start, end_time)",
                                      "splits_read": "all", **counts}
    inputs = [{"path": p.relative_to(corpus_dir).as_posix(), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
              for p in [cfg_path, inst_path, cases_path, *shards, *extra_inputs]]
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
        "extras": list(extras), **extra_results,
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
    p.add_argument("--extras", nargs="*", default=(), choices=EXTRAS,
                   help="optional case files beside the three: topology.json (deployed-service call edges) and/or spans.csv "
                        "(the end-ordered request view's spans per case); the three registered files are unchanged")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    rec = RunRecord(args.out, "export_rca", vars(args))
    results = export(args.corpus, args.rung, args.out, workers=args.workers, extras=tuple(args.extras))
    rec.finish(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
