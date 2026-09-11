"""python -m tracebench.correlate --corpus <dir> [--enable-ip] [--gap-min 30]

Runs the bundled correlator over a corpus's raw feed, one shard at a time, and
writes the four correlated views, the vocabulary files and the correlation-loss
report. Reads raw/ (and, for op ids and thresholds, graphs/alphabet.json and
slow-thresholds.json); never the oracle except to compute the report.

Memory is bounded by one shard. Every join key (request, correlation, trace
and session ids) lives inside one session, and a session's records all lie in
the shard of its arrival, so key resolution, attribution, request trees and
session-level sequences are shard-local. Device-, user- and ip-level sequences
span sessions, so they are stitched across shards: a run stays open until no
record that could still join it can arrive (after shard k every record whose
timestamp precedes the end of shard k has been read).
"""
from __future__ import annotations

import argparse
import datetime as dt
from collections import defaultdict
from pathlib import Path

import yaml

from ..constants import (
    ALPHABET_JSON, ATTR_NONE, ATTR_SESSION, CORRELATION_REPORT_JSON, FAULTS_JSON, GRAPHS_DIR, LABELS_DIR, RAW_DIR,
    REPORTS_DIR, SESSION_GAP_MIN_DEFAULT, SLOW_THRESHOLDS_JSON,
)
from ..log import log
from ..record import RunRecord, read_json, write_json
from ..rng import stable_id
from ..shards import completed_shards, read_records, shard_name
from .normalize import normalize
from .report import ReportAccumulator
from .resolve import attribute, build_resolution
from .trees import build_trees
from .views import OpIndex, ViewBuilder, build_spans

# Emitted timestamps carry clock skew of tens of milliseconds; a record of
# shard k may therefore be stamped slightly past the shard's end. The stitcher
# treats everything before (shard end - this margin) as fully known.
SKEW_MARGIN_MS = 5_000


def load_raw_shard(corpus_dir, shard):
    d = Path(corpus_dir) / RAW_DIR / shard_name(shard)
    records = []
    for name in ("vl.jsonl.gz", "sentry.jsonl.gz"):
        p = d / name
        if p.exists():
            records.extend(read_records(p))
    return records


class SequenceStitcher:
    """Attribution-level sequences in bounded memory.

    A member is `(ts_ms, span_id, Span or None, true_session_gid or None)`.
    Session-level groups are never gap-split and complete within a shard;
    other levels split at `gap_ms` of silence, and a run closes only once
    every record that could still join it has been read (`known_until_ms`).
    Sequence ids are a stable hash of (level, key, first-minute bucket), so
    they do not depend on how the corpus is sharded."""

    def __init__(self, gap_ms):
        self.gap_ms = gap_ms
        self.open = {}

    def add_shard(self, groups, known_until_ms):
        closed = []
        for gk in sorted(set(self.open) | set(groups)):
            members = self.open.pop(gk, []) + groups.get(gk, [])
            members.sort(key=lambda m: (m[0], m[1]))
            level, key = gk
            if level == ATTR_SESSION:
                closed.append(self._sequence(level, key, members))
                continue
            runs = [[members[0]]]
            for m in members[1:]:
                if m[0] - runs[-1][-1][0] > self.gap_ms:
                    runs.append([m])
                else:
                    runs[-1].append(m)
            keep = []
            for run in runs:
                # once a run stays open every later run does too: a record not
                # yet read could bridge them
                if not keep and run[-1][0] + self.gap_ms < known_until_ms:
                    closed.append(self._sequence(level, key, run))
                else:
                    keep.extend(run)
            if keep:
                self.open[gk] = keep
        return closed

    def close_all(self):
        return self.add_shard({}, float("inf"))

    @staticmethod
    def _sequence(level, key, members):
        first = members[0][0]
        bucket = "" if level == ATTR_SESSION else str(first // 60000)
        return {"sequence_id": stable_id("seq", level, key, bucket), "level": level, "key": key,
                "members": members, "ts_first": first, "ts_last": members[-1][0]}


def client_links(spans, res):
    """client span id -> edge hop span id: tagged errors by request id; transactions through their front-end trace."""
    access_by_request = {s["keys"]["request_id"]: s["span_id"] for s in spans if s["kind"] == "vl.access" and s["keys"].get("request_id")}
    request_of_strace = {}
    for s in spans:
        if s["kind"] == "sentry.error" and s["keys"].get("request_id") and s["keys"].get("sentry_trace_id"):
            request_of_strace[s["keys"]["sentry_trace_id"]] = s["keys"]["request_id"]
    link = {}
    for s in spans:
        if s["kind"] == "sentry.error" and s["keys"].get("request_id") in access_by_request:
            link[s["span_id"]] = access_by_request[s["keys"]["request_id"]]
        elif s["kind"] == "sentry.transaction":
            rid = request_of_strace.get(s["keys"].get("sentry_trace_id"))
            if rid in access_by_request:
                link[s["span_id"]] = access_by_request[rid]
    return link


def _window_epoch_ms(config):
    start = config["run"]["window"]["start"]
    if isinstance(start, str):
        start = dt.datetime.fromisoformat(start.replace("Z", "+00:00"))
    if start.tzinfo is None:
        start = start.replace(tzinfo=dt.timezone.utc)
    return int(start.timestamp() * 1000)


def correlate(corpus_dir, enable_ip=False, gap_min=SESSION_GAP_MIN_DEFAULT, min_count=None):
    corpus_dir = Path(corpus_dir)
    inst = read_json(corpus_dir / "instantiation.json")
    config = yaml.safe_load((corpus_dir / "config.yaml").read_text())
    if min_count is None:
        min_count = config["vocab"]["min_count"]
    epoch_ms = _window_epoch_ms(config)
    shard_ms = int(config["run"]["shard_ticks"]) * int(config["run"].get("tick_s", 1)) * 1000
    service_ips = {svc["ip"]: svc["name"] for svc in inst["topology"]["services"] if svc["ip"]}
    alphabet = read_json(corpus_dir / GRAPHS_DIR / ALPHABET_JSON)
    opindex = OpIndex(alphabet)
    thr = read_json(corpus_dir / SLOW_THRESHOLDS_JSON)
    thresholds = {t["op_id"]: t["threshold_s"] for t in thr["thresholds"]}
    faults = read_json(corpus_dir / LABELS_DIR / FAULTS_JSON)["faults"] if (corpus_dir / LABELS_DIR / FAULTS_JSON).exists() else []
    name_to_op = {(o["service"], o["name"]): oid for oid, o in opindex.ops.items()}
    for f in faults:
        f["_rc_ops"] = [name_to_op.get((f["service"], e)) for e in f["endpoints"] if (f["service"], e) in name_to_op]
    views = ViewBuilder(corpus_dir, opindex, thresholds, faults, min_count, epoch_ms)
    report = ReportAccumulator()
    stitcher = SequenceStitcher(gap_min * 60 * 1000)
    for shard in completed_shards(corpus_dir):
        records = load_raw_shard(corpus_dir, shard)
        spans, norm_stats = normalize(records)
        del records
        res = build_resolution(spans)
        attribution = {s["span_id"]: attribute(s, res, enable_ip) for s in spans}
        trees = build_trees(spans, service_ips)
        link = client_links(spans, res)
        all_spans = build_spans(spans, trees, attribution, opindex, thresholds, link)
        gid_of = report.add_shard(corpus_dir, shard, spans, trees, attribution, link, norm_stats)
        views.add_requests(all_spans, trees, attribution)
        groups = defaultdict(list)
        for s in spans:
            level, key = attribution[s["span_id"]]
            if level == ATTR_NONE:
                continue
            groups[(level, key)].append((s["ts_ms"], s["span_id"], all_spans.get(s["span_id"]), gid_of.get(s["span_id"])))
        known_until = epoch_ms + (shard + 1) * shard_ms - SKEW_MARGIN_MS
        closed = stitcher.add_shard(groups, known_until)
        for seq in closed:
            views.add_sequence(seq)
            report.add_sequence(seq)
        log({"event": "correlate_shard", "shard": shard, "spans": len(spans), "trees": len(trees),
             "sequences_closed": len(closed), "sequences_open": len(stitcher.open)})
        del spans, res, attribution, trees, link, all_spans, groups, closed
    for seq in stitcher.close_all():
        views.add_sequence(seq)
        report.add_sequence(seq)
    stats = views.finalize()
    rep = report.finalize()
    (corpus_dir / REPORTS_DIR).mkdir(exist_ok=True)
    write_json(corpus_dir / REPORTS_DIR / CORRELATION_REPORT_JSON, rep)
    return stats, rep


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", required=True)
    p.add_argument("--enable-ip", action="store_true", help="allow ip-level attribution (off by default, as in the real feed)")
    p.add_argument("--gap-min", type=float, default=SESSION_GAP_MIN_DEFAULT)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    rec = RunRecord(args.corpus, "correlate", vars(args))
    stats, report = correlate(args.corpus, args.enable_ip, args.gap_min)
    log({"event": "correlate", "parent_link_f1": report["parent_link"]["all"]["f1"],
         "unattributed_fraction": report["unattributed_fraction"], "alphabet_realized": stats["alphabet_size_realized_train"]})
    from ..manifest import write_manifest
    manifest = write_manifest(args.corpus)  # the views changed the file set; re-list and re-hash
    rec.finish({"stats": stats, "report": {k: v for k, v in report.items() if k != "description"},
                "manifest_files": len(manifest["files"])})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
