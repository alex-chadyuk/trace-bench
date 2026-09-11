"""The correlated views: four trace-cmi-shaped parquet roots, vocabulary,
prevalence, stats and sequence labels.

    views/{end,start}-{request,session}/sequences/split=<split>/date=<day>/part-NNNN.parquet

Columns: the 13 trace-cmi columns first (`trace_id, ops, outcomes, offsets,
durations, parent_pos, healthy, n_err_spans, n_spans, truncated,
scenario_hash, scenario_sid, start_ns`), then `attribution_level` (int8) and
`sequence_kind` (string). `end` ordering sorts spans by their emitted
(completion) time so a callee precedes its caller and `parent_pos[j] > j`;
`start` ordering sorts by start time, `parent_pos[j] < j`. A client-side
record is the root of its request tree (the client calls the edge), so under
`end` ordering the edge hop's parent is the client record that follows it.

Op ids are the mechanism's (from graphs/alphabet.json), so the tokens a
consumer builds are the scoring target's tokens. Fault-phase labels go to
labels/ (not method-readable), never into the views.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ..constants import (
    ATTR_NONE, ATTR_SESSION, GRAINS, KIND_CLIENT, LABELS_DIR, N_SPECIALS, ORDERINGS, OUTCOME_4XX, OUTCOME_5XX,
    OUTCOME_ERR, OUTCOME_OK, OUTCOME_SLOW, PHASE_NONE, PHASE_POST, PHASE_PRE, PHASE_STRADDLE, SEQUENCES_DIR,
    SLOW_THRESHOLDS_JSON, SPLITS, VIEWS_DIR,
)
from ..constants import OUTCOME_NAMES, PAD, BOS, EOS, UNK
from ..rng import stable_id

ARROW_SCHEMA = pa.schema([
    ("trace_id", pa.string()),
    ("ops", pa.list_(pa.int32())),
    ("outcomes", pa.list_(pa.int8())),
    ("offsets", pa.list_(pa.int64())),
    ("durations", pa.list_(pa.int64())),
    ("parent_pos", pa.list_(pa.int32())),
    ("healthy", pa.bool_()),
    ("n_err_spans", pa.int16()),
    ("n_spans", pa.int32()),
    ("truncated", pa.bool_()),
    ("scenario_hash", pa.string()),
    ("scenario_sid", pa.int32()),
    ("start_ns", pa.int64()),
    ("attribution_level", pa.int8()),
    ("sequence_kind", pa.string()),
])
LABELS_SCHEMA = pa.schema([
    ("trace_id", pa.string()), ("grain", pa.string()), ("fault_index", pa.int16()), ("phase", pa.int8()),
    ("rc_positions_end", pa.list_(pa.int32())), ("rc_positions_start", pa.list_(pa.int32())), ("rc_took", pa.bool_()),
])
WRITE_BATCH = 50_000
EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)


def outcome_of(status, request_time_s, threshold_s):
    if status is None or status == 0:
        return OUTCOME_ERR
    if 400 <= status < 500:
        return OUTCOME_4XX
    if status >= 500:
        return OUTCOME_5XX
    if threshold_s is not None and request_time_s is not None and request_time_s > threshold_s:
        return OUTCOME_SLOW
    return OUTCOME_OK


def scenario_hash(pairs):
    payload = ";".join(f"{s},{d}" for s, d in sorted(pairs))
    return hashlib.sha1(payload.encode()).hexdigest()


def split_of(key):
    h = int(stable_id("split", key)[:8], 16) % 100
    return "train" if h < 80 else ("val" if h < 90 else "test")


def day_of(ts_ms):
    return (EPOCH + dt.timedelta(milliseconds=int(ts_ms))).strftime("%Y-%m-%d")


class OpIndex:
    """(service, path) -> op id from the alphabet; client pages by path."""

    def __init__(self, alphabet):
        self.by_key = {}
        self.client_by_path = {}
        self.ops = {}
        for t in alphabet["tokens"]:
            if t["op_id"] is None:
                continue
            self.ops.setdefault(t["op_id"], {"id": t["op_id"], "service": t["service"], "name": t["name"], "kind": t["kind"]})
            if t["kind"] == KIND_CLIENT:
                self.client_by_path[t["name"]] = t["op_id"]
            else:
                self.by_key[(t["service"], t["name"])] = t["op_id"]

    def op(self, service, path):
        return self.by_key.get((service, path))

    def client_op(self, page_url):
        path = "/" + page_url.split("/", 3)[-1] if page_url and "//" in page_url else page_url
        return self.client_by_path.get(path)


class Span:
    __slots__ = ("span_id", "op", "outcome", "start_ms", "end_ms", "parent", "kind", "trace_id", "level")

    def __init__(self, span_id, op, outcome, start_ms, end_ms, parent, kind, trace_id, level):
        self.span_id, self.op, self.outcome, self.start_ms, self.end_ms = span_id, op, outcome, start_ms, end_ms
        self.parent, self.kind, self.trace_id, self.level = parent, kind, trace_id, level


def build_spans(spans, trees, attribution, opindex, thresholds, link):
    """Hop and client spans with op ids, outcomes, times and parent span ids.
    `link` maps a client span id to the edge hop span id it resolves to."""
    out = {}
    for tid, tree in trees.items():
        for h in tree["hops"]:
            op = opindex.op(h["service"], h["path"])
            if op is None:
                continue
            thr = thresholds.get(op)
            outcome = outcome_of(h["status"], h["request_time_ms"] / 1000.0, thr)
            level = attribution.get(h["span_id"], (ATTR_NONE, None))[0]
            out[h["span_id"]] = Span(h["span_id"], op, outcome, h["start_ms"], h["ts_ms"], h["parent"], "hop", tid, level)
    edge_of_hop = {}
    for tid, tree in trees.items():
        if tree["edge"] is not None:
            edge_of_hop[tree["edge"]["span_id"]] = tid
    for s in spans:
        if s["kind"] not in ("sentry.error", "sentry.transaction"):
            continue
        op = opindex.client_op(s["http"].get("path"))
        if op is None:
            continue
        outcome = OUTCOME_ERR if s["kind"] == "sentry.error" else OUTCOME_OK
        edge_id = link.get(s["span_id"])
        tid = edge_of_hop.get(edge_id) if edge_id else None
        level = attribution.get(s["span_id"], (ATTR_NONE, None))[0]
        if edge_id and edge_id in out:
            start = out[edge_id].start_ms - 1
            # the client calls the edge: the client record is the edge hop's parent
            out[edge_id].parent = s["span_id"]
        else:
            dur = int((s["http"].get("request_time") or 0) * 1000)
            start = s["ts_ms"] - max(dur, 1)
        out[s["span_id"]] = Span(s["span_id"], op, outcome, start, s["ts_ms"], None, "client", tid, level)
    return out


def _depths(members):
    """Depth of every member under the inferred parent links (roots and spans
    whose parent lies outside the row are 0; a link cycle stops at its head)."""
    parent = {s.span_id: s.parent for s in members}
    depth = {}
    for s in members:
        chain, cur = [], s.span_id
        while cur is not None and cur not in depth and cur in parent and cur not in chain:
            chain.append(cur)
            cur = parent[cur]
        base = depth.get(cur, -1) if cur is not None and cur in depth else -1
        for i, sid in enumerate(reversed(chain)):
            depth[sid] = base + 1 + i
    return depth


def _rows_for(members, ordering, trace_id, level, kind):
    """One view row from a list of Span objects."""
    if not members:
        return None
    # Ties on the millisecond clock are broken by containment, then by the
    # inferred tree depth: a caller starts no later than its callee and ends no
    # earlier, so on a tied start the later-ending span is the parent (start
    # ordering) and on a tied end the later-starting span is the child (end
    # ordering); when both stamps tie (a caller whose own time rounds to zero)
    # the timestamps carry no order and the inferred link is the only evidence.
    # Span ids break the rest. Genuine inversions (a rounding step, browser
    # skew) are left as they are and counted in export-stats.json.
    depth = _depths(members)
    if ordering == "end":
        members = sorted(members, key=lambda s: (s.end_ms, -s.start_ms, -depth[s.span_id], s.span_id))
    else:
        members = sorted(members, key=lambda s: (s.start_ms, -s.end_ms, depth[s.span_id], s.span_id))
    pos = {s.span_id: i for i, s in enumerate(members)}
    t0 = min(s.start_ms for s in members)
    ops = [s.op for s in members]
    outcomes = [s.outcome for s in members]
    offsets = [(s.start_ms - t0) * 1_000_000 for s in members]
    durations = [max(0, s.end_ms - s.start_ms) * 1_000_000 for s in members]
    parent_pos = [pos.get(s.parent, -1) if s.parent else -1 for s in members]
    pairs = set()
    for s in members:
        src = members[pos[s.parent]].op if s.parent in pos else -1
        pairs.add((src, s.op))
    n_err = sum(1 for o in outcomes if o in (OUTCOME_4XX, OUTCOME_5XX, OUTCOME_ERR))
    return {"trace_id": trace_id, "ops": ops, "outcomes": outcomes, "offsets": offsets, "durations": durations,
            "parent_pos": parent_pos, "healthy": n_err == 0, "n_err_spans": n_err, "n_spans": len(members),
            "truncated": False, "scenario_hash": scenario_hash(pairs), "scenario_sid": -1, "start_ns": t0 * 1_000_000,
            "attribution_level": level, "sequence_kind": kind, "_day": day_of(t0), "_members": members}


class ViewWriter:
    def __init__(self, root):
        self.root = Path(root) / SEQUENCES_DIR
        self.buffers = defaultdict(list)
        self.counters = Counter()

    def add(self, split, day, row):
        buf = self.buffers[(split, day)]
        buf.append(row)
        if len(buf) >= WRITE_BATCH:
            self.flush(split, day)

    def flush(self, split, day):
        rows = self.buffers.pop((split, day), None)
        if not rows:
            return
        d = self.root / f"split={split}" / f"date={day}"
        d.mkdir(parents=True, exist_ok=True)
        n = self.counters[(split, day)]
        self.counters[(split, day)] += 1
        cols = {f.name: [r[f.name] for r in rows] for f in ARROW_SCHEMA}
        pq.write_table(pa.table(cols, schema=ARROW_SCHEMA), d / f"part-{n:04d}.parquet", compression="zstd")

    def flush_all(self):
        for key in sorted(self.buffers):
            self.flush(*key)


class ViewBuilder:
    """Streams view rows shard by shard; `finalize()` writes the vocabulary,
    prevalence, stats, thresholds and the sequence labels. Request rows are
    added per shard (a request tree lies inside one shard); session-grain rows
    are added as the stitcher closes sequences."""

    def __init__(self, corpus_dir, opindex, thresholds, faults, min_count, epoch_ms):
        self.corpus_dir = Path(corpus_dir)
        self.opindex, self.thresholds, self.faults, self.min_count = opindex, thresholds, faults, min_count
        self.epoch_ms = epoch_ms
        self.writers = {(o, g): ViewWriter(self.corpus_dir / VIEWS_DIR / f"{o}-{g}") for o in ORDERINGS for g in GRAINS}
        self.stats = {"rows": Counter(), "tokens_train": Counter(), "pairs_train": Counter(), "levels_request": Counter(),
                      "spans_in_views": 0, "links": Counter(), "violations": Counter()}
        (self.corpus_dir / LABELS_DIR).mkdir(exist_ok=True)
        self.label_rows = []
        self.label_writer = None

    def add_requests(self, all_spans, trees, attribution):
        by_trace = defaultdict(list)
        for s in all_spans.values():
            if s.trace_id is not None:
                by_trace[s.trace_id].append(s)
        stats = self.stats
        for tid in sorted(by_trace):
            members = by_trace[tid]
            edge = trees[tid]["edge"]
            # session key of a trace (for split assignment): the edge hop's attribution key when attributed
            level, key = attribution.get(edge["span_id"], (ATTR_NONE, None)) if edge else (ATTR_NONE, None)
            split = split_of(key if key else tid)
            stats["levels_request"][level] += 1
            row_start = None
            for ordering in ORDERINGS:
                row = _rows_for(members, ordering, tid, level, "request")
                if row is None:
                    continue
                self.writers[(ordering, "request")].add(split, row["_day"], row)
                stats["rows"][(ordering, "request", split)] += 1
                for j, pp in enumerate(row["parent_pos"]):
                    if pp >= 0:
                        stats["links"][ordering] += 1
                        if (pp <= j) if ordering == "end" else (pp >= j):
                            stats["violations"][ordering] += 1
                if ordering == "start":
                    row_start = row
                if ordering == "end":
                    if split == "train":
                        for op, o in zip(row["ops"], row["outcomes"]):
                            stats["tokens_train"][(op, o)] += 1
                        stats["pairs_train"][row["scenario_hash"]] += 1
                    stats["spans_in_views"] += row["n_spans"]
                    row_end = row
            if row_start is not None:
                self.label_rows.extend(_labels(tid, "request", row_end, row_start, self.faults, self.epoch_ms))
        self._flush_labels()

    def add_sequence(self, seq):
        members = [m[2] for m in seq["members"] if m[2] is not None]
        if not members:
            return
        split = split_of(seq["key"])
        for ordering in ORDERINGS:
            row = _rows_for(members, ordering, seq["sequence_id"], seq["level"], "session")
            self.writers[(ordering, "session")].add(split, row["_day"], row)
            self.stats["rows"][(ordering, "session", split)] += 1

    def _flush_labels(self, final=False):
        if not self.label_rows or (not final and len(self.label_rows) < WRITE_BATCH):
            return
        cols = {f.name: [r[f.name] for r in self.label_rows] for f in LABELS_SCHEMA}
        if self.label_writer is None:
            self.label_writer = pq.ParquetWriter(self.corpus_dir / LABELS_DIR / "sequence-labels.parquet", LABELS_SCHEMA, compression="zstd")
        self.label_writer.write_table(pa.table(cols, schema=LABELS_SCHEMA))
        self.label_rows = []

    def finalize(self):
        for w in self.writers.values():
            w.flush_all()
        self._flush_labels(final=True)
        if self.label_writer is not None:
            self.label_writer.close()
        else:
            pq.write_table(pa.table({f.name: [] for f in LABELS_SCHEMA}, schema=LABELS_SCHEMA),
                           self.corpus_dir / LABELS_DIR / "sequence-labels.parquet", compression="zstd")
        stats, opindex, thresholds, min_count = self.stats, self.opindex, self.thresholds, self.min_count
        base_ops = [opindex.ops[i] for i in sorted(opindex.ops)]
        variants = sorted((op, o) for (op, o), c in stats["tokens_train"].items() if o != OUTCOME_OK and c >= min_count)
        vocab = {"version": 1, "normalizer_version": "tracebench-1", "n_specials": N_SPECIALS,
                 "specials": {"PAD": PAD, "BOS": BOS, "EOS": EOS, "UNK": UNK},
                 "vocab_size": N_SPECIALS + len(base_ops) + len(variants),
                 "base_ops": [{"id": o["id"], "service": o["service"], "name": o["name"], "kind": o["kind"]} for o in base_ops],
                 "variants": [{"op_id": op, "outcome": OUTCOME_NAMES[o], "outcome_id": o} for op, o in variants]}
        for ordering in ORDERINGS:
            for grain in GRAINS:
                root = self.corpus_dir / VIEWS_DIR / f"{ordering}-{grain}"
                root.mkdir(parents=True, exist_ok=True)
                (root / "model-vocab.json").write_text(json.dumps(vocab, sort_keys=True))
                (root / "scenario-prevalence.json").write_text(json.dumps(dict(sorted(stats["pairs_train"].items())), sort_keys=True))
                (root / SLOW_THRESHOLDS_JSON).write_text(json.dumps({"thresholds_s": {str(k): v for k, v in sorted(thresholds.items())}}, sort_keys=True))
        export_stats = {
            "k_ops": len(base_ops), "vocab_size": vocab["vocab_size"], "n_variants": len(variants),
            "alphabet_size_realized_train": len(stats["tokens_train"]),
            "outcome_pairs_observed_train": sum(1 for (op, o) in stats["tokens_train"] if o != OUTCOME_OK),
            "rows": {f"{o}-{g}-{s}": stats["rows"][(o, g, s)] for o in ORDERINGS for g in GRAINS for s in SPLITS},
            "spans_in_request_views": stats["spans_in_views"],
            "attribution_level_of_requests": {str(k): v for k, v in sorted(stats["levels_request"].items())},
            "ordering": {"end": "spans sorted by emitted (completion) time; callee precedes caller; parent_pos[j] > j",
                         "start": "spans sorted by start time; parent_pos[j] < j"},
            "orientation_violations": {o: {"links": stats["links"][o], "violations": stats["violations"][o],
                                           "rate": stats["violations"][o] / max(stats["links"][o], 1),
                                           "note": "links whose parent does not follow (end) / precede (start) the child under emitted clock skew; counted, never dropped"}
                                       for o in ORDERINGS},
            "scenario_hash": "sha1 over sorted distinct (parent_op, op) pairs, -1 for roots",
            "split_rule": "blake2b(session key or trace id) mod 100: <80 train, <90 val, else test",
        }
        for ordering in ORDERINGS:
            for grain in GRAINS:
                (self.corpus_dir / VIEWS_DIR / f"{ordering}-{grain}" / "export-stats.json").write_text(json.dumps(export_stats, sort_keys=True, indent=1))
        return export_stats


def _labels(tid, grain, row_end, row_start, faults, epoch_ms):
    """Fault-phase labels of one row. Fault intervals are seconds from the
    window start; row times are epoch milliseconds, hence `epoch_ms`."""
    out = []
    t0_ms = row_end["start_ns"] // 1_000_000 - epoch_ms
    t1_ms = max(s.end_ms for s in row_end["_members"]) - epoch_ms
    for f in faults:
        lo, hi = f["start_s"] * 1000, f["end_s"] * 1000
        if t1_ms < lo or t0_ms >= hi:
            phase = PHASE_PRE if t1_ms < lo else PHASE_POST
            if t0_ms >= hi:
                phase = PHASE_NONE
        elif t0_ms >= lo and t1_ms < hi:
            phase = PHASE_POST
        else:
            phase = PHASE_STRADDLE
        rc_ops = set(f["_rc_ops"])
        rc_end = [i for i, s in enumerate(row_end["_members"]) if s.op in rc_ops]
        rc_start = [i for i, s in enumerate(row_start["_members"]) if s.op in rc_ops]
        took = any(row_end["_members"][i].outcome != OUTCOME_OK for i in rc_end)
        out.append({"trace_id": tid, "grain": grain, "fault_index": f["index"], "phase": phase,
                    "rc_positions_end": rc_end, "rc_positions_start": rc_start, "rc_took": took})
    return out
