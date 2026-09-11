"""Shard files: atomic writes, done markers, resume.

A shard's outputs are written to temporary names and renamed; the marker
`raw/shard=NNNN.done` (per-file sha256) is written last, so a shard is either
complete or absent. A resumed run skips shards with a marker (PRD scenario
24): because every draw is a function of identifiers, the skipped shards are
byte-identical to what an uninterrupted run would have written.

Layout per shard:
    raw/shard=NNNN/vl.jsonl.gz          access, app, audit records (emission order)
    raw/shard=NNNN/sentry.jsonl.gz      client-side records
    raw/shard=NNNN/state.jsonl.gz       twin only: latent state changes
    oracle/shard=NNNN/linkage.parquet   true parent record and membership of every record
    oracle/shard=NNNN/spans.parquet     true request trees
    oracle/shard=NNNN/sessions.parquet  true sessions
"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .constants import (
    KIND_ACCESS, KIND_APP, KIND_AUDIT, KIND_SENTRY_ERROR, KIND_SENTRY_TXN, KIND_STATE_EVENT, ORACLE_DIR, RAW_DIR,
)
from .record import sha256_file, write_json

VL_KINDS = (KIND_ACCESS, KIND_APP, KIND_AUDIT)
SENTRY_KINDS = (KIND_SENTRY_ERROR, KIND_SENTRY_TXN)

LINKAGE_SCHEMA = pa.schema([
    ("log_id", pa.string()), ("part", pa.int16()), ("kind", pa.string()), ("true_ts_ms", pa.int64()),
    ("component", pa.string()), ("op", pa.int32()), ("pod", pa.string()), ("request_id", pa.string()),
    ("edge_request_id", pa.string()), ("parent_log_id", pa.string()), ("session_gid", pa.int64()),
    ("device_index", pa.int64()), ("req_gid", pa.int64()), ("request_row", pa.int64()), ("attempt", pa.int16()),
    ("hop_row", pa.int64()),
])
SPANS_SCHEMA = pa.schema([
    ("span_id", pa.string()), ("parent_span_id", pa.string()), ("op", pa.int32()), ("service", pa.string()),
    ("name", pa.string()), ("outcome", pa.int8()), ("status", pa.int32()), ("attempt", pa.int16()),
    ("start_ms", pa.int64()), ("dur_ms", pa.int64()), ("own_ms", pa.int64()), ("own_us", pa.int64()), ("dur_us", pa.int64()),
    ("req_gid", pa.int64()),
    ("edge_request_id", pa.string()), ("session_gid", pa.int64()), ("is_edge", pa.bool_()), ("tick", pa.int64()),
    ("state_health", pa.int8()), ("state_pool", pa.int8()), ("state_cache", pa.int8()), ("state_load", pa.int8()),
    ("state_intensity", pa.int8()),
])
SESSIONS_SCHEMA = pa.schema([
    ("gid", pa.int64()), ("scenario", pa.int32()), ("arrival_ms", pa.int64()), ("device_index", pa.int64()),
    ("device_id", pa.string()), ("session_id", pa.string()), ("cart_id", pa.string()), ("logged_in", pa.bool_()),
    ("net", pa.int8()), ("auth", pa.int8()), ("steps_done", pa.int32()), ("final_ok", pa.bool_()),
])


def shard_name(shard):
    return f"shard={shard:04d}"


def done_path(corpus_dir, shard):
    return Path(corpus_dir) / RAW_DIR / (shard_name(shard) + ".done")


def completed_shards(corpus_dir):
    raw = Path(corpus_dir) / RAW_DIR
    if not raw.exists():
        return []
    out = []
    for p in sorted(raw.glob("shard=*.done")):
        try:
            out.append(int(p.name[len("shard="):-len(".done")]))
        except ValueError:
            continue
    return out


def _write_jsonl_gz(path, records):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.GzipFile(filename="", mode="wb", fileobj=open(tmp, "wb"), mtime=0, compresslevel=6) as gz:
        for r in records:
            gz.write((json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8"))
    os.replace(tmp, path)
    return path


def _write_parquet(path, rows, schema):
    tmp = path.with_suffix(path.suffix + ".tmp")
    cols = {f.name: [r.get(f.name) for r in rows] for f in schema}
    table = pa.table(cols, schema=schema)
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)
    return path


def write_shard(corpus_dir, shard, records, oracle, twin=False, extra=None):
    """Writes every file of a shard atomically, then the done marker."""
    corpus_dir = Path(corpus_dir)
    raw = corpus_dir / RAW_DIR / shard_name(shard)
    ora = corpus_dir / ORACLE_DIR / shard_name(shard)
    raw.mkdir(parents=True, exist_ok=True)
    ora.mkdir(parents=True, exist_ok=True)
    written = []
    written.append(_write_jsonl_gz(raw / "vl.jsonl.gz", [r for r in records if r["kind"] in VL_KINDS]))
    written.append(_write_jsonl_gz(raw / "sentry.jsonl.gz", [r for r in records if r["kind"] in SENTRY_KINDS]))
    if twin:
        written.append(_write_jsonl_gz(raw / "state.jsonl.gz", [r for r in records if r["kind"] == KIND_STATE_EVENT]))
    written.append(_write_parquet(ora / "linkage.parquet", oracle["linkage"], LINKAGE_SCHEMA))
    written.append(_write_parquet(ora / "spans.parquet", oracle["spans"], SPANS_SCHEMA))
    written.append(_write_parquet(ora / "sessions.parquet", oracle["sessions"], SESSIONS_SCHEMA))
    files = {str(p.relative_to(corpus_dir).as_posix()): {"bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in written}
    marker = {"shard": shard, "files": files, "n_records": len(records), "n_spans": len(oracle["spans"]),
              "n_sessions": len(oracle["sessions"]), **(extra or {})}
    tmp = done_path(corpus_dir, shard).with_suffix(".done.tmp")
    write_json(tmp, marker)
    os.replace(tmp, done_path(corpus_dir, shard))
    return marker


def read_records(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def shard_markers(corpus_dir):
    from .record import read_json
    return {s: read_json(done_path(corpus_dir, s)) for s in completed_shards(corpus_dir)}
