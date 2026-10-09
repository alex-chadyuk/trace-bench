"""The metrics channel of a partially observable variant (D-TB-21).

A real cluster exports its infrastructure state as metric time series: one
series per metric family and topology label set, sampled on a regular scrape
grid, each sample the instantaneous level at the scrape instant. The metrics
variant gives a method exactly that for the exposed latent groups — traffic
intensity (one global series), per-service load and pool state, per-endpoint
health — and nothing for the hidden ones (cache contents, per-session network
and auth state), which a real system does not export either.

The series are a function of the latent chain alone: `latents.simulate_latents`
advances it from `instantiation.json` and the seed with the fault forcings
applied, exactly as `generate` did, so the samples agree with the state the
engine used for every hop and with the twin's state log. Sampling is lossy on
purpose: a state that flips and flips back between two scrape instants never
shows. The oracle keeps the tick-resolution change log with a per-change
`visible_at_scrape` flag, so the share of changes the channel misses is a
measured property of the corpus (`oracle/state/sampling.json`).

Files (method-readable under `metrics/`; `oracle/state/` is oracle):

    metrics/series.json                      the catalogue: cadence, families, every series
                                             with its state-token stem (`state:load:3`)
    metrics/shard=NNNN/samples.parquet       long rows (ts_ms, family, service, endpoint, value)
    metrics/shard=NNNN.done                  per-shard marker (resume)
    oracle/state/shard=NNNN/changes.parquet  every latent value change, exposed or hidden
    oracle/state/sampling.json               changes invisible at the scrape grid, per group

`ts_ms` counts milliseconds from the simulated window's start, the clock every
raw record's true emission time is on; the series carry no clock skew (the
scraper's clock is the oracle's). Levels are the mechanism's value names.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .constants import (
    LATENT_GROUPS, METRIC_FAMILIES, METRICS_DIR, METRICS_SAMPLES_PARQUET, METRICS_SCHEMA, METRICS_SCRAPE_S,
    METRICS_SERIES_JSON, ORACLE_STATE_DIR, STATE_CHANGES_PARQUET, STATE_SAMPLING_JSON, exposed_groups,
)
from .generate import _n_ticks, _shard_bounds
from .latents import Slots, compile_fault_forcings, initial_state, simulate_latents, state_events
from .log import log
from .mechanism import CACHE_VALUES, HEALTH_VALUES, INTENSITY_VALUES, LOAD_VALUES, POOL_VALUES
from .record import read_json, sha256_file, write_json
from .shards import shard_name

SAMPLES_SCHEMA = pa.schema([
    ("ts_ms", pa.int64()), ("family", pa.string()), ("service", pa.string()), ("endpoint", pa.string()),
    ("value", pa.string()),
])
CHANGES_SCHEMA = pa.schema([
    ("tick", pa.int64()), ("ts_ms", pa.int64()), ("node", pa.string()), ("group", pa.string()),
    ("family", pa.string()), ("service", pa.string()), ("endpoint", pa.string()),
    ("value", pa.string()), ("previous", pa.string()), ("exposed", pa.bool_()), ("visible_at_scrape", pa.bool_()),
])
VALUE_NAMES = {"intensity": INTENSITY_VALUES, "load": LOAD_VALUES, "pool": POOL_VALUES, "cache": CACHE_VALUES,
               "health": HEALTH_VALUES}
# groups a metrics channel can carry: the tick-level ones (session latents have no time series)
TICK_GROUPS = ("intensity", "load", "pool", "cache", "health")


def scrape_ticks(cfg, scrape_s=METRICS_SCRAPE_S):
    """Ticks between two scrapes; the grid must divide the tick and the shard."""
    tick_s, shard_ticks = cfg.run.tick_s, cfg.run.shard_ticks
    if scrape_s % tick_s:
        raise ValueError(f"scrape period {scrape_s}s is not a multiple of the tick ({tick_s}s)")
    k = scrape_s // tick_s
    if shard_ticks % k:
        raise ValueError(f"shard of {shard_ticks} ticks is not a multiple of the scrape grid ({k} ticks)")
    return k


def grid_ticks(t0, t1, k):
    """Scrape instants in [t0, t1): every tick that is a multiple of `k`."""
    first = -(-t0 // k) * k
    return list(range(first, t1, k))


class Catalogue:
    """The series of one variant in file order (family, service, endpoint), with
    the positions and latent-array slots that fill them from a LatentSlice."""

    def __init__(self, inst, slots: Slots, variant):
        self.variant = variant
        self.exposed = exposed_groups(variant)
        self.hidden = sorted(set(LATENT_GROUPS) - self.exposed)
        topo = inst.topo
        entries = []
        for group in TICK_GROUPS:
            if group not in self.exposed:
                continue
            family = METRIC_FAMILIES[group]
            if group == "intensity":
                entries.append((family, None, None, group, 0, "intensity"))
            elif group == "health":
                for slot, op_id in enumerate(slots.op_of_slot):
                    op = topo.ops[op_id]
                    entries.append((family, topo.services[op.service].name, op.name, group, slot, f"health:{op_id}"))
            else:
                for slot, svc_index in enumerate(slots.svc_of_slot):
                    entries.append((family, topo.services[svc_index].name, None, group, slot, f"{group}:{svc_index}"))
        entries.sort(key=lambda e: (e[0], e[1] or "", e[2] or ""))
        self.entries = entries
        self.family = [e[0] for e in entries]
        self.service = [e[1] for e in entries]
        self.endpoint = [e[2] for e in entries]
        self.node = [e[5] for e in entries]
        # per group: catalogue positions and the latent-array slots at those positions
        self.positions = {}
        for group in TICK_GROUPS:
            idx = [i for i, e in enumerate(entries) if e[3] == group]
            self.positions[group] = (np.array(idx, dtype=np.int64), np.array([entries[i][4] for i in idx], dtype=np.int64))
        self.names = {g: np.array(VALUE_NAMES[g], dtype=object) for g in TICK_GROUPS}

    def __len__(self):
        return len(self.entries)

    def values_at(self, sl, i):
        """Level name of every series at row `i` of a LatentSlice."""
        out = np.empty(len(self.entries), dtype=object)
        for group, (pos, slot) in self.positions.items():
            if not len(pos):
                continue
            if group == "intensity":
                out[pos] = self.names[group][int(sl.intensity[i])]
            else:
                out[pos] = self.names[group][getattr(sl, group)[i, slot]]
        return out

    def families_json(self):
        return {METRIC_FAMILIES[g]: {"group": g, "levels": list(VALUE_NAMES[g]),
                                     "labels": [] if g == "intensity" else (["service", "endpoint"] if g == "health" else ["service"])}
                for g in TICK_GROUPS if g in self.exposed}

    def series_json(self):
        return [{"family": f, "service": s, "endpoint": e, "token_stem": f"state:{node}", "levels": list(VALUE_NAMES[g])}
                for (f, s, e, g, _slot, node) in self.entries]


def _write_parquet_columns(path, cols, schema):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    table = pa.table({f.name: pa.array(cols[f.name], type=f.type) for f in schema}, schema=schema)
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)
    return path


def sample_columns(cat: Catalogue, sl, ticks, tick_s):
    """Long-format columns of the samples at `ticks` (a subset of the slice)."""
    n = len(cat)
    values = [cat.values_at(sl, t - sl.t0) for t in ticks]
    return {
        "ts_ms": np.repeat(np.array([t * tick_s * 1000 for t in ticks], dtype=np.int64), n),
        "family": cat.family * len(ticks), "service": cat.service * len(ticks), "endpoint": cat.endpoint * len(ticks),
        "value": list(np.concatenate(values)) if ticks else [],
    }


def _node_parts(ev):
    group = ev["node"].split(":")[0]
    endpoint = ev["subject"] if group == "health" else None
    return group, endpoint


def change_rows(sl, slots, topo, prev_state, prev_intensity, exposed, k, tick_s, next_row=None):
    """The latent value changes of one slice as CHANGES_SCHEMA rows, each with
    whether the next scrape shows the new value. `next_row` is the state at the
    first tick after the slice (the following shard's first row), needed for
    changes in the slice's last scrape interval; None leaves them unresolved."""
    events = state_events(sl, slots, topo, prev_state)
    if prev_intensity is not None and int(sl.intensity[0]) != int(prev_intensity):
        events.append({"tick": sl.t0, "node": "intensity", "service": None, "subject": "intensity",
                       "value": INTENSITY_VALUES[int(sl.intensity[0])], "previous": INTENSITY_VALUES[int(prev_intensity)]})
        events.sort(key=lambda e: (e["tick"], e["node"]))
    t1 = sl.t0 + len(sl.intensity)
    rows = []
    for ev in events:
        group, endpoint = _node_parts(ev)
        g = -(-ev["tick"] // k) * k                       # the next scrape instant at or after the change
        if g < t1:
            at = sl.at(g)
        elif next_row is not None:
            at = next_row
        else:
            at = None
        if at is None:
            visible = None
        elif group == "intensity":
            visible = INTENSITY_VALUES[int(at["intensity"])] == ev["value"]
        else:
            slot = slots.slot_of_op[int(ev["node"].split(":")[1])] if group == "health" else slots.slot_of_svc[int(ev["node"].split(":")[1])]
            visible = VALUE_NAMES[group][int(at[group][slot])] == ev["value"]
        rows.append({"tick": ev["tick"], "ts_ms": ev["tick"] * tick_s * 1000, "node": ev["node"], "group": group,
                     "family": METRIC_FAMILIES.get(group), "service": ev["service"], "endpoint": endpoint,
                     "value": ev["value"], "previous": ev["previous"], "exposed": group in exposed,
                     "visible_at_scrape": visible})
    return rows


def _rows_to_columns(rows, schema):
    return {f.name: [r.get(f.name) for r in rows] for f in schema}


def done_path(corpus_dir, shard):
    return Path(corpus_dir) / METRICS_DIR / (shard_name(shard) + ".done")


def derive_metrics(corpus_dir, inst, seed, variant, resume=False, scrape_s=METRICS_SCRAPE_S):
    """Write the metrics channel and the oracle state log of a corpus whose
    `instantiation.json`, `config.yaml` and `constants.json` are in place. One
    sequential pass over the window: the chain is advanced shard by shard from
    all-nominal with the fault forcings, exactly as `generate` did, and every
    shard's samples and changes are written atomically behind a marker. With
    `resume`, a shard whose marker exists is simulated (the chain must advance)
    but not rewritten. Returns the summary written to `sampling.json`."""
    corpus_dir = Path(corpus_dir)
    cfg, topo = inst.cfg, inst.topo
    tick_s = cfg.run.tick_s
    k = scrape_ticks(cfg, scrape_s)
    slots = Slots.build(topo)
    cat = Catalogue(inst, slots, variant)
    forcings, _ = compile_fault_forcings(inst, slots)
    bounds = _shard_bounds(cfg)
    n_ticks = _n_ticks(cfg)
    counts = {g: {"changes": 0, "invisible_at_scrape": 0, "unresolved": 0} for g in TICK_GROUPS}
    n_samples = 0
    written = []

    def flush(shard, sl, prev_state, prev_intensity, next_row):
        nonlocal n_samples
        _, t0, t1 = bounds[shard]
        ticks = grid_ticks(t0, t1, k)
        rows = change_rows(sl, slots, topo, prev_state, prev_intensity, cat.exposed, k, tick_s, next_row)
        for r in rows:
            c = counts[r["group"]]
            c["changes"] += 1
            if r["visible_at_scrape"] is None:
                c["unresolved"] += 1
            elif not r["visible_at_scrape"]:
                c["invisible_at_scrape"] += 1
        n_samples += len(ticks) * len(cat)
        marker = done_path(corpus_dir, shard)
        if resume and marker.exists():
            log({"event": "metrics_shard", "shard": shard, "status": "kept", "samples": len(ticks) * len(cat), "changes": len(rows)})
            return
        sdir = corpus_dir / METRICS_DIR / shard_name(shard)
        odir = corpus_dir / ORACLE_STATE_DIR / shard_name(shard)
        files = [
            _write_parquet_columns(sdir / METRICS_SAMPLES_PARQUET, sample_columns(cat, sl, ticks, tick_s), SAMPLES_SCHEMA),
            _write_parquet_columns(odir / STATE_CHANGES_PARQUET, _rows_to_columns(rows, CHANGES_SCHEMA), CHANGES_SCHEMA),
        ]
        rec = {"shard": shard, "t0": t0, "t1": t1, "n_samples": len(ticks) * len(cat), "n_changes": len(rows),
               "files": {p.relative_to(corpus_dir).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in files}}
        tmp = marker.with_suffix(".done.tmp")
        write_json(tmp, rec)
        os.replace(tmp, marker)
        written.append(shard)
        log({"event": "metrics_shard", "shard": shard, "status": "written", "samples": rec["n_samples"], "changes": len(rows)})

    state = initial_state(slots)
    prev_intensity = None
    pending = None            # (shard, slice, prev_state, prev_intensity) awaiting the next shard's first row
    for shard, t0, t1 in bounds:
        prev_state = {a: v.copy() for a, v in state.items()}
        sl, state = simulate_latents(inst, seed, slots, state, t0, t1, forcings)
        if pending is not None:
            flush(*pending, next_row=sl.at(t0))
        pending = (shard, sl, prev_state, prev_intensity)
        prev_intensity = int(sl.intensity[-1])
    if pending is not None:
        flush(*pending, next_row=None)

    per_group = {}
    for g, c in counts.items():
        resolved = c["changes"] - c["unresolved"]
        per_group[g] = {**c, "exposed": g in cat.exposed,
                        "share_invisible_at_scrape": (c["invisible_at_scrape"] / resolved) if resolved else 0.0}
    sampling = {"schema": METRICS_SCHEMA, "variant": variant, "scrape_s": scrape_s, "tick_s": tick_s,
                "exposed_groups": sorted(cat.exposed), "hidden_groups": cat.hidden,
                "n_series": len(cat), "n_samples": n_samples, "n_shards": len(bounds), "per_group": per_group,
                "note": "a change is invisible when the next scrape at or after it already shows a later value; "
                        "changes in the last interval of the window are unresolved (no later scrape)"}
    write_json(corpus_dir / ORACLE_STATE_DIR / STATE_SAMPLING_JSON, sampling)
    ticks_all = grid_ticks(0, n_ticks, k)
    series = {"schema": METRICS_SCHEMA, "variant": variant, "scrape_s": scrape_s, "tick_s": tick_s,
              "epoch": cfg.run.window.start,
              "ts_semantics": "ts_ms = milliseconds since `epoch` of the scrape instant; a sample is the instantaneous "
                              "level at that tick (after fault injection), the state the engine used for the hops of "
                              "that tick; no clock skew (the raw feed's `ts` carries per-pod skew); no 'unknown' "
                              "samples: scraping starts with the window and misses no scrape",
              "window_ticks": n_ticks, "n_shards": len(bounds), "shard_ticks": cfg.run.shard_ticks,
              "n_samples_per_series": len(ticks_all), "first_ts_ms": ticks_all[0] * tick_s * 1000 if ticks_all else None,
              "last_ts_ms": ticks_all[-1] * tick_s * 1000 if ticks_all else None,
              "families": cat.families_json(), "n_series": len(cat), "series": cat.series_json(),
              "exposed_groups": sorted(cat.exposed), "hidden_groups": cat.hidden,
              "token_semantics": "a sample (series, value) is the state token `<token_stem>=<value>` of the alphabet",
              "note": "one series per (family, service[, endpoint]); service-level, no per-pod replication; "
                      "levels are the mechanism's categorical values, not fitted numeric gauges"}
    write_json(corpus_dir / METRICS_DIR / METRICS_SERIES_JSON, series)
    return {**sampling, "shards_written": written}


def completed_metrics_shards(corpus_dir):
    d = Path(corpus_dir) / METRICS_DIR
    if not d.exists():
        return []
    return sorted(int(p.name[len("shard="):-len(".done")]) for p in d.glob("shard=*.done"))


def read_samples(corpus_dir, shard):
    return pq.read_table(Path(corpus_dir) / METRICS_DIR / shard_name(shard) / METRICS_SAMPLES_PARQUET).to_pydict()


def read_changes(corpus_dir, shard):
    return pq.read_table(Path(corpus_dir) / ORACLE_STATE_DIR / shard_name(shard) / STATE_CHANGES_PARQUET).to_pydict()


def series_catalogue(corpus_dir):
    return read_json(Path(corpus_dir) / METRICS_DIR / METRICS_SERIES_JSON)
