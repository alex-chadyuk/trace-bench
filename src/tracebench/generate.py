"""Generate a corpus from a named-instance configuration.

    python -m tracebench.generate --config configs/instances/xs.yaml --seed 0 --out corpora \\
        [--twin] [--override-cap] [--resume] [--stop-after-shard N] [--workers N]

Order of operations (every step is a pure function of configuration, seed,
tool version and constants):
  1. validate the configuration and load the fitted constants (never any
     network source);
  2. instantiate the system and write its record (instantiation.json,
     topology, thresholds);
  3. estimate the output size and refuse above the cap unless overridden;
  4. write the ground truth BEFORE any data: mechanism graphs per regime,
     alphabet, scoring targets, floor sweep, views;
  5. compute the latent trajectory's shard checkpoints once, sequentially;
  6. generate shards (skipping completed ones on --resume);
  7. write fault and case labels, the shard index and the run record.
The correlated views are produced by `python -m tracebench.correlate`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .constants import (
    CASES_JSON, COMPLETE_MARKER, ESTIMATE_JSON, FAULTS_JSON, LABELS_DIR, RUN_DIR, VARIANT_LATENT, VARIANT_TWIN,
)
from .emit import Emitter
from .engine import Engine, SpillExceeded
from .estimate import alphabet_estimate, expected_invocations
from .graphs import write_graph_artifacts
from .instantiate import instantiate_from_paths, load_instantiation, write_instantiation
from .latents import Slots, compile_fault_forcings, shard_checkpoints, simulate_latents
from .log import log
from .record import RunRecord, write_json
from .shards import completed_shards, write_shard


class CapExceeded(RuntimeError):
    pass


def corpus_dir_for(out, cfg_name, variant, seed):
    return Path(out) / cfg_name / variant / f"seed={seed}"


def estimate_size(inst):
    """Expected corpus size (GB) and CPU-hours from expected counts and the
    fitted per-record byte constants; recorded before any shard is written."""
    c = inst.constants
    inv, attempts, sessions = expected_invocations(inst)
    topo = inst.topo
    hops = sum(inv[op.id] * attempts[op.id] for op in topo.ops if op.kind != 2)
    requests = sum(inv[b] for b in topo.bff_ops)
    err_rate = 0.05
    app_lines = hops * ((1 - err_rate) * c["records_per_request.app_ok"] + err_rate * c["records_per_request.app_err"])
    audit = requests * c["records_per_request.audit"] + sum(inv[o] for o in topo.external_ops)
    split_extra = (app_lines + audit) * c["split.rate"] * 1.5
    client = requests * (c["client.txn_sampling"] + c["client.error_rate"] + 0.02)
    n_pods = sum(len(s.pods) for s in topo.services)
    window_s = inst.cfg.window_seconds
    health = n_pods * window_s / c["health_check_period_s"]
    background = c["background.app_lines_per_service_s"] * window_s * sum(1 for s in topo.services if s.kind != 2)
    vl_records = hops + app_lines + audit + split_extra + health + background
    sentry_records = client
    raw_bytes = (vl_records * c["bytes_per_record.vl"] + sentry_records * c["bytes_per_record.sentry"]) * c["gzip_ratio"]
    oracle_bytes = (vl_records + sentry_records) * 90 + hops * 60
    views_bytes = hops * 12 * 4
    total = raw_bytes + oracle_bytes + views_bytes
    cpu_hours = (vl_records + sentry_records) / 20000.0 / 3600.0 * 1.5
    return {
        "expected_sessions": sessions, "expected_requests": requests, "expected_hops": hops,
        "expected_vl_records": vl_records, "expected_sentry_records": sentry_records,
        "estimate_gb": total / 1e9, "estimate_raw_gb": raw_bytes / 1e9, "cap_gb": inst.cfg.run.cap_gb,
        "estimate_cpu_hours": cpu_hours,
    }


def _n_ticks(cfg):
    return int(round(cfg.window_seconds / cfg.run.tick_s))


def _shard_bounds(cfg):
    n = _n_ticks(cfg)
    st = cfg.run.shard_ticks
    bounds = []
    t = 0
    while t < n:
        bounds.append((len(bounds), t, min(t + st, n)))
        t += st
    return bounds


SPILL_RETRY_MARGIN = 600     # ticks added beyond what the failed attempt asked for
SPILL_RETRY_MAX = 4


def run_one_shard(inst, corpus_dir, variant, seed, shard, t0, t1, checkpoint, forcings, spill_ticks):
    slots = Slots.build(inst.topo)
    n_total = _n_ticks(inst.cfg)
    # `spill_allowance` is a tail heuristic, not a bound: a long enough journey
    # outruns it (D-TB-16). When that happens the engine says how many ticks it
    # needed, and the shard is re-simulated and re-run at that size. This is
    # byte-identical to having started with the larger allowance, because the
    # latent uniforms are keyed by the absolute tick and the chain advances from
    # `checkpoint` — a longer slice holds the same values at every tick the
    # shorter one held. The retry costs one shard's work.
    spill = spill_ticks
    for attempt in range(SPILL_RETRY_MAX):
        latents, _ = simulate_latents(inst, seed, slots, checkpoint, t0, min(t1 + spill, n_total + spill), forcings)
        engine = Engine(inst, seed, slots, forcings)
        try:
            res = engine.run_shard(shard, t0, t1, latents)
        except SpillExceeded as e:
            needed = e.needed_ticks - (t1 - t0) + SPILL_RETRY_MARGIN
            if attempt == SPILL_RETRY_MAX - 1 or needed <= spill:
                raise RuntimeError(
                    f"shard {shard}: a journey spilled past the latent slice and {SPILL_RETRY_MAX} "
                    f"attempts did not suffice (held {e.held_ticks} ticks, needed {e.needed_ticks}); "
                    f"raise the spill allowance") from e
            log({"event": "spill_retry", "shard": shard, "attempt": attempt + 1,
                 "held_ticks": e.held_ticks, "needed_ticks": e.needed_ticks,
                 "spill_ticks": spill, "next_spill_ticks": needed})
            spill = needed
            continue
        break
    if res.spill_ticks > spill:
        raise RuntimeError(f"shard {shard}: journeys spilled {res.spill_ticks} ticks past the slice; raise the spill allowance")
    emitter = Emitter(inst, seed, slots, variant)
    records, oracle = emitter.emit_shard(res)
    marker = write_shard(corpus_dir, shard, records, oracle, twin=(variant == VARIANT_TWIN),
                         extra={"t0": t0, "t1": t1, "n_requests": len(res.requests), "n_hops": len(res.hops),
                                "n_clients": len(res.clients)})
    return marker


def spill_allowance(inst):
    """Ticks a journey may extend past its arrival shard: steps x attempts x
    (max step gap + retry backoff), rounded up to whole ticks."""
    cfg = inst.cfg
    max_steps = max(s.steps for s in cfg.scenarios)
    max_att = max(s.retry.max_retries for s in cfg.scenarios) + 1
    gap_max = min(3 * inst.constants["session.step_gap_quantiles.p99"], 3600.0)
    per_request = 3 * inst.constants["latency_quantiles.bff.p99"] * 4 + 2.0
    seconds = max_steps * (gap_max + max_att * per_request) + 5
    return int(np.ceil(seconds / cfg.run.tick_s))


def generate(config_path, seed, out, twin=False, override_cap=False, resume=False, stop_after_shard=None, workers=1,
             correlate=True):
    inst = instantiate_from_paths(config_path, seed=seed)
    cfg = inst.cfg
    variant = VARIANT_TWIN if twin else VARIANT_LATENT
    corpus_dir = corpus_dir_for(out, cfg.name, variant, seed)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    rec = RunRecord(corpus_dir, "generate", {"config": str(config_path), "seed": seed, "out": str(out), "twin": twin,
                                             "override_cap": override_cap, "resume": resume,
                                             "stop_after_shard": stop_after_shard, "workers": workers})
    # The call-graph artifact's `derived` date is the simulated window's start
    # date, a function of the configuration alone — not the wall clock, which
    # made a corpus regenerated on another day differ by one file (D-TB-18).
    derived = cfg.run.window.start[:10]
    # 2. instantiation record
    write_instantiation(inst, corpus_dir, derived)
    # 3. size estimate and cap
    est = estimate_size(inst)
    est["alphabet"] = alphabet_estimate(inst)
    write_json(corpus_dir / RUN_DIR / ESTIMATE_JSON, est)
    rec.note(estimate_gb=est["estimate_gb"], cap_gb=est["cap_gb"], estimate_cpu_hours=est["estimate_cpu_hours"])
    log({"event": "estimate", **{k: v for k, v in est.items() if k != "alphabet"}})
    if est["estimate_gb"] > cfg.run.cap_gb and not override_cap:
        rec.finish({"refused": True, "estimate_gb": est["estimate_gb"], "cap_gb": cfg.run.cap_gb}, status="refused")
        raise CapExceeded(f"estimated {est['estimate_gb']:.2f} GB exceeds the cap of {cfg.run.cap_gb:g} GB; pass --override-cap to proceed")
    if est["estimate_cpu_hours"] > 12.0 and cfg.rung == "local":
        log({"event": "warning", "message": f"estimated {est['estimate_cpu_hours']:.1f} CPU-hours exceeds the 12-hour local budget"})
    # 4. ground truth before any data
    graphs_summary = write_graph_artifacts(inst, corpus_dir, variant)
    rec.note(graphs_written_before_shards=True, **{f"graphs_{k}": v for k, v in graphs_summary.items() if not isinstance(v, dict)})
    log({"event": "graphs", **{k: v for k, v in graphs_summary.items()}})
    # 5. latent checkpoints
    slots = Slots.build(inst.topo)
    forcings, fault_records = compile_fault_forcings(inst, slots)
    bounds = _shard_bounds(cfg)
    checkpoints = shard_checkpoints(inst, seed, slots, cfg.run.shard_ticks, _n_ticks(cfg), forcings)
    spill = spill_allowance(inst)
    # 6. shards
    done = set(completed_shards(corpus_dir)) if resume else set()
    todo = [(k, t0, t1) for (k, t0, t1) in bounds if k not in done]
    if stop_after_shard is not None:
        todo = [b for b in todo if b[0] <= stop_after_shard]
    markers = {}
    if workers > 1 and len(todo) > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor  # a dead worker raises BrokenProcessPool instead of hanging
        jobs = [(variant, seed, k, t0, t1, checkpoints[k], forcings, spill) for (k, t0, t1) in todo]
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_worker_init,
                                 initargs=(str(corpus_dir),)) as pool:
            for m in pool.map(_worker_run, jobs):
                markers[m["shard"]] = m
                log({"event": "shard", "shard": m["shard"], "records": m["n_records"], "spans": m["n_spans"]})
    else:
        for (k, t0, t1) in todo:
            m = run_one_shard(inst, corpus_dir, variant, seed, k, t0, t1, checkpoints[k], forcings, spill)
            markers[k] = m
            log({"event": "shard", "shard": k, "records": m["n_records"], "spans": m["n_spans"]})
    # 7. labels and record
    (corpus_dir / LABELS_DIR).mkdir(exist_ok=True)
    write_json(corpus_dir / LABELS_DIR / FAULTS_JSON, {"faults": fault_records, "window_s": cfg.window_seconds})
    write_json(corpus_dir / LABELS_DIR / CASES_JSON, {"cases": cases_from_faults(inst, fault_records)})
    complete = stop_after_shard is None and len(completed_shards(corpus_dir)) == len(bounds)
    results = {"instance": cfg.name, "variant": variant, "seed": seed, "corpus_dir": str(corpus_dir),
               "n_shards": len(bounds), "shards_written": sorted(markers), "shards_completed": completed_shards(corpus_dir),
               "complete": complete, "spill_ticks": spill, "estimate": {k: v for k, v in est.items() if k != "alphabet"},
               "graphs": graphs_summary}
    if complete and correlate:
        from .correlate.__main__ import correlate as run_correlate
        stats, report = run_correlate(corpus_dir)
        results["views"] = stats
        results["correlation"] = {k: v for k, v in report.items() if k in ("parent_link", "unattributed_fraction", "session_recovery")}
        log({"event": "correlate", "parent_link_f1": report["parent_link"]["all"]["f1"], "unattributed_fraction": report["unattributed_fraction"]})
    if complete:
        # The manifest hashes every shipped file (run/ excluded); COMPLETE is
        # written only after it exists.
        from .manifest import write_manifest
        manifest = write_manifest(corpus_dir)
        results["manifest"] = {"files": len(manifest["files"]), "alphabet_size_realized_train": manifest["alphabet_size_realized_train"]}
        (corpus_dir / COMPLETE_MARKER).write_text("shards complete" + ("; views written" if correlate else "; run `python -m tracebench.correlate`") + "\n")
    rec.finish(results, status="ok" if complete else "partial")
    return results


def cases_from_faults(inst, fault_records):
    cfg = inst.cfg
    return [{"case_id": f"{cfg.name}_{f['service']}_{f['kind']}_{f['index']}", "system": cfg.name,
             "root_cause_component": f["service"], "root_cause_endpoints": f["endpoints"], "indicator": f["indicator"],
             "fault_kind": f["kind"], "inject_time_s": f["start_s"], "end_time_s": f["end_s"],
             "forced_nodes": f["forced_nodes"], "forced_value": f["forced_value"]} for f in fault_records]


_WORKER = {}


def _worker_init(corpus_dir):
    _WORKER["inst"] = load_instantiation(corpus_dir)
    _WORKER["corpus_dir"] = corpus_dir


def _worker_run(job):
    variant, seed, k, t0, t1, checkpoint, forcings, spill = job
    return run_one_shard(_WORKER["inst"], _WORKER["corpus_dir"], variant, seed, k, t0, t1, checkpoint, forcings, spill)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--out", required=True)
    p.add_argument("--twin", action="store_true", help="generate the fully-observable twin (latent values written onto records)")
    p.add_argument("--override-cap", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after-shard", type=int, default=None)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--skip-correlate", action="store_true", help="do not run the bundled correlator after the shards")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        res = generate(args.config, args.seed, args.out, twin=args.twin, override_cap=args.override_cap,
                       resume=args.resume, stop_after_shard=args.stop_after_shard, workers=args.workers,
                       correlate=not args.skip_correlate)
    except CapExceeded as e:
        log({"event": "refused", "reason": str(e)})
        return 3
    log({"event": "done", "complete": res["complete"], "corpus_dir": res["corpus_dir"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
