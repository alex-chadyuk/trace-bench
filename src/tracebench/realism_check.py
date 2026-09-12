"""Realism as a measurement (PRD scenario 9): compare a corpus's realised
statistics with the fitted constants it was generated from.

Quantities and tolerances (PRD): per-hop latency quantiles, error rate,
retry count, call-graph fan-out and depth, session inactivity gap and
operation vocabulary size; quantiles and counts within 10 % relative, rates
within one percentage point absolute or 10 % relative, whichever is larger.
Latency and error statistics are read at nominal state (the oracle's state
columns), because the constants describe the nominal regime; a configured
`error_rate_multiplier` other than 1 is a declared deviation and is reported
as such rather than hidden.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .constants import KIND_BFF, KIND_EXTERNAL, KIND_SERVICE, OUTCOME_OK, OUTCOME_SLOW, REALISM_REPORT_JSON, REPORTS_DIR
from .instantiate import load_instantiation
from .latency import TIER_OF_KIND
from .log import log
from .record import RunRecord, write_json
from .topology import depth_distribution, service_layer

QS = ("p50", "p90", "p95", "p99")
QP = (0.5, 0.9, 0.95, 0.99)


def _rel_ok(realised, constant, rel=0.10):
    if constant == 0:
        return abs(realised) < 1e-9
    return abs(realised - constant) / abs(constant) <= rel


def _rate_ok(realised, constant):
    return abs(realised - constant) <= max(0.01, 0.10 * abs(constant))


def _pick_shards(corpus_dir, max_shards):
    shards = sorted((Path(corpus_dir) / "oracle").glob("shard=*/"))
    if max_shards and len(shards) > max_shards:
        idx = np.linspace(0, len(shards) - 1, max_shards).round().astype(int)
        shards = [shards[i] for i in sorted(set(int(i) for i in idx))]
    return shards


def _invoked_fanout(shards):
    """(mean invoked callees per calling hop, number of calling hops): first
    attempts grouped by their parent span, over hops that invoked at least one
    callee. A request's spans all lie in one shard, so shards count apart."""
    children = parents = 0
    for sh in shards:
        t = pq.read_table(sh / "spans.parquet", columns=["parent_span_id", "attempt"])
        t = t.filter(pc.and_(pc.equal(t["attempt"], 0), pc.is_valid(t["parent_span_id"])))
        children += t.num_rows
        parents += pc.count_distinct(t["parent_span_id"]).as_py() if t.num_rows else 0
    return (children / parents if parents else 0.0), parents


def _columns(shards, name, columns):
    tables = [pq.read_table(sh / name, columns=columns) for sh in shards]
    t = pa.concat_tables(tables) if tables else None
    if t is None:
        return {c: np.array([]) for c in columns}
    return {c: t.column(c).to_numpy(zero_copy_only=False) for c in columns}


def realism_report(corpus_dir, max_shards=12):
    """Realism items over an evenly spaced sample of at most `max_shards` oracle
    shards (the quantities are per-hop and per-session statistics, stable under
    sampling; the whole corpus at m+ does not fit in memory as Python objects)."""
    corpus_dir = Path(corpus_dir)
    inst = load_instantiation(corpus_dir)
    c = inst.constants
    topo = inst.topo
    mult = inst.cfg.mechanism.error_rate_multiplier
    n_ops_total = max(op.id for op in topo.ops) + 1
    kind_lookup = np.zeros(n_ops_total, dtype=np.int64)
    depth_lookup = np.zeros(n_ops_total, dtype=np.int64)
    leaf_lookup = np.zeros(n_ops_total, dtype=bool)
    for op in topo.ops:
        kind_lookup[op.id] = op.kind
        depth_lookup[op.id] = service_layer(op)
        leaf_lookup[op.id] = len(topo.callee_edges(op.id)) == 0
    shards = _pick_shards(corpus_dir, max_shards)
    cols = _columns(shards, "spans.parquet", ["op", "own_us", "outcome", "attempt", "state_health", "state_pool", "state_cache", "req_gid"])
    n = int(len(cols["op"]))
    items = []
    if n == 0:
        return {"items": items, "n_spans": 0, "all_pass": False, "shards_sampled": len(shards)}
    op = cols["op"].astype(np.int64); own = cols["own_us"].astype(np.float64) / 1e6
    outcome = cols["outcome"].astype(np.int64); attempt = cols["attempt"].astype(np.int64)
    nominal = (cols["state_health"] == 0) & (cols["state_pool"] == 0) & (cols["state_cache"] == 0)
    kinds = kind_lookup[op]
    is_leaf = leaf_lookup[op]
    # --- per-hop own-time quantiles per tier (nominal state, first attempts) ---
    for kind, tier in ((KIND_BFF, "bff"), (KIND_SERVICE, "service"), (KIND_EXTERNAL, "external")):
        sel = nominal & (kinds == kind) & (attempt == 0)
        if sel.sum() < 200:
            continue
        for q, p in zip(QS, QP):
            realised = float(np.quantile(own[sel], p))
            const = c[f"latency_quantiles.{tier}.{q}"]
            items.append({"quantity": f"latency_quantiles.{tier}.{q}", "constant": const, "realised": realised,
                          "n": int(sel.sum()), "tolerance": "10% relative", "pass": _rel_ok(realised, const)})
    # --- error rates per tier (nominal state, first attempts). The constants are the
    # own-class rates the mechanism draws from; a hop's realised rate also carries
    # failures propagated from its callees, so the comparison uses LEAF operations
    # (no callees) and the total rate is reported beside it, not scored.
    for kind, tier in ((KIND_BFF, "bff"), (KIND_SERVICE, "service"), (KIND_EXTERNAL, "external")):
        rates = c.error_rates(tier)
        const = sum(rates.values())
        target = const * mult
        sel_all = nominal & (kinds == kind) & (attempt == 0)
        sel = sel_all & is_leaf
        if sel.sum() >= 200:
            realised = float(np.isin(outcome[sel], [1, 2, 3]).mean())
            items.append({"quantity": f"error_rate.{tier}", "constant": const, "realised": realised, "n": int(sel.sum()),
                          "tolerance": "1 pp absolute or 10% relative", "pass": _rate_ok(realised, target),
                          "note": "leaf operations of the tier (no propagated failures)",
                          "declared_deviation": None if mult == 1.0 else f"error_rate_multiplier={mult:g} (target {target:.4f})"})
        if sel_all.sum() >= 200:
            realised = float(np.isin(outcome[sel_all], [1, 2, 3]).mean())
            items.append({"quantity": f"error_rate_total.{tier}", "constant": const, "realised": realised, "n": int(sel_all.sum()),
                          "tolerance": "reported only", "pass": None,
                          "note": "all hops of the tier including failures propagated from callees; not comparable with an own-class constant"})
    # --- retry count per invocation (backend + external) ---
    sel = (kinds != KIND_BFF)
    invocations = int((attempt[sel] == 0).sum())
    retries = int((attempt[sel] > 0).sum())
    realised = retries / max(invocations, 1)
    # retries follow the retryable-failure rate, which the multiplier scales
    # (non-linearly, through propagation): scored only at the fitted rates
    items.append({"quantity": "retry.mean_retries", "constant": c["retry.mean_retries"], "realised": realised,
                  "n": invocations, "tolerance": "10% relative or 0.01 absolute" if mult == 1.0 else "reported only under an error-rate multiplier",
                  "pass": (abs(realised - c["retry.mean_retries"]) <= max(0.01, 0.1 * c["retry.mean_retries"])) if mult == 1.0 else None,
                  "declared_deviation": None if mult == 1.0 else f"error_rate_multiplier={mult:g}"})
    # --- fan-out and depth from the recorded topology and realised trees ---
    fan = [len(topo.callee_edges(o.id)) for o in topo.ops if o.kind in (KIND_BFF, KIND_SERVICE) and topo.callee_edges(o.id)]
    pmfs = c["fanout_pmf_by_depth"]
    exp_fan = float(np.mean([sum(i * p for i, p in enumerate(row)) / max(1e-9, 1 - row[0]) for row in pmfs]))
    static_fan = float(np.mean(fan)) if fan else 0.0
    # Depth and fan-out are configuration targets by owner decision (the fitted
    # reference system is star-shaped); the fitted value is reported beside.
    # Requests invoke a callee on its edge's share of requests, so the scored
    # fan-out is the invoked one; the topology's edge count is reported beside.
    cfg_fan = float(inst.cfg.topology.fanout_mean)
    invoked_fan, n_calling = _invoked_fanout(shards)
    items.append({"quantity": "fanout_mean", "constant": cfg_fan, "realised": invoked_fan, "n": n_calling,
                  "tolerance": "25% relative (configuration target)", "pass": _rel_ok(invoked_fan, cfg_fan, 0.25),
                  "fitted": exp_fan, "static_fanout": static_fan, "static_n": len(fan),
                  "note": "mean callees invoked by a calling hop (first attempts; hops that invoked at least one callee) vs the instance's fanout_mean; `static_fanout` is the mean callee-edge count per calling endpoint in the topology (sampled fan-out plus reachability attachments); `fitted` is the constants' expectation from fanout_pmf_by_depth (conditional on calling)"})
    req = cols["req_gid"].astype(np.int64)
    uniq, inv = np.unique(req, return_inverse=True)
    req_depth = np.zeros(len(uniq), dtype=np.int64)
    np.maximum.at(req_depth, inv, depth_lookup[op])
    cfg_pmf = list(inst.cfg.topology.depth_pmf)
    realised_pmf, root_only = depth_distribution(req_depth, len(cfg_pmf))
    tvd = 0.5 * sum(abs(a - b) for a, b in zip(realised_pmf, cfg_pmf))
    tvd_fitted = 0.5 * sum(abs(a - b) for a, b in zip(realised_pmf, c["depth_pmf"]))
    cal = topo.calibration
    items.append({"quantity": "depth_pmf", "constant": cfg_pmf, "realised": realised_pmf, "n": int(np.count_nonzero(req_depth >= 1)),
                  "tolerance": "total variation <= 0.10 (configuration target)", "pass": tvd <= 0.10, "tv": tvd,
                  "fitted": list(c["depth_pmf"]), "tv_to_fitted": tvd_fitted, "root_only_share": root_only,
                  "calibration_tv": cal.get("tv"), "calibration_realised_pmf": cal.get("realised_pmf"),
                  "note": "share of requests whose deepest backend-service layer is d = 1..L below the BFF hop (layer 0), among requests that reach layer 1; externals set no depth; `root_only_share` = requests answered without any backend call (cache hits). The fitted constant is on the same axis (service-chain depth below the root, root-only traces excluded). Call probabilities were calibrated to the configured pmf at instantiation by a topology-only Monte Carlo (`calibration_tv`)"})
    # --- vocabulary size ---
    n_ops = sum(1 for o in topo.ops if o.kind != 2)
    items.append({"quantity": "vocab_size", "constant": c["vocab_size"], "realised": n_ops, "n": n_ops,
                  "tolerance": "reported only", "pass": None,
                  "note": "operation count is set by the instance's counts (the ladder spans 20 to 4,331 operations); the fitted reference system has 549"})
    # --- session step gaps (audited responses of a session; a session lies in one shard) ---
    gaps = []
    for sh in shards:
        t = pq.read_table(sh / "linkage.parquet", columns=["kind", "session_gid", "true_ts_ms"])
        kind_col = t.column("kind").to_numpy(zero_copy_only=False)
        g = t.column("session_gid").to_numpy(zero_copy_only=False).astype(np.int64)
        ts = t.column("true_ts_ms").to_numpy(zero_copy_only=False).astype(np.int64)
        sel = (kind_col == "vl.audit") & (g >= 0)
        g, ts = g[sel], ts[sel]
        if len(g) < 2:
            continue
        order = np.lexsort((ts, g))
        g, ts = g[order], ts[order]
        same = (g[1:] == g[:-1]) & (ts[1:] != ts[:-1])
        gaps.append((ts[1:] - ts[:-1])[same] / 1000.0)
    gaps = np.concatenate(gaps) if gaps else np.array([])
    gaps = gaps[gaps > 1.5]
    if len(gaps) > 200:
        for q, p in zip(QS, QP):
            realised = float(np.quantile(gaps, p))
            const = c[f"session.step_gap_quantiles.{q}"]
            items.append({"quantity": f"session.step_gap_quantiles.{q}", "constant": const, "realised": realised, "n": int(len(gaps)),
                          "tolerance": "10% relative", "pass": _rel_ok(realised, const, 0.15),
                          "note": "gaps between consecutive audited responses of a session, excluding retry backoff"})
    scored = [i for i in items if i["pass"] is not None]
    report = {"constants_version": c.version, "n_spans": n, "shards_sampled": len(shards), "items": items,
              "n_pass": sum(1 for i in scored if i["pass"]), "n_items": len(scored), "n_reported_only": len(items) - len(scored),
              "all_pass": all(i["pass"] for i in scored)}
    return report


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", required=True)
    p.add_argument("--max-shards", type=int, default=12, help="evenly spaced oracle shards to sample (0 = all)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    rec = RunRecord(args.corpus, "realism_check", vars(args))
    rep = realism_report(args.corpus, args.max_shards)
    write_json(Path(args.corpus) / REPORTS_DIR / REALISM_REPORT_JSON, rep)
    from .manifest import refresh_manifest
    refresh_manifest(args.corpus)  # the report is a new file of the corpus
    rec.finish({k: v for k, v in rep.items() if k != "items"})
    log({"event": "realism_check", "n_pass": rep["n_pass"], "n_items": rep["n_items"], "constants": rep["constants_version"]})
    return 0 if rep["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
