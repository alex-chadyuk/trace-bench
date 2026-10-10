"""Ranking axis for root-cause methods (docs/scorers-service-rca.md, 2026-10-09).

AC@1, AC@3, AC@5 and Avg@5 of one component ranking per fault case against the shipped labels
(`labels/cases.json`, `root_cause_component` = a service name), with a seeded bootstrap interval
over cases and the analytic expectation of a uniformly random ranking.

    python -m tracebench.score_rca --corpus <dir> --prediction <json> \
        --bootstrap-b 2000 --bootstrap-seed 0 --out <json>
    python -m tracebench.score_rca --pool <result.json> [<result.json> ...] \
        --bootstrap-b 2000 --bootstrap-seed 0 --out <json>

Prediction shape: `{"cases": {"<case_id>": ["<component>", ...], ...}}` — one ranking per case,
best first, component names as the labels spell them. A case without a ranking, or whose root
cause is absent from its ranking, counts as a miss at every k and is listed. AC@k = the share of
cases whose root cause is among the first k entries; Avg@5 = the mean of AC@1 … AC@5. Random
expectation: AC@k = min(k, n) / n for n candidates — the deployed services of the corpus, clients
excluded — and Avg@5 its mean over k. `--pool` concatenates the cases of several results (one
per corpus) into one table with its own bootstrap.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .constants import CASES_JSON, INSTANTIATION_JSON, LABELS_DIR
from .log import log
from .record import read_json, write_json
from .score_service import deployed_services

KS = (1, 3, 5)
AVG_K = 5
LEVEL = 0.95


def load_cases(corpus_dir):
    return read_json(Path(corpus_dir) / LABELS_DIR / CASES_JSON)["cases"]


def case_ranks(cases, prediction):
    """One row per labelled case: the 1-based rank of its root cause in the method's ranking, or
    None when the case has no ranking or the root cause is not in it."""
    rankings = prediction.get("cases", {})
    rows = []
    for c in cases:
        r = rankings.get(c["case_id"])
        rank = None
        if r is not None:
            r = [str(x) for x in r]
            if c["root_cause_component"] in r:
                rank = r.index(c["root_cause_component"]) + 1
        rows.append({"case_id": c["case_id"], "root_cause_component": c["root_cause_component"],
                     "fault_kind": c.get("fault_kind"), "rank": rank,
                     "n_ranked": None if r is None else len(r), "missing": r is None})
    return rows


def _ac(ranks, k):
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def metrics_of(ranks):
    ranks = list(ranks)
    if not ranks:
        return {**{f"ac@{k}": None for k in KS}, f"avg@{AVG_K}": None, "n_cases": 0}
    return {**{f"ac@{k}": _ac(ranks, k) for k in KS},
            f"avg@{AVG_K}": sum(_ac(ranks, k) for k in range(1, AVG_K + 1)) / AVG_K, "n_cases": len(ranks)}


def bootstrap(ranks, b, seed):
    """Percentile intervals (LEVEL) of every metric over `b` resamples of the cases with
    replacement, drawn from `numpy.random.default_rng(seed)`."""
    ranks = list(ranks)
    n = len(ranks)
    if n == 0 or b <= 0:
        return {"b": int(b), "seed": int(seed), "level": LEVEL, "intervals": {}}
    arr = np.array([-1 if r is None else int(r) for r in ranks], dtype=np.int64)
    samp = arr[np.random.default_rng(seed).integers(0, n, size=(int(b), n))]
    hits = {k: ((samp > 0) & (samp <= k)).mean(axis=1) for k in range(1, AVG_K + 1)}
    lo, hi = (1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2

    def ci(v):
        return {"low": float(np.quantile(v, lo)), "high": float(np.quantile(v, hi))}

    intervals = {f"ac@{k}": ci(hits[k]) for k in KS}
    intervals[f"avg@{AVG_K}"] = ci(np.mean([hits[k] for k in range(1, AVG_K + 1)], axis=0))
    return {"b": int(b), "seed": int(seed), "level": LEVEL, "intervals": intervals}


def random_expectation(n_candidates):
    """A uniformly random ranking of `n_candidates` components places the root cause in the first k
    with probability min(k, n) / n."""
    if not n_candidates:
        return None
    n = int(n_candidates)
    ac = {k: min(k, n) / n for k in range(1, AVG_K + 1)}
    return {**{f"ac@{k}": ac[k] for k in KS}, f"avg@{AVG_K}": sum(ac.values()) / AVG_K, "n_candidates": n}


def by_fault_kind(rows):
    kinds = sorted({r["fault_kind"] for r in rows if r["fault_kind"] is not None})
    return {kind: metrics_of([r["rank"] for r in rows if r["fault_kind"] == kind]) for kind in kinds}


def score_rca(corpus_dir, prediction, b, seed):
    corpus_dir = Path(corpus_dir)
    rows = case_ranks(load_cases(corpus_dir), prediction)
    ranks = [r["rank"] for r in rows]
    n_cand = len(deployed_services(corpus_dir)) if (corpus_dir / INSTANTIATION_JSON).exists() else None
    return {"axis": "ranking", "corpus_dir_name": corpus_dir.name, "metrics": metrics_of(ranks),
            "bootstrap": bootstrap(ranks, b, seed), "random": random_expectation(n_cand),
            "n_missing": sum(1 for r in rows if r["missing"]), "by_fault_kind": by_fault_kind(rows), "cases": rows}


def pool(results, b, seed):
    """One table over the cases of several per-corpus results (the same metrics and bootstrap)."""
    rows = [r for res in results for r in res["cases"]]
    ranks = [r["rank"] for r in rows]
    cands = {(res.get("random") or {}).get("n_candidates") for res in results}
    return {"axis": "ranking", "pooled_from": [res.get("corpus_dir_name") for res in results],
            "n_cases_per_result": [len(res["cases"]) for res in results], "metrics": metrics_of(ranks),
            "bootstrap": bootstrap(ranks, b, seed),
            "random": random_expectation(cands.pop()) if len(cands) == 1 else None,
            "random_per_result": [res.get("random") for res in results],
            "n_missing": sum(1 for r in rows if r["missing"]), "by_fault_kind": by_fault_kind(rows)}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", help="a corpus with labels/ (and instantiation.json for the random expectation)")
    p.add_argument("--prediction", help="the per-case rankings JSON")
    p.add_argument("--pool", nargs="+", help="per-corpus result files to pool instead of scoring one corpus")
    p.add_argument("--bootstrap-b", required=True, type=int, help="bootstrap resamples (0 = none)")
    p.add_argument("--bootstrap-seed", required=True, type=int)
    p.add_argument("--out", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.pool:
        if args.corpus or args.prediction:
            raise SystemExit("--pool takes result files only; --corpus / --prediction score one corpus")
        result = pool([read_json(p) for p in args.pool], args.bootstrap_b, args.bootstrap_seed)
    else:
        if not (args.corpus and args.prediction):
            raise SystemExit("score one corpus with --corpus and --prediction, or pool results with --pool")
        prediction = json.loads(Path(args.prediction).read_text(encoding="utf-8"))
        result = score_rca(args.corpus, prediction, args.bootstrap_b, args.bootstrap_seed)
    write_json(args.out, result)
    log({"event": "score_rca", **{k: result["metrics"][k] for k in result["metrics"]}, "n_missing": result["n_missing"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
