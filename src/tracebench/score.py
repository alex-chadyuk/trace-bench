"""Scoring a predicted graph against the scoring target.

Structural axes on the mixed token graph (PRD success bar): precision /
recall / F1 and SHD on directed edges (SHD = fp + fn on an ordered-pair
universe), skeleton F1 / SHD, edge-orientation accuracy (all directed true
positives, and the compelled subset via the CPDAG), AUROC and average
precision from edge scores (skeleton, directed, bidirected), the bidirected
class as its own precision / recall / F1, a mixed-graph SHD (one per unordered
pair whose edge state differs), and the trivial baselines SHD is charged
against (empty graph, top-k by score). The causal-validity axis (SID, parent-
and ancestor-AID) is computed on the observable twin's DAG with `gadjid` and
reported not-applicable-with-reason elsewhere.

Universe. Only ordered token pairs whose operations can co-occur (the target's
`support_op_pairs`) are scoreable; predictions outside it are counted, never
scored. Within-operation token pairs are never scored.

Prediction file:
    {"directed": [{"src": "<op>:<outcome>", "dst": "...", "score": 0.9}, ...],
     "bidirected": [{"a": "...", "b": "...", "score": 0.4}, ...]}
An edge listed is predicted present; `score` (default 1.0) ranks edges for the
threshold-free axes. Absent edges score 0.

`self_check` feeds the target back as a prediction and asserts every axis is
perfect (PRD scenario 4).
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np

from .constants import ALPHABET_JSON, GRAPHS_DIR, SCORING_TARGET_JSON, VARIANT_TWIN
from .cpdag import cpdag_from_dag
from .log import log
from .projection import is_acyclic
from .record import RunRecord, read_json, write_json

SCORING_TARGET_SESSION_JSON = "scoring-target-session.json"


# --- universe --------------------------------------------------------------------------
def _op_of(tok):
    return None if tok.startswith("state:") else int(tok.split(":")[0])


def build_universe(alphabet, target):
    """Ordered token pairs (a, b) with a's op != b's op and the op pair in
    support; state tokens (twin) pair with every token they reach in the target."""
    tokens = [t["token"] for t in alphabet["tokens"]]
    by_op = {}
    state_tokens = []
    for t in tokens:
        op = _op_of(t)
        if op is None:
            state_tokens.append(t)
        else:
            by_op.setdefault(op, []).append(t)
    support = {tuple(p) for p in target.get("support_op_pairs", [])}
    ordered = []
    for a, b in sorted(support):
        for ta in by_op.get(a, []):
            for tb in by_op.get(b, []):
                ordered.append((ta, tb))
                ordered.append((tb, ta))
    if state_tokens:
        reached = {}
        for e in target["directed"]:
            if e["src"].startswith("state:"):
                reached.setdefault(e["src"].split("=")[0], set()).add(_op_of(e["dst"]))
        for st in state_tokens:
            var = st.split("=")[0]
            for op in sorted(reached.get(var, ())):
                for tb in by_op.get(op, []):
                    ordered.append((st, tb))
    ordered = sorted(set(ordered))
    unordered = sorted({(a, b) if a < b else (b, a) for a, b in ordered})
    return ordered, unordered


# --- helpers -----------------------------------------------------------------------------------
def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1, "shd": int(fp + fn), "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def auroc(y, s):
    y = np.asarray(y, dtype=bool)
    s = np.asarray(s, dtype=np.float64)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y, s):
    y = np.asarray(y, dtype=bool)
    s = np.asarray(s, dtype=np.float64)
    if y.sum() == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    return float((prec * ys).sum() / ys.sum())


# --- scoring ---------------------------------------------------------------------------------------
def truth_sets(target, floor):
    d = {(e["src"], e["dst"]) for e in target["directed"] if e["strength"] >= floor}
    b = {(e["a"], e["b"]) for e in target["bidirected"] if e["strength"] >= floor}
    return d, b


def prediction_sets(prediction):
    d = {}
    for e in prediction.get("directed", []):
        d[(e["src"], e["dst"])] = float(e.get("score", 1.0))
    b = {}
    for e in prediction.get("bidirected", []):
        key = (e["a"], e["b"]) if e["a"] < e["b"] else (e["b"], e["a"])
        b[key] = float(e.get("score", 1.0))
    return d, b


def _pair_state(a, b, directed, bidirected):
    s = set()
    if (a, b) in directed:
        s.add("->")
    if (b, a) in directed:
        s.add("<-")
    if (a, b) in bidirected:
        s.add("<->")
    return frozenset(s)


def score_at_floor(target, alphabet, prediction, floor, ordered=None, unordered=None):
    if ordered is None:
        ordered, unordered = build_universe(alphabet, target)
    uo = set(ordered)
    uu = set(unordered)
    td, tb = truth_sets(target, floor)
    pd_all, pb_all = prediction_sets(prediction)
    outside = sum(1 for k in pd_all if k not in uo) + sum(1 for k in pb_all if k not in uu)
    pd = {k: v for k, v in pd_all.items() if k in uo}
    pb = {k: v for k, v in pb_all.items() if k in uu}
    # directed
    tp = len(td & set(pd)); fp = len(set(pd) - td); fn = len(td - set(pd))
    directed = prf(tp, fp, fn)
    # skeleton
    t_skel = {(a, b) if a < b else (b, a) for a, b in td} | set(tb)
    p_skel = {(a, b) if a < b else (b, a) for a, b in pd} | set(pb)
    skeleton = prf(len(t_skel & p_skel), len(p_skel - t_skel), len(t_skel - p_skel))
    # orientation on skeleton true positives where the truth has one direction and no bidirected edge
    orient_n = orient_ok = 0
    truth_dir_edges = sorted(td)
    acyclic = is_acyclic(truth_dir_edges)
    compelled, _ = cpdag_from_dag(truth_dir_edges) if acyclic else (set(), set())
    comp_n = comp_ok = 0
    for a, b in sorted(t_skel & p_skel):
        fwd, rev = (a, b) in td, (b, a) in td
        if fwd == rev or (a, b) in tb:
            continue
        s, t = (a, b) if fwd else (b, a)
        ok = (s, t) in pd and (t, s) not in pd
        orient_n += 1
        orient_ok += ok
        if (s, t) in compelled:
            comp_n += 1
            comp_ok += ok
    # bidirected class
    bidirected = prf(len(tb & set(pb)), len(set(pb) - tb), len(tb - set(pb)))
    # mixed SHD: one per unordered pair whose state differs
    shd_mixed = 0
    for a, b in unordered:
        if _pair_state(a, b, td, tb) != _pair_state(a, b, set(pd), set(pb)):
            shd_mixed += 1
    # threshold-free axes
    y_d = [(a, b) in td for a, b in ordered]
    s_d = [pd.get((a, b), 0.0) for a, b in ordered]
    y_s = [(a, b) in t_skel for a, b in unordered]
    s_s = [max(pd.get((a, b), 0.0), pd.get((b, a), 0.0), pb.get((a, b), 0.0)) for a, b in unordered]
    y_b = [(a, b) in tb for a, b in unordered]
    s_b = [pb.get((a, b), 0.0) for a, b in unordered]
    # baselines
    k = len(td)
    topk = sorted(pd.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    topk_set = {e for e, _ in topk}
    shd_topk = len(td - topk_set) + len(topk_set - td)
    return {
        "floor": floor,
        "universe": {"ordered_pairs": len(ordered), "unordered_pairs": len(unordered),
                     "truth_directed": len(td), "truth_bidirected": len(tb), "predictions_outside_universe": outside},
        "directed": directed,
        "skeleton": skeleton,
        "orientation": {"accuracy": (orient_ok / orient_n) if orient_n else None, "n": orient_n,
                        "accuracy_compelled": (comp_ok / comp_n) if comp_n else None, "n_compelled": comp_n,
                        "truth_directed_acyclic": acyclic},
        "bidirected": bidirected,
        "shd_mixed": shd_mixed,
        "auroc": {"directed": auroc(y_d, s_d), "skeleton": auroc(y_s, s_s), "bidirected": auroc(y_b, s_b)},
        "average_precision": {"directed": average_precision(y_d, s_d), "skeleton": average_precision(y_s, s_s),
                              "bidirected": average_precision(y_b, s_b)},
        "baselines": {"shd_empty_directed": len(td), "shd_empty_skeleton": len(t_skel),
                      "shd_empty_bidirected": len(tb), "shd_topk_directed": shd_topk},
    }


def causal_validity(target, prediction, floor, variant):
    """SID / parent-AID / ancestor-AID on the twin's DAG over tokens (gadjid).
    Not applicable, with the reason stated, on the latent-bearing instance or
    when either directed graph is cyclic."""
    if variant != VARIANT_TWIN:
        return {"value": None, "reason": "target is an ADMG with bidirected edges; SID/AID are defined over DAGs; see the observable twin"}
    td, _ = truth_sets(target, floor)
    pd, _ = prediction_sets(prediction)
    if not is_acyclic(sorted(td)):
        return {"value": None, "reason": "the twin's directed target is cyclic at this floor"}
    if not is_acyclic(sorted(pd)):
        return {"value": None, "reason": "the predicted directed graph is cyclic"}
    try:
        import gadjid
    except ImportError:
        return {"value": None, "reason": "gadjid not installed (pip install 'trace-bench[sid]')"}
    nodes = sorted({t for e in td for t in e} | {t for e in pd for t in e})
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    if n == 0:
        return {"value": {"sid": 0.0, "parent_aid": 0.0, "ancestor_aid": 0.0, "n_nodes": 0}, "reason": None}
    T = np.zeros((n, n), dtype=np.int8)
    G = np.zeros((n, n), dtype=np.int8)
    for a, b in td:
        T[idx[a], idx[b]] = 1
    for a, b in pd:
        G[idx[a], idx[b]] = 1
    kw = {"edge_direction": "from row to column"}
    sid = gadjid.sid(T, G, **kw)
    pa = gadjid.parent_aid(T, G, **kw)
    an = gadjid.ancestor_aid(T, G, **kw)
    return {"value": {"sid": float(sid[0]), "sid_mistakes": int(sid[1]), "parent_aid": float(pa[0]), "parent_aid_mistakes": int(pa[1]),
                      "ancestor_aid": float(an[0]), "ancestor_aid_mistakes": int(an[1]), "n_nodes": n}, "reason": None}


def score_corpus(corpus_dir, prediction, floor=None, grain="request", variant=None):
    corpus_dir = Path(corpus_dir)
    gdir = corpus_dir / GRAPHS_DIR
    target = read_json(gdir / (SCORING_TARGET_JSON if grain == "request" else SCORING_TARGET_SESSION_JSON))
    alphabet = read_json(gdir / ALPHABET_JSON)
    variant = variant or target.get("variant", "latent")
    floor = target["default_floor"] if floor is None else float(floor)
    ordered, unordered = build_universe(alphabet, target)
    result = score_at_floor(target, alphabet, prediction, floor, ordered, unordered)
    result["causal_validity"] = causal_validity(target, prediction, floor, variant)
    sweep = read_json(gdir / "floor-sensitivity.json")["sweep"] if (gdir / "floor-sensitivity.json").exists() else []
    result["sweep"] = [
        {"floor": r["floor"], **{k: v for k, v in score_at_floor(target, alphabet, prediction, r["floor"], ordered, unordered).items()
                                  if k in ("directed", "skeleton", "bidirected", "shd_mixed", "orientation")}}
        for r in sweep
    ]
    result["grain"] = grain
    result["variant"] = variant
    return result


def target_as_prediction(target, floor=None):
    floor = target["default_floor"] if floor is None else floor
    return {"directed": [{"src": e["src"], "dst": e["dst"], "score": e["strength"]} for e in target["directed"] if e["strength"] >= floor],
            "bidirected": [{"a": e["a"], "b": e["b"], "score": e["strength"]} for e in target["bidirected"] if e["strength"] >= floor]}


def self_check(corpus_dir, grain="request"):
    """Score the target against itself: every structural axis must be perfect."""
    corpus_dir = Path(corpus_dir)
    gdir = corpus_dir / GRAPHS_DIR
    target = read_json(gdir / (SCORING_TARGET_JSON if grain == "request" else SCORING_TARGET_SESSION_JSON))
    res = score_corpus(corpus_dir, target_as_prediction(target), grain=grain)
    problems = []
    for axis in ("directed", "skeleton", "bidirected"):
        m = res[axis]
        if m["tp"] + m["fn"] > 0 and (m["f1"] != 1.0 or m["shd"] != 0):
            problems.append(f"{axis}: f1={m['f1']} shd={m['shd']}")
    if res["orientation"]["n"] and res["orientation"]["accuracy"] != 1.0:
        problems.append(f"orientation: {res['orientation']}")
    for axis, v in res["auroc"].items():
        if v is not None and v < 1.0 - 1e-12:
            problems.append(f"auroc {axis}: {v}")
    if res["shd_mixed"] != 0:
        problems.append(f"shd_mixed: {res['shd_mixed']}")
    cv = res["causal_validity"]["value"]
    if cv and (cv["sid"] != 0.0 or cv["parent_aid"] != 0.0):
        problems.append(f"causal validity: {cv}")
    res["self_check_passed"] = not problems
    res["self_check_problems"] = problems
    return res


def headline(per_seed_results, dotted_key):
    """Central tendency and dispersion of one metric over the seeds of a named
    instance (PRD scenario 27): a headline number is never a single seed."""
    vals = []
    for r in per_seed_results:
        cur = r
        for k in dotted_key.split("."):
            cur = cur[k]
        vals.append(float(cur))
    a = np.asarray(vals, dtype=np.float64)
    if len(a) < 5:
        raise ValueError("a headline needs at least five seeds")
    return {"metric": dotted_key, "n_seeds": int(len(a)), "mean": float(a.mean()), "std": float(a.std(ddof=1)),
            "p10": float(np.quantile(a, 0.1)), "p50": float(np.quantile(a, 0.5)), "p90": float(np.quantile(a, 0.9)),
            "values": vals}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", required=True)
    p.add_argument("--prediction", required=True, help="prediction JSON, or 'self' to score the target against itself")
    p.add_argument("--grain", required=True, choices=["request", "session"])
    p.add_argument("--floor", type=float, default=None, help="strength floor (default: the target's default floor)")
    p.add_argument("--out", required=True, help="where to write the score report")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.prediction == "self":
        res = self_check(args.corpus, grain=args.grain)
    else:
        res = score_corpus(args.corpus, read_json(args.prediction), floor=args.floor, grain=args.grain)
    write_json(args.out, res)
    log({"event": "score", "grain": args.grain, "f1_directed": res["directed"]["f1"], "shd_mixed": res["shd_mixed"],
         "self_check": res.get("self_check_passed")})
    return 0 if res.get("self_check_passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
