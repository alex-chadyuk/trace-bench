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

Cost. A `ScoreContext` holds what depends on the target alone (the sorted
universe, and per floor the truth sets, their universe positions and the
CPDAG); with it a call costs O(truth + predicted edges) plus three array sums
over the universe, whatever the universe's size. `_score_at_floor_dense` is
the original universe-walking implementation, kept as the reference the sparse
path must equal bit for bit and as the fallback for non-finite scores.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
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


class ScoreContext:
    """What a score needs from the target alone, computed once and shared across calls
    (every τ of a sweep, every floor of the sensitivity sweep, every cell of a read)."""

    def __init__(self, target, alphabet, ordered=None, unordered=None):
        if ordered is None:
            ordered, unordered = build_universe(alphabet, target)
        self.target = target
        self.alphabet = alphabet
        self.ordered = ordered
        self.unordered = unordered
        # positions are found by bisection, which needs the order `build_universe` returns
        self.sparse = _strictly_increasing(ordered) and _strictly_increasing(unordered)
        self._truth = {}

    def truth(self, floor):
        t = self._truth.get(floor)
        if t is None:
            td, tb = truth_sets(self.target, floor)
            t_skel = {(a, b) if a < b else (b, a) for a, b in td} | set(tb)
            truth_dir_edges = sorted(td)
            acyclic = is_acyclic(truth_dir_edges)
            compelled, _ = cpdag_from_dag(truth_dir_edges) if acyclic else (set(), set())
            t = {"td": td, "tb": tb, "t_skel": t_skel, "acyclic": acyclic, "compelled": compelled,
                 "idx_d": _positions(self.ordered, td), "idx_s": _positions(self.unordered, t_skel),
                 "idx_b": _positions(self.unordered, tb),
                 "skel_in_universe": {k for k in t_skel if _position(self.unordered, k) >= 0}}
            self._truth[floor] = t
        return t


def _strictly_increasing(pairs):
    return all(a < b for a, b in zip(pairs, pairs[1:]))


def _position(sorted_pairs, key):
    """Index of `key` in a sorted pair list, or -1 (tokens are strings, so anything else is outside)."""
    if not (isinstance(key[0], str) and isinstance(key[1], str)):
        return -1
    i = bisect_left(sorted_pairs, key)
    return i if i < len(sorted_pairs) and sorted_pairs[i] == key else -1


def _positions(sorted_pairs, keys):
    """Sorted universe positions of the keys that are in the universe."""
    idx = [i for i in (_position(sorted_pairs, k) for k in keys) if i >= 0]
    return np.array(sorted(idx), dtype=np.int64)


def _explicit(scored):
    """`{position: score}` -> (positions ascending, scores), without the zeros: a zero score ties with
    every absent pair and sorts among them by position, so it is an absent pair."""
    items = sorted((i, v) for i, v in scored.items() if v != 0.0)
    return (np.array([i for i, _ in items], dtype=np.int64), np.array([v for _, v in items], dtype=np.float64))


def _lookup(e_idx, pos_idx):
    """For each truth position: its slot in the explicit positions (or the count of explicit
    positions below it), and whether it is there."""
    loc = np.searchsorted(e_idx, pos_idx)
    hit = np.zeros(len(pos_idx), dtype=bool)
    inside = loc < len(e_idx)
    hit[inside] = e_idx[loc[inside]] == pos_idx[inside]
    return loc, hit


def _auroc_sparse(n, pos_idx, e_idx, e_val):
    """`auroc` over a length-n score vector that is zero except at `e_idx`, with the positives at
    `pos_idx`. Rank sums are half-integers far below 2**53, so the arithmetic is exact."""
    n_pos, n_neg = int(len(pos_idx)), int(n - len(pos_idx))
    if n_pos == 0 or n_neg == 0:
        return None
    m = len(e_idx)
    z = n - m                                            # the block of pairs tied at zero
    order = np.argsort(e_val, kind="mergesort")
    ev = e_val[order]
    m_neg = int((ev < 0).sum())
    at = np.arange(m, dtype=np.int64) + np.where(ev > 0, z, 0)   # place in the ascending order
    ranks_e = np.empty(m, dtype=np.float64)
    if m:
        first = np.flatnonzero(np.r_[True, ev[1:] != ev[:-1]])
        counts = np.diff(np.r_[first, m])
        i = at[first]
        ranks_e[order] = np.repeat((i + (i + counts - 1)) / 2.0 + 1.0, counts)
    zero_rank = (m_neg + (m_neg + z - 1)) / 2.0 + 1.0
    loc, hit = _lookup(e_idx, pos_idx)
    rank_sum = np.float64(ranks_e[loc[hit]].sum() + int((~hit).sum()) * zero_rank)
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _average_precision_sparse(n, pos_idx, e_idx, e_val):
    """`average_precision` over the same sparse score vector. The stable descending order is the
    positive scores, then the zero block in universe order, then the negative scores; the terms
    are summed from a length-n array so the summation order is the dense one's."""
    n_pos = int(len(pos_idx))
    if n_pos == 0:
        return None
    m = len(e_idx)
    z = n - m
    rank_e = np.empty(m, dtype=np.int64)
    rank_e[np.lexsort((e_idx, -e_val))] = np.arange(m, dtype=np.int64)
    at_e = rank_e + np.where(e_val > 0, 0, z)
    m_pos = int((e_val > 0).sum())
    loc, hit = _lookup(e_idx, pos_idx)
    at = np.empty(n_pos, dtype=np.int64)
    at[hit] = at_e[loc[hit]]
    at[~hit] = m_pos + (pos_idx[~hit] - loc[~hit])
    at.sort()
    terms = np.zeros(n, dtype=np.float64)
    terms[at] = np.arange(1, n_pos + 1, dtype=np.int64) / (at + 1)
    return float(terms.sum() / np.int64(n_pos))


def score_at_floor(target, alphabet, prediction, floor, ordered=None, unordered=None, context=None):
    if context is None:
        context = ScoreContext(target, alphabet, ordered, unordered)
    elif context.target is not target:
        raise ValueError("the score context was built from a different target")
    ordered, unordered = context.ordered, context.unordered
    pd_all, pb_all = prediction_sets(prediction)
    finite = all(np.isfinite(v) for v in pd_all.values()) and all(np.isfinite(v) for v in pb_all.values())
    if not (context.sparse and finite):
        return _score_at_floor_dense(target, alphabet, prediction, floor, ordered, unordered)
    truth = context.truth(floor)
    td, tb, t_skel = truth["td"], truth["tb"], truth["t_skel"]
    at_d = {k: _position(ordered, k) for k in pd_all}
    at_b = {k: _position(unordered, k) for k in pb_all}
    outside = sum(1 for i in at_d.values() if i < 0) + sum(1 for i in at_b.values() if i < 0)
    pd = {k: v for k, v in pd_all.items() if at_d[k] >= 0}
    pb = {k: v for k, v in pb_all.items() if at_b[k] >= 0}
    spd, spb = set(pd), set(pb)
    # directed
    directed = prf(len(td & spd), len(spd - td), len(td - spd))
    # skeleton
    p_skel = {(a, b) if a < b else (b, a) for a, b in pd} | spb
    skeleton = prf(len(t_skel & p_skel), len(p_skel - t_skel), len(t_skel - p_skel))
    # orientation on skeleton true positives where the truth has one direction and no bidirected edge
    orient_n = orient_ok = 0
    acyclic, compelled = truth["acyclic"], truth["compelled"]
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
    bidirected = prf(len(tb & spb), len(spb - tb), len(tb - spb))
    # mixed SHD: one per unordered pair whose state differs; a pair in neither skeleton is empty on both sides
    shd_mixed = 0
    for a, b in truth["skel_in_universe"] | p_skel:      # the predicted skeleton is inside the universe already
        if _pair_state(a, b, td, tb) != _pair_state(a, b, spd, spb):
            shd_mixed += 1
    # threshold-free axes
    e_d = _explicit({at_d[k]: v for k, v in pd.items()})
    e_s = _explicit({_position(unordered, (a, b)): max(pd.get((a, b), 0.0), pd.get((b, a), 0.0), pb.get((a, b), 0.0))
                     for a, b in p_skel})
    e_b = _explicit({at_b[k]: v for k, v in pb.items()})
    n_o, n_u = len(ordered), len(unordered)
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
        "auroc": {"directed": _auroc_sparse(n_o, truth["idx_d"], *e_d), "skeleton": _auroc_sparse(n_u, truth["idx_s"], *e_s),
                  "bidirected": _auroc_sparse(n_u, truth["idx_b"], *e_b)},
        "average_precision": {"directed": _average_precision_sparse(n_o, truth["idx_d"], *e_d),
                              "skeleton": _average_precision_sparse(n_u, truth["idx_s"], *e_s),
                              "bidirected": _average_precision_sparse(n_u, truth["idx_b"], *e_b)},
        "baselines": {"shd_empty_directed": len(td), "shd_empty_skeleton": len(t_skel),
                      "shd_empty_bidirected": len(tb), "shd_topk_directed": shd_topk},
    }


def _score_at_floor_dense(target, alphabet, prediction, floor, ordered=None, unordered=None):
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
    spd, spb = set(pd), set(pb)  # built once: rebuilding them per pair made this loop quadratic
    for a, b in unordered:
        if _pair_state(a, b, td, tb) != _pair_state(a, b, spd, spb):
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


def load_context(corpus_dir, grain="request"):
    """The score context of a corpus's target at a grain."""
    gdir = Path(corpus_dir) / GRAPHS_DIR
    target = read_json(gdir / (SCORING_TARGET_JSON if grain == "request" else SCORING_TARGET_SESSION_JSON))
    return ScoreContext(target, read_json(gdir / ALPHABET_JSON))


def score_corpus(corpus_dir, prediction, floor=None, grain="request", variant=None, context=None):
    """`context` (from `load_context` at the same grain) spares re-reading the target and
    rebuilding the universe when several predictions are scored against one corpus."""
    corpus_dir = Path(corpus_dir)
    gdir = corpus_dir / GRAPHS_DIR
    if context is None:
        context = load_context(corpus_dir, grain)
    target, alphabet = context.target, context.alphabet
    variant = variant or target.get("variant", "latent")
    floor = target["default_floor"] if floor is None else float(floor)
    result = score_at_floor(target, alphabet, prediction, floor, context=context)
    result["causal_validity"] = causal_validity(target, prediction, floor, variant)
    sweep = read_json(gdir / "floor-sensitivity.json")["sweep"] if (gdir / "floor-sensitivity.json").exists() else []
    result["sweep"] = [
        {"floor": r["floor"], **{k: v for k, v in score_at_floor(target, alphabet, prediction, r["floor"], context=context).items()
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
