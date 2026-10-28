"""Per-trace root-cause axis for root-cause methods (docs/scorers-service-rca.md §4, 2026-10-31).

The root-cause question asked per failed request trace rather than per fault case: hit@1 and hit@3
of the root-cause service among a trace's own non-client services, of a root-cause position among
its positions, and the exact set (the method's reached-service set equals {root-cause service}),
with a seeded bootstrap over cases and the analytic expectation of a uniformly random ranking.

    python -m tracebench.score_rca_trace --corpus <dir> --cell <ranking-trace.json | cell.json> \
        --mode {arm,topology,random,callees,broadcast} --hop0 {0,1} [--reading {hop1,hop0}] \
        [--prediction <json>] --bootstrap-b 2000 --bootstrap-seed 0 --out <json>
    python -m tracebench.score_rca_trace --pool <result.json> [<result.json> ...] \
        --bootstrap-b 2000 --bootstrap-seed 0 --out <json>

Selection (rule R, label-free, identical on the method side): per case every request trace of the
end-ordered view, all splits, wholly inside the fault window (`inject_ns <= start_ns` and
`start_ns + max(offsets + durations) < end_ns`, integer ns) with at least one failed non-client
span (`outcomes` in {5xx, err, slow} — 4xx is not a failure on this axis), ordered by
`(start_ns, trace_id)`; `cap` 0 = all. The method side's `selected` list must equal the list
recomputed here (refused otherwise) and `skipped` must be a subset of it. The scored set C of a
case = the selected traces in which the fault took: a root-cause position (`rc_positions_end` of
`labels/sequence-labels.parquet`) with an outcome in the same non-OK set, recomputed from the view
row; the label's `rc_took` also counts 4xx and is only asserted (took ⇒ rc_took).

Truth per trace: the case's `root_cause_component` and its `rc_positions_end` (indices into the
row's `ops`). An arm reads window positions `p = i - c + 1`, reachable for
`c - 1 <= i < min(len(ops), max_len)`; a root-cause position in the context or past the length cap
is counted (`n_rc_unreachable_context`, `n_rc_unreachable_truncation`) and is a miss. The
services order everywhere is the name order of the deployed services (clients excluded), the
export's `services.json` order.

Modes: `arm` (the harness's `ranking-trace.json`: per probed trace the ranked reached services as
indices into that order and the ranked window positions; a selected trace the method skipped or
did not report is a miss, `n_unprobed`), `topology` (the rule on the true call tree: for every
failed non-client span its `parent_pos` descendants, ranked by the number of failed spans whose
subtree holds the item, then the smallest hop, then the services order; positions are ops
indices), `random` (analytic over the trace's services and candidate positions), `callees`
(analytic over the failed spans' descendant services) and `broadcast` (a case-level ranking in
`score_rca`'s shape, or a list of per-run rankings averaged per trace, filtered to the trace's
services with the order kept and the absent services appended in services order). `hop0` adds a
failed span itself at hop 0 (`topology`, `callees`; `arm` reads the method's hop-0 lists).

Per case the metrics are means over C; the reported point is the mean over cases, with a
percentile bootstrap over cases (`numpy.random.default_rng(seed).integers(0, n, (b, n))`, 95 %);
beside it the trace-weighted means, per fault kind and per split (train / val+test). `--pool`
concatenates the case rows of several results (one per corpus) into one table with its own
bootstrap. The per-trace rows are written with the result (`traces`); the caller may strip them.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

from .constants import (CASES_JSON, CONFIG_YAML, INSTANTIATION_JSON, KIND_CLIENT, LABELS_DIR, OUTCOME_5XX,
                        OUTCOME_ERR, OUTCOME_SLOW)
from .export_rca import base_ops, case_dirs, deployed_services, view_inputs, window_epoch
from .log import log
from .record import read_json, write_json

KS = (1, 3)
LEVEL = 0.95
AXIS = "trace"
NONOK_IDS = (OUTCOME_5XX, OUTCOME_ERR, OUTCOME_SLOW)
SEQUENCE_LABELS_PARQUET = "sequence-labels.parquet"
SPANS_VIEW = "end-request"
MODES = ("arm", "topology", "random", "callees", "broadcast")
MODE_ARM, MODE_TOPOLOGY, MODE_RANDOM, MODE_CALLEES, MODE_BROADCAST = MODES
SELECT_WINDOW, SELECT_POST_NONOK = "window", "post-nonok"
SELECTS = (SELECT_WINDOW, SELECT_POST_NONOK)
READINGS = {"hop1": ("s", "p", 0), "hop0": ("s0", "p0", 1)}  # reading -> (services key, positions key, hop0)
GRAIN_REQUEST = "request"
SPLIT_TRAIN = "train"
NS_PER_S = 10**9
ROW_COLUMNS = ("trace_id", "ops", "outcomes", "offsets", "durations", "parent_pos", "start_ns")
BATCH_ROWS = 20_000
HIT_KEYS = ("s@1", "s@3", "p@1", "p@3", "exact")
RANDOM_KEYS = ("s@1", "s@3", "p@1", "p@3")
BOOTSTRAP_KEYS = HIT_KEYS


# --- corpus inputs -----------------------------------------------------------------------------

def load_labels(corpus_dir):
    """`(trace_id, fault_index) -> {phase, rc_positions_end, rc_took}` of the request-grain label rows."""
    path = Path(corpus_dir) / LABELS_DIR / SEQUENCE_LABELS_PARQUET
    out = {}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_ROWS):
        for r in batch.to_pylist():
            if r["grain"] != GRAIN_REQUEST:
                continue
            key = (str(r["trace_id"]), int(r["fault_index"]))
            if key in out:
                raise ValueError(f"duplicate label row for trace {key[0]} fault {key[1]}")
            out[key] = {"phase": int(r["phase"]), "rc_positions_end": [int(i) for i in r["rc_positions_end"]],
                        "rc_took": bool(r["rc_took"])}
    return out


def load_ops(corpus_dir):
    """`(op id -> (service, name), client op ids)` from the end-ordered request view's vocabulary."""
    vocab_path, _ = view_inputs(corpus_dir, SPANS_VIEW)
    vocab = read_json(vocab_path)
    client = {int(o["id"]) for o in vocab["base_ops"] if int(o["kind"]) == KIND_CLIENT}
    return base_ops(vocab), client


def services_sorted(corpus_dir):
    """The deployed services (clients excluded) in name order — the export's `services.json` order."""
    return [name for name, _ in deployed_services(read_json(Path(corpus_dir) / INSTANTIATION_JSON))]


def fault_index_of(case):
    """The label's `fault_index` of a case: the trailing integer of its `case_id`
    (`<system>_<service>_<kind>_<index>`, the only field of `cases.json` that carries it)."""
    try:
        return int(str(case["case_id"]).rsplit("_", 1)[1])
    except (IndexError, ValueError):
        raise ValueError(f"case_id {case['case_id']!r} does not end in the fault index") from None


def case_windows_ns(corpus_dir):
    """`[(case_dir, case, inject_ns, end_ns)]` in `case_dirs` order; the window start is the corpus
    clock of `config.yaml`, the times `labels/cases.json`'s integral seconds."""
    corpus_dir = Path(corpus_dir)
    with open(corpus_dir / CONFIG_YAML, encoding="utf-8") as f:
        epoch, _ = window_epoch(yaml.safe_load(f))
    out = []
    for case_dir, case in case_dirs(read_json(corpus_dir / LABELS_DIR / CASES_JSON)["cases"]):
        bounds = []
        for key in ("inject_time_s", "end_time_s"):
            v = float(case[key])
            if not v.is_integer():
                raise ValueError(f"{case['case_id']}: {key} = {v!r} is not an integral number of seconds")
            bounds.append((epoch + int(v)) * NS_PER_S)
        out.append((case_dir, case, bounds[0], bounds[1]))
    return out


# --- rule R -----------------------------------------------------------------------------------

def trace_end_ns(row):
    return int(row["start_ns"]) + max(int(o) + int(d) for o, d in zip(row["offsets"], row["durations"]))


def is_post_trace(row, lo_ns, hi_ns):
    """The trace lies wholly inside `[lo_ns, hi_ns)`: rule R without its non-OK clause (= phase POST)."""
    return lo_ns <= int(row["start_ns"]) and trace_end_ns(row) < hi_ns


def nonok_positions(row, client_ops):
    """Ops indices whose outcome is in NONOK_IDS and whose op is not a client op."""
    return [i for i, (op, o) in enumerate(zip(row["ops"], row["outcomes"])) if int(o) in NONOK_IDS and int(op) not in client_ops]


def is_r_trace(row, lo_ns, hi_ns, client_ops):
    return is_post_trace(row, lo_ns, hi_ns) and bool(nonok_positions(row, client_ops))


def is_window_trace(row, lo_ns, hi_ns):
    """The legacy selection: the trace starts inside the window (no non-OK test)."""
    return lo_ns <= int(row["start_ns"]) < hi_ns


def split_of(path):
    for part in Path(path).parts:
        if part.startswith("split="):
            return part[len("split="):]
    raise ValueError(f"{path} carries no split=<name> directory")


def select_traces(corpus_dir, windows_ns, client_ops, select, cap):
    """`case_dir -> [row]` of the selected traces of every case (rows carry ROW_COLUMNS and `split`),
    sorted by `(start_ns, trace_id)`, the first `cap` when `cap > 0`."""
    if select not in SELECTS:
        raise ValueError(f"unknown selection {select!r}; registered: {SELECTS}")
    cap = int(cap)
    if cap < 0:
        raise ValueError(f"cap must be >= 0, got {cap}")
    _, files = view_inputs(corpus_dir, SPANS_VIEW)
    picked = {case_dir: [] for case_dir, _, _, _ in windows_ns}
    for f in files:
        split = split_of(f)
        for batch in pq.ParquetFile(f).iter_batches(batch_size=BATCH_ROWS, columns=list(ROW_COLUMNS)):
            for r in batch.to_pylist():
                for case_dir, _, lo, hi in windows_ns:
                    taken = is_r_trace(r, lo, hi, client_ops) if select == SELECT_POST_NONOK else is_window_trace(r, lo, hi)
                    if taken:
                        picked[case_dir].append({**{k: r[k] for k in ROW_COLUMNS}, "split": split})
    for rows in picked.values():
        rows.sort(key=lambda r: (int(r["start_ns"]), str(r["trace_id"])))
        if cap:
            del rows[cap:]
    return picked


def check_selection(case_dir, rows, cell):
    """The method side's `selected` list equals the recomputed R list (same order); `skipped` is a subset."""
    want = (cell.get("selected") or {}).get(case_dir)
    if want is None:
        raise ValueError(f"{case_dir}: the cell carries no selected list for this case")
    got = [str(r["trace_id"]) for r in rows]
    if got != [str(t) for t in want]:
        raise ValueError(f"{case_dir}: the method side selected {len(want)} traces, the score side recomputes "
                         f"{len(got)} (first difference at index {next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))})")
    skipped = [str(t) for t in (cell.get("skipped") or {}).get(case_dir, [])]
    extra = sorted(set(skipped) - set(got))
    if extra:
        raise ValueError(f"{case_dir}: skipped traces not in the selected list: {extra[:5]}")
    return set(skipped)


def took_nonok(row, label):
    """The fault took in this trace: a root-cause position with an outcome in NONOK_IDS. Asserts the
    label's `rc_took` (which also counts 4xx) agrees: took ⇒ rc_took."""
    outcomes = row["outcomes"]
    took = any(int(outcomes[i]) in NONOK_IDS for i in label["rc_positions_end"])
    if took and not label["rc_took"]:
        raise ValueError(f"trace {row['trace_id']}: a root-cause position is non-OK but the label says rc_took is false")
    return took


# --- positions, call tree, ranking ------------------------------------------------------------

def window_pos(i, c, max_len):
    """The arm's window position of ops index `i` (`i - c + 1`), None in the context (`i < c - 1`)
    or past the length cap (`i >= max_len`)."""
    if i < c - 1 or i >= max_len:
        return None
    return i - c + 1


def descendants(parent_pos, q):
    """`{position: hop >= 1}` of the call-tree subtree below `q`: the positions whose parent chain
    reaches `q`, by following child links from `q` with a visited guard (an inverted link,
    `parent_pos[j] < j`, is followed like any other). `q` itself is excluded."""
    children = defaultdict(list)
    for j, p in enumerate(parent_pos):
        p = int(p)
        if p >= 0:
            children[p].append(j)
    out, seen, frontier, hop = {}, {q}, [q], 0
    while frontier:
        hop += 1
        nxt = []
        for u in frontier:
            for v in children.get(u, ()):
                if v not in seen:
                    seen.add(v)
                    out[v] = hop
                    nxt.append(v)
        frontier = nxt
    return out


def rank_by_closure(closures, order):
    """Rank the items of several closures (`{item: hop}`, one per failed position) by the number of
    closures holding the item (desc), the smallest hop, then the item's index in `order`; reached
    items only."""
    index = {item: k for k, item in enumerate(order)}
    count, min_hop = Counter(), {}
    for cl in closures:
        for item, hop in cl.items():
            if item not in index:
                raise ValueError(f"item {item!r} is not in the ranking order")
            count[item] += 1
            min_hop[item] = min(int(hop), min_hop.get(item, int(hop)))
    return sorted(count, key=lambda it: (-count[it], min_hop[it], index[it]))


def topology_rank(row, ops, client_ops, hop0, services_order):
    """The rule on the true call tree: for every failed non-client position its descendants (and
    itself at hop 0 when `hop0`), client positions dropped, mapped to services through `ops`;
    `{"services": [reached, ranked], "positions": [reached ops indices, ranked], "set": sorted services}`."""
    ops_row = [int(x) for x in row["ops"]]
    svc_closures, pos_closures = [], []
    for q in nonok_positions(row, client_ops):
        desc = descendants(row["parent_pos"], q)
        if hop0:
            desc[q] = 0
        pos = {p: h for p, h in desc.items() if ops_row[p] not in client_ops}
        svc = {}
        for p, h in pos.items():
            s = ops[ops_row[p]][0]
            svc[s] = min(h, svc.get(s, h))
        svc_closures.append(svc)
        pos_closures.append(pos)
    services = rank_by_closure(svc_closures, services_order)
    positions = rank_by_closure(pos_closures, list(range(len(ops_row))))
    return {"services": services, "positions": positions, "set": sorted(services)}


def trace_services(row, ops, client_ops):
    """Sorted distinct non-client services of the whole trace (the trace's candidate services)."""
    return sorted({ops[int(op)][0] for op in row["ops"] if int(op) not in client_ops})


# --- metrics ----------------------------------------------------------------------------------

def hit(ranked, targets, k):
    """1 iff any of the first `k` ranked items is a target."""
    return int(any(x in targets for x in list(ranked)[:k]))


def exact_set(reached_services, rc_service):
    """`(1 iff the reached set is exactly {rc_service}, its size)`."""
    s = set(reached_services)
    return int(s == {rc_service}), len(s)


def random_services(n, k):
    """A uniformly random ranking of `n` services places the root cause in the first k with probability min(k, n) / n."""
    n = int(n)
    return 0.0 if n <= 0 else min(int(k), n) / n


def random_positions(m, r, k):
    """A uniformly random ranking of `m` positions, `r` of them root-cause, hits one in the first
    k' = min(k, m) with probability 1 - C(m - r, k') / C(m, k'); 0 when r = 0 or m = 0."""
    m, r, k = int(m), int(r), int(k)
    if m <= 0 or r <= 0:
        return 0.0
    if r > m:
        raise ValueError(f"r = {r} root-cause positions exceed m = {m} candidates")
    kk = min(k, m)
    return 1.0 - math.comb(m - r, kk) / math.comb(m, kk)


def filter_broadcast(case_ranking, trace_services, order):
    """A case-level ranking restricted to the trace's services, order kept, absent services appended in `order`."""
    present = set(trace_services)
    ranked = [str(s) for s in case_ranking]
    return [s for s in ranked if s in present] + [s for s in order if s in present and s not in ranked]


def _mean(values):
    vals = [v for v in values if v is not None]
    return None if not vals else float(sum(vals)) / len(vals)


# --- per-trace rows ---------------------------------------------------------------------------

def _window_frame(row, rc, c, max_len, client_ops):
    """The arm's position frame: the reachable root-cause window positions, the unreachable counts
    (context / truncation) and the number of non-client window positions (`m`)."""
    n = len(row["ops"])
    reachable, ctx, trunc = [], 0, 0
    for i in rc:
        p = window_pos(i, c, max_len)
        if p is None:
            if i < c - 1:
                ctx += 1
            else:
                trunc += 1
        else:
            reachable.append(p)
    m = sum(1 for i in range(max(c - 1, 0), min(n, max_len)) if int(row["ops"][i]) not in client_ops)
    return reachable, ctx, trunc, m


def _broadcast_hits(ranking, services, order, rc_service):
    """hit@k of a case-level ranking (or the mean over a list of per-run rankings, a None run 0)."""
    if ranking is None:
        return {str(k): 0.0 for k in KS}
    runs = list(ranking) if ranking and isinstance(ranking[0], (list, tuple, type(None))) else [ranking]
    hits = {str(k): [] for k in KS}
    for run in runs:
        ranked = [] if run is None else filter_broadcast(run, services, order)
        for k in KS:
            hits[str(k)].append(0 if run is None else hit(ranked, {rc_service}, k))
    return {k: float(sum(v)) / len(v) if v else 0.0 for k, v in hits.items()}


def per_trace_rows(case_dir, case, rows, labels, cell, mode, hop0, prediction, ops, client_ops, services_order, skipped):
    """The per-trace table of one case over its scored set C, plus the case's counters
    (`n_unprobed`, `n_rc_unreachable_context`, `n_rc_unreachable_truncation`, `n_took_4xx_only`)."""
    c, max_len = int(cell["c"]), int(cell["max_len"])
    rc_service = str(case["root_cause_component"])
    fault_index = fault_index_of(case)
    counters = Counter()
    out = []
    arm_cases = (prediction or {}).get("cases", {}).get(case_dir, {}) if mode == MODE_ARM else None
    ranking = (prediction or {}).get("cases", {}).get(case["case_id"]) if mode == MODE_BROADCAST else None
    for row in rows:
        tid = str(row["trace_id"])
        label = labels.get((tid, fault_index))
        if label is None:
            raise ValueError(f"{case_dir}: trace {tid} has no label row for fault {fault_index}")
        took = took_nonok(row, label)
        if not took:
            counters["n_took_4xx_only"] += int(bool(label["rc_took"]))
            continue
        rc = [int(i) for i in label["rc_positions_end"]]
        reachable, ctx, trunc, m_window = _window_frame(row, rc, c, max_len, client_ops)
        counters["n_rc_unreachable_context"] += ctx
        counters["n_rc_unreachable_truncation"] += trunc
        services = trace_services(row, ops, client_ops)
        n_services = len(services)
        random_s = {str(k): random_services(n_services, k) for k in KS}
        random_p = {str(k): random_positions(m_window, len(reachable), k) for k in KS}
        n_rc_reachable = len(reachable)
        hit_s, hit_p, exact, set_size, probed = None, None, None, None, True
        if mode == MODE_ARM:
            entry = None if tid in skipped else arm_cases.get(tid)
            if entry is None:
                probed = False
                hit_s, hit_p, exact, set_size = {str(k): 0 for k in KS}, {str(k): 0 for k in KS}, 0, 0
            else:
                ranked_s = [services_order[int(i)] for i in entry["s"]]
                ranked_p = [int(p) for p in entry["p"]]
                hit_s = {str(k): hit(ranked_s, {rc_service}, k) for k in KS}
                hit_p = {str(k): hit(ranked_p, set(reachable), k) for k in KS}
                exact, set_size = exact_set(ranked_s, rc_service)
        elif mode == MODE_TOPOLOGY:
            tr = topology_rank(row, ops, client_ops, hop0, services_order)
            rc_targets = {i for i in rc if int(row["ops"][i]) not in client_ops}
            n_rc_reachable = len(rc_targets)
            m_ops = sum(1 for op in row["ops"] if int(op) not in client_ops)
            random_p = {str(k): random_positions(m_ops, len(rc_targets), k) for k in KS}
            hit_s = {str(k): hit(tr["services"], {rc_service}, k) for k in KS}
            hit_p = {str(k): hit(tr["positions"], rc_targets, k) for k in KS}
            exact, set_size = exact_set(tr["set"], rc_service)
        elif mode == MODE_RANDOM:
            hit_s, hit_p = dict(random_s), dict(random_p)
        elif mode == MODE_CALLEES:
            callees = topology_rank(row, ops, client_ops, hop0, services_order)["set"]
            m_t = len(callees)
            hit_s = {str(k): (min(k, m_t) / m_t if rc_service in callees and m_t else 0.0) for k in KS}
            random_p = None
        elif mode == MODE_BROADCAST:
            hit_s = _broadcast_hits(ranking, services, services_order, rc_service)
            random_p = None
        else:
            raise ValueError(f"unknown mode {mode!r}; registered: {MODES}")
        counters["n_unprobed"] += int(not probed)
        out.append({"case_dir": case_dir, "case_id": case["case_id"], "trace_id": tid, "split": row.get("split"),
                    "n_services": n_services, "n_positions": len(row["ops"]), "n_rc": len(rc), "n_rc_reachable": n_rc_reachable,
                    "hit_s": hit_s, "hit_p": hit_p, "exact": exact, "set_size": set_size,
                    "random_s": random_s, "random_p": random_p, "probed": probed})
    return out, counters


# --- case rows, aggregates --------------------------------------------------------------------

def hit_rates(trace_rows):
    """Means over trace rows of the hit indicators: `{s@1, s@3, p@1, p@3, exact}` (None where n/a)."""
    rates = {}
    for k in KS:
        rates[f"s@{k}"] = _mean(r["hit_s"][str(k)] for r in trace_rows)
        rates[f"p@{k}"] = _mean(None if r["hit_p"] is None else r["hit_p"][str(k)] for r in trace_rows)
    rates["exact"] = _mean(r["exact"] for r in trace_rows)
    return rates


def random_rates(trace_rows):
    rates = {}
    for k in KS:
        rates[f"s@{k}"] = _mean(r["random_s"][str(k)] for r in trace_rows)
        rates[f"p@{k}"] = _mean(None if r["random_p"] is None else r["random_p"][str(k)] for r in trace_rows)
    return rates


def case_row(case_dir, case, n_selected, n_skipped, trace_rows, counters):
    return {"case_dir": case_dir, "case_id": case["case_id"], "fault_kind": case.get("fault_kind"),
            "root_cause_component": case["root_cause_component"], "n_selected": int(n_selected), "n_skipped": int(n_skipped),
            "n_traces": len(trace_rows), "n_unprobed": int(counters["n_unprobed"]), "hit_rate": hit_rates(trace_rows),
            "mean_set_size": _mean(r["set_size"] for r in trace_rows), "random": random_rates(trace_rows),
            "n_rc_unreachable_context": int(counters["n_rc_unreachable_context"]),
            "n_rc_unreachable_truncation": int(counters["n_rc_unreachable_truncation"]),
            "n_took_4xx_only": int(counters["n_took_4xx_only"])}


def metrics_of_cases(case_rows):
    """Means over cases of the case-mean rates (the reported point); `n_cases`, `n_traces` = sum |C|."""
    scored = [r for r in case_rows if r["n_traces"] > 0]
    out = {key: _mean(r["hit_rate"][key] for r in scored) for key in HIT_KEYS}
    out["mean_set_size"] = _mean(r["mean_set_size"] for r in scored)
    out["n_cases"] = len(case_rows)
    out["n_traces"] = int(sum(r["n_traces"] for r in case_rows))
    return out


def metrics_of_traces(trace_rows):
    """The same keys over trace rows directly (trace-weighted)."""
    out = dict(hit_rates(trace_rows))
    out["mean_set_size"] = _mean(r["set_size"] for r in trace_rows)
    out["n_cases"] = len({r["case_dir"] for r in trace_rows})
    out["n_traces"] = len(trace_rows)
    return out


def random_of_cases(case_rows):
    scored = [r for r in case_rows if r["n_traces"] > 0]
    return {key: _mean(r["random"][key] for r in scored) for key in RANDOM_KEYS}


def by_fault_kind(case_rows):
    kinds = sorted({r["fault_kind"] for r in case_rows if r["fault_kind"] is not None})
    return {kind: metrics_of_cases([r for r in case_rows if r["fault_kind"] == kind]) for kind in kinds}


def by_split(trace_rows):
    train = [r for r in trace_rows if r["split"] == SPLIT_TRAIN]
    rest = [r for r in trace_rows if r["split"] != SPLIT_TRAIN]
    return {SPLIT_TRAIN: metrics_of_traces(train), "val+test": metrics_of_traces(rest)}


def bootstrap(case_rows, metric_keys, b, seed):
    """Percentile intervals (LEVEL) of the mean over cases of the case-mean rates, over `b` resamples
    of the scored cases with replacement drawn from `numpy.random.default_rng(seed)`; a metric that
    is n/a (None) in any case is skipped."""
    rows = [r for r in case_rows if r["n_traces"] > 0]
    n = len(rows)
    if n == 0 or b <= 0:
        return {"b": int(b), "seed": int(seed), "level": LEVEL, "intervals": {}}
    keys = [k for k in metric_keys if all(r["hit_rate"].get(k) is not None for r in rows)]
    idx = np.random.default_rng(seed).integers(0, n, size=(int(b), n))
    if not keys:
        return {"b": int(b), "seed": int(seed), "level": LEVEL, "intervals": {}}
    mat = np.array([[float(r["hit_rate"][k]) for k in keys] for r in rows], dtype=np.float64)
    means = mat[idx].mean(axis=1)  # b × metrics
    lo, hi = (1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2
    intervals = {k: {"low": float(np.quantile(means[:, j], lo)), "high": float(np.quantile(means[:, j], hi))}
                 for j, k in enumerate(keys)}
    return {"b": int(b), "seed": int(seed), "level": LEVEL, "intervals": intervals}


# --- score, pool ------------------------------------------------------------------------------

def _check_prediction(mode, prediction, services_order):
    if mode == MODE_ARM:
        if not isinstance(prediction, dict) or "cases" not in prediction:
            raise ValueError("mode arm needs a prediction {services, cases}")
        if [str(s) for s in prediction.get("services", [])] != list(services_order):
            raise ValueError("the prediction's services order differs from the corpus's deployed services in name order")
    elif mode == MODE_BROADCAST:
        if not isinstance(prediction, dict) or "cases" not in prediction:
            raise ValueError("mode broadcast needs a prediction {cases: {case_id: ranking}}")
    elif prediction is not None:
        raise ValueError(f"mode {mode} takes no prediction")


def load_inputs(corpus_dir, cell):
    """Everything `score` reads from the corpus for one cell — the labels, the ops, the services order, the case windows
    and the selected rows — loaded once so a caller scoring several rows of one cell (the harness's `scoretrace`:
    the arm's two readings, the floor, the controls and the broadcast rows) pays the view and label reads once.
    `score(..., inputs=load_inputs(corpus_dir, cell))` is identical to `score(...)` without it."""
    corpus_dir = Path(corpus_dir)
    select, cap = str(cell["select"]), int(cell["cap"])
    ops, client_ops = load_ops(corpus_dir)
    windows = case_windows_ns(corpus_dir)
    return {"corpus_dir": corpus_dir, "select": select, "cap": cap, "labels": load_labels(corpus_dir), "ops": ops,
            "client_ops": client_ops, "services_order": services_sorted(corpus_dir), "windows": windows,
            "selected": select_traces(corpus_dir, windows, client_ops, select, cap)}


def score(corpus_dir, cell, prediction, mode, hop0, b, seed, inputs=None):
    corpus_dir = Path(corpus_dir)
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; registered: {MODES}")
    hop0 = int(hop0)
    if hop0 not in (0, 1):
        raise ValueError(f"hop0 must be 0 or 1, got {hop0}")
    select, cap = str(cell["select"]), int(cell["cap"])
    if inputs is None:
        inputs = load_inputs(corpus_dir, cell)
    elif (inputs["corpus_dir"], inputs["select"], inputs["cap"]) != (corpus_dir, select, cap):
        raise ValueError("the preloaded inputs are of another corpus or selection")
    labels, ops, client_ops = inputs["labels"], inputs["ops"], inputs["client_ops"]
    services_order, windows, selected = inputs["services_order"], inputs["windows"], inputs["selected"]
    _check_prediction(mode, prediction, services_order)
    case_rows, trace_rows = [], []
    for case_dir, case, _, _ in windows:
        rows = selected[case_dir]
        skipped = check_selection(case_dir, rows, cell)
        traces, counters = per_trace_rows(case_dir, case, rows, labels, cell, mode, hop0, prediction, ops, client_ops,
                                          services_order, skipped)
        case_rows.append(case_row(case_dir, case, len(rows), len(skipped), traces, counters))
        trace_rows.extend(traces)
    counts = {"n_candidates": len(trace_rows), "n_selected": int(sum(r["n_selected"] for r in case_rows)),
              "n_unprobed": int(sum(r["n_unprobed"] for r in case_rows)),
              "n_rc_unreachable_context": int(sum(r["n_rc_unreachable_context"] for r in case_rows)),
              "n_rc_unreachable_truncation": int(sum(r["n_rc_unreachable_truncation"] for r in case_rows)),
              "n_took_4xx_only": int(sum(r["n_took_4xx_only"] for r in case_rows)),
              "mean_n_services": _mean(r["n_services"] for r in trace_rows)}
    return {"axis": AXIS, "mode": mode, "hop0": hop0, "select": select, "cap": cap, "c": int(cell["c"]),
            "max_len": int(cell["max_len"]), "corpus_dir_name": corpus_dir.name, "metrics": metrics_of_cases(case_rows),
            "trace_weighted": metrics_of_traces(trace_rows), "bootstrap": bootstrap(case_rows, BOOTSTRAP_KEYS, b, seed),
            "random": random_of_cases(case_rows), "by_fault_kind": by_fault_kind(case_rows), "by_split": by_split(trace_rows),
            "counts": counts, "cases": case_rows, "traces": trace_rows}


def pool(results, b, seed):
    """One table over the case rows of several per-corpus results of one mode and hop-0 setting
    (the resampling unit stays the (corpus, case) pair)."""
    results = list(results)
    if not results:
        raise ValueError("nothing to pool")
    modes = {(res.get("axis"), res.get("mode"), int(res.get("hop0"))) for res in results}
    if len(modes) != 1:
        raise ValueError(f"results of different axes / modes / hop-0 settings cannot be pooled: {sorted(modes)}")
    axis, mode, hop0 = modes.pop()
    rows = [r for res in results for r in res["cases"]]
    return {"axis": axis, "mode": mode, "hop0": hop0, "pooled_from": [res.get("corpus_dir_name") for res in results],
            "n_cases_per_result": [len(res["cases"]) for res in results], "metrics": metrics_of_cases(rows),
            "bootstrap": bootstrap(rows, BOOTSTRAP_KEYS, b, seed), "random": random_of_cases(rows),
            "by_fault_kind": by_fault_kind(rows)}


# --- CLI --------------------------------------------------------------------------------------

def is_ranking_trace(doc):
    """The harness's `ranking-trace.json` (per-case `selected` / `skipped` / `traces`) rather than a bare cell."""
    return "selected" not in doc and isinstance(doc.get("cases"), dict)


def cell_of(doc):
    """The scorer's cell `{c, max_len, select, cap, selected, skipped}` from a cell json or a `ranking-trace.json`."""
    if not is_ranking_trace(doc):
        return {"c": doc["c"], "max_len": doc["max_len"], "select": doc["select"], "cap": doc["cap"],
                "selected": doc["selected"], "skipped": doc.get("skipped", {})}
    cases = doc["cases"]
    return {"c": doc["c"], "max_len": doc["max_len"], "select": doc["select"], "cap": doc["cap"],
            "selected": {cd: v["selected"] for cd, v in cases.items()},
            "skipped": {cd: v.get("skipped", []) for cd, v in cases.items()}}


def prediction_of(doc, reading):
    """Mode `arm`'s prediction from a `ranking-trace.json` at one reading (`s/p` for hop1, `s0/p0` for hop0)."""
    s_key, p_key, _ = READINGS[reading]
    return {"services": list(doc["services"]),
            "cases": {cd: {tid: {"s": t[s_key], "p": t[p_key]} for tid, t in v.get("traces", {}).items()} for cd, v in doc["cases"].items()}}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", help="a corpus with labels/, instantiation.json, config.yaml and views/end-request/")
    p.add_argument("--cell", help="the harness's ranking-trace.json, or a cell json {c, max_len, select, cap, selected, skipped}")
    p.add_argument("--mode", choices=MODES)
    p.add_argument("--hop0", type=int, choices=(0, 1), help="1 = a failed span itself is a candidate at hop 0")
    p.add_argument("--reading", choices=tuple(READINGS), help="mode arm with a ranking-trace.json: take s/p (hop1) or s0/p0 (hop0) from it")
    p.add_argument("--prediction", help="mode broadcast (or arm without --reading): the prediction JSON")
    p.add_argument("--pool", nargs="+", help="per-corpus result files to pool instead of scoring one corpus")
    p.add_argument("--bootstrap-b", required=True, type=int, help="bootstrap resamples (0 = none)")
    p.add_argument("--bootstrap-seed", required=True, type=int)
    p.add_argument("--out", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.pool:
        if args.corpus or args.cell or args.mode or args.hop0 is not None or args.prediction or args.reading:
            raise SystemExit("--pool takes result files only; --corpus / --cell / --mode / --hop0 score one corpus")
        result = pool([read_json(p) for p in args.pool], args.bootstrap_b, args.bootstrap_seed)
        write_json(args.out, result)
        log({"event": "score_rca_trace_pool", "mode": result["mode"], "hop0": result["hop0"], **result["metrics"]})
        return 0
    if not (args.corpus and args.cell and args.mode and args.hop0 is not None):
        raise SystemExit("score one corpus with --corpus, --cell, --mode and --hop0, or pool results with --pool")
    doc = json.loads(Path(args.cell).read_text(encoding="utf-8"))
    cell = cell_of(doc)
    prediction = None
    if args.reading:
        if args.mode != MODE_ARM or not is_ranking_trace(doc) or "services" not in doc:
            raise SystemExit("--reading applies to mode arm with --cell = the harness's ranking-trace.json")
        if args.prediction:
            raise SystemExit("--reading takes the prediction from --cell; drop --prediction")
        if READINGS[args.reading][2] != args.hop0:
            raise SystemExit(f"--reading {args.reading} is the hop0={READINGS[args.reading][2]} row; --hop0 {args.hop0} contradicts it")
        prediction = prediction_of(doc, args.reading)
    elif args.mode in (MODE_ARM, MODE_BROADCAST):
        if not args.prediction:
            raise SystemExit(f"mode {args.mode} needs --prediction (or, for arm, --reading with a ranking-trace.json)")
        prediction = json.loads(Path(args.prediction).read_text(encoding="utf-8"))
    elif args.prediction:
        raise SystemExit(f"mode {args.mode} takes no --prediction")
    result = score(args.corpus, cell, prediction, args.mode, args.hop0, args.bootstrap_b, args.bootstrap_seed)
    write_json(args.out, result)
    log({"event": "score_rca_trace", "mode": args.mode, "hop0": args.hop0, **result["metrics"], **result["counts"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
