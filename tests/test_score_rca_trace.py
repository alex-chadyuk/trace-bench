"""The per-trace root-cause axis (docs/scorers-service-rca.md §4, 2026-10-31): rule R against the shipped
labels on a real xs corpus, the window / call-tree / ranking primitives by hand and by enumeration, the
bootstrap against `score_rca`'s draws, pooling, the broadcast filter, and every mode end to end on a
hand-written corpus with known answers (including the selection-mismatch refusal and the CLI)."""
import itertools
import json
import math

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from corpus_fixture import xs_corpus

from tracebench import score_rca
from tracebench.constants import (KIND_BFF, KIND_CLIENT, KIND_SERVICE, OUTCOME_4XX, OUTCOME_5XX, OUTCOME_ERR,
                                  OUTCOME_OK, OUTCOME_SLOW, PHASE_NONE, PHASE_POST, PHASE_PRE, PHASE_STRADDLE)
from tracebench.correlate.views import ARROW_SCHEMA, LABELS_SCHEMA
from tracebench.export_rca import case_dirs, window_epoch
from tracebench.record import read_json, write_json
from tracebench.score_rca_trace import (MODES, NONOK_IDS, SELECT_POST_NONOK, SELECT_WINDOW, bootstrap,
                                        case_windows_ns, check_selection, descendants, exact_set, fault_index_of,
                                        filter_broadcast, hit, is_post_trace, is_r_trace, load_labels, load_ops,
                                        main, pool, random_positions, random_services, rank_by_closure, score,
                                        select_traces, services_sorted, took_nonok, topology_rank, window_pos)

# --- 1. rule R against the shipped labels on a real corpus ---------------------------------------------


@pytest.fixture(scope="module")
def xs_inputs():
    corpus = xs_corpus()
    ops, client_ops = load_ops(corpus)
    labels = load_labels(corpus)
    windows = case_windows_ns(corpus)
    import glob
    files = sorted(glob.glob(str(corpus / "views" / "end-request" / "sequences" / "split=*" / "date=*" / "*.parquet")))
    rows = {r["trace_id"]: r for f in files for r in pq.read_table(f).to_pylist()}
    return corpus, ops, client_ops, labels, windows, rows


def test_post_clause_of_rule_r_is_the_labels_phase_post(xs_inputs):
    corpus, _, client_ops, labels, windows, rows = xs_inputs
    assert len(windows) == 2 and {fault_index_of(c) for _, c, _, _ in windows} == {0, 1}
    n = 0
    for case_dir, case, lo, hi in windows:
        fi = fault_index_of(case)
        for (tid, fault_index), lab in labels.items():
            if fault_index != fi:
                continue
            assert is_post_trace(rows[tid], lo, hi) == (lab["phase"] == PHASE_POST), (case_dir, tid)
            n += 1
    assert n == len(labels) and n > 1000
    # the window selection reproduces the legacy start-inside rule; the R list is POST-only and non-empty
    sel = select_traces(corpus, windows, client_ops, SELECT_POST_NONOK, 0)
    for case_dir, case, lo, hi in windows:
        ids = [r["trace_id"] for r in sel[case_dir]]
        assert ids and ids == sorted(ids, key=lambda t: (rows[t]["start_ns"], t))
        assert all(labels[(t, fault_index_of(case))]["phase"] == PHASE_POST for t in ids)
        assert all(is_r_trace(rows[t], lo, hi, client_ops) for t in ids)
        assert {r["split"] for r in sel[case_dir]} <= {"train", "val", "test"}
    capped = select_traces(corpus, windows, client_ops, SELECT_POST_NONOK, 3)
    assert all(len(v) == 3 for v in capped.values())
    legacy = select_traces(corpus, windows, client_ops, SELECT_WINDOW, 0)
    assert all(len(legacy[d]) >= len(sel[d]) for d in sel)


def test_rc_positions_end_are_the_root_cause_endpoint_positions_and_took_implies_rc_took(xs_inputs):
    _, ops, _, labels, windows, rows = xs_inputs
    cases = {fault_index_of(c): c for _, c, _, _ in windows}
    n_took = 0
    for (tid, fault_index), lab in labels.items():
        case = cases[fault_index]
        rc_ops = {oid for oid, (svc, name) in ops.items() if svc == case["root_cause_component"] and name in case["root_cause_endpoints"]}
        assert rc_ops
        row = rows[tid]
        assert lab["rc_positions_end"] == [i for i, op in enumerate(row["ops"]) if op in rc_ops], tid
        assert all(ops[row["ops"][i]][0] == case["root_cause_component"] for i in lab["rc_positions_end"])
        n_took += took_nonok(row, lab)  # raises if took and not rc_took
    assert n_took > 0
    with pytest.raises(ValueError):
        took_nonok({"trace_id": "x", "outcomes": [OUTCOME_5XX]}, {"rc_positions_end": [0], "rc_took": False})
    assert not took_nonok({"trace_id": "x", "outcomes": [OUTCOME_4XX]}, {"rc_positions_end": [0], "rc_took": True})


# --- 2–4. primitives -------------------------------------------------------------------------------------


def test_window_pos_context_and_truncation():
    for c in (1, 2, 3):
        got = [window_pos(i, c, 64) for i in range(70)]
        assert got[:c - 1] == [None] * (c - 1)  # the context
        assert got[c - 1:64] == list(range(64 - (c - 1)))  # p = i - c + 1
        assert got[64:] == [None] * 6  # past the length cap
    assert window_pos(0, 1, 64) == 0 and window_pos(5, 3, 64) == 3 and window_pos(63, 1, 64) == 63 and window_pos(63, 2, 64) == 62


def test_descendants_tolerate_inverted_links_and_return_hops():
    parent = [2, 2, 6, 6, 3, 4, -1]  # position 5's parent is 4 (an inverted link, parent < child)
    assert descendants(parent, 6) == {2: 1, 3: 1, 0: 2, 1: 2, 4: 2, 5: 3}
    assert descendants(parent, 3) == {4: 1, 5: 2}
    assert descendants(parent, 4) == {5: 1} and descendants(parent, 5) == {} and descendants(parent, 0) == {}
    assert descendants([1, 0], 0) == {1: 1}  # a two-cycle: the visited guard stops the walk
    ops = {0: ("svc-a", "/a"), 1: ("svc-b", "/b"), 3: ("svc-c", "/c"), 4: ("web", "/p")}
    row = {"ops": [1, 1, 0, 0, 1, 3, 4], "outcomes": [0, 0, 0, 0, 2, 2, 0], "parent_pos": parent}
    order = ["svc-a", "svc-b", "svc-c"]
    r1 = topology_rank(row, ops, {4}, 0, order)
    assert r1 == {"services": ["svc-c"], "positions": [5], "set": ["svc-c"]}
    r0 = topology_rank(row, ops, {4}, 1, order)
    assert r0 == {"services": ["svc-c", "svc-b"], "positions": [5, 4], "set": ["svc-b", "svc-c"]}
    assert rank_by_closure([{"a": 2, "b": 1}, {"a": 1}], ["a", "b"]) == ["a", "b"]
    assert rank_by_closure([{"a": 2}, {"b": 1}], ["a", "b"]) == ["b", "a"]  # ties by the smallest hop
    assert rank_by_closure([{"a": 1}, {"b": 1}], ["b", "a"]) == ["b", "a"]  # then by the order


def test_hit_exact_set_and_random_expectations_by_enumeration():
    assert hit(["x", "y"], {"y"}, 1) == 0 and hit(["x", "y"], {"y"}, 3) == 1 and hit([], {"y"}, 3) == 0
    assert exact_set(["svc-c"], "svc-c") == (1, 1) and exact_set(["svc-c", "svc-b"], "svc-c") == (0, 2)
    assert exact_set([], "svc-c") == (0, 0) and exact_set(["svc-b"], "svc-c") == (0, 1)
    for n in range(6):
        for k in (1, 3):
            expect = 0.0 if n == 0 else sum(1 for p in itertools.permutations(range(n)) if 0 in p[:k]) / math.factorial(n)
            assert random_services(n, k) == pytest.approx(expect)
    for m in range(7):
        for r in range(m + 1):
            for k in (1, 3):
                kk = min(k, m)
                if m == 0:
                    expect = 0.0
                else:
                    subsets = list(itertools.combinations(range(m), kk))
                    expect = sum(1 for s in subsets if any(i < r for i in s)) / len(subsets)
                assert random_positions(m, r, k) == pytest.approx(expect), (m, r, k)
    with pytest.raises(ValueError):
        random_positions(2, 3, 1)


# --- 5–7. bootstrap, pool, broadcast filter ------------------------------------------------------------

def _case(rate, n_traces=1, kind="crash", d="case-00"):
    return {"case_dir": d, "case_id": d, "fault_kind": kind, "root_cause_component": "svc", "n_selected": n_traces,
            "n_skipped": 0, "n_traces": n_traces, "n_unprobed": 0,
            "hit_rate": {"s@1": rate, "s@3": rate, "p@1": None, "p@3": None, "exact": rate}, "mean_set_size": 1.0,
            "random": {"s@1": 0.5, "s@3": 1.0, "p@1": None, "p@3": None},
            "n_rc_unreachable_context": 0, "n_rc_unreachable_truncation": 0, "n_took_4xx_only": 0}


def test_bootstrap_matches_score_rca_on_binary_case_rates():
    rates = [1, 0, 1, 1, 0, 0, 1]
    rows = [_case(float(r)) for r in rates]
    mine = bootstrap(rows, ("s@1", "s@3", "p@1", "p@3", "exact"), 300, 3)
    ref = score_rca.bootstrap([1 if r else None for r in rates], 300, 3)
    assert mine["intervals"]["s@1"] == ref["intervals"]["ac@1"] == mine["intervals"]["exact"]
    assert set(mine["intervals"]) == {"s@1", "s@3", "exact"}  # the None columns are skipped
    assert mine["b"] == 300 and mine["seed"] == 3 and mine["level"] == 0.95
    assert bootstrap(rows, ("s@1",), 300, 3) == {**mine, "intervals": {"s@1": mine["intervals"]["s@1"]}}
    assert bootstrap(rows, ("s@1",), 300, 4) != bootstrap(rows, ("s@1",), 300, 3)
    assert bootstrap([], ("s@1",), 300, 3)["intervals"] == {} and bootstrap(rows, ("s@1",), 0, 3)["intervals"] == {}
    assert bootstrap([_case(None, n_traces=0)], ("s@1",), 10, 0)["intervals"] == {}  # an empty C is not a scored case


def test_filter_broadcast_and_run_averaged_hit():
    from tracebench.score_rca_trace import _broadcast_hits
    order = ["svc-a", "svc-b", "svc-c", "svc-d"]
    assert filter_broadcast(["svc-c", "svc-x", "svc-a"], ["svc-a", "svc-b", "svc-c"], order) == ["svc-c", "svc-a", "svc-b"]
    assert filter_broadcast([], ["svc-b", "svc-c"], order) == ["svc-b", "svc-c"]
    assert filter_broadcast(["svc-d"], ["svc-a"], order) == ["svc-a"]
    svcs = ["svc-a", "svc-b", "svc-c"]
    assert _broadcast_hits(["svc-b", "svc-c"], svcs, order, "svc-c") == {"1": 0.0, "3": 1.0}
    assert _broadcast_hits(None, svcs, order, "svc-c") == {"1": 0.0, "3": 0.0}
    runs = [["svc-c", "svc-b"], None, ["svc-b", "svc-c"], ["svc-c"]]
    assert _broadcast_hits(runs, svcs, order, "svc-c") == {"1": 0.5, "3": 0.75}
    assert _broadcast_hits([None, None], svcs, order, "svc-c") == {"1": 0.0, "3": 0.0}


# --- 8. end to end on a hand-written corpus -----------------------------------------------------------

EPOCH = 1767603600  # 2026-01-05T09:00:00Z
MS = 1_000_000
SERVICES = ["svc-a", "svc-b", "svc-c"]  # name order = index order of an arm's "s" lists
CASES = [
    {"case_id": "hand_svc-c_crash_0", "system": "hand", "root_cause_component": "svc-c", "root_cause_endpoints": ["/c"],
     "fault_kind": "crash", "inject_time_s": 600.0, "end_time_s": 900.0},
    {"case_id": "hand_svc-b_degrade_1", "system": "hand", "root_cause_component": "svc-b", "root_cause_endpoints": ["/b1"],
     "fault_kind": "degrade", "inject_time_s": 1200.0, "end_time_s": 1500.0},
]
BASE_OPS = [{"id": 0, "service": "svc-a", "name": "/a", "kind": KIND_BFF}, {"id": 1, "service": "svc-b", "name": "/b1", "kind": KIND_SERVICE},
            {"id": 2, "service": "svc-b", "name": "/b2", "kind": KIND_SERVICE}, {"id": 3, "service": "svc-c", "name": "/c", "kind": KIND_SERVICE},
            {"id": 4, "service": "web", "name": "/page", "kind": KIND_CLIENT}]
RC_OP = {0: 3, 1: 1}  # fault index -> root-cause op
OK, X4, X5, ERR, SLOW = OUTCOME_OK, OUTCOME_4XX, OUTCOME_5XX, OUTCOME_ERR, OUTCOME_SLOW
# (trace_id, start_s, split, ops, outcomes, parent_pos, offsets_ms, durations_ms); end-ordered (callee first)
TRACES = [
    ("ta", 610, "train", [1, 3, 0, 4], [OK, X5, OK, OK], [2, 2, 3, -1], [2, 3, 1, 0], [10, 20, 90, 100]),  # only failed span = rc span
    ("tb", 620, "val", [3, 2, 0, 4], [X4, X5, OK, OK], [1, 2, 3, -1], [3, 2, 1, 0], [20, 50, 90, 100]),  # rc span 4xx only: in R, not in C
    ("tc", 630, "train", [3, 1, 0, 4], [ERR, ERR, OK, OK], [1, 2, 3, -1], [3, 2, 1, 0], [20, 50, 90, 100]),  # rc span in the context (c = 2)
    ("td", 640, "val", [3, 4], [X5, OK], [1, -1], [1, 0], [10, 20]),  # skipped by the method
    ("te", 650, "train", [1, 2, 0, 0, 1, 3, 4], [OK, OK, OK, OK, X5, X5, OK], [2, 2, 6, 6, 3, 4, -1],
     [2, 2, 1, 1, 2, 3, 0], [5, 5, 40, 40, 20, 10, 100]),  # inverted link 5 -> 4; rc past max_len 5
    ("tf", 1210, "test", [3, 1, 0, 4], [OK, SLOW, SLOW, OK], [1, 2, 3, -1], [3, 2, 1, 0], [20, 50, 90, 100]),  # case 1, rc reachable
    ("tg", 1220, "test", [2, 0, 4], [X5, X5, OK], [1, 2, -1], [2, 1, 0], [10, 50, 60]),  # case 1: /b2 is not an rc endpoint
    ("th", 1100, "train", [3, 0, 4], [X5, X5, OK], [1, 2, -1], [2, 1, 0], [10, 50, 60]),  # outside both windows
    ("ti", 700, "train", [3, 0, 4], [OK, OK, OK], [1, 2, -1], [2, 1, 0], [10, 50, 60]),  # POST but healthy
    ("tj", 1490, "test", [1, 0, 4], [X5, X5, OK], [1, 2, -1], [1, 1, 0], [100, 19000, 20000]),  # straddles the end
    ("tk", 660, "val", [3, 4], [OK, ERR], [1, -1], [1, 0], [10, 20]),  # only the client span failed
]
CELL = {"c": 2, "max_len": 5, "select": "post-nonok", "cap": 0,
        "selected": {"case-00": ["ta", "tb", "tc", "td", "te"], "case-01": ["tf", "tg"]},
        "skipped": {"case-00": ["td"], "case-01": []}}
ARM_HOP1 = {"services": SERVICES, "cases": {
    "case-00": {"ta": {"s": [2, 1], "p": [1, 0]}, "tc": {"s": [2], "p": [0]}, "te": {"s": [1], "p": [3]}},
    "case-01": {"tf": {"s": [2, 1, 0], "p": [0, 1]}}}}
ARM_HOP0 = {"services": SERVICES, "cases": {
    "case-00": {"ta": {"s": [2], "p": [0]}, "tc": {"s": [1, 2], "p": [1, 0]}, "te": {"s": [2, 1], "p": [4, 3]}},
    "case-01": {"tf": {"s": [1], "p": [0]}}}}
BROADCAST = {"cases": {"hand_svc-c_crash_0": ["svc-b", "svc-c"], "hand_svc-b_degrade_1": ["svc-a", "svc-c", "svc-b"]}}
BROADCAST_RUNS = {"cases": {"hand_svc-c_crash_0": [["svc-c", "svc-b"], None, ["svc-b", "svc-c"], ["svc-c"]]}}


def _phase(start_s, end_s, lo, hi):
    if end_s < lo:
        return PHASE_PRE
    if start_s >= hi:
        return PHASE_NONE
    if start_s >= lo and end_s < hi:
        return PHASE_POST
    return PHASE_STRADDLE


def _view_row(tid, start_s, ops, outcomes, parent, offs_ms, durs_ms):
    return {"trace_id": tid, "ops": ops, "outcomes": outcomes, "offsets": [o * MS for o in offs_ms], "durations": [d * MS for d in durs_ms],
            "parent_pos": parent, "healthy": all(o == OK for o in outcomes), "n_err_spans": sum(1 for o in outcomes if o != OK),
            "n_spans": len(ops), "truncated": False, "scenario_hash": "h", "scenario_sid": -1, "start_ns": (EPOCH + start_s) * 10**9,
            "attribution_level": 0, "sequence_kind": "request"}


@pytest.fixture(scope="module")
def hand_corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("hand") / "hand-corpus"
    d.mkdir()
    (d / "config.yaml").write_text(yaml.safe_dump({"run": {"window": {"start": "2026-01-05T09:00:00Z", "duration_hours": 1.0}}}))
    services = [{"index": i, "name": n, "kind": k} for i, (n, k) in enumerate([("svc-a", KIND_BFF), ("svc-b", KIND_SERVICE),
                                                                              ("svc-c", KIND_SERVICE), ("web", KIND_CLIENT)])]
    write_json(d / "instantiation.json", {"topology": {"services": services}})
    write_json(d / "labels" / "cases.json", {"cases": CASES})
    write_json(d / "views" / "end-request" / "model-vocab.json", {"base_ops": BASE_OPS})
    by_split = {}
    labels = []
    for tid, start_s, split, ops, outcomes, parent, offs, durs in TRACES:
        by_split.setdefault(split, []).append(_view_row(tid, start_s, ops, outcomes, parent, offs, durs))
        end_s = start_s + max(o + du for o, du in zip(offs, durs)) / 1000.0
        for fi, case in enumerate(CASES):
            rc = [i for i, op in enumerate(ops) if op == RC_OP[fi]]
            labels.append({"trace_id": tid, "grain": "request", "fault_index": fi,
                           "phase": _phase(start_s, end_s, case["inject_time_s"], case["end_time_s"]),
                           "rc_positions_end": rc, "rc_positions_start": rc, "rc_took": any(outcomes[i] != OK for i in rc)})
        labels.append({"trace_id": tid, "grain": "session", "fault_index": 0, "phase": PHASE_NONE, "rc_positions_end": [],
                       "rc_positions_start": [], "rc_took": False})  # a session-grain row is never read
    for split, rows in by_split.items():
        part = d / "views" / "end-request" / "sequences" / f"split={split}" / "date=2026-01-05" / "part-0000.parquet"
        part.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows, schema=ARROW_SCHEMA), part)
    pq.write_table(pa.Table.from_pylist(labels, schema=LABELS_SCHEMA), d / "labels" / "sequence-labels.parquet")
    return d


def _rates(res, case_dir):
    return next(c for c in res["cases"] if c["case_dir"] == case_dir)["hit_rate"]


def _trace(res, tid):
    return next(t for t in res["traces"] if t["trace_id"] == tid)


def test_hand_corpus_inputs_and_selection(hand_corpus):
    assert services_sorted(hand_corpus) == SERVICES
    ops, client_ops = load_ops(hand_corpus)
    assert client_ops == {4} and ops[3] == ("svc-c", "/c")
    windows = case_windows_ns(hand_corpus)
    assert [(cd, lo // 10**9 - EPOCH, hi // 10**9 - EPOCH) for cd, _, lo, hi in windows] == [("case-00", 600, 900), ("case-01", 1200, 1500)]
    assert window_epoch(yaml.safe_load((hand_corpus / "config.yaml").read_text()))[0] == EPOCH
    assert [cd for cd, _ in case_dirs(CASES)] == ["case-00", "case-01"]
    sel = select_traces(hand_corpus, windows, client_ops, SELECT_POST_NONOK, 0)
    assert {cd: [r["trace_id"] for r in rows] for cd, rows in sel.items()} == CELL["selected"]
    assert [r["split"] for r in sel["case-00"]] == ["train", "val", "train", "val", "train"]
    assert set(sel["case-00"][0]) == {"trace_id", "ops", "outcomes", "offsets", "durations", "parent_pos", "start_ns", "split"}
    legacy = select_traces(hand_corpus, windows, client_ops, SELECT_WINDOW, 0)
    assert [r["trace_id"] for r in legacy["case-00"]] == ["ta", "tb", "tc", "td", "te", "tk", "ti"]  # start inside, no non-OK test
    assert [r["trace_id"] for r in legacy["case-01"]] == ["tf", "tg", "tj"]
    assert [r["trace_id"] for r in select_traces(hand_corpus, windows, client_ops, SELECT_POST_NONOK, 2)["case-00"]] == ["ta", "tb"]
    assert len(load_labels(hand_corpus)) == 2 * len(TRACES)  # the session-grain rows are dropped
    with pytest.raises(ValueError):
        select_traces(hand_corpus, windows, client_ops, "phase", 0)
    with pytest.raises(ValueError, match="case-00"):
        check_selection("case-00", sel["case-00"], {"selected": {"case-00": ["ta", "tb", "tc", "te"]}, "skipped": {}})
    with pytest.raises(ValueError, match="case-00"):
        check_selection("case-00", sel["case-00"], {"selected": CELL["selected"], "skipped": {"case-00": ["tz"]}})
    with pytest.raises(ValueError, match="case-01"):
        check_selection("case-01", sel["case-01"], {"selected": {"case-00": CELL["selected"]["case-00"]}, "skipped": {}})
    bad_cell = {**CELL, "selected": {"case-00": ["ta", "tb", "tc", "td"], "case-01": ["tf", "tg"]}}
    with pytest.raises(ValueError, match="case-00"):
        score(hand_corpus, bad_cell, None, "topology", 0, 0, 0)
    assert case_windows_ns(hand_corpus)[0][1]["case_id"] == "hand_svc-c_crash_0"


def test_counts_and_truth_are_mode_independent(hand_corpus):
    for mode, pred in (("topology", None), ("random", None), ("callees", None), ("broadcast", BROADCAST)):
        res = score(hand_corpus, CELL, pred, mode, 0, 0, 0)
        assert res["axis"] == "trace" and res["mode"] == mode and res["hop0"] == 0 and res["select"] == "post-nonok" and res["cap"] == 0
        assert res["corpus_dir_name"] == "hand-corpus"
        c0, c1 = res["cases"]
        assert (c0["n_selected"], c0["n_skipped"], c0["n_traces"], c0["n_took_4xx_only"]) == (5, 1, 4, 1)
        assert (c0["n_rc_unreachable_context"], c0["n_rc_unreachable_truncation"]) == (2, 1)
        assert (c1["n_selected"], c1["n_skipped"], c1["n_traces"], c1["n_took_4xx_only"]) == (2, 0, 1, 0)
        assert c0["fault_kind"] == "crash" and c1["root_cause_component"] == "svc-b"
        assert res["counts"] == {"n_candidates": 5, "n_selected": 7, "n_unprobed": 0, "n_rc_unreachable_context": 2,
                                 "n_rc_unreachable_truncation": 1, "n_took_4xx_only": 1, "mean_n_services": pytest.approx(2.6)}
        assert [t["trace_id"] for t in res["traces"]] == ["ta", "tc", "td", "te", "tf"]
        assert [t["split"] for t in res["traces"]] == ["train", "train", "val", "train", "test"]
        te = _trace(res, "te")
        assert (te["n_positions"], te["n_rc"], te["n_services"]) == (7, 1, 3) and te["probed"]
        assert res["metrics"]["n_cases"] == 2 and res["metrics"]["n_traces"] == 5 and res["trace_weighted"]["n_traces"] == 5
        assert set(res["by_fault_kind"]) == {"crash", "degrade"} and res["by_fault_kind"]["degrade"]["n_traces"] == 1
        assert res["by_split"]["train"]["n_traces"] == 3 and res["by_split"]["val+test"]["n_traces"] == 2
        # the per-trace random expectation (arm frame: c = 2, max_len = 5) is the same in every mode but topology
        assert _trace(res, "ta")["random_s"] == {"1": pytest.approx(1 / 3), "3": 1.0}
        assert _trace(res, "td")["random_s"] == {"1": 1.0, "3": 1.0}
        if mode in ("broadcast", "callees"):
            assert _trace(res, "ta")["random_p"] is None and res["random"]["p@1"] is None
        elif mode == "random":
            assert _trace(res, "ta")["random_p"] == {"1": 0.5, "3": 1.0} and _trace(res, "tc")["random_p"] == {"1": 0.0, "3": 0.0}
            assert _trace(res, "td")["random_p"] == {"1": 0.0, "3": 0.0} and _trace(res, "tf")["random_p"] == {"1": 0.5, "3": 1.0}
        assert res["random"]["s@1"] == pytest.approx((0.5 + 1 / 3) / 2) and res["random"]["s@3"] == 1.0


def test_topology_floor_known_answers(hand_corpus):
    res = score(hand_corpus, CELL, None, "topology", 0, 200, 0)
    per = {t["trace_id"]: (t["hit_s"]["1"], t["hit_s"]["3"], t["hit_p"]["1"], t["hit_p"]["3"], t["exact"], t["set_size"]) for t in res["traces"]}
    assert per == {"ta": (0, 0, 0, 0, 0, 0), "tc": (1, 1, 1, 1, 1, 1), "td": (0, 0, 0, 0, 0, 0), "te": (1, 1, 1, 1, 1, 1), "tf": (0, 1, 0, 1, 0, 2)}
    assert _rates(res, "case-00") == {"s@1": 0.5, "s@3": 0.5, "p@1": 0.5, "p@3": 0.5, "exact": 0.5}
    assert _rates(res, "case-01") == {"s@1": 0.0, "s@3": 1.0, "p@1": 0.0, "p@3": 1.0, "exact": 0.0}
    assert res["metrics"] == {"s@1": 0.25, "s@3": 0.75, "p@1": 0.25, "p@3": 0.75, "exact": 0.25, "mean_set_size": 1.25, "n_cases": 2, "n_traces": 5}
    assert res["trace_weighted"] == {"s@1": 0.4, "s@3": 0.6, "p@1": 0.4, "p@3": 0.6, "exact": 0.4, "mean_set_size": 0.8, "n_cases": 2, "n_traces": 5}
    assert res["by_split"]["train"]["s@1"] == pytest.approx(2 / 3) and res["by_split"]["val+test"]["s@1"] == 0.0
    assert res["by_fault_kind"]["crash"]["s@1"] == 0.5 and res["by_fault_kind"]["degrade"]["exact"] == 0.0
    assert set(res["bootstrap"]["intervals"]) == {"s@1", "s@3", "p@1", "p@3", "exact"} and res["bootstrap"]["b"] == 200
    for key, iv in res["bootstrap"]["intervals"].items():
        assert 0.0 <= iv["low"] <= res["metrics"][key] <= iv["high"] <= 1.0
    # positions are ops indices here: n_rc_reachable counts the root-cause positions themselves, the random frame is the whole trace
    assert _trace(res, "te")["n_rc_reachable"] == 1 and _trace(res, "ta")["random_p"] == {"1": pytest.approx(1 / 3), "3": 1.0}
    hop0 = score(hand_corpus, CELL, None, "topology", 1, 0, 0)
    per0 = {t["trace_id"]: (t["hit_s"]["1"], t["hit_p"]["1"], t["exact"], t["set_size"]) for t in hop0["traces"]}
    assert per0 == {"ta": (1, 1, 1, 1), "tc": (1, 1, 0, 2), "td": (1, 1, 1, 1), "te": (1, 1, 0, 2), "tf": (1, 1, 0, 3)}
    assert hop0["metrics"] == {"s@1": 1.0, "s@3": 1.0, "p@1": 1.0, "p@3": 1.0, "exact": 0.25, "mean_set_size": 2.25, "n_cases": 2, "n_traces": 5}
    assert hop0["hop0"] == 1 and hop0["bootstrap"]["intervals"] == {}
    with pytest.raises(ValueError):
        score(hand_corpus, CELL, BROADCAST, "topology", 0, 0, 0)


def test_random_and_callees_controls_known_answers(hand_corpus):
    rnd = score(hand_corpus, CELL, None, "random", 0, 50, 0)
    assert rnd["metrics"]["s@1"] == pytest.approx(5 / 12) and rnd["metrics"]["s@3"] == 1.0
    assert rnd["metrics"]["p@1"] == pytest.approx(0.3125) and rnd["metrics"]["p@3"] == pytest.approx(0.625)
    assert rnd["metrics"]["exact"] is None and rnd["metrics"]["mean_set_size"] is None
    assert _rates(rnd, "case-00") == {"s@1": pytest.approx(0.5), "s@3": 1.0, "p@1": 0.125, "p@3": 0.25, "exact": None}
    assert {k: rnd["metrics"][k] for k in ("s@1", "s@3", "p@1", "p@3")} == rnd["random"]
    assert set(rnd["bootstrap"]["intervals"]) == {"s@1", "s@3", "p@1", "p@3"}
    assert _trace(rnd, "ta")["exact"] is None and _trace(rnd, "ta")["set_size"] is None
    cal = score(hand_corpus, CELL, None, "callees", 0, 0, 0)
    per = {t["trace_id"]: (t["hit_s"]["1"], t["hit_s"]["3"]) for t in cal["traces"]}
    assert per == {"ta": (0.0, 0.0), "tc": (1.0, 1.0), "td": (0.0, 0.0), "te": (1.0, 1.0), "tf": (0.5, 1.0)}
    assert cal["metrics"]["s@1"] == 0.5 and cal["metrics"]["s@3"] == 0.75 and cal["metrics"]["p@1"] is None and cal["metrics"]["exact"] is None
    assert _trace(cal, "ta")["hit_p"] is None and _trace(cal, "ta")["random_p"] is None
    cal0 = score(hand_corpus, CELL, None, "callees", 1, 0, 0)
    per0 = {t["trace_id"]: (t["hit_s"]["1"], t["hit_s"]["3"]) for t in cal0["traces"]}
    assert per0 == {"ta": (1.0, 1.0), "tc": (0.5, 1.0), "td": (1.0, 1.0), "te": (0.5, 1.0), "tf": (pytest.approx(1 / 3), 1.0)}
    assert cal0["metrics"]["s@1"] == pytest.approx(13 / 24) and cal0["metrics"]["s@3"] == 1.0


def test_broadcast_plain_and_run_averaged(hand_corpus):
    res = score(hand_corpus, CELL, BROADCAST, "broadcast", 0, 0, 0)
    per = {t["trace_id"]: (t["hit_s"]["1"], t["hit_s"]["3"], t["hit_p"], t["exact"], t["set_size"]) for t in res["traces"]}
    assert per == {"ta": (0.0, 1.0, None, None, None), "tc": (0.0, 1.0, None, None, None), "td": (1.0, 1.0, None, None, None),
                   "te": (0.0, 1.0, None, None, None), "tf": (0.0, 1.0, None, None, None)}
    assert res["metrics"] == {"s@1": 0.125, "s@3": 1.0, "p@1": None, "p@3": None, "exact": None, "mean_set_size": None, "n_cases": 2, "n_traces": 5}
    assert set(score(hand_corpus, CELL, BROADCAST, "broadcast", 0, 20, 0)["bootstrap"]["intervals"]) == {"s@1", "s@3"}
    runs = score(hand_corpus, CELL, BROADCAST_RUNS, "broadcast", 0, 0, 0)
    per = {t["trace_id"]: (t["hit_s"]["1"], t["hit_s"]["3"]) for t in runs["traces"]}
    assert per == {"ta": (0.5, 0.75), "tc": (0.5, 0.75), "td": (0.75, 0.75), "te": (0.5, 0.75), "tf": (0.0, 0.0)}  # case 1 unranked
    assert _rates(runs, "case-00")["s@1"] == 0.5625 and runs["metrics"]["s@1"] == 0.28125
    with pytest.raises(ValueError):
        score(hand_corpus, CELL, None, "broadcast", 0, 0, 0)


def test_arm_known_answers_hop1_and_hop0(hand_corpus):
    res = score(hand_corpus, CELL, ARM_HOP1, "arm", 0, 100, 0)
    per = {t["trace_id"]: (t["hit_s"]["1"], t["hit_s"]["3"], t["hit_p"]["1"], t["hit_p"]["3"], t["exact"], t["set_size"], t["probed"]) for t in res["traces"]}
    assert per == {"ta": (1, 1, 0, 1, 0, 2, True), "tc": (1, 1, 0, 0, 1, 1, True), "td": (0, 0, 0, 0, 0, 0, False),
                   "te": (0, 0, 0, 0, 0, 1, True), "tf": (0, 1, 1, 1, 0, 3, True)}
    assert _rates(res, "case-00") == {"s@1": 0.5, "s@3": 0.5, "p@1": 0.0, "p@3": 0.25, "exact": 0.25}
    assert res["cases"][0]["mean_set_size"] == 1.0 and res["cases"][0]["n_unprobed"] == 1 and res["cases"][1]["n_unprobed"] == 0
    assert res["metrics"] == {"s@1": 0.25, "s@3": 0.75, "p@1": 0.5, "p@3": 0.625, "exact": 0.125, "mean_set_size": 2.0, "n_cases": 2, "n_traces": 5}
    assert res["counts"]["n_unprobed"] == 1 and res["counts"]["n_candidates"] == 5
    assert (_trace(res, "ta")["n_rc_reachable"], _trace(res, "tc")["n_rc_reachable"], _trace(res, "te")["n_rc_reachable"]) == (1, 0, 0)
    assert res["cases"][0]["random"] == {"s@1": pytest.approx(0.5), "s@3": 1.0, "p@1": 0.125, "p@3": 0.25}
    missing = {"services": SERVICES, "cases": {"case-00": {k: v for k, v in ARM_HOP1["cases"]["case-00"].items() if k != "te"}, "case-01": ARM_HOP1["cases"]["case-01"]}}
    assert score(hand_corpus, CELL, missing, "arm", 0, 0, 0)["counts"]["n_unprobed"] == 2
    hop0 = score(hand_corpus, CELL, ARM_HOP0, "arm", 1, 0, 0)
    assert _rates(hop0, "case-00") == {"s@1": 0.5, "s@3": 0.75, "p@1": 0.25, "p@3": 0.25, "exact": 0.25}
    assert _rates(hop0, "case-01") == {"s@1": 1.0, "s@3": 1.0, "p@1": 1.0, "p@3": 1.0, "exact": 1.0}
    assert hop0["metrics"] == {"s@1": 0.75, "s@3": 0.875, "p@1": 0.625, "p@3": 0.625, "exact": 0.625, "mean_set_size": 1.125, "n_cases": 2, "n_traces": 5}
    with pytest.raises(ValueError):
        score(hand_corpus, CELL, {"services": ["svc-c", "svc-b", "svc-a"], "cases": {}}, "arm", 0, 0, 0)
    with pytest.raises(ValueError):
        score(hand_corpus, CELL, ARM_HOP1, "window", 0, 0, 0)


def test_pool_concatenates_cases(hand_corpus):
    a = score(hand_corpus, CELL, None, "topology", 0, 100, 0)
    b = score(hand_corpus, CELL, ARM_HOP1, "arm", 0, 100, 0)
    pooled = pool([a, a], 100, 0)
    assert pooled["metrics"]["n_cases"] == 4 and pooled["metrics"]["n_traces"] == 10 and pooled["n_cases_per_result"] == [2, 2]
    assert {k: pooled["metrics"][k] for k in ("s@1", "s@3", "p@1", "p@3", "exact", "mean_set_size")} == {k: a["metrics"][k] for k in ("s@1", "s@3", "p@1", "p@3", "exact", "mean_set_size")}
    assert pooled["pooled_from"] == ["hand-corpus", "hand-corpus"] and pooled["mode"] == "topology" and pooled["hop0"] == 0 and pooled["axis"] == "trace"
    assert pooled["random"] == a["random"] and set(pooled["by_fault_kind"]) == {"crash", "degrade"} and "traces" not in pooled
    assert pooled["bootstrap"]["b"] == 100 and set(pooled["bootstrap"]["intervals"]) == set(a["bootstrap"]["intervals"])
    assert pooled["bootstrap"] == pool([a, a], 100, 0)["bootstrap"]  # seeded
    with pytest.raises(ValueError):
        pool([a, b], 100, 0)
    with pytest.raises(ValueError):
        pool([], 100, 0)


def test_cli_round_trip_including_ranking_trace_and_pool(hand_corpus, tmp_path):
    cell = tmp_path / "cell.json"
    cell.write_text(json.dumps(CELL))
    out = tmp_path / "topo.json"
    assert main(["--corpus", str(hand_corpus), "--cell", str(cell), "--mode", "topology", "--hop0", "0", "--bootstrap-b", "50", "--bootstrap-seed", "1", "--out", str(out)]) == 0
    r = read_json(out)
    assert r["metrics"]["s@1"] == 0.25 and r["bootstrap"]["b"] == 50 and len(r["traces"]) == 5 and r["c"] == 2 and r["max_len"] == 5
    pooled = tmp_path / "pooled.json"
    assert main(["--pool", str(out), str(out), "--bootstrap-b", "50", "--bootstrap-seed", "1", "--out", str(pooled)]) == 0
    assert read_json(pooled)["metrics"]["n_cases"] == 4
    # the harness's ranking-trace.json as --cell: mode arm takes the prediction from it at the chosen reading
    rt = {"schema": "ranking-trace@1", "arm": "rca/trace", "cell": "xs-latent-c2", "c": 2, "max_len": 5, "select": "post-nonok", "cap": 0,
          "device": "cpu", "services": SERVICES,
          "cases": {cd: {"selected": CELL["selected"][cd], "skipped": CELL["skipped"][cd],
                         "traces": {tid: {**ARM_HOP1["cases"][cd][tid], "s0": ARM_HOP0["cases"][cd][tid]["s"], "p0": ARM_HOP0["cases"][cd][tid]["p"]}
                                    for tid in ARM_HOP1["cases"][cd]}} for cd in CELL["selected"]}}
    rt_path = tmp_path / "ranking-trace.json"
    rt_path.write_text(json.dumps(rt))
    out1, out0 = tmp_path / "arm1.json", tmp_path / "arm0.json"
    assert main(["--corpus", str(hand_corpus), "--cell", str(rt_path), "--mode", "arm", "--hop0", "0", "--reading", "hop1", "--bootstrap-b", "0", "--bootstrap-seed", "0", "--out", str(out1)]) == 0
    assert main(["--corpus", str(hand_corpus), "--cell", str(rt_path), "--mode", "arm", "--hop0", "1", "--reading", "hop0", "--bootstrap-b", "0", "--bootstrap-seed", "0", "--out", str(out0)]) == 0
    assert read_json(out1)["metrics"] == score(hand_corpus, CELL, ARM_HOP1, "arm", 0, 0, 0)["metrics"]
    assert read_json(out0)["metrics"] == score(hand_corpus, CELL, ARM_HOP0, "arm", 1, 0, 0)["metrics"]
    # broadcast with --prediction; the ranking-trace header doubles as the cell
    pred = tmp_path / "broadcast.json"
    pred.write_text(json.dumps(BROADCAST))
    outb = tmp_path / "bc.json"
    assert main(["--corpus", str(hand_corpus), "--cell", str(rt_path), "--mode", "broadcast", "--hop0", "0", "--prediction", str(pred), "--bootstrap-b", "0", "--bootstrap-seed", "0", "--out", str(outb)]) == 0
    assert read_json(outb)["metrics"]["s@1"] == 0.125
    base = ["--corpus", str(hand_corpus), "--cell", str(rt_path), "--bootstrap-b", "0", "--bootstrap-seed", "0", "--out", str(tmp_path / "x.json")]
    for bad in (["--mode", "arm", "--hop0", "0", "--reading", "hop0"],  # the reading contradicts --hop0
                ["--mode", "arm", "--hop0", "0"],  # arm without a prediction source
                ["--mode", "topology", "--hop0", "0", "--reading", "hop1"],  # --reading outside mode arm
                ["--mode", "topology", "--hop0", "0", "--prediction", str(pred)],  # a prediction for a mode without one
                ["--mode", "random"],  # --hop0 is required
                ["--mode", "arm", "--hop0", "0", "--reading", "hop1", "--prediction", str(pred)]):
        with pytest.raises(SystemExit):
            main(base + bad)
    with pytest.raises(SystemExit):
        main(["--pool", str(out), "--corpus", str(hand_corpus), "--bootstrap-b", "0", "--bootstrap-seed", "0", "--out", str(tmp_path / "x.json")])
    with pytest.raises(SystemExit):
        main(["--bootstrap-b", "1", "--bootstrap-seed", "1", "--out", str(tmp_path / "x.json")])
    assert tuple(MODES) == ("arm", "topology", "random", "callees", "broadcast") and set(NONOK_IDS) == {OUTCOME_5XX, OUTCOME_ERR, OUTCOME_SLOW}
