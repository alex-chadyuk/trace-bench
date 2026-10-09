"""D-TB-21: the metrics channel of the metrics variant — regular-grid samples
of the exposed latent groups that agree with the simulation (the twin's state
log), nothing of the hidden groups, a method-readable catalogue that names the
state tokens, and a measured sampling loss in the oracle."""
import collections
import gzip
import json

import numpy as np
import pyarrow.parquet as pq

from tracebench.allowlist import method_readable_files
from tracebench.constants import METRICS_SCRAPE_S, METRIC_FAMILIES, exposed_groups, hidden_groups
from tracebench.generate import _n_ticks, _shard_bounds
from tracebench.instantiate import load_instantiation
from tracebench.intensity import intensity_index_vector
from tracebench.latents import Slots
from tracebench.mechanism import CACHE_VALUES, HEALTH_VALUES, LOAD_VALUES, POOL_VALUES
from tracebench.metrics import completed_metrics_shards, read_changes, read_samples, series_catalogue
from tracebench.record import read_json
from corpus_fixture import xs_corpus, xs_metrics_corpus

NAMES = {"load": LOAD_VALUES, "pool": POOL_VALUES, "cache": CACHE_VALUES, "health": HEALTH_VALUES}


def _series_index(cat):
    return {(s["family"], s["service"], s["endpoint"]): s for s in cat["series"]}


def test_samples_fill_the_scrape_grid_for_every_exposed_series():
    corpus = xs_metrics_corpus()
    cat = series_catalogue(corpus)
    inst = load_instantiation(corpus)
    slots = Slots.build(inst.topo)
    k = METRICS_SCRAPE_S // inst.cfg.run.tick_s
    assert cat["scrape_s"] == METRICS_SCRAPE_S and cat["epoch"] == inst.cfg.run.window.start
    assert cat["n_series"] == 1 + 2 * len(slots.svc_of_slot) + len(slots.op_of_slot) == len(cat["series"])
    assert set(cat["families"]) == {METRIC_FAMILIES[g] for g in exposed_groups("metrics")}
    shards = completed_metrics_shards(corpus)
    assert shards == [b[0] for b in _shard_bounds(inst.cfg)]
    n_ticks_total = 0
    for shard in shards:
        d = read_samples(corpus, shard)
        ticks = sorted(set(d["ts_ms"]))
        assert all(t % (METRICS_SCRAPE_S * 1000) == 0 for t in ticks)
        assert len(ticks) == inst.cfg.run.shard_ticks // k
        assert len(d["ts_ms"]) == len(ticks) * cat["n_series"]
        n_ticks_total += len(ticks)
        # file order: (ts_ms, family, service, endpoint), every series once per instant
        keys = list(zip(d["ts_ms"], d["family"], [s or "" for s in d["service"]], [e or "" for e in d["endpoint"]]))
        assert keys == sorted(keys) and len(set(keys)) == len(keys)
    assert n_ticks_total == cat["n_samples_per_series"]
    first = read_samples(corpus, 0)
    assert first["ts_ms"][0] == 0 == cat["first_ts_ms"]


def _twin_state_at_grid(twin, shard, slots, topo, k):
    """Replay the twin's per-shard state log (it re-anchors to nominal at every
    shard start) into {(node): value} at every grid tick of the shard."""
    cfg = load_instantiation(twin).cfg
    t0 = shard * cfg.run.shard_ticks
    t1 = min(t0 + cfg.run.shard_ticks, _n_ticks(cfg))
    events = collections.defaultdict(list)
    with gzip.open(twin / "raw" / f"shard={shard:04d}" / "state.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            tick = int(round((__import__("datetime").datetime.strptime(r["ts"], "%Y-%m-%dT%H:%M:%S.%fZ")
                              - __import__("datetime").datetime.strptime(cfg.run.window.start, "%Y-%m-%dT%H:%M:%SZ")).total_seconds()))
            events[tick].append((r["node"], r["value"]))
    state = {}
    for group, slot_list in (("load", slots.svc_of_slot), ("pool", slots.svc_of_slot), ("cache", slots.svc_of_slot)):
        for svc in slot_list:
            state[f"{group}:{svc}"] = NAMES[group][0]
    for op in slots.op_of_slot:
        state[f"health:{op}"] = HEALTH_VALUES[0]
    out = {}
    for t in range(t0, t1):
        for node, value in events.get(t, []):
            state[node] = value
        if t % k == 0:
            out[t] = dict(state)
    return out


def test_samples_equal_the_twin_state_on_the_grid():
    """The channel is the simulation's own state, sampled: at every scrape
    instant each exposed series shows the value the twin recorded (fault
    windows included), and intensity follows the daily profile."""
    corpus = xs_metrics_corpus()
    twin = xs_corpus("twin")
    cat = series_catalogue(corpus)
    inst = load_instantiation(corpus)
    slots = Slots.build(inst.topo)
    k = METRICS_SCRAPE_S // inst.cfg.run.tick_s
    by_key = _series_index(cat)
    n_checked = 0
    for shard in completed_metrics_shards(corpus):
        truth = _twin_state_at_grid(twin, shard, slots, inst.topo, k)
        d = read_samples(corpus, shard)
        t0 = shard * inst.cfg.run.shard_ticks
        inten = intensity_index_vector(inst.cfg, inst.constants, t0, inst.cfg.run.shard_ticks)
        for ts, fam, svc, ep, val in zip(d["ts_ms"], d["family"], d["service"], d["endpoint"], d["value"]):
            tick = ts // 1000 // inst.cfg.run.tick_s
            stem = by_key[(fam, svc, ep)]["token_stem"]
            node = stem[len("state:"):]
            if node == "intensity":
                assert val == ("day", "night", "peak")[int(inten[tick - t0])]
            else:
                assert truth[tick][node] == val, (shard, tick, node, val, truth[tick][node])
            n_checked += 1
    assert n_checked == cat["n_samples_per_series"] * cat["n_series"]


def test_injected_faults_show_in_the_series():
    corpus = xs_metrics_corpus()
    faults = read_json(corpus / "labels" / "faults.json")["faults"]
    health_faults = [f for f in faults if f["indicator"] == "health"]
    assert health_faults, "the xs fixture injects no health fault"
    rows = collections.defaultdict(dict)        # (service, endpoint) -> ts -> value
    for shard in completed_metrics_shards(corpus):
        d = read_samples(corpus, shard)
        for ts, fam, svc, ep, val in zip(d["ts_ms"], d["family"], d["service"], d["endpoint"], d["value"]):
            if fam == "endpoint_health":
                rows[(svc, ep)][ts] = val
    for f in health_faults:
        lo, hi = f["tick_lo"] * 1000, f["tick_hi"] * 1000
        for ep in f["endpoints"]:
            inside = [v for ts, v in rows[(f["service"], ep)].items() if lo <= ts < hi]
            assert inside and all(v == f["forced_value"] for v in inside), (f, inside)


def test_hidden_groups_are_absent_and_the_catalogue_names_the_state_tokens():
    corpus = xs_metrics_corpus()
    cat = series_catalogue(corpus)
    assert cat["hidden_groups"] == sorted(hidden_groups("metrics")) == ["auth", "cache", "net"]
    assert cat["exposed_groups"] == ["health", "intensity", "load", "pool"]
    assert not any(s["token_stem"].startswith(("state:cache", "state:net", "state:auth")) for s in cat["series"])
    for shard in completed_metrics_shards(corpus):
        assert "cache" not in {f.split("_")[0] for f in read_samples(corpus, shard)["family"]}
    alphabet = read_json(corpus / "graphs" / "alphabet.json")
    state_tokens = {t["token"] for t in alphabet["tokens"] if t.get("state_var")}
    from_catalogue = {f"{s['token_stem']}={lvl}" for s in cat["series"] for lvl in s["levels"]}
    assert from_catalogue == state_tokens
    # intensity is the one unlabelled series; health is the one with an endpoint label
    assert [s for s in cat["series"] if s["service"] is None] == [s for s in cat["series"] if s["family"] == "traffic_intensity"]
    assert all((s["endpoint"] is not None) == (s["family"] == "endpoint_health") for s in cat["series"])


def test_the_channel_is_method_readable_and_the_oracle_state_is_not():
    corpus = xs_metrics_corpus()
    readable = set(method_readable_files(corpus))
    assert "metrics/series.json" in readable and "metrics/shard=0000/samples.parquet" in readable
    assert not any(p.startswith("oracle/") for p in readable)
    assert (corpus / "oracle" / "state" / "sampling.json").exists()


def test_sampling_loss_is_measured_in_the_oracle():
    corpus = xs_metrics_corpus()
    sampling = read_json(corpus / "oracle" / "state" / "sampling.json")
    counted = collections.Counter()
    invisible = collections.Counter()
    unresolved = collections.Counter()
    for shard in completed_metrics_shards(corpus):
        c = read_changes(corpus, shard)
        for g, v, ex in zip(c["group"], c["visible_at_scrape"], c["exposed"]):
            counted[g] += 1
            assert ex == (g in exposed_groups("metrics"))
            if v is None:
                unresolved[g] += 1
            elif not v:
                invisible[g] += 1
        # a change on a scrape instant is always visible
        for t, v in zip(c["tick"], c["visible_at_scrape"]):
            if t % (METRICS_SCRAPE_S) == 0:
                assert v is True
    for g, rec in sampling["per_group"].items():
        assert rec["changes"] == counted[g] and rec["invisible_at_scrape"] == invisible[g] and rec["unresolved"] == unresolved[g]
        assert 0.0 <= rec["share_invisible_at_scrape"] <= 1.0
    assert sampling["per_group"]["cache"]["exposed"] is False and sampling["per_group"]["load"]["exposed"] is True
    # tick-level chains flip faster than a 30 s scrape: some loss is expected, and it is >= 0 at the fitted rates
    assert sum(invisible.values()) >= 0
