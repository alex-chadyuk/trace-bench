"""PRD scenarios 12 (byte-identical regeneration), 24 (resume equals an
uninterrupted run), 22 (size cap refusal and override) and the twin's
identity with the latent instance."""
import json
from collections import defaultdict

import pyarrow.parquet as pq
import pytest

from tracebench.engine import H_CALL, H_HIT, Engine, SpillExceeded
from tracebench.generate import (
    CapExceeded, _n_ticks, _shard_bounds, generate, run_one_shard, spill_allowance,
)
from tracebench.hashing import D_HOP, uniforms
from tracebench.instantiate import instantiate
from tracebench.latents import Slots, compile_fault_forcings, initial_state, shard_checkpoints, simulate_latents
from tracebench.shards import completed_shards, read_records
from corpus_fixture import corpus_checksums, scratch_dir, write_variant_config, xs_corpus
from xs_fixture import xs_instantiation


def test_regeneration_is_byte_identical():
    a = xs_corpus("latent", True, "base")
    b = xs_corpus("latent", True, "again")
    ca, cb = corpus_checksums(a), corpus_checksums(b)
    assert ca.keys() == cb.keys()
    assert all(ca[k] == cb[k] for k in ca), [k for k in ca if ca[k] != cb[k]]
    assert any(k.startswith("raw/") for k in ca) and any(k.startswith("oracle/") for k in ca)
    assert (a / "COMPLETE").exists()


def test_invocation_composes_the_call_draw_with_the_cache():
    """Request by request the engine invokes exactly the ops an independent
    top-down evaluation of `caller invoked, call draw < p_call, and not (cache
    WARM and hit draw < p_hit)` gives. With every p_call = 1 that is the rule
    without call probabilities (a request traverses its reachable subtree less
    cache hits), which invokes more hops."""
    base = xs_instantiation()
    t1 = 600
    hop_counts = {}
    for label in ("calibrated", "p_call=1"):
        inst = instantiate(base.cfg, base.constants, 0)          # a private copy: its edges are edited below
        if label == "p_call=1":
            for e in inst.topo.edges:
                e.p_call = 1.0
        topo = inst.topo
        slots = Slots.build(topo)
        latents, _ = simulate_latents(inst, 0, slots, initial_state(slots), 0, t1 + spill_allowance(inst))
        res = Engine(inst, 0, slots).run_shard(0, 0, t1, latents)
        h, q = res.hops.cols, res.requests.cols
        engine_ops = defaultdict(set)
        for row, op, k in zip(h["request_row"], h["op"], h["attempt"]):
            if k == 0:
                engine_ops[int(row)].add(int(op))
        assert len(q["request_row"]) > 500
        for row, b, rg, tick in zip(q["request_row"], q["bff_op"], q["req_gid"], q["tick"]):
            invoked = {int(b)}
            for v in topo.reachable_from(int(b))[1:]:
                for e in topo.caller_edges(v):
                    if e.caller not in invoked or float(uniforms(0, D_HOP, int(rg), e.index, H_CALL)) >= e.p_call:
                        continue
                    warm = latents.cache[int(tick) - latents.t0, slots.svc_slot_of_op[e.caller]] == 0
                    if e.cached and warm and float(uniforms(0, D_HOP, int(rg), e.index, H_HIT)) < e.p_hit:
                        continue
                    invoked.add(v)
                    break
            assert invoked == engine_ops[int(row)], (label, int(row), invoked, engine_ops[int(row)])
        hop_counts[label] = sum(len(s) for s in engine_ops.values())
    assert hop_counts["p_call=1"] > hop_counts["calibrated"], hop_counts


def test_resume_equals_uninterrupted_run():
    full = xs_corpus("latent", True, "base")
    cfg_path = write_variant_config(None, "xs-resume")
    out = scratch_dir("xs-resume-out")
    partial = generate(cfg_path, 0, out, stop_after_shard=1)
    corpus = partial["corpus_dir"]
    assert completed_shards(corpus) == [0, 1] and not partial["complete"]
    resumed = generate(cfg_path, 0, out, resume=True)
    assert resumed["complete"] and completed_shards(corpus) == [0, 1, 2, 3]
    assert resumed["shards_written"] == [2, 3]
    cf, cr = corpus_checksums(full), corpus_checksums(corpus)
    assert cf == cr


def test_size_cap_refuses_before_writing_and_override_proceeds():
    cfg_path = write_variant_config(lambda d: d["run"].__setitem__("cap_gb", 0.001), "xs-cap")
    out = scratch_dir("xs-cap-out")
    with pytest.raises(CapExceeded) as ei:
        generate(cfg_path, 0, out)
    assert "exceeds the cap" in str(ei.value)
    corpus = out / "xs" / "latent" / "seed=0"
    assert not (corpus / "raw").exists() and not (corpus / "graphs").exists()
    est = json.loads((corpus / "run" / "estimate.json").read_text())
    assert est["estimate_gb"] > est["cap_gb"]
    res = generate(cfg_path, 0, out, override_cap=True, stop_after_shard=0)
    assert completed_shards(res["corpus_dir"]) == [0]


def test_twin_is_the_same_simulation_with_latents_exposed():
    latent = xs_corpus("latent", True, "base")
    twin = xs_corpus("twin", True, "base")
    s_l = pq.read_table(latent / "oracle" / "shard=0000" / "spans.parquet").to_pydict()
    s_t = pq.read_table(twin / "oracle" / "shard=0000" / "spans.parquet").to_pydict()
    assert s_l == s_t
    recs_l = list(read_records(latent / "raw" / "shard=0000" / "vl.jsonl.gz"))
    recs_t = list(read_records(twin / "raw" / "shard=0000" / "vl.jsonl.gz"))
    assert len(recs_l) == len(recs_t)
    stripped = [{k: v for k, v in r.items() if not k.startswith("state_")} for r in recs_t]
    assert stripped == recs_l
    access = [r for r in recs_t if r["kind"] == "vl.access" and r.get("trace_id")]
    assert all("state_health" in r and "state_pool" in r for r in access)
    assert (twin / "raw" / "shard=0000" / "state.jsonl.gz").exists()
    assert not any(k.startswith("state_") for r in recs_l for k in r)


# --- D-TB-16: the spill allowance is a heuristic, so overspill must be recoverable ---------
def _shard_zero_inputs(inst, seed=0):
    slots = Slots.build(inst.topo)
    forcings, _ = compile_fault_forcings(inst, slots)
    checkpoints = shard_checkpoints(inst, seed, slots, inst.cfg.run.shard_ticks, _n_ticks(inst.cfg), forcings)
    shard, t0, t1 = _shard_bounds(inst.cfg)[0]
    return slots, forcings, checkpoints[shard], shard, t0, t1


def test_a_journey_past_the_latent_slice_raises_a_typed_actionable_error():
    """The failure the s rung hit at seeds 1 and 4 (2026-09-12) surfaced as
    `IndexError: index 2872 is out of bounds for axis 0 with size 2839` from
    inside numpy, which named neither the cause nor the remedy. It must name
    both, and say how many ticks would have sufficed."""
    inst = xs_instantiation(0)
    slots, forcings, checkpoint, shard, t0, t1 = _shard_zero_inputs(inst)
    # one tick of headroom past the slice: journeys certainly outrun it
    latents, _ = simulate_latents(inst, 0, slots, checkpoint, t0, t1 + 1, forcings)
    with pytest.raises(SpillExceeded) as ei:
        Engine(inst, 0, slots, forcings).run_shard(shard, t0, t1, latents)
    e = ei.value
    assert e.needed_ticks > e.held_ticks == (t1 - t0) + 1
    assert "journey needs latent tick" in str(e) and str(e.needed_ticks) in str(e)


def test_a_retried_shard_is_byte_identical_to_one_with_enough_allowance():
    """The fix rests on one property: re-simulating a taller slice does not
    change the latent values at the ticks the short slice already held (the
    uniforms are keyed by the ABSOLUTE tick and the chain advances from the
    recorded checkpoint). So a shard that had to retry must be byte-identical
    to a shard that never did."""
    inst = xs_instantiation(0)
    slots, forcings, checkpoint, shard, t0, t1 = _shard_zero_inputs(inst)

    generous = scratch_dir("spill-generous")
    retried = scratch_dir("spill-retried")
    ok = run_one_shard(inst, generous, "latent", 0, shard, t0, t1, checkpoint, forcings, spill_allowance(inst))
    # a 1-tick allowance forces the engine to raise and run_one_shard to re-simulate
    fixed = run_one_shard(inst, retried, "latent", 0, shard, t0, t1, checkpoint, forcings, 1)
    assert ok["n_records"] == fixed["n_records"] and ok["n_spans"] == fixed["n_spans"]

    a, b = corpus_checksums(generous), corpus_checksums(retried)
    assert a and a.keys() == b.keys()
    assert all(a[k] == b[k] for k in a), [k for k in a if a[k] != b[k]]


def test_overspill_is_refused_with_the_knob_named_once_the_retries_run_out():
    """The retries are bounded: when they run out the run refuses and names what
    to raise, rather than looping. (Each retry does make progress in practice —
    a taller slice reveals journeys the truncated one never reached, so the
    needed size grows — which is why the bound is what has to be tested.)"""
    import tracebench.generate as gen
    inst = xs_instantiation(0)
    slots, forcings, checkpoint, shard, t0, t1 = _shard_zero_inputs(inst)
    out = scratch_dir("spill-refused")
    original = gen.SPILL_RETRY_MAX
    gen.SPILL_RETRY_MAX = 1             # a single attempt, which a 1-tick allowance cannot survive
    try:
        with pytest.raises(RuntimeError, match="raise the spill allowance") as ei:
            run_one_shard(inst, out, "latent", 0, shard, t0, t1, checkpoint, forcings, 1)
        assert isinstance(ei.value.__cause__, SpillExceeded)
        assert "needed" in str(ei.value)
    finally:
        gen.SPILL_RETRY_MAX = original


def test_the_real_allowance_needs_no_retry_at_xs():
    """The retry is a safety net, not the normal path: at the shipped allowance
    the xs rung must not need it (the s rung needed 6 and 33 ticks more than its
    1,939-tick allowance, which one retry with the 600-tick margin covers)."""
    import tracebench.generate as gen
    inst = xs_instantiation(0)
    slots, forcings, checkpoint, shard, t0, t1 = _shard_zero_inputs(inst)
    latents, _ = simulate_latents(inst, 0, slots, checkpoint, t0, t1 + spill_allowance(inst), forcings)
    res = Engine(inst, 0, slots, forcings).run_shard(shard, t0, t1, latents)   # must not raise
    assert res.spill_ticks <= spill_allowance(inst)
    assert gen.SPILL_RETRY_MARGIN > 0 and gen.SPILL_RETRY_MAX >= 2
