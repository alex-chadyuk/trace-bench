"""PRD scenario 12: a named instance at a given seed regenerates
byte-identically on a different machine at the same tool and constants version.

`tests/fixtures/xs-checksums.json` is the committed pin: the per-file sha256 of
the xs corpora (latent and twin, seed 0) generated from the shipped
`configs/instances/xs.yaml`. Run locally, this test regenerates them and
compares. The cross-machine half of the scenario is the same comparison against
manifests produced elsewhere:

    TRACEBENCH_XS_MANIFEST_DIR=<dir of pulled manifest.json files> \\
        pytest tests/test_byte_identity.py

Freeze the fixture after any change that alters corpus bytes (a tool-version
bump included):

    python -m tracebench.manifest freeze --corpus <xs/latent/seed=0> \\
        --corpus <xs/twin/seed=0> --out tests/fixtures/xs-checksums.json
"""
import os
from pathlib import Path

import pytest

from tracebench.constants import MC_DECIMALS
from tracebench.manifest import corpus_key
from tracebench.record import read_json
from corpus_fixture import shipped_xs_pipeline
from xs_fixture import xs_instantiation

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "xs-checksums.json"
MANIFEST_DIR_ENV = "TRACEBENCH_XS_MANIFEST_DIR"


def _frozen():
    if not FIXTURE.exists():
        pytest.fail(f"{FIXTURE} is missing; freeze it with `python -m tracebench.manifest freeze` "
                    f"(see this module's docstring)")
    return read_json(FIXTURE)


def _manifests_from(directory):
    """Every manifest under a directory, keyed by the corpus identity it names —
    so the pulled layout (versioned prefix, rung, variant) need not match ours."""
    out = {}
    for p in sorted(Path(directory).rglob("manifest.json")):
        m = read_json(p)
        if m.get("schema", "").startswith("tracebench/manifest"):
            out[corpus_key(m)] = m
    return out


def _regenerated():
    out, _res, _calls = shipped_xs_pipeline()
    return _manifests_from(out)


def _compare(frozen, manifests, key):
    m = manifests[key]
    assert m["tool_version"] == frozen["tool_version"], f"{key}: tool version differs"
    assert m["constants_version"] == frozen["constants_version"], f"{key}: constants version differs"
    assert m["config_hash"] == frozen["corpora"][key]["config_hash"], f"{key}: configuration differs"
    expected = frozen["corpora"][key]["files"]
    actual = {f["path"]: f["sha256"] for f in m["files"]}
    changed = sorted(p for p in expected.keys() & actual.keys() if expected[p] != actual[p])
    assert not changed, f"{key}: {len(changed)} file(s) differ: {changed[:10]}"
    assert not sorted(expected.keys() - actual.keys()), f"{key}: files missing: {sorted(expected.keys() - actual.keys())[:10]}"
    assert not sorted(actual.keys() - expected.keys()), f"{key}: files not in the fixture: {sorted(actual.keys() - expected.keys())[:10]}"


def test_xs_seed0_matches_the_frozen_checksums():
    frozen = _frozen()
    pulled = os.environ.get(MANIFEST_DIR_ENV)
    manifests = _manifests_from(pulled) if pulled else _regenerated()
    assert manifests, f"no manifest found in {pulled}" if pulled else "no corpus regenerated"
    missing = sorted(set(frozen["corpora"]) - set(manifests))
    assert not missing, f"the fixture pins corpora that are not here: {missing}"
    if not pulled:
        # a local run produces exactly the pinned corpora and nothing else
        assert set(manifests) == set(frozen["corpora"])
    for key in sorted(frozen["corpora"]):
        _compare(frozen, manifests, key)


def test_the_fixture_pins_both_variants_of_seed_zero():
    frozen = _frozen()
    assert set(frozen["corpora"]) == {"xs/latent/seed=0", "xs/twin/seed=0"}
    assert all(len(c["files"]) > 20 and all(len(s) == 64 for s in c["files"].values())
               for c in frozen["corpora"].values())


def test_every_monte_carlo_derived_float_is_quantised():
    """The guard behind D-TB-15: a fitted float that reaches an artifact unrounded
    is not reproducible across CPU architectures (one SLOW threshold came out 1
    ULP apart on arm64 and x86-64 on 2026-09-12, while every data file stayed
    byte-identical). Rounding is what makes it reproducible, so a future change
    that adds an unrounded fitted quantity fails here rather than a rung later."""
    inst = xs_instantiation(0)
    thresholds = inst.latency.thresholds
    assert thresholds, "the xs instantiation fits no thresholds"
    unrounded = {k: v for k, v in thresholds.items() if v != round(v, MC_DECIMALS)}
    assert not unrounded, f"SLOW thresholds not quantised to {MC_DECIMALS} decimals: {unrounded}"

    p_calls = {i: e.p_call for i, e in enumerate(inst.topo.edges)}
    assert p_calls, "the xs topology has no call edges"
    unrounded = {k: v for k, v in p_calls.items() if v != round(v, MC_DECIMALS)}
    assert not unrounded, f"call probabilities not quantised to {MC_DECIMALS} decimals: {unrounded}"

    # and as serialised: the artifacts carry the rounded values, not a longer repr
    from tracebench.instantiate import instantiation_record
    from tracebench.latency import slow_thresholds_json
    rec = instantiation_record(inst)
    for key, value in rec["latency"]["thresholds_s"].items():
        assert value == round(value, MC_DECIMALS), f"instantiation.json latency.thresholds_s[{key}]"
    for row in slow_thresholds_json(inst.latency, inst.topo)["thresholds"]:
        assert row["threshold_s"] == round(row["threshold_s"], MC_DECIMALS), row["op_id"]
    for edge in rec["topology"]["edges"]:
        assert edge["p_call"] == round(edge["p_call"], MC_DECIMALS), edge


def test_the_corpus_carries_no_wall_clock_date():
    """The guard behind D-TB-18: `topology/callgraph.json` is dated by the
    simulated window's start, a function of the configuration, never by the day
    the corpus was generated — the latter made the same configuration and seed
    regenerate differently on any day but the freeze day."""
    import datetime as dt
    import yaml

    from xs_fixture import XS

    out, _res, _calls = shipped_xs_pipeline()
    window_start = yaml.safe_load(XS.read_text())["run"]["window"]["start"]
    for corpus in sorted(Path(out).glob("xs/*/seed=0")):
        cg = read_json(corpus / "topology" / "callgraph.json")
        assert cg["derived"] == window_start[:10], corpus
        assert cg["derived"] != dt.date.today().isoformat() or window_start[:10] == dt.date.today().isoformat()
