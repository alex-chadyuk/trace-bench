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

from tracebench.manifest import corpus_key
from tracebench.record import read_json
from corpus_fixture import shipped_xs_pipeline

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
