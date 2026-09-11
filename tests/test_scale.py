"""PRD scenario 17: the top rung's event-type alphabet exceeds 8,000, and the
run record states each rung's alphabet size and node count. Instantiation
only — no corpus is generated here."""
from pathlib import Path

import pytest

from tracebench.estimate import alphabet_estimate
from tracebench.instantiate import instantiate_from_paths
from xs_fixture import xs_instantiation

REPO = Path(__file__).resolve().parents[1]


def test_xs_alphabet_estimate_is_small():
    est = alphabet_estimate(xs_instantiation())
    assert 0 < est["expected_realized_alphabet"] <= est["potential_alphabet"] <= 5 * est["n_ops"]
    assert est["expected_sessions"] > 0


@pytest.mark.slow
def test_top_rung_alphabet_exceeds_8000():
    inst = instantiate_from_paths(REPO / "configs" / "instances" / "xl.yaml", seed=0)
    est = alphabet_estimate(inst)
    assert est["potential_alphabet"] > 8000
    assert est["expected_realized_alphabet"] > 8000, est
    assert len(inst.mechanism.nodes) > 8000
