"""D-TB-21: the exposure registry is the whole truth about what a variant
observes, and the streaming JSON writer produces the bytes `json.dumps` did."""
import json

from tracebench.constants import (
    DERIVED_FROM, EXPOSURE_PROFILES, GENERATED_VARIANTS, LATENT_GROUPS, METRIC_FAMILIES, VARIANTS, exposed_groups,
    hidden_groups,
)
from tracebench.record import write_json
from xs_fixture import xs_instantiation


def test_latent_groups_are_exactly_the_mechanism_latent_groups():
    mech = xs_instantiation().mechanism
    groups = {mech.nodes[n].var.group for n in mech.latent_ids()}
    assert groups == set(LATENT_GROUPS)
    assert not any(mech.nodes[n].var.group in LATENT_GROUPS for n in mech.observable_ids())


def test_profiles_cover_every_variant_and_stay_within_the_latent_groups():
    assert set(EXPOSURE_PROFILES) == set(VARIANTS)
    for v in VARIANTS:
        assert exposed_groups(v) <= set(LATENT_GROUPS)
        assert exposed_groups(v) | hidden_groups(v) == set(LATENT_GROUPS)
    assert exposed_groups("latent") == set() and exposed_groups("twin") == set(LATENT_GROUPS)
    assert exposed_groups("metrics") == {"intensity", "load", "pool", "health"}
    assert "cache" in hidden_groups("metrics")
    assert set(METRIC_FAMILIES) == exposed_groups("metrics")
    assert set(DERIVED_FROM) == set(VARIANTS) - set(GENERATED_VARIANTS) and DERIVED_FROM["metrics"] == "latent"


def test_streaming_write_json_matches_dumps(tmp_path):
    obj = {"b": [1, 2.5, {"z": None, "a": "ü"}], "a": {"nested": [[], {}, "x\n"]}, "n": 10 ** 12}
    p = write_json(tmp_path / "x.json", obj)
    expected = json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
    assert p.read_text(encoding="utf-8") == expected
