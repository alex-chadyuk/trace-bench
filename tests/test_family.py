"""PRD scenarios 26 (a sampled family: distinct systems, disjoint train/test
split) and 27 (at least five seeds per named instance; headline numbers as
central tendency and dispersion)."""
import glob
from pathlib import Path

import yaml

from tracebench.config import load_family_config, load_instance_config
from tracebench.family import sample_family, write_family
from tracebench.instantiate import instantiate_from_paths
from xs_fixture import REPO


def test_family_systems_are_distinct_and_split_is_disjoint(tmp_path):
    fam = load_family_config(REPO / "configs" / "family" / "small.yaml")
    systems = sample_family(fam)
    manifest = write_family(fam, systems, tmp_path)
    assert len(systems) == fam.n_systems
    assert set(manifest["split"]["train"]).isdisjoint(manifest["split"]["test"])
    assert manifest["split"]["test"] and manifest["split"]["train"]
    hashes = {s["config_hash"] for s in systems}
    assert len(hashes) == len(systems)
    # two systems instantiate to different topologies and mechanism graphs
    a = instantiate_from_paths(tmp_path / "systems" / f"{systems[0]['name']}.yaml")
    b = instantiate_from_paths(tmp_path / "systems" / f"{systems[1]['name']}.yaml")
    assert a.topo.to_dict() != b.topo.to_dict()
    ga, gb = a.mechanism.graph_json(ctxmax=False), b.mechanism.graph_json(ctxmax=False)
    assert {(e["src"], e["dst"]) for e in ga["edges"]} != {(e["src"], e["dst"]) for e in gb["edges"]}
    # resampling with the same seed reproduces the family
    again = sample_family(fam)
    assert [s["config_hash"] for s in again] == [s["config_hash"] for s in systems]


def test_every_named_instance_ships_at_least_five_seeds():
    for path in sorted(glob.glob(str(REPO / "configs" / "instances" / "*.yaml"))):
        cfg = load_instance_config(path)
        assert len(cfg.run.seeds) >= 5, path
        assert cfg.run.twin is True, path
        assert cfg.constants.endswith("realism-v1.json"), path


def test_headline_reports_central_tendency_and_dispersion():
    from tracebench.score import headline
    per_seed = [{"directed": {"f1": 0.8}}, {"directed": {"f1": 0.9}}, {"directed": {"f1": 0.85}}, {"directed": {"f1": 0.8}}, {"directed": {"f1": 0.9}}]
    h = headline(per_seed, "directed.f1")
    assert h["n_seeds"] == 5 and abs(h["mean"] - 0.85) < 1e-9 and h["std"] > 0 and "p50" in h
