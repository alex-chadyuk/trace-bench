"""Instantiation: configuration + constants + seed -> a concrete system.

`instantiate()` builds names, topology, scenarios, latency thresholds and the
mechanism in a fixed order from the INSTANTIATE and LATENCY streams, and
`write_instantiation()` records everything a later, read-only derivation needs
(`instantiation.json`, `topology/callgraph.json`, `topology/prior.json`,
`slow-thresholds.json`). The mechanism graph itself is derived from these by
`graphs.py`, so scenario 6's "reproduces without consulting the raw feed"
holds for every graph artifact.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import InstanceConfig, load_instance_config
from .constants import (
    CALLGRAPH_JSON, CONFIG_YAML, CONSTANTS_JSON, INSTANTIATION_JSON, PRIOR_JSON,
    SLOW_THRESHOLDS_JSON, TOPOLOGY_DIR, Stream,
)
from .latency import LatencyModel, slow_thresholds_json
from .mechanism import Mechanism
from .naming import Namer
from .realism import RealismConstants, load_realism, resolve_constants_path
from .record import canonical_hash, write_json
from .rng import instantiation_generator, numpy_minor_version
from .scenarios import ScenarioSet, build_scenarios
from .topology import Topology, build_topology, callgraph_json, prior_json
from . import __version__


@dataclass
class Instantiation:
    cfg: InstanceConfig
    constants: RealismConstants
    seed: int
    topo: Topology
    sset: ScenarioSet
    latency: LatencyModel
    mechanism: Mechanism

    @property
    def config_hash(self):
        return canonical_hash(self.cfg.resolved())


def instantiate(cfg: InstanceConfig, constants: RealismConstants, seed: int) -> Instantiation:
    rng = instantiation_generator(seed, Stream.INSTANTIATE)
    namer = Namer(rng)
    topo = build_topology(cfg, constants, rng, namer, sum(s.steps for s in cfg.scenarios))
    sset = build_scenarios(cfg, topo, rng)
    latency = LatencyModel(topo, constants, cfg.slow.quantile, mc_samples=cfg.mechanism.mc_samples, cfg=cfg)
    latency.fit_thresholds(instantiation_generator(seed, Stream.LATENCY))
    mech = Mechanism(cfg, constants, topo, sset, latency, seed)
    return Instantiation(cfg, constants, seed, topo, sset, latency, mech)


def instantiate_from_paths(config_path, seed=None) -> Instantiation:
    cfg = load_instance_config(config_path)
    constants = load_realism(resolve_constants_path(config_path, cfg.constants))
    return instantiate(cfg, constants, cfg.run.seed if seed is None else int(seed))


def instantiation_record(inst: Instantiation):
    return {
        "schema": "tracebench/instantiation@1",
        "tool_version": __version__,
        "numpy_minor": numpy_minor_version(),
        "instance": inst.cfg.name,
        "seed": inst.seed,
        "config_hash": inst.config_hash,
        "constants_version": inst.constants.version,
        "topology": inst.topo.to_dict(),
        "scenarios": inst.sset.to_dict(),
        "latency": inst.latency.to_dict(),
        "counts": {
            "services": sum(1 for s in inst.topo.services if s.kind == 1),
            "ops": len(inst.topo.ops),
            "call_edges": len(inst.topo.edges),
            "mechanism_nodes": len(inst.mechanism.nodes),
            "mechanism_latent_nodes": len(inst.mechanism.latent_ids()),
        },
    }


def write_instantiation(inst: Instantiation, corpus_dir, derived_date):
    corpus_dir = Path(corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / CONFIG_YAML).write_text(
        __import__("tracebench.config", fromlist=["dump_config_yaml"]).dump_config_yaml(inst.cfg), encoding="utf-8")
    write_json(corpus_dir / CONSTANTS_JSON, inst.constants.as_dict())
    write_json(corpus_dir / INSTANTIATION_JSON, instantiation_record(inst))
    write_json(corpus_dir / TOPOLOGY_DIR / CALLGRAPH_JSON, callgraph_json(inst.topo, derived_date, inst.cfg.name))
    write_json(corpus_dir / TOPOLOGY_DIR / PRIOR_JSON, prior_json(inst.topo))
    write_json(corpus_dir / SLOW_THRESHOLDS_JSON, slow_thresholds_json(inst.latency, inst.topo))
    return corpus_dir


def load_instantiation(corpus_dir) -> Instantiation:
    """Rebuild the instantiation from a corpus directory without any RNG use
    beyond the recorded seed: configuration and constants are re-read from the
    corpus, so the rebuild is exact (asserted against the recorded hash)."""
    corpus_dir = Path(corpus_dir)
    from .config import load_instance_config as _load
    cfg = _load(corpus_dir / CONFIG_YAML)
    constants = RealismConstants(__import__("tracebench.record", fromlist=["read_json"]).read_json(corpus_dir / CONSTANTS_JSON),
                                 path=str(corpus_dir / CONSTANTS_JSON))
    rec = __import__("tracebench.record", fromlist=["read_json"]).read_json(corpus_dir / INSTANTIATION_JSON)
    inst = instantiate(cfg, constants, int(rec["seed"]))
    if inst.config_hash != rec["config_hash"]:
        raise ValueError("instantiation.json config_hash does not match the corpus config.yaml")
    if inst.topo.to_dict() != rec["topology"]:
        raise ValueError("re-instantiated topology differs from instantiation.json (numpy version drift?)")
    return inst
