"""Sample a family of whole systems (PRD scenario 26).

    python -m tracebench.family --config configs/family/<name>.yaml --out <dir> [--generate] [--seed N]

Each sampled system gets its own instance configuration (counts, fan-out,
scenario lengths drawn from the family's ranges; everything else from the
template), its own topology, mechanism graph and corpus. A declared
train/test split partitions systems, so no test system's topology or graph
appears in the training portion. The family manifest records the split and
every system's configuration hash.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import yaml

from .config import dump_config_yaml, load_family_config, parse_instance_config
from .constants import Stream
from .log import log
from .record import RunRecord, canonical_hash, write_json
from .rng import generator

SCENARIO_NAMES = ["browse", "search", "detail", "cart", "checkout", "account", "support", "settings", "history",
                  "invoice", "upgrade", "cancel", "notify", "export", "import", "report", "auditlog", "billing", "refer", "promo"]


def _draw_int(rng, r):
    return int(rng.integers(r.lo, r.hi + 1))


def sample_family(fam, seed=None):
    seed = fam.seed if seed is None else int(seed)
    rng = generator(seed, 2, 0, Stream.INSTANTIATE)
    systems = []
    n_test = max(1, int(round(fam.n_systems * fam.split.test_fraction)))
    perm = rng.permutation(fam.n_systems)
    test_set = set(int(i) for i in perm[:n_test])
    for k in range(fam.n_systems):
        d = copy.deepcopy(fam.template)
        counts = {"clients": _draw_int(rng, fam.ranges.clients), "services": _draw_int(rng, fam.ranges.services),
                  "endpoints_per_service": _draw_int(rng, fam.ranges.endpoints_per_service),
                  "bff_endpoints": _draw_int(rng, fam.ranges.bff_endpoints), "scenarios": _draw_int(rng, fam.ranges.scenarios)}
        d["counts"] = counts
        d["name"] = f"{fam.name}-{k:03d}"
        d.setdefault("topology", {})["fanout_mean"] = float(rng.uniform(fam.ranges.fanout_mean.lo, fam.ranges.fanout_mean.hi))
        scen_template = d.get("scenarios", [{}])[0] if d.get("scenarios") else {}
        scenarios = []
        for i in range(counts["scenarios"]):
            sc = copy.deepcopy(scen_template)
            sc["name"] = SCENARIO_NAMES[i % len(SCENARIO_NAMES)] + ("" if i < len(SCENARIO_NAMES) else str(i))
            sc["steps"] = min(counts["bff_endpoints"], _draw_int(rng, fam.ranges.steps))
            sc["weight"] = float(rng.uniform(0.5, 3.0))
            scenarios.append(sc)
        d["scenarios"] = scenarios
        d.setdefault("run", {})["seed"] = int(rng.integers(0, 2 ** 31 - 1))
        d["run"]["seeds"] = sorted({d["run"]["seed"]} | {int(x) for x in rng.integers(0, 2 ** 31 - 1, size=4)})
        if len(d["run"]["seeds"]) < 5:
            d["run"]["seeds"] = sorted(set(d["run"]["seeds"]) | {d["run"]["seed"] + i + 1 for i in range(5)})[:5]
            if d["run"]["seed"] not in d["run"]["seeds"]:
                d["run"]["seeds"][0] = d["run"]["seed"]
        cfg = parse_instance_config(d)
        systems.append({"index": k, "name": cfg.name, "split": "test" if k in test_set else "train",
                        "config": cfg, "config_hash": canonical_hash(cfg.resolved())})
    return systems


def write_family(fam, systems, out):
    out = Path(out)
    (out / "systems").mkdir(parents=True, exist_ok=True)
    for s in systems:
        (out / "systems" / f"{s['name']}.yaml").write_text(dump_config_yaml(s["config"]))
    manifest = {"schema": "tracebench/family@1", "family": fam.name, "n_systems": len(systems), "seed": fam.seed,
                "split": {"test_fraction": fam.split.test_fraction,
                          "train": [s["name"] for s in systems if s["split"] == "train"],
                          "test": [s["name"] for s in systems if s["split"] == "test"]},
                "systems": [{"name": s["name"], "split": s["split"], "config_hash": s["config_hash"],
                             "config": f"systems/{s['name']}.yaml", "seed": s["config"].run.seed} for s in systems]}
    write_json(out / "family-manifest.json", manifest)
    return manifest


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--generate", action="store_true", help="also generate every system's corpus under <out>/corpora")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    fam = load_family_config(args.config)
    rec = RunRecord(args.out, "family", vars(args))
    systems = sample_family(fam, args.seed)
    manifest = write_family(fam, systems, args.out)
    results = {"n_systems": len(systems), "n_test": len(manifest["split"]["test"])}
    if args.generate:
        from .generate import generate
        for s in systems:
            res = generate(Path(args.out) / "systems" / f"{s['name']}.yaml", s["config"].run.seed, Path(args.out) / "corpora",
                           twin=s["config"].run.twin)
            log({"event": "family_system", "name": s["name"], "complete": res["complete"]})
    rec.finish(results)
    log({"event": "family", **results})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
