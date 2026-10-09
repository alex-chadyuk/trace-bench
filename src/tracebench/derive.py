"""Derive a partially observable variant from a complete latent corpus (D-TB-21).

    python -m tracebench.derive --from <latent corpus dir> --variant metrics --out <root> [--resume]

The metrics variant is the latent corpus plus a metrics channel, not a new
simulation: the raw feed, the correlated views, the oracle linkage, the labels,
the reports, the topology and the instantiation are the source's files,
hard-linked (copied across filesystems) and byte-identical; the tick-level
latent chain is recomputed from `instantiation.json` and the seed — it depends
on nothing the requests do — and sampled onto the scrape grid (`metrics.py`);
the ground truth (`graphs/`) is re-derived under the variant's exposure profile.
The derived corpus is complete on its own: it verifies against its own manifest,
which also records where it came from (`derived_from`).

Order: validate the source → link its files → write the ground truth (before
any new data, as `generate` does) → write the metrics channel and the oracle
state log → manifest → COMPLETE. `--resume` keeps the metrics shards already
written; everything else is cheap and redone.

Two tool stamps: `instantiation.json` is the source's file and names the tool
that instantiated the system; `manifest.json` names the tool that derived the
corpus. `load_instantiation` re-instantiates and asserts the configuration
hash and the topology, which is the guard that the deriving tool did not move
the simulation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .constants import (
    COMPLETE_MARKER, DERIVED_FROM, GRAPHS_DIR, MANIFEST_JSON, METRICS_DIR, ORACLE_STATE_DIR, RUN_DIR, VARIANTS,
)
from .generate import corpus_dir_for
from .graphs import write_graph_artifacts
from .instantiate import load_instantiation
from .log import log
from .manifest import corpus_key, finalize_corpus, verify_manifest
from .metrics import derive_metrics
from .record import RunRecord, link_or_copy, read_json, sha256_file

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 3
# the source's files a derived corpus does not take over: it writes its own
NOT_LINKED_PREFIXES = (GRAPHS_DIR + "/", RUN_DIR + "/", METRICS_DIR + "/", ORACLE_STATE_DIR + "/")
NOT_LINKED_NAMES = (MANIFEST_JSON, COMPLETE_MARKER, "artifacts.json")
MIN_SOURCE_TOOL_VERSION = (0, 3, 0)


class DeriveRefused(ValueError):
    pass


def _version_tuple(v):
    try:
        return tuple(int(x) for x in str(v).split("."))
    except ValueError:
        return ()


def validate_source(src, variant):
    """The source must be a complete, self-consistent corpus of the variant the
    requested one derives from, made by a tool that produced the data this
    tool reads (0.3.0 or later). Returns its manifest."""
    src = Path(src)
    if variant not in DERIVED_FROM:
        raise DeriveRefused(f"variant {variant!r} is not derived; derivable: {sorted(DERIVED_FROM)} (all: {VARIANTS})")
    if not (src / COMPLETE_MARKER).exists():
        raise DeriveRefused(f"{src} is not complete (no {COMPLETE_MARKER})")
    if not (src / MANIFEST_JSON).exists():
        raise DeriveRefused(f"{src} has no {MANIFEST_JSON}")
    m = read_json(src / MANIFEST_JSON)
    want = DERIVED_FROM[variant]
    if m.get("variant") != want:
        raise DeriveRefused(f"{src} is a {m.get('variant')!r} corpus; the {variant} variant derives from a {want!r} one")
    if _version_tuple(m.get("tool_version")) < MIN_SOURCE_TOOL_VERSION:
        raise DeriveRefused(f"{src} was made by tool {m.get('tool_version')!r}; "
                            f"{'.'.join(map(str, MIN_SOURCE_TOOL_VERSION))} or later is required")
    problems = verify_manifest(src)
    if problems:
        raise DeriveRefused(f"{src} does not verify against its manifest ({len(problems)} problems): {problems[:5]}")
    return m


def link_source(src, dst):
    """Every source file except what the derived corpus writes itself."""
    src, dst = Path(src), Path(dst)
    counts = {"linked": 0, "copied": 0, "kept": 0}
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(src).as_posix()
        if rel in NOT_LINKED_NAMES or any(rel.startswith(x) for x in NOT_LINKED_PREFIXES):
            continue
        target = dst / rel
        if target.exists():
            if target.stat().st_size == p.stat().st_size and (target.samefile(p) or sha256_file(target) == sha256_file(p)):
                counts["kept"] += 1
                continue
            target.unlink()
        counts[link_or_copy(p, target)] += 1
    return counts


def derive(src, variant, out, resume=False):
    src = Path(src).resolve()
    source_manifest = validate_source(src, variant)
    instance, seed = source_manifest["instance"], int(source_manifest["seed"])
    corpus_dir = corpus_dir_for(out, instance, variant, seed)
    if (corpus_dir / COMPLETE_MARKER).exists():
        raise DeriveRefused(f"{corpus_dir} is already complete; remove it to derive again")
    corpus_dir.mkdir(parents=True, exist_ok=True)
    rec = RunRecord(corpus_dir, "derive", {"from": str(src), "variant": variant, "out": str(out), "resume": resume})
    derived_from = {"corpus_key": corpus_key(source_manifest), "variant": source_manifest["variant"],
                    "config_hash": source_manifest["config_hash"], "tool_version": source_manifest["tool_version"],
                    "constants_version": source_manifest["constants_version"],
                    "manifest_sha256": sha256_file(src / MANIFEST_JSON)}
    # 1. the source's files, byte-identical
    counts = link_source(src, corpus_dir)
    derived_from.update({"n_files_linked": counts["linked"] + counts["kept"], "n_files_copied": counts["copied"]})
    log({"event": "derive_link", "corpus": str(corpus_dir), **counts})
    # 2. the same system (the hash and topology guard), then the ground truth under the variant's exposure
    inst = load_instantiation(corpus_dir)
    graphs_summary = write_graph_artifacts(inst, corpus_dir, variant)
    log({"event": "graphs", **graphs_summary})
    rec.note(graphs_written_before_metrics=True)
    # 3. the metrics channel and the oracle state log
    sampling = derive_metrics(corpus_dir, inst, seed, variant, resume=resume)
    log({"event": "metrics", "n_series": sampling["n_series"], "n_samples": sampling["n_samples"],
         "shards_written": len(sampling["shards_written"]),
         "share_invisible": {g: round(v["share_invisible_at_scrape"], 4) for g, v in sampling["per_group"].items() if v["exposed"]}})
    # 4. manifest and completion
    manifest = finalize_corpus(corpus_dir, f"derived from {derived_from['corpus_key']}; metrics written", derived_from=derived_from)
    results = {"instance": instance, "variant": variant, "seed": seed, "corpus_dir": str(corpus_dir),
               "derived_from": derived_from, "link_counts": counts, "graphs": graphs_summary,
               "metrics": {k: v for k, v in sampling.items() if k != "shards_written"},
               "manifest": {"files": len(manifest["files"])}, "complete": True}
    rec.finish(results)
    return results


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="source", required=True, help="a complete corpus of the variant this one derives from")
    p.add_argument("--variant", required=True, choices=sorted(DERIVED_FROM))
    p.add_argument("--out", required=True, help="corpus root; the result lands at <out>/<instance>/<variant>/seed=<k>")
    p.add_argument("--resume", action="store_true", help="keep the metrics shards already written")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        res = derive(args.source, args.variant, args.out, resume=args.resume)
    except DeriveRefused as e:
        log({"event": "derive_refused", "reason": str(e)})
        return EXIT_REFUSED
    log({"event": "derive", "corpus_dir": res["corpus_dir"], "n_series": res["metrics"]["n_series"],
         "n_files": res["manifest"]["files"]})
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
