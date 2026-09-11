"""Publish a frozen, citable release of one or more corpora to the public dataset host.

    python -m tracebench.publish --corpus <dir> [--corpus <dir> ...] --version vX.Y.Z \\
        [--repo chadyuk/trace-bench] [--dry-run]

Layout on the dataset host: `instances/<name>/<variant>/seed=<s>/...` (each
corpus with its manifest), a top-level `release.json` index, a dataset card
(README.md) and the CC BY 4.0 licence. A version is a tag on the dataset
repository; publishing a version whose tag already exists is refused (PRD
scenario 14). Every corpus must be complete and carry a verified manifest.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from . import __version__
from .constants import COMPLETE_MARKER, MANIFEST_JSON
from .log import log
from .manifest import verify_manifest
from .record import read_json, write_json

DEFAULT_REPO = "chadyuk/trace-bench"
CARD = """---
license: cc-by-4.0
pretty_name: trace-bench
tags: [causal-discovery, root-cause-analysis, microservices, synthetic, benchmark]
---

# trace-bench {version}

A simulated microservice trace benchmark with ground-truth causal graphs.
Each named instance ships a raw heterogeneous log feed (no parent pointers),
the correlated sequence views, the mechanism graph that generated the data
(latent state variables flagged), its latent projection over
`(operation, outcome)` tokens as the scoring target, the deployment call
topology as a candidate structural prior, injected-fault labels and the
oracle linkage that makes correlation loss a measured quantity. Every corpus
is wholly synthetic and reproduces byte-identically from its configuration,
seed, tool version and constants version (see each `manifest.json`).

Generator (MIT): https://github.com/alex-chadyuk/trace-bench — tool version {tool_version}.
Corpus licence: CC BY 4.0. Fitted realism constants: {constants}.

| instance | variant | seed | alphabet (realized, train) | mechanism nodes | directed / bidirected at floor |
|---|---|---|---|---|---|
{rows}

To score a method: read only `raw/` and `views/`; score against
`graphs/scoring-target.json` (request grain) with `python -m tracebench.score`.
"""


def _row(m):
    tc = m.get("target_counts") or {}
    return f"| {m['instance']} | {m['variant']} | {m['seed']} | {m.get('alphabet_size_realized_train')} | {m.get('n_nodes_mechanism')} | {tc.get('n_directed_at_floor')} / {tc.get('n_bidirected_at_floor')} |"


def plan_release(corpora, version):
    entries = []
    for c in corpora:
        c = Path(c)
        if not (c / COMPLETE_MARKER).exists():
            raise ValueError(f"{c} is not complete")
        if not (c / MANIFEST_JSON).exists():
            raise ValueError(f"{c} has no manifest; run `python -m tracebench.manifest write`")
        problems = verify_manifest(c)
        if problems:
            raise ValueError(f"{c}: manifest verification failed: {problems[:3]}")
        m = read_json(c / MANIFEST_JSON)
        entries.append({"corpus_dir": str(c), "path_in_repo": f"instances/{m['instance']}/{m['variant']}/seed={m['seed']}",
                        "manifest": m})
    constants = sorted({e["manifest"]["constants_version"] for e in entries})
    card = CARD.format(version=version, tool_version=__version__, constants=", ".join(constants),
                       rows="\n".join(_row(e["manifest"]) for e in entries))
    index = {"schema": "tracebench/release@1", "version": version, "tool_version": __version__,
             "published_at": dt.date.today().isoformat(), "license": "CC-BY-4.0",
             "corpora": [{"path": e["path_in_repo"], "instance": e["manifest"]["instance"], "variant": e["manifest"]["variant"],
                          "seed": e["manifest"]["seed"], "config_hash": e["manifest"]["config_hash"],
                          "constants_version": e["manifest"]["constants_version"], "n_files": len(e["manifest"]["files"])}
                         for e in entries]}
    return entries, card, index


def publish(corpora, version, repo=DEFAULT_REPO, dry_run=False, api=None):
    entries, card, index = plan_release(corpora, version)
    if dry_run:
        return {"dry_run": True, "repo": repo, "version": version, "corpora": [e["path_in_repo"] for e in entries]}
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True)
    refs = api.list_repo_refs(repo_id=repo, repo_type="dataset")
    tags = {t.name for t in getattr(refs, "tags", [])}
    if version in tags:
        raise RuntimeError(f"version {version} already published to {repo}; bump the version")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "README.md").write_text(card)
        (tmp / "LICENSE").write_text("Creative Commons Attribution 4.0 International (CC BY 4.0)\nhttps://creativecommons.org/licenses/by/4.0/\n")
        write_json(tmp / "release.json", index)
        api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(tmp), path_in_repo=".",
                          commit_message=f"trace-bench {version}: release index and card")
    for e in entries:
        api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=e["corpus_dir"], path_in_repo=e["path_in_repo"],
                          commit_message=f"trace-bench {version}: {e['path_in_repo']}", ignore_patterns=["run/*"])
    api.create_tag(repo_id=repo, repo_type="dataset", tag=version, tag_message=f"trace-bench {version}")
    return {"dry_run": False, "repo": repo, "version": version, "corpora": [e["path_in_repo"] for e in entries]}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", action="append", required=True)
    p.add_argument("--version", required=True, help="release version, e.g. v0.1.0 (a dataset tag)")
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        res = publish(args.corpus, args.version, args.repo, args.dry_run)
    except (ValueError, RuntimeError) as e:
        log({"event": "publish_refused", "reason": str(e)})
        return 3
    log({"event": "publish", **res})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
