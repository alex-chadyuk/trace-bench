"""Publish corpora to the public dataset host, in two steps.

    python -m tracebench.publish upload  --corpus <dir> [--corpus <dir> ...] \\
        [--corpus-root <dir>] [--repo chadyuk/trace-bench] [--stage-dir <dir>] \\
        [--workers N] [--replace] [--keep-stage] [--dry-run]
    python -m tracebench.publish release --version vX.Y.Z [--repo chadyuk/trace-bench] [--dry-run]

`upload` puts one or more complete corpora on the dataset's default branch at
`instances/<name>/<variant>/seed=<s>/...` and is what an unattended generation
job runs: it is per-corpus, resumable, and refuses a path that already exists
unless `--replace` is given. It publishes no release metadata, so many
independent jobs can accumulate the corpora of one release.

`release` is the single-writer step that freezes what is on the host: it
verifies every remote corpus against its own `manifest.json` (present,
complete, same size and — where the host exposes it — the same sha256), writes
the dataset card, the licence and `release.json`, and tags the repository.
Releasing a version whose tag already exists is refused (PRD scenario 14).

Corpora are versioned by the tool version that generated them (every
`manifest.json` names it); a release is a tag over corpora already uploaded.

Uploads go through a staging tree of hard links (`<stage>/instances/...`), so
`upload_large_folder` — which has no `path_in_repo` — sees the repository
layout without a byte being copied. `run/` and `artifacts.json` never enter
the stage: the first carries host and wall-clock facts, the second the private
object-store location.
"""
from __future__ import annotations

import argparse
import datetime as dt
import errno
import os
import shutil
import sys
import tempfile
from pathlib import Path

from . import __version__
from .constants import COMPLETE_MARKER, MANIFEST_JSON, RUN_DIR
from .log import log
from .manifest import verify_manifest
from .record import read_json, write_json

DEFAULT_REPO = "chadyuk/trace-bench"
REPO_TYPE = "dataset"
DEFAULT_REVISION = "main"
INSTANCES_PREFIX = "instances"
STAGE_DIR_NAME = ".hf-stage"
# Never uploaded: the run record (host, wall clock, argv) is not part of a
# corpus, and artifacts.json names the private object store (PRD scenario 20).
NOT_UPLOADED = (RUN_DIR + "/", "artifacts.json")
IGNORE_PATTERNS = ["**/run/*", "**/artifacts.json"]

CARD = """---
license: cc-by-4.0
pretty_name: trace-bench
viewer: false
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

The dataset viewer is disabled: a corpus is a tree of gzipped JSON lines and
parquet files with several schemas, not one table.
"""


def _row(m):
    tc = m.get("target_counts") or {}
    return f"| {m['instance']} | {m['variant']} | {m['seed']} | {m.get('alphabet_size_realized_train')} | {m.get('n_nodes_mechanism')} | {tc.get('n_directed_at_floor')} / {tc.get('n_bidirected_at_floor')} |"


def path_in_repo(manifest):
    """Where a corpus lives on the dataset host — a pure function of its identity."""
    return f"{INSTANCES_PREFIX}/{manifest['instance']}/{manifest['variant']}/seed={manifest['seed']}"


def discover_corpora(root):
    """Every complete corpus under `root` (`<instance>/<variant>/seed=<k>/COMPLETE`)."""
    root = Path(root)
    return sorted(p.parent for p in root.glob(f"*/*/seed=*/{COMPLETE_MARKER}"))


# --- upload -------------------------------------------------------------------------------
def plan_upload(corpora):
    """Local, network-free check of what would be uploaded. Every corpus must be
    complete, carry a manifest and verify against it; two corpora may not claim
    the same remote path."""
    entries = []
    seen = {}
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
        target = path_in_repo(m)
        if target in seen:
            raise ValueError(f"{c} and {seen[target]} both claim {target}")
        seen[target] = c
        entries.append({"corpus_dir": str(c), "path_in_repo": target, "manifest": m})
    if not entries:
        raise ValueError("no corpora to upload")
    return entries


def default_stage_dir(entries):
    """Beside the corpus root (`<out>/.hf-stage`), never inside a corpus."""
    return Path(entries[0]["corpus_dir"]).resolve().parents[2] / STAGE_DIR_NAME


def _link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError as e:
        if e.errno not in (errno.EXDEV, errno.EPERM, errno.EMLINK):
            raise
        shutil.copy2(src, dst)   # a different filesystem: pay the copy


def _stage(entries, stage_root):
    """Hard-link every uploadable file of every corpus into `<stage_root>/instances/...`.

    The stage is what `upload_large_folder` walks, so it holds the repository
    layout; hard links make it free in space and identical in bytes.
    """
    stage_root = Path(stage_root).resolve()
    for e in entries:
        corpus = Path(e["corpus_dir"]).resolve()
        if stage_root == corpus or stage_root.is_relative_to(corpus):
            raise ValueError(f"stage directory {stage_root} is inside the corpus {corpus}; "
                             f"pass --stage-dir outside every corpus (it would be uploaded and hashed)")
        if corpus.is_relative_to(stage_root):
            raise ValueError(f"corpus {corpus} is inside the stage directory {stage_root}")
    n_files = 0
    n_bytes = 0
    for e in entries:
        corpus = Path(e["corpus_dir"]).resolve()
        dest_root = stage_root / e["path_in_repo"]
        if dest_root.exists():
            shutil.rmtree(dest_root)   # a resumed upload re-links from the corpus of record
        for p in sorted(corpus.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(corpus).as_posix()
            if rel.startswith(NOT_UPLOADED[0]) or rel in NOT_UPLOADED[1:]:
                continue
            _link(p, dest_root / rel)
            n_files += 1
            n_bytes += p.stat().st_size
    return {"stage_dir": str(stage_root), "n_files": n_files, "bytes": n_bytes}


def remote_has(api, repo, path, revision=DEFAULT_REVISION):
    return bool(api.get_paths_info(repo, [path], repo_type=REPO_TYPE, revision=revision))


def upload(corpora, repo=DEFAULT_REPO, dry_run=False, replace=False, stage_dir=None, workers=None,
           keep_stage=False, api=None):
    entries = plan_upload(corpora)
    stage = Path(stage_dir) if stage_dir else default_stage_dir(entries)
    if dry_run:
        return {"dry_run": True, "repo": repo, "stage_dir": str(stage),
                "corpora": [e["path_in_repo"] for e in entries]}
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    # Public from creation: private storage counts against the account quota,
    # and the corpora are CC BY 4.0 in any case.
    api.create_repo(repo_id=repo, repo_type=REPO_TYPE, exist_ok=True, private=False)
    for e in entries:
        if remote_has(api, repo, e["path_in_repo"]):
            if not replace:
                raise RuntimeError(f"{e['path_in_repo']} already exists in {repo}; pass --replace to overwrite it")
            api.delete_folder(path_in_repo=e["path_in_repo"], repo_id=repo, repo_type=REPO_TYPE,
                              revision=DEFAULT_REVISION,
                              commit_message=f"trace-bench: replace {e['path_in_repo']}")
    staged = _stage(entries, stage)
    api.upload_large_folder(repo_id=repo, folder_path=str(stage), repo_type=REPO_TYPE, revision=DEFAULT_REVISION,
                            ignore_patterns=IGNORE_PATTERNS, num_workers=workers, print_report_every=300)
    missing = [e["path_in_repo"] for e in entries
               if not remote_has(api, repo, f"{e['path_in_repo']}/{MANIFEST_JSON}")]
    if missing:
        # The stage carries upload_large_folder's own resume state: keep it.
        raise RuntimeError(f"upload finished but these corpora have no remote manifest: {missing}; "
                           f"re-run the same command to resume from {stage}")
    if not keep_stage:
        shutil.rmtree(stage, ignore_errors=True)
    return {"dry_run": False, "repo": repo, "corpora": [e["path_in_repo"] for e in entries],
            "staged": staged, "stage_kept": keep_stage}


# --- release ------------------------------------------------------------------------------
def remote_tree(api, repo, revision=DEFAULT_REVISION):
    """`{path: (size, sha256 or None)}` for every file on the host. Xet-backed
    files may expose no LFS sha256, in which case size is the only check."""
    out = {}
    for item in api.list_repo_tree(repo, repo_type=REPO_TYPE, revision=revision, recursive=True):
        size = getattr(item, "size", None)
        if size is None:
            continue   # a folder
        lfs = getattr(item, "lfs", None)
        out[item.path] = (size, getattr(lfs, "sha256", None) if lfs is not None else None)
    return out


def plan_release(api, repo, version, tmp):
    """Verify every remote corpus against its own manifest and build the release
    metadata. Reads the host; writes only into `tmp`."""
    refs = api.list_repo_refs(repo_id=repo, repo_type=REPO_TYPE)
    tags = {t.name for t in getattr(refs, "tags", [])}
    if version in tags:
        raise RuntimeError(f"version {version} already published to {repo}; bump the version")
    tree = remote_tree(api, repo)
    manifest_paths = sorted(p for p in tree
                            if p.startswith(INSTANCES_PREFIX + "/") and p.endswith("/" + MANIFEST_JSON)
                            and len(p.split("/")) == 5 and p.split("/")[3].startswith("seed="))
    if not manifest_paths:
        raise RuntimeError(f"{repo} holds no corpus under {INSTANCES_PREFIX}/; run `publish upload` first")
    tmp = Path(tmp)
    entries = []
    problems = []
    sha_checked = 0
    for mp in manifest_paths:
        corpus_path = mp[: -len("/" + MANIFEST_JSON)]
        local = api.hf_hub_download(repo, mp, repo_type=REPO_TYPE, revision=DEFAULT_REVISION,
                                    local_dir=str(tmp))
        m = read_json(local)
        if f"{corpus_path}/{COMPLETE_MARKER}" not in tree:
            problems.append(f"{corpus_path}: no {COMPLETE_MARKER} on the host")
        if path_in_repo(m) != corpus_path:
            problems.append(f"{corpus_path}: manifest identity is {path_in_repo(m)}")
        for f in m["files"]:
            rp = f"{corpus_path}/{f['path']}"
            found = tree.get(rp)
            if found is None:
                problems.append(f"missing: {rp}")
                continue
            size, sha = found
            if size != f["bytes"]:
                problems.append(f"size differs: {rp} ({size} on the host, {f['bytes']} in the manifest)")
            elif sha is not None:
                sha_checked += 1
                if sha != f["sha256"]:
                    problems.append(f"sha256 differs: {rp}")
        entries.append({"path_in_repo": corpus_path, "manifest": m})
    if problems:
        raise RuntimeError(f"{repo} does not match its manifests ({len(problems)} problems): {problems[:5]}")
    constants = sorted({e["manifest"]["constants_version"] for e in entries})
    tool_versions = sorted({e["manifest"]["tool_version"] for e in entries})
    card = CARD.format(version=version, tool_version=", ".join(tool_versions), constants=", ".join(constants),
                       rows="\n".join(_row(e["manifest"]) for e in sorted(
                           entries, key=lambda e: (e["manifest"]["instance"], e["manifest"]["variant"],
                                                   e["manifest"]["seed"]))))
    index = {"schema": "tracebench/release@1", "version": version, "tool_version": __version__,
             "corpus_tool_versions": tool_versions,
             "published_at": dt.date.today().isoformat(), "license": "CC-BY-4.0",
             "n_files_verified": sum(len(e["manifest"]["files"]) for e in entries),
             "n_files_sha256_verified": sha_checked,
             "corpora": [{"path": e["path_in_repo"], "instance": e["manifest"]["instance"],
                          "variant": e["manifest"]["variant"], "seed": e["manifest"]["seed"],
                          "config_hash": e["manifest"]["config_hash"],
                          "constants_version": e["manifest"]["constants_version"],
                          "tool_version": e["manifest"]["tool_version"],
                          "n_files": len(e["manifest"]["files"])}
                         for e in sorted(entries, key=lambda e: e["path_in_repo"])]}
    return entries, card, index


def release(version, repo=DEFAULT_REPO, dry_run=False, api=None):
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        entries, card, index = plan_release(api, repo, version, tmp / "manifests")
        result = {"dry_run": dry_run, "repo": repo, "version": version,
                  "corpora": [e["path_in_repo"] for e in entries],
                  "n_files_verified": index["n_files_verified"],
                  "n_files_sha256_verified": index["n_files_sha256_verified"]}
        if dry_run:
            return result
        meta = tmp / "release"
        meta.mkdir()
        (meta / "README.md").write_text(card, encoding="utf-8")
        (meta / "LICENSE").write_text(
            "Creative Commons Attribution 4.0 International (CC BY 4.0)\n"
            "https://creativecommons.org/licenses/by/4.0/\n", encoding="utf-8")
        write_json(meta / "release.json", index)
        api.upload_folder(repo_id=repo, repo_type=REPO_TYPE, folder_path=str(meta), path_in_repo=".",
                          revision=DEFAULT_REVISION,
                          commit_message=f"trace-bench {version}: release index and card")
    api.create_tag(repo_id=repo, repo_type=REPO_TYPE, tag=version, revision=DEFAULT_REVISION,
                   tag_message=f"trace-bench {version}")
    return result


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("upload", help="put complete corpora on the dataset host (no release metadata, no tag)")
    u.add_argument("--corpus", action="append", default=[], help="a complete corpus directory (repeatable)")
    u.add_argument("--corpus-root", default=None, help="upload every complete corpus under this directory")
    u.add_argument("--repo", default=DEFAULT_REPO)
    u.add_argument("--stage-dir", default=None,
                   help=f"hard-link staging tree (default <out>/{STAGE_DIR_NAME}); must be outside every corpus")
    u.add_argument("--workers", type=int, default=None, help="upload workers (default: the host library's own)")
    u.add_argument("--replace", action="store_true", help="delete and re-upload a corpus that is already there")
    u.add_argument("--keep-stage", action="store_true", help="keep the staging tree after a successful upload")
    u.add_argument("--dry-run", action="store_true", help="check the corpora locally; touch no network")
    r = sub.add_parser("release", help="verify the host against every remote manifest, then card + index + tag")
    r.add_argument("--version", required=True, help="release version, e.g. v0.2.0 (a dataset tag)")
    r.add_argument("--repo", default=DEFAULT_REPO)
    r.add_argument("--dry-run", action="store_true", help="verify the host; write nothing")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "upload":
            corpora = list(args.corpus) + (discover_corpora(args.corpus_root) if args.corpus_root else [])
            if not corpora:
                raise ValueError("no corpora: pass --corpus (repeatable) or --corpus-root with complete corpora under it")
            res = upload(corpora, repo=args.repo, dry_run=args.dry_run, replace=args.replace,
                         stage_dir=args.stage_dir, workers=args.workers, keep_stage=args.keep_stage)
        else:
            res = release(args.version, repo=args.repo, dry_run=args.dry_run)
    except (ValueError, RuntimeError) as e:
        log({"event": "publish_refused", "command": args.cmd, "reason": str(e)})
        return 3
    log({"event": "publish", "command": args.cmd, **res})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
