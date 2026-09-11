"""Binary run artifacts live in object storage, never in git.

    python -m tracebench.artifacts push   <corpus-dir> --s3-uri s3://<bucket>/<prefix>
    python -m tracebench.artifacts pull   <corpus-dir> --s3-uri s3://<bucket>/<prefix>
    python -m tracebench.artifacts verify <corpus-dir>
    python -m tracebench.artifacts status <corpus-dir>

Shells out to the AWS CLI (the lab's convention; no boto3 dependency). The
object-store location comes only from the command line or the
TRACEBENCH_S3_URI environment variable, and the manifest it writes
(`artifacts.json`) sits INSIDE the corpus directory, which is never tracked,
so no bucket name can reach the repository (PRD scenario 20). The same
`push <folder> --s3-uri` surface as the lab's other repositories lets the
private execution configuration call it unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .record import read_json, sha256_file, write_json

ARTIFACTS_JSON = "artifacts.json"
BINARY_SUFFIXES = (".parquet", ".gz", ".npz", ".npy", ".jsonl")


def _run(cmd, check=True):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:3])} failed: {proc.stderr.strip()[:400]}")
    return proc


def local_binaries(folder):
    folder = Path(folder)
    out = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix in BINARY_SUFFIXES:
            out.append({"path": p.relative_to(folder).as_posix(), "bytes": p.stat().st_size, "sha256": sha256_file(p)})
    return out


def push(folder, s3_uri, profile=None):
    folder = Path(folder)
    files = local_binaries(folder)
    cmd = ["aws", "s3", "sync", str(folder), s3_uri.rstrip("/"), "--no-progress", "--exclude", "run/*"]
    if profile:
        cmd += ["--profile", profile]
    _run(cmd)
    manifest = {"schema": "tracebench/artifacts@1", "s3_uri": s3_uri.rstrip("/"), "files": files}
    write_json(folder / ARTIFACTS_JSON, manifest)
    return manifest


def pull(folder, s3_uri, profile=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cmd = ["aws", "s3", "sync", s3_uri.rstrip("/"), str(folder), "--no-progress"]
    if profile:
        cmd += ["--profile", profile]
    _run(cmd)
    return verify(folder)


def verify(folder):
    folder = Path(folder)
    m = read_json(folder / ARTIFACTS_JSON)
    problems = []
    for f in m["files"]:
        p = folder / f["path"]
        if not p.exists():
            problems.append(f"missing {f['path']}")
        elif sha256_file(p) != f["sha256"]:
            problems.append(f"changed {f['path']}")
    return problems


def status(folder):
    folder = Path(folder)
    have = local_binaries(folder)
    m = read_json(folder / ARTIFACTS_JSON) if (folder / ARTIFACTS_JSON).exists() else None
    return {"local_binaries": len(have), "local_bytes": sum(f["bytes"] for f in have),
            "manifest": None if m is None else {"s3_uri": m["s3_uri"], "n_files": len(m["files"])}}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("push", "pull"):
        sp = sub.add_parser(name)
        sp.add_argument("folder")
        sp.add_argument("--s3-uri", default=os.environ.get("TRACEBENCH_S3_URI"), help="object-store prefix (or TRACEBENCH_S3_URI)")
    for name in ("verify", "status"):
        sp = sub.add_parser(name)
        sp.add_argument("folder")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd in ("push", "pull") and not args.s3_uri:
        print("an object-store prefix is required (--s3-uri or TRACEBENCH_S3_URI); it is never read from the repository", file=sys.stderr)
        return 2
    if args.cmd == "push":
        m = push(args.folder, args.s3_uri, args.profile)
        print(json.dumps({"pushed": len(m["files"]), "s3_uri": m["s3_uri"]}))
        return 0
    if args.cmd == "pull":
        problems = pull(args.folder, args.s3_uri, args.profile)
        print(json.dumps({"problems": problems}))
        return 0 if not problems else 1
    if args.cmd == "verify":
        problems = verify(args.folder)
        print(json.dumps({"problems": problems}))
        return 0 if not problems else 1
    print(json.dumps(status(args.folder)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
