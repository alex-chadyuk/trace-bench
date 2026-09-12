"""Corpus manifest and verification (PRD scenarios 12, 13, 19, 20, 24).

The manifest lists every file with its size and sha256 (parquet files also
carry a content hash over their canonical row stream, so a reader can check
identity across pyarrow versions), the configuration hash, seed, tool and
constants versions, licence, alphabet sizes and counts, and which files a
method may read. `verify` re-checks a corpus against its manifest and scans
every artifact for the public name grammar (and, when a private denylist is
supplied, for real names).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pyarrow.parquet as pq

from . import __version__
from .allowlist import is_method_readable
from .constants import (
    ALPHABET_JSON, COMPLETE_MARKER, GRAPHS_DIR, INSTANTIATION_JSON, MANIFEST_JSON, MANIFEST_SCHEMA, RUN_DIR,
    SCORING_TARGET_JSON,
)
from .log import log
from .naming import BFF_SERVICE, NAME_PATTERNS
from .record import RunRecord, canonical_json, read_json, sha256_file, write_json
from .rng import numpy_minor_version

BYTE_IDENTITY_EXCLUDES = ("run/", "manifest.json", "COMPLETE")
# files that live in a corpus directory but are not part of the corpus: the
# manifest itself, the completion marker, and the object-store pointer the
# artifacts command writes after a push
_UNLISTED = frozenset({MANIFEST_JSON, COMPLETE_MARKER, "artifacts.json"})


def content_hash_parquet(path):
    """sha256 over the canonical JSON of every row in file order (pyarrow-version independent)."""
    t = pq.read_table(path)
    h = hashlib.sha256()
    for batch in t.to_batches():
        for row in batch.to_pylist():
            h.update(canonical_json(row).encode("utf-8"))
            h.update(b"\n")
    return h.hexdigest()


def build_manifest(corpus_dir, label=None):
    corpus_dir = Path(corpus_dir)
    inst = read_json(corpus_dir / INSTANTIATION_JSON)
    files = []
    for p in sorted(corpus_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(corpus_dir).as_posix()
        if rel in _UNLISTED or rel.startswith(RUN_DIR + "/"):
            continue
        entry = {"path": rel, "bytes": p.stat().st_size, "sha256": sha256_file(p), "method_readable": is_method_readable(rel)}
        if p.suffix == ".parquet":
            entry["content_sha256"] = content_hash_parquet(p)
        files.append(entry)
    alphabet = read_json(corpus_dir / GRAPHS_DIR / ALPHABET_JSON) if (corpus_dir / GRAPHS_DIR / ALPHABET_JSON).exists() else {"n_tokens": None}
    target = read_json(corpus_dir / GRAPHS_DIR / SCORING_TARGET_JSON) if (corpus_dir / GRAPHS_DIR / SCORING_TARGET_JSON).exists() else {}
    stats = {}
    for root in sorted((corpus_dir / "views").glob("*/export-stats.json")) if (corpus_dir / "views").exists() else []:
        stats = read_json(root)
        break
    shards = sorted(int(p.name[len("shard="):-len(".done")]) for p in (corpus_dir / "raw").glob("shard=*.done")) if (corpus_dir / "raw").exists() else []
    manifest = {
        "schema": MANIFEST_SCHEMA, "label": label,
        "instance": inst["instance"], "variant": corpus_dir.parent.name, "seed": inst["seed"],
        "tool": "trace-bench", "tool_version": __version__, "config_hash": inst["config_hash"],
        "constants_version": inst["constants_version"], "numpy_minor": numpy_minor_version(),
        "license": "CC-BY-4.0", "code_license": "MIT",
        "default_floor": target.get("default_floor"), "orderings": ["end", "start"], "grains": ["request", "session"],
        "alphabet_size_potential": alphabet.get("n_tokens"),
        "alphabet_size_realized_train": stats.get("alphabet_size_realized_train"),
        "alphabet_size_vocab": stats.get("vocab_size"),
        "n_nodes_mechanism": inst["counts"]["mechanism_nodes"], "n_latent_nodes": inst["counts"]["mechanism_latent_nodes"],
        "counts": {**inst["counts"], "shards": len(shards), "view_rows": stats.get("rows")},
        "target_counts": {k: target.get(k) for k in ("n_directed", "n_bidirected", "n_directed_at_floor", "n_bidirected_at_floor")},
        "byte_identity_excludes": list(BYTE_IDENTITY_EXCLUDES),
        "files": files,
    }
    return manifest


def write_manifest(corpus_dir, label=None):
    m = build_manifest(corpus_dir, label)
    write_json(Path(corpus_dir) / MANIFEST_JSON, m)
    return m


def refresh_manifest(corpus_dir):
    """Rewrite an existing manifest after a command adds a report to the corpus
    (keeps the label). No-op when the corpus has no manifest yet."""
    corpus_dir = Path(corpus_dir)
    if not (corpus_dir / MANIFEST_JSON).exists():
        return None
    label = read_json(corpus_dir / MANIFEST_JSON).get("label")
    return write_manifest(corpus_dir, label)


# --- verification -------------------------------------------------------------------------
def verify_manifest(corpus_dir):
    corpus_dir = Path(corpus_dir)
    m = read_json(corpus_dir / MANIFEST_JSON)
    problems = []
    listed = {f["path"] for f in m["files"]}
    for f in m["files"]:
        p = corpus_dir / f["path"]
        if not p.exists():
            problems.append(f"missing: {f['path']}")
            continue
        if p.stat().st_size != f["bytes"] or sha256_file(p) != f["sha256"]:
            problems.append(f"changed: {f['path']}")
    for p in sorted(corpus_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(corpus_dir).as_posix()
            if rel not in listed and rel not in _UNLISTED and not rel.startswith(RUN_DIR + "/"):
                problems.append(f"unlisted: {rel}")
    return problems


_JSON_NAME_KEYS = ("service", "name", "pod", "host", "path", "caller_service", "callee_service", "caller_endpoint", "callee_endpoint")
_GRAMMAR = {
    "service": re.compile(r"^(tb-[a-z]{4,12}|tb-edge|ext-[a-z]{4,12}|client)$"),
    "pod": NAME_PATTERNS["pod"],
    "host": re.compile(r"^(node-\d+\.tb\.internal|)$"),
    "path": re.compile(r"^(/v1/[a-z]{4,12}/[a-z]{4,12}|/page/[a-z]{4,12}|/healthz)$"),
    "name": re.compile(r"^(/v1/[a-z]{4,12}/[a-z]{4,12}|/page/[a-z]{4,12}|[a-zA-Z:_\d/.-]+)$"),
}


def _trie_pattern(words):
    """Regex matching any of `words` (literal), compiled from a prefix trie so
    shared prefixes are tested once."""
    trie = {}
    for w in words:
        node = trie
        for ch in w:
            node = node.setdefault(ch, {})
        node[""] = True  # end marker

    def emit(node):
        end = "" in node
        keys = sorted(k for k in node if k != "")
        if not keys:
            return ""
        alts = []
        for k in keys:
            sub = emit(node[k])
            alts.append(re.escape(k) + sub)
        if len(alts) == 1 and not end:
            return alts[0]
        body = "(?:" + "|".join(alts) + ")"
        return body + ("?" if end else "")

    return emit(trie)


def scan_names(corpus_dir, denylist=None, max_records=200000):
    """Every name in every artifact matches the public grammar; none is on the denylist."""
    corpus_dir = Path(corpus_dir)
    deny_re = None
    if denylist:
        d = read_json(denylist)
        names = sorted({x.lower() for x in d.get("names", []) if x}, key=len, reverse=True)
        if names:
            # a prefix-trie pattern: one linear scan per field instead of a
            # Python loop (or a flat alternation's backtracking) over thousands of names
            deny_re = re.compile(_trie_pattern(names))
    problems = []
    seen = 0
    import gzip

    def check(value, where):
        nonlocal problems
        if deny_re is not None and isinstance(value, str):
            m = deny_re.search(value.lower())
            if m:
                problems.append(f"denylisted name {m.group(0)!r} in {where}")

    for p in sorted(corpus_dir.rglob("*.jsonl.gz")):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                seen += 1
                if seen > max_records:
                    break
                r = json.loads(line)
                for k in ("service", "pod", "host", "path"):
                    if k in r and r[k] is not None and not _GRAMMAR[k].match(str(r[k])):
                        problems.append(f"{p.name}: {k}={r[k]!r} outside the public grammar")
                for v in r.values():
                    check(v, p.name)
    inst = read_json(corpus_dir / INSTANTIATION_JSON)
    for svc in inst["topology"]["services"]:
        if not _GRAMMAR["service"].match(svc["name"]):
            problems.append(f"instantiation service {svc['name']!r} outside the public grammar")
        check(svc["name"], "instantiation")
    for op in inst["topology"]["ops"]:
        check(op["name"], "instantiation")
    cg = read_json(corpus_dir / "topology" / "callgraph.json")
    recorded = {(e["caller"], e["callee"]) for e in inst["topology"]["edges"]}
    if len(cg["edges"]) != len(recorded):
        problems.append("callgraph.json edge count differs from the instantiation")
    return problems[:200], seen


def verify(corpus_dir, denylist=None):
    problems = verify_manifest(corpus_dir)
    name_problems, scanned = scan_names(corpus_dir, denylist)
    return {"manifest_ok": not problems, "manifest_problems": problems, "names_ok": not name_problems,
            "name_problems": name_problems, "records_scanned": scanned, "denylist_used": bool(denylist)}


# --- frozen checksums ---------------------------------------------------------------------
CHECKSUMS_SCHEMA = "tracebench/checksums@1"


def corpus_key(m):
    """`<instance>/<variant>/seed=<seed>` — a corpus's identity, independent of
    where it was generated."""
    return f"{m['instance']}/{m['variant']}/seed={m['seed']}"


def freeze_checksums(corpora):
    """The per-file sha256 of one or more corpora, as the committed fixture that
    pins byte-identical regeneration across machines (PRD scenario 12).

    Every corpus must be at the same tool and constants version: a fixture that
    mixed versions would pin nothing.
    """
    manifests = [read_json(Path(c) / MANIFEST_JSON) for c in corpora]
    if not manifests:
        raise ValueError("no corpora to freeze")
    versions = {(m["tool_version"], m["constants_version"], m["numpy_minor"]) for m in manifests}
    if len(versions) != 1:
        raise ValueError(f"corpora span several versions: {sorted(versions)}")
    tool_version, constants_version, numpy_minor = versions.pop()
    return {
        "schema": CHECKSUMS_SCHEMA, "tool_version": tool_version, "constants_version": constants_version,
        "numpy_minor": numpy_minor,
        "corpora": {corpus_key(m): {"config_hash": m["config_hash"],
                                    "files": {f["path"]: f["sha256"] for f in m["files"]}}
                    for m in manifests},
    }


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write", help="write manifest.json for a complete corpus")
    w.add_argument("--corpus", required=True)
    w.add_argument("--label", default=None)
    v = sub.add_parser("verify", help="re-check a corpus against its manifest and the public name grammar")
    v.add_argument("--corpus", required=True)
    v.add_argument("--denylist", default=None, help="private JSON {names: [...]} of real names that must not appear")
    f = sub.add_parser("freeze", help="write the committed per-file checksum fixture for one or more corpora")
    f.add_argument("--corpus", action="append", required=True, help="a corpus directory with a manifest (repeatable)")
    f.add_argument("--out", required=True, help="fixture path, e.g. tests/fixtures/xs-checksums.json")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "write":
        rec = RunRecord(args.corpus, "manifest", vars(args))
        m = write_manifest(args.corpus, args.label)
        rec.finish({"n_files": len(m["files"]), "alphabet_size_realized_train": m["alphabet_size_realized_train"]})
        log({"event": "manifest", "n_files": len(m["files"])})
        return 0
    if args.cmd == "freeze":
        try:
            frozen = freeze_checksums(args.corpus)
        except ValueError as e:
            log({"event": "freeze_refused", "reason": str(e)})
            return 3
        write_json(args.out, frozen)
        log({"event": "freeze", "out": args.out, "tool_version": frozen["tool_version"],
             "corpora": sorted(frozen["corpora"]),
             "n_files": {k: len(v["files"]) for k, v in sorted(frozen["corpora"].items())}})
        return 0
    res = verify(args.corpus, args.denylist)
    log({"event": "verify", **{k: v for k, v in res.items() if not k.endswith("problems")}})
    for pr in res["manifest_problems"] + res["name_problems"]:
        log({"event": "verify_problem", "problem": pr})
    return 0 if res["manifest_ok"] and res["names_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
