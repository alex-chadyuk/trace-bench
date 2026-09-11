"""What a method under evaluation may read (PRD scenario 25).

A method sees the raw feed and the correlated views only. Fault records, case
labels, the mechanism graph, the scoring target and the oracle linkage are
outside that set. `open_for_method` is the one door: it refuses any path that
is not under a method-readable prefix of a corpus.
"""
from __future__ import annotations

from pathlib import Path

from .constants import METHOD_READABLE_PREFIXES


class NotMethodReadable(PermissionError):
    pass


def is_method_readable(relative_path):
    rel = str(relative_path).replace("\\", "/").lstrip("./")
    return any(rel.startswith(p) for p in METHOD_READABLE_PREFIXES)


def method_readable_files(corpus_dir):
    corpus_dir = Path(corpus_dir)
    out = []
    for p in sorted(corpus_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(corpus_dir).as_posix()
            if is_method_readable(rel):
                out.append(rel)
    return out


def open_for_method(corpus_dir, relative_path, mode="rb"):
    """Open a corpus file on behalf of a method; refuses labels, graphs and the oracle."""
    corpus_dir = Path(corpus_dir).resolve()
    target = (corpus_dir / relative_path).resolve()
    try:
        rel = target.relative_to(corpus_dir).as_posix()
    except ValueError:
        raise NotMethodReadable(f"{relative_path!r} escapes the corpus directory") from None
    if not is_method_readable(rel):
        raise NotMethodReadable(
            f"{rel!r} is not method-readable; a method may read only {METHOD_READABLE_PREFIXES}")
    return open(target, mode)
