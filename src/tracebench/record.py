"""Run records and deterministic JSON/file helpers.

Every command writes `run/arguments.json`, `run/results.json` and
`run/run_meta.json`. `run_meta.json` is the one artifact excluded from the
byte-identity claim (it carries wall-clock and host facts); everything else
must be a pure function of configuration, seed, tool version and constants.
"""
import errno
import hashlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path

from . import __version__
from .constants import ARGUMENTS_JSON, RESULTS_JSON, RUN_DIR, RUN_META_JSON, TOOL_NAME
from .log import now_iso


def canonical_json(obj):
    """Canonical serialisation used for hashing and for every content-addressed
    artifact: sorted keys, no whitespace, UTF-8, no NaN."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_hash(obj, digest_size=16):
    return hashlib.blake2b(canonical_json(obj).encode("utf-8"), digest_size=digest_size).hexdigest()


def write_json(path, obj, indent=1):
    """Deterministic pretty JSON (sorted keys, fixed indent, trailing newline).
    Streamed through `json.dump`: the same bytes `json.dumps` would give, without
    materialising the text (the xl session target is several GB; 0.4.0)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, sort_keys=True, indent=indent, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    return path


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def link_or_copy(src, dst):
    """Hard-link `src` at `dst` (free in space, identical in bytes); copy when
    the filesystem refuses. Returns "linked" or "copied"."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return "linked"
    except OSError as e:
        if e.errno not in (errno.EXDEV, errno.EPERM, errno.EMLINK):
            raise
        shutil.copy2(src, dst)   # a different filesystem: pay the copy
        return "copied"


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def tool_versions():
    import numpy
    import pyarrow

    return {
        "tool": TOOL_NAME,
        "tool_version": __version__,
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pyarrow": pyarrow.__version__,
    }


class RunRecord:
    """Writes the three per-command record files into `<out>/run/`."""

    def __init__(self, out_dir, command, arguments, groups=None):
        self.run_dir = Path(out_dir) / RUN_DIR
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.command = command
        self.started = now_iso()
        write_json(self.run_dir / ARGUMENTS_JSON, {"command": command, "arguments": arguments, "_groups": groups or {}})
        self._meta_extra = {}

    def note(self, **facts):
        """Facts recorded in run_meta.json (never in results.json)."""
        self._meta_extra.update(facts)

    def finish(self, results, status="ok"):
        write_json(self.run_dir / RESULTS_JSON, results)
        meta = {
            "command": self.command,
            "status": status,
            "started": self.started,
            "finished": now_iso(),
            "host": platform.node(),
            "platform": platform.platform(),
            "argv": sys.argv,
            **tool_versions(),
            **self._meta_extra,
        }
        write_json(self.run_dir / RUN_META_JSON, meta)
        return meta
