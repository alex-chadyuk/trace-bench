"""JSON-line logging to stdout (house idiom: one dict per line, flushed)."""
import datetime as dt
import json
import sys


def now_iso():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def log(record, stream=None):
    payload = dict(record)
    payload.setdefault("ts", now_iso())
    out = stream or sys.stdout
    out.write(json.dumps(payload, sort_keys=True, default=_default) + "\n")
    out.flush()


def _default(obj):
    # numpy scalars and paths render as their Python equivalents; anything else
    # is a bug at the logging boundary, so fail loudly rather than emit repr().
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "__fspath__"):
        return str(obj)
    raise TypeError(f"not JSON serialisable at the log boundary: {type(obj).__name__}")
