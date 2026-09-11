"""Raw records -> flat spans: merge split parts, harvest join keys from text.

Nothing here knows the simulator. Every rule is one the documented real
correlation model applies: parts of one line share a `log_id` and are merged
in `part` order before any key is read; audit and application keys live in
free text and are harvested by pattern; a session id equal to the cart id is
the platform's silent fallback and is refused as a session key.
"""
from __future__ import annotations

import datetime as dt
import re
from collections import defaultdict

_KV = {
    k: re.compile(r'"%s":\s*"([^"]+)"' % k)
    for k in ("request_id", "trace_id", "correlation_id", "session_id", "cart_id", "client_unique_header", "sub", "url", "external_url")
}
_STATUS = re.compile(r'"response_status":\s*(\d+)')
EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)


def parse_ts_ms(ts):
    t = dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=dt.timezone.utc)
    return int((t - EPOCH).total_seconds() * 1000)


def merge_parts(records):
    """Merge multipart app/audit lines by log_id (part order); other records pass through."""
    parts = defaultdict(list)
    out = []
    for r in records:
        if r.get("total_parts", 1) > 1 and r["kind"] in ("vl.app", "vl.audit"):
            parts[r["log_id"]].append(r)
        else:
            out.append(r)
    merged_count = 0
    for log_id, group in parts.items():
        group = sorted(group, key=lambda r: r["part"])
        base = dict(group[0])
        base["message"] = "".join(r["message"] for r in group)
        base["merged_parts"] = len(group)
        base["parts_seen"] = [r["part"] for r in group]
        out.append(base)
        merged_count += 1
    return out, merged_count


def harvest(message):
    keys = {}
    for k, rx in _KV.items():
        m = rx.search(message)
        if m:
            keys[k] = m.group(1)
    m = _STATUS.search(message)
    if m:
        keys["response_status"] = int(m.group(1))
    return keys


def normalize(records):
    """-> (spans, stats). A span is a dict with span_id, kind, ts_ms, keys{}, http{}, service, pod, raw fields."""
    merged, n_merged = merge_parts(records)
    spans = []
    fallback = 0
    for r in merged:
        kind = r["kind"]
        keys = {}
        http = {}
        if kind == "vl.access":
            keys = {"request_id": r.get("request_id"), "trace_id": r.get("trace_id"), "client_ip": r.get("client_ip")}
            http = {"method": r.get("method"), "path": r.get("path"), "status": r.get("status"),
                    "request_time": r.get("request_time")}
        elif kind in ("vl.app", "vl.audit"):
            h = harvest(r.get("message", ""))
            keys = {"request_id": h.get("request_id"), "trace_id": h.get("trace_id"), "correlation_id": h.get("correlation_id"),
                    "session_id": h.get("session_id"), "cart_id": h.get("cart_id"), "device_id": h.get("client_unique_header"),
                    "account": h.get("sub")}
            if keys.get("session_id") and keys.get("cart_id") and keys["session_id"] == keys["cart_id"]:
                keys["session_id"] = None
                keys["session_id_is_cart_fallback"] = True
                fallback += 1
            http = {"path": h.get("url") or h.get("external_url"), "status": h.get("response_status")}
        elif kind == "sentry.error":
            keys = {"request_id": r.get("request_id"), "device_id": r.get("device_id"),
                    "account": r.get("account_id") if r.get("account_id") not in (None, "empty") else None,
                    "sentry_trace_id": r.get("sentry_trace_id")}
            http = {"path": r.get("page_url"), "status": None}
        elif kind == "sentry.transaction":
            keys = {"device_id": r.get("device_id"), "sentry_trace_id": r.get("sentry_trace_id")}
            http = {"path": r.get("page_url"), "status": r.get("status"), "request_time": (r.get("duration_ms") or 0) / 1000.0}
        else:
            continue
        keys = {k: v for k, v in keys.items() if v is not None}
        spans.append({"span_id": r["log_id"], "kind": kind, "ts_ms": parse_ts_ms(r["ts"]), "keys": keys, "http": http,
                      "service": r.get("service"), "pod": r.get("pod"), "log_source": r.get("log_source")})
    spans.sort(key=lambda s: (s["ts_ms"], s["span_id"]))
    return spans, {"records_in": len(records), "records_merged_lines": n_merged, "spans": len(spans), "cart_fallback_refused": fallback}
