"""Parent-link inference: request trees from access lines, with no parent pointer.

Each hop attempt is one access line. The edge request of a trace is the hop
whose client address is not an internal address (or which other hops name as
their correlation id). An internal hop's parent is the hop of the same trace
whose service address equals the hop's client address and whose interval
contains it most tightly (start = ts - request_time, end = ts), with a
tolerance for clock skew; failing that, the edge hop. Client-side records link
to the edge hop through the request id they carry (tagged failures only).
"""
from __future__ import annotations

from collections import defaultdict

SKEW_TOLERANCE_MS = 60      # about three standard deviations of server clock skew


def build_trees(spans, service_ips):
    """-> traces: {trace_id: {"hops": [hop dicts in start order], "edge": hop or None}}.
    A hop dict: span_id, ts_ms (end), start_ms, request_id, correlation_id, service, path, status,
    request_time_ms, client_ip, parent (span_id or None), is_edge."""
    corr_of_request = {}
    for s in spans:
        if s["kind"] in ("vl.app", "vl.audit"):
            rid, cid = s["keys"].get("request_id"), s["keys"].get("correlation_id")
            if rid and cid and rid != cid:
                corr_of_request[rid] = cid
    traces = defaultdict(list)
    for s in spans:
        if s["kind"] != "vl.access" or not s["keys"].get("trace_id"):
            continue
        rt = s["http"].get("request_time") or 0.0
        hop = {"span_id": s["span_id"], "ts_ms": s["ts_ms"], "start_ms": s["ts_ms"] - int(rt * 1000),
               "request_id": s["keys"].get("request_id"), "correlation_id": corr_of_request.get(s["keys"].get("request_id")),
               "service": s["service"], "path": s["http"].get("path"), "status": s["http"].get("status"),
               "request_time_ms": int(rt * 1000), "client_ip": s["keys"].get("client_ip"), "parent": None, "is_edge": False}
        traces[s["keys"]["trace_id"]].append(hop)
    out = {}
    for tid, hops in traces.items():
        named_edges = {h["correlation_id"] for h in hops if h["correlation_id"]}
        for h in hops:
            if h["request_id"] in named_edges or not (h["client_ip"] or "").startswith("10."):
                h["is_edge"] = True
        edges = [h for h in hops if h["is_edge"]]
        edge = min(edges, key=lambda h: (h["start_ms"], h["span_id"])) if edges else None
        by_service = defaultdict(list)
        for h in hops:
            by_service[h["service"]].append(h)
        for h in hops:
            if h["is_edge"]:
                continue
            caller_service = service_ips.get(h["client_ip"])
            best = None
            for c in by_service.get(caller_service, ()):
                if c is h:
                    continue
                # containment violation (how far the child sticks out of the candidate), then tightness
                violation = max(0, c["start_ms"] - h["start_ms"]) + max(0, h["ts_ms"] - c["ts_ms"])
                if violation > SKEW_TOLERANCE_MS:
                    continue
                width = c["ts_ms"] - c["start_ms"]
                key = (violation, width, c["span_id"])
                if best is None or key < best[0]:
                    best = (key, c)
            h["parent"] = best[1]["span_id"] if best else (edge["span_id"] if edge and edge is not h else None)
        hops.sort(key=lambda h: (h["start_ms"], h["span_id"]))
        out[tid] = {"hops": hops, "edge": edge}
    return out
