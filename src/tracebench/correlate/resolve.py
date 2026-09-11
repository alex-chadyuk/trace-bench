"""Connector / identity resolution and attribution (one propagation round).

Identity keys name an actor: session, device, user (account hash). Connector
keys glue spans together: request id, correlation id (an internal hop's link
back to the edge request), backend trace id, front-end trace id, cart id.
Every connector accumulates the identity keys seen next to it on any span;
one propagation round then chains indirect links (front-end trace -> request
-> session). A span is attributed at the strongest level reachable from its
own keys, session > device > user (> ip when enabled), else `none`.
"""
from __future__ import annotations

from collections import defaultdict

from ..constants import ATTR_DEVICE, ATTR_IP, ATTR_NONE, ATTR_SESSION, ATTR_USER

LEVELS = (("session", ATTR_SESSION), ("device", ATTR_DEVICE), ("user", ATTR_USER), ("ip", ATTR_IP))


def identity_keys(span):
    k = span["keys"]
    out = {}
    if k.get("session_id"):
        out["session"] = {k["session_id"]}
    if k.get("device_id"):
        out["device"] = {k["device_id"]}
    if k.get("account"):
        out["user"] = {"acc=" + k["account"]}
    if span["kind"] == "vl.access" and k.get("client_ip") and not k["client_ip"].startswith("10."):
        out["ip"] = {k["client_ip"]}
    return out


def connectors(span):
    k = span["keys"]
    out = []
    if k.get("request_id"):
        out.append(("request", k["request_id"]))
    if k.get("correlation_id") and k["correlation_id"] != k.get("request_id"):
        out.append(("request", k["correlation_id"]))
    if k.get("trace_id"):
        out.append(("btrace", k["trace_id"]))
    if k.get("sentry_trace_id"):
        out.append(("strace", k["sentry_trace_id"]))
    if k.get("cart_id"):
        out.append(("cart", k["cart_id"]))
    return out


def build_resolution(spans):
    """connector -> {level: set(identity keys)} with one propagation round."""
    res = defaultdict(lambda: defaultdict(set))
    co_request = defaultdict(set)      # non-request connector -> request ids seen on the same span
    for s in spans:
        ids = identity_keys(s)
        cons = connectors(s)
        for c in cons:
            for level, vals in ids.items():
                res[c][level].update(vals)
        reqs = [c for c in cons if c[0] == "request"]
        for c in cons:
            if c[0] != "request":
                co_request[c].update(reqs)
    for c, reqs in co_request.items():
        for r in reqs:
            for level, vals in res.get(r, {}).items():
                res[c][level].update(vals)
    return res


def attribute(span, res, enable_ip=False):
    ids = identity_keys(span)
    cons = connectors(span)
    for level, code in LEVELS:
        if level == "ip" and not enable_ip:
            continue
        cands = set(ids.get(level, ()))
        for c in cons:
            cands.update(res.get(c, {}).get(level, ()))
        if cands:
            return code, sorted(cands)[0]
    return ATTR_NONE, None
