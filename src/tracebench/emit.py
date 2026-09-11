"""Emission: the raw heterogeneous log feed and the oracle linkage.

Components emit what they would really know and nothing more. Six record
kinds mirror the real feed: an access line per hop attempt at every nginx
(edge and internal; emitted at completion), error-gated application lines,
audit lines (the correlation hub, split into parts like the real feed, with
the cart-id-as-session fallback), client-side error events (a request id only
on app-tagged API failures), sampled client performance spans, and health
checks plus keyless background lines that no actor owns. No record carries a
parent pointer. Clock skew is applied per pod and per browser session; the
oracle keeps the true emission time, the true parent record, the true
request tree and the true session.

On the observable twin the same records additionally carry the latent state
values that influenced them (`state_*` fields) and a state-change log.
"""
from __future__ import annotations

import datetime as dt
import json
import math

import numpy as np

from .constants import (
    KIND_ACCESS, KIND_APP, KIND_AUDIT, KIND_BFF, KIND_CLIENT, KIND_EXTERNAL, KIND_SENTRY_ERROR,
    KIND_SENTRY_TXN, KIND_SERVICE, KIND_STATE_EVENT, T_VALUES,
)
from .engine import OUTCOME_ABSENT, OUTCOME_ERR, OUTCOME_OK, OUTCOME_SLOW, ShardResult
from .hashing import D_DEFECT, D_EMIT, D_SKEW, uniforms
from .intensity import window_start
from .latents import Slots, state_events
from .mechanism import AUTH_VALUES, CACHE_VALUES, HEALTH_VALUES, INTENSITY_VALUES, LOAD_VALUES, NET_VALUES, POOL_VALUES
from .naming import Namer
from .rng import stable_hex, stable_uuid

METHODS = ("GET", "POST", "POST", "GET", "PUT", "DELETE")
FILLER = "context payload " * 8
ERROR_MODULES = ("OSA", "ORD", "CAT", "USR", "PAY", "AUTH")
MAX_BREADCRUMBS = 50


def _iso(ms, epoch):
    t = epoch + dt.timedelta(milliseconds=int(ms))
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"


def _normal_from_uniforms(u1, u2):
    u1 = np.clip(u1, 1e-12, 1.0)
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


def _poisson_small(u, lam):
    """Inverse-CDF Poisson for small lambda from one uniform per draw (vectorised)."""
    lam = np.broadcast_to(np.asarray(lam, dtype=np.float64), np.shape(u))
    k = np.zeros(np.shape(u), dtype=np.int64)
    p = np.exp(-lam)
    cdf = p.copy()
    for i in range(1, 12):
        more = u >= cdf
        k += more
        p = p * lam / i
        cdf = cdf + p
    return k


class Emitter:
    def __init__(self, inst, seed, slots: Slots, variant="latent"):
        self.inst = inst
        self.seed = int(seed)
        self.slots = slots
        self.twin = variant == "twin"
        self.topo = inst.topo
        self.cfg = inst.cfg
        self.c = inst.constants
        self.epoch = window_start(inst.cfg)
        self.n_devices = inst.cfg.counts.clients
        topo = self.topo
        self.svc_name = [s.name for s in topo.services]
        self.svc_host = [s.host for s in topo.services]
        self.svc_ip = [s.ip for s in topo.services]
        self.op_svc = np.array([op.service for op in topo.ops])
        self.op_name = [op.name for op in topo.ops]
        self.op_kind = np.array([op.kind for op in topo.ops])
        self.op_method = [METHODS[int(uniforms(self.seed, D_EMIT, op.id, 0) * len(METHODS))] for op in topo.ops]
        # per-pod clock skew (ms), fixed for the corpus
        self.pod_skew = {}
        for s in topo.services:
            for j, pod in enumerate(s.pods):
                u1, u2 = uniforms(self.seed, D_SKEW, s.index, j, 0), uniforms(self.seed, D_SKEW, s.index, j, 1)
                self.pod_skew[(s.index, j)] = float(_normal_from_uniforms(u1, u2) * self.c["clock_skew_ms.server_sd"])
        self.split_rate = self.c["split.rate"]
        self.parts_cdf = np.cumsum(self.c["split.parts_pmf"])
        self.key_part_cdf = np.cumsum(self.c["split.key_part_pmf"])
        self.cut = int(self.c["split.cut_chars"])

    # --- identities ------------------------------------------------------------------
    def device_index(self, gid):
        return (uniforms(self.seed, D_EMIT, gid, 1) * self.n_devices).astype(np.int64)

    def device_id(self, dev):
        return stable_uuid(self.seed, "device", int(dev))

    def session_id(self, gid):
        return stable_uuid(self.seed, "session", int(gid))

    def cart_id(self, gid):
        return stable_uuid(self.seed, "cart", int(gid))

    def account_hash(self, dev):
        return stable_hex(self.seed, "account", int(dev), length=64)

    def request_id(self, req_gid, op, attempt):
        return stable_hex(self.seed, "req", int(req_gid), int(op), int(attempt), length=32)

    def trace_id(self, req_gid):
        return stable_hex(self.seed, "trace", int(req_gid), length=32)

    def log_id(self, *parts):
        return stable_hex(self.seed, "log", *parts, length=24)

    def browser_skew(self, gid):
        u1, u2 = uniforms(self.seed, D_SKEW, gid, 9, 0), uniforms(self.seed, D_SKEW, gid, 9, 1)
        return _normal_from_uniforms(u1, u2) * self.c["clock_skew_ms.browser_sd"]

    # --- main ------------------------------------------------------------------------
    def emit_shard(self, res: ShardResult):
        """Returns (records: list of dicts in emission order, oracle: dict of tables)."""
        records = []
        linkage = []
        hops = res.hops.cols
        reqs = res.requests.cols
        sess = res.sessions
        n_h = len(hops["hop_row"]) if hops else 0
        # --- per-request identities ---
        req_trace = {}
        edge_req_id = {}
        for i in range(len(reqs["req_gid"])):
            rg = int(reqs["req_gid"][i])
            req_trace[rg] = self.trace_id(rg)
            edge_req_id[rg] = self.request_id(rg, int(reqs["bff_op"][i]), 0)
        dev_of_session = self.device_index(sess["gid"]) if len(sess["gid"]) else np.zeros(0, np.int64)
        # --- hops: access + app + audit ---
        access_log_id = np.empty(n_h, dtype=object)
        for i in range(n_h):
            rg, op, k = int(hops["req_gid"][i]), int(hops["op"][i]), int(hops["attempt"][i])
            access_log_id[i] = self.log_id("access", rg, op, k)
        for i in range(n_h):
            rg, op, k = int(hops["req_gid"][i]), int(hops["op"][i]), int(hops["attempt"][i])
            svc = int(self.op_svc[op])
            pod_j = int(hops["pod"][i])
            pod = self.topo.services[svc].pods[pod_j]
            skew = self.pod_skew[(svc, pod_j)]
            end_ms = int(hops["end_ms"][i])
            outcome = int(hops["outcome"][i])
            is_edge = bool(hops["is_edge"][i])
            gid = int(hops["gid"][i])
            srow = int(np.searchsorted(sess["gid"], gid))
            dev = int(dev_of_session[srow])
            rid = self.request_id(rg, op, k)
            parent_hop = int(hops["parent_hop"][i])
            if is_edge:
                client_ip = Namer.client_ip(dev)
                ua = "tb-web/2.4 (browser)"
            else:
                pop = int(hops["parent_op"][i])
                psvc = int(self.op_svc[pop]) if pop >= 0 else svc
                client_ip = self.svc_ip[psvc]
                ua = "tb-http/1.1 (service)"
            state = self._state_fields(hops, i, sess, srow) if self.twin else {}
            rec = {
                "kind": KIND_ACCESS, "ts": _iso(end_ms + skew, self.epoch), "log_id": access_log_id[i],
                "pod": pod, "service": self.svc_name[svc], "host": self.svc_host[svc],
                "method": self.op_method[op], "path": self.op_name[op], "status": int(hops["status"][i]),
                "request_time": round(int(hops["duration_ms"][i]) / 1000.0, 3),
                "upstream_response_time": round(int(hops["own_ms"][i]) / 1000.0, 3) if self.op_kind[op] == KIND_BFF else "-",
                "request_id": rid, "trace_id": req_trace[rg], "client_ip": client_ip, "user_agent": ua,
                "body_bytes": int(200 + uniforms(self.seed, D_EMIT, rg, op, k, 3) * 4000), **state,
            }
            records.append(rec)
            parent_log = access_log_id[parent_hop] if parent_hop >= 0 else None
            linkage.append(self._link(rec, end_ms, svc, op, pod, rid, edge_req_id[rg], parent_log, gid, dev, rg, hops["request_row"][i], k, i))
            # application lines (error-gated)
            lam = self.c["records_per_request.app_err"] if outcome in (1, 2, 3) else self.c["records_per_request.app_ok"]
            n_app = int(_poisson_small(uniforms(self.seed, D_EMIT, rg, op, k, 4), lam))
            for j in range(n_app):
                u_t = uniforms(self.seed, D_EMIT, rg, op, k, 5, j)
                ts = int(hops["start_ms"][i] + u_t * max(1, hops["duration_ms"][i]))
                level = "ERROR" if outcome in (1, 2, 3) and j == n_app - 1 else "INFO"
                keys = {"request_id": rid, "trace_id": req_trace[rg]}
                if not is_edge:
                    keys["correlation_id"] = edge_req_id[rg]
                if bool(sess["logged_in"][srow]):
                    keys["sub"] = self.account_hash(dev)
                body = f"[{self.svc_name[svc]}:{self.op_name[op].split('/')[-1]}] request_context: " + _kv(keys) + " " + FILLER * (1 + j % 3)
                for part in self._split(body, rg, op, k, 10 + j, level, pod, svc, ts + skew, "app", state):
                    records.append(part)
                    linkage.append(self._link(part, ts, svc, op, pod, rid, edge_req_id[rg], access_log_id[i], gid, dev, rg, hops["request_row"][i], k, i))
            # audit: one per consumer response (edge) and per outbound external call
            is_ext = self.op_kind[op] == KIND_EXTERNAL
            if is_edge or is_ext:
                if is_edge:
                    sid = self.session_id(gid)
                    cart = self.cart_id(gid)
                    if uniforms(self.seed, D_DEFECT, rg, op, k, 1) < self.c["cart_fallback_rate"]:
                        sid = cart                      # the platform's silent fallback
                    keys = {"request_id": rid, "correlation_id": rid, "trace_id": req_trace[rg],
                            "session_id": sid, "cart_id": cart, "client_unique_header": self.device_id(dev),
                            "url": self.op_name[op], "response_status": int(hops["status"][i])}
                    audit_svc, audit_pod, audit_op = svc, pod, op
                    ts = end_ms - 1
                else:
                    pop = int(hops["parent_op"][i])
                    prow = parent_hop
                    caller_rid = self.request_id(rg, pop, int(hops["attempt"][prow])) if prow >= 0 else rid
                    keys = {"request_id": caller_rid, "correlation_id": edge_req_id[rg], "trace_id": req_trace[rg],
                            "session_id": self.session_id(gid),
                            "external_url": f"https://{self.svc_name[svc]}.example{self.op_name[op]}",
                            "response_status": int(hops["status"][i])}
                    audit_svc = int(self.op_svc[pop]) if pop >= 0 else svc
                    audit_pod = self.topo.services[audit_svc].pods[int(hops["pod"][prow])] if prow >= 0 else pod
                    audit_op = pop if pop >= 0 else op
                    ts = end_ms
                body = "[audit] third_party_logging_message_v1 " + _kv(keys) + " " + FILLER
                for part in self._split(body, rg, op, k, 20, "error", audit_pod, audit_svc, ts + skew, "audit", state):
                    records.append(part)
                    linkage.append(self._link(part, ts, audit_svc, audit_op, audit_pod, keys["request_id"], edge_req_id[rg],
                                              access_log_id[i] if is_edge else (access_log_id[prow] if prow >= 0 else access_log_id[i]),
                                              gid, dev, rg, hops["request_row"][i], k, i))
        # --- client records ---
        cl = res.clients.cols
        crumbs = {}      # gid -> list of breadcrumbs so far
        if cl:
            order = np.lexsort((cl["ts_ms"], cl["gid"]))
            for i in order:
                gid = int(cl["gid"][i])
                srow = int(cl["session_row"][i])
                dev = int(dev_of_session[srow])
                step, k = int(cl["step"][i]), int(cl["attempt"][i])
                rrow = int(cl["request_row"][i])
                rg = int(reqs["req_gid"][rrow])
                bff_op = int(cl["bff_op"][i])
                ts_true = int(cl["ts_ms"][i])
                ts = ts_true + self.browser_skew(gid)
                strace = stable_hex(self.seed, "strace", gid, step, k, length=32)
                edge_rid = edge_req_id[rg]
                page = self.op_name[int(cl["client_op"][i])]
                status = int(reqs["outcome"][rrow])
                http_status = 200 if status in (OUTCOME_OK, OUTCOME_SLOW) else (400 if status == 1 else (500 if status == 2 else 0))
                crumb = {"ts": _iso(ts, self.epoch), "method": self.op_method[bff_op], "status": http_status, "url": f"https://app.example{self.op_name[bff_op]}"}
                history = crumbs.setdefault(gid, [])
                state = self._client_state(sess, srow) if self.twin else {}
                if int(cl["outcome"][i]) == 1:
                    tagged = bool(cl["tagged"][i]) and status in (1, 2, 3)
                    rec = {"kind": KIND_SENTRY_ERROR, "ts": _iso(ts, self.epoch), "log_id": self.log_id("sentry", gid, step, k),
                           "event_id": stable_hex(self.seed, "sev", gid, step, k, length=32),
                           "title": "AxiosError: Request failed with status code %d" % http_status if status in (1, 2) else "Error: Network Error",
                           "page_url": f"https://app.example{page}", "sentry_trace_id": strace,
                           "error_module": ERROR_MODULES[int(uniforms(self.seed, D_EMIT, gid, step, k, 6) * len(ERROR_MODULES))],
                           "error_code": str(int(uniforms(self.seed, D_EMIT, gid, step, k, 7) * 9) + 1),
                           "http_failed": "BE" if status in (1, 2, 3) else "FE",
                           "breadcrumbs": list(history[-MAX_BREADCRUMBS:]) + [crumb], **state}
                    if tagged:
                        rec["request_id"] = edge_rid
                        rec["device_id"] = self.device_id(dev)
                        rec["account_id"] = self.account_hash(dev) if bool(sess["logged_in"][srow]) else "empty"
                    records.append(rec)
                    linkage.append(self._link(rec, ts_true, -1, int(cl["client_op"][i]), "browser", edge_rid if tagged else None, edge_rid,
                                              self.log_id("access", rg, bff_op, 0), gid, dev, rg, rrow, k, -1))
                if bool(cl["sampled_txn"][i]):
                    rec = {"kind": KIND_SENTRY_TXN, "ts": _iso(ts, self.epoch), "log_id": self.log_id("txn", gid, step, k),
                           "event_id": stable_hex(self.seed, "txn", gid, step, k, length=32), "device_id": self.device_id(dev),
                           "sentry_trace_id": strace, "page_url": f"https://app.example{page}", "op": "http.client",
                           "description": f"{self.op_method[bff_op]} {self.op_name[bff_op]}",
                           "duration_ms": int(reqs["end_ms"][rrow] - reqs["start_ms"][rrow]) + 40, "status": http_status, **state}
                    records.append(rec)
                    linkage.append(self._link(rec, ts_true, -1, int(cl["client_op"][i]), "browser", None, edge_rid,
                                              self.log_id("access", rg, bff_op, 0), gid, dev, rg, rrow, k, -1))
                history.append(crumb)
        # --- machinery nobody owns: health checks and keyless background lines ---
        period = self.c["health_check_period_s"]
        t0_ms, t1_ms = res.t0 * self.cfg.run.tick_s * 1000, res.t1 * self.cfg.run.tick_s * 1000
        for s in self.topo.services:
            if s.kind == KIND_CLIENT:
                continue
            for j, pod in enumerate(s.pods):
                phase = uniforms(self.seed, D_EMIT, s.index, j, 8) * period
                t = t0_ms + int(phase * 1000)
                nchk = 0
                while t < t1_ms:
                    rec = {"kind": KIND_ACCESS, "ts": _iso(t + self.pod_skew[(s.index, j)], self.epoch),
                           "log_id": self.log_id("healthz", s.index, j, t), "pod": pod, "service": s.name, "host": s.host,
                           "method": "GET", "path": "/healthz", "status": 200, "request_time": 0.001, "upstream_response_time": "-",
                           "request_id": stable_hex(self.seed, "hc", s.index, j, t, length=32), "trace_id": None,
                           "client_ip": "10.77.0.1", "user_agent": "kube-probe/1.29", "body_bytes": 2}
                    records.append(rec)
                    linkage.append(self._link(rec, t, s.index, -1, pod, rec["request_id"], None, None, -1, -1, -1, -1, 0, -1))
                    t += int(period * 1000)
                    nchk += 1
            rate = self.c["background.app_lines_per_service_s"] * (res.t1 - res.t0) * self.cfg.run.tick_s
            n_bg = int(_poisson_small(uniforms(self.seed, D_EMIT, s.index, res.shard, 9), min(rate, 11.0))) if rate < 11 else int(rate)
            for j in range(n_bg):
                t = t0_ms + int(uniforms(self.seed, D_EMIT, s.index, res.shard, 10, j) * (t1_ms - t0_ms))
                pod = s.pods[int(uniforms(self.seed, D_EMIT, s.index, res.shard, 11, j) * len(s.pods))]
                rec = {"kind": KIND_APP, "ts": _iso(t, self.epoch), "log_id": self.log_id("bg", s.index, res.shard, j),
                       "part": 1, "total_parts": 1, "pod": pod, "service": s.name, "level": "INFO",
                       "log_source": f"/srv/{s.name}/daemons/Scheduler.lua:{40 + j % 60}",
                       "message": "[Scheduler] tick: queue depth %d, workers idle" % int(uniforms(self.seed, D_EMIT, s.index, res.shard, 12, j) * 40)}
                records.append(rec)
                linkage.append(self._link(rec, t, s.index, -1, pod, None, None, None, -1, -1, -1, -1, 0, -1))
        # --- twin: state-change log ---
        if self.twin:
            for ev in state_events(res.latents, self.slots, self.topo):
                if res.t0 <= ev["tick"] < res.t1:
                    t = ev["tick"] * self.cfg.run.tick_s * 1000
                    rec = {"kind": KIND_STATE_EVENT, "ts": _iso(t, self.epoch), "log_id": self.log_id("state", ev["tick"], ev["node"]),
                           "node": ev["node"], "service": ev["service"], "subject": ev["subject"], "value": ev["value"], "previous": ev["previous"]}
                    records.append(rec)
                    linkage.append(self._link(rec, t, -1, -1, "simulator", None, None, None, -1, -1, -1, -1, 0, -1))
        records.sort(key=lambda r: (r["ts"], r["log_id"]))
        oracle = {"linkage": linkage, "spans": self._spans(res, access_log_id, edge_req_id, dev_of_session),
                  "sessions": self._sessions(res, dev_of_session)}
        return records, oracle

    # --- helpers --------------------------------------------------------------------------
    def _state_fields(self, hops, i, sess, srow):
        return {"state_health": HEALTH_VALUES[int(hops["state_health"][i])], "state_pool": POOL_VALUES[int(hops["state_pool"][i])],
                "state_cache": CACHE_VALUES[int(hops["state_cache"][i])], "state_load": LOAD_VALUES[int(hops["state_load"][i])],
                "state_intensity": INTENSITY_VALUES[int(hops["state_intensity"][i])], "state_retry": int(hops["attempt"][i])}

    def _client_state(self, sess, srow):
        return {"state_net": NET_VALUES[int(sess["net"][srow])], "state_auth": AUTH_VALUES[int(sess["auth"][srow])]}

    def _split(self, body, rg, op, k, tag, level, pod, svc, ts_ms, source, state):
        """One app/audit line, split into parts like the real feed when long."""
        log_id = self.log_id(source, rg, op, k, tag)
        if len(body) > self.cut or uniforms(self.seed, D_DEFECT, rg, op, k, tag) < self.split_rate:
            n_parts = int(np.searchsorted(self.parts_cdf, uniforms(self.seed, D_DEFECT, rg, op, k, tag, 1))) + 1
            n_parts = max(2, n_parts)
            key_part = min(n_parts, int(np.searchsorted(self.key_part_cdf, uniforms(self.seed, D_DEFECT, rg, op, k, tag, 2))) + 1)
            # place the key-bearing text in part `key_part`; pad the others with filler
            pad = (FILLER * 200)[: self.cut]
            parts = []
            for p in range(1, n_parts + 1):
                text = body if p == key_part else pad
                parts.append(text)
        else:
            parts = [body]
        out = []
        for p, text in enumerate(parts, start=1):
            out.append({"kind": KIND_APP if source == "app" else KIND_AUDIT, "ts": _iso(ts_ms, self.epoch), "log_id": log_id,
                        "part": p, "total_parts": len(parts), "pod": pod, "service": self.svc_name[svc], "level": level,
                        "log_source": (f"/srv/{self.svc_name[svc]}/routes/Handler.lua:{28 + tag}" if source == "app" else f"/srv/libs/http/AuditSink.lua:{100 + tag}"),
                        "message": text, **state})
        return out

    def _link(self, rec, true_ts, svc, op, pod, request_id, edge_request_id, parent_log_id, gid, dev, req_gid, request_row, attempt, hop_row):
        return {"log_id": rec["log_id"], "part": rec.get("part", 1), "kind": rec["kind"], "true_ts_ms": int(true_ts),
                "component": self.svc_name[svc] if svc >= 0 else ("browser" if rec["kind"].startswith("sentry") else "simulator"),
                "op": int(op), "pod": pod, "request_id": request_id, "edge_request_id": edge_request_id,
                "parent_log_id": parent_log_id, "session_gid": int(gid), "device_index": int(dev), "req_gid": int(req_gid),
                "request_row": int(request_row), "attempt": int(attempt), "hop_row": int(hop_row)}

    def _spans(self, res, access_log_id, edge_req_id, dev_of_session):
        h = res.hops.cols
        if not h:
            return []
        out = []
        for i in range(len(h["hop_row"])):
            ph = int(h["parent_hop"][i])
            rg = int(h["req_gid"][i])
            rec = {"span_id": access_log_id[i], "parent_span_id": access_log_id[ph] if ph >= 0 else None,
                   "op": int(h["op"][i]), "service": self.svc_name[int(self.op_svc[int(h["op"][i])])], "name": self.op_name[int(h["op"][i])],
                   "outcome": int(h["outcome"][i]), "status": int(h["status"][i]), "attempt": int(h["attempt"][i]),
                   "start_ms": int(h["start_ms"][i]), "dur_ms": int(h["duration_ms"][i]), "own_ms": int(h["own_ms"][i]),
                   "own_us": int(h["own_us"][i]), "dur_us": int(h["duration_us"][i]),
                   "req_gid": rg, "edge_request_id": edge_req_id[rg], "session_gid": int(h["gid"][i]),
                   "is_edge": bool(h["is_edge"][i]), "tick": int(h["tick"][i]),
                   "state_health": int(h["state_health"][i]), "state_pool": int(h["state_pool"][i]), "state_cache": int(h["state_cache"][i]),
                   "state_load": int(h["state_load"][i]), "state_intensity": int(h["state_intensity"][i])}
            out.append(rec)
        return out

    def _sessions(self, res, dev_of_session):
        s = res.sessions
        n = len(s["gid"])
        return [{"gid": int(s["gid"][i]), "scenario": int(s["scenario"][i]), "arrival_ms": int(s["arrival_ms"][i]),
                 "device_index": int(dev_of_session[i]), "device_id": self.device_id(int(dev_of_session[i])),
                 "session_id": self.session_id(int(s["gid"][i])), "cart_id": self.cart_id(int(s["gid"][i])),
                 "logged_in": bool(s["logged_in"][i]), "net": int(s["net"][i]), "auth": int(s["auth"][i]),
                 "steps_done": int(s["steps_done"][i]), "final_ok": bool(s["final_ok"][i])} for i in range(n)]


def _kv(keys):
    return "{" + ", ".join(f"{json.dumps(k)}: {json.dumps(v)}" for k, v in keys.items()) + "}"
