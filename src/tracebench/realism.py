"""Fitted realism constants: the versioned public file every corpus is generated from.

The file is produced by a private fitter that reads the internal sources; this
module only loads and validates it. Loading fails closed on any missing or
mis-shaped leaf (PRD scenario 11's public half): the generator never invents a
value. Every leaf is `{"value": ..., "source": <label>, "n": <int|null>,
"note": <str|null>}`; source labels are opaque.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .constants import REALISM_SCHEMA
from .record import read_json

QUANTILE_KEYS = ("p50", "p90", "p95", "p99")
TIERS = ("bff", "service", "external")

# Leaf path -> validator name. A validator returns None or an error string.
LEAF_SPEC: dict[str, str] = {
    **{f"latency_quantiles.{t}.{q}": "positive" for t in TIERS for q in QUANTILE_KEYS},
    "latency_multipliers.degraded": "ge1",
    "latency_multipliers.pool_tight": "ge1",
    "latency_multipliers.pool_exhausted": "ge1",
    "latency_multipliers.cache_cold": "ge1",
    "depth_pmf": "pmf",
    "fanout_pmf_by_depth": "pmf_list",
    **{f"error_rates.{t}.{c}": "prob" for t in TIERS for c in ("4xx", "5xx", "err")},
    "error_shift.degraded": "prob_map",
    "error_shift.failed": "prob_map",
    "error_shift.pool_exhausted": "prob_map",
    "retry.p_retry_5xx": "prob",
    "retry.mean_retries": "nonneg",
    "vocab_size": "posint",
    **{f"async_gap_quantiles.{q}": "positive" for q in QUANTILE_KEYS},
    "daily_profile": "profile24",
    "weekday": "profile7",
    "session.inactivity_gap_min": "positive",
    **{f"session.step_gap_quantiles.{q}": "positive" for q in QUANTILE_KEYS},
    "session.steps_pmf": "pmf",
    "records_per_request.access": "nonneg",
    "records_per_request.app_ok": "nonneg",
    "records_per_request.app_err": "nonneg",
    "records_per_request.audit": "nonneg",
    "split.rate": "prob",
    "split.parts_pmf": "pmf",
    "split.key_part_pmf": "pmf",
    "split.cut_chars": "posint",
    "unattributed_fraction": "prob",
    **{f"identity_level_mix.{k}": "prob" for k in ("session", "device", "user", "ip", "none")},
    "client.error_rate": "prob",
    "client.txn_sampling": "prob",
    "client.auto_captured_share": "prob",
    "client.net_flaky_share": "prob",
    "client.auth_expired_share": "prob",
    "client.logged_in_share": "prob",
    "background.app_lines_per_service_s": "nonneg",
    "cart_fallback_rate": "prob",
    "clock_skew_ms.server_sd": "nonneg",
    "clock_skew_ms.browser_sd": "nonneg",
    "health_check_period_s": "positive",
    "pods_per_service": "pmf",
    "state_dynamics.load_persist": "prob",
    "state_dynamics.pool_tight_given_high": "prob",
    "state_dynamics.pool_exhausted_given_high": "prob",
    "state_dynamics.cache_cold_given_high": "prob",
    "state_dynamics.cache_persist_cold": "prob",
    "state_dynamics.health_degraded_given_exhausted": "prob",
    "state_dynamics.health_degraded_base": "prob",
    "state_dynamics.health_failed_given_degraded": "prob",
    "state_dynamics.health_recover": "prob",
    "state_dynamics.pool_recover": "prob",
    "bytes_per_record.vl": "positive",
    "bytes_per_record.sentry": "positive",
    "gzip_ratio": "prob",
}


# Leaves a private fitter measures from real sources ("fitted") versus leaves that
# are structural design choices of the mechanism ("spec": declared with a
# rationale, never measured). The distinction is published with the file.
SPEC_LEAVES = {
    "latency_multipliers.degraded", "latency_multipliers.pool_tight", "latency_multipliers.pool_exhausted",
    "latency_multipliers.cache_cold", "error_shift.degraded", "error_shift.failed", "error_shift.pool_exhausted",
    "state_dynamics.load_persist", "state_dynamics.pool_tight_given_high", "state_dynamics.pool_exhausted_given_high",
    "state_dynamics.cache_cold_given_high", "state_dynamics.cache_persist_cold",
    "state_dynamics.health_degraded_given_exhausted", "state_dynamics.health_degraded_base",
    "state_dynamics.health_failed_given_degraded", "state_dynamics.health_recover", "state_dynamics.pool_recover",
    "client.net_flaky_share", "client.auth_expired_share",
    # platform facts the merged feed cannot expose (parts are merged before normalisation)
    "split.key_part_pmf", "split.cut_chars",
}
FITTED_LEAVES = [k for k in LEAF_SPEC if k not in SPEC_LEAVES]


class RealismError(ValueError):
    pass


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _check(kind, v):
    if kind == "positive":
        return None if _is_num(v) and v > 0 else "must be a positive finite number"
    if kind == "nonneg":
        return None if _is_num(v) and v >= 0 else "must be a non-negative finite number"
    if kind == "ge1":
        return None if _is_num(v) and v >= 1 else "must be a finite number >= 1"
    if kind == "prob":
        return None if _is_num(v) and 0 <= v <= 1 else "must be a probability in [0, 1]"
    if kind == "posint":
        return None if isinstance(v, int) and not isinstance(v, bool) and v > 0 else "must be a positive integer"
    if kind == "pmf":
        ok = isinstance(v, list) and v and all(_is_num(p) and p >= 0 for p in v) and abs(sum(v) - 1) < 1e-6
        return None if ok else "must be a non-empty list of non-negative numbers summing to 1"
    if kind == "pmf_list":
        ok = isinstance(v, list) and v and all(_check("pmf", row) is None for row in v)
        return None if ok else "must be a non-empty list of pmfs"
    if kind == "profile24":
        ok = isinstance(v, list) and len(v) == 24 and all(_is_num(p) and p > 0 for p in v)
        return None if ok else "must be 24 positive multipliers"
    if kind == "profile7":
        ok = isinstance(v, list) and len(v) == 7 and all(_is_num(p) and p > 0 for p in v)
        return None if ok else "must be 7 positive multipliers"
    if kind == "multiplier_map":
        ok = isinstance(v, dict) and set(v) == {"4xx", "5xx", "err"} and all(_is_num(x) and x >= 0 for x in v.values())
        return None if ok else "must map 4xx/5xx/err to non-negative multipliers"
    if kind == "prob_map":
        ok = isinstance(v, dict) and set(v) == {"ok", "4xx", "5xx", "err"} and all(_is_num(x) and 0 <= x <= 1 for x in v.values()) and abs(sum(v.values()) - 1) < 1e-6
        return None if ok else "must map ok/4xx/5xx/err to probabilities summing to 1"
    raise KeyError(kind)


class RealismConstants:
    def __init__(self, payload: dict[str, Any], path: str | None = None):
        self.path = path
        problems = []
        if payload.get("schema") != REALISM_SCHEMA:
            problems.append(f"schema: expected {REALISM_SCHEMA!r}, got {payload.get('schema')!r}")
        for key in ("version", "fitted_at", "sources", "leaves"):
            if key not in payload:
                problems.append(f"{key}: missing")
        if problems:
            raise RealismError("; ".join(problems))
        self.version = str(payload["version"])
        self.fitted_at = str(payload["fitted_at"])
        self.sources = dict(payload["sources"])
        self.leaves = payload["leaves"]
        self._values: dict[str, Any] = {}
        for path_, kind in LEAF_SPEC.items():
            leaf = _dig(self.leaves, path_)
            if leaf is None:
                problems.append(f"{path_}: missing leaf")
                continue
            if not isinstance(leaf, dict) or "value" not in leaf or "source" not in leaf:
                problems.append(f"{path_}: leaf must be {{value, source, n, note}}")
                continue
            if leaf["source"] not in self.sources:
                problems.append(f"{path_}: source {leaf['source']!r} not declared in sources")
            err = _check(kind, leaf["value"])
            if err:
                problems.append(f"{path_}: {err} (got {leaf['value']!r})")
            self._values[path_] = leaf["value"]
        if problems:
            raise RealismError("realism constants invalid — " + "; ".join(problems))

    def __getitem__(self, path_):
        return self._values[path_]

    def get(self, path_, default=None):
        return self._values.get(path_, default)

    def quantiles(self, tier):
        return [self[f"latency_quantiles.{tier}.{q}"] for q in QUANTILE_KEYS]

    def error_rates(self, tier):
        return {c: self[f"error_rates.{tier}.{c}"] for c in ("4xx", "5xx", "err")}

    def as_dict(self):
        return {"schema": REALISM_SCHEMA, "version": self.version, "fitted_at": self.fitted_at,
                "sources": self.sources, "leaves": self.leaves}


def _dig(d, dotted):
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def load_realism(path) -> RealismConstants:
    path = Path(path)
    if not path.exists():
        raise RealismError(f"realism constants file not found: {path}")
    return RealismConstants(read_json(path), path=str(path))


def resolve_constants_path(config_path, constants_field):
    """`constants` in a config is relative to the config file's repository root
    (the directory holding `configs/`), else to the config file's directory."""
    p = Path(constants_field)
    if p.is_absolute():
        return p
    cfg_dir = Path(config_path).resolve().parent
    for base in (cfg_dir, *cfg_dir.parents):
        cand = base / p
        if cand.exists():
            return cand
    return cfg_dir / p
