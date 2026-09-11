"""Regimes: opt-in mechanism changes at recorded changepoints.

A regime overlay is a patch applied to the fitted constants and/or the
mechanism parameters ("a deploy or a configuration change"); the topology and
the scenarios are unchanged. Each regime carries its own mechanism graph; the
changepoints and the set of edges whose strength changed are recorded
(PRD scenario 18).

Overlay shape (validated here, not in config.py, because it references
constants leaves):

    overlay:
      constants: {"<leaf path>": <value>, ...}     # e.g. error_shift.degraded -> {...}
      mechanism: {error_rate_multiplier: <float>}
"""
from __future__ import annotations

import copy

from .config import ConfigError, _problem
from .realism import LEAF_SPEC, RealismConstants, _check


def validate_overlay(overlay, index):
    problems = []
    allowed = {"constants", "mechanism"}
    extra = set(overlay) - allowed
    if extra:
        problems.append(_problem(f"schedules.regimes.{index}.overlay", sorted(extra), f"keys must be among {sorted(allowed)}"))
    for path, value in (overlay.get("constants") or {}).items():
        if path not in LEAF_SPEC:
            problems.append(_problem(f"schedules.regimes.{index}.overlay.constants.{path}", value, "not a known constants leaf"))
            continue
        err = _check(LEAF_SPEC[path], value)
        if err:
            problems.append(_problem(f"schedules.regimes.{index}.overlay.constants.{path}", value, err))
    mech = overlay.get("mechanism") or {}
    for key, value in mech.items():
        if key != "error_rate_multiplier":
            problems.append(_problem(f"schedules.regimes.{index}.overlay.mechanism.{key}", value, "only error_rate_multiplier may change at a changepoint"))
        elif not isinstance(value, (int, float)) or value <= 0:
            problems.append(_problem(f"schedules.regimes.{index}.overlay.mechanism.{key}", value, "must be a positive number"))
    if problems:
        raise ConfigError(problems)


def apply_overlay(cfg, constants: RealismConstants, overlay):
    """Return (cfg', constants') with the overlay applied; inputs untouched."""
    payload = copy.deepcopy(constants.as_dict())
    for path, value in (overlay.get("constants") or {}).items():
        cur = payload["leaves"]
        parts = path.split(".")
        for p in parts[:-1]:
            cur = cur[p]
        leaf = cur[parts[-1]]
        leaf["value"] = value
        leaf["note"] = (leaf.get("note") or "") + " [regime overlay]"
    new_constants = RealismConstants(payload, path=constants.path)
    new_cfg = cfg.model_copy(deep=True)
    for key, value in (overlay.get("mechanism") or {}).items():
        setattr(new_cfg.mechanism, key, value)
    return new_cfg, new_constants


def regime_intervals(cfg):
    """[(name, start_s, end_s)] covering the whole window; the base regime is
    named 'base' and every changepoint opens a new regime."""
    window = cfg.window_seconds
    points = [(r.name, r.at_s) for r in cfg.schedules.regimes]
    out = []
    prev_name, prev_t = "base", 0.0
    for name, t in points:
        out.append((prev_name, prev_t, t))
        prev_name, prev_t = name, t
    out.append((prev_name, prev_t, window))
    return out


def changed_edges(before, after, eps=1e-9, floor=None):
    """Edges whose strength differs by more than eps, or whose floored
    membership flips when a floor is given. Keys are (src, dst)."""
    b = {(e["src"], e["dst"]): e["strength"] for e in before}
    a = {(e["src"], e["dst"]): e["strength"] for e in after}
    changed = []
    for key in sorted(set(b) | set(a)):
        sb, sa = b.get(key, 0.0), a.get(key, 0.0)
        flipped = floor is not None and ((sb >= floor) != (sa >= floor))
        if abs(sb - sa) > eps or flipped:
            changed.append({"src": key[0], "dst": key[1], "before": sb, "after": sa, "flipped_at_floor": bool(flipped)})
    return changed
