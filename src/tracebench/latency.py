"""Latency model: per-hop own service time, nested totals and the SLOW class.

Durations are part of the mechanism. A hop's own service time is drawn from a
piecewise-linear quantile function fitted to the tier's latency quantiles and
scaled by state multipliers (endpoint health, connection pool, cache miss). A
hop's total duration is its own time plus the totals of the callees it
invoked. SLOW is a deterministic function of the total: total > the op's
threshold, where the threshold is the configured quantile of the op's total
duration under nominal state. The threshold is estimated once at instantiation
by Monte Carlo from the INSTANTIATE stream (recorded sample count), which is
how the mechanism graph's slow-related strengths are computed as well.

All samplers here are vectorised and shared by the mechanism (strengths) and
the engine (generation), so the graph and the data come from one model.
"""
from __future__ import annotations

import numpy as np

from .constants import KIND_BFF, KIND_CLIENT, KIND_EXTERNAL, KIND_SERVICE, MC_DECIMALS, T_VALUES
from .realism import QUANTILE_KEYS
from .tables import build_class_tables, nominal_retry_probability

RETRY_BACKOFF_S = 0.05     # pause before a retried attempt, seconds (also engine.RETRY_BACKOFF_S)

QUANTILE_PROBS = np.array([0.0, 0.5, 0.9, 0.95, 0.99, 1.0])
TIER_OF_KIND = {KIND_BFF: "bff", KIND_SERVICE: "service", KIND_EXTERNAL: "external"}
# Own-time multipliers by class of the hop's own outcome: errors fail fast(ish).
CLASS_TIME_MULT = {"ok": 1.0, "4xx": 0.6, "5xx": 1.0, "err": 0.4}


def quantile_function(quantiles):
    """Knots of a piecewise-linear quantile function from (p50, p90, p95, p99);
    p0 = 0 and p100 = 3 x p99 (a bounded, fat tail)."""
    p50, p90, p95, p99 = quantiles
    return np.array([0.0, p50, p90, p95, p99, 3.0 * p99], dtype=np.float64)


def sample_own_time(rng, knots, n, multiplier=1.0):
    u = rng.random(n)
    return np.interp(u, QUANTILE_PROBS, knots) * multiplier


def sample_own_time_from_u(u, knots, multiplier):
    """Same map without drawing: `u` are uniforms already drawn (common random
    numbers), `multiplier` a scalar or an array broadcastable to `u`."""
    return np.interp(u, QUANTILE_PROBS, knots) * multiplier


class LatencyModel:
    """Holds per-op knots and state multipliers; computes nominal total
    distributions and SLOW thresholds bottom-up."""

    def __init__(self, topo, constants, slow_quantile, mc_samples=20000, cfg=None):
        self.topo = topo
        self.cfg = cfg
        self.slow_quantile = float(slow_quantile)
        self.mc_samples = int(mc_samples)
        self.mult_degraded = constants["latency_multipliers.degraded"]
        self.mult_pool_tight = constants["latency_multipliers.pool_tight"]
        self.mult_pool_exhausted = constants["latency_multipliers.pool_exhausted"]
        self.mult_cache_cold = constants["latency_multipliers.cache_cold"]
        self.knots = {}
        for op in topo.ops:
            if op.kind == KIND_CLIENT:
                continue
            self.knots[op.id] = quantile_function(constants.quantiles(TIER_OF_KIND[op.kind]))
        self.thresholds = {}          # op id -> SLOW threshold (seconds)
        self.nominal_samples = {}     # op id -> np.ndarray of nominal total durations
        self.nominal_p_slow = {}      # realised P(total > threshold) on the same samples
        # nominal per-op retry probability (a retried attempt adds own time + backoff to the caller)
        self.retry_p = {}
        if cfg is not None:
            tables = build_class_tables(cfg, constants)
            mask = set(cfg.endpoints.repertoire)
            for op in topo.ops:
                if op.kind in (KIND_SERVICE, KIND_EXTERNAL) and cfg.endpoints.retry.max_retries > 0:
                    self.retry_p[op.id] = nominal_retry_probability(tables, TIER_OF_KIND[op.kind], cfg.endpoints.retry.retry_on,
                                                                    constants["retry.p_retry_5xx"], mask)
                else:
                    self.retry_p[op.id] = 0.0

    def state_multiplier(self, health, pool, cache_miss):
        """health: 0 healthy / 1 degraded / 2 failed; pool: 0 free / 1 tight /
        2 exhausted; cache_miss: bool. Arrays broadcast."""
        m = np.ones(np.broadcast(health, pool, cache_miss).shape, dtype=np.float64)
        m = m * np.where(np.asarray(health) == 1, self.mult_degraded, 1.0)
        m = m * np.where(np.asarray(pool) == 1, self.mult_pool_tight, 1.0)
        m = m * np.where(np.asarray(pool) == 2, self.mult_pool_exhausted, 1.0)
        m = m * np.where(np.asarray(cache_miss), self.mult_cache_cold, 1.0)
        return m

    # --- nominal totals and thresholds (instantiation) ------------------------------
    def fit_thresholds(self, rng):
        """Bottom-up over call depth: an op's nominal total = own time (nominal
        state) + the totals of callees invoked under nominal cache warmth (a
        callee is called with its edge's call probability and a cached call is
        answered by the cache with probability p_hit; a callee not invoked
        costs nothing, retries included). Uses the given generator;
        deterministic."""
        topo = self.topo
        order = sorted((op for op in topo.ops if op.id in self.knots), key=lambda o: -o.depth)
        n = self.mc_samples
        for op in order:
            total = sample_own_time(rng, self.knots[op.id], n)
            for e in topo.callee_edges(op.id):
                callee = self.nominal_samples[e.callee]
                # resample the callee's nominal totals (independent hop)
                idx = rng.integers(len(callee), size=n)
                contrib = callee[idx]
                # a retried callee attempt costs its own time again plus the backoff
                rp = self.retry_p.get(e.callee, 0.0)
                if rp > 0:
                    retried = rng.random(n) < rp
                    contrib = contrib + np.where(retried, sample_own_time(rng, self.knots[e.callee], n) + RETRY_BACKOFF_S, 0.0)
                total = total + np.where(self._skipped(rng, e, n), 0.0, contrib)
            self.nominal_samples[op.id] = total
            # Rounded before it is stored, so the threshold that ships is the
            # threshold the engine classifies against and neither depends on the
            # CPU architecture's last bit (MC_DECIMALS). nominal_p_slow is left
            # as measured: it is a count over mc_samples, so rounding could not
            # absorb a sample crossing the threshold anyway.
            thr = round(float(np.quantile(total, self.slow_quantile)), MC_DECIMALS)
            self.thresholds[op.id] = thr
            self.nominal_p_slow[op.id] = float(np.mean(total > thr))
        return self.thresholds

    @staticmethod
    def _skipped(rng, e, n):
        """Requests on which a nominal caller does not invoke the callee: not
        called (call probability) or answered by a WARM cache."""
        skipped = np.zeros(n, bool)
        if e.p_call < 1.0:
            skipped |= rng.random(n) >= e.p_call
        if e.cached:
            skipped |= rng.random(n) < e.p_hit
        return skipped

    def total_samples(self, rng, op_id, health=0, pool=0, cache_miss=False,
                      callee_classes=None, n=None, callee_retry=None):
        """Monte Carlo samples of an op's total duration under a given own
        state and, optionally, forced callee final classes
        ({callee_op_id: class_name}); other callees at nominal. Used for the
        mechanism's slow-related strengths."""
        n = n or self.mc_samples
        topo = self.topo
        mult = float(self.state_multiplier(health, pool, cache_miss))
        total = sample_own_time(rng, self.knots[op_id], n, mult)
        for e in topo.callee_edges(op_id):
            forced = None if callee_classes is None else callee_classes.get(e.callee)
            base = self.nominal_samples[e.callee]
            thr = self.thresholds[e.callee]
            if forced == "absent":
                continue
            idx = rng.integers(len(base), size=n)
            contrib = base[idx]
            # a forced final class places the callee's total in that class's band (clamped,
            # exactly as the engine does when the class is forced)
            if forced == "slow":
                contrib = np.maximum(contrib, thr * 1.0001)
            elif forced in ("ok", "4xx", "5xx", "err"):
                contrib = np.minimum(contrib, thr)
            # retry time: forced by the callee's first-attempt class when given, nominal otherwise
            if callee_retry is not None and e.callee in callee_retry:
                rp = callee_retry[e.callee]
            else:
                rp = self.retry_p.get(e.callee, 0.0) if forced is None else 0.0
            if rp > 0:
                retried = rng.random(n) < rp
                contrib = contrib + np.where(retried, sample_own_time(rng, self.knots[e.callee], n) + RETRY_BACKOFF_S, 0.0)
            if forced is None:
                # a callee left at nominal may not be invoked; a forced class implies it was
                contrib = np.where(self._skipped(rng, e, n), 0.0, contrib)
            total = total + contrib
        return total

    def p_slow(self, rng, op_id, **kw):
        thr = self.thresholds[op_id]
        return float(np.mean(self.total_samples(rng, op_id, **kw) > thr))

    def to_dict(self):
        return {
            "slow_quantile": self.slow_quantile,
            "mc_samples": self.mc_samples,
            "thresholds_s": {str(k): v for k, v in sorted(self.thresholds.items())},
            "nominal_p_slow": {str(k): v for k, v in sorted(self.nominal_p_slow.items())},
        }


def slow_thresholds_json(model: LatencyModel, topo):
    return {
        "description": "per-operation SLOW threshold: the configured quantile of the op's total request time under nominal state, estimated at instantiation",
        "quantile": model.slow_quantile,
        "mc_samples": model.mc_samples,
        "thresholds": [
            {"op_id": op.id, "service": topo.services[op.service].name, "name": op.name,
             "threshold_s": model.thresholds[op.id], "nominal_p_slow": model.nominal_p_slow[op.id]}
            for op in topo.ops if op.id in model.thresholds
        ],
    }
