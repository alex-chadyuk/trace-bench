"""Categorical class tables shared by the mechanism (strengths), the latency
model (nominal retry time) and the engine (generation), built once from the
configuration and the fitted constants."""
from __future__ import annotations

import numpy as np

HEALTH_VALUES = ("healthy", "degraded", "failed")
POOL_VALUES = ("free", "tight", "exhausted")
CLASS_VALUES = ("ok", "4xx", "5xx", "err")
TIERS = ("bff", "service", "external")


def build_class_tables(cfg, constants):
    """{tier: table[health][pool] -> pmf over CLASS_VALUES} at worst = ok."""
    mult = cfg.mechanism.error_rate_multiplier
    out = {}
    for tier in TIERS:
        rates = constants.error_rates(tier)
        tab = np.zeros((3, 3, 4))
        for hi, health in enumerate(HEALTH_VALUES):
            for pi, pool in enumerate(POOL_VALUES):
                if health == "failed":
                    f = constants["error_shift.failed"]
                    tab[hi, pi] = (f["ok"], f["4xx"], f["5xx"], f["err"])
                    continue
                r = {k: v * mult for k, v in rates.items()}
                tot = sum(r.values())
                if tot > 0.95:
                    r = {k: v * 0.95 / tot for k, v in r.items()}
                    tot = 0.95
                pmf = np.array([1 - tot, r["4xx"], r["5xx"], r["err"]])
                # a degraded endpoint / an exhausted pool fails an explicit share of
                # requests on top of the nominal classes (mixture with the shift pmf)
                for cond, key in ((health == "degraded", "error_shift.degraded"), (pool == "exhausted", "error_shift.pool_exhausted")):
                    if cond:
                        sh = constants[key]
                        shift = np.array([sh["ok"], sh["4xx"], sh["5xx"], sh["err"]])
                        share = 1.0 - sh["ok"]
                        pmf = (1 - share) * pmf + share * (shift / max(share, 1e-12) * np.array([0, 1, 1, 1]))
                        pmf = pmf / pmf.sum()
                tab[hi, pi] = pmf
        out[tier] = tab
    return out


def nominal_retry_probability(class_tables, tier, retry_on, p_retry, mask):
    """P(an attempt at nominal state triggers a retry) = P(class in retry_on) * p_retry."""
    pmf = class_tables[tier][0, 0].copy()
    pmf = pmf * np.array([1.0, "4xx" in mask, "5xx" in mask, "err" in mask])
    pmf = pmf / pmf.sum()
    p = sum(pmf[CLASS_VALUES.index(c)] for c in retry_on if c in CLASS_VALUES)
    return float(p * p_retry)
