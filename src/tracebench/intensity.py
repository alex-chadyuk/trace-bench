"""Request intensity: a daily and weekly cycle over the simulated window.

Intensity is never stationary (PRD Behavior); structure is. lambda(t) is the
expected session-arrival rate at simulated time t and `intensity_value(t)`
buckets it into the tick-level latent root `intensity` (day / night / peak).
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from .mechanism import INTENSITY_VALUES

NIGHT_BELOW = 0.5      # lambda / base below this -> night
PEAK_ABOVE = 1.15      # lambda / base above this -> peak


def window_start(cfg):
    return dt.datetime.strptime(cfg.run.window.start, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)


def profile_multiplier(cfg, constants, t_seconds):
    """Multiplier of base_rps at simulated offset t (seconds from the window start)."""
    start = window_start(cfg)
    hourly = constants["daily_profile"]
    weekly = constants["weekday"]
    t = start + dt.timedelta(seconds=float(t_seconds))
    return hourly[t.hour] * weekly[t.weekday()]


def lambda_at(cfg, constants, t_seconds):
    return cfg.run.base_rps * profile_multiplier(cfg, constants, t_seconds)


def tick_lambdas(cfg, constants, t0_tick, n_ticks):
    """Vector of lambda for ticks t0..t0+n-1 (each `tick_s` seconds)."""
    tick_s = cfg.run.tick_s
    start = window_start(cfg)
    hourly = np.asarray(constants["daily_profile"], dtype=np.float64)
    weekly = np.asarray(constants["weekday"], dtype=np.float64)
    ticks = np.arange(t0_tick, t0_tick + n_ticks, dtype=np.int64)
    secs = ticks * tick_s
    # hour-of-day and weekday for every tick without per-tick datetime objects
    base_sec = start.hour * 3600 + start.minute * 60 + start.second
    abs_sec = base_sec + secs
    hour = (abs_sec // 3600) % 24
    day = (abs_sec // 86400) + start.weekday()
    weekday = day % 7
    return cfg.run.base_rps * hourly[hour] * weekly[weekday]


def intensity_value(cfg, constants, t_seconds):
    """Name of the `intensity` latent at offset t."""
    r = profile_multiplier(cfg, constants, t_seconds)
    if r < NIGHT_BELOW:
        return INTENSITY_VALUES[1]
    if r > PEAK_ABOVE:
        return INTENSITY_VALUES[2]
    return INTENSITY_VALUES[0]


def intensity_index_vector(cfg, constants, t0_tick, n_ticks):
    lam = tick_lambdas(cfg, constants, t0_tick, n_ticks) / cfg.run.base_rps
    out = np.zeros(n_ticks, dtype=np.int8)
    out[lam < NIGHT_BELOW] = 1
    out[lam > PEAK_ABOVE] = 2
    return out


def expected_sessions(cfg, constants):
    n_ticks = int(round(cfg.window_seconds / cfg.run.tick_s))
    return float(tick_lambdas(cfg, constants, 0, n_ticks).sum() * cfg.run.tick_s)
