"""Counter-keyed uniforms: every random draw of the engine is a pure function of
identifiers (run seed, domain, and up to four integer coordinates), never of
call order or realised outcomes.

This is what makes byte-identical reproducibility, sharded resume and common
random numbers hold at once: a fault forcing changes only the mapping from
draws to outcomes, never the draws; a resumed shard draws exactly what the
uninterrupted run drew; two machines agree bit for bit because uint64
arithmetic wraps identically everywhere. The mixer is splitmix64.
"""
from __future__ import annotations

import numpy as np

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_U53 = np.float64(1.0 / (1 << 53))

# Draw domains (never reuse a number).
D_LATENT = 1
D_ARRIVAL = 2
D_SESSION = 3
D_REQUEST = 4
D_HOP = 5
D_EMIT = 6
D_SKEW = 7
D_DEFECT = 8
D_CLIENT = 9


def splitmix64(z):
    z = np.asarray(z, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = z + _GOLDEN
        z = (z ^ (z >> np.uint64(30))) * _M1
        z = (z ^ (z >> np.uint64(27))) * _M2
        z = z ^ (z >> np.uint64(31))
    return z


def mix(seed, domain, *coords):
    """uint64 hash of the coordinates (any number; arrays broadcast). The
    number of coordinates is part of the key, so (a,) and (a, 0) differ."""
    with np.errstate(over="ignore"):
        h = splitmix64(np.uint64(int(seed) & 0xFFFFFFFFFFFFFFFF) ^ (np.uint64(int(domain)) * _M1))
        h = splitmix64(h ^ np.uint64(len(coords)))
        for part in coords:
            p = np.asarray(part, dtype=np.int64).astype(np.uint64)
            h = splitmix64(h ^ (p * _GOLDEN))
    return h


def uniforms(seed, domain, *coords):
    """float64 in [0, 1) with the broadcast shape of the coordinates."""
    h = mix(seed, domain, *coords)
    return (h >> np.uint64(11)).astype(np.float64) * _U53


def categorical(u, cdf_rows):
    """Index sampled from per-row CDFs: `cdf_rows[i]` is the cumulative pmf
    for draw `u[i]`. Rows must end at 1 (or above)."""
    u = np.asarray(u, dtype=np.float64)
    cdf_rows = np.asarray(cdf_rows, dtype=np.float64)
    if cdf_rows.ndim == 1:
        return int(np.searchsorted(cdf_rows, u, side="right").clip(0, len(cdf_rows) - 1))
    idx = (u[:, None] >= cdf_rows).sum(axis=1)
    return np.minimum(idx, cdf_rows.shape[1] - 1)
