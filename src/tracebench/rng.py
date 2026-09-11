"""Deterministic random-number and identifier derivation.

Byte-identical reproducibility across machines (PRD scenario 12) rests on three
rules, all enforced here:

1. Every random stream is `Generator(PCG64(SeedSequence([run_seed, scope, shard,
   stream])))`. Shard boundaries and per-shard seeds are therefore pure
   functions of the run seed, which is what lets a resumed run reproduce an
   uninterrupted one (scenario 24).
2. Identifiers never come from `hash()` (salted per process) or `uuid4()`; they
   are blake2b digests over the canonical serialisation of their inputs.
3. The numpy minor version is recorded in the manifest and asserted at load,
   because Generator bit streams are only guaranteed stable within a version.
"""
import hashlib

import numpy as np

from .constants import INSTANTIATE_SHARD, SCOPE_GENERATE, SCOPE_INSTANTIATE, Stream


def seed_sequence(run_seed, scope, shard, stream):
    return np.random.SeedSequence([int(run_seed), int(scope), int(shard) & 0xFFFFFFFF, int(stream)])


def generator(run_seed, scope, shard, stream):
    return np.random.Generator(np.random.PCG64(seed_sequence(run_seed, scope, shard, stream)))


def instantiation_generator(run_seed, stream=Stream.INSTANTIATE):
    return generator(run_seed, SCOPE_INSTANTIATE, INSTANTIATE_SHARD, stream)


def shard_generator(run_seed, shard, stream):
    return generator(run_seed, SCOPE_GENERATE, shard, stream)


def stable_id(*parts, digest_size=16):
    """Hex identifier that depends only on its inputs (never on the process)."""
    h = hashlib.blake2b(digest_size=digest_size)
    for p in parts:
        if isinstance(p, bytes):
            h.update(p)
        else:
            h.update(str(p).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def stable_hex(*parts, length=32):
    """Fixed-length lowercase hex, e.g. a 32-hex request id."""
    return stable_id(*parts, digest_size=(length + 1) // 2)[:length]


def stable_uuid(*parts):
    """UUID-shaped identifier (8-4-4-4-12) derived deterministically."""
    h = stable_hex(*parts, length=32)
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def numpy_minor_version():
    major, minor = np.__version__.split(".")[:2]
    return f"{major}.{minor}"


def assert_numpy_version(expected_minor):
    have = numpy_minor_version()
    if have != expected_minor:
        raise RuntimeError(
            f"numpy minor version {have} differs from the corpus's {expected_minor}; "
            "Generator streams are only stable within a minor version"
        )
