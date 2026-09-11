"""Determinism primitives: seeds, streams and identifiers are pure functions
of their inputs (the foundation of PRD scenarios 12 and 24)."""
import numpy as np

from tracebench.constants import Stream
from tracebench.rng import (
    instantiation_generator, numpy_minor_version, shard_generator, stable_hex,
    stable_id, stable_uuid,
)


def test_same_inputs_same_stream():
    a = shard_generator(7, 3, Stream.OUTCOME).random(5)
    b = shard_generator(7, 3, Stream.OUTCOME).random(5)
    assert np.array_equal(a, b)


def test_streams_shards_and_seeds_are_independent():
    base = shard_generator(7, 3, Stream.OUTCOME).random(5)
    assert not np.array_equal(base, shard_generator(7, 4, Stream.OUTCOME).random(5))
    assert not np.array_equal(base, shard_generator(7, 3, Stream.LATENCY).random(5))
    assert not np.array_equal(base, shard_generator(8, 3, Stream.OUTCOME).random(5))
    assert not np.array_equal(base, instantiation_generator(7).random(5))


def test_stable_ids_are_process_independent():
    # A fixed expectation pins the derivation itself, not just self-consistency.
    assert stable_id("request", 0, 12) == stable_id("request", 0, 12)
    assert stable_hex("a", 1, length=32) == "2e62af1e126c050a3305ca52c162b6e9"
    u = stable_uuid("session", 5)
    assert len(u) == 36 and u.count("-") == 4
    assert stable_id("x") != stable_id("y")


def test_numpy_minor_version_shape():
    assert numpy_minor_version().count(".") == 1
