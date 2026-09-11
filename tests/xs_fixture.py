"""Shared xs instantiation for the test suite (built once per process)."""
import functools
from pathlib import Path

from tracebench.instantiate import instantiate_from_paths

REPO = Path(__file__).resolve().parents[1]
XS = REPO / "configs" / "instances" / "xs.yaml"


@functools.lru_cache(maxsize=None)
def xs_instantiation(seed=0):
    return instantiate_from_paths(XS, seed=seed)
