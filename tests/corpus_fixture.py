"""Generated xs corpora shared across the test suite (built once per process)."""
import copy
import functools
import shutil
import tempfile
from pathlib import Path

import yaml

from tracebench.config import dump_config_yaml, parse_instance_config
from tracebench.generate import generate
from tracebench.record import sha256_file
from xs_fixture import REPO, XS

_ROOT = Path(tempfile.mkdtemp(prefix="tracebench-tests-"))


def _write_config(mutate, name):
    data = yaml.safe_load(XS.read_text())
    # the shipped constants; realism-dev.json is the labelled placeholder used only by the loader tests
    data["constants"] = str((REPO / "constants" / "realism-v1.json").resolve())
    if mutate:
        mutate(data)
    cfg = parse_instance_config(data)
    path = _ROOT / f"{name}.yaml"
    path.write_text(dump_config_yaml(cfg))
    return path


@functools.lru_cache(maxsize=None)
def xs_corpus(variant="latent", faults=True, tag="base"):
    """Path of a generated xs corpus for (variant, faults)."""
    def mutate(d):
        if not faults:
            d["schedules"]["faults"] = []
    name = f"xs-{variant}-{'faults' if faults else 'nofaults'}-{tag}"
    cfg_path = _write_config(mutate, name)
    out = _ROOT / name
    res = generate(cfg_path, 0, out, twin=(variant == "twin"))
    return Path(res["corpus_dir"])


def corpus_checksums(corpus_dir, exclude_prefixes=("run/",)):
    corpus_dir = Path(corpus_dir)
    out = {}
    for p in sorted(corpus_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(corpus_dir).as_posix()
            if any(rel.startswith(x) for x in exclude_prefixes):
                continue
            out[rel] = sha256_file(p)
    return out


def scratch_dir(name):
    d = _ROOT / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def write_variant_config(mutate, name):
    return _write_config(mutate, name)
