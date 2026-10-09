"""D-TB-21: a derived corpus is the source corpus plus its channel — every
source file byte-identical, the ground truth re-derived, complete and
verifiable on its own, with its provenance in the manifest; refusals name why."""
import shutil

import pytest

from tracebench import derive as derive_mod
from tracebench.derive import DeriveRefused, derive
from tracebench.generate import generate
from tracebench.manifest import verify
from tracebench.record import read_json, sha256_file, write_json
from corpus_fixture import corpus_checksums, scratch_dir, xs_corpus, xs_metrics_corpus
from xs_fixture import XS


def test_linked_files_are_byte_identical_to_the_source():
    src = xs_corpus("latent")
    out = xs_metrics_corpus()
    sm, om = read_json(src / "manifest.json"), read_json(out / "manifest.json")
    source_files = {f["path"]: f["sha256"] for f in sm["files"]}
    derived_files = {f["path"]: f["sha256"] for f in om["files"]}
    for path, sha in source_files.items():
        if path.startswith("graphs/"):
            assert derived_files[path] != sha or path.endswith(("mechanism-graph.json", "changepoints.json")), path
        else:
            assert derived_files[path] == sha, path
    assert all(p.startswith(("metrics/", "oracle/state/", "graphs/")) for p in set(derived_files) - set(source_files))
    assert om["variant"] == "metrics" and om["config_hash"] == sm["config_hash"]
    assert read_json(out / "graphs" / "scoring-target.json")["variant"] == "metrics"
    assert read_json(src / "graphs" / "scoring-target.json")["variant"] == "latent"
    d = om["derived_from"]
    assert d["corpus_key"] == "xs/latent/seed=0" and d["variant"] == "latent"
    assert d["manifest_sha256"] == sha256_file(src / "manifest.json") and d["tool_version"] == sm["tool_version"]
    assert d["n_files_linked"] + d["n_files_copied"] == sum(1 for p in source_files if not p.startswith("graphs/"))
    assert om["exposure"] == {"exposed_groups": ["health", "intensity", "load", "pool"], "hidden_groups": ["auth", "cache", "net"]}
    assert sm["exposure"]["exposed_groups"] == [] and sm["derived_from"] is None and sm["metrics"] is None
    assert om["metrics"]["scrape_s"] == 30 and om["metrics"]["n_series"] > 0


def test_derived_corpus_is_complete_and_verifies_on_its_own():
    out = xs_metrics_corpus()
    assert (out / "COMPLETE").exists()
    res = verify(out)
    assert res["manifest_ok"] and res["names_ok"] and res["records_scanned"] > 0
    rec = read_json(out / "run" / "results.json")
    assert rec["complete"] and rec["derived_from"]["corpus_key"] == "xs/latent/seed=0"
    assert read_json(out / "run" / "run_meta.json")["command"] == "derive"


def test_resume_rewrites_only_the_missing_shard_and_reproduces_the_bytes():
    out = xs_metrics_corpus()
    src = xs_corpus("latent")
    before = corpus_checksums(out)
    root = scratch_dir("derive-resume")
    copy = root / "xs" / "metrics" / "seed=0"
    shutil.copytree(out, copy)
    (copy / "COMPLETE").unlink()
    (copy / "metrics" / "shard=0001.done").unlink()
    shutil.rmtree(copy / "metrics" / "shard=0001")
    shutil.rmtree(copy / "oracle" / "state" / "shard=0001")
    res = derive(src, "metrics", root, resume=True)
    assert res["metrics"]["n_shards"] == 4 and res["complete"]
    after = corpus_checksums(copy)
    assert after == before
    rec = read_json(copy / "run" / "results.json")
    assert rec["link_counts"]["kept"] > 0


def test_refusals(tmp_path):
    src = xs_corpus("latent")
    # already derived
    with pytest.raises(DeriveRefused, match="already complete"):
        derive(src, "metrics", xs_metrics_corpus().parents[2])
    # a twin is not a source
    with pytest.raises(DeriveRefused, match="derives from a 'latent'"):
        derive(xs_corpus("twin"), "metrics", tmp_path)
    # an incomplete source
    incomplete = tmp_path / "incomplete" / "xs" / "latent" / "seed=0"
    shutil.copytree(src, incomplete)
    (incomplete / "COMPLETE").unlink()
    with pytest.raises(DeriveRefused, match="not complete"):
        derive(incomplete, "metrics", tmp_path / "out1")
    # a source from a tool that predates the data this tool reads
    (incomplete / "COMPLETE").write_text("x\n")
    m = read_json(incomplete / "manifest.json")
    write_json(incomplete / "manifest.json", dict(m, tool_version="0.2.3"))
    with pytest.raises(DeriveRefused, match="0.3.0 or later"):
        derive(incomplete, "metrics", tmp_path / "out2")
    # a source that does not verify
    write_json(incomplete / "manifest.json", m)
    (incomplete / "graphs" / "alphabet.json").write_text("{}")
    with pytest.raises(DeriveRefused, match="does not verify"):
        derive(incomplete, "metrics", tmp_path / "out3")
    # not a derived variant
    with pytest.raises(DeriveRefused, match="not derived"):
        derive(src, "twin", tmp_path / "out4")
    # the CLI reports a refusal as exit 3
    assert derive_mod.main(["--from", str(src), "--variant", "metrics", "--out", str(xs_metrics_corpus().parents[2])]) == 3
    assert not (tmp_path / "out1").exists() or not (tmp_path / "out1" / "xs" / "metrics" / "seed=0" / "COMPLETE").exists()


def test_generate_refuses_a_derived_variant(tmp_path):
    with pytest.raises(ValueError, match="tracebench.derive"):
        generate(XS, 0, tmp_path, variant="metrics")
    assert not (tmp_path / "xs").exists()
