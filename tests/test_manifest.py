"""PRD scenarios 13 (a release carries configuration, seed, tool version,
constants version, licence and per-file checksums, and verifies against its
manifest), 14 (re-publishing a version is refused) and 19/20 (public name
grammar; nothing private in the repository)."""
import json

import pytest

from tracebench.manifest import build_manifest, verify, verify_manifest, write_manifest
from tracebench.publish import plan_release, publish
from corpus_fixture import xs_corpus


def test_manifest_fields_and_verification(tmp_path):
    corpus = xs_corpus()
    m = write_manifest(corpus, label="test")
    for key in ("instance", "variant", "seed", "tool_version", "config_hash", "constants_version", "license", "files",
                "alphabet_size_realized_train", "n_nodes_mechanism", "default_floor", "byte_identity_excludes"):
        assert key in m, key
    assert m["license"] == "CC-BY-4.0" and m["constants_version"] == "1"
    assert all({"path", "bytes", "sha256", "method_readable"} <= set(f) for f in m["files"])
    assert any(f["path"].endswith(".parquet") and "content_sha256" in f for f in m["files"])
    assert not verify_manifest(corpus)
    res = verify(corpus)
    assert res["manifest_ok"] and res["names_ok"] and res["records_scanned"] > 0
    # tampering is detected
    target = corpus / "graphs" / "alphabet.json"
    original = target.read_text()
    target.write_text(original + "\n")
    try:
        problems = verify_manifest(corpus)
        assert any("changed" in p and "alphabet.json" in p for p in problems)
    finally:
        target.write_text(original)
    assert not verify_manifest(corpus)


class _StubApi:
    def __init__(self, tags):
        self.tags = tags
        self.uploads = []
        self.created = []

    def create_repo(self, **kw):
        self.created.append(kw)

    def list_repo_refs(self, **kw):
        class T:
            def __init__(self, n):
                self.name = n
        class R:
            pass
        r = R()
        r.tags = [T(t) for t in self.tags]
        return r

    def upload_folder(self, **kw):
        self.uploads.append(kw)

    def create_tag(self, **kw):
        self.tags.append(kw["tag"])


def test_publish_refuses_an_existing_version_and_tags_a_new_one():
    corpus = xs_corpus()
    write_manifest(corpus)
    api = _StubApi(tags=["v0.1.0"])
    with pytest.raises(RuntimeError) as ei:
        publish([corpus], "v0.1.0", repo="chadyuk/trace-bench", api=api)
    assert "already published" in str(ei.value) and not api.uploads
    res = publish([corpus], "v0.1.1", repo="chadyuk/trace-bench", api=api)
    assert res["version"] == "v0.1.1" and "v0.1.1" in api.tags
    assert any(u["path_in_repo"].startswith("instances/xs/latent/seed=0") for u in api.uploads)
    entries, card, index = plan_release([corpus], "v0.1.1")
    assert "cc-by-4.0" in card and index["corpora"][0]["constants_version"] == "1"


def test_dry_run_needs_no_network():
    corpus = xs_corpus()
    write_manifest(corpus)
    res = publish([corpus], "v9.9.9", dry_run=True)
    assert res["dry_run"] and res["corpora"] == ["instances/xs/latent/seed=0"]
