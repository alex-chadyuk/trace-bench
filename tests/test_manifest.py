"""PRD scenarios 13 (a release carries configuration, seed, tool version,
constants version, licence and per-file checksums, and verifies against its
manifest), 14 (re-publishing a version is refused) and 19/20 (public name
grammar; nothing private in the repository).

The dataset host is stubbed by `_StubApi`, an in-memory repository: the upload
path is exercised end to end (staging tree, duplicate refusal, replace) and so
is the release path (remote verification against every manifest, tag refusal,
card and index)."""
import hashlib
import sys
from pathlib import Path

import pytest

from tracebench.manifest import build_manifest, freeze_checksums, verify, verify_manifest, write_manifest
from tracebench.publish import path_in_repo, plan_release, plan_upload, release, upload
from tracebench.record import read_json, write_json
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


def test_freeze_checksums_pins_every_file(tmp_path):
    corpus = xs_corpus()
    m = write_manifest(corpus)
    frozen = freeze_checksums([corpus])
    assert frozen["tool_version"] == m["tool_version"] and frozen["constants_version"] == m["constants_version"]
    key = "xs/latent/seed=0"
    assert set(frozen["corpora"]) == {key}
    assert frozen["corpora"][key]["config_hash"] == m["config_hash"]
    assert frozen["corpora"][key]["files"] == {f["path"]: f["sha256"] for f in m["files"]}
    # a mixed-version fixture would pin nothing
    other = dict(m, tool_version="9.9.9")
    write_json(tmp_path / "manifest.json", other)
    (tmp_path / "COMPLETE").write_text("x\n")
    with pytest.raises(ValueError, match="several versions"):
        freeze_checksums([corpus, tmp_path])


# --- the stubbed dataset host -------------------------------------------------------------
class _RepoFile:
    def __init__(self, path, size, sha256):
        self.path = path
        self.size = size
        self.lfs = None if sha256 is None else type("Lfs", (), {"sha256": sha256})()


class _RepoFolder:
    def __init__(self, path):
        self.path = path
        self.tree_id = "folder"


class _StubApi:
    """An in-memory dataset repository: `remote` maps a repo path to
    (size, sha256, bytes). `expose_sha256 = False` models a host that returns no
    per-file hash (Xet-backed storage), where size is the only remote check."""

    def __init__(self, tags=(), expose_sha256=True):
        self.tags = list(tags)
        self.remote = {}
        self.created = []
        self.deleted = []
        self.large_uploads = []
        self.folder_uploads = []
        self.expose_sha256 = expose_sha256

    # -- writes
    def create_repo(self, **kw):
        self.created.append(kw)

    def _absorb(self, folder_path, prefix=""):
        folder_path = Path(folder_path)
        for p in sorted(folder_path.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(folder_path).as_posix()
            assert not rel.startswith("run/") and "/run/" not in rel, f"run/ reached the host: {rel}"
            assert Path(rel).name != "artifacts.json", f"artifacts.json reached the host: {rel}"
            data = p.read_bytes()
            key = f"{prefix}{rel}" if prefix else rel
            self.remote[key] = (len(data), hashlib.sha256(data).hexdigest(), data)

    def upload_large_folder(self, repo_id, folder_path, **kw):
        self.large_uploads.append({"repo_id": repo_id, "folder_path": str(folder_path), **kw})
        self._absorb(folder_path)

    def upload_folder(self, *, repo_id, folder_path, path_in_repo=".", **kw):
        self.folder_uploads.append({"repo_id": repo_id, "path_in_repo": path_in_repo, **kw})
        self._absorb(folder_path, "" if path_in_repo in (".", "", None) else path_in_repo.rstrip("/") + "/")

    def delete_folder(self, path_in_repo, repo_id, **kw):
        prefix = path_in_repo.rstrip("/") + "/"
        for key in [k for k in self.remote if k == path_in_repo or k.startswith(prefix)]:
            del self.remote[key]
        self.deleted.append(path_in_repo)

    def create_tag(self, repo_id, *, tag, **kw):
        self.tags.append(tag)

    # -- reads
    def get_paths_info(self, repo_id, paths, **kw):
        out = []
        for p in paths:
            if p in self.remote:
                size, sha, _ = self.remote[p]
                out.append(_RepoFile(p, size, sha if self.expose_sha256 else None))
            elif any(k.startswith(p.rstrip("/") + "/") for k in self.remote):
                out.append(_RepoFolder(p))
        return out

    def list_repo_tree(self, repo_id, path_in_repo=None, **kw):
        for path, (size, sha, _) in sorted(self.remote.items()):
            yield _RepoFile(path, size, sha if self.expose_sha256 else None)

    def hf_hub_download(self, repo_id, filename, *, local_dir=None, **kw):
        dest = Path(local_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.remote[filename][2])
        return str(dest)

    def list_repo_refs(self, **kw):
        return type("Refs", (), {"tags": [type("Tag", (), {"name": t})() for t in self.tags]})()


def _corpus_with_private_files():
    """The xs corpus as an unattended job leaves it: a `run/` record from
    generation and the object-store pointer a push writes."""
    corpus = xs_corpus()
    write_manifest(corpus)
    write_json(corpus / "artifacts.json", {"schema": "tracebench/artifacts@1", "files": []})
    assert (corpus / "run").is_dir() and not verify_manifest(corpus)
    return corpus


def test_upload_stages_without_the_run_record_or_the_store_pointer(tmp_path):
    corpus = _corpus_with_private_files()
    api = _StubApi()
    res = upload([corpus], repo="chadyuk/trace-bench", stage_dir=tmp_path / "stage", api=api)
    assert res["corpora"] == ["instances/xs/latent/seed=0"]
    assert api.created[0]["private"] is False and api.created[0]["repo_type"] == "dataset"
    # the host absorbed the staged tree (the stub asserts run/ and artifacts.json never arrive)
    assert "instances/xs/latent/seed=0/manifest.json" in api.remote
    assert "instances/xs/latent/seed=0/COMPLETE" in api.remote
    assert any(k.startswith("instances/xs/latent/seed=0/raw/") for k in api.remote)
    assert not [k for k in api.remote if "/run/" in k or k.endswith("artifacts.json")]
    # every listed file, plus the two the manifest does not list itself: COMPLETE and manifest.json
    assert res["staged"]["n_files"] == len(read_json(corpus / "manifest.json")["files"]) + 2
    assert not (tmp_path / "stage").exists()   # removed after a successful upload
    # a corpus already on the host is refused, and replaced only on request
    with pytest.raises(RuntimeError, match="already exists"):
        upload([corpus], repo="chadyuk/trace-bench", stage_dir=tmp_path / "stage2", api=api)
    assert not api.deleted
    upload([corpus], repo="chadyuk/trace-bench", stage_dir=tmp_path / "stage3", replace=True, api=api)
    assert api.deleted == ["instances/xs/latent/seed=0"]
    assert "instances/xs/latent/seed=0/manifest.json" in api.remote


def test_upload_refuses_an_incomplete_corpus_and_a_stage_inside_a_corpus(tmp_path):
    corpus = _corpus_with_private_files()
    assert plan_upload([corpus])[0]["path_in_repo"] == path_in_repo(read_json(corpus / "manifest.json"))
    with pytest.raises(ValueError, match="not complete"):
        plan_upload([tmp_path])
    with pytest.raises(ValueError, match="both claim"):
        plan_upload([corpus, corpus])
    api = _StubApi()
    with pytest.raises(ValueError, match="inside the corpus"):
        upload([corpus], repo="chadyuk/trace-bench", stage_dir=Path(corpus) / "hf-stage", api=api)
    assert not api.large_uploads


def test_upload_dry_run_needs_no_network(monkeypatch, tmp_path):
    corpus = _corpus_with_private_files()
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)   # any import of it now raises
    res = upload([corpus], dry_run=True, stage_dir=tmp_path / "stage")
    assert res["dry_run"] and res["corpora"] == ["instances/xs/latent/seed=0"]
    assert not (tmp_path / "stage").exists()


def test_release_verifies_the_host_then_tags_it(tmp_path):
    corpus = _corpus_with_private_files()
    api = _StubApi(tags=["v0.1.0"])
    upload([corpus], repo="chadyuk/trace-bench", stage_dir=tmp_path / "stage", api=api)

    with pytest.raises(RuntimeError, match="already published"):
        release("v0.1.0", repo="chadyuk/trace-bench", api=api)
    assert not api.folder_uploads

    # the remote must match the manifest it ships with
    path = "instances/xs/latent/seed=0/graphs/alphabet.json"
    size, sha, data = api.remote[path]
    api.remote[path] = (size + 1, sha, data)
    with pytest.raises(RuntimeError, match="size differs"):
        release("v0.2.0", repo="chadyuk/trace-bench", api=api)
    api.remote[path] = (size, "0" * 64, data)
    with pytest.raises(RuntimeError, match="sha256 differs"):
        release("v0.2.0", repo="chadyuk/trace-bench", api=api)
    api.remote[path] = (size, sha, data)

    dry = release("v0.2.0", repo="chadyuk/trace-bench", dry_run=True, api=api)
    assert dry["dry_run"] and dry["corpora"] == ["instances/xs/latent/seed=0"]
    assert dry["n_files_sha256_verified"] == dry["n_files_verified"] > 0
    assert "v0.2.0" not in api.tags and not api.folder_uploads

    res = release("v0.2.0", repo="chadyuk/trace-bench", api=api)
    assert res["version"] == "v0.2.0" and "v0.2.0" in api.tags
    card = api.remote["README.md"][2].decode()
    assert "viewer: false" in card and "cc-by-4.0" in card and "| xs | latent | 0 |" in card
    index = api.remote["release.json"][2].decode()
    assert '"version": "v0.2.0"' in index and '"path": "instances/xs/latent/seed=0"' in index
    assert "LICENSE" in api.remote


def test_release_refuses_a_missing_file_and_an_empty_host(tmp_path):
    corpus = _corpus_with_private_files()
    api = _StubApi()
    with pytest.raises(RuntimeError, match="holds no corpus"):
        release("v0.2.0", repo="chadyuk/trace-bench", api=api)
    upload([corpus], repo="chadyuk/trace-bench", stage_dir=tmp_path / "stage", api=api)
    del api.remote["instances/xs/latent/seed=0/graphs/alphabet.json"]
    with pytest.raises(RuntimeError, match="missing"):
        release("v0.2.0", repo="chadyuk/trace-bench", api=api)


def test_release_falls_back_to_size_when_the_host_exposes_no_hash(tmp_path):
    corpus = _corpus_with_private_files()
    api = _StubApi(expose_sha256=False)
    upload([corpus], repo="chadyuk/trace-bench", stage_dir=tmp_path / "stage", api=api)
    entries, card, index = plan_release(api, "chadyuk/trace-bench", "v0.2.0", tmp_path / "manifests")
    assert index["n_files_sha256_verified"] == 0 and index["n_files_verified"] > 0
    assert entries[0]["manifest"]["instance"] == "xs"
