"""PRD scenario 20: the public repository contains no account identifier,
bucket name, role name or network configuration, and no generated corpus or
binary run artifact is tracked."""
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

FORBIDDEN = [
    (re.compile(r"s3://[a-z0-9]"), "S3 URI"),
    (re.compile(r"arn:aws"), "AWS ARN"),
    (re.compile(r"\b\d{12}\b"), "12-digit AWS account id"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS access key id"),
    (re.compile(r"\baidev\b|--profile [a-z]"), "AWS profile name"),
    (re.compile(r"lotusflare|\.mgmt\.|qlab0\d|dc9\d-\d"), "real host or organisation name"),
    (re.compile(r"\blf-[a-z]+\b"), "real service name prefix"),
]
BINARY_SUFFIXES = (".parquet", ".npz", ".npy", ".gz", ".pt", ".jsonl")
TEXT_SUFFIXES = (".py", ".yaml", ".yml", ".md", ".toml", ".txt", ".json", ".cff", ".gitignore")


def _tracked_files():
    """Tracked files plus untracked files git would not ignore — i.e. everything
    that could reach the public remote."""
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO, check=True, capture_output=True, text=True,
    ).stdout
    return [REPO / line for line in sorted(set(out.splitlines())) if line]


def test_no_private_identifiers_in_tracked_text():
    offenders = []
    for path in _tracked_files():
        if path.suffix not in TEXT_SUFFIXES and path.name != ".gitignore":
            continue
        if path == Path(__file__).resolve():
            continue  # this file holds the pattern table itself
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern, what in FORBIDDEN:
            for m in pattern.finditer(text):
                line = text.count("\n", 0, m.start()) + 1
                offenders.append(f"{path.relative_to(REPO)}:{line}: {what} ({m.group(0)!r})")
    assert not offenders, "\n".join(offenders)


def test_no_binary_or_corpus_artifacts_tracked():
    tracked = [p for p in _tracked_files() if p.suffix in BINARY_SUFFIXES or "corpora/" in str(p) or "/data/" in str(p)]
    assert not tracked, [str(p.relative_to(REPO)) for p in tracked]


def test_gitignore_covers_generated_data():
    text = (REPO / ".gitignore").read_text()
    for pattern in ("corpora/", "data/", "*.parquet", "*.npz", "*.gz", "*.tfvars"):
        assert pattern in text.split(), f".gitignore must list {pattern}"
    assert "tracerca" not in text


def test_no_tracked_file_is_large():
    big = [(p, p.stat().st_size) for p in _tracked_files() if p.exists() and p.stat().st_size > 1_000_000]
    assert not big, big
