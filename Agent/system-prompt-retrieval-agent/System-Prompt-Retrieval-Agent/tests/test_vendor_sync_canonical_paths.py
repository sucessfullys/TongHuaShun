"""Failure-path tests for the deterministic vendor-sync hard gate.

Covers the four failure paths required by S00.02c plus the happy path:

* Byte-mismatch between master and vendored copy        -> exit 2
* Missing master                                        -> exit 3
* Missing or unparseable provenance JSON                -> exit 2
* Provenance ``sha256`` not equal to master's sha256    -> exit 2
* Byte-identical + provenance current                   -> exit 0

Each test instantiates an isolated ``--repo-root`` under ``tmp_path`` so
the script's own real vendored copy is never touched.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "System-Prompt-Retrieval-Agent" / "scripts" / "sync_vendored_canonical_paths.py"
MASTER_REL = "Image-Generater-Remote/server/canonical_paths.py"
VENDORED_REL = (
    "System-Prompt-Retrieval-Agent/src/system_prompt_retrieval_agent/"
    "remote/_vendored/canonical_paths.py"
)
PROVENANCE_REL = (
    "System-Prompt-Retrieval-Agent/src/system_prompt_retrieval_agent/"
    "remote/_vendored/_canonical_paths_provenance.json"
)


def _run(repo_root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--repo-root", str(repo_root), *extra],
        capture_output=True,
        text=True,
    )


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """Build an isolated repo skeleton seeded from the real master."""
    real_master = REPO_ROOT / MASTER_REL
    assert real_master.is_file(), "real master must exist for tests"

    fake_master = tmp_path / MASTER_REL
    fake_master.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(real_master, fake_master)
    return tmp_path


def test_sync_then_check_exit_zero(fake_repo: Path) -> None:
    sync = _run(fake_repo)
    assert sync.returncode == 0, sync.stderr

    check = _run(fake_repo, "--check")
    assert check.returncode == 0, check.stderr
    assert "byte-identical and provenance current" in check.stdout


def test_byte_mismatch_returns_two(fake_repo: Path) -> None:
    sync = _run(fake_repo)
    assert sync.returncode == 0, sync.stderr

    vendored = fake_repo / VENDORED_REL
    vendored.write_bytes(vendored.read_bytes() + b"# tampered\n")

    check = _run(fake_repo, "--check")
    assert check.returncode == 2
    assert "byte-equality FAILED" in check.stderr


def test_missing_master_returns_three(fake_repo: Path) -> None:
    (fake_repo / MASTER_REL).unlink()

    check = _run(fake_repo, "--check")
    assert check.returncode == 3
    assert "master missing" in check.stderr


def test_missing_provenance_returns_two(fake_repo: Path) -> None:
    sync = _run(fake_repo)
    assert sync.returncode == 0, sync.stderr

    (fake_repo / PROVENANCE_REL).unlink()

    check = _run(fake_repo, "--check")
    assert check.returncode == 2
    assert "provenance missing" in check.stderr


def test_provenance_sha_mismatch_returns_two(fake_repo: Path) -> None:
    sync = _run(fake_repo)
    assert sync.returncode == 0, sync.stderr

    prov_path = fake_repo / PROVENANCE_REL
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    prov["sha256"] = "0" * 64
    prov_path.write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n")

    check = _run(fake_repo, "--check")
    assert check.returncode == 2
    assert "provenance sha256 mismatch" in check.stderr


def test_missing_vendored_copy_returns_two(fake_repo: Path) -> None:
    sync = _run(fake_repo)
    assert sync.returncode == 0, sync.stderr

    (fake_repo / VENDORED_REL).unlink()

    check = _run(fake_repo, "--check")
    assert check.returncode == 2
    assert "vendored copy missing" in check.stderr


def test_sync_writes_correct_provenance(fake_repo: Path) -> None:
    sync = _run(fake_repo)
    assert sync.returncode == 0, sync.stderr

    master_bytes = (fake_repo / MASTER_REL).read_bytes()
    expected_sha = hashlib.sha256(master_bytes).hexdigest()
    prov = json.loads((fake_repo / PROVENANCE_REL).read_text(encoding="utf-8"))
    assert prov["sha256"] == expected_sha
    assert prov["size_bytes"] == len(master_bytes)
    assert prov["source_path"] == MASTER_REL
    assert prov["source_mtime_utc"].endswith("Z")
    assert prov["sync_timestamp_utc"].endswith("Z")
