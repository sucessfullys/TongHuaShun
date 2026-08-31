"""Tests for the ERA environment preflight script (plugin/scripts/preflight.sh)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from era.paths import repo_root

PREFLIGHT = repo_root() / "plugin" / "scripts" / "preflight.sh"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(PREFLIGHT), *args],
        capture_output=True,
        text=True,
    )


def test_preflight_script_exists():
    assert PREFLIGHT.is_file()


def test_preflight_passes_on_real_repo():
    """No argument → the script self-derives the (healthy) ERA repo root."""
    result = _run()
    assert result.returncode == 0, result.stderr
    assert "PASS" in result.stdout


def test_preflight_reports_loop_engine():
    """Preflight names the loop engine so /era:start can pick a mode without a
    cryptic mid-run hook crash (plugin when jq is present, manual fallback when
    not). jq is not fatal — preflight still passes either way."""
    result = _run()
    assert result.returncode == 0, result.stderr
    assert "loop engine =" in result.stdout
    assert ("ralph-loop plugin" in result.stdout
            or "manual fallback" in result.stdout)


def test_preflight_fails_on_missing_repo(tmp_path: Path):
    result = _run(str(tmp_path / "nonexistent"))
    assert result.returncode == 1
    assert "FAIL" in result.stderr
    assert "not found" in result.stderr


def test_preflight_fails_without_venv(tmp_path: Path):
    """An existing directory with no .venv → a clear venv-missing failure."""
    result = _run(str(tmp_path))
    assert result.returncode == 1
    assert "FAIL" in result.stderr
    assert "venv" in result.stderr.lower()
