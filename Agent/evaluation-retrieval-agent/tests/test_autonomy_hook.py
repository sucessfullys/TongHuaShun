"""Tests for ERA Phase D-3 — the PreToolUse autonomy hook.

The hook (era.cli check-autonomy) is wired into .claude/settings.json
via era_claude_settings(). It reads <cwd>/status.json and decides
allow/block for AskUserQuestion calls:

- exit 0 (allow): awaiting_human (legitimate Stage 8) / idle / stopped /
  done / no status file / malformed JSON (fail-open)
- exit 2 (block): run_state == running or blocked (mid-loop, iron rule
  forbids asking).
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

from era.cli import main


def _write_status(dir: Path, run_state: str | None) -> None:
    """Stub a workspace status.json with the given run_state."""
    payload = {"stage": "research", "stage_index": 1, "iteration": 1}
    if run_state is not None:
        payload["run_state"] = run_state
    (dir / "status.json").write_text(
        json.dumps(payload), encoding="utf-8",
    )


def _run_check_autonomy(monkeypatch, cwd: Path) -> tuple[int, str]:
    """Invoke `era.cli check-autonomy` with cwd as the workspace; return
    (exit_code, stderr)."""
    monkeypatch.chdir(cwd)
    # The CLI takes no stdin for check-autonomy.
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    err_buf = io.StringIO()
    monkeypatch.setattr("sys.stderr", err_buf)
    rc = main(["check-autonomy"])
    return rc, err_buf.getvalue()


def test_check_autonomy_allows_awaiting_human(tmp_path: Path, monkeypatch):
    """Legitimate Stage 8 site: the era-human-feedback skill sets
    run_state=awaiting_human BEFORE its prompt — the hook must allow."""
    _write_status(tmp_path, "awaiting_human")
    rc, err = _run_check_autonomy(monkeypatch, tmp_path)
    assert rc == 0
    assert err == ""


def test_check_autonomy_allows_idle(tmp_path: Path, monkeypatch):
    _write_status(tmp_path, "idle")
    rc, _ = _run_check_autonomy(monkeypatch, tmp_path)
    assert rc == 0


def test_check_autonomy_allows_stopped(tmp_path: Path, monkeypatch):
    _write_status(tmp_path, "stopped")
    rc, _ = _run_check_autonomy(monkeypatch, tmp_path)
    assert rc == 0


def test_check_autonomy_allows_done(tmp_path: Path, monkeypatch):
    _write_status(tmp_path, "done")
    rc, _ = _run_check_autonomy(monkeypatch, tmp_path)
    assert rc == 0


def test_check_autonomy_allows_missing_status(tmp_path: Path, monkeypatch):
    """Fail-open: no status.json in cwd → operator using Claude Code
    interactively outside /era:start. AskUserQuestion is legitimate."""
    # tmp_path has no status.json.
    rc, _ = _run_check_autonomy(monkeypatch, tmp_path)
    assert rc == 0


def test_check_autonomy_allows_missing_run_state(tmp_path: Path, monkeypatch):
    """A status.json that exists but doesn't carry run_state should also
    fail-open — we don't know enough to block."""
    (tmp_path / "status.json").write_text(
        json.dumps({"stage": "x", "stage_index": 1}), encoding="utf-8")
    rc, _ = _run_check_autonomy(monkeypatch, tmp_path)
    assert rc == 0


def test_check_autonomy_allows_malformed_status(tmp_path: Path, monkeypatch):
    """Fail-open on unparseable JSON — better than silently bricking
    AskUserQuestion forever."""
    (tmp_path / "status.json").write_text("{ not valid }", encoding="utf-8")
    rc, _ = _run_check_autonomy(monkeypatch, tmp_path)
    assert rc == 0


def test_check_autonomy_blocks_running(tmp_path: Path, monkeypatch):
    """The user's exact bug: run_state=running, agent tries to ask
    'should I proceed with Stage 6'. Block must fire."""
    _write_status(tmp_path, "running")
    rc, err = _run_check_autonomy(monkeypatch, tmp_path)
    assert rc == 2
    assert "iron autonomy rule" in err.lower()
    assert "askuserquestion" in err.lower()


def test_check_autonomy_blocks_blocked_state(tmp_path: Path, monkeypatch):
    """Pre-Stage-8 blocked state is the agent's signal to auto-revise,
    not to ask. Still blocked."""
    _write_status(tmp_path, "blocked")
    rc, _ = _run_check_autonomy(monkeypatch, tmp_path)
    assert rc == 2
