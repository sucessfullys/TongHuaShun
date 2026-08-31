"""Tests for the annotation server lifecycle (era/orchestration/annotate.py).

These tests do **not** spawn a real subprocess. They monkeypatch the
``subprocess.Popen`` call and the readiness poll so the lifecycle logic
(PID file write, idempotency, stop) is exercised in-process.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from era.orchestration import annotate as orchestrate


class _FakePopen:
    """Stand-in for subprocess.Popen — records construction, exposes pid."""
    _next_pid = 99000

    def __init__(self, *args, **kwargs):
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.args = args
        self.kwargs = kwargs

    def poll(self):
        return None  # "still running"

    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0


@pytest.fixture
def _stub_lifecycle(monkeypatch):
    """Replace subprocess.Popen + the readiness poll + alive check + killpg.

    Each test wires in its own Popen so we can vary the fake PID. The
    readiness poll always succeeds quickly. PIDs we report alive are
    tracked in ``_alive``; ``stop_annotate`` calls ``os.killpg`` which we
    drop into ``_killed``.
    """
    state = {"alive": set(), "killed": []}

    def fake_popen(*args, **kwargs):
        proc = _FakePopen(*args, **kwargs)
        state["alive"].add(proc.pid)
        return proc

    def fake_pid_alive(pid):
        return pid in state["alive"]

    def fake_wait_responsive(host, port, timeout):
        return True

    def fake_port_responds(host, port):
        return True

    def fake_pick_port(host, preferred):
        return preferred or orchestrate.DEFAULT_PORT

    def fake_killpg(pgid, sig):
        # In the test, pgid == pid (no real process group)
        state["killed"].append(pgid)
        state["alive"].discard(pgid)

    def fake_getpgid(pid):
        return pid

    monkeypatch.setattr(orchestrate.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(orchestrate, "_pid_alive", fake_pid_alive)
    monkeypatch.setattr(orchestrate, "_wait_responsive", fake_wait_responsive)
    monkeypatch.setattr(orchestrate, "_port_responds", fake_port_responds)
    monkeypatch.setattr(orchestrate, "_pick_port", fake_pick_port)
    monkeypatch.setattr(orchestrate.os, "killpg", fake_killpg)
    monkeypatch.setattr(orchestrate.os, "getpgid", fake_getpgid)
    return state


def _make_dataset(root: Path, methods: list[str]) -> None:
    """Minimal dataset for the validator (just needs a method subdir)."""
    for m in methods:
        (root / m).mkdir(parents=True, exist_ok=True)


# ---- validation --------------------------------------------------------

def test_serve_rejects_missing_dataset(tmp_path: Path, _stub_lifecycle):
    r = orchestrate.serve_annotate(tmp_path / "does-not-exist")
    assert r["error"] == "no_dataset"


def test_serve_rejects_empty_dataset(tmp_path: Path, _stub_lifecycle):
    # Empty dir is not a usable dataset
    r = orchestrate.serve_annotate(tmp_path)
    assert r["error"] == "empty_dataset"


# ---- serve --------------------------------------------------------------

def test_serve_starts_and_writes_pidfile(tmp_path: Path, _stub_lifecycle):
    _make_dataset(tmp_path, ["method_a"])
    r = orchestrate.serve_annotate(tmp_path)
    assert r["status"] == "ok"
    assert r["host"] == "127.0.0.1"
    assert r["port"] == orchestrate.DEFAULT_PORT
    assert r["url"].startswith("http://127.0.0.1:")
    assert r["responsive"] is True
    pidfile = tmp_path / orchestrate.PIDFILE_NAME
    assert pidfile.is_file()
    rec = json.loads(pidfile.read_text(encoding="utf-8"))
    assert rec["pid"] == r["pid"]
    assert rec["dataset_root"] == str(tmp_path.resolve())


def test_serve_idempotent_when_already_running(tmp_path: Path, _stub_lifecycle):
    _make_dataset(tmp_path, ["method_a"])
    first = orchestrate.serve_annotate(tmp_path)
    second = orchestrate.serve_annotate(tmp_path)
    assert second["status"] == "already_running"
    assert second["pid"] == first["pid"]


def test_serve_replaces_stale_pidfile(tmp_path: Path, _stub_lifecycle):
    """A pidfile whose PID is dead must be cleared so a fresh launch happens."""
    _make_dataset(tmp_path, ["method_a"])
    # Hand-write a pidfile for a non-running PID (not in _alive)
    pidfile = tmp_path / orchestrate.PIDFILE_NAME
    pidfile.write_text(json.dumps({
        "pid": 1, "host": "127.0.0.1", "port": orchestrate.DEFAULT_PORT,
    }), encoding="utf-8")
    r = orchestrate.serve_annotate(tmp_path)
    assert r["status"] == "ok"
    # New PID, not the stale 1
    assert r["pid"] != 1


# ---- status -------------------------------------------------------------

def test_status_reports_running_after_serve(tmp_path: Path, _stub_lifecycle):
    _make_dataset(tmp_path, ["method_a"])
    served = orchestrate.serve_annotate(tmp_path)
    s = orchestrate.annotate_status(tmp_path)
    assert s["server"]["running"] is True
    assert s["server"]["responsive"] is True
    assert s["server"]["pid"] == served["pid"]


def test_status_reports_idle_with_no_pidfile(tmp_path: Path, _stub_lifecycle):
    _make_dataset(tmp_path, ["method_a"])
    s = orchestrate.annotate_status(tmp_path)
    assert s["server"]["running"] is False
    assert s["server"]["pid"] is None


# ---- stop ---------------------------------------------------------------

def test_stop_kills_and_unlinks_pidfile(tmp_path: Path, _stub_lifecycle):
    _make_dataset(tmp_path, ["method_a"])
    served = orchestrate.serve_annotate(tmp_path)
    r = orchestrate.stop_annotate(tmp_path)
    assert r["status"] == "ok"
    assert r["killed"] is True
    assert r["pid"] == served["pid"]
    assert not (tmp_path / orchestrate.PIDFILE_NAME).is_file()
    assert served["pid"] in _stub_lifecycle["killed"]


def test_stop_no_op_when_not_running(tmp_path: Path, _stub_lifecycle):
    _make_dataset(tmp_path, ["method_a"])
    r = orchestrate.stop_annotate(tmp_path)
    assert r["status"] == "not_running"


# ---- CLI roundtrip ------------------------------------------------------

def test_cli_serve_annotate_then_stop(
    tmp_path: Path, _stub_lifecycle, monkeypatch, capsys,
):
    """End-to-end through era.cli: serve → status → stop."""
    import io
    import json as _json

    from era.cli import main

    _make_dataset(tmp_path, ["method_a"])

    monkeypatch.setattr("sys.stdin", io.StringIO(_json.dumps(
        {"dataset_root": str(tmp_path)})))
    code = main(["serve-annotate"])
    out = _json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["status"] == "ok"

    monkeypatch.setattr("sys.stdin", io.StringIO(_json.dumps(
        {"dataset_root": str(tmp_path)})))
    code = main(["annotate-status"])
    out = _json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["server"]["running"] is True

    monkeypatch.setattr("sys.stdin", io.StringIO(_json.dumps(
        {"dataset_root": str(tmp_path)})))
    code = main(["stop-annotate"])
    out = _json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["status"] == "ok"
    assert not (tmp_path / orchestrate.PIDFILE_NAME).is_file()


def test_cli_serve_annotate_missing_dataset_root(
    monkeypatch, capsys,
):
    import io
    import json as _json

    from era.cli import main

    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
    code = main(["serve-annotate"])
    out = _json.loads(capsys.readouterr().out)
    assert code == 1
    assert out["error"] == "missing_dataset_root"
