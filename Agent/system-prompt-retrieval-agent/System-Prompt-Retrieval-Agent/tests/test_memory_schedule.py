"""S02.07 — Tests for memory.schedule: atomic write, default fields."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from system_prompt_retrieval_agent.memory.schedule import (
    default_schedule,
    read_schedule,
    write_schedule,
)


def test_default_schedule_has_schema_version():
    sched = default_schedule()
    assert "schema_version" in sched
    assert sched["schema_version"] != ""


def test_default_schedule_fields():
    sched = default_schedule(max_rounds=8)
    assert sched["max_rounds"] == 8
    assert sched["status"] == "in_progress"
    assert sched["current_round"] == 0
    assert sched["fallback_prompt_generation"] is False
    assert sched["consecutive_fallback_rounds"] == 0
    assert isinstance(sched["blockers"], list)
    assert sched["stop_reason"] is None
    assert sched["next_action"] is not None
    assert "last_updated" in sched
    assert isinstance(sched["history"], list)


def test_write_then_read_roundtrip(tmp_path: Path):
    path = tmp_path / "schedule.json"
    data = default_schedule(max_rounds=3)
    write_schedule(path, data)
    loaded = read_schedule(path)
    assert loaded == data


def test_atomic_write_uses_tmp_then_rename(tmp_path: Path, monkeypatch):
    """Verify that write_schedule writes to .tmp file first then renames."""
    path = tmp_path / "schedule.json"
    tmp_path_file = path.with_suffix(path.suffix + ".tmp")

    written_tmp = []

    real_replace = __import__("os").replace

    def mock_replace(src, dst):
        written_tmp.append(src)
        real_replace(src, dst)

    monkeypatch.setattr("system_prompt_retrieval_agent.memory.schedule.os.replace", mock_replace)

    data = default_schedule()
    write_schedule(path, data)

    # os.replace should have been called with the .tmp file as source
    assert len(written_tmp) == 1
    assert Path(written_tmp[0]) == tmp_path_file

    # Final file must exist with correct content
    assert path.exists()
    assert not tmp_path_file.exists()
    loaded = json.loads(path.read_text())
    assert loaded["schema_version"] == data["schema_version"]


def test_no_tmp_file_left_after_write(tmp_path: Path):
    path = tmp_path / "schedule.json"
    write_schedule(path, default_schedule())
    tmp_file = path.with_suffix(path.suffix + ".tmp")
    assert not tmp_file.exists()


def test_read_schedule_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        read_schedule(tmp_path / "nonexistent.json")


def test_overwrite_updates_content(tmp_path: Path):
    path = tmp_path / "schedule.json"
    sched = default_schedule()
    write_schedule(path, sched)

    sched2 = {**sched, "current_round": 3, "status": "completed"}
    write_schedule(path, sched2)

    loaded = read_schedule(path)
    assert loaded["current_round"] == 3
    assert loaded["status"] == "completed"
