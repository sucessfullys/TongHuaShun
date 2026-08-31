"""Tests for the checkpoint probe."""

from __future__ import annotations

from pathlib import Path

from conftest import touch

from era.probe import probe_checkpoints


def test_detects_model_repos(tmp_path: Path):
    root = tmp_path / "models"
    # a HuggingFace-style repo with config.json
    touch(root / "qwen2-vl-7b" / "config.json")
    # a repo identified by weight files
    touch(root / "internvl-8b" / "model.safetensors")
    # a non-model directory
    (root / "scratch").mkdir(parents=True)
    result = probe_checkpoints(str(root))
    assert result["probe_ok"] is True
    assert set(result["detected"]) == {"qwen2-vl-7b", "internvl-8b"}


def test_missing_model_root(tmp_path: Path):
    result = probe_checkpoints(str(tmp_path / "nope"))
    assert result["probe_ok"] is False
    assert "not found" in result["error"]


def test_empty_model_root(tmp_path: Path):
    root = tmp_path / "models"
    root.mkdir()
    result = probe_checkpoints(str(root))
    assert result["probe_ok"] is True
    assert result["detected"] == []
