"""Tests for the GPU probe (monkeypatched subprocess — no real GPU needed)."""

from __future__ import annotations

import subprocess

from era.probe import gpu as gpu_mod
from era.probe import probe_gpus

_CSV = (
    "0, NVIDIA A100-SXM4-80GB, 81920, 535.104.05\n"
    "1, NVIDIA A100-SXM4-80GB, 81920, 535.104.05\n"
)


def _fake_run_ok(cmd, **_kwargs):
    if any("--query-gpu" in str(c) for c in cmd):
        return subprocess.CompletedProcess(cmd, 0, stdout=_CSV, stderr="")
    return subprocess.CompletedProcess(
        cmd, 0, stdout="| NVIDIA-SMI 535  CUDA Version: 12.2  |", stderr=""
    )


def test_probe_gpus_ok(monkeypatch):
    monkeypatch.setattr(gpu_mod.subprocess, "run", _fake_run_ok)
    result = probe_gpus([0, 1])
    assert result["probe_ok"] is True
    assert result["all_gpu_ids"] == [0, 1]
    assert result["gpu_model"].startswith("NVIDIA A100")
    assert result["per_gpu_memory_gb"] == 80.0
    assert result["driver_version"] == "535.104.05"
    assert result["cuda_version"] == "12.2"
    assert result["error"] == ""


def test_probe_gpus_missing_binary(monkeypatch):
    def _raise(cmd, **_kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(gpu_mod.subprocess, "run", _raise)
    result = probe_gpus([0])
    assert result["probe_ok"] is False
    assert "not found" in result["error"]


def test_probe_gpus_requested_id_not_visible(monkeypatch):
    monkeypatch.setattr(gpu_mod.subprocess, "run", _fake_run_ok)
    result = probe_gpus([0, 9])
    assert result["probe_ok"] is True
    assert "9" in result["error"]  # mismatch flagged
