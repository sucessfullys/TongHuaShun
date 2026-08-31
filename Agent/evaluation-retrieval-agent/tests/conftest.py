"""Shared pytest fixtures for ERA Stage 0 tests."""

from __future__ import annotations

from pathlib import Path

import pytest


def touch(path: Path) -> None:
    """Create an empty file (probes classify by extension, not content)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


@pytest.fixture
def per_sample_root(tmp_path: Path) -> Path:
    """3 per-sample dirs, each with two inputs + one generated output."""
    root = tmp_path / "tryon_results"
    for i in range(3):
        d = root / f"sample_{i:03d}"
        touch(d / "input_cloth.png")
        touch(d / "input_model.png")
        touch(d / "tryon_result_flux2klein.png")
    return root


@pytest.fixture
def flat_root(tmp_path: Path) -> Path:
    """A single flat directory of images."""
    root = tmp_path / "outputs"
    for i in range(5):
        touch(root / f"img_{i:03d}.png")
    return root


@pytest.fixture
def multi_method_roots(tmp_path: Path) -> list[str]:
    """Two method directories passed as separate roots, each with per-sample
    subdirs holding a source input + a generated result."""
    roots = []
    for method in ("method_a", "method_b"):
        d = tmp_path / method
        for i in range(4):
            leaf = d / f"sample_{i}"
            touch(leaf / "source.png")
            touch(leaf / "result.png")
        roots.append(str(d))
    return roots


@pytest.fixture
def nested_methods_root(tmp_path: Path) -> Path:
    """One root with 3 method dirs, each nesting sample dirs 3 levels deep
    (category/scene/sample). Method 3 uses a different output filename and
    also drops intermediate `_agent_N` artifacts in each sample dir."""
    root = tmp_path / "tryon_results"
    methods = {
        "TryOn_results": "tryon_result_flux2klein.png",
        "TryOn_results_PE": "tryon_result_flux2klein.png",
        "TryOn_results_PE_v5": "tryon_result_flux2klein_agent.png",
    }
    for method, output in methods.items():
        for category in ("dress", "upper"):
            for s in range(2):
                leaf = (root / method / category / f"{category}_scene"
                        / f"{category}_{s}")
                touch(leaf / "input_cloth.png")
                touch(leaf / "input_model.png")
                touch(leaf / output)
                if method == "TryOn_results_PE_v5":
                    for i in range(3):
                        touch(leaf / f"tryon_result_flux2klein_agent_{i}.png")
    return root


def valid_params(tmp_path: Path, per_sample_root: Path) -> dict:
    """A complete, valid set of init-workspace parameters."""
    return {
        "project_name": "tryon-eval",
        "mission": "Evaluate virtual try-on transformed images.",
        "task_family": "editing",
        "task_adapter": "virtual_tryon",
        "created_at": "2026-01-01T00:00:00+00:00",
        "hardware": {
            "gpu_model": "NVIDIA A100-SXM4-80GB",
            "visible_gpu_ids": [4, 5, 6, 7],
            "reserve_gpu_ids": [],
            "max_gpus_per_run": 4,
            "per_gpu_memory_gb": 80.0,
            "driver_version": "535.104.05",
            "cuda_version": "12.2",
            "probe_ok": True,
        },
        "checkpoints": {
            "local_model_root": "/mnt/model",
            "detected": ["qwen2-vl-7b"],
            "user_checkpoints": [],
        },
        "data": {
            "data_root": str(per_sample_root),
            "layout": "per_sample_dirs",
            "methods": [
                {"method_id": "flux2klein", "path": str(per_sample_root),
                 "output_file": "tryon_result_flux2klein.png"},
            ],
            "sample_glob": "*",
            "sample_count": 3,
            "iter_sample_count": 3,
            "use_annotation_evidence": True,
            "input_roles": {
                "input_cloth": "input_cloth.png",
                "input_model": "input_model.png",
            },
            "input_root": "",
            "sample_key": "relpath",
        },
        "credentials": {
            "env_file": "/abs/.env",
            "openai": True,
            "anthropic": True,
            "google": False,
        },
        "budget": {"api_cost_cap_usd": 0.0, "wallclock_cap_hours": 24.0},
        "probe": {
            "gpu": {"probe_ok": True},
            "data": {"layout": "per_sample_dirs"},
            "checkpoints": {"detected": ["qwen2-vl-7b"]},
            "credentials": {"openai": True},
        },
    }
