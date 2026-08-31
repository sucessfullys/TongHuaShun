"""GPU inventory probe — parses ``nvidia-smi``. Never raises."""

from __future__ import annotations

import re
import subprocess


def probe_gpus(gpu_ids: list[int] | None = None) -> dict:
    """Inventory the local GPUs via ``nvidia-smi``.

    ``gpu_ids`` is the operator-requested subset; ``per_gpu_memory_gb`` is the
    conservative minimum over that subset. On any failure ``probe_ok`` is
    ``False`` and ``error`` explains why — the caller then asks the operator to
    enter hardware values manually.
    """
    requested = list(gpu_ids) if gpu_ids else []
    result: dict = {
        "probe_ok": False,
        "gpu_model": "",
        "all_gpu_ids": [],
        "requested_gpu_ids": requested,
        "per_gpu_memory_gb": 0.0,
        "driver_version": "",
        "cuda_version": "",
        "gpus": [],
        "error": "",
    }

    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        result["error"] = "nvidia-smi not found on PATH"
        return result
    except subprocess.TimeoutExpired:
        result["error"] = "nvidia-smi timed out"
        return result
    except OSError as exc:  # pragma: no cover - defensive
        result["error"] = f"nvidia-smi failed to launch: {exc}"
        return result

    if proc.returncode != 0:
        result["error"] = (
            f"nvidia-smi exited {proc.returncode}: {proc.stderr.strip()}"
        )
        return result

    gpus: list[dict] = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            idx = int(parts[0])
            mem_mb = float(parts[2])
        except ValueError:
            continue
        gpus.append({
            "index": idx,
            "name": parts[1],
            "memory_gb": round(mem_mb / 1024.0, 1),
            "driver_version": parts[3],
        })

    if not gpus:
        result["error"] = "nvidia-smi returned no GPUs"
        return result

    result["probe_ok"] = True
    result["gpus"] = gpus
    result["all_gpu_ids"] = [g["index"] for g in gpus]
    result["gpu_model"] = gpus[0]["name"]
    result["driver_version"] = gpus[0]["driver_version"]

    subset = requested or result["all_gpu_ids"]
    mems = [g["memory_gb"] for g in gpus if g["index"] in subset]
    if not mems:
        mems = [g["memory_gb"] for g in gpus]
    result["per_gpu_memory_gb"] = min(mems)

    missing = [i for i in requested if i not in result["all_gpu_ids"]]
    if missing:
        result["error"] = (
            f"requested GPU ids {missing} are not visible "
            f"(visible: {result['all_gpu_ids']})"
        )

    result["cuda_version"] = _probe_cuda_version()
    return result


def _probe_cuda_version() -> str:
    """CUDA runtime version from the plain ``nvidia-smi`` header (best effort)."""
    try:
        proc = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=30
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ""
    if proc.returncode != 0:
        return ""
    match = re.search(r"CUDA Version:\s*([0-9.]+)", proc.stdout)
    return match.group(1) if match else ""
