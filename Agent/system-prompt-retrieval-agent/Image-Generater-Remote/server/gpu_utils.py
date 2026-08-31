"""GPU memory probe and process management utilities.

Designed to run on the remote 3H100 server. All functions degrade gracefully
when nvidia-smi or GPU hardware is unavailable, so the module can be imported
on macOS without crashing.
"""

from __future__ import annotations

import subprocess
import time
from typing import Optional


# ── Internal helpers ──────────────────────────────────────────────────────────

def _run_nvidia_smi(extra_args: list[str]) -> Optional[str]:
    """Run nvidia-smi with extra_args; return stdout or None on failure."""
    try:
        result = subprocess.run(
            ["nvidia-smi"] + extra_args,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _find_pids(pattern: str) -> list[int]:
    """Return PIDs of processes matching pattern via pgrep."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", pattern],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return [int(p) for p in result.stdout.strip().split() if p.strip().isdigit()]
        return []
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []


# ── Public API ────────────────────────────────────────────────────────────────

def get_gpu_snapshot(gpu_id: Optional[int] = None) -> list[dict]:
    """Return GPU memory/utilization info for each GPU (or a single GPU).

    Uses ``nvidia-smi --query-gpu=index,memory.used,memory.total,memory.free,
    utilization.gpu --format=csv,noheader,nounits``.

    Returns an empty list when nvidia-smi is unavailable or returns no data.

    Each dict has keys:
        gpu_id (int), mem_used_gib (float), mem_total_gib (float),
        mem_free_gib (float), utilization_pct (float or None).
    """
    output = _run_nvidia_smi([
        "--query-gpu=index,memory.used,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    if output is None:
        return []

    snapshots: list[dict] = []
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gid = int(parts[0])
            mem_used_mib = float(parts[1])
            mem_total_mib = float(parts[2])
            mem_free_mib = float(parts[3])
            # utilization.gpu can be "[Not Supported]" on some hardware
            try:
                util_pct: Optional[float] = float(parts[4]) if len(parts) >= 5 else None
            except (ValueError, IndexError):
                util_pct = None

            if gpu_id is not None and gid != gpu_id:
                continue

            snapshots.append({
                "gpu_id": gid,
                "mem_used_gib": round(mem_used_mib / 1024.0, 3),
                "mem_total_gib": round(mem_total_mib / 1024.0, 3),
                "mem_free_gib": round(mem_free_mib / 1024.0, 3),
                "utilization_pct": util_pct,
            })
        except (ValueError, IndexError):
            continue

    return snapshots


def assert_memory_below_ceiling(gpu_id: int, ceiling_pct: int = 90) -> bool:
    """Check that GPU memory usage is below ``ceiling_pct`` percent.

    Returns True when usage is safely below the ceiling.
    Raises RuntimeError when usage meets or exceeds the ceiling.
    Returns True (safe) when nvidia-smi is unavailable — callers that need a
    hard guarantee must handle the unavailable case separately.
    """
    snapshots = get_gpu_snapshot(gpu_id=gpu_id)
    if not snapshots:
        # No data available — cannot assert; treat as safe so callers don't
        # block on non-GPU machines during testing.
        return True

    snap = snapshots[0]
    total = snap["mem_total_gib"]
    used = snap["mem_used_gib"]
    if total <= 0:
        return True

    used_pct = (used / total) * 100.0
    if used_pct >= ceiling_pct:
        raise RuntimeError(
            f"GPU {gpu_id} memory usage {used_pct:.1f}% meets or exceeds "
            f"ceiling {ceiling_pct}% (used={used:.2f} GiB / total={total:.2f} GiB)"
        )
    return True


def kill_nogpualarm(pattern: str = "NoGPUAlarmNew.py") -> dict:
    """Kill processes matching *pattern*.

    Strategy:
    1. ``pkill -f {pattern}`` (SIGTERM).
    2. Wait 2 seconds.
    3. Check if any processes are still running.
    4. ``pkill -9 -f {pattern}`` (SIGKILL) only if still running.

    Returns a dict with keys:
        killed (bool): True if any processes were found and signalled.
        signal_used (str): "SIGTERM", "SIGKILL", "SIGTERM+SIGKILL", or "none".
        pids_found (list[int]): PIDs found *before* the first kill attempt.
    """
    pids_before = _find_pids(pattern)

    if not pids_before:
        return {"killed": False, "signal_used": "none", "pids_found": []}

    # SIGTERM first
    try:
        subprocess.run(["pkill", "-f", pattern], capture_output=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    signal_used = "SIGTERM"

    time.sleep(2)

    pids_after = _find_pids(pattern)
    if pids_after:
        # Still running — escalate to SIGKILL
        try:
            subprocess.run(["pkill", "-9", "-f", pattern], capture_output=True, timeout=10)
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass
        signal_used = "SIGTERM+SIGKILL"

    return {
        "killed": True,
        "signal_used": signal_used,
        "pids_found": pids_before,
    }


def verify_post_unload_free(gpu_id: int, min_free_gib: float) -> dict:
    """Check whether free VRAM on *gpu_id* is at least *min_free_gib* GiB.

    Returns a dict with keys:
        ok (bool): True when free VRAM >= min_free_gib.
        free_gib (float): Actual free VRAM in GiB (0.0 if unavailable).
        min_required_gib (float): The threshold that was checked.
    """
    snapshots = get_gpu_snapshot(gpu_id=gpu_id)
    if not snapshots:
        # nvidia-smi unavailable — report 0 free but still return structured result
        return {"ok": False, "free_gib": 0.0, "min_required_gib": min_free_gib}

    free_gib = snapshots[0]["mem_free_gib"]
    return {
        "ok": free_gib >= min_free_gib,
        "free_gib": free_gib,
        "min_required_gib": min_free_gib,
    }
