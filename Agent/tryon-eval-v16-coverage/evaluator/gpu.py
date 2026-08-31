#!/usr/bin/env python3
"""GPU environment helpers — watchdog teardown + free-card selection.

This box runs a GPU watchdog (``NoGPUAlarmNew.py``) that holds otherwise-idle
cards at ~35 GB / 100 % util. vLLM's free-memory check fails while it runs, so
it must be killed BEFORE serving; it should be restarted after the evaluation
(``restart_watchdog``) so idle cards stay protected.
"""
from __future__ import annotations

import subprocess
import time

WATCHDOG_PATTERN = "NoGPUAlarmNew.py"
WATCHDOG_START = "cd /mnt/image-edit/datasets/xywang/code/GPU_OCU/ && bash start.sh"
# A judge with tensor_parallel=4 needs ~70 GB free per card on H100-80GB at
# gpu_memory_utilization 0.92.
MIN_FREE_MB = 70000


def _run(cmd: list, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def watchdog_alive() -> bool:
    try:
        return _run(["pgrep", "-f", WATCHDOG_PATTERN]).returncode == 0
    except Exception:
        return False


def kill_watchdog(log=print) -> bool:
    """Kill the GPU watchdog (plain pkill, then sudo -n). True when gone."""
    if not watchdog_alive():
        return True
    log("killing the GPU watchdog (it holds idle cards and blocks vLLM)...")
    try:
        _run(["pkill", "-9", "-f", WATCHDOG_PATTERN])
    except Exception:
        pass
    time.sleep(2)
    if watchdog_alive():
        try:
            _run(["sudo", "-n", "pkill", "-9", "-f", WATCHDOG_PATTERN])
        except Exception:
            pass
        time.sleep(2)
    alive = watchdog_alive()
    if alive:
        log("! the watchdog is still alive after sudo pkill — kill it "
            "manually:\n    sudo pkill -9 -f \"python3 -u NoGPUAlarmNew.py\"")
    return not alive


def restart_watchdog(log=print) -> None:
    """Re-arm the watchdog if it is not running (best-effort)."""
    if watchdog_alive():
        return
    try:
        subprocess.Popen(WATCHDOG_START, shell=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
        log("restarted the GPU watchdog")
    except Exception as e:
        log(f"! could not restart the GPU watchdog: {e}")


def gpu_table() -> list[dict]:
    """Per-GPU snapshot from nvidia-smi (index, used/total MB, util %)."""
    out = _run(["nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits"])
    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpus.append({"index": int(parts[0]), "used_mb": int(parts[1]),
                         "total_mb": int(parts[2]), "util": int(parts[3])})
        except ValueError:
            continue
    return gpus


def format_gpu_table(gpus: list[dict]) -> str:
    lines = ["  GPU   used/total MB   util%"]
    for g in gpus:
        lines.append(f"  {g['index']:>3}   {g['used_mb']:>6}/{g['total_mb']:<6}"
                     f"   {g['util']:>4}")
    return "\n".join(lines)


def pick_metric_gpu(min_free_mb: int = 6000,
                    exclude: list[int] | None = None) -> int | None:
    """A card with headroom for the small metric models (DINO/LPIPS), or
    None for CPU. Judge cards are excluded — at 0.92 memory utilization a
    cuBLAS handle on them fails to allocate."""
    exclude = set(exclude or ())
    best = None
    best_free = min_free_mb
    for g in gpu_table():
        if g["index"] in exclude:
            continue
        free = g["total_mb"] - g["used_mb"]
        if free >= best_free:
            best, best_free = g["index"], free
    return best


def pick_gpus(count: int, min_free_mb: int = MIN_FREE_MB,
              log=print) -> list[int]:
    """Choose ``count`` cards with the most free memory, or exit with a table."""
    gpus = gpu_table()
    if not gpus:
        raise SystemExit("nvidia-smi returned no GPUs — is this a GPU box?")
    free = sorted(
        (g for g in gpus if (g["total_mb"] - g["used_mb"]) >= min_free_mb),
        key=lambda g: g["used_mb"])
    if len(free) < count:
        log(format_gpu_table(gpus))
        raise SystemExit(
            f"need {count} GPUs with >= {min_free_mb} MB free, found "
            f"{len(free)}. Free some cards (or attach to a running judge "
            f"with --endpoint) and retry.")
    chosen = sorted(g["index"] for g in free[:count])
    log(f"using GPUs {chosen}")
    return chosen
