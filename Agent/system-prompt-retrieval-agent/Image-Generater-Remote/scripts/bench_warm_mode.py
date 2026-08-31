#!/usr/bin/env python3
"""B8 CPU-warm lifecycle benchmark.

Measures cold-load and warm-to-GPU load times for Gemma-31B, FLUX-9B, and
Qwen-VL-8B on the 3h100 remote host. Intended as an OPERATOR-RUN benchmark;
never starts the production controller.

Decision rule (S10b.02):
    If (warm_to_gpu_s < 0.5 * cold_load_s) AND (host_ram_peak_gib < 64):
        recommend lifecycle_mode=warm
    Else:
        recommend lifecycle_mode=cold  (V0.2.1 default)

Usage:
    python3 scripts/bench_warm_mode.py \\
        --gemma-path  /path/to/gemma \\
        --flux-path   /path/to/flux \\
        --qwen-path   /path/to/qwen \\
        --gpu-ids 0,1,2 \\
        --output-csv  /path/to/results.csv

Output CSV columns per model:
    model, cold_load_s, warm_to_gpu_s, gpu_unload_s,
    gpu_vram_peak_gib, host_ram_peak_gib, recommendation
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
import urllib.request
from typing import Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("bench_warm_mode")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Ports used exclusively for this benchmark (disjoint from production ports).
_BENCH_PORTS = [9431, 9432, 9433]

# Minimum free VRAM (GiB) required before each measurement phase.
_MIN_FREE_VRAM_GIB = 40.0

# Host-RAM headroom threshold (GiB) for the warm-mode decision (S10b.02).
_WARM_RAM_CAP_GIB = 64.0

# CSV column order.
_CSV_COLUMNS = [
    "model",
    "cold_load_s",
    "warm_to_gpu_s",
    "gpu_unload_s",
    "gpu_vram_peak_gib",
    "host_ram_peak_gib",
    "recommendation",
]

# ---------------------------------------------------------------------------
# Hardware helpers
# ---------------------------------------------------------------------------


def _gpu_free_vram_gib(gpu_ids: list[int]) -> dict[int, float]:
    """Return {gpu_id: free_gib} for the requested GPUs.  Returns {} on error."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            log.warning("nvidia-smi failed; VRAM stats unavailable")
            return {}
        out: dict[int, float] = {}
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            gid = int(parts[0])
            if gid in gpu_ids:
                out[gid] = float(parts[1]) / 1024.0
        return out
    except Exception as exc:
        log.warning("GPU VRAM query failed: %s", exc)
        return {}


def _gpu_used_vram_gib(gpu_ids: list[int]) -> dict[int, float]:
    """Return {gpu_id: used_gib}."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return {}
        out: dict[int, float] = {}
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            gid = int(parts[0])
            if gid in gpu_ids:
                out[gid] = float(parts[1]) / 1024.0
        return out
    except Exception:
        return {}


def _host_ram_available_gib() -> float:
    """Return host RAM available (GiB) using /proc/meminfo (Linux)."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb / (1024.0 ** 2)
    except Exception:
        pass
    return float("nan")


def _host_ram_used_gib() -> float:
    """Return approximate host RAM currently used (GiB)."""
    try:
        with open("/proc/meminfo") as fh:
            info: dict[str, int] = {}
            for line in fh:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1])
        total = info.get("MemTotal", 0)
        available = info.get("MemAvailable", 0)
        return (total - available) / (1024.0 ** 2)
    except Exception:
        return float("nan")


def _verify_gpus_free(gpu_ids: list[int], min_free_gib: float = _MIN_FREE_VRAM_GIB) -> bool:
    free = _gpu_free_vram_gib(gpu_ids)
    for gid in gpu_ids:
        if gid in free and free[gid] < min_free_gib:
            log.warning("GPU %d: only %.1f GiB free (need %.1f)", gid, free[gid], min_free_gib)
            return False
    return True


# ---------------------------------------------------------------------------
# Process helpers (mirrors gpu_pool.py logic, self-contained for the bench)
# ---------------------------------------------------------------------------


def _kill_hogs(pattern: str = "NoGPUAlarmNew.py") -> None:
    """Kill GPU-hogging watchdog processes."""
    try:
        subprocess.run(["pkill", "-f", pattern], capture_output=True, timeout=10)
        time.sleep(1)
        subprocess.run(["pkill", "-9", "-f", pattern], capture_output=True, timeout=10)
    except Exception as exc:
        log.debug("pkill %s: %s", pattern, exc)


def _server_ready(port: int, timeout_s: float = 3.0) -> bool:
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.status == 200
    except Exception:
        return False


def _wait_servers_ready(
    ports: list[int],
    procs: list[subprocess.Popen],
    startup_timeout_s: int = 900,
) -> None:
    """Block until all ports are ready, or raise RuntimeError on timeout/crash."""
    log.info("Waiting for vLLM servers on ports %s ...", ports)
    deadline = time.monotonic() + startup_timeout_s
    ready = [False] * len(ports)
    while time.monotonic() < deadline:
        all_ok = True
        for i, port in enumerate(ports):
            if ready[i]:
                continue
            if _server_ready(port):
                ready[i] = True
                log.info("  port %d ready", port)
            else:
                all_ok = False
                if procs[i].poll() is not None:
                    raise RuntimeError(
                        f"vLLM server on port {port} died "
                        f"(exit code {procs[i].returncode})"
                    )
        if all_ok:
            return
        time.sleep(5)
    not_ready = [ports[i] for i in range(len(ports)) if not ready[i]]
    raise RuntimeError(f"Ports {not_ready} not ready within {startup_timeout_s}s")


def _start_vllm(
    model_path: str,
    gpu_ids: list[int],
    ports: list[int],
    gpu_memory_utilization: float = 0.95,
    max_model_len: int = 8192,
    extra_args: Optional[list[str]] = None,
) -> list[subprocess.Popen]:
    """Start one vLLM server per GPU for benchmarking."""
    assert len(gpu_ids) == len(ports)
    procs = []
    log_dir = os.environ.get("LOG_DIR", "/tmp")
    for port, gid in zip(ports, gpu_ids):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gid)
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--port", str(port),
            "--host", "127.0.0.1",
            "--trust-remote-code",
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--max-model-len", str(max_model_len),
        ]
        if extra_args:
            cmd.extend(extra_args)
        log_file = os.path.join(log_dir, f"bench_vllm_{port}.log")
        log.info("  Starting vLLM GPU=%d port=%d → %s", gid, port, log_file)
        proc = subprocess.Popen(
            cmd, env=env,
            stdout=open(log_file, "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        procs.append(proc)
    return procs


def _stop_procs(procs: list[subprocess.Popen]) -> None:
    """SIGTERM → wait 30 s → SIGKILL entire process group."""
    import signal
    for proc in procs:
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 30
    for proc in procs:
        remaining = max(0, deadline - time.monotonic())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                proc.kill()
            proc.wait()
    log.info("Benchmark vLLM servers stopped.")


# ---------------------------------------------------------------------------
# FLUX subprocess helper (mirrors gpu_pool.start_flux_workers intent)
# ---------------------------------------------------------------------------


def _start_flux_bench(
    model_path: str,
    gpu_ids: list[int],
    tmp_dir: str,
) -> list[subprocess.Popen]:
    """
    Start one FLUX subprocess per GPU using a minimal inline inference script
    that loads the pipeline and immediately saves a tiny test image, then exits.
    This measures disk→GPU load time without running a real production shard.
    """
    procs = []
    for gid in gpu_ids:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gid)
        # Inline Python: load pipeline, save tiny image, exit.
        inline = (
            "import torch, time\n"
            "from diffusers import FluxPipeline\n"
            f"pipe = FluxPipeline.from_pretrained('{model_path}', torch_dtype=torch.bfloat16)\n"
            "pipe = pipe.to('cuda')\n"
            "img = pipe('a red circle', num_inference_steps=1, height=64, width=64).images[0]\n"
            f"img.save('{tmp_dir}/bench_flux_gpu{gid}.png')\n"
        )
        cmd = [sys.executable, "-c", inline]
        log_file = os.path.join(tmp_dir, f"bench_flux_gpu{gid}.log")
        log.info("  Starting FLUX bench worker GPU=%d", gid)
        proc = subprocess.Popen(
            cmd, env=env,
            stdout=open(log_file, "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        procs.append(proc)
    return procs


def _wait_procs(procs: list[subprocess.Popen], timeout_s: int = 900) -> list[int]:
    """Wait for all procs; return exit codes."""
    codes = []
    deadline = time.monotonic() + timeout_s
    for proc in procs:
        remaining = max(1, deadline - time.monotonic())
        try:
            proc.wait(timeout=remaining)
            codes.append(proc.returncode)
        except subprocess.TimeoutExpired:
            log.error("Bench worker PID=%d timed out, killing", proc.pid)
            proc.kill()
            proc.wait()
            codes.append(-1)
    return codes


# ---------------------------------------------------------------------------
# CPU-prefetch helper: load model weights into host RAM via torch.load
# ---------------------------------------------------------------------------


def _prefetch_to_cpu(model_path: str, tmp_dir: str) -> subprocess.Popen:
    """
    Spawn a subprocess that loads a vLLM/transformers model into host RAM on
    the CPU only (no GPU allocation).  The subprocess blocks until terminated.
    This simulates the `cpu_prefetched` lifecycle state.

    For FLUX (diffusers): uses FluxPipeline(..., device="cpu").
    For vLLM-compatible models (Gemma/Qwen): uses AutoModelForCausalLM on CPU.
    """
    inline = (
        "import os, sys, time, signal\n"
        "model_path = sys.argv[1]\n"
        "# Try transformers AutoModel first (Gemma / Qwen)\n"
        "try:\n"
        "    from transformers import AutoModelForCausalLM, AutoTokenizer\n"
        "    print(f'Loading {model_path} to CPU (transformers)...', flush=True)\n"
        "    model = AutoModelForCausalLM.from_pretrained(\n"
        "        model_path, torch_dtype='auto', device_map='cpu',\n"
        "        trust_remote_code=True,\n"
        "    )\n"
        "    print('CPU prefetch ready (transformers)', flush=True)\n"
        "except Exception as e:\n"
        "    print(f'transformers failed ({e}), trying diffusers...', flush=True)\n"
        "    from diffusers import FluxPipeline\n"
        "    import torch\n"
        "    model = FluxPipeline.from_pretrained(\n"
        "        model_path, torch_dtype=torch.bfloat16, device_map='cpu',\n"
        "    )\n"
        "    print('CPU prefetch ready (diffusers)', flush=True)\n"
        "# Block until killed\n"
        "def _noop(sig, frame): pass\n"
        "signal.signal(signal.SIGTERM, _noop)\n"
        "while True: time.sleep(60)\n"
    )
    log_file = os.path.join(tmp_dir, "bench_cpu_prefetch.log")
    proc = subprocess.Popen(
        [sys.executable, "-c", inline, model_path],
        stdout=open(log_file, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc


def _wait_for_cpu_prefetch_ready(proc: subprocess.Popen, timeout_s: int = 900) -> bool:
    """Poll the prefetch log for 'CPU prefetch ready' line."""
    log_file = None
    # Locate log path from proc; we know it's bench_cpu_prefetch.log in tmp_dir
    # but here we scan stdout captured in the Popen — we reopen the file.
    # The log file path is not directly accessible; scan /proc/<pid>/fd instead.
    deadline = time.monotonic() + timeout_s
    log_path = None
    try:
        import glob
        fds = glob.glob(f"/proc/{proc.pid}/fd/*")
        for fd in fds:
            try:
                target = os.readlink(fd)
                if "bench_cpu_prefetch" in target:
                    log_path = target
                    break
            except Exception:
                pass
    except Exception:
        pass

    if log_path is None:
        # Fallback: just sleep proportionally and assume success
        log.warning("Could not determine prefetch log path; sleeping 120s as proxy")
        time.sleep(120)
        return proc.poll() is None

    while time.monotonic() < deadline:
        if proc.poll() is not None:
            log.error("CPU prefetch subprocess died (exit %d)", proc.returncode)
            return False
        try:
            with open(log_path) as fh:
                if "CPU prefetch ready" in fh.read():
                    return True
        except Exception:
            pass
        time.sleep(5)
    log.error("CPU prefetch did not signal ready within %ds", timeout_s)
    return False


# ---------------------------------------------------------------------------
# Per-model benchmark logic
# ---------------------------------------------------------------------------


class _Snapshot:
    """Snapshot of GPU VRAM usage and host RAM usage at a point in time."""

    def __init__(self, gpu_ids: list[int]) -> None:
        self.vram_used = _gpu_used_vram_gib(gpu_ids)
        self.host_ram_used = _host_ram_used_gib()

    def vram_delta(self, baseline: "_Snapshot") -> float:
        """Peak VRAM increase across all tracked GPUs (GiB)."""
        delta = 0.0
        for gid, used in self.vram_used.items():
            base = baseline.vram_used.get(gid, 0.0)
            delta = max(delta, used - base)
        return max(delta, 0.0)

    def host_ram_delta(self, baseline: "_Snapshot") -> float:
        return max(self.host_ram_used - baseline.host_ram_used, 0.0)


def _bench_vllm_model(
    label: str,
    model_path: str,
    gpu_ids: list[int],
    ports: list[int],
    gpu_memory_utilization: float = 0.95,
    max_model_len: int = 8192,
    extra_args: Optional[list[str]] = None,
    tmp_dir: str = "/tmp",
) -> dict:
    """
    Benchmark a vLLM-served model (Gemma or Qwen).

    Returns a dict matching CSV columns (without 'recommendation').
    """
    log.info("=" * 60)
    log.info("Benchmarking %s (vLLM)", label)
    log.info("=" * 60)

    # --- cold load -----------------------------------------------------------
    _kill_hogs()
    time.sleep(3)
    _verify_gpus_free(gpu_ids)

    base_snap = _Snapshot(gpu_ids)
    t0 = time.monotonic()
    procs = _start_vllm(
        model_path, gpu_ids, ports,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        extra_args=extra_args,
    )
    try:
        _wait_servers_ready(ports, procs)
    except RuntimeError as exc:
        log.error("Cold load failed for %s: %s", label, exc)
        _stop_procs(procs)
        return _error_row(label)

    cold_load_s = time.monotonic() - t0
    after_load_snap = _Snapshot(gpu_ids)
    gpu_vram_peak_gib = after_load_snap.vram_delta(base_snap)
    log.info("%s cold_load_s=%.2f  gpu_vram_peak_gib=%.2f",
             label, cold_load_s, gpu_vram_peak_gib)

    # --- gpu unload ----------------------------------------------------------
    t_unload = time.monotonic()
    _stop_procs(procs)
    time.sleep(5)
    gpu_unload_s = time.monotonic() - t_unload
    log.info("%s gpu_unload_s=%.2f", label, gpu_unload_s)

    # --- CPU prefetch --------------------------------------------------------
    _verify_gpus_free(gpu_ids)
    base_ram_snap = _Snapshot(gpu_ids)
    log.info("Prefetching %s to CPU RAM ...", label)
    prefetch_proc = _prefetch_to_cpu(model_path, tmp_dir)
    prefetch_ok = _wait_for_cpu_prefetch_ready(prefetch_proc)
    after_prefetch_snap = _Snapshot(gpu_ids)
    host_ram_peak_gib = after_prefetch_snap.host_ram_delta(base_ram_snap)
    log.info("%s host_ram_peak_gib=%.2f (prefetch ok=%s)",
             label, host_ram_peak_gib, prefetch_ok)

    # --- warm GPU reload -----------------------------------------------------
    base_warm_snap = _Snapshot(gpu_ids)
    t_warm = time.monotonic()
    warm_procs = _start_vllm(
        model_path, gpu_ids, ports,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        extra_args=extra_args,
    )
    try:
        _wait_servers_ready(ports, warm_procs)
    except RuntimeError as exc:
        log.error("Warm reload failed for %s: %s", label, exc)
        _stop_procs(warm_procs)
        _stop_procs([prefetch_proc])
        return _partial_row(label, cold_load_s, None, gpu_unload_s,
                            gpu_vram_peak_gib, host_ram_peak_gib)

    warm_to_gpu_s = time.monotonic() - t_warm
    log.info("%s warm_to_gpu_s=%.2f", label, warm_to_gpu_s)

    # --- cleanup -------------------------------------------------------------
    _stop_procs(warm_procs)
    _stop_procs([prefetch_proc])
    time.sleep(5)

    return {
        "model": label,
        "cold_load_s": round(cold_load_s, 2),
        "warm_to_gpu_s": round(warm_to_gpu_s, 2),
        "gpu_unload_s": round(gpu_unload_s, 2),
        "gpu_vram_peak_gib": round(gpu_vram_peak_gib, 2),
        "host_ram_peak_gib": round(host_ram_peak_gib, 2),
        "recommendation": _decide(warm_to_gpu_s, cold_load_s, host_ram_peak_gib),
    }


def _bench_flux_model(
    label: str,
    model_path: str,
    gpu_ids: list[int],
    tmp_dir: str = "/tmp",
) -> dict:
    """
    Benchmark FLUX (diffusers-based, subprocess-isolated).

    Returns a dict matching CSV columns (without 'recommendation').
    """
    log.info("=" * 60)
    log.info("Benchmarking %s (FLUX/diffusers)", label)
    log.info("=" * 60)

    # --- cold load -----------------------------------------------------------
    _kill_hogs()
    time.sleep(3)
    _verify_gpus_free(gpu_ids)

    base_snap = _Snapshot(gpu_ids)
    t0 = time.monotonic()
    flux_procs = _start_flux_bench(model_path, gpu_ids, tmp_dir)
    exit_codes = _wait_procs(flux_procs, timeout_s=900)
    cold_load_s = time.monotonic() - t0

    after_load_snap = _Snapshot(gpu_ids)
    gpu_vram_peak_gib = after_load_snap.vram_delta(base_snap)

    if any(c != 0 for c in exit_codes):
        log.error("FLUX cold bench returned non-zero exit codes: %s", exit_codes)
        return _error_row(label)

    log.info("%s cold_load_s=%.2f  gpu_vram_peak_gib=%.2f",
             label, cold_load_s, gpu_vram_peak_gib)

    # gpu_unload_s: FLUX subprocesses exit on their own; VRAM is freed by the OS.
    t_unload = time.monotonic()
    time.sleep(5)  # Wait for OS to reclaim VRAM
    gpu_unload_s = time.monotonic() - t_unload
    log.info("%s gpu_unload_s=%.2f (subprocess exit, OS reclaim)", label, gpu_unload_s)

    # --- CPU prefetch --------------------------------------------------------
    _verify_gpus_free(gpu_ids)
    base_ram_snap = _Snapshot(gpu_ids)
    log.info("Prefetching %s to CPU RAM ...", label)
    prefetch_proc = _prefetch_to_cpu(model_path, tmp_dir)
    prefetch_ok = _wait_for_cpu_prefetch_ready(prefetch_proc)
    after_prefetch_snap = _Snapshot(gpu_ids)
    host_ram_peak_gib = after_prefetch_snap.host_ram_delta(base_ram_snap)
    log.info("%s host_ram_peak_gib=%.2f (prefetch ok=%s)",
             label, host_ram_peak_gib, prefetch_ok)

    # --- warm GPU reload -----------------------------------------------------
    t_warm = time.monotonic()
    warm_procs = _start_flux_bench(model_path, gpu_ids, tmp_dir)
    warm_exit_codes = _wait_procs(warm_procs, timeout_s=900)
    warm_to_gpu_s = time.monotonic() - t_warm

    _stop_procs([prefetch_proc])
    time.sleep(3)

    if any(c != 0 for c in warm_exit_codes):
        log.error("FLUX warm bench returned non-zero exit codes: %s", warm_exit_codes)
        return _partial_row(label, cold_load_s, None, gpu_unload_s,
                            gpu_vram_peak_gib, host_ram_peak_gib)

    log.info("%s warm_to_gpu_s=%.2f", label, warm_to_gpu_s)

    return {
        "model": label,
        "cold_load_s": round(cold_load_s, 2),
        "warm_to_gpu_s": round(warm_to_gpu_s, 2),
        "gpu_unload_s": round(gpu_unload_s, 2),
        "gpu_vram_peak_gib": round(gpu_vram_peak_gib, 2),
        "host_ram_peak_gib": round(host_ram_peak_gib, 2),
        "recommendation": _decide(warm_to_gpu_s, cold_load_s, host_ram_peak_gib),
    }


# ---------------------------------------------------------------------------
# Decision rule (S10b.02)
# ---------------------------------------------------------------------------


def _decide(
    warm_to_gpu_s: Optional[float],
    cold_load_s: float,
    host_ram_peak_gib: float,
) -> str:
    """
    S10b.02 decision rule:
      WARM if (warm_to_gpu_s < 0.5 * cold_load_s) AND (host_ram_peak_gib < 64)
      COLD otherwise.
    """
    if warm_to_gpu_s is None:
        return "cold (warm-reload failed)"
    if warm_to_gpu_s < 0.5 * cold_load_s and host_ram_peak_gib < _WARM_RAM_CAP_GIB:
        return "warm"
    reasons = []
    if warm_to_gpu_s >= 0.5 * cold_load_s:
        reasons.append(f"warm_to_gpu_s={warm_to_gpu_s:.1f} >= 0.5*cold={0.5*cold_load_s:.1f}")
    if host_ram_peak_gib >= _WARM_RAM_CAP_GIB:
        reasons.append(f"host_ram={host_ram_peak_gib:.1f}GiB >= {_WARM_RAM_CAP_GIB}GiB")
    return "cold (" + "; ".join(reasons) + ")"


def _error_row(label: str) -> dict:
    return {col: ("ERROR" if col in ("model", "recommendation") else float("nan"))
            for col in _CSV_COLUMNS} | {"model": label, "recommendation": "cold (error)"}


def _partial_row(
    label: str,
    cold_load_s: float,
    warm_to_gpu_s: Optional[float],
    gpu_unload_s: float,
    gpu_vram_peak_gib: float,
    host_ram_peak_gib: float,
) -> dict:
    rec = _decide(warm_to_gpu_s, cold_load_s, host_ram_peak_gib)
    return {
        "model": label,
        "cold_load_s": round(cold_load_s, 2),
        "warm_to_gpu_s": "nan" if warm_to_gpu_s is None else round(warm_to_gpu_s, 2),
        "gpu_unload_s": round(gpu_unload_s, 2),
        "gpu_vram_peak_gib": round(gpu_vram_peak_gib, 2),
        "host_ram_peak_gib": round(host_ram_peak_gib, 2),
        "recommendation": rec,
    }


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def _write_csv(rows: list[dict], output_csv: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in _CSV_COLUMNS})
    log.info("Results written to %s", output_csv)


def _print_table(rows: list[dict]) -> None:
    header = "  ".join(f"{c:>22}" for c in _CSV_COLUMNS)
    log.info("\n%s", header)
    log.info("-" * len(header))
    for row in rows:
        line = "  ".join(f"{str(row.get(c, '')):>22}" for c in _CSV_COLUMNS)
        log.info(line)


# ---------------------------------------------------------------------------
# Main / CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="B8 CPU-warm lifecycle benchmark for Gemma/FLUX/Qwen on 3h100.",
    )
    p.add_argument(
        "--gemma-path",
        required=True,
        help="Absolute path to Gemma model weights on 3h100.",
    )
    p.add_argument(
        "--flux-path",
        required=True,
        help="Absolute path to FLUX model weights on 3h100.",
    )
    p.add_argument(
        "--qwen-path",
        required=True,
        help="Absolute path to Qwen-VL model weights on 3h100.",
    )
    p.add_argument(
        "--gpu-ids",
        default="0,1,2",
        help="Comma-separated GPU IDs to benchmark (default: 0,1,2).",
    )
    p.add_argument(
        "--output-csv",
        default="bench_warm_results.csv",
        help="Path for the output CSV file (default: bench_warm_results.csv).",
    )
    p.add_argument(
        "--tmp-dir",
        default="/tmp/bench_warm",
        help="Scratch directory for logs and temp files (default: /tmp/bench_warm).",
    )
    p.add_argument(
        "--skip-gemma", action="store_true", help="Skip Gemma benchmark."
    )
    p.add_argument(
        "--skip-flux", action="store_true", help="Skip FLUX benchmark."
    )
    p.add_argument(
        "--skip-qwen", action="store_true", help="Skip Qwen benchmark."
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
    ports = _BENCH_PORTS[: len(gpu_ids)]

    os.makedirs(args.tmp_dir, exist_ok=True)
    os.environ["LOG_DIR"] = args.tmp_dir

    log.info("B8 CPU-warm lifecycle benchmark starting")
    log.info("  GPU IDs    : %s", gpu_ids)
    log.info("  Output CSV : %s", args.output_csv)
    log.info("  Scratch dir: %s", args.tmp_dir)

    rows: list[dict] = []

    # Gemma ----------------------------------------------------------------
    if not args.skip_gemma:
        row = _bench_vllm_model(
            label="Gemma-31B",
            model_path=args.gemma_path,
            gpu_ids=gpu_ids,
            ports=ports,
            gpu_memory_utilization=0.95,
            max_model_len=8192,
            tmp_dir=args.tmp_dir,
        )
        rows.append(row)
        log.info("Gemma row: %s", json.dumps(row, default=str))
    else:
        log.info("Skipping Gemma (--skip-gemma)")

    # FLUX -----------------------------------------------------------------
    if not args.skip_flux:
        row = _bench_flux_model(
            label="FLUX-9B",
            model_path=args.flux_path,
            gpu_ids=gpu_ids,
            tmp_dir=args.tmp_dir,
        )
        rows.append(row)
        log.info("FLUX row: %s", json.dumps(row, default=str))
    else:
        log.info("Skipping FLUX (--skip-flux)")

    # Qwen -----------------------------------------------------------------
    if not args.skip_qwen:
        row = _bench_vllm_model(
            label="Qwen-VL-8B",
            model_path=args.qwen_path,
            gpu_ids=gpu_ids,
            ports=ports,
            gpu_memory_utilization=0.90,
            max_model_len=32768,
            extra_args=["--limit-mm-per-prompt", '{"image": 3}'],
            tmp_dir=args.tmp_dir,
        )
        rows.append(row)
        log.info("Qwen row: %s", json.dumps(row, default=str))
    else:
        log.info("Skipping Qwen (--skip-qwen)")

    if not rows:
        log.warning("No models were benchmarked. Nothing written.")
        return

    _print_table(rows)
    _write_csv(rows, args.output_csv)

    # Aggregate recommendation -------------------------------------------
    warm_count = sum(1 for r in rows if str(r.get("recommendation", "")).startswith("warm"))
    total = len(rows)
    log.info(
        "Overall recommendation: %d/%d models qualify for warm mode.",
        warm_count, total,
    )
    if warm_count == total:
        log.info("Suggestion: set lifecycle_mode=warm in config.yaml (requires manual review).")
    else:
        log.info("Suggestion: keep lifecycle_mode=cold (V0.2.1 default).")


if __name__ == "__main__":
    main()
