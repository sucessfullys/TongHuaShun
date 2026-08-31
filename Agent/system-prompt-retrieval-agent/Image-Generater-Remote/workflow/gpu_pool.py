"""GPU pool manager — manages model lifecycle across multiple GPUs.

Handles vLLM server startup/shutdown and FLUX subprocess management
with clean VRAM reclamation between stages.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.request
from typing import Any, Optional

log = logging.getLogger("gpu_pool")

GPU_IDS = [0, 1, 2]


def kill_gpu_hogs(pattern: str = "NoGPUAlarmNew.py") -> None:
    """Kill GPU-hogging processes (e.g. NoGPUAlarm)."""
    try:
        subprocess.run(["pkill", "-f", pattern], capture_output=True, timeout=10)
        time.sleep(2)
        subprocess.run(["pkill", "-9", "-f", pattern], capture_output=True, timeout=10)
        log.info("Killed processes matching '%s'", pattern)
    except Exception as e:
        log.warning("Failed to kill '%s': %s", pattern, e)


def _check_server_ready(port: int, timeout_s: float = 3) -> bool:
    """Check if a vLLM server on localhost:port is responding."""
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.status == 200
    except Exception:
        return False


def start_vllm_servers(
    model_path: str,
    ports: list[int],
    gpu_ids: list[int],
    extra_args: Optional[list[str]] = None,
    gpu_memory_utilization: float = 0.95,
    max_model_len: int = 8192,
    startup_timeout_s: int = 900,
) -> list[subprocess.Popen]:
    """Start one vLLM OpenAI-compatible server per GPU.

    Args:
        model_path: Path to model weights on disk.
        ports: Port for each server (one per GPU).
        gpu_ids: GPU indices (e.g. [0, 1, 2]).
        extra_args: Additional vLLM CLI args.
        gpu_memory_utilization: Fraction of GPU memory to use.
        max_model_len: Maximum sequence length.
        startup_timeout_s: Max seconds to wait for all servers to become ready.

    Returns:
        List of Popen processes (one per server).
    """
    assert len(ports) == len(gpu_ids), "ports and gpu_ids must have same length"

    procs = []
    log_dir = os.environ.get("LOG_DIR", ".")

    for port, gpu_id in zip(ports, gpu_ids):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

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

        log_file = os.path.join(log_dir, f"vllm_{port}.log")
        log.info("Starting vLLM server: GPU %d, port %d, model %s",
                 gpu_id, port, os.path.basename(model_path))

        proc = subprocess.Popen(
            cmd, env=env,
            stdout=open(log_file, "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,  # Create new process group for clean kill
        )
        procs.append(proc)
        log.info("  PID=%d → log=%s", proc.pid, log_file)

    # Wait for ALL servers to become ready
    log.info("Waiting for %d vLLM servers to become ready...", len(procs))
    deadline = time.monotonic() + startup_timeout_s
    ready = [False] * len(ports)

    while time.monotonic() < deadline:
        all_ready = True
        for i, port in enumerate(ports):
            if ready[i]:
                continue
            if _check_server_ready(port):
                ready[i] = True
                log.info("  Port %d: READY", port)
            else:
                all_ready = False
                # Check if process died
                if procs[i].poll() is not None:
                    log.error("  Port %d: vLLM server DIED (exit code %d)",
                              port, procs[i].returncode)
                    # Kill the rest and fail
                    stop_servers(procs)
                    raise RuntimeError(
                        f"vLLM server on port {port} died with code {procs[i].returncode}. "
                        f"Check log: vllm_{port}.log"
                    )
        if all_ready:
            log.info("All %d vLLM servers ready.", len(procs))
            return procs
        time.sleep(5)

    # Timeout
    not_ready = [ports[i] for i in range(len(ports)) if not ready[i]]
    stop_servers(procs)
    raise RuntimeError(f"vLLM servers on ports {not_ready} did not become ready "
                       f"within {startup_timeout_s}s")


def stop_servers(procs: list[subprocess.Popen]) -> None:
    """Stop vLLM server processes and their children (EngineCore).

    vLLM 0.19+ spawns child VLLM::EngineCore processes that hold GPU memory.
    We must kill the entire process group, not just the parent, otherwise
    EngineCore zombies leak VRAM until the machine is rebooted.
    """
    for proc in procs:
        if proc.poll() is None:
            log.info("Stopping vLLM server PID=%d (pgid=%d)...", proc.pid, os.getpgid(proc.pid))
            # Send SIGTERM to entire process group
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass

    # Wait up to 30s for graceful shutdown
    deadline = time.monotonic() + 30
    for proc in procs:
        remaining = max(0, deadline - time.monotonic())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            log.warning("Force-killing vLLM process group PID=%d", proc.pid)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                proc.kill()
            proc.wait()

    log.info("All vLLM servers and child processes stopped.")


def verify_gpus_free(gpu_ids: list[int], min_free_gib: float = 70.0) -> bool:
    """Verify that all specified GPUs have sufficient free VRAM."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            log.warning("nvidia-smi failed; assuming GPUs are free")
            return True

        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            gpu_id = int(parts[0])
            free_mib = float(parts[1])
            free_gib = free_mib / 1024.0
            if gpu_id in gpu_ids and free_gib < min_free_gib:
                log.warning("GPU %d has only %.1f GiB free (need %.1f)",
                            gpu_id, free_gib, min_free_gib)
                return False
        return True
    except Exception as e:
        log.warning("GPU check failed: %s", e)
        return True


def start_flux_workers(
    model_path: str,
    gpu_ids: list[int],
    shard_input_files: list[str],
    shard_output_files: list[str],
    config: dict[str, Any],
) -> list[subprocess.Popen]:
    """Start FLUX processing subprocesses, one per GPU.

    Each subprocess loads the FLUX pipeline, processes its shard,
    and writes results to its output file. Subprocess isolation ensures
    clean VRAM reclamation on exit.

    Args:
        model_path: Path to FLUX model weights.
        gpu_ids: GPU indices.
        shard_input_files: JSON files with per-shard work items.
        shard_output_files: JSON files where each subprocess writes results.
        config: FLUX-specific config (height, width, steps, etc.).

    Returns:
        List of Popen processes.
    """
    assert len(gpu_ids) == len(shard_input_files) == len(shard_output_files)

    # The subprocess entry point is workflow/stage_flux.py
    script_path = os.path.join(os.path.dirname(__file__), "stage_flux.py")
    procs = []

    for gpu_id, input_file, output_file in zip(gpu_ids, shard_input_files, shard_output_files):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        cmd = [
            sys.executable, script_path,
            "--input", input_file,
            "--output", output_file,
            "--model-path", model_path,
            "--config", json.dumps(config),
        ]

        log.info("Starting FLUX worker: GPU %d, input=%s", gpu_id, input_file)
        proc = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True,  # Clean process group for kill
        )
        procs.append(proc)
        log.info("  PID=%d", proc.pid)

    return procs


def wait_for_flux_workers(
    procs: list[subprocess.Popen],
    timeout_s: int = 7200,
) -> list[int]:
    """Wait for all FLUX subprocesses to finish. Returns exit codes."""
    exit_codes = []
    deadline = time.monotonic() + timeout_s

    for i, proc in enumerate(procs):
        remaining = max(1, deadline - time.monotonic())
        try:
            proc.wait(timeout=remaining)
            exit_codes.append(proc.returncode)
            if proc.returncode != 0:
                # Read stderr for error context
                stdout = proc.stdout.read().decode("utf-8", errors="replace") if proc.stdout else ""
                log.error("FLUX worker %d exited with code %d:\n%s",
                          i, proc.returncode, stdout[-2000:])
            else:
                log.info("FLUX worker %d completed successfully.", i)
        except subprocess.TimeoutExpired:
            log.error("FLUX worker %d timed out after %ds — killing", i, timeout_s)
            proc.kill()
            proc.wait()
            exit_codes.append(-1)

    return exit_codes
