#!/usr/bin/env python3
"""Self-contained vLLM judge launcher for the standalone try-on evaluator.

Ported from the ERA iter_038 Stage-6 serve launcher (serve-122b-pilot_launch.py)
— stdlib only, NO era dependency. Serves Qwen3.5-122B-A10B (MoE FP16, tp4) on an
OpenAI-compatible endpoint the scorer closure calls over HTTP.

Public API:
  * launch(gpu_ids, port, model_path=None, log_file=None) -> dict
        {"proc", "base_url", "served_model_name", "port", "gpu_ids", ...}
  * probe_endpoint(endpoint, served=None) -> (base_url, served_model_name)
        attach to an already-running server instead of launching one.
  * shutdown_judge(proc, served, graceful_timeout_s=30) -> None
        safe process-group teardown (SIGTERM pgid -> poll -> SIGKILL -> sweep).

The GPU watchdog kill / free-card pick / watchdog restart is owned by run.py
(evaluator/gpu.py); this module only launches + tears down vLLM.
"""
from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

SERVED_MODEL_NAME = "Qwen3.5-122B-A10B"
TENSOR_PARALLEL = 4
GPU_MEM_UTIL = 0.90            # headroom for CUDA-graph capture (no --enforce-eager)
MAX_MODEL_LEN = 12288         # 3 images + think + JSON; smaller => more concurrency
STARTUP_TIMEOUT_S = 2400      # MoE FP16 load is several minutes
PORT_RANGE = (8000, 8099)

MODEL_PATH_CANDIDATES = [
    "/dev/shm/models/Qwen3.5-122B-A10B",
    "/mnt/image-edit/models/Qwen/Qwen3.5-122B-A10B",
]


def _log(msg: str) -> None:
    print(f"[serve_judge] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# environment prep
# --------------------------------------------------------------------------- #
def resolve_model_path(model_path: str | None = None) -> str:
    cands = [model_path] if model_path else []
    cands += MODEL_PATH_CANDIDATES
    for cand in cands:
        if cand and Path(cand).is_dir() and (Path(cand) / "config.json").is_file():
            return cand
    raise SystemExit(f"no Qwen3.5-122B weights found among {cands}")


def apply_rope_patch() -> None:
    """Re-apply the Qwen3.5-MoE rope list->set patch if a vLLM reinstall reverted
    it (no-op when already a set literal, which is the live install's state)."""
    try:
        import vllm  # noqa
        cfg = (Path(vllm.__file__).resolve().parent /
               "transformers_utils" / "configs" / "qwen3_5_moe.py")
    except Exception as e:
        _log(f"rope-patch: could not locate vllm config ({e}); skipping")
        return
    if not cfg.is_file():
        return
    try:
        text = cfg.read_text()
    except Exception:
        return
    m = re.search(r'ignore_keys_at_rope_validation"\]\s*=\s*([\[{])', text)
    if not m:
        return
    if m.group(1) == "{":
        _log("rope-patch: already applied (set literal) — no-op")
        return
    _log("rope-patch: list literal detected — attempting in-place fix via sudo")
    patched = re.sub(r'(ignore_keys_at_rope_validation"\]\s*=\s*)\[',
                     r"\1{", text, count=1)
    patched = re.sub(r'(ignore_keys_at_rope_validation"\]\s*=\s*\{[^\]]*?)\]',
                     r"\1}", patched, count=1)
    tmp = Path(f"/tmp/qwen3_5_moe.{os.getpid()}.py")
    try:
        tmp.write_text(patched)
        subprocess.run(["sudo", "cp", str(cfg), str(cfg) + ".era-bak"],
                       capture_output=True, timeout=15)
        r = subprocess.run(["sudo", "cp", str(tmp), str(cfg)],
                           capture_output=True, timeout=15)
        _log("rope-patch: applied" if r.returncode == 0
             else f"rope-patch: sudo cp failed ({r.stderr.decode()[:200]})")
    except Exception as e:
        _log(f"rope-patch: failed ({e}) — continuing (may already be patched)")
    finally:
        tmp.unlink(missing_ok=True)


def find_vllm() -> str:
    for cand in ("/usr/local/bin/vllm", "/usr/bin/vllm"):
        if Path(cand).is_file() and os.access(cand, os.X_OK):
            return cand
    import shutil
    found = shutil.which("vllm")
    if found:
        return found
    raise SystemExit("vllm binary not found on PATH")


def pick_port(preferred: int) -> int:
    cands = [preferred] + [p for p in range(*PORT_RANGE) if p != preferred]
    for port in cands:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise SystemExit(f"no free port near {preferred} / in {PORT_RANGE}")


# --------------------------------------------------------------------------- #
# probes
# --------------------------------------------------------------------------- #
def models_at(base_url: str, timeout: int = 5) -> list[str]:
    with urllib.request.urlopen(f"{base_url}/models", timeout=timeout) as r:
        return [m.get("id") for m in json.loads(r.read()).get("data", [])]


def chat_ping(base_url: str, model: str, timeout: int = 120) -> None:
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": "ping"}],
                       "max_tokens": 4}).encode()
    req = urllib.request.Request(f"{base_url}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        json.loads(r.read())


def _tail(path: Path, n: int = 40) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return "(log unreadable)"


def probe_endpoint(endpoint: str, served: str | None = None) -> tuple[str, str]:
    """Attach to a running OpenAI-compatible server. Returns (base_url, served)."""
    base = endpoint.rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    try:
        models = models_at(base)
    except Exception as e:
        raise SystemExit(f"could not reach judge at {base}: {e}")
    if served and served not in models:
        raise SystemExit(f"{served!r} not served at {base}; available: {models}")
    return base, (served or (models[0] if models else SERVED_MODEL_NAME))


# --------------------------------------------------------------------------- #
# launch / teardown
# --------------------------------------------------------------------------- #
def launch(gpu_ids, port, model_path=None, log_file=None,
           gpu_mem_util: float = GPU_MEM_UTIL) -> dict:
    """Launch vLLM detached on the given GPUs, block until /v1/models lists the
    served model + a chat ping succeeds, and return the endpoint block (with the
    live subprocess handle under 'proc'). The GPU watchdog MUST already be killed
    by the caller (vLLM's free-mem check fails while it holds ~35 GB/card)."""
    gpu_ids = [str(x) for x in gpu_ids]
    if len(gpu_ids) != TENSOR_PARALLEL:
        _log(f"WARNING: {len(gpu_ids)} GPUs given but tp={TENSOR_PARALLEL}")
    model_path = resolve_model_path(model_path)
    apply_rope_patch()
    port = pick_port(int(port))
    vllm = find_vllm()
    log_file = Path(log_file) if log_file else Path(f"/tmp/vllm-judge-{port}.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [vllm, "serve", model_path,
           "--served-model-name", SERVED_MODEL_NAME,
           "--tensor-parallel-size", str(TENSOR_PARALLEL),
           "--port", str(port),
           "--max-model-len", str(MAX_MODEL_LEN),
           "--gpu-memory-utilization", str(gpu_mem_util),
           "--limit-mm-per-prompt", '{"image": 3}',
           "--trust-remote-code"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    env["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"   # co-resident TP safety
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"     # fast weight load

    _log(f"model={model_path} gpus={gpu_ids} tp={TENSOR_PARALLEL} port={port}")
    _log(f"launching: {' '.join(cmd)}")
    fh = open(log_file, "w")
    proc = subprocess.Popen(cmd, start_new_session=True,
                            stdout=fh, stderr=subprocess.STDOUT, env=env)

    base = f"http://127.0.0.1:{port}/v1"
    deadline = time.time() + STARTUP_TIMEOUT_S
    t0 = time.time()
    last_err = None
    ready = False
    while time.time() < deadline:
        time.sleep(5)
        if proc.poll() is not None:
            raise SystemExit(
                f"vLLM exited with code {proc.returncode} during startup\n"
                f"--- tail {log_file} ---\n{_tail(log_file)}")
        try:
            if SERVED_MODEL_NAME in models_at(base):
                ready = True
                break
        except Exception as e:
            last_err = str(e)
            el = int(time.time() - t0)
            if el and el % 60 < 5:
                _log(f"  waiting for judge... {el}s")
    if not ready:
        shutdown_judge(proc, SERVED_MODEL_NAME)
        raise SystemExit(f"judge not up within {STARTUP_TIMEOUT_S}s "
                         f"(last: {last_err})\n--- tail {log_file} ---\n"
                         f"{_tail(log_file)}")
    chat_ping(base, SERVED_MODEL_NAME)
    startup = int(time.time() - t0)
    _log(f"READY: {SERVED_MODEL_NAME} at {base} ({startup}s startup)")
    return {"proc": proc, "base_url": base, "served_model_name": SERVED_MODEL_NAME,
            "host": "127.0.0.1", "port": port, "tensor_parallel": TENSOR_PARALLEL,
            "gpu_ids": gpu_ids, "pid": proc.pid, "model_path": model_path,
            "log_file": str(log_file)}


def shutdown_judge(proc, served: str | None = None,
                   graceful_timeout_s: int = 30) -> None:
    """Safe teardown: SIGTERM the process group -> poll -> SIGKILL -> orphan
    sweep. vLLM is launched with start_new_session=True so parent + tp workers
    share one pgid. Raw kill(pid) would leak tp workers and wedge GPUs."""
    if proc is None:
        return
    try:
        if proc.poll() is not None:
            return
    except Exception:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        pgid = None
    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGTERM)
        else:
            proc.terminate()
    except Exception:
        pass
    deadline = time.time() + graceful_timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        time.sleep(1)
    if proc.poll() is None:
        _log("graceful teardown timed out — SIGKILL")
        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            pass
    # orphan sweep — a crashed tp worker can survive the pgid kill.
    if served:
        try:
            subprocess.run(["pkill", "-TERM", "-f", served],
                           capture_output=True, timeout=15)
        except Exception:
            pass
    _log("judge torn down")
