"""FastAPI supervisor — one per GPU, manages a single backend worker child."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from server.config import load_config, deep_get, model_config, port_to_gpu, all_host_ports
from server.schemas import (
    HealthResponse, GpuStatusResponse, GpuSnapshot, StatusResponse,
    LoadRequest, LoadResponse, PrefetchRequest,
    UnloadRequest, UnloadResponse,
    GemmaSingleRequest, GemmaSingleResponse,
    FluxSingleRequest, FluxSingleResponse,
    QwenSingleRequest, QwenSingleResponse,
    InferListRequest, InferTemplateRequest,
    BatchResponse, BatchItemResult,
    CallbackPayload, ErrorResponse, ErrorDetail,
)
from server.gpu_utils import get_gpu_snapshot, kill_nogpualarm, verify_post_unload_free
from server.manifest import verify_manifest
from server.paths import validate_path_exists, is_path_allowed
from server.worker_rpc import WorkerConnection
from server.callback import send_callback_if_required

logging.basicConfig(level=logging.INFO, format="%(asctime)s [supervisor] %(levelname)s %(message)s")
log = logging.getLogger("supervisor")

# ── Global state ───────────────────────────────────────────────────

_cfg: dict[str, Any] = {}
_gpu_id: int = 0
_port: int = 17610
_lock = asyncio.Lock()       # guards /load + /unload
_infer_lock = asyncio.Lock() # guards all /infer/* RPC (B6 fix)
_worker: Optional[WorkerConnection] = None
_loaded_model: Optional[str] = None
_last_request_model: Optional[str] = None
_last_request_time: Optional[str] = None
_load_result: dict[str, Any] = {}


def _make_error(code: str, message: str, retryable: bool = False, **details) -> JSONResponse:
    return JSONResponse(
        status_code=400 if code in ("invalid_input", "path_not_found", "path_not_allowed", "manifest_mismatch") else 500,
        content=ErrorResponse(
            ok=False,
            error=ErrorDetail(error_code=code, message=message, retryable=retryable, details=details)
        ).model_dump(),
    )


# ── Lifespan ───────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(application: FastAPI):
    global _cfg, _gpu_id, _port
    config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    _cfg = load_config(config_path)
    _port = int(os.environ.get("SUPERVISOR_PORT", "17610"))
    _gpu_id = port_to_gpu(_cfg, _port)
    log.info("Supervisor started: port=%d gpu_id=%d", _port, _gpu_id)
    yield
    if _worker and _worker.is_alive():
        _worker.terminate(grace_s=10)


app = FastAPI(title="V0.1 Supervisor", lifespan=lifespan)


# ── Health / Status ────────────────────────────────────────────────

@app.get("/health")
async def health():
    snap = get_gpu_snapshot(gpu_id=_gpu_id)
    gpu_snap = snap[0] if snap else {}
    return HealthResponse(
        status="ok",
        loaded_model=_loaded_model,
        gpu_id=_gpu_id,
        gpu_mem_used_gib=gpu_snap.get("mem_used_gib"),
        gpu_mem_total_gib=gpu_snap.get("mem_total_gib"),
        backend_pid=_worker.pid if _worker and _worker.is_alive() else None,
    )


@app.get("/gpu/status")
async def gpu_status():
    snaps = get_gpu_snapshot()
    return GpuStatusResponse(gpus=[GpuSnapshot(**s) for s in snaps])


@app.get("/status")
async def status():
    return StatusResponse(
        last_request_model=_last_request_model,
        last_request_time=_last_request_time,
        lock_held=_lock.locked(),
        lock_holder=_loaded_model if _lock.locked() else None,
    )


# ── Worker spawn (F4: TP-aware visible_devices) ───────────────────

def _spawn_worker(visible_devices: Optional[str] = None) -> WorkerConnection:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = visible_devices if visible_devices is not None else str(_gpu_id)
    proc = subprocess.Popen(
        [sys.executable, "-m", "server.backend_worker"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, bufsize=1,
        env=env,
    )
    return WorkerConnection(proc)


# ── Load (F2: read LoadResult from worker; F3: Qwen fallback; F4: TP retry) ──

@app.post("/load")
async def load_model(req: LoadRequest):
    global _worker, _loaded_model, _load_result
    async with _lock:
        if _worker and _worker.is_alive():
            return _make_error("load_failed", "A model is already loaded — /unload first")

        try:
            mcfg = model_config(_cfg, req.model_name)
        except KeyError:
            return _make_error("load_failed", f"Unknown model: {req.model_name}")

        dm = mcfg.get("deployment_mode", "single_gpu_replicated")

        # Kill GPU preservation if configured
        from server.config import kill_gpu_preservation_on_load, gpu_preservation_pattern
        if kill_gpu_preservation_on_load(_cfg):
            kill_nogpualarm(gpu_preservation_pattern(_cfg))

        merged_config = {**mcfg, **req.config_overrides}
        timeout = deep_get(_cfg, "server.request_timeout_s", 1800)

        # --- Determine visible_devices (F4: TP-aware) ---
        visible = str(_gpu_id)
        if dm == "tensor_parallel_single_service":
            if _gpu_id != 0:
                return _make_error("load_failed",
                    f"Model {req.model_name} uses tensor_parallel_single_service — only sup_gpu0 hosts it")
            tp = int(mcfg.get("tensor_parallel_size", 2))
            visible = ",".join(str(i) for i in range(tp))

        t0 = time.monotonic()
        _worker = _spawn_worker(visible_devices=visible)
        _worker.send({
            "type": "load",
            "model_name": req.model_name,
            "gpu_id": _gpu_id,
            "config": merged_config,
        })

        resp = _worker.recv(timeout_s=timeout)

        # --- F4: Bounded TP retry ladder for Gemma OOM ---
        if req.model_name == "gemma4" and (resp is None or resp.get("type") != "ready"):
            error_code = resp.get("error_code", "load_failed") if resp else "worker_crashed"
            if error_code in ("oom", "load_failed", "worker_crashed"):
                for tp_try in (2, 3):
                    log.warning("Gemma load failed — retrying with tp_size=%d", tp_try)
                    if _worker and _worker.is_alive():
                        _worker.terminate(grace_s=5)
                    merged_config["tensor_parallel_size"] = tp_try
                    merged_config["deployment_mode"] = "tensor_parallel_single_service"
                    visible = ",".join(str(i) for i in range(tp_try))
                    _worker = _spawn_worker(visible_devices=visible)
                    _worker.send({"type": "load", "model_name": "gemma4",
                                  "gpu_id": 0, "config": merged_config})
                    resp = _worker.recv(timeout_s=timeout)
                    if resp and resp.get("type") == "ready":
                        dm = "tensor_parallel_single_service"
                        break
                    if _worker and _worker.is_alive():
                        _worker.terminate(grace_s=5)
                    _worker = None

        if resp is None or resp.get("type") != "ready":
            error_msg = resp.get("message", "Worker did not report ready") if resp else "Worker process died"
            error_code = resp.get("error_code", "load_failed") if resp else "worker_crashed"
            if _worker and _worker.is_alive():
                _worker.terminate(grace_s=5)
            _worker = None
            return _make_error(error_code, error_msg)

        # --- F2: Read LoadResult from worker reply (not from config) ---
        lr = resp.get("load_result") or {}
        load_ms = (time.monotonic() - t0) * 1000
        _loaded_model = req.model_name
        _load_result = {
            "adapter_attached": lr.get("adapter_attached"),
            "adapter_backend_used": lr.get("adapter_backend_used") or mcfg.get("adapter_backend"),
            "deployment_mode": lr.get("deployment_mode") or dm,
            "tp_size": int(lr.get("tp_size", merged_config.get("tensor_parallel_size", 1))),
        }

        # --- F3: Qwen swift_infer fallback ---
        if (req.model_name == "qwen3_vl"
                and _load_result.get("adapter_backend_used") == "swift_infer"
                and lr.get("adapter_attached") is False
                and not merged_config.get("adapter_required", False)):
            log.warning("Qwen LoRA attach declined — falling back to qwen3_vl_swift")
            _worker.terminate(grace_s=10)
            _worker = _spawn_worker(visible_devices=visible)
            _worker.send({"type": "load", "model_name": "qwen3_vl_swift",
                          "gpu_id": _gpu_id, "config": merged_config})
            resp2 = _worker.recv(timeout_s=timeout)
            if resp2 is None or resp2.get("type") != "ready":
                if _worker and _worker.is_alive():
                    _worker.terminate(grace_s=5)
                _worker = None
                _loaded_model = None
                return _make_error("adapter_load_error",
                                   "Qwen swift_infer fallback failed to load")
            lr2 = resp2.get("load_result") or {}
            _load_result.update({
                "adapter_attached": lr2.get("adapter_attached", False),
                "adapter_backend_used": "swift_infer",
                "deployment_mode": lr2.get("deployment_mode", "external_backend"),
                "tp_size": 1,
            })
            load_ms = (time.monotonic() - t0) * 1000

        return LoadResponse(
            status="loaded",
            load_latency_ms=round(load_ms, 1),
            adapter_attached=_load_result.get("adapter_attached"),
            adapter_backend_used=_load_result.get("adapter_backend_used"),
            deployment_mode=_load_result.get("deployment_mode", dm),
            tp_size=_load_result.get("tp_size", 1),
            backend_pid=_worker.pid,
        )


# ── Prefetch ───────────────────────────────────────────────────────

@app.post("/prefetch")
async def prefetch(req: PrefetchRequest):
    if not _worker or not _worker.is_alive() or not _loaded_model:
        return _make_error("load_failed", "No model loaded")
    return {"status": "prefetch_ok", "model_name": _loaded_model}


# ── Unload (F6: acquire _infer_lock; L8: use get_running_loop) ────

@app.post("/unload")
async def unload_model(req: UnloadRequest = UnloadRequest()):
    global _worker, _loaded_model, _load_result
    # Acquire _infer_lock first so no inference is in flight
    async with _infer_lock:
        async with _lock:
            if not _worker:
                return UnloadResponse(status="no_model_loaded", mem_before_gib=0, mem_after_gib=0)

            snap_before = get_gpu_snapshot(gpu_id=_gpu_id)
            mem_before = snap_before[0]["mem_used_gib"] if snap_before else 0

            grace = deep_get(_cfg, "gpu.unload_grace_s", 30)
            _worker.terminate(grace_s=grace)
            _worker = None
            _loaded_model = None
            _load_result = {}

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            snap_after = get_gpu_snapshot(gpu_id=_gpu_id)
            mem_after = snap_after[0]["mem_used_gib"] if snap_after else 0

            min_free = deep_get(_cfg, "gpu.post_unload_min_free_gib", 70)
            vram_check = verify_post_unload_free(_gpu_id, min_free)
            restarted = False
            if not vram_check["ok"]:
                log.error("Post-unload VRAM too low: free=%.1f GiB < min=%.1f GiB — self-restarting",
                          vram_check["free_gib"], min_free)
                restarted = True
                # L8 fix: use get_running_loop() instead of deprecated get_event_loop()
                asyncio.get_running_loop().call_later(2.0, lambda: os._exit(1))

            return UnloadResponse(
                status="unloaded",
                mem_before_gib=round(mem_before, 2),
                mem_after_gib=round(mem_after, 2),
                restarted=restarted,
            )


# ── Inference helpers (F6: all under _infer_lock) ─────────────────

def _check_loaded(model_name: str = None):
    if not _worker or not _worker.is_alive() or not _loaded_model:
        return _make_error("load_failed", "No model loaded — POST /load first")
    if model_name and _loaded_model != model_name:
        return _make_error("invalid_input", f"Loaded model is {_loaded_model}, not {model_name}")
    return None


def _validate_image_paths(body: dict) -> Optional[JSONResponse]:
    """Validate image paths in the request body against allowed_read_roots."""
    allowed = deep_get(_cfg, "paths.allowed_read_roots", [])
    for key in ("model_image_path", "cloth_image_path", "result_image_path"):
        p = body.get(key)
        if p:
            try:
                validate_path_exists(p, allowed)
            except ValueError as e:
                code = "path_not_found" if "path_not_found" in str(e) else "path_not_allowed"
                return _make_error(code, str(e))
    return None


def _infer_one_sync(payload: dict) -> dict:
    """Synchronous RPC to worker for a single inference (runs in thread)."""
    rid = str(uuid.uuid4())
    _worker.send({"type": "infer", "request_id": rid, "payload": payload})
    timeout = deep_get(_cfg, "server.request_timeout_s", 1800)
    resp = _worker.recv(timeout_s=timeout)
    if resp is None:
        raise RuntimeError("worker_unresponsive: Worker did not respond")
    if resp.get("type") == "error":
        raise RuntimeError(f"{resp.get('error_code', 'internal_error')}: {resp.get('message', '')}")
    return resp.get("payload", {})


def _infer_batch_sync(items: list[dict]) -> list[dict]:
    """Synchronous RPC to worker for batch inference via infer_list (runs in thread)."""
    rid = str(uuid.uuid4())
    _worker.send({"type": "infer", "request_id": rid, "payload": {"_batch": True, "items": items}})
    timeout = deep_get(_cfg, "server.request_timeout_s", 1800)
    resp = _worker.recv(timeout_s=timeout)
    if resp is None:
        raise RuntimeError("worker_unresponsive: Worker did not respond")
    if resp.get("type") == "error":
        raise RuntimeError(f"{resp.get('error_code', 'internal_error')}: {resp.get('message', '')}")
    return resp.get("payload", {}).get("items", [])


async def _send_callback_checked(body: dict, ok: bool, ok_count: int,
                                  error_count: int, total: int, latency_ms: float):
    """Send callback if URL present; honor callback_required (L3 fix: consistent behavior)."""
    cb_url = body.get("callback_url", "")
    if not cb_url:
        return
    cb_required = deep_get(_cfg, "callback.callback_required", True)
    payload = CallbackPayload(
        run_id=body.get("run_id", ""),
        stage_id=body.get("stage_id", ""),
        cell_id=body.get("cell_id", ""),
        shard_id=body.get("shard_id", 0),
        sample_id=body.get("sample_id", ""),
        host_port=_port,
        model_name=_loaded_model or "",
        status="completed", ok=ok,
        ok_count=ok_count, error_count=error_count, total=total,
        latency_ms=round(latency_ms, 1),
    ).model_dump()
    try:
        await send_callback_if_required(cb_url, payload, required=cb_required)
    except RuntimeError:
        if cb_required:
            raise  # propagate to caller


# ── /infer/single (F6: under _infer_lock) ─────────────────────────

@app.post("/infer/single")
async def infer_single(request: Request):
    global _last_request_model, _last_request_time
    body = await request.json()

    async with _infer_lock:
        err = _check_loaded()
        if err:
            return err

        _last_request_model = _loaded_model
        _last_request_time = time.strftime("%Y-%m-%dT%H:%M:%S")

        # Validate manifest hash if provided
        manifest_hash = body.get("manifest_hash", "")
        if manifest_hash and body.get("_manifest_items"):
            if not verify_manifest(manifest_hash, body["_manifest_items"]):
                return _make_error("manifest_mismatch", "Manifest hash does not match items")

        # Validate image paths
        path_err = _validate_image_paths(body)
        if path_err:
            return path_err

        try:
            result = await asyncio.to_thread(_infer_one_sync, body)
        except RuntimeError as e:
            code = str(e).split(":")[0] if ":" in str(e) else "internal_error"
            return _make_error(code, str(e))

    # Callback (outside infer_lock)
    try:
        await _send_callback_checked(body, ok=True, ok_count=1, error_count=0, total=1, latency_ms=0)
    except RuntimeError:
        return _make_error("callback_failed", f"Callback to {body.get('callback_url', '')} failed")

    return result


# ── /infer/list (F5: manifest/shard/path validation + batch RPC; F6: locked) ──

@app.post("/infer/list")
async def infer_list(request: Request):
    global _last_request_model, _last_request_time
    body = await request.json()

    async with _infer_lock:
        err = _check_loaded()
        if err:
            return err

        _last_request_model = _loaded_model
        _last_request_time = time.strftime("%Y-%m-%dT%H:%M:%S")

        items = body.get("items", [])
        manifest_hash = body.get("manifest_hash", "")
        shard_indices = set(body.get("shard_indices", []))

        # F5: Validate manifest hash if provided with items
        if manifest_hash and body.get("_manifest_items"):
            if not verify_manifest(manifest_hash, body["_manifest_items"]):
                return _make_error("manifest_mismatch", "Manifest hash does not match items")

        # F5: Validate shard indices membership
        if shard_indices:
            for item in items:
                idx = item.get("index")
                if idx is not None and idx not in shard_indices:
                    return _make_error("invalid_input",
                        f"Item index {idx} not in shard_indices {sorted(shard_indices)}")

        # F5: Validate image paths for all items
        allowed = deep_get(_cfg, "paths.allowed_read_roots", [])
        for item in items:
            for key in ("model_image_path", "cloth_image_path", "result_image_path"):
                p = item.get(key)
                if p:
                    try:
                        validate_path_exists(p, allowed)
                    except ValueError as e:
                        code = "path_not_found" if "path_not_found" in str(e) else "path_not_allowed"
                        return _make_error(code, str(e))

        # F5/L1: Batch RPC to worker (single round-trip via infer_list)
        t0 = time.monotonic()
        ok_count = 0
        error_count = 0
        results = []

        try:
            batch_results = await asyncio.to_thread(_infer_batch_sync, items)
            for i, item in enumerate(items):
                if i < len(batch_results):
                    results.append(BatchItemResult(
                        index=item.get("index", 0),
                        sample_id=item.get("sample_id", ""),
                        ok=True, result=batch_results[i] if isinstance(batch_results[i], dict) else {},
                    ))
                    ok_count += 1
                else:
                    results.append(BatchItemResult(
                        index=item.get("index", 0),
                        sample_id=item.get("sample_id", ""),
                        ok=False, error="Missing result from worker",
                    ))
                    error_count += 1
        except RuntimeError as e:
            # Entire batch failed
            for item in items:
                results.append(BatchItemResult(
                    index=item.get("index", 0),
                    sample_id=item.get("sample_id", ""),
                    ok=False, error=str(e),
                ))
                error_count += 1

        total_ms = (time.monotonic() - t0) * 1000

    resp = BatchResponse(
        run_id=body.get("run_id", ""),
        stage_id=body.get("stage_id", ""),
        cell_id=body.get("cell_id", ""),
        host_port=_port,
        model_name=_loaded_model or "",
        ok=error_count == 0,
        ok_count=ok_count,
        error_count=error_count,
        total=len(items),
        items=[r.model_dump() for r in results],
        latency_ms=round(total_ms, 1),
    )

    # L3 fix: callback with same required behavior as /infer/single
    try:
        await _send_callback_checked(body, ok=error_count == 0, ok_count=ok_count,
                                      error_count=error_count, total=len(items), latency_ms=total_ms)
    except RuntimeError:
        return _make_error("callback_failed", f"Callback to {body.get('callback_url', '')} failed")

    return resp.model_dump()


# ── /infer/template (F5: path validation; F6: locked) ─────────────

@app.post("/infer/template")
async def infer_template(request: Request):
    global _last_request_model, _last_request_time
    body = await request.json()

    async with _infer_lock:
        err = _check_loaded()
        if err:
            return err
        _last_request_model = _loaded_model
        _last_request_time = time.strftime("%Y-%m-%dT%H:%M:%S")

        # F5: manifest check
        manifest_hash = body.get("manifest_hash", "")
        if manifest_hash and body.get("_manifest_items"):
            if not verify_manifest(manifest_hash, body["_manifest_items"]):
                return _make_error("manifest_mismatch", "Manifest hash does not match items")

        try:
            result = await asyncio.to_thread(_infer_one_sync, {
                "_template_mode": True,
                **body,
            })
        except RuntimeError as e:
            return _make_error("internal_error", str(e))

    return result


# ── Error handler ──────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_error_handler(request: Request, exc: Exception):
    log.exception("Unhandled exception: %s", exc)
    return _make_error("internal_error", str(exc))


# ── CLI entry point ───────────────────────────────────────────────

def main():
    import uvicorn
    port = int(os.environ.get("SUPERVISOR_PORT", "17610"))
    host = os.environ.get("SUPERVISOR_HOST", "127.0.0.1")
    config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    log.info("Starting supervisor on %s:%d with config %s", host, port, config_path)
    uvicorn.run(app, host=host, port=port, workers=1, log_level="info")


if __name__ == "__main__":
    main()
