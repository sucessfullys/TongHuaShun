"""Backend worker child process — loads one model adapter and handles RPC commands.

Run as::

    python -m server.backend_worker

The process reads JSON-line commands from stdin and writes JSON-line responses to
stdout.  stderr is used for internal logging only and is NOT part of the protocol.

Supported commands (supervisor -> worker)
-----------------------------------------
{"type": "load",   "model_name": "gemma4", "gpu_id": 0, "config": {...}}
{"type": "infer",  "request_id": "abc123", "payload": {...}}
{"type": "unload"}
{"type": "exit"}

Responses (worker -> supervisor)
---------------------------------
{"type": "ready",   "model_name": "gemma4", "load_latency_ms": 1234}
{"type": "result",  "request_id": "abc123", "ok": true,  "payload": {...}}
{"type": "error",   "request_id": "abc123", "error_code": "oom", "message": "..."}
{"type": "status",  "message": "..."}
{"type": "exiting"}

Special flags
-------------
--self-check   Import this module, print "backend_worker self-check ok", exit 0.
               Does NOT import any model adapters (they may require a GPU).
"""

from __future__ import annotations

import importlib
import logging
import sys
import time
from typing import Any, Optional

# Internal RPC helpers — imported from our own write scope.
from server.worker_rpc import recv_message, send_message

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [backend_worker] %(levelname)s %(message)s",
)
log = logging.getLogger("backend_worker")

# ── Adapter registry ───────────────────────────────────────────────────────────

ADAPTER_REGISTRY: dict[str, tuple[str, str]] = {
    "gemma4":         ("models.gemma4",         "Gemma4Adapter"),
    "flux2_klein":    ("models.flux2_klein",     "Flux2KleinAdapter"),
    "qwen3_vl":       ("models.qwen3_vl",        "Qwen3VLAdapter"),
    "qwen3_vl_swift": ("models.qwen3_vl_swift",  "Qwen3VLSwiftAdapter"),
}

# Seconds of idle (no command received) before auto-exit after a model is loaded.
IDLE_TIMEOUT_S: float = 300.0


# ── Helpers ────────────────────────────────────────────────────────────────────

def _reply(msg: dict) -> None:
    """Write a single JSON-line response to stdout."""
    send_message(sys.stdout, msg)


def _reply_status(message: str) -> None:
    _reply({"type": "status", "message": message})


def _reply_error(request_id: Optional[str], error_code: str, message: str) -> None:
    resp: dict[str, Any] = {"type": "error", "error_code": error_code, "message": message}
    if request_id is not None:
        resp["request_id"] = request_id
    _reply(resp)


# ── Command handlers ───────────────────────────────────────────────────────────

def _handle_load(cmd: dict) -> Optional[Any]:
    """Dynamically import and instantiate the adapter.  Returns adapter or None."""
    model_name: str = cmd.get("model_name", "")
    gpu_id: int = int(cmd.get("gpu_id", 0))
    config_override: dict = cmd.get("config", {})

    if model_name not in ADAPTER_REGISTRY:
        _reply_error(
            None,
            "load_failed",
            f"Unknown model_name '{model_name}'. "
            f"Known: {list(ADAPTER_REGISTRY.keys())}",
        )
        return None

    module_path, class_name = ADAPTER_REGISTRY[model_name]
    log.info("Loading adapter %s from %s …", class_name, module_path)
    t0 = time.monotonic()

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        _reply_error(
            None,
            "adapter_load_error",
            f"Cannot import '{module_path}': {exc}",
        )
        return None
    except Exception as exc:  # noqa: BLE001
        _reply_error(
            None,
            "adapter_load_error",
            f"Unexpected error importing '{module_path}': {exc}",
        )
        return None

    try:
        adapter_cls = getattr(module, class_name)
    except AttributeError:
        _reply_error(
            None,
            "adapter_load_error",
            f"Module '{module_path}' has no attribute '{class_name}'",
        )
        return None

    try:
        adapter = adapter_cls(gpu_id=gpu_id, config=config_override)
    except Exception as exc:  # noqa: BLE001
        _reply_error(
            None,
            "adapter_load_error",
            f"Failed to instantiate {class_name}: {exc}",
        )
        return None

    # Actually load weights — this is the operation the supervisor is waiting for.
    try:
        load_result = adapter.load() or {}
    except MemoryError as exc:
        _reply_error(None, "oom", str(exc))
        return None
    except Exception as exc:  # noqa: BLE001
        _reply_error(None, "load_failed", f"{class_name}.load() raised: {exc}")
        return None

    load_latency_ms = (time.monotonic() - t0) * 1000.0
    log.info("Adapter %s loaded in %.1f ms", class_name, load_latency_ms)
    _reply({
        "type": "ready",
        "model_name": model_name,
        "load_latency_ms": round(load_latency_ms, 1),
        "load_result": {
            "adapter_attached": load_result.get("adapter_attached"),
            "adapter_backend_used": load_result.get("adapter_backend_used"),
            "deployment_mode": load_result.get("deployment_mode"),
            "tp_size": load_result.get("tp_size", 1),
            "resident_mem_gib": load_result.get("resident_mem_gib", 0.0),
        },
    })
    return adapter


def _handle_infer(adapter: Optional[Any], cmd: dict) -> None:
    """Delegate to adapter.infer_single() or infer_list(); send result or error."""
    request_id: Optional[str] = cmd.get("request_id")
    payload: dict = cmd.get("payload", {})

    if adapter is None:
        _reply_error(request_id, "load_failed", "No model loaded — send 'load' first")
        return

    try:
        # Batch mode: payload contains {"_batch": True, "items": [...]}
        if payload.get("_batch"):
            items = payload.get("items", [])
            results = adapter.infer_list(items)
            _reply({
                "type": "result",
                "request_id": request_id,
                "ok": True,
                "payload": {"items": results},
            })
            return

        result = adapter.infer_single(payload)
    except MemoryError as exc:
        log.exception("OOM during infer")
        _reply_error(request_id, "oom", str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        log.exception("Exception during infer for request_id=%s", request_id)
        _reply_error(request_id, "internal_error", str(exc))
        return

    _reply({
        "type": "result",
        "request_id": request_id,
        "ok": True,
        "payload": result if isinstance(result, dict) else {"value": result},
    })


def _handle_unload(adapter: Optional[Any]) -> None:
    """Call adapter.unload() if available, then send a status message."""
    if adapter is None:
        _reply_status("No model loaded — nothing to unload")
        return

    try:
        adapter.unload()
        log.info("Adapter unloaded")
        _reply_status("unloaded")
    except Exception as exc:  # noqa: BLE001
        log.exception("Exception during unload")
        _reply_error(None, "unload_failed", str(exc))


# ── Main event loop ────────────────────────────────────────────────────────────

def _main_loop() -> int:
    """Read commands from stdin and dispatch until 'exit' or EOF.

    Returns the process exit code (0 = clean).
    """
    adapter: Optional[Any] = None
    model_loaded: bool = False

    log.info("backend_worker started (pid=%d)", __import__("os").getpid())

    while True:
        # Use the idle timeout only after a model has been loaded.
        timeout = IDLE_TIMEOUT_S if model_loaded else None
        cmd = recv_message(sys.stdin, timeout_s=timeout)

        if cmd is None:
            if model_loaded:
                # Idle timeout — auto-exit.
                log.warning(
                    "No command received for %.0f s — auto-exiting", IDLE_TIMEOUT_S
                )
                _reply_status(f"idle timeout after {IDLE_TIMEOUT_S:.0f}s — exiting")
                _reply({"type": "exiting"})
            else:
                log.info("EOF on stdin — exiting")
            return 0

        msg_type: str = cmd.get("type", "")

        if msg_type == "load":
            adapter = _handle_load(cmd)
            if adapter is not None:
                model_loaded = True

        elif msg_type == "infer":
            _handle_infer(adapter, cmd)

        elif msg_type == "unload":
            _handle_unload(adapter)
            adapter = None
            model_loaded = False

        elif msg_type == "exit":
            log.info("Received 'exit' command — shutting down cleanly")
            _reply({"type": "exiting"})
            return 0

        else:
            log.warning("Unknown command type: %r", msg_type)
            _reply_error(
                cmd.get("request_id"),
                "internal_error",
                f"Unknown command type '{msg_type}'",
            )


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point: parse flags, then run the main loop or self-check."""
    if "--self-check" in sys.argv:
        # Must work without any adapter imports (no GPU required).
        print("backend_worker self-check ok")
        sys.exit(0)

    try:
        exit_code = _main_loop()
    except KeyboardInterrupt:
        log.info("Interrupted — exiting")
        exit_code = 0
    except Exception as exc:  # noqa: BLE001
        log.exception("Unhandled exception in main loop: %s", exc)
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
