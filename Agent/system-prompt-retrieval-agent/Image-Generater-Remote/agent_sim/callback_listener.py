"""Lightweight FastAPI server that receives status callbacks from worker hosts.

Listens on 127.0.0.1:17700 (POST /status).  Stores payloads in an in-memory
dict and appends each received payload to a JSONL file on disk so callers can
reconstruct history across restarts.

Typical usage
-------------
Start standalone::

    python -m agent_sim.callback_listener

Query from test code::

    from agent_sim.callback_listener import get_callbacks, clear
    matches = get_callbacks(run_id="r1", stage_id="gemma", cell_id="c0")
    clear()
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Pydantic model for the inbound payload
# ---------------------------------------------------------------------------

class CallbackPayload(BaseModel):
    """Mirrors server.schemas.CallbackPayload + CorrelationMixin.

    All fields are optional / have defaults so the listener accepts both
    minimal and fully-populated payloads without crashing.
    """

    run_id: str = ""
    stage_id: str = ""
    cell_id: str = ""
    shard_id: int = 0
    sample_id: str = ""
    host_port: int = 0
    model_name: str = ""

    status: str = "completed"
    ok: bool = True
    ok_count: int = 0
    error_count: int = 0
    total: int = 0
    latency_ms: float = 0
    error_message: str | None = None

    model_config = {"extra": "allow"}   # accept any extra fields sent by workers


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------

# Key: (run_id, stage_id, cell_id, host_port, shard_id)
# Value: list of raw dicts (one per received callback for that key)
_store: dict[tuple[str, str, str, int, int], list[dict[str, Any]]] = {}

# Path to the JSONL file.  Can be overridden before the app starts by
# setting the CALLBACK_LISTENER_JSONL env var, useful in tests.
_DEFAULT_JSONL = Path("callbacks.jsonl")


def _jsonl_path() -> Path:
    env = os.environ.get("CALLBACK_LISTENER_JSONL")
    return Path(env) if env else _DEFAULT_JSONL


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="callback-listener", version="0.1.0")


@app.post("/status", status_code=200)
async def receive_status(payload: CallbackPayload) -> dict[str, str]:
    """Accept a callback payload, persist it, and return ``{"ok": "true"}``."""
    raw: dict[str, Any] = payload.model_dump()

    key: tuple[str, str, str, int, int] = (
        raw["run_id"],
        raw["stage_id"],
        raw["cell_id"],
        raw["host_port"],
        raw["shard_id"],
    )

    # In-memory append
    if key not in _store:
        _store[key] = []
    _store[key].append(raw)

    # Disk append
    try:
        with _jsonl_path().open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(raw, ensure_ascii=False) + "\n")
    except OSError as exc:
        # Non-fatal: log to stderr and continue — do not break the response.
        import sys
        print(f"[callback_listener] WARNING: could not write to JSONL: {exc}", file=sys.stderr)

    return {"ok": "true"}


# ---------------------------------------------------------------------------
# Query helpers (importable by tests / coordinator)
# ---------------------------------------------------------------------------

def get_callbacks(
    run_id: str,
    stage_id: str,
    cell_id: str,
) -> list[dict[str, Any]]:
    """Return all stored callbacks matching *(run_id, stage_id, cell_id)*.

    The *host_port* and *shard_id* dimensions are not filtered here — callers
    receive every shard/host entry for the given cell so they can aggregate
    across workers.

    Returns an empty list when no matching callbacks have been received.
    """
    results: list[dict[str, Any]] = []
    for (r, s, c, _hp, _sh), payloads in _store.items():
        if r == run_id and s == stage_id and c == cell_id:
            results.extend(payloads)
    return results


def clear() -> None:
    """Reset all in-memory state (does not truncate the JSONL file)."""
    _store.clear()


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "agent_sim.callback_listener:app",
        host="127.0.0.1",
        port=17700,
        reload=False,
        log_level="info",
    )
