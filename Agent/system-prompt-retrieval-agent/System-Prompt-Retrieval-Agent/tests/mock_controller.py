"""Mock FastAPI controller for remote-client tests.

All stage scenarios are triggered by the ``run_id`` field of the request body.

V0.1 / V0.2 legacy scenarios (RemoteStageRequest):
  run_id == "LOCK_BUSY"      → HTTP 400 {"detail": "locked"}
  run_id == "TIMEOUT"        → sleeps 30 s then 200 happy-path
  run_id == "PARTIAL"        → 200 ok=1 errors=1 total=2
  run_id == "MISSING_SHARD"  → 200 ok=1 errors=0 total=2 (count mismatch)
  run_id == "VRAM_LEAK"      → 200 vram_free_gib=[50,60,55]
  run_id == "DUPLICATE_ID"   → 200 duplicate sample_id entries
  anything else              → happy path ok=2 errors=0 total=2

V0.2.1 stage scenarios (GemmaStageRequest / FluxStageRequest / QwenStageRequest):
  run_id == "V021_HAPPY"     → StageManifest with 2 pairs × 2 user_prompts,
                               all ok; echoes user_prompt_corpus_hash.
  run_id == "V021_ONE_PAIR_FAILED" → pair_B in failed_pairs[], pair_A surviving.
  run_id == "V021_CELL_FAIL" → pair_A has one (pair, user_prompt) cell failed
                               but pair_A is in failed_pairs (strict default).
  run_id == "V021_PARTIAL"   → pair_A has one cell failed but stays in
                               surviving (allow_partial mode).
  run_id == "V021_CORPUS_MISMATCH" → echoes wrong corpus hash.
  run_id == "V021_NO_PURPOSE" → HTTP 400 missing_manifest_purpose when
                               sample_manifest_path set without purpose.
  run_id == "V021_LIFECYCLE_VIOLATION" → lifecycle_state_after="gpu_loaded"
                               (not a valid post-stage state).

Usage (ASGITransport — no real port binding)::

    import httpx
    from tests.mock_controller import mock_app

    transport = httpx.ASGITransport(app=mock_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.json() == {"status": "ok"}
"""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

mock_app = FastAPI(title="MockController")

# ---------------------------------------------------------------------------
# State exposed to tests for side-effect inspection.
# ---------------------------------------------------------------------------
call_counts: dict[str, int] = {}


def _inc(key: str) -> int:
    call_counts[key] = call_counts.get(key, 0) + 1
    return call_counts[key]


def reset_counts() -> None:
    """Reset all call counters (call from test setUp / fixture)."""
    call_counts.clear()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@mock_app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@mock_app.get("/status")
async def status() -> dict[str, Any]:
    return {"current_stage": None, "workers": [], "vram_free_gib": None}


def _make_manifest(
    stage: str,
    run_id: str,
    ok: int,
    errors: int,
    total: int,
    entries: list[dict],
    vram_free_gib: list[float] | None = None,
) -> dict:
    workers = [{"id": i, "status": "done"} for i in range(ok + errors)]
    return {
        "stage": stage,
        "run_id": run_id,
        "ok": ok,
        "errors": errors,
        "total": total,
        "entries": entries,
        "workers": workers,
        "vram_free_gib": vram_free_gib,
    }


def _happy_manifest(run_id: str) -> dict:
    entries = [
        {"sample_id": "s1", "status": "ok", "output_path": "out/s1.jpg", "error": None},
        {"sample_id": "s2", "status": "ok", "output_path": "out/s2.jpg", "error": None},
    ]
    return _make_manifest(
        stage="all",
        run_id=run_id,
        ok=2,
        errors=0,
        total=2,
        entries=entries,
        vram_free_gib=[72.0, 73.0, 74.0],
    )


@mock_app.post("/stage/{stage}")
async def run_stage(stage: str, request: Request) -> JSONResponse:
    body: dict = await request.json()
    run_id: str = body.get("run_id", "")

    _inc(f"stage/{stage}")

    # Route V0.2.1 batched payloads (those that carry prompt_pairs[]).
    if body.get("prompt_pairs") is not None or run_id.startswith("V021"):
        return _dispatch_v021(stage, body)

    if run_id == "LOCK_BUSY":
        return JSONResponse(status_code=400, content={"detail": "locked"})

    if run_id == "TIMEOUT":
        await asyncio.sleep(30)
        manifest = _happy_manifest(run_id)
        return JSONResponse(
            status_code=200,
            content={"ok": True, "stage": stage, "message": "", "run_id": run_id, "manifest": manifest},
        )

    if run_id == "PARTIAL":
        entries = [
            {"sample_id": "s1", "status": "ok", "output_path": "out/s1.jpg", "error": None},
            {"sample_id": "s2", "status": "error", "output_path": None, "error": "oops"},
        ]
        manifest = _make_manifest(
            stage=stage, run_id=run_id, ok=1, errors=1, total=2,
            entries=entries, vram_free_gib=[72.0, 73.0],
        )
        return JSONResponse(
            status_code=200,
            content={"ok": False, "stage": stage, "message": "partial", "run_id": run_id, "manifest": manifest},
        )

    if run_id == "MISSING_SHARD":
        entries = [
            {"sample_id": "s1", "status": "ok", "output_path": "out/s1.jpg", "error": None},
        ]
        manifest = _make_manifest(
            stage=stage, run_id=run_id, ok=1, errors=0, total=2,
            entries=entries, vram_free_gib=[72.0, 73.0],
        )
        return JSONResponse(
            status_code=200,
            content={"ok": False, "stage": stage, "message": "shard missing", "run_id": run_id, "manifest": manifest},
        )

    if run_id == "VRAM_LEAK":
        manifest = _make_manifest(
            stage=stage, run_id=run_id, ok=2, errors=0, total=2,
            entries=[
                {"sample_id": "s1", "status": "ok", "output_path": "out/s1.jpg", "error": None},
                {"sample_id": "s2", "status": "ok", "output_path": "out/s2.jpg", "error": None},
            ],
            vram_free_gib=[50.0, 60.0, 55.0],
        )
        return JSONResponse(
            status_code=200,
            content={"ok": True, "stage": stage, "message": "", "run_id": run_id, "manifest": manifest},
        )

    if run_id == "DUPLICATE_ID":
        entries = [
            {"sample_id": "dup", "status": "ok", "output_path": "out/dup.jpg", "error": None},
            {"sample_id": "dup", "status": "ok", "output_path": "out/dup2.jpg", "error": None},
        ]
        manifest = _make_manifest(
            stage=stage, run_id=run_id, ok=2, errors=0, total=2,
            entries=entries, vram_free_gib=[72.0, 73.0],
        )
        return JSONResponse(
            status_code=200,
            content={"ok": True, "stage": stage, "message": "", "run_id": run_id, "manifest": manifest},
        )

    # Default: happy path.
    manifest = _happy_manifest(run_id)
    return JSONResponse(
        status_code=200,
        content={"ok": True, "stage": stage, "message": "", "run_id": run_id, "manifest": manifest},
    )


@mock_app.post("/callback")
async def callback(request: Request) -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# V0.2.1 stage-manifest helpers
# ---------------------------------------------------------------------------


def _make_per_user_prompt(user_prompt_ids: list[str], ok: int = 1, errors: int = 0) -> dict:
    """Build a per_user_prompt rollup dict."""
    result = {}
    for up_id in user_prompt_ids:
        result[up_id] = {"ok": ok, "errors": errors, "total": ok + errors}
    return result


def _make_stage_manifest(
    stage: str,
    run_id: str,
    pairs_ok: list[str],
    pairs_failed: list[str],
    user_prompt_ids: list[str],
    corpus_hash: str,
    lifecycle_state_after: str = "disk_unloaded",
    vram_free_gib: list[float] | None = None,
    per_pair_ok: int = 1,
    per_pair_errors: int = 0,
    host_ram_free_gib: float | None = None,
) -> dict:
    """Build a V0.2.1 StageManifest response body."""
    n_up = len(user_prompt_ids)
    pairs = {}

    for pair_id in pairs_ok:
        per_up = _make_per_user_prompt(user_prompt_ids, ok=per_pair_ok, errors=per_pair_errors)
        pairs[pair_id] = {
            "prompt_pair_id": pair_id,
            "ok": per_pair_ok * n_up,
            "errors": per_pair_errors * n_up,
            "total": (per_pair_ok + per_pair_errors) * n_up,
            "per_user_prompt": per_up,
        }

    for pair_id in pairs_failed:
        per_up = _make_per_user_prompt(user_prompt_ids, ok=0, errors=1)
        pairs[pair_id] = {
            "prompt_pair_id": pair_id,
            "ok": 0,
            "errors": n_up,
            "total": n_up,
            "failure_reason": "worker_crash",
            "per_user_prompt": per_up,
        }

    failed_pairs_list = [
        {"prompt_pair_id": p, "failure_reason": "worker_crash"} for p in pairs_failed
    ]

    manifest = {
        "stage": stage,
        "run_id": run_id,
        "pairs": pairs,
        "surviving_pairs": list(pairs_ok),
        "failed_pairs": failed_pairs_list,
        "user_prompt_corpus_hash": corpus_hash,
        "lifecycle_state_after": lifecycle_state_after,
        "vram_free_gib": vram_free_gib or [72.0, 73.0, 74.0],
    }
    if host_ram_free_gib is not None:
        manifest["host_ram_free_gib"] = host_ram_free_gib

    return manifest


def _dispatch_v021(stage: str, body: dict) -> JSONResponse:
    """Handle V0.2.1 batched stage requests."""
    run_id: str = body.get("run_id", "")
    corpus_hash: str = body.get("user_prompt_corpus_hash", "a" * 64)
    user_prompts: list[dict] = body.get("user_prompts", [])
    user_prompt_ids = [up["user_prompt_id"] for up in user_prompts] or ["zh_001", "en_001"]
    prompt_pairs: list[dict] = body.get("prompt_pairs", [])
    pair_ids = [p["prompt_pair_id"] for p in prompt_pairs] or ["pair_A", "pair_B"]
    sample_manifest_path = body.get("sample_manifest_path")
    sample_manifest_path_purpose = body.get("sample_manifest_path_purpose")

    # Validate that sample_manifest_path requires a purpose.
    if sample_manifest_path is not None and sample_manifest_path_purpose is None:
        return JSONResponse(
            status_code=400,
            content={"detail": "missing_manifest_purpose"},
        )

    # Validate no system_prompt_text on FLUX/Qwen pairs.
    if stage in ("flux", "qwen"):
        for pair in prompt_pairs:
            if pair.get("system_prompt_text") is not None:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "system_prompt_text_forbidden"},
                )

    if run_id == "V021_HAPPY" or (run_id not in _V021_SPECIAL_RUN_IDS and run_id.startswith("V021")):
        # Generic V0.2.1 happy path.
        manifest = _make_stage_manifest(
            stage=stage,
            run_id=run_id,
            pairs_ok=pair_ids,
            pairs_failed=[],
            user_prompt_ids=user_prompt_ids,
            corpus_hash=corpus_hash,
        )
        return JSONResponse(status_code=200, content=manifest)

    if run_id == "V021_ONE_PAIR_FAILED":
        ok_pairs = pair_ids[:-1]
        fail_pairs = pair_ids[-1:]
        manifest = _make_stage_manifest(
            stage=stage,
            run_id=run_id,
            pairs_ok=ok_pairs,
            pairs_failed=fail_pairs,
            user_prompt_ids=user_prompt_ids,
            corpus_hash=corpus_hash,
        )
        return JSONResponse(status_code=200, content=manifest)

    if run_id == "V021_CELL_FAIL":
        # First pair has one (pair, user_prompt) cell failed — strict mode
        # should put entire pair in failed_pairs.
        # For simplicity: pair_A fails, pair_B survives.
        ok_pairs = pair_ids[1:] if len(pair_ids) > 1 else []
        fail_pairs = [pair_ids[0]] if pair_ids else []
        manifest = _make_stage_manifest(
            stage=stage,
            run_id=run_id,
            pairs_ok=ok_pairs,
            pairs_failed=fail_pairs,
            user_prompt_ids=user_prompt_ids,
            corpus_hash=corpus_hash,
        )
        return JSONResponse(status_code=200, content=manifest)

    if run_id == "V021_PARTIAL":
        # All pairs survive but first pair has one cell with errors
        # (allow_partial mode).
        n_up = len(user_prompt_ids)
        pairs_data = {}
        for i, pair_id in enumerate(pair_ids):
            if i == 0 and n_up > 1:
                # First user_prompt ok, second fails.
                per_up = {
                    user_prompt_ids[0]: {"ok": 1, "errors": 0, "total": 1},
                    user_prompt_ids[1]: {"ok": 0, "errors": 1, "total": 1},
                }
                pairs_data[pair_id] = {
                    "prompt_pair_id": pair_id,
                    "ok": 1,
                    "errors": 1,
                    "total": 2,
                    "per_user_prompt": per_up,
                }
            else:
                per_up = _make_per_user_prompt(user_prompt_ids)
                pairs_data[pair_id] = {
                    "prompt_pair_id": pair_id,
                    "ok": n_up,
                    "errors": 0,
                    "total": n_up,
                    "per_user_prompt": per_up,
                }
        manifest = {
            "stage": stage,
            "run_id": run_id,
            "pairs": pairs_data,
            "surviving_pairs": list(pair_ids),
            "failed_pairs": [],
            "user_prompt_corpus_hash": corpus_hash,
            "lifecycle_state_after": "disk_unloaded",
            "vram_free_gib": [72.0, 73.0, 74.0],
        }
        return JSONResponse(status_code=200, content=manifest)

    if run_id == "V021_CORPUS_MISMATCH":
        wrong_hash = "b" * 64
        manifest = _make_stage_manifest(
            stage=stage,
            run_id=run_id,
            pairs_ok=pair_ids,
            pairs_failed=[],
            user_prompt_ids=user_prompt_ids,
            corpus_hash=wrong_hash,  # intentionally different
        )
        return JSONResponse(status_code=200, content=manifest)

    if run_id == "V021_NO_PURPOSE":
        return JSONResponse(
            status_code=400,
            content={"detail": "missing_manifest_purpose"},
        )

    if run_id == "V021_LIFECYCLE_VIOLATION":
        manifest = _make_stage_manifest(
            stage=stage,
            run_id=run_id,
            pairs_ok=pair_ids,
            pairs_failed=[],
            user_prompt_ids=user_prompt_ids,
            corpus_hash=corpus_hash,
            lifecycle_state_after="gpu_loaded",  # not a valid post-stage state
        )
        return JSONResponse(status_code=200, content=manifest)

    # Fallback: return happy-path for unknown V021_ run_ids.
    manifest = _make_stage_manifest(
        stage=stage,
        run_id=run_id,
        pairs_ok=pair_ids,
        pairs_failed=[],
        user_prompt_ids=user_prompt_ids,
        corpus_hash=corpus_hash,
    )
    return JSONResponse(status_code=200, content=manifest)


_V021_SPECIAL_RUN_IDS = {
    "V021_HAPPY", "V021_ONE_PAIR_FAILED", "V021_CELL_FAIL", "V021_PARTIAL",
    "V021_CORPUS_MISMATCH", "V021_NO_PURPOSE", "V021_LIFECYCLE_VIOLATION",
}


# ---------------------------------------------------------------------------
# Override the generic stage route to handle V0.2.1 as well.
# ---------------------------------------------------------------------------

# Re-register /stage/{stage} to handle both legacy and V0.2.1 payloads.

_original_run_stage = mock_app.routes[-2].endpoint if len(mock_app.routes) >= 2 else None  # type: ignore[attr-defined]


@mock_app.post("/stage/v021/{stage}")
async def run_stage_v021(stage: str, request: Request) -> JSONResponse:
    """V0.2.1-only stage endpoint (dedicated path for clean testing)."""
    body: dict = await request.json()
    _inc(f"stage/v021/{stage}")
    return _dispatch_v021(stage, body)
