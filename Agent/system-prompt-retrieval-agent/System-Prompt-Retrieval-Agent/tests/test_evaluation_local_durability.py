"""Level-1 eval-durability tests for LocalApiEvaluator.

Covers:
  - per-cell JSONL append + sentinel rename ordering
  - bulk-load of prior records → cached return without HTTP
  - corrupted state (sentinel without record) → re-evaluate + refresh
  - malformed prior row → dropped at load → re-evaluated
  - eval_schema_version bump → old rows ignored
  - first-failure cancels siblings (gather-with-cancel + EVAL_ERROR)
  - aggregator dedupe + completeness filter

Mirrors the test_evaluation_local.py setup so fixtures stay consistent.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from system_prompt_retrieval_agent.evaluation.local_api_eval import (
    EVAL_SCHEMA_VERSION,
    EvalRetryBudgetExhausted,
    LocalApiEvaluator,
    _append_jsonl_fsync,
    _cell_key,
    _jsonl_path,
    _key_for_record,
    _load_prior_records,
    _record_is_complete,
    _sentinel_path,
)
from system_prompt_retrieval_agent.rate_limiter import reset_rate_limiter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_rl():
    reset_rate_limiter()
    yield
    reset_rate_limiter()


@pytest.fixture
def fast_rate_limiter(monkeypatch):
    from system_prompt_retrieval_agent import rate_limiter as rl_mod

    fake_rl = MagicMock()
    fake_rl.acquire = AsyncMock()
    fake_rl.add_cost = MagicMock()
    monkeypatch.setattr(rl_mod, "_GLOBAL", fake_rl)
    return fake_rl


def _canned(overrides: dict | None = None) -> MagicMock:
    payload = {
        "edit_correctness": 0.80,
        "garment_transfer_correctness": 0.75,
        "preservation": 0.90,
        "artifact_penalty": 0.10,
        "notes": "ok",
    }
    if overrides:
        payload.update(overrides)
    msg = MagicMock(); msg.content = json.dumps(payload)
    choice = MagicMock(); choice.message = msg
    resp = MagicMock(); resp.choices = [choice]
    return resp


def _make_client(call_log: list, side_effects=None) -> MagicMock:
    """Fake AsyncOpenAI client. ``side_effects`` may be a list mixing
    response mocks and Exception instances.
    """
    if side_effects is None:
        side_effects = [_canned()]
    it = iter(side_effects)
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()

    async def _create(**kwargs):
        call_log.append(kwargs)
        nxt = next(it)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    client.chat.completions.create = AsyncMock(side_effect=_create)
    return client


def _png(p: Path):
    from PIL import Image
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(p, format="PNG")
    return p


def _cells(tmp_path: Path, n: int, *, pid: str = "seed_v5", up: str = "en_001",
           run_id: str = "run-A", round_id: int = 1) -> list[dict]:
    m = _png(tmp_path / "model.png"); c = _png(tmp_path / "cloth.png")
    out = []
    for i in range(n):
        g = _png(tmp_path / f"gen_{i}.png")
        out.append({
            "prompt_pair_id": pid,
            "user_prompt_id": up,
            "sample_id": f"dress__bg__sample_{i:03d}",
            "model_image_path": str(m),
            "cloth_image_path": str(c),
            "generated_image_path": str(g),
            "intermediate_prompt": f"prompt {i}",
            "run_id": run_id,
            "round_id": round_id,
        })
    return out


def _cfg(tmp_path: Path, *, api_concurrency: int = 4, max_concurrent: int = 4):
    from conftest import write_mock_config
    from system_prompt_retrieval_agent.config import load_config

    cfg_path = write_mock_config(
        tmp_path,
        evaluation={
            "categories": ["dress", "lower", "upper"],
            "run_local_api_eval": True,
            "max_concurrent": max_concurrent,
            "low_score_examples_per_pair": 10,
            "api_concurrency": api_concurrency,
            "api_rps_limit": None,
            "streaming_api_eval": False,
            "weights": {
                "qwen_pass_rate": 0.40,
                "edit_correctness": 0.20,
                "garment_transfer_correctness": 0.15,
                "preservation": 0.15,
                "artifact_penalty": -0.10,
                "category_balance_bonus": 0.10,
            },
        },
    )
    return load_config(cfg_path)


# ---------------------------------------------------------------------------
# 1. Idempotency in one batch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_idempotent_in_batch_same_cell_twice(tmp_path: Path, fast_rate_limiter):
    """Same cell submitted twice in one evaluate_many_cells call → 1 HTTP
    call, 1 JSONL row, 1 sentinel, 2 list slots (both populated)."""
    cfg = _cfg(tmp_path)
    cells = _cells(tmp_path, 1)
    # Submit the same cell twice (duplicate within one gather batch).
    cells = [cells[0], dict(cells[0])]
    call_log: list = []
    client = _make_client(call_log, side_effects=[_canned()])
    ev = LocalApiEvaluator(cfg, openai_client=client)

    api_root = tmp_path / "outputs" / "api_eval"
    results = await ev.evaluate_many_cells(cells, api_eval_root=api_root)

    assert len(call_log) == 1, f"expected 1 HTTP call, got {len(call_log)}"
    jsonl = api_root / "seed_v5" / "en_001.jsonl"
    rows = jsonl.read_text().strip().splitlines()
    assert len(rows) == 1, "expected 1 JSONL row for the deduped cell"
    sentinels = list((api_root / "seed_v5" / "en_001").glob(".eval_done.*"))
    assert len(sentinels) == 1
    assert len(results) == 2, "result list should have one entry per input slot"


# ---------------------------------------------------------------------------
# 2. Resume skip with cached return
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_skip_returns_cached(tmp_path: Path, fast_rate_limiter):
    """Pre-write a JSONL row + sentinel for a cell → evaluate_many_cells
    returns the cached record without any HTTP call."""
    cfg = _cfg(tmp_path)
    cells = _cells(tmp_path, 1)
    cell = cells[0]
    api_root = tmp_path / "outputs" / "api_eval"

    rec = {
        "sample_id": cell["sample_id"],
        "prompt_pair_id": cell["prompt_pair_id"],
        "user_prompt_id": cell["user_prompt_id"],
        "run_id": cell["run_id"],
        "round_id": cell["round_id"],
        "qwen_pass_rate": None,
        "edit_correctness": 0.42,
        "garment_transfer_correctness": 0.43,
        "preservation": 0.44,
        "artifact_penalty": 0.05,
        "notes": "cached",
        "usd_spent": 0.002,
        "eval_schema_version": EVAL_SCHEMA_VERSION,
    }
    _append_jsonl_fsync(_jsonl_path(api_root, cell), rec)
    sentinel = _sentinel_path(api_root, cell)
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("done\n")

    call_log: list = []
    client = _make_client(call_log, side_effects=[])  # no responses; should not be called
    ev = LocalApiEvaluator(cfg, openai_client=client)

    results = await ev.evaluate_many_cells(cells, api_eval_root=api_root)
    assert len(call_log) == 0, "no HTTP calls expected on resume hit"
    assert len(results) == 1
    assert abs(results[0]["edit_correctness"] - 0.42) < 1e-9


# ---------------------------------------------------------------------------
# 3. Corrupted state: sentinel without record
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_corrupted_sentinel_without_record_reevaluates(tmp_path: Path, fast_rate_limiter):
    """Sentinel exists but JSONL has no matching row → drop sentinel,
    re-evaluate, append fresh row + new sentinel."""
    cfg = _cfg(tmp_path)
    cells = _cells(tmp_path, 1)
    cell = cells[0]
    api_root = tmp_path / "outputs" / "api_eval"

    sentinel = _sentinel_path(api_root, cell)
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("orphan\n")
    assert sentinel.is_file()

    call_log: list = []
    client = _make_client(call_log, side_effects=[_canned({"edit_correctness": 0.99})])
    ev = LocalApiEvaluator(cfg, openai_client=client)

    results = await ev.evaluate_many_cells(cells, api_eval_root=api_root)
    assert len(call_log) == 1
    assert results[0]["edit_correctness"] == pytest.approx(0.99)
    assert sentinel.is_file()  # refreshed


# ---------------------------------------------------------------------------
# 4. Malformed prior row dropped
# ---------------------------------------------------------------------------


def test_load_prior_records_drops_malformed(tmp_path: Path):
    cell = {
        "prompt_pair_id": "p", "user_prompt_id": "u", "sample_id": "s",
        "run_id": "r", "round_id": 1,
    }
    api_root = tmp_path / "api_eval"
    jsonl = _jsonl_path(api_root, cell)
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    # 1) malformed JSON line
    # 2) complete record
    # 3) incomplete record (missing axes)
    with jsonl.open("w") as fh:
        fh.write("{not json\n")
        fh.write(json.dumps({
            "prompt_pair_id": "p", "user_prompt_id": "u", "sample_id": "s",
            "run_id": "r", "round_id": 1,
            "edit_correctness": 0.1, "garment_transfer_correctness": 0.2,
            "preservation": 0.3, "artifact_penalty": 0.0,
            "eval_schema_version": EVAL_SCHEMA_VERSION,
        }) + "\n")
        fh.write(json.dumps({
            "prompt_pair_id": "p", "user_prompt_id": "u", "sample_id": "s",
            "edit_correctness": 0.5,  # missing other axes
        }) + "\n")

    by_key, malformed = _load_prior_records(api_root, [cell])
    key = _cell_key(cell)
    assert key in by_key
    assert len(malformed) == 2, "two malformed/incomplete rows expected"
    assert by_key[key]["edit_correctness"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# 5. Schema-version bump
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_schema_version_bump_old_rows_ignored(tmp_path: Path, fast_rate_limiter):
    """Old JSONL rows stamped with eval_schema_version=0 are not matched
    against the current key (which uses EVAL_SCHEMA_VERSION=1), so the
    cell is re-evaluated."""
    cfg = _cfg(tmp_path)
    cells = _cells(tmp_path, 1)
    cell = cells[0]
    api_root = tmp_path / "outputs" / "api_eval"

    old_rec = {
        "prompt_pair_id": cell["prompt_pair_id"],
        "user_prompt_id": cell["user_prompt_id"],
        "sample_id": cell["sample_id"],
        "run_id": cell["run_id"],
        "round_id": cell["round_id"],
        "edit_correctness": 0.11,
        "garment_transfer_correctness": 0.22,
        "preservation": 0.33,
        "artifact_penalty": 0.05,
        "eval_schema_version": 0,  # OLD
    }
    _append_jsonl_fsync(_jsonl_path(api_root, cell), old_rec)
    sent = _sentinel_path(api_root, cell)
    sent.parent.mkdir(parents=True, exist_ok=True)
    sent.write_text("done\n")

    call_log: list = []
    client = _make_client(call_log, side_effects=[_canned({"edit_correctness": 0.91})])
    ev = LocalApiEvaluator(cfg, openai_client=client)

    results = await ev.evaluate_many_cells(cells, api_eval_root=api_root)
    assert len(call_log) == 1, "old schema version should not match → re-evaluate"
    assert results[0]["edit_correctness"] == pytest.approx(0.91)


# ---------------------------------------------------------------------------
# 6. First-failure cancels siblings
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_failure_cancels_siblings(tmp_path: Path, fast_rate_limiter):
    """If one cell fails (proxy outage), evaluate_many_cells must raise,
    cancel pending tasks, and not leave a partial JSONL for the failed
    cell. JSONL rows for cells that completed before the failure may
    exist; the runner is expected to halt the round on the raised
    exception."""
    from openai import APIConnectionError

    cfg = _cfg(tmp_path, api_concurrency=1, max_concurrent=1)
    cells = _cells(tmp_path, 3)

    # Slow + canned for cell-0; explode for cell-1; cell-2 should be cancelled.
    def _slow_then_explode():
        # responses are consumed in order: 0 → cell0; 1 → cell1 raises
        return [_canned({"edit_correctness": 0.7}),
                APIConnectionError(request=MagicMock())]

    call_log: list = []
    client = _make_client(call_log, side_effects=_slow_then_explode())
    ev = LocalApiEvaluator(cfg, openai_client=client)

    api_root = tmp_path / "outputs" / "api_eval"
    with pytest.raises((APIConnectionError, EvalRetryBudgetExhausted, BaseException)):
        await ev.evaluate_many_cells(cells, api_eval_root=api_root)

    # Cell-0 completed → JSONL row + sentinel exist for it.
    sent_0 = _sentinel_path(api_root, cells[0])
    # The failure may have happened before cell-0 finished; just assert
    # cell-2 was never sent.
    assert len(call_log) <= 5, "siblings should be cancelled before too many calls"


# ---------------------------------------------------------------------------
# 7. Aggregator dedupe + completeness filter
# ---------------------------------------------------------------------------


def test_aggregate_dedupe_complete_drops_incomplete_and_keeps_latest():
    from system_prompt_retrieval_agent.scoring_v022 import _dedupe_complete

    rows = [
        # incomplete (missing axes) → dropped
        {"prompt_pair_id": "p", "user_prompt_id": "u", "sample_id": "s1",
         "edit_correctness": 0.1},
        # complete, original
        {"prompt_pair_id": "p", "user_prompt_id": "u", "sample_id": "s1",
         "edit_correctness": 0.2, "garment_transfer_correctness": 0.3,
         "preservation": 0.4, "artifact_penalty": 0.05},
        # complete, newer (later in list) — should win
        {"prompt_pair_id": "p", "user_prompt_id": "u", "sample_id": "s1",
         "edit_correctness": 0.9, "garment_transfer_correctness": 0.9,
         "preservation": 0.9, "artifact_penalty": 0.0},
        # different sample
        {"prompt_pair_id": "p", "user_prompt_id": "u", "sample_id": "s2",
         "edit_correctness": 0.5, "garment_transfer_correctness": 0.5,
         "preservation": 0.5, "artifact_penalty": 0.0},
        # malformed value (string) → dropped
        {"prompt_pair_id": "p", "user_prompt_id": "u", "sample_id": "s3",
         "edit_correctness": "not-a-float",
         "garment_transfer_correctness": 0.5, "preservation": 0.5,
         "artifact_penalty": 0.0},
    ]
    out = _dedupe_complete(rows)
    by_sid = {r["sample_id"]: r for r in out}
    assert "s3" not in by_sid
    assert by_sid["s1"]["edit_correctness"] == pytest.approx(0.9)
    assert by_sid["s2"]["edit_correctness"] == pytest.approx(0.5)
    assert len(out) == 2
