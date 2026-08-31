"""S05B production-wiring tests.

Covers:
* S05B.02 — RemoteControllerClient.post_v022_stage hits /v022/stage/{stage}
* S05B.07 — --resume-from-run-id reaches V0.2.2 runner with hash drift gate
* S05B.11 — production CLI invokes V022Orchestrator and never /stage/all
* S05B.13 — production-mode regression: /stage/all is hard-failed
* S05B.14 — transport URL test for gemma/flux/qwen
* S05B.15 — distinct user_prompt_ids → distinct artifact paths/scoring rows
* S05B.20–S05B.22 — production sync StageDispatcher returns dict
* S05B.31 — RoundExecutionResult dataclass shape
* S05B.32 — V0.2.2 runner does not import scoring.ranking.pair_row_for_csv
* S05B.33 — dispatcher owns client lifecycle on a single private loop
"""

from __future__ import annotations

import asyncio
import json
import inspect
import threading
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from system_prompt_retrieval_agent.agent_loop_v022 import (
    RoundExecutionResult,
    StageDispatcher,
    V022Orchestrator,
    V022ProductionViolation,
)
from system_prompt_retrieval_agent.execution_mode_v022 import (
    PRODUCTION_MODE,
    assert_production_mode_does_not_call_stage_all,
    resolve_execution_mode,
)
from system_prompt_retrieval_agent.remote.client import RemoteControllerClient
from system_prompt_retrieval_agent.scoring_v022 import (
    EvalCell,
    PairRankingRow,
    build_next_round_context_v022,
    build_pair_ranking_rows,
)


# ---------------------------------------------------------------------------
# S05B.02 / S05B.14 — transport URL routing
# ---------------------------------------------------------------------------


class _StubResp:
    status_code = 200

    def __init__(self, payload: dict):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


class _StubAsyncClient:
    def __init__(self):
        self.posted: list[tuple[str, dict]] = []
        self.aclose_calls = 0

    async def post(self, url, json=None):  # noqa: ARG002
        self.posted.append((url, json))
        return _StubResp({"stage": url.rstrip("/").rsplit("/", 1)[-1], "ok": True})

    async def aclose(self):
        self.aclose_calls += 1


@pytest.mark.asyncio
async def test_post_v022_stage_routes_to_v022_endpoint():
    stub = _StubAsyncClient()
    client = RemoteControllerClient("http://h:1", http_client=stub)
    for stage in ("gemma", "flux", "qwen"):
        body = await client.post_v022_stage(stage, {"any": 1})
        assert body == {"stage": stage, "ok": True}
    urls = [u for u, _ in stub.posted]
    assert urls == [
        "http://h:1/v022/stage/gemma",
        "http://h:1/v022/stage/flux",
        "http://h:1/v022/stage/qwen",
    ]


@pytest.mark.asyncio
async def test_post_v022_stage_never_hits_stage_all():
    stub = _StubAsyncClient()
    client = RemoteControllerClient("http://h:1", http_client=stub)
    await client.post_v022_stage("gemma", {"x": 1})
    assert all("/v022/stage/" in u for u, _ in stub.posted)
    assert not any(u.endswith("/stage/all") for u, _ in stub.posted)


# ---------------------------------------------------------------------------
# S05B.20–S05B.22 / S05B.33 — production sync StageDispatcher
# ---------------------------------------------------------------------------


class _AsyncFakeClient:
    """Async client double for StageDispatcher.from_client tests."""

    def __init__(self):
        self.entered = False
        self.exited = False
        self.posts: list[tuple[str, dict]] = []
        self.loop_when_post: asyncio.AbstractEventLoop | None = None

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, *_):
        self.exited = True

    async def post_v022_stage(self, stage: str, payload: dict) -> dict:
        self.loop_when_post = asyncio.get_event_loop()
        self.posts.append((stage, payload))
        return {"stage": stage, "ok": True}


def test_from_client_returns_dict_not_coroutine():
    fc = _AsyncFakeClient()
    d = StageDispatcher.from_client(lambda: fc, production_mode=True)
    try:
        for stage in ("gemma", "flux", "qwen"):
            res = d.dispatch(stage, {"x": stage})
            assert isinstance(res, dict)
            assert not asyncio.iscoroutine(res)
            assert res == {"stage": stage, "ok": True}
        assert d.dispatched_endpoints == [
            "/v022/stage/gemma", "/v022/stage/flux", "/v022/stage/qwen",
        ]
    finally:
        d.close()


def test_from_client_owns_single_client_lifecycle_on_private_loop():
    fc = _AsyncFakeClient()
    d = StageDispatcher.from_client(lambda: fc, production_mode=True)
    private_loop = d._private_loop  # type: ignore[attr-defined]
    try:
        d.dispatch("gemma", {})
        d.dispatch("flux", {})
        d.dispatch("qwen", {})
        # Same client across all three dispatches (no per-call construction).
        assert len(fc.posts) == 3
        # Client was entered exactly once on the private loop.
        assert fc.entered is True
        assert fc.loop_when_post is private_loop
    finally:
        d.close()
    assert fc.exited is True
    assert private_loop.is_closed() is True


def test_from_client_dispatch_stage_all_hard_fails():
    d = StageDispatcher.from_client(lambda: _AsyncFakeClient(), production_mode=True)
    try:
        with pytest.raises(V022ProductionViolation, match="/stage/all"):
            d.dispatch_stage_all()
    finally:
        d.close()


def test_dispatch_rejects_coroutine_payload():
    async def _bad(_s, _p):  # async function returning coroutine, simulates misuse
        return {"ok": True}

    d = StageDispatcher(post_v022=_bad, production_mode=True)
    with pytest.raises(V022ProductionViolation, match="must return dict"):
        d.dispatch("gemma", {})


# ---------------------------------------------------------------------------
# S05B.31 — RoundExecutionResult shape
# ---------------------------------------------------------------------------


def test_round_execution_result_is_dataclass_with_stage_results():
    r = RoundExecutionResult(run_id="x", round_id=2)
    assert r.stage_results == {}
    assert r.summary == {}
    r.stage_results["gemma"] = "X"  # smoke insertion ok
    assert r.stage_results["gemma"] == "X"


# ---------------------------------------------------------------------------
# S05B.13 — production mode hard-fails on /stage/all
# ---------------------------------------------------------------------------


def test_production_mode_hard_fails_on_stage_all():
    from system_prompt_retrieval_agent.execution_mode_v022 import ExecutionModeError

    with pytest.raises(ExecutionModeError, match="/stage/all"):
        assert_production_mode_does_not_call_stage_all(
            PRODUCTION_MODE, "/stage/all",
        )


def test_production_mode_default_resolves_to_v022_stage_major():
    assert resolve_execution_mode({}) == PRODUCTION_MODE
    assert PRODUCTION_MODE == "v022_stage_major"


# ---------------------------------------------------------------------------
# S05B.11 — CLI dispatch path (no /stage/all in production)
# ---------------------------------------------------------------------------


def test_cli_dispatch_run_routes_production_to_v022_runner(tmp_path, monkeypatch):
    """Production-mode CLI dispatch must invoke V022Runner, not AgentLoop."""
    from system_prompt_retrieval_agent import cli
    from system_prompt_retrieval_agent.config import (
        AppConfig, PathsConfig, RemoteConfig, ApiConfig, ExecutionConfig,
    )

    cfg = AppConfig(
        paths=PathsConfig(
            local_project_root=str(tmp_path),
            local_agent_root=str(tmp_path),
            remote_image_service_root="/tmp/remote",
            dataset_root=str(tmp_path / "data"),
            output_root=str(tmp_path / "out"),
            memory_root=str(tmp_path / "mem"),
            prompt_seed_file=str(tmp_path / "seed"),
            env_file=str(tmp_path / ".env"),
        ),
        remote=RemoteConfig(tunnel_command="ssh -L"),
        api=ApiConfig(),
        execution=ExecutionConfig(mode="v022_stage_major"),
    )

    # Replace V022Runner with a spy so we can confirm it was instantiated.
    invoked = {"hit": False}

    class _SpyRunner:
        def __init__(self, *a, **kw):
            invoked["hit"] = True

        async def run(self):
            from system_prompt_retrieval_agent.runner_v022 import V022RunnerOutcome
            return V022RunnerOutcome(run_id="x", stop_reason="threshold_met")

    monkeypatch.setattr(
        "system_prompt_retrieval_agent.runner_v022.V022Runner", _SpyRunner,
    )
    rc = cli._dispatch_run(cfg)
    assert rc == 0
    assert invoked["hit"] is True


def test_cli_dispatch_legacy_mode_still_uses_agent_loop(tmp_path, monkeypatch):
    """Legacy compat modes are explicitly reachable; production path is not."""
    from system_prompt_retrieval_agent import cli
    from system_prompt_retrieval_agent.config import (
        AppConfig, PathsConfig, RemoteConfig, ApiConfig, ExecutionConfig,
    )

    cfg = AppConfig(
        paths=PathsConfig(
            local_project_root=str(tmp_path),
            local_agent_root=str(tmp_path),
            remote_image_service_root="/tmp/remote",
            dataset_root=str(tmp_path / "data"),
            output_root=str(tmp_path / "out"),
            memory_root=str(tmp_path / "mem"),
            prompt_seed_file=str(tmp_path / "seed"),
            env_file=str(tmp_path / ".env"),
        ),
        remote=RemoteConfig(tunnel_command="ssh -L"),
        api=ApiConfig(),
        execution=ExecutionConfig(mode="legacy_cartesian_compat"),
    )

    invoked = {"hit": False}

    class _SpyAgentLoop:
        def __init__(self, *a, **kw):
            invoked["hit"] = True

        async def run(self):
            return 0

    monkeypatch.setattr(
        "system_prompt_retrieval_agent.agent_loop.AgentLoop", _SpyAgentLoop,
    )
    rc = cli._dispatch_run(cfg)
    assert rc == 0
    assert invoked["hit"] is True


# ---------------------------------------------------------------------------
# S05B.32 — V0.2.2 runner must not import legacy pair_row_for_csv
# ---------------------------------------------------------------------------


def test_runner_v022_does_not_import_legacy_pair_row_for_csv():
    """The runner module must not import or call the legacy ranking helper.

    Use AST inspection rather than substring scanning so the forbidden
    symbol may still appear in docstrings (which document the rule).
    """
    import ast

    mod = __import__(
        "system_prompt_retrieval_agent.runner_v022", fromlist=["x"]
    )
    assert not hasattr(mod, "pair_row_for_csv")
    src_path = Path(inspect.getfile(mod))
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "scoring.ranking" not in node.module, (
                f"runner_v022 must not import from scoring.ranking; "
                f"got {node.module}"
            )
            for alias in node.names:
                assert alias.name != "pair_row_for_csv"
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "scoring.ranking" not in alias.name


# ---------------------------------------------------------------------------
# S05B.15 — distinct user_prompt_ids produce distinct ranking rows + paths
# ---------------------------------------------------------------------------


def test_distinct_user_prompt_ids_yield_distinct_artifact_paths():
    from system_prompt_retrieval_agent.remote._vendored import canonical_paths as cp

    p1 = cp.cell_artifact_path("20260101T000000Z-deadbeef", "gemma", 0, "PP1", "UP_A", "S1")
    p2 = cp.cell_artifact_path("20260101T000000Z-deadbeef", "gemma", 0, "PP1", "UP_B", "S1")
    assert p1 != p2
    assert "UP_A" in p1
    assert "UP_B" in p2


def test_pair_ranking_rows_keep_per_user_prompt_means():
    cells_by_pair = {
        "PP1": [
            EvalCell(
                prompt_pair_id="PP1", user_prompt_id="UP_A", sample_id="S1",
                intermediate_prompt="x", generated_image_path="x",
                qwen_eval_json_path="x", model_image_path="x", cloth_image_path="x",
            ),
            EvalCell(
                prompt_pair_id="PP1", user_prompt_id="UP_B", sample_id="S1",
                intermediate_prompt="x", generated_image_path="x",
                qwen_eval_json_path="x", model_image_path="x", cloth_image_path="x",
            ),
        ]
    }
    # Each row must carry the 4 canonical axes so the aggregator's
    # _dedupe_complete filter (Level-1) accepts it. Values are tuned so
    # _cell_score → 0.8 / 0.4 with the default weights, matching the
    # original assertion targets.
    results_by_pair = {
        "PP1": [
            {"user_prompt_id": "UP_A", "sample_id": "S1", "prompt_pair_id": "PP1",
             "score": 0.8,
             "edit_correctness": 0.8, "garment_transfer_correctness": 0.8,
             "preservation": 0.8, "artifact_penalty": 0.0},
            {"user_prompt_id": "UP_B", "sample_id": "S1", "prompt_pair_id": "PP1",
             "score": 0.4,
             "edit_correctness": 0.4, "garment_transfer_correctness": 0.4,
             "preservation": 0.4, "artifact_penalty": 0.0},
        ],
    }
    rows = build_pair_ranking_rows(
        run_id="20260101T000000Z-deadbeef", round_id=0,
        eval_cells_by_pair=cells_by_pair,
        cell_results_by_pair=results_by_pair,
    )
    assert len(rows) == 1
    r = rows[0]
    assert r.prompt_pair_id == "PP1"
    assert r.n_user_prompts == 2
    assert r.n_cells == 2
    assert r.per_user_prompt_overall["UP_A"] == pytest.approx(0.8)
    assert r.per_user_prompt_overall["UP_B"] == pytest.approx(0.4)
    assert r.pair_overall == pytest.approx(0.6)
    # run_id-keyed, not legacy schema
    d = r.as_dict()
    assert d["run_id"] == "20260101T000000Z-deadbeef"
    assert "schema_version" not in d  # not legacy schema


def test_build_next_round_context_v022_returns_top_n_dicts():
    rows = [
        PairRankingRow(run_id="r", round_id=0, prompt_pair_id="PP1", pair_overall=0.9, n_cells=1, n_user_prompts=1),
        PairRankingRow(run_id="r", round_id=0, prompt_pair_id="PP2", pair_overall=0.7, n_cells=1, n_user_prompts=1),
        PairRankingRow(run_id="r", round_id=0, prompt_pair_id="PP3", pair_overall=0.5, n_cells=1, n_user_prompts=1),
    ]
    out = build_next_round_context_v022(rows, top_n=2)
    assert len(out) == 2
    assert out[0]["prompt_pair_id"] == "PP1"
    assert out[1]["prompt_pair_id"] == "PP2"


# ---------------------------------------------------------------------------
# S05B.07 — resume regex / runner accepts the flag
# ---------------------------------------------------------------------------


def test_runner_rejects_invalid_resume_run_id(tmp_path):
    from system_prompt_retrieval_agent.config import (
        AppConfig, PathsConfig, RemoteConfig, ApiConfig, ExecutionConfig,
    )
    from system_prompt_retrieval_agent.execution_mode_v022 import ExecutionModeError
    from system_prompt_retrieval_agent.runner_v022 import V022Runner

    cfg = AppConfig(
        paths=PathsConfig(
            local_project_root=str(tmp_path),
            local_agent_root=str(tmp_path),
            remote_image_service_root="/tmp/remote",
            dataset_root=str(tmp_path / "data"),
            output_root=str(tmp_path / "out"),
            memory_root=str(tmp_path / "mem"),
            prompt_seed_file=str(tmp_path / "seed"),
            env_file=str(tmp_path / ".env"),
        ),
        remote=RemoteConfig(tunnel_command="ssh -L"),
        api=ApiConfig(),
        execution=ExecutionConfig(mode="v022_stage_major"),
    )
    with pytest.raises(ExecutionModeError, match="canonical regex"):
        V022Runner(cfg, resume_from_run_id="bogus-run-id")
