"""S05B mocked end-to-end test (S05B.12, S05B.30).

Exercises the V0.2.2 production runner against a fake stage dispatcher
that mimics the remote /v022/stage/{stage} endpoint, plus a stub local
evaluator. Verifies:

* CLI -> V022Runner -> V022Orchestrator path runs to completion.
* Canonical cell-keyed artifacts land under outputs/v02/{run_id}/...
* memory_v022.upsert_long_memory_rows is called with rows that include run_id.
* No /stage/all endpoint is touched; only /v022/stage/{stage}.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from system_prompt_retrieval_agent.agent_loop_v022 import StageDispatcher
from system_prompt_retrieval_agent.config import (
    AppConfig, ApiConfig, BudgetConfig, EvaluationConfig, ExecutionConfig,
    GemmaUserPromptEntry, GemmaUserPromptsConfig, PathsConfig,
    PromptGenerationConfig, RemoteConfig, WorkflowConfig,
)
from system_prompt_retrieval_agent.remote._vendored import canonical_paths as cp
from system_prompt_retrieval_agent.runner_v022 import V022Runner


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeMemoryManager:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._schedule: dict = {"current_round": 1}
        self.long_rows: list[dict] = []

    def load_schedule(self):
        return dict(self._schedule)

    def save_schedule(self, data):
        self._schedule = dict(data)

    def determine_round_id(self):
        return 1

    def mark_fallback_round(self, round_id, is_fallback):
        return None

    def append_long_memory(self, row):
        self.long_rows.append(dict(row))

    def write_pair(self, *_a, **_kw):
        return None

    def prune_long_memory(self, *_a, **_kw):
        return None

    def archive_run_state(self, *_a, **_kw):
        return []

    def read_pair(self, *_a, **_kw):
        return None

    def load_for_generation(self, *_a, **_kw):
        return {}

    def mark_round_complete(self, *_a, **_kw):
        return None

    def reset_fallback(self, *_a, **_kw):
        return None

    def write_round_summary(self, *_a, **_kw):
        return None


class _FakeLocalEvaluator:
    def __init__(self):
        self.called = False

    async def evaluate_many_cells(self, cells, *, api_eval_root=None):  # noqa: ARG002
        self.called = True
        out = []
        for c in cells:
            out.append({
                "prompt_pair_id": c["prompt_pair_id"],
                "user_prompt_id": c["user_prompt_id"],
                "sample_id": c["sample_id"],
                "score": 0.85,
                "qwen_pass_rate": 0.85,
                "edit_correctness": 0.85,
                "garment_transfer_correctness": 0.80,
                "preservation": 0.90,
                "artifact_penalty": 0.05,
            })
        return out


class _FakeStageDispatcher(StageDispatcher):
    """In-memory dispatcher that lays down a remote tree the copy-back can pull."""

    def __init__(self, *, remote_root: Path, run_id: str, round_id: int):
        self._remote_root = remote_root
        self._run_id = run_id
        self._round_id = round_id
        super().__init__(post_v022=self._mock_post, production_mode=True)

    def _mock_post(self, stage: str, payload: dict) -> dict:
        attempt_id = payload["attempt_id"]
        cells_out = []
        for row in payload["dispatch_cells"]:
            relpath = cp.cell_artifact_path(
                self._run_id, stage, self._round_id,
                row["prompt_pair_id"], row["user_prompt_id"], row["sample_id"],
            )
            full = self._remote_root / relpath
            full.parent.mkdir(parents=True, exist_ok=True)
            payload_bytes = (
                f"{stage}|{attempt_id}|{row['prompt_pair_id']}|"
                f"{row['user_prompt_id']}|{row['sample_id']}"
            ).encode("utf-8")
            full.write_bytes(payload_bytes)
            cells_out.append({
                **row, "status": "ok", "artifact_relpath": relpath,
                "artifact_size_bytes": len(payload_bytes),
                "artifact_sha256": hashlib.sha256(payload_bytes).hexdigest(),
            })
        per_up: dict = {}
        pair_rollups: dict = {}
        for c in cells_out:
            pair_rollups.setdefault(c["prompt_pair_id"], {"ok": 0, "errors": 0, "total": 0})
            pair_rollups[c["prompt_pair_id"]]["ok"] += 1
            pair_rollups[c["prompt_pair_id"]]["total"] += 1
            per_up.setdefault(c["prompt_pair_id"], {})
            per_up[c["prompt_pair_id"]].setdefault(c["user_prompt_id"], {"ok": 0, "errors": 0, "total": 0})
            per_up[c["prompt_pair_id"]][c["user_prompt_id"]]["ok"] += 1
            per_up[c["prompt_pair_id"]][c["user_prompt_id"]]["total"] += 1
        manifest = {
            **{k: payload[k] for k in cp.MATCH_FIELDS},
            "attempt_id": attempt_id,
            "lifecycle_mode": payload["lifecycle_mode"],
            "lifecycle_state_after": payload["lifecycle_state_after"],
            "cells": cells_out,
            "pair_rollups": pair_rollups,
            "per_user_prompt": per_up,
        }
        relpath_mf = cp.stage_manifest_attempt_path(
            self._run_id, stage, self._round_id, attempt_id,
        )
        full_mf = self._remote_root / relpath_mf
        full_mf.parent.mkdir(parents=True, exist_ok=True)
        full_mf.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return manifest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_cfg(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(
        paths=PathsConfig(
            local_project_root=str(tmp_path),
            local_agent_root=str(tmp_path),
            remote_image_service_root=str(tmp_path / "remote_root"),
            dataset_root=str(tmp_path / "data"),
            output_root=str(tmp_path / "out"),
            memory_root=str(tmp_path / "mem"),
            prompt_seed_file=str(tmp_path / "seed.json"),
            env_file=str(tmp_path / ".env"),
        ),
        remote=RemoteConfig(tunnel_command="ssh -L"),
        api=ApiConfig(),
        workflow=WorkflowConfig(max_rounds=1, score_threshold=2.0),
        budget=BudgetConfig(daily_usd_cap=100.0, per_round_usd_cap=10.0),
        prompt_generation=PromptGenerationConfig(prompt_pairs_per_round=2),
        gemma_user_prompts=GemmaUserPromptsConfig(
            library=[
                GemmaUserPromptEntry(user_prompt_id="UP_A", language="en", text="A", enabled=True),
                GemmaUserPromptEntry(user_prompt_id="UP_B", language="en", text="B", enabled=True),
            ],
            corpus_id="ut1",
        ),
        evaluation=EvaluationConfig(run_local_api_eval=True),
        execution=ExecutionConfig(mode="v022_stage_major"),
    )
    # Build a minimal sample manifest at the dataset root
    Path(cfg.paths.dataset_root).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.dataset_root, "sample_manifest.json").write_text(
        json.dumps([
            {"sample_id": "S1", "model_image_path": "/d/m1.png", "cloth_image_path": "/d/c1.png"},
            {"sample_id": "S2", "model_image_path": "/d/m2.png", "cloth_image_path": "/d/c2.png"},
        ])
    )
    Path(cfg.paths.memory_root).mkdir(parents=True, exist_ok=True)
    return cfg


# ---------------------------------------------------------------------------
# S05B.12 — full mocked end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v022_runner_end_to_end_writes_canonical_artifacts_and_memory(
    tmp_path, monkeypatch
):
    cfg = _make_cfg(tmp_path)

    # Stub prompt generation so we don't need OpenAI.
    from system_prompt_retrieval_agent.schemas import PromptPair

    async def _fake_generate(cfg_, mem, round_id, n, existing):  # noqa: ARG001
        out = [
            PromptPair(
                prompt_pair_id=f"PP{i+1}",
                round_id=round_id,
                system_prompt_id=f"sp{i+1}",
                system_prompt=f"system prompt {i+1}",
                negative_prompt_id=f"np{i+1}",
                negative_prompt=f"neg {i+1}",
            )
            for i in range(n)
        ]
        return out, False

    monkeypatch.setattr(
        "system_prompt_retrieval_agent.runner_v022.generate_prompt_pairs",
        _fake_generate,
    )

    # Stub copy-back transport: the orchestrator's copy-back resolves
    # cells under the live artifact root (cfg.paths.output_root) by
    # default; we redirect dispatcher to write into a separate "remote"
    # tree and use a local-copy transport.
    remote_root = tmp_path / "remote_root" / "outputs" / "v02"
    remote_root.mkdir(parents=True, exist_ok=True)

    # We need the runner to use our fake dispatcher whose remote_root
    # matches the orchestrator copy-back source. Inject a custom
    # dispatcher; build it ad-hoc per-run since attempt_ids/run_id
    # differ.
    cfg.paths.output_root = str(tmp_path / "live")
    Path(cfg.paths.output_root).mkdir(parents=True, exist_ok=True)

    fake_mem = _FakeMemoryManager(Path(cfg.paths.memory_root))
    fake_eval = _FakeLocalEvaluator()

    # Create the runner without dispatcher first to learn the run_id;
    # then build the dispatcher around that run_id and inject it.
    from system_prompt_retrieval_agent.agent_loop_v022 import V022Orchestrator

    runner = V022Runner(
        cfg, max_rounds=1,
        memory=fake_mem, local_evaluator=fake_eval,
        dispatcher=StageDispatcher(post_v022=lambda *_: {}, production_mode=True),
    )
    # Replace the dispatcher with a real fake bound to runner.run_id.
    fake_disp = _FakeStageDispatcher(
        remote_root=remote_root, run_id=runner.run_id, round_id=1,
    )
    runner._dispatcher = fake_disp

    # Patch the orchestrator's copy-back transport so it knows where to
    # find cells (our remote tree).
    orig_init = V022Orchestrator.__init__

    def _init(self, *, dispatcher, copy_back_transport=None):  # type: ignore[no-redef]
        def _local_copy(*, remote_root, local_temp, cell_relpaths, manifest_relpath):
            relpaths = list(cell_relpaths)
            if manifest_relpath:
                relpaths.append(manifest_relpath)
            for relpath in relpaths:
                src = Path(remote_root) / relpath
                if not src.is_file():
                    continue
                dst = Path(local_temp) / relpath
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

        orig_init(self, dispatcher=dispatcher, copy_back_transport=_local_copy)

    monkeypatch.setattr(V022Orchestrator, "__init__", _init)

    # Override the orchestrator's remote artifact root inside RoundContext
    # so copy-back finds our `remote_root`. The RoundContext built by the
    # runner uses `cfg.paths.remote_image_service_root + "/outputs/v02"`,
    # which we already set to `tmp_path / "remote_root"` — matches `remote_root`.

    outcome = await runner.run()
    assert outcome.run_id == runner.run_id
    assert outcome.rounds_completed == 1
    # Every dispatched endpoint went to /v022/stage/, never /stage/all
    assert all(e.startswith("/v022/stage/") for e in fake_disp.dispatched_endpoints)
    assert "/stage/all" not in "".join(fake_disp.dispatched_endpoints)

    # Canonical artifacts on disk under live output root
    for stage in ("gemma", "flux", "qwen"):
        for pid in ("PP1", "PP2"):
            for upid in ("UP_A", "UP_B"):
                for sid in ("S1", "S2"):
                    p = (
                        Path(cfg.paths.output_root)
                        / cp.cell_artifact_path(runner.run_id, stage, 1, pid, upid, sid)
                    )
                    assert p.is_file(), f"missing canonical artifact: {p}"

    # memory_v022 row written with run_id
    long_mem = Path(cfg.paths.memory_root) / "long_memory.csv"
    assert long_mem.is_file(), "memory_v022 long_memory.csv was not written"
    with long_mem.open() as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert rows, "memory_v022 wrote no rows"
    for r in rows:
        assert r.get("run_id") == runner.run_id
        assert r.get("schema_version") == "0.2.2"

    # next_round_context persisted
    nrc = Path(cfg.paths.output_root) / "runs" / runner.run_id / "rounds" / "round_001" / "next_round_context.json"
    assert nrc.is_file()
    nrc_payload = json.loads(nrc.read_text())
    assert nrc_payload, "next_round_context empty"
    # New V0.2.2 fields propagate through as_dict() into next-round handoff.
    for k in ("mean_qwen_pass_rate", "mean_edit_correctness",
              "mean_garment_transfer_correctness", "mean_preservation",
              "mean_artifact_penalty", "total_score", "per_category",
              "worst_cells"):
        assert k in nrc_payload[0], f"next_round_context missing {k}"

    # failed_cells.json is written unconditionally (bottom-N policy).
    fc = (Path(cfg.paths.output_root) / "runs" / runner.run_id
          / "rounds" / "round_001" / "failures" / "failed_cells.json")
    assert fc.is_file(), "failed_cells.json was not written"
    fc_payload = json.loads(fc.read_text())
    assert isinstance(fc_payload, list)
    # Even with all overall=0.85 (well above any threshold), bottom-N should
    # surface up to 50 cells across both pairs.
    assert fc_payload, "failed_cells.json should be non-empty (bottom-N, no threshold)"
    for entry in fc_payload:
        assert "prompt_pair_id" in entry
        assert "user_prompt_id" in entry
        assert "sample_id" in entry
        assert "overall" in entry

    # long_memory.csv: sub-score columns now non-null AND scores_missing
    # sentinels cleared (the bug we fixed).
    nonnull_qpr = sum(1 for r in rows if r.get("qwen_pass_rate") not in (None, ""))
    nonnull_ec = sum(1 for r in rows if r.get("edit_correctness") not in (None, ""))
    assert nonnull_qpr == len(rows), (
        f"qwen_pass_rate should be populated on every row; got {nonnull_qpr}/{len(rows)}"
    )
    assert nonnull_ec == len(rows), (
        f"edit_correctness should be populated on every row; got {nonnull_ec}/{len(rows)}"
    )
    sentinel_count = sum(
        1 for r in rows if r.get("missing_score_reason") == "scores_missing"
    )
    assert sentinel_count == 0, (
        f"missing_score_reason='scores_missing' should be cleared after backfill; "
        f"got {sentinel_count}/{len(rows)} rows still flagged"
    )
