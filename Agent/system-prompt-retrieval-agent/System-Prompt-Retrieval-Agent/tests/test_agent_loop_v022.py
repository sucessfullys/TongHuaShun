"""S05 + S05.06 + S05.10–S05.13 tests for the V0.2.2 orchestrator."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from system_prompt_retrieval_agent.agent_loop_v022 import (
    CorpusContext,
    RoundContext,
    StageDispatcher,
    V022Orchestrator,
    V022ProductionViolation,
    gate_resume_from_run_id,
)
from system_prompt_retrieval_agent.remote._vendored import canonical_paths as cp
from system_prompt_retrieval_agent.resume_from_run_id import ResumeDriftError


_RID = "20260426T010000Z-deadbeef"


def _corpus(**overrides) -> CorpusContext:
    base = dict(
        enabled_prompt_pair_ids=("PP1", "PP2"),
        enabled_user_prompt_ids=("UP1",),
        configured_sample_ids=("S1", "S2"),
        user_prompt_corpus_id="v0",
        user_prompt_corpus_hash="u" * 64,
        prompt_pair_corpus_hash="p" * 64,
        sample_corpus_hash="s" * 64,
        config_hash="c" * 64,
    )
    base.update(overrides)
    return CorpusContext(**base)


def _ctx(tmp_path: Path, **overrides) -> RoundContext:
    base = dict(
        run_id=_RID, round_id=0,
        artifact_root=tmp_path / "live",
        remote_artifact_root=str(tmp_path / "remote"),
        corpus=_corpus(),
    )
    base.update(overrides)
    return RoundContext(**base)


def _expected_match(stage: str, ctx: RoundContext) -> dict:
    return {
        "schema_version": cp.SCHEMA_VERSION, "run_id": ctx.run_id,
        "round_id": ctx.round_id, "stage": stage,
        "config_hash": ctx.corpus.config_hash,
        "user_prompt_corpus_id": ctx.corpus.user_prompt_corpus_id,
        "user_prompt_corpus_hash": ctx.corpus.user_prompt_corpus_hash,
        "prompt_pair_corpus_hash": ctx.corpus.prompt_pair_corpus_hash,
        "sample_corpus_hash": ctx.corpus.sample_corpus_hash,
    }


# ---------------------------------------------------------------------------
# Fake transport + dispatcher: writes a remote tree the copy-back can pull
# ---------------------------------------------------------------------------


class _FakeStageDispatcher(StageDispatcher):
    """Dispatcher backed by an in-memory fake remote runner.

    For every dispatch, lays down the per-attempt manifest + cell
    artifacts under a 'remote' tmp tree, then returns the manifest as
    the stage response (mimicking the real /v022/stage/{stage}
    endpoint behavior).
    """

    def __init__(self, ctx: RoundContext, *, fail_for_cells=()):
        self._ctx = ctx
        self._fail = set(fail_for_cells)
        super().__init__(post_v022=self._mock_post, production_mode=True)

    def _mock_post(self, stage: str, payload: dict) -> dict:
        remote_root = Path(self._ctx.remote_artifact_root)
        attempt_id = payload["attempt_id"]
        cells_out = []
        for row in payload["dispatch_cells"]:
            key = (row["prompt_pair_id"], row["user_prompt_id"], row["sample_id"])
            relpath = cp.cell_artifact_path(
                self._ctx.run_id, stage, self._ctx.round_id,
                row["prompt_pair_id"], row["user_prompt_id"], row["sample_id"],
            )
            full = remote_root / relpath
            full.parent.mkdir(parents=True, exist_ok=True)
            if key in self._fail:
                cells_out.append({**row, "status": "error",
                                  "artifact_relpath": "",
                                  "error_reason": "stub-fail"})
                continue
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
        pair_rollups: dict = {}
        per_up: dict = {}
        for c in cells_out:
            ok = 1 if c["status"] in cp.SUCCESSFUL_CELL_STATUSES else 0
            pair_rollups.setdefault(c["prompt_pair_id"], {"ok": 0, "errors": 0, "total": 0})
            pair_rollups[c["prompt_pair_id"]]["ok"] += ok
            pair_rollups[c["prompt_pair_id"]]["errors"] += 1 - ok
            pair_rollups[c["prompt_pair_id"]]["total"] += 1
            per_up.setdefault(c["prompt_pair_id"], {})
            per_up[c["prompt_pair_id"]].setdefault(c["user_prompt_id"], {"ok": 0, "errors": 0, "total": 0})
            per_up[c["prompt_pair_id"]][c["user_prompt_id"]]["ok"] += ok
            per_up[c["prompt_pair_id"]][c["user_prompt_id"]]["errors"] += 1 - ok
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
            self._ctx.run_id, stage, self._ctx.round_id, attempt_id
        )
        full_mf = remote_root / relpath_mf
        full_mf.parent.mkdir(parents=True, exist_ok=True)
        full_mf.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return manifest


def _local_copy_transport(remote_root: Path):
    """Local copy-back transport used by the orchestrator under test."""

    def transport(*, remote_root, local_temp, cell_relpaths, manifest_relpath):
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
    return transport


# ---------------------------------------------------------------------------
# S05.05 / S05.06 / S05.10 — production must not call /stage/all
# ---------------------------------------------------------------------------


def test_dispatcher_dispatch_stage_all_hard_fails_in_production():
    d = StageDispatcher(post_v022=lambda s, p: {}, production_mode=True)
    with pytest.raises(V022ProductionViolation, match="/stage/all"):
        d.dispatch_stage_all()


def test_orchestrator_dispatches_only_v022_endpoints(tmp_path):
    ctx = _ctx(tmp_path)
    dispatcher = _FakeStageDispatcher(ctx)
    orch = V022Orchestrator(
        dispatcher=dispatcher,
        copy_back_transport=_local_copy_transport(Path(ctx.remote_artifact_root)),
    )
    orch.run_round(ctx)
    assert dispatcher.dispatched_endpoints == [
        "/v022/stage/gemma", "/v022/stage/flux", "/v022/stage/qwen"
    ]


def test_orchestrator_rejects_arbitrary_run_id(tmp_path):
    ctx = _ctx(tmp_path, run_id="bogus")
    orch = V022Orchestrator(dispatcher=_FakeStageDispatcher(ctx))
    with pytest.raises(V022ProductionViolation, match="canonical regex"):
        orch.run_round(ctx)


# ---------------------------------------------------------------------------
# S05.11 — mocked end-to-end
# ---------------------------------------------------------------------------


def test_mocked_end_to_end_writes_canonical_artifacts_and_manifests(tmp_path):
    ctx = _ctx(tmp_path)
    dispatcher = _FakeStageDispatcher(ctx)
    orch = V022Orchestrator(
        dispatcher=dispatcher,
        copy_back_transport=_local_copy_transport(Path(ctx.remote_artifact_root)),
    )
    summary = orch.run_round(ctx).summary
    # Every stage produced cells; merged manifests written; pointer retargeted.
    for stage in ("gemma", "flux", "qwen"):
        merged_path = ctx.artifact_root / cp.stage_manifest_merged_path(_RID, stage, 0)
        pointer_path = ctx.artifact_root / cp.stage_manifest_pointer_path(_RID, stage, 0)
        assert merged_path.is_file()
        assert pointer_path.exists()
        # Survivor JSONL written
        surv = ctx.artifact_root / cp.manifests_dir(_RID) / f"survivors_{stage}_round_0.jsonl"
        assert surv.is_file()
    # Cartesian dispatch_count for stage 0 = 2 pairs × 1 user_prompt × 2 samples = 4
    assert summary["stages"]["gemma"]["dispatch_count"] == 4
    assert summary["stages"]["gemma"]["remote_called"] is True
    # Downstream stages dispatch only on Gemma's surviving cells (all 4 OK).
    assert summary["stages"]["flux"]["dispatch_count"] == 4


# ---------------------------------------------------------------------------
# S05.12 — partial resume: D = S - L excludes locally-valid cells
# ---------------------------------------------------------------------------


def test_resume_dispatches_only_missing_cells(tmp_path):
    ctx = _ctx(tmp_path)
    dispatcher = _FakeStageDispatcher(ctx)
    orch = V022Orchestrator(
        dispatcher=dispatcher,
        copy_back_transport=_local_copy_transport(Path(ctx.remote_artifact_root)),
    )
    orch.run_round(ctx)
    # Delete one Gemma cell artifact + its per-attempt manifest reference.
    target = ctx.artifact_root / cp.cell_artifact_path(_RID, "gemma", 0, "PP1", "UP1", "S1")
    target.unlink()
    # Reset remote tree so a second dispatch produces fresh content for
    # only the missing cell. (The fake remote regenerates cells per-call.)
    shutil.rmtree(ctx.remote_artifact_root, ignore_errors=True)
    # Second round on the same run_id (resume).
    dispatcher2 = _FakeStageDispatcher(ctx)
    orch2 = V022Orchestrator(
        dispatcher=dispatcher2,
        copy_back_transport=_local_copy_transport(Path(ctx.remote_artifact_root)),
    )
    summary = orch2.run_round(ctx).summary
    # Gemma dispatch should only contain the deleted cell (1 of 4).
    assert summary["stages"]["gemma"]["dispatch_count"] == 1
    # Merged-view contains ALL 4 cells (3 carried_over + 1 fresh ok).
    merged = json.loads(
        (ctx.artifact_root / cp.stage_manifest_merged_path(_RID, "gemma", 0)).read_text()
    )
    statuses = sorted(c["status"] for c in merged["cells"])
    assert statuses.count("carried_over") == 3
    assert statuses.count("ok") == 1


# ---------------------------------------------------------------------------
# S05.07a — empty-dispatch short-circuit per stage (D = ∅)
# ---------------------------------------------------------------------------


def test_full_resume_with_no_changes_skips_all_remote_calls(tmp_path):
    ctx = _ctx(tmp_path)
    dispatcher = _FakeStageDispatcher(ctx)
    orch = V022Orchestrator(
        dispatcher=dispatcher,
        copy_back_transport=_local_copy_transport(Path(ctx.remote_artifact_root)),
    )
    orch.run_round(ctx)

    # Now resume — every cell on disk → D = ∅ for every stage.
    dispatcher2 = _FakeStageDispatcher(ctx)
    orch2 = V022Orchestrator(
        dispatcher=dispatcher2,
        copy_back_transport=_local_copy_transport(Path(ctx.remote_artifact_root)),
    )
    summary = orch2.run_round(ctx).summary
    assert dispatcher2.dispatched_endpoints == []  # no remote calls at all
    for stage in ("gemma", "flux", "qwen"):
        assert summary["stages"][stage]["dispatch_count"] == 0
        assert summary["stages"][stage]["remote_called"] is False


# ---------------------------------------------------------------------------
# S05.13 — --resume-from-run-id integration (drift gate runs before dispatch)
# ---------------------------------------------------------------------------


def test_resume_from_run_id_aborts_on_corpus_drift(tmp_path):
    ctx = _ctx(tmp_path)
    dispatcher = _FakeStageDispatcher(ctx)
    orch = V022Orchestrator(
        dispatcher=dispatcher,
        copy_back_transport=_local_copy_transport(Path(ctx.remote_artifact_root)),
    )
    orch.run_round(ctx)
    drifted = _ctx(
        tmp_path,
        corpus=_corpus(prompt_pair_corpus_hash="0" * 64),
    )
    # Same artifact_root, same run_id → drift gate applies.
    drifted.run_id = ctx.run_id
    with pytest.raises(ResumeDriftError, match="prompt_pair_corpus_hash"):
        gate_resume_from_run_id(ctx=drifted, resume_from_run_id=ctx.run_id)


def test_resume_from_run_id_passes_when_hashes_unchanged(tmp_path):
    ctx = _ctx(tmp_path)
    dispatcher = _FakeStageDispatcher(ctx)
    orch = V022Orchestrator(
        dispatcher=dispatcher,
        copy_back_transport=_local_copy_transport(Path(ctx.remote_artifact_root)),
    )
    orch.run_round(ctx)
    prior = gate_resume_from_run_id(ctx=ctx, resume_from_run_id=ctx.run_id)
    assert prior.exists()
