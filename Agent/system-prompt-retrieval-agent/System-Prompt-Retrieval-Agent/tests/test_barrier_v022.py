"""S03 tests for V0.2.2 stage barrier (remote + local artifact)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from system_prompt_retrieval_agent.barrier_v022 import (
    BarrierViolation,
    PartialModeThresholds,
    apply_survival_policy,
    run_local_artifact_barrier,
    run_remote_stage_barrier,
)
from system_prompt_retrieval_agent.remote._vendored import canonical_paths as cp


def _expected_match(rid="20260426T010000Z-deadbeef", stage="gemma", round_id=0):
    return {
        "schema_version": cp.SCHEMA_VERSION,
        "run_id": rid,
        "round_id": round_id,
        "stage": stage,
        "config_hash": "c" * 64,
        "user_prompt_corpus_id": "v0",
        "user_prompt_corpus_hash": "u" * 64,
        "prompt_pair_corpus_hash": "p" * 64,
        "sample_corpus_hash": "s" * 64,
    }


_FAKE_RUN_ID = "20260426T010000Z-deadbeef"


def _ok_cell(pid, upid, sid, *, relpath=None, size=64, sha=None, stage="gemma"):
    relpath = relpath or cp.cell_artifact_path(_FAKE_RUN_ID, stage, 0, pid, upid, sid)
    return {
        "prompt_pair_id": pid, "user_prompt_id": upid, "sample_id": sid,
        "status": "ok", "artifact_relpath": relpath,
        "artifact_size_bytes": size, "artifact_sha256": (sha or "a" * 64),
    }


def _manifest(cells, *, stage="gemma", lifecycle_state="disk_unloaded", lifecycle_mode="cold"):
    pair_rollups: dict = {}
    per_up: dict = {}
    for c in cells:
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
    return {
        **_expected_match(stage=stage),
        "attempt_id": "20260426T010100Z-bee0",
        "lifecycle_mode": lifecycle_mode,
        "lifecycle_state_after": lifecycle_state,
        "cells": list(cells),
        "pair_rollups": pair_rollups,
        "per_user_prompt": per_up,
    }


# ---------------------------------------------------------------------------
# Remote stage barrier
# ---------------------------------------------------------------------------


def test_remote_barrier_passes_on_well_formed_manifest():
    cells = [_ok_cell("PP1", "UP1", "S1"), _ok_cell("PP1", "UP1", "S2")]
    run_remote_stage_barrier(_manifest(cells), expected_match_fields=_expected_match())


def test_remote_barrier_rejects_match_field_drift():
    cells = [_ok_cell("PP1", "UP1", "S1")]
    expected = _expected_match()
    expected["config_hash"] = "0" * 64
    with pytest.raises(BarrierViolation, match="config_hash"):
        run_remote_stage_barrier(_manifest(cells), expected_match_fields=expected)


def test_remote_barrier_rejects_duplicate_cell_keys():
    c = _ok_cell("PP1", "UP1", "S1")
    cells = [c, dict(c)]  # duplicate
    with pytest.raises(BarrierViolation, match="duplicate"):
        run_remote_stage_barrier(_manifest(cells), expected_match_fields=_expected_match())


def test_remote_barrier_rejects_carried_over_status_from_remote():
    cells = [_ok_cell("PP1", "UP1", "S1"), {**_ok_cell("PP1", "UP1", "S2"), "status": "carried_over"}]
    with pytest.raises(BarrierViolation, match="carried_over"):
        run_remote_stage_barrier(_manifest(cells), expected_match_fields=_expected_match())


def test_remote_barrier_rejects_invalid_lifecycle_state():
    cells = [_ok_cell("PP1", "UP1", "S1")]
    bad = _manifest(cells, lifecycle_state="cpu_prefetched", lifecycle_mode="cold")
    with pytest.raises(ValueError):
        run_remote_stage_barrier(bad, expected_match_fields=_expected_match())


def test_remote_barrier_rejects_dispatch_set_mismatch():
    cells = [_ok_cell("PP1", "UP1", "S1")]
    expected_keys = [("PP1", "UP1", "S1"), ("PP1", "UP1", "S2")]
    with pytest.raises(BarrierViolation, match="dispatch set"):
        run_remote_stage_barrier(
            _manifest(cells),
            expected_match_fields=_expected_match(),
            expected_cell_keys=expected_keys,
        )


def test_remote_barrier_rejects_rollup_inconsistency():
    cells = [_ok_cell("PP1", "UP1", "S1")]
    m = _manifest(cells)
    m["pair_rollups"]["PP1"]["ok"] = 99  # tampered
    with pytest.raises(BarrierViolation, match="pair_rollups"):
        run_remote_stage_barrier(m, expected_match_fields=_expected_match())


def test_remote_barrier_rejects_empty_cells():
    m = _manifest([])
    with pytest.raises(BarrierViolation, match="empty"):
        run_remote_stage_barrier(m, expected_match_fields=_expected_match())


def test_remote_barrier_rejects_fabricated_success_without_size_or_sha():
    bad = {**_ok_cell("PP1", "UP1", "S1")}
    del bad["artifact_size_bytes"]
    with pytest.raises(ValueError):
        run_remote_stage_barrier(_manifest([bad]), expected_match_fields=_expected_match())


# ---------------------------------------------------------------------------
# Local artifact barrier
# ---------------------------------------------------------------------------


def _seed_cell(tmp_path: Path, pid, upid, sid, payload: bytes, stage="gemma"):
    relpath = cp.cell_artifact_path(_FAKE_RUN_ID, stage, 0, pid, upid, sid)
    full = tmp_path / relpath
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_bytes(payload)
    return relpath, len(payload), hashlib.sha256(payload).hexdigest()


def test_local_artifact_barrier_passes_on_consistent_disk(tmp_path):
    relpath, size, sha = _seed_cell(tmp_path, "PP1", "UP1", "S1", b"ok-bytes")
    cell = _ok_cell("PP1", "UP1", "S1", relpath=relpath, size=size, sha=sha)
    run_local_artifact_barrier(
        copied_manifest=_manifest([cell]),
        expected_match_fields=_expected_match(),
        artifact_root=tmp_path,
    )


def test_local_artifact_barrier_rejects_missing_artifact(tmp_path):
    relpath, size, sha = _seed_cell(tmp_path, "PP1", "UP1", "S1", b"ok-bytes")
    cell = _ok_cell("PP1", "UP1", "S1", relpath=relpath, size=size, sha=sha)
    (tmp_path / relpath).unlink()
    with pytest.raises(BarrierViolation, match="missing on-disk artifact"):
        run_local_artifact_barrier(
            copied_manifest=_manifest([cell]),
            expected_match_fields=_expected_match(),
            artifact_root=tmp_path,
        )


def test_local_artifact_barrier_rejects_byte_truncation(tmp_path):
    relpath, size, sha = _seed_cell(tmp_path, "PP1", "UP1", "S1", b"original-bytes")
    cell = _ok_cell("PP1", "UP1", "S1", relpath=relpath, size=size, sha=sha)
    (tmp_path / relpath).write_bytes(b"truncated")  # size + sha both diverge
    with pytest.raises(BarrierViolation, match="size"):
        run_local_artifact_barrier(
            copied_manifest=_manifest([cell]),
            expected_match_fields=_expected_match(),
            artifact_root=tmp_path,
        )


def test_local_artifact_barrier_lifecycle_mode_drift_does_not_invalidate(tmp_path):
    """S03.08 — informational fields are not part of match-field comparison."""
    relpath, size, sha = _seed_cell(tmp_path, "PP1", "UP1", "S1", b"ok-bytes")
    cell = _ok_cell("PP1", "UP1", "S1", relpath=relpath, size=size, sha=sha)
    manifest = _manifest([cell], lifecycle_state="disk_unloaded", lifecycle_mode="cold")
    # Even if the prior run used a different lifecycle_mode/state, the
    # local barrier must not reject — only match_fields matter.
    expected = _expected_match()  # same match-fields
    run_local_artifact_barrier(
        copied_manifest=manifest,
        expected_match_fields=expected,
        artifact_root=tmp_path,
    )


# ---------------------------------------------------------------------------
# Survival policy
# ---------------------------------------------------------------------------


def test_strict_mode_requires_full_coverage():
    cells = [_ok_cell("PP1", "UP1", "S1"), _ok_cell("PP1", "UP1", "S2"),
             _ok_cell("PP1", "UP2", "S1"), _ok_cell("PP1", "UP2", "S2")]
    out = apply_survival_policy(
        _manifest(cells),
        enabled_user_prompt_ids=("UP1", "UP2"),
        enabled_sample_ids=("S1", "S2"),
        allow_partial=False,
    )
    assert "PP1" in out.surviving_pairs


def test_strict_mode_fails_on_any_missing_cell():
    cells = [_ok_cell("PP1", "UP1", "S1")]  # missing UP1/S2 + UP2/* coverage
    out = apply_survival_policy(
        _manifest(cells),
        enabled_user_prompt_ids=("UP1", "UP2"),
        enabled_sample_ids=("S1", "S2"),
        allow_partial=False,
    )
    assert out.surviving_pairs == []
    assert out.failed_pairs[0]["failure_reason"] == "strict_mode_incomplete"


def test_partial_mode_below_min_user_prompts_fails():
    cells = [_ok_cell("PP1", "UP1", "S1")]
    out = apply_survival_policy(
        _manifest(cells),
        enabled_user_prompt_ids=("UP1", "UP2"),
        enabled_sample_ids=("S1", "S2"),
        allow_partial=True,
        thresholds=PartialModeThresholds(min_surviving_user_prompts=2),
    )
    assert out.failed_pairs[0]["failure_reason"] == "below_min_user_prompts"


def test_partial_mode_below_min_sample_ratio_fails():
    cells = [_ok_cell("PP1", "UP1", "S1"),
             {**_ok_cell("PP1", "UP1", "S2"), "status": "error",
              "artifact_size_bytes": 0, "artifact_sha256": "0" * 64}]
    out = apply_survival_policy(
        _manifest(cells),
        enabled_user_prompt_ids=("UP1",),
        enabled_sample_ids=("S1", "S2"),
        allow_partial=True,
        thresholds=PartialModeThresholds(min_sample_ratio=0.75),
    )
    assert out.failed_pairs[0]["failure_reason"] == "below_min_sample_ratio"


def test_partial_mode_required_categories_must_be_covered():
    cells = [_ok_cell("PP1", "UP1", "S1"), _ok_cell("PP1", "UP1", "S2")]
    cats = {("PP1", "UP1", "S1"): "dress", ("PP1", "UP1", "S2"): "dress"}
    out = apply_survival_policy(
        _manifest(cells),
        enabled_user_prompt_ids=("UP1",),
        enabled_sample_ids=("S1", "S2"),
        allow_partial=True,
        thresholds=PartialModeThresholds(required_categories=("dress", "upper")),
        cell_categories=cats,
    )
    assert out.failed_pairs[0]["failure_reason"].startswith("category_coverage_missing")
