"""S02 tests for V0.2.2 survivor + resume + merged-view manifests.

Covers:

* S02.01 — legacy ``build_survivor_manifest`` is hard-deprecated and refuses to run
* S02.02 — Gemma → FLUX survivor from merged-view cells (status ∈ ok/carried_over only)
* S02.03 — FLUX → Qwen survivor from merged-view cells
* S02.04 — survivor manifests contain only real 3-key rows
* S02.05 — duplicate rows rejected
* S02.06 — unknown / missing-key / placeholder rows rejected
* S02.07 — survivor rows are a strict subset of upstream successful cells
* S02.08 — ``D = S - L`` correctly skips locally-valid cells
* S02.09 — local L-validation rejects size / sha mismatch + missing files
* S02.10 — bundled rejection cases
* S02.11 — survivor generation actually runs for both stage transitions
* S02.12 — merged-view preservation under partial resume (zero already-valid dropped)
* S02.13 — merge ownership: remote may not emit carried_over; local agent owns merged path
* S02.14 — empty-dispatch (D = ∅): no remote call, merged manifest still written
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from system_prompt_retrieval_agent.remote._vendored import canonical_paths as cp
from system_prompt_retrieval_agent.survivor_resume_v022 import (
    CellKey,
    LocalArtifactView,
    SurvivorValidationError,
    build_gemma_base_survivor_set,
    build_merged_view,
    build_survivor_manifest_v022,
    compute_dispatch_set,
    compute_local_artifact_view,
    validate_cells,
    write_merged_view,
    write_resume_missing_cells_jsonl,
    write_survivor_jsonl,
)


_ENABLED_PP = ("PP1", "PP2")
_ENABLED_UP = ("UP1", "UP2")
_ENABLED_SS = ("S1", "S2", "S3")


def _ok_cell(pid="PP1", upid="UP1", sid="S1", **extra):
    base = {
        "prompt_pair_id": pid,
        "user_prompt_id": upid,
        "sample_id": sid,
        "status": "ok",
        "artifact_relpath": f"outputs/v02/RUN/gemma/round_0/{pid}/{upid}/{sid}/intermediate_prompt.txt",
        "artifact_size_bytes": 64,
        "artifact_sha256": "a" * 64,
    }
    base.update(extra)
    return base


def _merged_manifest(cells):
    return {
        "schema_version": cp.SCHEMA_VERSION,
        "run_id": "20260426T010000Z-deadbeef",
        "round_id": 0,
        "stage": "gemma",
        "config_hash": "c" * 64,
        "user_prompt_corpus_id": "v0",
        "user_prompt_corpus_hash": "u" * 64,
        "prompt_pair_corpus_hash": "p" * 64,
        "sample_corpus_hash": "s" * 64,
        "merged_view": True,
        "cells": list(cells),
    }


# ---------------------------------------------------------------------------
# S02.01 — legacy refusal
# ---------------------------------------------------------------------------


def test_legacy_build_survivor_manifest_refuses(tmp_path):
    from system_prompt_retrieval_agent.remote.resume import build_survivor_manifest
    from system_prompt_retrieval_agent.schemas import (
        StageManifest,
        PerPairManifest,
        PerUserPromptManifest,
    )

    mf = StageManifest(
        stage="gemma",
        run_id="run",
        round_id=0,
        pairs={
            "PP1": PerPairManifest(
                prompt_pair_id="PP1",
                ok=2, errors=0, total=2,
                per_user_prompt={
                    "UP1": PerUserPromptManifest(ok=2, errors=0, total=2)
                },
            )
        },
        surviving_pairs=["PP1"],
        failed_pairs=[],
        ok=2, errors=0, total=2,
    )
    os.environ.pop("SPRA_LEGACY_SURVIVOR_MANIFEST_OK", None)
    with pytest.raises(RuntimeError, match="hard-deprecated"):
        build_survivor_manifest(mf, local_manifests_root=tmp_path)


# ---------------------------------------------------------------------------
# S02.02 / S02.03 / S02.04 / S02.07 — survivor extraction
# ---------------------------------------------------------------------------


def test_survivor_extracts_only_successful_cells():
    cells = [
        _ok_cell(pid="PP1", upid="UP1", sid="S1"),
        _ok_cell(pid="PP1", upid="UP1", sid="S2", status="error",
                 **{"artifact_relpath": "", "artifact_size_bytes": 0,
                    "artifact_sha256": "0" * 64}),
        _ok_cell(pid="PP2", upid="UP2", sid="S3", status="carried_over"),
    ]
    survivors = build_survivor_manifest_v022(
        _merged_manifest(cells),
        enabled_prompt_pair_ids=_ENABLED_PP,
        enabled_user_prompt_ids=_ENABLED_UP,
        enabled_sample_ids=_ENABLED_SS,
    )
    keys = {s.as_tuple() for s in survivors}
    assert keys == {("PP1", "UP1", "S1"), ("PP2", "UP2", "S3")}
    for s in survivors:
        row = s.as_row()
        assert set(row.keys()) == {"prompt_pair_id", "user_prompt_id", "sample_id"}


# ---------------------------------------------------------------------------
# S02.05 / S02.06 / S02.10 — validation rejections
# ---------------------------------------------------------------------------


def test_duplicate_cells_rejected():
    cells = [CellKey("PP1", "UP1", "S1"), CellKey("PP1", "UP1", "S1")]
    with pytest.raises(SurvivorValidationError, match="duplicate"):
        validate_cells(
            cells,
            enabled_prompt_pair_ids=_ENABLED_PP,
            enabled_user_prompt_ids=_ENABLED_UP,
            enabled_sample_ids=_ENABLED_SS,
        )


@pytest.mark.parametrize("placeholder", ["__all__", "*", ""])
def test_placeholder_sample_id_rejected(placeholder):
    cells = [CellKey("PP1", "UP1", placeholder)]
    with pytest.raises(SurvivorValidationError, match="placeholder"):
        validate_cells(
            cells,
            enabled_prompt_pair_ids=_ENABLED_PP,
            enabled_user_prompt_ids=_ENABLED_UP,
            enabled_sample_ids=_ENABLED_SS,
        )


@pytest.mark.parametrize(
    "field,bad",
    [
        ("prompt_pair_id", "PP_BOGUS"),
        ("user_prompt_id", "UP_BOGUS"),
        ("sample_id", "S_BOGUS"),
    ],
)
def test_unknown_id_rejected(field, bad):
    kwargs = {"prompt_pair_id": "PP1", "user_prompt_id": "UP1", "sample_id": "S1"}
    kwargs[field] = bad
    cells = [CellKey(**kwargs)]
    with pytest.raises(SurvivorValidationError, match=f"unknown {field}"):
        validate_cells(
            cells,
            enabled_prompt_pair_ids=_ENABLED_PP,
            enabled_user_prompt_ids=_ENABLED_UP,
            enabled_sample_ids=_ENABLED_SS,
        )


def test_missing_key_rejected_during_extraction():
    cells = [{"prompt_pair_id": "PP1", "user_prompt_id": "UP1", "status": "ok",
              "artifact_relpath": "x", "artifact_size_bytes": 1,
              "artifact_sha256": "a" * 64}]  # no sample_id
    with pytest.raises(SurvivorValidationError, match="sample_id"):
        build_survivor_manifest_v022(
            _merged_manifest(cells),
            enabled_prompt_pair_ids=_ENABLED_PP,
            enabled_user_prompt_ids=_ENABLED_UP,
            enabled_sample_ids=_ENABLED_SS,
        )


# ---------------------------------------------------------------------------
# Gemma base set (S05.02 / S00.16a path)
# ---------------------------------------------------------------------------


def test_gemma_base_survivor_set_full_cartesian():
    cells = build_gemma_base_survivor_set(
        enabled_prompt_pair_ids=_ENABLED_PP,
        enabled_user_prompt_ids=_ENABLED_UP,
        configured_sample_ids=_ENABLED_SS,
    )
    assert len(cells) == 2 * 2 * 3
    assert {c.as_tuple() for c in cells[:3]} == {
        ("PP1", "UP1", s) for s in _ENABLED_SS
    }


def test_gemma_base_set_empty_aborts():
    with pytest.raises(SurvivorValidationError, match="S_gemma is empty"):
        build_gemma_base_survivor_set(
            enabled_prompt_pair_ids=_ENABLED_PP,
            enabled_user_prompt_ids=(),
            configured_sample_ids=_ENABLED_SS,
        )


# ---------------------------------------------------------------------------
# S02.08 / S02.09 — D = S - L + L-validation
# ---------------------------------------------------------------------------


def _seed_local_artifact(
    artifact_root: Path,
    *,
    run_id: str,
    stage: str,
    round_id: int,
    pair: str,
    upid: str,
    sid: str,
    payload: bytes,
) -> tuple[str, int, str]:
    relpath = cp.cell_artifact_path(run_id, stage, round_id, pair, upid, sid)
    full = artifact_root / relpath
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_bytes(payload)
    return relpath, len(payload), hashlib.sha256(payload).hexdigest()


def _seed_attempt_manifest(
    artifact_root: Path,
    *,
    run_id: str,
    stage: str,
    round_id: int,
    attempt_id: str,
    cells: list[dict],
    match_overrides: dict | None = None,
):
    base = {
        "schema_version": cp.SCHEMA_VERSION,
        "run_id": run_id,
        "round_id": round_id,
        "stage": stage,
        "config_hash": "c" * 64,
        "user_prompt_corpus_id": "v0",
        "user_prompt_corpus_hash": "u" * 64,
        "prompt_pair_corpus_hash": "p" * 64,
        "sample_corpus_hash": "s" * 64,
        "attempt_id": attempt_id,
        "lifecycle_mode": "cold",
        "lifecycle_state_after": "disk_unloaded",
        "cells": cells,
    }
    if match_overrides:
        base.update(match_overrides)
    relpath = cp.stage_manifest_attempt_path(run_id, stage, round_id, attempt_id)
    target = artifact_root / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(base, indent=2, sort_keys=True))
    return base


@pytest.fixture
def seeded_local(tmp_path: Path):
    rid = "20260426T010000Z-deadbeef"
    aid = "20260426T010005Z-cafe"
    relpath, size, sha = _seed_local_artifact(
        tmp_path, run_id=rid, stage="gemma", round_id=0,
        pair="PP1", upid="UP1", sid="S1", payload=b"prompt-1"
    )
    _seed_attempt_manifest(
        tmp_path, run_id=rid, stage="gemma", round_id=0, attempt_id=aid,
        cells=[
            {
                "prompt_pair_id": "PP1", "user_prompt_id": "UP1", "sample_id": "S1",
                "status": "ok", "artifact_relpath": relpath,
                "artifact_size_bytes": size, "artifact_sha256": sha,
            }
        ],
    )
    return tmp_path, rid


def _expected_match(rid):
    return {
        "schema_version": cp.SCHEMA_VERSION,
        "run_id": rid,
        "round_id": 0,
        "stage": "gemma",
        "config_hash": "c" * 64,
        "user_prompt_corpus_id": "v0",
        "user_prompt_corpus_hash": "u" * 64,
        "prompt_pair_corpus_hash": "p" * 64,
        "sample_corpus_hash": "s" * 64,
    }


def test_compute_dispatch_set_excludes_locally_valid(seeded_local):
    artifact_root, rid = seeded_local
    survivor = [
        CellKey("PP1", "UP1", "S1"),
        CellKey("PP1", "UP1", "S2"),
        CellKey("PP2", "UP2", "S3"),
    ]
    L = compute_local_artifact_view(
        run_id=rid, stage="gemma", round_id=0,
        artifact_root=artifact_root,
        expected_match_fields=_expected_match(rid),
        survivor_set=survivor,
    )
    D = compute_dispatch_set(survivor, L)
    assert {c.as_tuple() for c in D} == {("PP1", "UP1", "S2"), ("PP2", "UP2", "S3")}


def test_l_validation_rejects_sha_mismatch(seeded_local, tmp_path):
    artifact_root, rid = seeded_local
    # Tamper the on-disk artifact AFTER manifest seed → sha mismatch
    relpath = cp.cell_artifact_path(rid, "gemma", 0, "PP1", "UP1", "S1")
    (artifact_root / relpath).write_bytes(b"tampered")
    L = compute_local_artifact_view(
        run_id=rid, stage="gemma", round_id=0,
        artifact_root=artifact_root,
        expected_match_fields=_expected_match(rid),
        survivor_set=[CellKey("PP1", "UP1", "S1")],
    )
    assert L.cell_ids() == set()


def test_l_validation_rejects_match_field_drift(seeded_local):
    artifact_root, rid = seeded_local
    expected = _expected_match(rid)
    expected["config_hash"] = "0" * 64  # drift
    L = compute_local_artifact_view(
        run_id=rid, stage="gemma", round_id=0,
        artifact_root=artifact_root,
        expected_match_fields=expected,
        survivor_set=[CellKey("PP1", "UP1", "S1")],
    )
    assert L.cell_ids() == set()


def test_l_validation_rejects_missing_artifact(seeded_local):
    artifact_root, rid = seeded_local
    relpath = cp.cell_artifact_path(rid, "gemma", 0, "PP1", "UP1", "S1")
    (artifact_root / relpath).unlink()
    L = compute_local_artifact_view(
        run_id=rid, stage="gemma", round_id=0,
        artifact_root=artifact_root,
        expected_match_fields=_expected_match(rid),
        survivor_set=[CellKey("PP1", "UP1", "S1")],
    )
    assert L.cell_ids() == set()


# ---------------------------------------------------------------------------
# S02.11 / S02.12 — survivor pipeline + merged-view preservation
# ---------------------------------------------------------------------------


def test_partial_resume_drops_zero_locally_valid_cells(seeded_local):
    artifact_root, rid = seeded_local
    survivor = [
        CellKey("PP1", "UP1", "S1"),
        CellKey("PP1", "UP1", "S2"),
    ]
    L = compute_local_artifact_view(
        run_id=rid, stage="gemma", round_id=0,
        artifact_root=artifact_root,
        expected_match_fields=_expected_match(rid),
        survivor_set=survivor,
    )
    D = compute_dispatch_set(survivor, L)
    # Simulate the remote response only for D
    remote_manifest = {
        **_expected_match(rid),
        "attempt_id": "20260426T010100Z-bee0",
        "cells": [
            {
                "prompt_pair_id": "PP1", "user_prompt_id": "UP1", "sample_id": "S2",
                "status": "ok",
                "artifact_relpath": cp.cell_artifact_path(rid, "gemma", 0, "PP1", "UP1", "S2"),
                "artifact_size_bytes": 9, "artifact_sha256": "b" * 64,
            }
        ],
    }
    merged = build_merged_view(
        survivor_set=survivor, local_view=L,
        current_remote_manifest=remote_manifest,
        current_attempt_id="20260426T010100Z-bee0",
    )
    keys = {(c["prompt_pair_id"], c["user_prompt_id"], c["sample_id"]) for c in merged["cells"]}
    assert keys == {("PP1", "UP1", "S1"), ("PP1", "UP1", "S2")}
    by_key = {(c["prompt_pair_id"], c["user_prompt_id"], c["sample_id"]): c
              for c in merged["cells"]}
    assert by_key[("PP1", "UP1", "S1")]["status"] == "carried_over"
    assert by_key[("PP1", "UP1", "S2")]["status"] == "ok"


# ---------------------------------------------------------------------------
# S02.13 — merge ownership (remote may not emit carried_over)
# ---------------------------------------------------------------------------


def test_remote_carried_over_status_rejected_by_merge():
    survivor = [CellKey("PP1", "UP1", "S1")]
    bad_remote = {
        **_expected_match("20260426T010000Z-deadbeef"),
        "attempt_id": "20260426T010100Z-bee0",
        "cells": [
            {"prompt_pair_id": "PP1", "user_prompt_id": "UP1", "sample_id": "S1",
             "status": "carried_over",
             "artifact_relpath": "x",
             "artifact_size_bytes": 1, "artifact_sha256": "a" * 64},
        ],
    }
    with pytest.raises(SurvivorValidationError, match="carried_over"):
        build_merged_view(
            survivor_set=survivor,
            local_view=LocalArtifactView(),
            current_remote_manifest=bad_remote,
            current_attempt_id="20260426T010100Z-bee0",
        )


def test_merged_view_path_owned_by_local_agent(seeded_local):
    artifact_root, rid = seeded_local
    survivor = [CellKey("PP1", "UP1", "S1")]
    L = compute_local_artifact_view(
        run_id=rid, stage="gemma", round_id=0,
        artifact_root=artifact_root,
        expected_match_fields=_expected_match(rid),
        survivor_set=survivor,
    )
    merged = build_merged_view(
        survivor_set=survivor, local_view=L,
        current_remote_manifest=None, current_attempt_id=None,
    )
    target = write_merged_view(
        artifact_root=artifact_root, run_id=rid, stage="gemma",
        round_id=0, merged_payload=merged,
    )
    expected_path = artifact_root / cp.stage_manifest_merged_path(rid, "gemma", 0)
    assert target == expected_path
    assert target.is_file()
    assert json.loads(target.read_text())["cells"][0]["status"] == "carried_over"


# ---------------------------------------------------------------------------
# S02.14 — empty-dispatch short-circuit
# ---------------------------------------------------------------------------


def test_empty_dispatch_yields_merged_view_from_l_only(seeded_local):
    artifact_root, rid = seeded_local
    survivor = [CellKey("PP1", "UP1", "S1")]
    L = compute_local_artifact_view(
        run_id=rid, stage="gemma", round_id=0,
        artifact_root=artifact_root,
        expected_match_fields=_expected_match(rid),
        survivor_set=survivor,
    )
    D = compute_dispatch_set(survivor, L)
    assert D == []  # empty dispatch
    # No remote call → current_remote_manifest=None
    merged = build_merged_view(
        survivor_set=survivor, local_view=L,
        current_remote_manifest=None, current_attempt_id=None,
    )
    assert len(merged["cells"]) == 1
    assert merged["cells"][0]["status"] == "carried_over"


# ---------------------------------------------------------------------------
# JSONL writers
# ---------------------------------------------------------------------------


def test_survivor_jsonl_emits_three_key_rows(tmp_path):
    cells = [CellKey("PP1", "UP1", "S1"), CellKey("PP2", "UP2", "S3")]
    target = tmp_path / "surv.jsonl"
    write_survivor_jsonl(cells, target)
    rows = [json.loads(line) for line in target.read_text().splitlines()]
    assert all(set(r.keys()) == {"prompt_pair_id", "user_prompt_id", "sample_id"} for r in rows)
    assert len(rows) == 2


def test_resume_missing_cells_jsonl_uses_same_format(tmp_path):
    cells = [CellKey("PP1", "UP1", "S1")]
    target = tmp_path / "resume.jsonl"
    write_resume_missing_cells_jsonl(cells, target)
    assert json.loads(target.read_text().strip()) == {
        "prompt_pair_id": "PP1", "user_prompt_id": "UP1", "sample_id": "S1"
    }
