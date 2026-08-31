"""S04 tests for V0.2.2 cell-scoped copy-back + atomic promotion."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from system_prompt_retrieval_agent.copy_back_v022 import (
    CopyBackError,
    CopyBackPlan,
    PromotionResult,
    assert_no_destructive_flags,
    build_missing_manifest,
    copy_back_dispatch_cells,
    finalize_pointer_after_merge,
)
from system_prompt_retrieval_agent.remote._vendored import canonical_paths as cp


_RID = "20260426T010000Z-deadbeef"
_AID = "20260426T010005Z-cafe"


def _expected_match(stage="gemma", round_id=0):
    return {
        "schema_version": cp.SCHEMA_VERSION,
        "run_id": _RID,
        "round_id": round_id,
        "stage": stage,
        "config_hash": "c" * 64,
        "user_prompt_corpus_id": "v0",
        "user_prompt_corpus_hash": "u" * 64,
        "prompt_pair_corpus_hash": "p" * 64,
        "sample_corpus_hash": "s" * 64,
    }


def _make_remote_tree(remote_root: Path, dispatch_cells, *, payloads):
    """Create a fake 'remote' source tree the transport will copy from."""
    cell_records = []
    for (pid, upid, sid) in dispatch_cells:
        relpath = cp.cell_artifact_path(_RID, "gemma", 0, pid, upid, sid)
        full = remote_root / relpath
        full.parent.mkdir(parents=True, exist_ok=True)
        payload = payloads.get((pid, upid, sid), b"prompt-bytes")
        full.write_bytes(payload)
        cell_records.append({
            "prompt_pair_id": pid, "user_prompt_id": upid, "sample_id": sid,
            "status": "ok", "artifact_relpath": relpath,
            "artifact_size_bytes": len(payload),
            "artifact_sha256": hashlib.sha256(payload).hexdigest(),
        })
    manifest_relpath = cp.stage_manifest_attempt_path(_RID, "gemma", 0, _AID)
    manifest_full = remote_root / manifest_relpath
    manifest_full.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **_expected_match(),
        "attempt_id": _AID,
        "lifecycle_mode": "cold",
        "lifecycle_state_after": "disk_unloaded",
        "cells": cell_records,
        "pair_rollups": {pid: {"ok": 1, "errors": 0, "total": 1}
                         for (pid, _, _) in dispatch_cells},
        "per_user_prompt": {pid: {upid: {"ok": 1, "errors": 0, "total": 1}}
                            for (pid, upid, _) in dispatch_cells},
    }
    manifest_full.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return manifest_relpath, payload, cell_records


def _make_transport(remote_root: Path):
    """Local transport that copies from a fake 'remote' tree using shutil."""
    import shutil as _sh

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
            _sh.copy2(src, dst)
    return transport


# ---------------------------------------------------------------------------
# S04.01 / S04.02 / S04.03 / S04.04 — happy path
# ---------------------------------------------------------------------------


def test_cell_scoped_copy_back_promotes_only_dispatch_cells(tmp_path):
    remote = tmp_path / "remote_root"
    local = tmp_path / "local_root"
    dispatch = [("PP1", "UP1", "S1"), ("PP1", "UP1", "S2")]
    _make_remote_tree(remote, dispatch, payloads={
        ("PP1", "UP1", "S1"): b"prompt-1",
        ("PP1", "UP1", "S2"): b"prompt-2",
    })
    plan = CopyBackPlan(
        run_id=_RID, stage="gemma", round_id=0, attempt_id=_AID,
        expected_match_fields=_expected_match(),
        dispatch_cells=dispatch,
        artifact_root=local,
        remote_artifact_root=str(remote),
        remote_manifest_root=str(remote),
    )
    result = copy_back_dispatch_cells(plan, transport=_make_transport(remote))
    assert sorted(result.promoted_cells) == sorted(dispatch)
    for (pid, upid, sid) in dispatch:
        live = local / cp.cell_artifact_path(_RID, "gemma", 0, pid, upid, sid)
        assert live.is_file()
    assert result.promoted_manifest_path.is_file()
    # Pointer must NOT yet be written by copy-back (S04.04: only after merge).
    pointer = local / cp.stage_manifest_pointer_path(_RID, "gemma", 0)
    assert not pointer.exists()


def test_finalize_pointer_after_merge_retargets_atomically(tmp_path):
    remote = tmp_path / "remote_root"
    local = tmp_path / "local_root"
    dispatch = [("PP1", "UP1", "S1")]
    _make_remote_tree(remote, dispatch, payloads={("PP1", "UP1", "S1"): b"x"})
    plan = CopyBackPlan(
        run_id=_RID, stage="gemma", round_id=0, attempt_id=_AID,
        expected_match_fields=_expected_match(),
        dispatch_cells=dispatch, artifact_root=local,
        remote_artifact_root=str(remote), remote_manifest_root=str(remote),
    )
    result = copy_back_dispatch_cells(plan, transport=_make_transport(remote))
    finalize_pointer_after_merge(
        attempt_path=result.pending_attempt_path,
        pointer_path=result.pending_pointer_path,
    )
    assert result.pending_pointer_path.exists()
    if result.pending_pointer_path.is_symlink():
        assert os.readlink(result.pending_pointer_path) == result.pending_attempt_path.name
    else:
        assert result.pending_pointer_path.read_bytes() == result.pending_attempt_path.read_bytes()


# ---------------------------------------------------------------------------
# S04.03 / S04.05 / S04.05a — verification failures abort
# ---------------------------------------------------------------------------


def test_match_field_drift_in_temp_manifest_aborts(tmp_path):
    remote = tmp_path / "remote_root"
    local = tmp_path / "local_root"
    dispatch = [("PP1", "UP1", "S1")]
    _make_remote_tree(remote, dispatch, payloads={("PP1", "UP1", "S1"): b"x"})
    expected = _expected_match()
    expected["config_hash"] = "0" * 64  # drift
    plan = CopyBackPlan(
        run_id=_RID, stage="gemma", round_id=0, attempt_id=_AID,
        expected_match_fields=expected, dispatch_cells=dispatch,
        artifact_root=local, remote_artifact_root=str(remote),
        remote_manifest_root=str(remote),
    )
    with pytest.raises(CopyBackError, match="match-field"):
        copy_back_dispatch_cells(plan, transport=_make_transport(remote))
    # No live artifacts written
    assert not (local / "outputs").exists() or not any((local / "outputs").rglob("*"))


def test_truncated_transfer_detected_via_sha256(tmp_path):
    remote = tmp_path / "remote_root"
    local = tmp_path / "local_root"
    dispatch = [("PP1", "UP1", "S1")]
    _make_remote_tree(remote, dispatch, payloads={("PP1", "UP1", "S1"): b"original"})
    relpath = cp.cell_artifact_path(_RID, "gemma", 0, "PP1", "UP1", "S1")
    # Corrupt the source AFTER manifest computed → sha mismatch on receiver
    (remote / relpath).write_bytes(b"truncated")
    plan = CopyBackPlan(
        run_id=_RID, stage="gemma", round_id=0, attempt_id=_AID,
        expected_match_fields=_expected_match(), dispatch_cells=dispatch,
        artifact_root=local, remote_artifact_root=str(remote),
        remote_manifest_root=str(remote),
    )
    with pytest.raises(CopyBackError, match="truncated transfer"):
        copy_back_dispatch_cells(plan, transport=_make_transport(remote))


def test_dispatch_set_mismatch_aborts(tmp_path):
    remote = tmp_path / "remote_root"
    local = tmp_path / "local_root"
    actual = [("PP1", "UP1", "S1")]
    _make_remote_tree(remote, actual, payloads={("PP1", "UP1", "S1"): b"x"})
    declared = [("PP1", "UP1", "S1"), ("PP1", "UP1", "S2")]
    plan = CopyBackPlan(
        run_id=_RID, stage="gemma", round_id=0, attempt_id=_AID,
        expected_match_fields=_expected_match(),
        dispatch_cells=declared,
        artifact_root=local, remote_artifact_root=str(remote),
        remote_manifest_root=str(remote),
    )
    with pytest.raises(CopyBackError, match="dispatch set"):
        copy_back_dispatch_cells(plan, transport=_make_transport(remote))


# ---------------------------------------------------------------------------
# S04.04 — non-overwriting per-attempt manifest path
# ---------------------------------------------------------------------------


def test_per_attempt_manifest_is_non_overwriting(tmp_path):
    remote = tmp_path / "remote_root"
    local = tmp_path / "local_root"
    dispatch = [("PP1", "UP1", "S1")]
    _make_remote_tree(remote, dispatch, payloads={("PP1", "UP1", "S1"): b"x"})
    plan = CopyBackPlan(
        run_id=_RID, stage="gemma", round_id=0, attempt_id=_AID,
        expected_match_fields=_expected_match(),
        dispatch_cells=dispatch, artifact_root=local,
        remote_artifact_root=str(remote), remote_manifest_root=str(remote),
    )
    copy_back_dispatch_cells(plan, transport=_make_transport(remote))
    # Re-running with same attempt_id must abort — would overwrite.
    with pytest.raises(CopyBackError, match="already occupied"):
        copy_back_dispatch_cells(plan, transport=_make_transport(remote))


# ---------------------------------------------------------------------------
# S04.06 — destructive flag guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["--delete", "--delete-after", "--delete-before",
                                  "--delete-during", "--delete-excluded"])
def test_destructive_rsync_flag_rejected(flag):
    with pytest.raises(CopyBackError, match="forbidden rsync flag"):
        assert_no_destructive_flags(["rsync", "-av", flag, "src/", "dst/"])


def test_safe_rsync_flags_accepted():
    assert_no_destructive_flags(["rsync", "-av", "src/", "dst/"])
    assert_no_destructive_flags(["rsync", "-av", "--files-from", "list", "src/", "dst/"])


# ---------------------------------------------------------------------------
# S04.07 / S04.08 — build_missing_manifest after a successful copy-back
# ---------------------------------------------------------------------------


def test_build_missing_manifest_after_copy_back_drops_locally_valid(tmp_path):
    remote = tmp_path / "remote_root"
    local = tmp_path / "local_root"
    dispatch = [("PP1", "UP1", "S1"), ("PP1", "UP1", "S2")]
    _make_remote_tree(remote, dispatch, payloads={
        ("PP1", "UP1", "S1"): b"a", ("PP1", "UP1", "S2"): b"bb",
    })
    plan = CopyBackPlan(
        run_id=_RID, stage="gemma", round_id=0, attempt_id=_AID,
        expected_match_fields=_expected_match(),
        dispatch_cells=dispatch, artifact_root=local,
        remote_artifact_root=str(remote), remote_manifest_root=str(remote),
    )
    copy_back_dispatch_cells(plan, transport=_make_transport(remote))

    survivor = [("PP1", "UP1", "S1"), ("PP1", "UP1", "S2"), ("PP1", "UP1", "S3")]
    D2 = build_missing_manifest(
        run_id=_RID, stage="gemma", round_id=0,
        artifact_root=local, expected_match_fields=_expected_match(),
        survivor_cells=survivor,
    )
    assert D2 == [("PP1", "UP1", "S3")]


def test_build_missing_manifest_under_corpus_drift_returns_full_survivor(tmp_path):
    remote = tmp_path / "remote_root"
    local = tmp_path / "local_root"
    dispatch = [("PP1", "UP1", "S1")]
    _make_remote_tree(remote, dispatch, payloads={("PP1", "UP1", "S1"): b"x"})
    plan = CopyBackPlan(
        run_id=_RID, stage="gemma", round_id=0, attempt_id=_AID,
        expected_match_fields=_expected_match(),
        dispatch_cells=dispatch, artifact_root=local,
        remote_artifact_root=str(remote), remote_manifest_root=str(remote),
    )
    copy_back_dispatch_cells(plan, transport=_make_transport(remote))

    drifted = _expected_match()
    drifted["prompt_pair_corpus_hash"] = "0" * 64  # corpus drift
    D2 = build_missing_manifest(
        run_id=_RID, stage="gemma", round_id=0,
        artifact_root=local, expected_match_fields=drifted,
        survivor_cells=dispatch,
    )
    # Drifted match-fields → L is empty → D2 == survivor
    assert D2 == dispatch


# ---------------------------------------------------------------------------
# Cells in L are never re-rsynced or overwritten by a D-only attempt
# ---------------------------------------------------------------------------


def test_l_cells_not_overwritten_by_subsequent_d_only_attempt(tmp_path):
    remote = tmp_path / "remote_root"
    local = tmp_path / "local_root"
    dispatch_first = [("PP1", "UP1", "S1")]
    _make_remote_tree(remote, dispatch_first, payloads={("PP1", "UP1", "S1"): b"first"})
    plan = CopyBackPlan(
        run_id=_RID, stage="gemma", round_id=0, attempt_id=_AID,
        expected_match_fields=_expected_match(),
        dispatch_cells=dispatch_first, artifact_root=local,
        remote_artifact_root=str(remote), remote_manifest_root=str(remote),
    )
    copy_back_dispatch_cells(plan, transport=_make_transport(remote))
    s1_path = local / cp.cell_artifact_path(_RID, "gemma", 0, "PP1", "UP1", "S1")
    s1_first = s1_path.read_bytes()

    # Second attempt with new attempt_id, dispatching only S2 (D = [S2]).
    aid2 = "20260426T010100Z-bee0"
    dispatch_second = [("PP1", "UP1", "S2")]
    _make_remote_tree(remote, dispatch_second, payloads={("PP1", "UP1", "S2"): b"second"})
    # The fake remote tree was rewritten — re-write the S2 attempt manifest
    # under aid2 so transport can pick it up.
    relpath2 = cp.cell_artifact_path(_RID, "gemma", 0, "PP1", "UP1", "S2")
    (remote / relpath2).write_bytes(b"second")
    payload = {
        **_expected_match(),
        "attempt_id": aid2,
        "lifecycle_mode": "cold", "lifecycle_state_after": "disk_unloaded",
        "cells": [
            {"prompt_pair_id": "PP1", "user_prompt_id": "UP1", "sample_id": "S2",
             "status": "ok", "artifact_relpath": relpath2,
             "artifact_size_bytes": 6,
             "artifact_sha256": hashlib.sha256(b"second").hexdigest()}
        ],
        "pair_rollups": {"PP1": {"ok": 1, "errors": 0, "total": 1}},
        "per_user_prompt": {"PP1": {"UP1": {"ok": 1, "errors": 0, "total": 1}}},
    }
    aid2_relpath = cp.stage_manifest_attempt_path(_RID, "gemma", 0, aid2)
    (remote / aid2_relpath).parent.mkdir(parents=True, exist_ok=True)
    (remote / aid2_relpath).write_text(json.dumps(payload, sort_keys=True))

    plan2 = CopyBackPlan(
        run_id=_RID, stage="gemma", round_id=0, attempt_id=aid2,
        expected_match_fields=_expected_match(),
        dispatch_cells=dispatch_second, artifact_root=local,
        remote_artifact_root=str(remote), remote_manifest_root=str(remote),
    )
    copy_back_dispatch_cells(plan2, transport=_make_transport(remote))

    # S1 must be untouched; S2 newly promoted; both per-attempt manifests preserved.
    assert s1_path.read_bytes() == s1_first
    s2_path = local / relpath2
    assert s2_path.read_bytes() == b"second"
    aid_relpath = cp.stage_manifest_attempt_path(_RID, "gemma", 0, _AID)
    aid2_full = local / aid2_relpath
    aid_full = local / aid_relpath
    assert aid_full.is_file() and aid2_full.is_file()


# ---------------------------------------------------------------------------
# F2-strict — status-aware copy-back (strict + partial)
# ---------------------------------------------------------------------------


def _make_remote_tree_with_failure(
    remote_root: Path,
    dispatch_cells,
    *,
    failed_keys,
    failed_status="error",
):
    """Variant of _make_remote_tree where some cells are status='error'
    and have NO artifact file on the remote tree (Gemma adapter behavior:
    failed cells emit a manifest row but skip intermediate_prompt.txt).
    """
    cell_records = []
    for (pid, upid, sid) in dispatch_cells:
        relpath = cp.cell_artifact_path(_RID, "gemma", 0, pid, upid, sid)
        if (pid, upid, sid) in failed_keys:
            cell_records.append({
                "prompt_pair_id": pid, "user_prompt_id": upid, "sample_id": sid,
                "status": failed_status, "artifact_relpath": None,
                "artifact_size_bytes": None, "artifact_sha256": None,
                "error_message": "fake remote error",
            })
            continue
        full = remote_root / relpath
        full.parent.mkdir(parents=True, exist_ok=True)
        payload = b"prompt-bytes"
        full.write_bytes(payload)
        cell_records.append({
            "prompt_pair_id": pid, "user_prompt_id": upid, "sample_id": sid,
            "status": "ok", "artifact_relpath": relpath,
            "artifact_size_bytes": len(payload),
            "artifact_sha256": hashlib.sha256(payload).hexdigest(),
        })
    manifest_relpath = cp.stage_manifest_attempt_path(_RID, "gemma", 0, _AID)
    manifest_full = remote_root / manifest_relpath
    manifest_full.parent.mkdir(parents=True, exist_ok=True)
    pair_rollups: dict = {}
    per_up: dict = {}
    for c in cell_records:
        pid = c["prompt_pair_id"]; upid = c["user_prompt_id"]
        ok = 1 if c["status"] in cp.SUCCESSFUL_CELL_STATUSES else 0
        err = 1 - ok
        pr = pair_rollups.setdefault(pid, {"ok": 0, "errors": 0, "total": 0})
        pr["ok"] += ok; pr["errors"] += err; pr["total"] += 1
        u = per_up.setdefault(pid, {}).setdefault(upid, {"ok": 0, "errors": 0, "total": 0})
        u["ok"] += ok; u["errors"] += err; u["total"] += 1
    payload = {
        **_expected_match(),
        "attempt_id": _AID,
        "lifecycle_mode": "cold",
        "lifecycle_state_after": "disk_unloaded",
        "cells": cell_records,
        "pair_rollups": pair_rollups,
        "per_user_prompt": per_up,
    }
    manifest_full.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return manifest_relpath, payload


def test_strict_mode_aborts_before_cell_rsync_when_manifest_has_failed_cell(tmp_path):
    """F2-strict: if the remote per-attempt manifest reports any non-success
    cell and ``allow_partial=False`` (default), copy-back must raise
    CopyBackError BEFORE attempting to pull cell artifacts. This proves the
    rsync rc=23 / link_stat failure mode is no longer reachable for
    legitimate per-cell failures."""
    remote = tmp_path / "remote_root"
    local = tmp_path / "local_root"
    dispatch = [("PP1", "UP1", "S1"), ("PP1", "UP1", "S2")]
    _make_remote_tree_with_failure(
        remote, dispatch, failed_keys={("PP1", "UP1", "S2")},
    )
    plan = CopyBackPlan(
        run_id=_RID, stage="gemma", round_id=0, attempt_id=_AID,
        expected_match_fields=_expected_match(),
        dispatch_cells=dispatch, artifact_root=local,
        remote_artifact_root=str(remote),
        remote_manifest_root=str(remote),
        # allow_partial defaults to False — strict mode
    )

    cell_pull_calls: list[tuple] = []
    base_transport = _make_transport(remote)

    def spy_transport(*, remote_root, local_temp, cell_relpaths, manifest_relpath):
        if cell_relpaths and not manifest_relpath:
            cell_pull_calls.append(tuple(cell_relpaths))
        base_transport(
            remote_root=remote_root, local_temp=local_temp,
            cell_relpaths=cell_relpaths, manifest_relpath=manifest_relpath,
        )

    with pytest.raises(CopyBackError, match="strict mode.*non-success"):
        copy_back_dispatch_cells(plan, transport=spy_transport)
    # Critical: the second-pass cell rsync must NOT have been issued.
    assert cell_pull_calls == []
    # And nothing was promoted into the live tree.
    assert not (local / "outputs").exists() or not any((local / "outputs").rglob("*"))


def test_partial_mode_skips_failed_cells_and_records_them(tmp_path):
    """F2-strict: with ``allow_partial=True``, a non-success cell is skipped
    at copy-back but recorded on PromotionResult.failed_cells so partial
    mode is auditable, not a silent-loss path."""
    remote = tmp_path / "remote_root"
    local = tmp_path / "local_root"
    dispatch = [("PP1", "UP1", "S1"), ("PP1", "UP1", "S2"), ("PP1", "UP1", "S3")]
    _make_remote_tree_with_failure(
        remote, dispatch, failed_keys={("PP1", "UP1", "S2")},
        failed_status="error",
    )
    plan = CopyBackPlan(
        run_id=_RID, stage="gemma", round_id=0, attempt_id=_AID,
        expected_match_fields=_expected_match(),
        dispatch_cells=dispatch, artifact_root=local,
        remote_artifact_root=str(remote),
        remote_manifest_root=str(remote),
        allow_partial=True,
    )
    result = copy_back_dispatch_cells(plan, transport=_make_transport(remote))
    assert result.partial is True
    assert result.failed_cells == [("PP1", "UP1", "S2", "error")]
    # Promoted set is the success-only subset, never includes the failed cell.
    assert set(result.promoted_cells) == {("PP1", "UP1", "S1"), ("PP1", "UP1", "S3")}
    # Live tree contains the success artifacts but NOT the failed one.
    for sid in ("S1", "S3"):
        live = local / cp.cell_artifact_path(_RID, "gemma", 0, "PP1", "UP1", sid)
        assert live.is_file()
    failed_live = local / cp.cell_artifact_path(_RID, "gemma", 0, "PP1", "UP1", "S2")
    assert not failed_live.exists()
