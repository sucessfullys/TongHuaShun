"""V0.2.2 stage barriers (S03.01–S03.08).

Two distinct barriers per stage:

* :func:`run_remote_stage_barrier` — runs **before** copy-back. Validates
  the per-attempt manifest's identity (``match_fields``), expected cell
  keys, duplicate cell keys, derived rollups, lifecycle state against
  the cold/warm matrix (plan §3.9), and the remote-declared
  ``artifact_relpath`` / ``artifact_size_bytes`` / ``artifact_sha256``
  shape on every cell.

* :func:`run_local_artifact_barrier` — runs **after** copy-back, before
  survivor generation, resume-merge, or scoring. Verifies the
  **copied** stage manifest's ``match_fields``, every successful cell's
  canonical local artifact existence, and on-disk size + sha-256
  matches the manifest declarations (S03.04).

Strict-mode (S03.05) and partial-mode threshold (S03.06) decisions are
applied by :func:`apply_survival_policy` after both barriers pass.

Lifecycle informational fields (``lifecycle_mode``,
``lifecycle_state_after``) are explicitly **not** part of resume
match-field comparison (S03.08).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .remote._vendored import canonical_paths as cp


class BarrierViolation(ValueError):
    """A barrier check failed — caller must abort the round."""


# ---------------------------------------------------------------------------
# Remote stage barrier (S03.03)
# ---------------------------------------------------------------------------


def run_remote_stage_barrier(
    manifest: Mapping[str, Any],
    *,
    expected_match_fields: Mapping[str, Any],
    expected_cell_keys: Iterable[tuple[str, str, str]] | None = None,
) -> None:
    """Validate a remote per-attempt stage manifest before copy-back.

    Raises :class:`BarrierViolation` on any failure. ``expected_cell_keys``
    is the dispatch set ``D`` if known; when supplied, the barrier
    additionally proves the manifest's ``cells[]`` covers exactly ``D``.
    """
    for key in cp.MATCH_FIELDS:
        if key not in manifest:
            raise BarrierViolation(f"manifest missing match-field {key!r}")
        if manifest[key] != expected_match_fields.get(key):
            raise BarrierViolation(
                f"manifest match-field {key!r} mismatch: "
                f"expected {expected_match_fields.get(key)!r}, got {manifest[key]!r}"
            )

    lifecycle_mode = manifest.get("lifecycle_mode")
    state_after = manifest.get("lifecycle_state_after")
    if lifecycle_mode is None or state_after is None:
        raise BarrierViolation(
            "manifest missing lifecycle informational fields"
        )
    cp.validate_lifecycle_state_after(lifecycle_mode, state_after)

    cells = manifest.get("cells")
    if not isinstance(cells, list):
        raise BarrierViolation("manifest 'cells' is not a list")
    if not cells:
        raise BarrierViolation(
            "manifest 'cells' is empty; remote must not be called for D=∅"
        )

    seen_keys: set[tuple[str, str, str]] = set()
    stage = manifest["stage"]
    for cell in cells:
        cp.validate_cell_record(cell, stage=stage)
        # Remote may not emit carried_over (S02.13 / merge ownership).
        if cell["status"] == cp.CELL_STATUS_CARRIED_OVER:
            raise BarrierViolation(
                "remote stage manifest must not emit status='carried_over'"
            )
        key = (cell["prompt_pair_id"], cell["user_prompt_id"], cell["sample_id"])
        if key in seen_keys:
            raise BarrierViolation(f"duplicate cell key in manifest: {key}")
        seen_keys.add(key)

    if expected_cell_keys is not None:
        expected_set = set(expected_cell_keys)
        if seen_keys != expected_set:
            missing = expected_set - seen_keys
            extra = seen_keys - expected_set
            raise BarrierViolation(
                f"manifest 'cells' set != dispatch set D: "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )

    _validate_derived_rollups(cells, manifest)


def _validate_derived_rollups(
    cells: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> None:
    """Re-derive rollups from cells[] and cross-check against the manifest's
    declared ``pair_rollups`` and ``per_user_prompt`` (S03.01 / S03.02)."""
    pair_rollups: dict[str, dict[str, int]] = {}
    per_up: dict[str, dict[str, dict[str, int]]] = {}
    for cell in cells:
        pid = cell["prompt_pair_id"]
        upid = cell["user_prompt_id"]
        ok = 1 if cell["status"] in cp.SUCCESSFUL_CELL_STATUSES else 0
        err = 1 - ok
        pair_rollups.setdefault(pid, {"ok": 0, "errors": 0, "total": 0})
        pair_rollups[pid]["ok"] += ok
        pair_rollups[pid]["errors"] += err
        pair_rollups[pid]["total"] += 1
        per_up.setdefault(pid, {})
        per_up[pid].setdefault(upid, {"ok": 0, "errors": 0, "total": 0})
        per_up[pid][upid]["ok"] += ok
        per_up[pid][upid]["errors"] += err
        per_up[pid][upid]["total"] += 1

    declared_pair = manifest.get("pair_rollups", {})
    declared_up = manifest.get("per_user_prompt", {})
    for pid, expected in pair_rollups.items():
        if declared_pair.get(pid) != expected:
            raise BarrierViolation(
                f"pair_rollups for {pid!r} mismatch: declared "
                f"{declared_pair.get(pid)!r}, derived {expected!r}"
            )
    for pid, ups in per_up.items():
        for upid, expected in ups.items():
            if declared_up.get(pid, {}).get(upid) != expected:
                raise BarrierViolation(
                    f"per_user_prompt[{pid}][{upid}] mismatch: declared "
                    f"{declared_up.get(pid, {}).get(upid)!r}, derived {expected!r}"
                )


# ---------------------------------------------------------------------------
# Local artifact barrier (S03.04)
# ---------------------------------------------------------------------------


def run_local_artifact_barrier(
    *,
    copied_manifest: Mapping[str, Any],
    expected_match_fields: Mapping[str, Any],
    artifact_root: Path,
) -> None:
    """Validate a copied stage manifest against on-disk artifacts.

    Raises :class:`BarrierViolation` on the first failure.
    """
    for key in cp.MATCH_FIELDS:
        if copied_manifest.get(key) != expected_match_fields.get(key):
            raise BarrierViolation(
                f"copied manifest match-field {key!r} mismatch: "
                f"expected {expected_match_fields.get(key)!r}, "
                f"got {copied_manifest.get(key)!r}"
            )

    cells = copied_manifest.get("cells", [])
    for cell in cells:
        if cell.get("status") not in cp.SUCCESSFUL_CELL_STATUSES:
            continue
        relpath = cell.get("artifact_relpath")
        if not relpath:
            raise BarrierViolation(
                f"successful cell missing artifact_relpath: "
                f"{cell.get('prompt_pair_id')}, {cell.get('user_prompt_id')}, "
                f"{cell.get('sample_id')}"
            )
        on_disk = artifact_root / relpath
        if not on_disk.is_file():
            raise BarrierViolation(
                f"successful cell missing on-disk artifact: {on_disk}"
            )
        try:
            size = on_disk.stat().st_size
        except OSError as exc:
            raise BarrierViolation(
                f"unable to stat artifact {on_disk}: {exc}"
            ) from exc
        if size != cell.get("artifact_size_bytes"):
            raise BarrierViolation(
                f"on-disk size {size} != manifest size "
                f"{cell.get('artifact_size_bytes')} at {on_disk}"
            )
        h = hashlib.sha256()
        with on_disk.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        sha = h.hexdigest()
        if sha != cell.get("artifact_sha256"):
            raise BarrierViolation(
                f"on-disk sha256 {sha} != manifest sha256 "
                f"{cell.get('artifact_sha256')} at {on_disk}"
            )


# ---------------------------------------------------------------------------
# Survival policy (S03.05 / S03.06)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PartialModeThresholds:
    min_surviving_user_prompts: int = 1
    min_sample_ratio: float = 1.0
    required_categories: tuple[str, ...] = ()


@dataclass
class SurvivalPolicyResult:
    surviving_pairs: list[str] = field(default_factory=list)
    failed_pairs: list[dict[str, str]] = field(default_factory=list)


def apply_survival_policy(
    merged_manifest: Mapping[str, Any],
    *,
    enabled_user_prompt_ids: Sequence[str],
    enabled_sample_ids: Sequence[str],
    allow_partial: bool,
    thresholds: PartialModeThresholds | None = None,
    cell_categories: Mapping[tuple[str, str, str], str] | None = None,
    sample_user_prompt_map: Mapping[str, str] | None = None,
) -> SurvivalPolicyResult:
    """Apply strict / partial survival rules to a merged stage view."""
    cells_by_pair: dict[str, list[Mapping[str, Any]]] = {}
    for cell in merged_manifest.get("cells", []):
        cells_by_pair.setdefault(cell["prompt_pair_id"], []).append(cell)

    result = SurvivalPolicyResult()
    th = thresholds or PartialModeThresholds()
    for pid, cells in cells_by_pair.items():
        if not allow_partial:
            if all(
                c["status"] in cp.SUCCESSFUL_CELL_STATUSES for c in cells
            ) and _covers_all_required(
                cells, enabled_user_prompt_ids, enabled_sample_ids,
                sample_user_prompt_map=sample_user_prompt_map,
            ):
                result.surviving_pairs.append(pid)
            else:
                result.failed_pairs.append(
                    {"prompt_pair_id": pid, "failure_reason": "strict_mode_incomplete"}
                )
            continue
        # Partial mode
        ok_cells = [c for c in cells if c["status"] in cp.SUCCESSFUL_CELL_STATUSES]
        per_up_with_ok = {c["user_prompt_id"] for c in ok_cells}
        if len(per_up_with_ok) < th.min_surviving_user_prompts:
            result.failed_pairs.append(
                {"prompt_pair_id": pid, "failure_reason": "below_min_user_prompts"}
            )
            continue
        sample_ratio = (
            len(ok_cells) / max(1, len(cells))
        )
        if sample_ratio < th.min_sample_ratio:
            result.failed_pairs.append(
                {"prompt_pair_id": pid, "failure_reason": "below_min_sample_ratio"}
            )
            continue
        if th.required_categories and cell_categories is not None:
            covered = {
                cell_categories.get(
                    (c["prompt_pair_id"], c["user_prompt_id"], c["sample_id"])
                )
                for c in ok_cells
            }
            missing = set(th.required_categories) - covered
            if missing:
                result.failed_pairs.append(
                    {"prompt_pair_id": pid,
                     "failure_reason": f"category_coverage_missing:{','.join(sorted(missing))}"}
                )
                continue
        result.surviving_pairs.append(pid)

    return result


def _covers_all_required(
    cells: Sequence[Mapping[str, Any]],
    enabled_user_prompt_ids: Sequence[str],
    enabled_sample_ids: Sequence[str],
    sample_user_prompt_map: Mapping[str, str] | None = None,
) -> bool:
    if sample_user_prompt_map is not None:
        expected = {(sample_user_prompt_map[s], s) for s in enabled_sample_ids
                    if s in sample_user_prompt_map}
    else:
        expected = {(u, s) for u in enabled_user_prompt_ids for s in enabled_sample_ids}
    have = {(c["user_prompt_id"], c["sample_id"]) for c in cells
            if c["status"] in cp.SUCCESSFUL_CELL_STATUSES}
    return expected.issubset(have)
