"""V0.2.2 survivor + resume + merged-view manifests (S02 + S04 contracts).

This module owns the local-agent side of the resume × survivor merge
rule (plan §2.7a):

    D = S - L

where:

* ``S`` is the upstream merged-view survivor cell set for the current
  stage (or, for Gemma, the §2.7c base set
  ``enabled_prompt_pairs × enabled_user_prompts × configured_sample_ids``).
* ``L`` is the set of cells already valid on local disk (per-attempt
  manifest match-fields match + size + sha-256 match the cached values).
* ``D`` is the dispatch set sent to the remote stage runner.

The remote stage manifest's ``cells[]`` contains entries only for
cells in ``D``. The local agent constructs the merged stage view at
``stage_manifest_merged_path(...)`` containing every cell in ``S``:
``L`` cells with ``status="carried_over"`` from the prior copied
manifest, ``D`` cells from the current remote response.

Empty dispatch (``D = ∅``) is the local agent's responsibility to
short-circuit per plan §2.7d: do not POST to the remote, write a
merged manifest with ``cells[] = L`` and run the local artifact
barrier on ``L``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .remote._vendored import canonical_paths as cp


PLACEHOLDER_SAMPLE_IDS = frozenset({"__all__", "*", ""})


@dataclass(frozen=True)
class CellKey:
    prompt_pair_id: str
    user_prompt_id: str
    sample_id: str

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.prompt_pair_id, self.user_prompt_id, self.sample_id)

    def as_row(self) -> dict[str, str]:
        return {
            "prompt_pair_id": self.prompt_pair_id,
            "user_prompt_id": self.user_prompt_id,
            "sample_id": self.sample_id,
        }


# ---------------------------------------------------------------------------
# Validation primitives (S02.04 / S02.05 / S02.06)
# ---------------------------------------------------------------------------


class SurvivorValidationError(ValueError):
    """Survivor / resume manifest validation failure."""


def _row_to_cell(row: Mapping[str, Any]) -> CellKey:
    for key in ("prompt_pair_id", "user_prompt_id", "sample_id"):
        if key not in row:
            raise SurvivorValidationError(f"row missing required key: {key!r}")
    return CellKey(
        prompt_pair_id=row["prompt_pair_id"],
        user_prompt_id=row["user_prompt_id"],
        sample_id=row["sample_id"],
    )


def validate_cells(
    cells: Iterable[CellKey],
    *,
    enabled_prompt_pair_ids: Iterable[str],
    enabled_user_prompt_ids: Iterable[str],
    enabled_sample_ids: Iterable[str],
) -> list[CellKey]:
    """Validate a list of survivor / resume cells against enabled corpora.

    Raises :class:`SurvivorValidationError` on the first violation:
    placeholder sample ID, unknown ID, or duplicate row.
    """
    pp = set(enabled_prompt_pair_ids)
    up = set(enabled_user_prompt_ids)
    ss = set(enabled_sample_ids)
    seen: set[tuple[str, str, str]] = set()
    out: list[CellKey] = []
    for cell in cells:
        if cell.sample_id in PLACEHOLDER_SAMPLE_IDS:
            raise SurvivorValidationError(
                f"placeholder sample_id is forbidden under V0.2.2: "
                f"{cell.sample_id!r} (cell={cell.as_tuple()})"
            )
        if cell.prompt_pair_id not in pp:
            raise SurvivorValidationError(
                f"unknown prompt_pair_id {cell.prompt_pair_id!r}"
            )
        if cell.user_prompt_id not in up:
            raise SurvivorValidationError(
                f"unknown user_prompt_id {cell.user_prompt_id!r}"
            )
        if cell.sample_id not in ss:
            raise SurvivorValidationError(
                f"unknown sample_id {cell.sample_id!r}"
            )
        key = cell.as_tuple()
        if key in seen:
            raise SurvivorValidationError(
                f"duplicate cell row: {key}"
            )
        seen.add(key)
        out.append(cell)
    return out


# ---------------------------------------------------------------------------
# Survivor manifest generation (S02.02 / S02.03 / S02.07)
# ---------------------------------------------------------------------------


def _successful_cells_from_manifest(
    manifest: Mapping[str, Any],
) -> list[CellKey]:
    out: list[CellKey] = []
    for cell in manifest.get("cells", []):
        status = cell.get("status")
        if status in cp.SUCCESSFUL_CELL_STATUSES:
            out.append(_row_to_cell(cell))
    return out


def build_survivor_manifest_v022(
    prior_merged_manifest: Mapping[str, Any],
    *,
    enabled_prompt_pair_ids: Iterable[str],
    enabled_user_prompt_ids: Iterable[str],
    enabled_sample_ids: Iterable[str],
) -> list[CellKey]:
    """Build the survivor cell set from an upstream **merged-view** manifest.

    Only cells with status ∈ ``SUCCESSFUL_CELL_STATUSES`` (``ok`` or
    ``carried_over``) survive. The result is validated against the
    enabled corpora — unknown IDs, placeholders, and duplicates abort.
    """
    survivors = _successful_cells_from_manifest(prior_merged_manifest)
    return validate_cells(
        survivors,
        enabled_prompt_pair_ids=enabled_prompt_pair_ids,
        enabled_user_prompt_ids=enabled_user_prompt_ids,
        enabled_sample_ids=enabled_sample_ids,
    )


def build_gemma_base_survivor_set(
    *,
    enabled_prompt_pair_ids: Sequence[str],
    enabled_user_prompt_ids: Sequence[str],
    configured_sample_ids: Sequence[str],
    sample_user_prompt_map: Mapping[str, str] | None = None,
) -> list[CellKey]:
    """Build ``S_gemma`` per plan §2.7c.

    Default: full cartesian (pair × user_prompt × sample).

    When ``sample_user_prompt_map`` is given (mapping ``sample_id`` →
    ``user_prompt_id``), the survivor set collapses the user-prompt
    dimension: for each (pair, sample) we emit exactly ONE cell whose
    ``user_prompt_id`` is taken from the map. This is the "per-sample
    random user_prompt" mode used for pilot/iteration speed — every
    sample picks one of the enabled user_prompts independently. The
    per-pair fan-out in ``pair_id/<up>/sample_id`` directories collapses
    to a single ``<up>`` per (pair, sample) instead of K.
    """
    out: list[CellKey] = []
    if sample_user_prompt_map is not None:
        for pid in enabled_prompt_pair_ids:
            for sid in configured_sample_ids:
                upid = sample_user_prompt_map.get(sid)
                if upid is None:
                    raise SurvivorValidationError(
                        f"sample_user_prompt_map missing entry for sample_id={sid!r}"
                    )
                out.append(CellKey(pid, upid, sid))
    else:
        for pid in enabled_prompt_pair_ids:
            for upid in enabled_user_prompt_ids:
                for sid in configured_sample_ids:
                    out.append(CellKey(pid, upid, sid))
    if not out:
        raise SurvivorValidationError(
            "S_gemma is empty; aborting round per plan §2.7c"
        )
    return out


def write_survivor_jsonl(cells: Sequence[CellKey], target_path: Path) -> Path:
    """Write a survivor manifest as JSONL with atomic temp + rename."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".surv_", dir=str(target_path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for cell in cells:
                fh.write(json.dumps(cell.as_row(), ensure_ascii=False, sort_keys=True))
                fh.write("\n")
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        os.replace(tmp, target_path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
    return target_path


# ---------------------------------------------------------------------------
# Local L-set computation (S02.08 / S04.07)
# ---------------------------------------------------------------------------


@dataclass
class LocalArtifactView:
    """Cells valid on local disk for the current ``(run_id, round_id, stage)``."""

    valid_cells: dict[tuple[str, str, str], dict[str, Any]] = field(default_factory=dict)

    def has(self, key: tuple[str, str, str]) -> bool:
        return key in self.valid_cells

    def cell_ids(self) -> set[tuple[str, str, str]]:
        return set(self.valid_cells)


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_local_artifact_view(
    *,
    run_id: str,
    stage: str,
    round_id: int,
    artifact_root: Path,
    expected_match_fields: Mapping[str, Any],
    survivor_set: Sequence[CellKey],
) -> LocalArtifactView:
    """Compute ``L`` for a stage by intersecting prior per-attempt manifests
    with on-disk artifact validation.

    A cell joins ``L`` iff:
      • some prior per-attempt manifest for ``(run_id, round_id, stage)``
        records the cell with status ∈ ``SUCCESSFUL_CELL_STATUSES``, AND
      • that manifest's ``match_fields`` equal the current invocation's
        expected match-fields, AND
      • the on-disk artifact at the canonical path exists and its
        size + sha-256 match the manifest declarations.
    """
    manifests_dir = artifact_root / cp.manifests_dir(run_id)
    if not manifests_dir.is_dir():
        return LocalArtifactView()
    survivor_keys = {c.as_tuple() for c in survivor_set}
    valid: dict[tuple[str, str, str], dict[str, Any]] = {}
    prefix = f"{stage}_round_{round_id}_attempt_"
    candidates = sorted(
        (p for p in manifests_dir.iterdir() if p.name.startswith(prefix) and p.suffix == ".json"),
        key=lambda p: p.name,
    )
    for manifest_path in candidates:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not _match_fields_equal(payload, expected_match_fields):
            continue
        for record in payload.get("cells", []):
            if record.get("status") not in cp.SUCCESSFUL_CELL_STATUSES:
                continue
            key = (
                record.get("prompt_pair_id"),
                record.get("user_prompt_id"),
                record.get("sample_id"),
            )
            if survivor_keys and key not in survivor_keys:
                continue
            relpath = record.get("artifact_relpath")
            if not relpath:
                continue
            on_disk = artifact_root / relpath
            if not on_disk.is_file():
                continue
            try:
                size_on_disk = on_disk.stat().st_size
                sha_on_disk = _hash_file(on_disk)
            except OSError:
                continue
            if (
                record.get("artifact_size_bytes") != size_on_disk
                or record.get("artifact_sha256") != sha_on_disk
            ):
                continue
            valid[key] = {
                "prompt_pair_id": record["prompt_pair_id"],
                "user_prompt_id": record["user_prompt_id"],
                "sample_id": record["sample_id"],
                "status": cp.CELL_STATUS_CARRIED_OVER,
                "artifact_relpath": relpath,
                "artifact_size_bytes": size_on_disk,
                "artifact_sha256": sha_on_disk,
            }
            for opt in ("parse_status", "parsed"):
                if opt in record:
                    valid[key][opt] = record[opt]
    return LocalArtifactView(valid_cells=valid)


def _match_fields_equal(
    payload: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    for key in cp.MATCH_FIELDS:
        if payload.get(key) != expected.get(key):
            return False
    return True


# ---------------------------------------------------------------------------
# Resume dispatch set D = S - L (S02.08 / S02.09)
# ---------------------------------------------------------------------------


def compute_dispatch_set(
    survivor_set: Sequence[CellKey], local_view: LocalArtifactView
) -> list[CellKey]:
    """Return ``D = S - L`` preserving survivor order."""
    locally_valid = local_view.cell_ids()
    return [c for c in survivor_set if c.as_tuple() not in locally_valid]


def write_resume_missing_cells_jsonl(
    cells: Sequence[CellKey], target_path: Path
) -> Path:
    """Write a ``resume_missing_cells`` JSONL manifest."""
    return write_survivor_jsonl(cells, target_path)


# ---------------------------------------------------------------------------
# Local merged-view construction (S05.08a contract surface)
# ---------------------------------------------------------------------------


def build_merged_view(
    *,
    survivor_set: Sequence[CellKey],
    local_view: LocalArtifactView,
    current_remote_manifest: Mapping[str, Any] | None,
    current_attempt_id: str | None,
    match_fields_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the local-agent merged stage view (plan §2.7a / S05.08a).

    ``S = L (carried_over) + D (current attempt)``. The remote response
    must contain only cells in ``D`` and must not emit
    ``status="carried_over"``. When ``current_remote_manifest`` is
    ``None`` (empty-dispatch short-circuit per plan §2.7d), the merged
    view contains only ``L`` cells.
    """
    if current_remote_manifest is not None:
        for record in current_remote_manifest.get("cells", []):
            if record.get("status") == cp.CELL_STATUS_CARRIED_OVER:
                raise SurvivorValidationError(
                    "remote stage manifest must not emit status='carried_over'; "
                    "merge ownership belongs to the local agent"
                )
        match_fields = {k: current_remote_manifest.get(k) for k in cp.MATCH_FIELDS}
        attempt_id = current_remote_manifest.get("attempt_id") or current_attempt_id
    else:
        match_fields = {}
        attempt_id = current_attempt_id
    if match_fields_override is not None:
        match_fields = {k: match_fields_override.get(k) for k in cp.MATCH_FIELDS}

    survivor_keys = [c.as_tuple() for c in survivor_set]
    by_key_local = local_view.valid_cells
    by_key_remote: dict[tuple[str, str, str], dict[str, Any]] = {}
    if current_remote_manifest is not None:
        for record in current_remote_manifest.get("cells", []):
            key = (
                record.get("prompt_pair_id"),
                record.get("user_prompt_id"),
                record.get("sample_id"),
            )
            by_key_remote[key] = dict(record)

    cells_out: list[dict[str, Any]] = []
    for key in survivor_keys:
        if key in by_key_remote:
            cells_out.append(by_key_remote[key])
        elif key in by_key_local:
            cells_out.append(dict(by_key_local[key]))
        # Otherwise the cell is in S but not in L and not in current
        # remote response → barrier will catch as missing.

    merged = {
        **{k: match_fields.get(k) for k in cp.MATCH_FIELDS},
        "merged_view": True,
        "merged_attempt_id": attempt_id,
        "cells": cells_out,
    }
    return merged


def write_merged_view(
    *,
    artifact_root: Path,
    run_id: str,
    stage: str,
    round_id: int,
    merged_payload: Mapping[str, Any],
) -> Path:
    """Write the locally-owned merged manifest with atomic temp + rename."""
    relpath = cp.stage_manifest_merged_path(run_id, stage, round_id)
    target = artifact_root / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".merged_", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(cp.canonical_json(merged_payload))
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        os.replace(tmp, target)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
    return target
