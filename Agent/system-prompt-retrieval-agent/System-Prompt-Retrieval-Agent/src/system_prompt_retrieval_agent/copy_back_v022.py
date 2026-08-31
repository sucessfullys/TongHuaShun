"""V0.2.2 cell-scoped copy-back + atomic promotion (S04.01–S04.10).

Core invariants (plan §2.7e + §2.7a):

* Copy-back pulls **only** the cell directories under
  ``outputs/v02/{run_id}/{stage}/round_{round_id}/...`` corresponding
  to cells in the dispatch set ``D``, plus the current per-attempt
  manifest at ``{stage}_round_{round_id}_attempt_{attempt_id}.json``.
* Never pulls the entire stage directory; never legacy
  ``outputs/v01/{run_id}/{cell_id}`` paths.
* Temp directory lives under the **same filesystem** as the live
  artifact root; promotion is per-cell atomic via ``os.replace``.
* Per-attempt manifest is renamed into its non-overwriting path; the
  canonical pointer is retargeted **only after** the merged manifest
  is durably written (this module exposes a hook the orchestrator
  calls in the right order — S05.08a).
* Cells in ``L`` are never re-rsynced or overwritten by a ``D``-only
  attempt.
* No ``rsync --delete`` against live artifact directories. No
  stage-directory-wide replace. Deletion is bounded to temp/staging
  dirs.
* Failed sha-256 verification (truncated rsync) aborts copy-back —
  file existence alone is not sufficient.

The actual rsync invocation is delegated to an injected ``transport``
callable so the module can run under unit tests without SSH. The
default transport is :func:`rsync_cells_via_ssh`.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .remote._vendored import canonical_paths as cp


CellKeyTuple = tuple[str, str, str]


class CopyBackError(RuntimeError):
    """Copy-back failed (verification, transport, or promotion)."""


@dataclass
class CopyBackPlan:
    run_id: str
    stage: str
    round_id: int
    attempt_id: str
    expected_match_fields: Mapping[str, Any]
    dispatch_cells: Sequence[CellKeyTuple]
    artifact_root: Path  # local live artifact root
    remote_artifact_root: str  # rsync source prefix (e.g. "3h100:/mnt/.../outputs/v02")
    remote_manifest_root: str  # often == remote_artifact_root
    # Strict mode (default): if the per-attempt manifest reports any cell with
    # status not in SUCCESSFUL_CELL_STATUSES, raise CopyBackError BEFORE pulling
    # any cell artifacts. In partial mode, failed cells are skipped at copy-back
    # but recorded on PromotionResult.failed_cells for visibility.
    allow_partial: bool = False


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_unlink(path: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        path.unlink()


def _temp_root_same_fs(parent: Path) -> Path:
    """Create a temp dir inside ``parent`` (guaranteed same filesystem)."""
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=".cb_v022_", dir=str(parent)))


# ---------------------------------------------------------------------------
# Transport adapter (injectable)
# ---------------------------------------------------------------------------


def rsync_cells_via_ssh(
    *,
    remote_root: str,
    local_temp: Path,
    cell_relpaths: Iterable[str],
    manifest_relpath: str | None,
) -> None:
    """Real rsync transport. Pulls the listed cell relpaths plus, if given,
    the per-attempt manifest. Pass ``manifest_relpath=None`` to pull cells
    only (used by the second pass of the status-aware copy-back). Never uses
    ``--delete``.
    """
    relpaths = list(cell_relpaths)
    if manifest_relpath:
        relpaths.append(manifest_relpath)
    if not relpaths:
        return
    files_from_path = local_temp / ".files-from.txt"
    files_from_path.write_text("\n".join(relpaths) + "\n")
    cmd = [
        "rsync", "-av",
        "--files-from", str(files_from_path),
        f"{remote_root}/", str(local_temp) + "/",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise CopyBackError(
            f"rsync failed (rc={proc.returncode}): {proc.stderr.strip()}"
        )


Transport = Callable[..., None]


# ---------------------------------------------------------------------------
# Copy-back orchestration
# ---------------------------------------------------------------------------


@dataclass
class PromotionResult:
    promoted_cells: list[CellKeyTuple] = field(default_factory=list)
    promoted_manifest_path: Path | None = None
    pending_pointer_path: Path | None = None
    pending_attempt_path: Path | None = None
    # Cells reported as non-success in the per-attempt manifest. Always populated
    # so partial mode does not become a silent-loss path. Each entry is
    # (prompt_pair_id, user_prompt_id, sample_id, status).
    failed_cells: list[tuple[str, str, str, str]] = field(default_factory=list)
    partial: bool = False


def copy_back_dispatch_cells(
    plan: CopyBackPlan,
    *,
    transport: Transport = rsync_cells_via_ssh,
) -> PromotionResult:
    """Execute the cell-scoped copy-back per ``plan``.

    Status-aware two-pass design (S04.03+):

    1. Pull ONLY the per-attempt manifest into a fresh temp dir.
    2. Parse it. In strict mode (``plan.allow_partial=False``) any cell whose
       status is not in :data:`cp.SUCCESSFUL_CELL_STATUSES` aborts the stage
       with :class:`CopyBackError` BEFORE any cell artifact is touched. This
       prevents the "rsync rc=23 / link_stat" failure mode from masking a
       real per-cell remote failure as a transport error.
    3. In partial mode, non-success cells are recorded on
       :attr:`PromotionResult.failed_cells` (never silently dropped) and the
       cell rsync is restricted to the successful subset.
    4. Pull the (filtered) success cells in a second rsync pass.
    5. Run :func:`_verify_temp_manifest` and :func:`_atomic_promote` against
       the success-only cell relpaths.

    Returns a :class:`PromotionResult`. The canonical pointer is **not**
    retargeted by this function; the caller must call
    :func:`finalize_pointer_after_merge` once the merged-view manifest is
    durably written (S04.04 / S05.08a).
    """
    manifest_relpath = cp.stage_manifest_attempt_path(
        plan.run_id, plan.stage, plan.round_id, plan.attempt_id
    )

    temp_root = _temp_root_same_fs(plan.artifact_root)
    try:
        # --- Pass 1: manifest only ----------------------------------------
        transport(
            remote_root=plan.remote_artifact_root,
            local_temp=temp_root,
            cell_relpaths=[],
            manifest_relpath=manifest_relpath,
        )
        manifest_temp = temp_root / manifest_relpath
        if not manifest_temp.is_file():
            raise CopyBackError(
                f"per-attempt manifest not transferred: {manifest_temp}"
            )
        manifest = json.loads(manifest_temp.read_text(encoding="utf-8"))

        # --- Status-aware filtering --------------------------------------
        success_cells: list[CellKeyTuple] = []
        failed_cells: list[tuple[str, str, str, str]] = []
        for cell in manifest.get("cells", []):
            key = (cell["prompt_pair_id"], cell["user_prompt_id"], cell["sample_id"])
            if cell.get("status") in cp.SUCCESSFUL_CELL_STATUSES:
                success_cells.append(key)
            else:
                failed_cells.append((*key, cell.get("status", "?")))

        if failed_cells and not plan.allow_partial:
            preview = failed_cells[:5]
            raise CopyBackError(
                f"strict mode: per-attempt manifest reports {len(failed_cells)} "
                f"non-success cell(s); aborting before cell rsync. "
                f"First {len(preview)}: {preview}"
            )

        cell_relpaths = [
            cp.cell_artifact_path(
                plan.run_id, plan.stage, plan.round_id, pid, upid, sid
            )
            for (pid, upid, sid) in success_cells
        ]

        # --- Pass 2: pull success cells only ------------------------------
        if cell_relpaths:
            transport(
                remote_root=plan.remote_artifact_root,
                local_temp=temp_root,
                cell_relpaths=cell_relpaths,
                manifest_relpath=None,
            )

        _verify_temp_manifest(manifest, plan, cell_relpaths, temp_root)
        result = _atomic_promote(
            plan, cell_relpaths, manifest_relpath, temp_root
        )
        result.failed_cells = failed_cells
        result.partial = bool(failed_cells)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    else:
        shutil.rmtree(temp_root, ignore_errors=True)
    return result


def _verify_temp_manifest(
    manifest: Mapping[str, Any],
    plan: CopyBackPlan,
    cell_relpaths: Sequence[str],
    temp_root: Path,
) -> None:
    """S04.03: verify per-attempt manifest match-fields + cell coverage +
    on-disk size/sha against the manifest declarations BEFORE promotion.
    """
    for key in cp.MATCH_FIELDS:
        if manifest.get(key) != plan.expected_match_fields.get(key):
            raise CopyBackError(
                f"copied manifest match-field {key!r} mismatch: "
                f"expected {plan.expected_match_fields.get(key)!r}, "
                f"got {manifest.get(key)!r}"
            )
    declared_keys = {
        (c["prompt_pair_id"], c["user_prompt_id"], c["sample_id"])
        for c in manifest.get("cells", [])
    }
    expected_keys = set(plan.dispatch_cells)
    if declared_keys != expected_keys:
        raise CopyBackError(
            f"per-attempt manifest cells != dispatch set D: "
            f"missing={sorted(expected_keys - declared_keys)} "
            f"extra={sorted(declared_keys - expected_keys)}"
        )
    relpaths_set = set(cell_relpaths)
    for cell in manifest.get("cells", []):
        if cell.get("status") not in cp.SUCCESSFUL_CELL_STATUSES:
            continue
        relpath = cell.get("artifact_relpath")
        if relpath not in relpaths_set:
            raise CopyBackError(
                f"successful cell relpath outside dispatch scope: {relpath}"
            )
        on_temp = temp_root / relpath
        if not on_temp.is_file():
            raise CopyBackError(
                f"successful cell artifact missing under temp: {on_temp}"
            )
        size = on_temp.stat().st_size
        sha = _hash_file(on_temp)
        if size != cell.get("artifact_size_bytes"):
            raise CopyBackError(
                f"truncated transfer: temp size {size} != manifest size "
                f"{cell.get('artifact_size_bytes')} at {on_temp}"
            )
        if sha != cell.get("artifact_sha256"):
            raise CopyBackError(
                f"truncated transfer: temp sha256 {sha} != manifest sha256 "
                f"{cell.get('artifact_sha256')} at {on_temp}"
            )


def _atomic_promote(
    plan: CopyBackPlan,
    cell_relpaths: Sequence[str],
    manifest_relpath: str,
    temp_root: Path,
) -> PromotionResult:
    """S04.04: atomic per-cell rename from temp to live; per-attempt
    manifest renamed into its non-overwriting path. Pointer NOT touched.
    """
    result = PromotionResult()
    for relpath in cell_relpaths:
        temp_path = temp_root / relpath
        if not temp_path.is_file():
            # Cell may have ended up status='error' / 'missing'; skip.
            continue
        live_path = plan.artifact_root / relpath
        live_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp_path, live_path)
        # Decompose relpath into (pid, upid, sid) for the result record.
        parts = relpath.split("/")
        # outputs/v02/<run_id>/<stage>/round_<n>/<pid>/<upid>/<sid>/<filename>
        if len(parts) >= 8:
            result.promoted_cells.append((parts[-4], parts[-3], parts[-2]))

    # Per-attempt manifest into its non-overwriting path.
    temp_manifest = temp_root / manifest_relpath
    live_manifest = plan.artifact_root / manifest_relpath
    if live_manifest.exists():
        raise CopyBackError(
            f"per-attempt manifest path already occupied (would overwrite): "
            f"{live_manifest}"
        )
    live_manifest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temp_manifest, live_manifest)
    result.promoted_manifest_path = live_manifest
    result.pending_attempt_path = live_manifest
    result.pending_pointer_path = plan.artifact_root / cp.stage_manifest_pointer_path(
        plan.run_id, plan.stage, plan.round_id
    )
    return result


def finalize_pointer_after_merge(
    *, attempt_path: Path, pointer_path: Path
) -> None:
    """S04.04 / S05.08a: atomically retarget the canonical pointer at the
    new per-attempt manifest. Caller must guarantee the merged-view
    manifest is already durably written before invoking.
    """
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_link = pointer_path.with_suffix(pointer_path.suffix + ".tmp")
    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()
    try:
        os.symlink(attempt_path.name, tmp_link)
        os.replace(tmp_link, pointer_path)
    except (OSError, NotImplementedError):
        # Filesystem fallback: byte-copy the per-attempt manifest into pointer.
        fd, tmp_name = tempfile.mkstemp(prefix=".cbpt_", dir=str(pointer_path.parent))
        try:
            with os.fdopen(fd, "wb") as out, attempt_path.open("rb") as src:
                shutil.copyfileobj(src, out)
                out.flush()
                try:
                    os.fsync(out.fileno())
                except OSError:
                    pass
            os.replace(tmp_name, pointer_path)
        except Exception:
            _safe_unlink(Path(tmp_name))
            raise


# ---------------------------------------------------------------------------
# Resume artifact view (S04.07 / S04.08)
# ---------------------------------------------------------------------------


def build_missing_manifest(
    *,
    run_id: str,
    stage: str,
    round_id: int,
    artifact_root: Path,
    expected_match_fields: Mapping[str, Any],
    survivor_cells: Sequence[CellKeyTuple],
) -> list[CellKeyTuple]:
    """Scan canonical local artifacts + prior per-attempt manifests under
    ``(run_id, round_id, stage)`` and return ``D = S - L``.
    """
    from .survivor_resume_v022 import (
        CellKey,
        compute_dispatch_set,
        compute_local_artifact_view,
    )

    survivor = [CellKey(*c) for c in survivor_cells]
    local_view = compute_local_artifact_view(
        run_id=run_id,
        stage=stage,
        round_id=round_id,
        artifact_root=artifact_root,
        expected_match_fields=expected_match_fields,
        survivor_set=survivor,
    )
    return [c.as_tuple() for c in compute_dispatch_set(survivor, local_view)]


# ---------------------------------------------------------------------------
# Anti-pattern guards (S04.06)
# ---------------------------------------------------------------------------


FORBIDDEN_RSYNC_FLAGS = ("--delete", "--delete-after", "--delete-before",
                         "--delete-during", "--delete-excluded")


def assert_no_destructive_flags(rsync_argv: Sequence[str]) -> None:
    """Raise :class:`CopyBackError` if any destructive flag appears in
    an rsync argv targeted at the live artifact root.
    """
    bad = [f for f in rsync_argv if f in FORBIDDEN_RSYNC_FLAGS]
    if bad:
        raise CopyBackError(
            f"forbidden rsync flag(s) against live artifact root: {bad}"
        )
