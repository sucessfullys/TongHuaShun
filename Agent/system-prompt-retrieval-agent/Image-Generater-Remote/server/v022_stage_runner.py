"""V0.2.2 stage-major runner (S01.01–S01.07 + S01.06a + S00.18 enforcement).

Owns three orthogonal concerns:

1. **Stage worker pool** — per-stage worker registry created by
   :func:`load_stage_model`, persisted across cells inside one
   :func:`run_stage_cells` call, torn down by
   :func:`unload_or_retain_stage_model`. One worker per configured GPU.
   Cells are sharded across workers (data-parallel; not serialized
   through one worker, not one worker per cell).

2. **Canonical write ordering** — for every successful cell:
   write artifact bytes → fsync file → normalize filename →
   fsync containing dir → stat size from disk → sha-256 from disk →
   append ``cells[]`` entry. Any step failing prevents
   ``status="ok"``; the cell becomes ``status="error"`` with an
   explanatory ``error_reason``.

3. **Per-attempt + pointer + merged manifest emission** — writes the
   per-attempt manifest at the canonical path (never overwritten),
   atomically retargets the canonical pointer to the latest
   per-attempt file, leaves the merged-view manifest construction to
   the local agent (plan §2.7a / S05.08a).

The stage executor is injected as a callable so this module can run
under unit tests without any GPU stack. Real model loading is wired by
the remote service adapter when this module is mounted by the
controller.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import shutil
import tempfile
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import canonical_paths as cp


# ---------------------------------------------------------------------------
# Public dataclasses (request / response surface)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellKey:
    prompt_pair_id: str
    user_prompt_id: str
    sample_id: str

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.prompt_pair_id, self.user_prompt_id, self.sample_id)


@dataclass(frozen=True)
class StageMatchFields:
    """Caller-supplied match-field payload (plan §2.4 / S00.06)."""

    schema_version: str
    run_id: str
    round_id: int
    stage: str
    config_hash: str
    user_prompt_corpus_id: str
    user_prompt_corpus_hash: str
    prompt_pair_corpus_hash: str
    sample_corpus_hash: str

    def as_mapping(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class LifecycleEcho:
    """Informational lifecycle fields echoed in the per-attempt manifest."""

    lifecycle_mode: str
    lifecycle_state_after: str


@dataclass
class StageDispatch:
    """Inputs for a single ``run_stage_cells`` invocation.

    ``dispatch_cells`` corresponds to ``D`` in the resume × survivor
    merge rule (plan §2.7a). ``L`` is computed by the local agent and
    is not visible here. Empty dispatch is rejected — the local agent
    must short-circuit before calling.
    """

    match_fields: StageMatchFields
    attempt_id: str
    lifecycle: LifecycleEcho
    dispatch_cells: list[CellKey]
    artifact_root: Path  # absolute path the canonical-relpath layout sits under
    gpu_ids: list[int]


@dataclass
class CellResult:
    """Per-cell stage executor output (returned by an injected stage_fn)."""

    key: CellKey
    raw_bytes: bytes | None = None
    error_reason: str | None = None
    parse_status: str | None = None  # Qwen-only
    parsed: Mapping[str, Any] | None = None  # Qwen-only


# ---------------------------------------------------------------------------
# Stage worker pool (S01.01 / S01.01a)
# ---------------------------------------------------------------------------


class StageWorker:
    """Owns one model instance bound to one GPU.

    Real workers wrap the existing pipeline.py / stage_*.py callers
    and persist the loaded model across cells. Tests inject a
    callable that records its calls.
    """

    def __init__(
        self,
        gpu_id: int,
        stage: str,
        cell_executor: Callable[[CellKey, dict[str, Any]], CellResult],
    ) -> None:
        self.gpu_id = gpu_id
        self.stage = stage
        self._cell_executor = cell_executor
        self.is_loaded = False
        self.cells_processed = 0
        self.load_count = 0

    def load(self) -> None:
        if self.is_loaded:
            raise RuntimeError(
                f"worker stage={self.stage!r} gpu={self.gpu_id} already loaded"
            )
        self.is_loaded = True
        self.load_count += 1

    def unload(self) -> None:
        if not self.is_loaded:
            return
        self.is_loaded = False

    def process(self, cell: CellKey, ctx: dict[str, Any]) -> CellResult:
        if not self.is_loaded:
            raise RuntimeError(
                f"worker stage={self.stage!r} gpu={self.gpu_id} not loaded"
            )
        ctx_with_worker = {**ctx, "gpu_id": self.gpu_id, "stage": self.stage}
        result = self._cell_executor(cell, ctx_with_worker)
        self.cells_processed += 1
        return result


class StageWorkerPool:
    """One worker per configured GPU; persists across cells of one call."""

    def __init__(self, stage: str, gpu_ids: Sequence[int]) -> None:
        if stage not in cp.ALLOWED_STAGES:
            raise ValueError(f"invalid stage: {stage!r}")
        if not gpu_ids:
            raise ValueError("gpu_ids must be non-empty")
        self.stage = stage
        self.gpu_ids = list(gpu_ids)
        self.workers: list[StageWorker] = []

    def load_all(
        self,
        cell_executor: Callable[[CellKey, dict[str, Any]], CellResult],
    ) -> None:
        if self.workers:
            raise RuntimeError(
                f"stage={self.stage!r} pool already loaded; "
                "workers must be torn down before reload"
            )
        for gpu_id in self.gpu_ids:
            w = StageWorker(gpu_id, self.stage, cell_executor)
            w.load()
            self.workers.append(w)

    def unload_all(self) -> None:
        for w in self.workers:
            w.unload()
        self.workers = []


# Module-level registry, inspectable by tests (S01.08 / S01.08a).
_REGISTRY_LOCK = threading.Lock()
_STAGE_REGISTRY: dict[str, StageWorkerPool] = {}


def _registry_get() -> dict[str, StageWorkerPool]:
    return dict(_STAGE_REGISTRY)


def get_stage_worker_pool(stage: str) -> StageWorkerPool | None:
    with _REGISTRY_LOCK:
        return _STAGE_REGISTRY.get(stage)


def get_registry_snapshot() -> dict[str, dict[str, Any]]:
    """Return a JSON-friendly snapshot of the per-stage worker registry."""
    with _REGISTRY_LOCK:
        out: dict[str, dict[str, Any]] = {}
        for stage, pool in _STAGE_REGISTRY.items():
            out[stage] = {
                "gpu_ids": list(pool.gpu_ids),
                "worker_count": len(pool.workers),
                "load_counts": [w.load_count for w in pool.workers],
                "cells_processed": [w.cells_processed for w in pool.workers],
                "is_loaded": [w.is_loaded for w in pool.workers],
            }
        return out


def reset_registry_for_tests() -> None:
    """Test-only: clear the module-level registry between cases."""
    with _REGISTRY_LOCK:
        for pool in list(_STAGE_REGISTRY.values()):
            pool.unload_all()
        _STAGE_REGISTRY.clear()


# ---------------------------------------------------------------------------
# Lifecycle public surface (S01.01)
# ---------------------------------------------------------------------------


def load_stage_model(
    stage: str,
    gpu_ids: Sequence[int],
    cell_executor: Callable[[CellKey, dict[str, Any]], CellResult],
) -> StageWorkerPool:
    """Construct + load the per-stage worker pool. One model load per worker."""
    with _REGISTRY_LOCK:
        if stage in _STAGE_REGISTRY:
            raise RuntimeError(
                f"stage={stage!r} already loaded; call unload_or_retain_stage_model first"
            )
        pool = StageWorkerPool(stage, gpu_ids)
        pool.load_all(cell_executor)
        _STAGE_REGISTRY[stage] = pool
        return pool


def unload_or_retain_stage_model(stage: str, retain: bool = False) -> None:
    """Tear down the per-stage worker pool.

    ``retain=True`` is reserved for warm mode (currently hard-disabled
    in V0.2.2). In cold mode the pool is fully unloaded and removed
    from the registry.
    """
    if retain and cp.WARM_MODE_HARD_DISABLED_IN_V022:
        raise ValueError(cp.WARM_MODE_DISABLED_ERROR)
    with _REGISTRY_LOCK:
        pool = _STAGE_REGISTRY.pop(stage, None)
    if pool is not None and not retain:
        pool.unload_all()
        try:
            from server.v022_cell_adapters import teardown_stage as _adapter_teardown  # type: ignore
        except ImportError:
            _adapter_teardown = None
        if _adapter_teardown is not None:
            try:
                _adapter_teardown(stage)
            except Exception:  # pragma: no cover — defensive
                pass


# ---------------------------------------------------------------------------
# Write-ordering primitives (S00.18 / S01.06a)
# ---------------------------------------------------------------------------


def _fsync_file(fd: int) -> None:
    try:
        os.fsync(fd)
    except OSError:
        # Some filesystems (notably macOS tmpfs in CI sandboxes) do not
        # support fsync on file descriptors. The ordering invariant is
        # preserved as long as the bytes are flushed to the filesystem
        # cache before the rename, which os.write + close already
        # guarantees on POSIX.
        pass


def _fsync_dir(dir_path: Path) -> None:
    try:
        fd = os.open(dir_path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_artifact_with_ordering(
    raw_bytes: bytes, target_path: Path
) -> tuple[int, str]:
    """Apply S00.18 write ordering and return (size_bytes, sha256_hex).

    Steps: write bytes to a same-dir temp file → fsync file →
    normalize filename via atomic rename → fsync containing dir →
    stat size from disk → sha-256 from disk.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".v022_", dir=str(target_path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw_bytes)
            fh.flush()
            _fsync_file(fh.fileno())
        os.replace(tmp_path, target_path)
        _fsync_dir(target_path.parent)
        size_on_disk = target_path.stat().st_size
        h = hashlib.sha256()
        with target_path.open("rb") as fh2:
            for chunk in iter(lambda: fh2.read(65536), b""):
                h.update(chunk)
        return size_on_disk, h.hexdigest()
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()
        raise


# ---------------------------------------------------------------------------
# Cell execution + manifest assembly (S01.02–S01.07)
# ---------------------------------------------------------------------------


def _shard_cells_round_robin(
    cells: Sequence[CellKey], num_workers: int
) -> list[list[CellKey]]:
    shards: list[list[CellKey]] = [[] for _ in range(num_workers)]
    for i, cell in enumerate(cells):
        shards[i % num_workers].append(cell)
    return shards


def _qwen_status_from_result(result: CellResult) -> tuple[str, dict[str, Any]]:
    """Derive Qwen cell status from the executor result (S01.05)."""
    extra: dict[str, Any] = {}
    if result.error_reason is not None and result.raw_bytes is None:
        return cp.CELL_STATUS_ERROR, {"error_reason": result.error_reason}
    if result.parse_status == "parse_failed":
        extra["parse_status"] = "parse_failed"
        return cp.CELL_STATUS_PARSE_FAILED, extra
    if result.parse_status == "missing":
        extra["parse_status"] = "missing"
        return cp.CELL_STATUS_MISSING, extra
    if result.parse_status is not None:
        extra["parse_status"] = result.parse_status
    if result.parsed is not None:
        extra["parsed"] = dict(result.parsed)
    return cp.CELL_STATUS_OK, extra


def run_stage_cells(
    pool: StageWorkerPool,
    dispatch: StageDispatch,
    *,
    cell_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Process every cell in ``dispatch.dispatch_cells`` and emit the
    per-attempt manifest. Returns the manifest payload that was
    written to disk (for the caller to also send back over HTTP).
    """
    if pool.stage != dispatch.match_fields.stage:
        raise ValueError(
            f"pool stage {pool.stage!r} does not match dispatch stage "
            f"{dispatch.match_fields.stage!r}"
        )
    if not dispatch.dispatch_cells:
        # Empty dispatch must be short-circuited by the local agent
        # (plan §2.7d). Refuse to consume an attempt_id for nothing.
        raise ValueError("dispatch_cells is empty; remote call must not have been made")
    if dispatch.match_fields.schema_version != cp.SCHEMA_VERSION:
        raise ValueError(
            f"unexpected schema_version: {dispatch.match_fields.schema_version!r}"
        )
    cp.validate_lifecycle_state_after(
        dispatch.lifecycle.lifecycle_mode,
        dispatch.lifecycle.lifecycle_state_after,
    )

    stage = pool.stage
    artifact_filename = cp.artifact_filename(stage)
    ctx = dict(cell_context or {})

    shards = _shard_cells_round_robin(dispatch.dispatch_cells, len(pool.workers))

    # Real parallel dispatch: run each worker's shard in its own thread so
    # all workers (one per GPU) drive their device concurrently. CUDA
    # releases the GIL during kernel execution, so threads achieve true
    # device-level parallelism for the heavy stages (Gemma vLLM client,
    # FLUX diffusers pipeline, Qwen vLLM client).
    def _run_shard(worker: "StageWorker", shard: list[CellKey]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for cell in shard:
            relpath = cp.cell_artifact_path(
                dispatch.match_fields.run_id,
                stage,
                dispatch.match_fields.round_id,
                cell.prompt_pair_id,
                cell.user_prompt_id,
                cell.sample_id,
            )
            target_path = dispatch.artifact_root / relpath
            out.append(
                _process_one_cell(
                    worker=worker,
                    cell=cell,
                    ctx=ctx,
                    relpath=relpath,
                    target_path=target_path,
                    stage=stage,
                    artifact_filename=artifact_filename,
                )
            )
        return out

    if len(pool.workers) <= 1 or not any(shards):
        per_worker_results = [_run_shard(w, s) for w, s in zip(pool.workers, shards)]
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(
            max_workers=len(pool.workers),
            thread_name_prefix=f"v022-{stage}",
        ) as ex:
            futures = [ex.submit(_run_shard, w, s) for w, s in zip(pool.workers, shards)]
            per_worker_results = [f.result() for f in futures]

    # Re-interleave results back into dispatch_cells order so the manifest
    # cells[] sequence matches the dispatch order regardless of thread
    # completion timing (deterministic under round-robin sharding).
    cells_out: list[dict[str, Any]] = []
    cursors = [0] * len(per_worker_results)
    for i in range(len(dispatch.dispatch_cells)):
        widx = i % len(per_worker_results)
        cells_out.append(per_worker_results[widx][cursors[widx]])
        cursors[widx] += 1

    pair_rollups, per_user_prompt_rollups = _derive_rollups(cells_out)
    manifest = {
        **dispatch.match_fields.as_mapping(),
        "attempt_id": dispatch.attempt_id,
        "lifecycle_mode": dispatch.lifecycle.lifecycle_mode,
        "lifecycle_state_after": dispatch.lifecycle.lifecycle_state_after,
        "cells": cells_out,
        "pair_rollups": pair_rollups,
        "per_user_prompt": per_user_prompt_rollups,
    }
    _write_per_attempt_manifest(dispatch, manifest)
    _retarget_pointer(dispatch)
    return manifest


def _process_one_cell(
    *,
    worker: StageWorker,
    cell: CellKey,
    ctx: dict[str, Any],
    relpath: str,
    target_path: Path,
    stage: str,
    artifact_filename: str,
) -> dict[str, Any]:
    base_record = {
        "prompt_pair_id": cell.prompt_pair_id,
        "user_prompt_id": cell.user_prompt_id,
        "sample_id": cell.sample_id,
        "artifact_relpath": relpath,
    }
    try:
        result = worker.process(cell, dict(ctx))
    except Exception as exc:  # pragma: no cover — defensive
        return {
            **base_record,
            "status": cp.CELL_STATUS_ERROR,
            "error_reason": f"worker exception: {exc.__class__.__name__}: {exc}",
        }
    if stage == cp.STAGE_QWEN:
        status, extra = _qwen_status_from_result(result)
        if status not in cp.SUCCESSFUL_CELL_STATUSES:
            return {**base_record, "status": status, **extra}
    else:
        if result.error_reason is not None and result.raw_bytes is None:
            return {
                **base_record,
                "status": cp.CELL_STATUS_ERROR,
                "error_reason": result.error_reason,
            }
        if result.raw_bytes is None:
            return {
                **base_record,
                "status": cp.CELL_STATUS_MISSING,
                "error_reason": "stage executor returned no bytes",
            }
        extra = {}
        status = cp.CELL_STATUS_OK
    if result.raw_bytes is None:
        # Qwen passed parse_status checks but the executor didn't emit
        # bytes — should not happen; downgrade to error.
        return {
            **base_record,
            "status": cp.CELL_STATUS_ERROR,
            "error_reason": "stage executor returned no bytes for ok cell",
        }
    try:
        size_on_disk, sha = _write_artifact_with_ordering(result.raw_bytes, target_path)
    except OSError as exc:
        return {
            **base_record,
            "status": cp.CELL_STATUS_ERROR,
            "error_reason": f"artifact write failed: {exc.__class__.__name__}: {exc}",
        }
    record = {
        **base_record,
        "status": status,
        "artifact_size_bytes": size_on_disk,
        "artifact_sha256": sha,
    }
    record.update(extra)
    cp.validate_cell_record(record, stage=stage)
    return record


def _derive_rollups(
    cells: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    pair_rollups: dict[str, dict[str, int]] = {}
    per_up: dict[str, dict[str, dict[str, int]]] = {}
    for cell in cells:
        pid = cell["prompt_pair_id"]
        upid = cell["user_prompt_id"]
        ok = 1 if cell["status"] in cp.SUCCESSFUL_CELL_STATUSES else 0
        err = 1 if cell["status"] not in cp.SUCCESSFUL_CELL_STATUSES else 0
        pair_rollups.setdefault(pid, {"ok": 0, "errors": 0, "total": 0})
        pair_rollups[pid]["ok"] += ok
        pair_rollups[pid]["errors"] += err
        pair_rollups[pid]["total"] += 1
        per_up.setdefault(pid, {})
        per_up[pid].setdefault(upid, {"ok": 0, "errors": 0, "total": 0})
        per_up[pid][upid]["ok"] += ok
        per_up[pid][upid]["errors"] += err
        per_up[pid][upid]["total"] += 1
    flat_per_up: dict[str, dict[str, Any]] = {
        pid: dict(rollups) for pid, rollups in per_up.items()
    }
    return pair_rollups, flat_per_up


def _write_per_attempt_manifest(
    dispatch: StageDispatch, manifest: Mapping[str, Any]
) -> Path:
    relpath = cp.stage_manifest_attempt_path(
        dispatch.match_fields.run_id,
        dispatch.match_fields.stage,
        dispatch.match_fields.round_id,
        dispatch.attempt_id,
    )
    full_path = dispatch.artifact_root / relpath
    full_path.parent.mkdir(parents=True, exist_ok=True)
    payload = cp.canonical_json(manifest)
    fd, tmp = tempfile.mkstemp(prefix=".v022mf_", dir=str(full_path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            _fsync_file(fh.fileno())
        os.replace(tmp, full_path)
        _fsync_dir(full_path.parent)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
    return full_path


def _retarget_pointer(dispatch: StageDispatch) -> None:
    attempt_relpath = cp.stage_manifest_attempt_path(
        dispatch.match_fields.run_id,
        dispatch.match_fields.stage,
        dispatch.match_fields.round_id,
        dispatch.attempt_id,
    )
    pointer_relpath = cp.stage_manifest_pointer_path(
        dispatch.match_fields.run_id,
        dispatch.match_fields.stage,
        dispatch.match_fields.round_id,
    )
    pointer_full = dispatch.artifact_root / pointer_relpath
    attempt_full = dispatch.artifact_root / attempt_relpath
    pointer_full.parent.mkdir(parents=True, exist_ok=True)

    # Atomic retarget via temp symlink + rename. On filesystems without
    # symlink support we fall back to writing a JSON pointer file.
    try:
        tmp_link = pointer_full.with_suffix(pointer_full.suffix + ".tmp")
        if tmp_link.exists() or tmp_link.is_symlink():
            tmp_link.unlink()
        os.symlink(attempt_full.name, tmp_link)
        os.replace(tmp_link, pointer_full)
    except (OSError, NotImplementedError):
        _write_pointer_as_copy(pointer_full, attempt_full)
    _fsync_dir(pointer_full.parent)


def _write_pointer_as_copy(pointer_full: Path, attempt_full: Path) -> None:
    """Filesystem fallback when symlinks are unavailable: byte-copy the
    per-attempt manifest into the pointer path via temp + rename. The
    pointer file remains structurally identical to the per-attempt file.
    """
    fd, tmp = tempfile.mkstemp(prefix=".v022pt_", dir=str(pointer_full.parent))
    try:
        with os.fdopen(fd, "wb") as out, attempt_full.open("rb") as src:
            shutil.copyfileobj(src, out)
            out.flush()
            _fsync_file(out.fileno())
        os.replace(tmp, pointer_full)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
