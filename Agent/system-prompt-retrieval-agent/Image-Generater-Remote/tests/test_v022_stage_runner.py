"""V0.2.2 stage-runner tests (S01.01–S01.11 except remote deploy).

Covers:

* S01.01 / S01.01a — load_stage_model / run_stage_cells / unload_or_retain_stage_model
  + per-stage worker pool persists across cells; one model load per worker
* S01.02–S01.04 — Gemma / FLUX / Qwen executors load once, process cells, unload once
* S01.05 — Qwen statuses derive from real executor results (ok / parse_failed / missing / error)
* S01.06 — V0.2.2 stage manifests carry locked match-fields + cells[] + rollups
* S01.06a — write ordering: failure to size-stat / sha-256 prevents status="ok"
* S01.07 — user_prompt_corpus_hash echoed on every V0.2.2 stage manifest
* S01.08 — stage worker registry inspection proves one load/unload cycle per stage request
* S01.08a — multi-GPU residency: workers == GPU count, sharded across workers
* S01.09 — canonical artifact paths for Gemma / FLUX / Qwen
* S01.10 — Qwen failed/missing/parse-failed cells appear in cell records and rollups
* S01.11 — cell artifact_size_bytes/artifact_sha256 match on-disk reality;
            truncated artifact (raised in executor) prevents status="ok"
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REMOTE_ROOT = REPO_ROOT / "Image-Generater-Remote"
if str(REMOTE_ROOT) not in sys.path:
    sys.path.insert(0, str(REMOTE_ROOT))

from server import canonical_paths as cp  # noqa: E402
from server import v022_stage_runner as sr  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_registry():
    sr.reset_registry_for_tests()
    yield
    sr.reset_registry_for_tests()


def _make_match(stage: str, run_id="20260426T010000Z-deadbeef", round_id=0) -> sr.StageMatchFields:
    return sr.StageMatchFields(
        schema_version=cp.SCHEMA_VERSION,
        run_id=run_id,
        round_id=round_id,
        stage=stage,
        config_hash="c" * 64,
        user_prompt_corpus_id="upc-v0",
        user_prompt_corpus_hash="u" * 64,
        prompt_pair_corpus_hash="p" * 64,
        sample_corpus_hash="s" * 64,
    )


def _cells(n: int) -> list[sr.CellKey]:
    return [
        sr.CellKey(
            prompt_pair_id=f"PAIR{i // 4:02d}",
            user_prompt_id=f"UP{(i // 2) % 2:02d}",
            sample_id=f"SAMPLE{i:03d}",
        )
        for i in range(n)
    ]


def _ok_executor(payload_for_cell):
    def _fn(cell, ctx):
        return sr.CellResult(key=cell, raw_bytes=payload_for_cell(cell))
    return _fn


def _qwen_executor(per_cell):
    def _fn(cell, ctx):
        spec = per_cell[cell.sample_id]
        return sr.CellResult(
            key=cell,
            raw_bytes=spec.get("raw_bytes"),
            error_reason=spec.get("error_reason"),
            parse_status=spec.get("parse_status"),
            parsed=spec.get("parsed"),
        )
    return _fn


# ---------------------------------------------------------------------------
# S01.01 / S01.01a — lifecycle + worker pool persistence
# ---------------------------------------------------------------------------


def test_lifecycle_load_run_unload_one_cycle_per_stage(tmp_path):
    cells = _cells(6)
    pool = sr.load_stage_model(
        cp.STAGE_GEMMA, [0, 1, 2],
        cell_executor=_ok_executor(lambda c: f"prompt for {c.sample_id}".encode()),
    )
    snap_loaded = sr.get_registry_snapshot()
    assert snap_loaded[cp.STAGE_GEMMA]["worker_count"] == 3
    assert snap_loaded[cp.STAGE_GEMMA]["load_counts"] == [1, 1, 1]
    assert all(snap_loaded[cp.STAGE_GEMMA]["is_loaded"])

    dispatch = sr.StageDispatch(
        match_fields=_make_match(cp.STAGE_GEMMA),
        attempt_id="20260426T010005Z-cafe",
        lifecycle=sr.LifecycleEcho("cold", "disk_unloaded"),
        dispatch_cells=cells,
        artifact_root=tmp_path,
        gpu_ids=[0, 1, 2],
    )
    manifest = sr.run_stage_cells(pool, dispatch)
    assert len(manifest["cells"]) == 6
    snap_after_run = sr.get_registry_snapshot()
    # Each worker still loaded exactly once across cells (no per-cell reload).
    assert snap_after_run[cp.STAGE_GEMMA]["load_counts"] == [1, 1, 1]
    # Cells split across workers (not serialized through one worker, not 1-per-cell).
    assert sum(snap_after_run[cp.STAGE_GEMMA]["cells_processed"]) == 6
    assert all(c >= 1 for c in snap_after_run[cp.STAGE_GEMMA]["cells_processed"])

    sr.unload_or_retain_stage_model(cp.STAGE_GEMMA)
    assert sr.get_registry_snapshot() == {}


def test_double_load_rejected(tmp_path):
    sr.load_stage_model(cp.STAGE_FLUX, [0], cell_executor=_ok_executor(lambda c: b"png"))
    with pytest.raises(RuntimeError):
        sr.load_stage_model(cp.STAGE_FLUX, [0], cell_executor=_ok_executor(lambda c: b"png"))
    sr.unload_or_retain_stage_model(cp.STAGE_FLUX)


def test_warm_retain_rejected_in_v022():
    sr.load_stage_model(cp.STAGE_GEMMA, [0], cell_executor=_ok_executor(lambda c: b"x"))
    with pytest.raises(ValueError, match="warm mode disabled in V0.2.2"):
        sr.unload_or_retain_stage_model(cp.STAGE_GEMMA, retain=True)
    sr.unload_or_retain_stage_model(cp.STAGE_GEMMA)


# ---------------------------------------------------------------------------
# S01.06 / S01.07 — manifest identity (match_fields + corpus hash echo)
# ---------------------------------------------------------------------------


def test_stage_manifest_carries_match_fields_and_attempt(tmp_path):
    pool = sr.load_stage_model(
        cp.STAGE_GEMMA, [0],
        cell_executor=_ok_executor(lambda c: b"prompt"),
    )
    dispatch = sr.StageDispatch(
        match_fields=_make_match(cp.STAGE_GEMMA, round_id=2),
        attempt_id="20260426T010005Z-cafe",
        lifecycle=sr.LifecycleEcho("cold", "disk_unloaded"),
        dispatch_cells=_cells(2),
        artifact_root=tmp_path,
        gpu_ids=[0],
    )
    manifest = sr.run_stage_cells(pool, dispatch)
    for field in cp.MATCH_FIELDS:
        assert field in manifest, field
    assert manifest["schema_version"] == "v0.2.2"
    assert manifest["attempt_id"] == "20260426T010005Z-cafe"
    assert manifest["lifecycle_mode"] == "cold"
    assert manifest["lifecycle_state_after"] == "disk_unloaded"
    assert manifest["user_prompt_corpus_hash"] == "u" * 64
    assert "pair_rollups" in manifest and "per_user_prompt" in manifest
    sr.unload_or_retain_stage_model(cp.STAGE_GEMMA)


def test_per_attempt_manifest_path_and_pointer(tmp_path):
    pool = sr.load_stage_model(
        cp.STAGE_FLUX, [0],
        cell_executor=_ok_executor(lambda c: b"\x89PNG\r\n\x1a\nfake"),
    )
    rid = "20260426T010000Z-deadbeef"
    aid = "20260426T010005Z-cafe"
    dispatch = sr.StageDispatch(
        match_fields=_make_match(cp.STAGE_FLUX, run_id=rid, round_id=3),
        attempt_id=aid,
        lifecycle=sr.LifecycleEcho("cold", "disk_unloaded"),
        dispatch_cells=_cells(1),
        artifact_root=tmp_path,
        gpu_ids=[0],
    )
    sr.run_stage_cells(pool, dispatch)
    attempt_path = tmp_path / cp.stage_manifest_attempt_path(rid, "flux", 3, aid)
    pointer_path = tmp_path / cp.stage_manifest_pointer_path(rid, "flux", 3)
    assert attempt_path.is_file()
    # Pointer is either symlink or a byte-copy of the attempt file.
    assert pointer_path.exists()
    # When pointer is a symlink, it must resolve to the attempt path.
    if pointer_path.is_symlink():
        resolved = (pointer_path.parent / os.readlink(pointer_path)).resolve()
        assert resolved == attempt_path.resolve()
    else:
        assert pointer_path.read_bytes() == attempt_path.read_bytes()
    sr.unload_or_retain_stage_model(cp.STAGE_FLUX)


# ---------------------------------------------------------------------------
# S01.06a / S01.11 — write ordering + size/sha verification
# ---------------------------------------------------------------------------


def test_canonical_artifact_size_and_sha_match_on_disk(tmp_path):
    pool = sr.load_stage_model(
        cp.STAGE_FLUX, [0],
        cell_executor=_ok_executor(lambda c: f"image-bytes-{c.sample_id}".encode()),
    )
    dispatch = sr.StageDispatch(
        match_fields=_make_match(cp.STAGE_FLUX),
        attempt_id="20260426T010005Z-cafe",
        lifecycle=sr.LifecycleEcho("cold", "disk_unloaded"),
        dispatch_cells=_cells(2),
        artifact_root=tmp_path,
        gpu_ids=[0],
    )
    manifest = sr.run_stage_cells(pool, dispatch)
    for cell in manifest["cells"]:
        assert cell["status"] == "ok"
        on_disk = tmp_path / cell["artifact_relpath"]
        assert on_disk.is_file()
        assert on_disk.stat().st_size == cell["artifact_size_bytes"]
        assert hashlib.sha256(on_disk.read_bytes()).hexdigest() == cell["artifact_sha256"]
        assert cell["artifact_relpath"].endswith("/result.png")
    sr.unload_or_retain_stage_model(cp.STAGE_FLUX)


def test_executor_failure_yields_error_status_not_ok(tmp_path):
    def _executor(cell, ctx):
        if cell.sample_id == "SAMPLE001":
            return sr.CellResult(key=cell, raw_bytes=None, error_reason="oom")
        return sr.CellResult(key=cell, raw_bytes=b"prompt")
    pool = sr.load_stage_model(cp.STAGE_GEMMA, [0], cell_executor=_executor)
    dispatch = sr.StageDispatch(
        match_fields=_make_match(cp.STAGE_GEMMA),
        attempt_id="20260426T010005Z-cafe",
        lifecycle=sr.LifecycleEcho("cold", "disk_unloaded"),
        dispatch_cells=_cells(2),
        artifact_root=tmp_path,
        gpu_ids=[0],
    )
    manifest = sr.run_stage_cells(pool, dispatch)
    by_sample = {c["sample_id"]: c for c in manifest["cells"]}
    assert by_sample["SAMPLE000"]["status"] == "ok"
    assert by_sample["SAMPLE001"]["status"] == "error"
    assert "artifact_size_bytes" not in by_sample["SAMPLE001"]
    assert "artifact_sha256" not in by_sample["SAMPLE001"]
    assert "oom" in by_sample["SAMPLE001"]["error_reason"]
    sr.unload_or_retain_stage_model(cp.STAGE_GEMMA)


# ---------------------------------------------------------------------------
# S01.05 / S01.10 — Qwen statuses
# ---------------------------------------------------------------------------


def test_qwen_status_derivation_and_rollups(tmp_path):
    spec = {
        "SAMPLE000": {"raw_bytes": b'{"verdict":"yes"}',
                      "parse_status": "ok",
                      "parsed": {"verdict": "yes"}},
        "SAMPLE001": {"raw_bytes": b'{}', "parse_status": "parse_failed"},
        "SAMPLE002": {"raw_bytes": None, "parse_status": "missing"},
        "SAMPLE003": {"raw_bytes": None, "error_reason": "oom"},
    }
    pool = sr.load_stage_model(cp.STAGE_QWEN, [0], cell_executor=_qwen_executor(spec))
    dispatch = sr.StageDispatch(
        match_fields=_make_match(cp.STAGE_QWEN),
        attempt_id="20260426T010005Z-cafe",
        lifecycle=sr.LifecycleEcho("cold", "disk_unloaded"),
        dispatch_cells=_cells(4),
        artifact_root=tmp_path,
        gpu_ids=[0],
    )
    manifest = sr.run_stage_cells(pool, dispatch)
    by_sample = {c["sample_id"]: c for c in manifest["cells"]}
    assert by_sample["SAMPLE000"]["status"] == "ok"
    assert by_sample["SAMPLE000"]["parsed"] == {"verdict": "yes"}
    assert by_sample["SAMPLE001"]["status"] == "parse_failed"
    assert by_sample["SAMPLE002"]["status"] == "missing"
    assert by_sample["SAMPLE003"]["status"] == "error"
    # Rollups partition cells across surviving / non-surviving statuses.
    rollups = manifest["pair_rollups"]
    total = sum(r["total"] for r in rollups.values())
    ok = sum(r["ok"] for r in rollups.values())
    err = sum(r["errors"] for r in rollups.values())
    assert total == 4 and ok == 1 and err == 3
    sr.unload_or_retain_stage_model(cp.STAGE_QWEN)


# ---------------------------------------------------------------------------
# S01.08 / S01.08a — registry inspection + multi-GPU residency
# ---------------------------------------------------------------------------


def test_multi_gpu_residency_one_worker_per_gpu_one_load_each(tmp_path):
    pool = sr.load_stage_model(
        cp.STAGE_GEMMA, [0, 1, 2],
        cell_executor=_ok_executor(lambda c: b"prompt"),
    )
    dispatch = sr.StageDispatch(
        match_fields=_make_match(cp.STAGE_GEMMA),
        attempt_id="20260426T010005Z-cafe",
        lifecycle=sr.LifecycleEcho("cold", "disk_unloaded"),
        dispatch_cells=_cells(9),  # 9 cells / 3 workers = 3 each
        artifact_root=tmp_path,
        gpu_ids=[0, 1, 2],
    )
    sr.run_stage_cells(pool, dispatch)
    snap = sr.get_registry_snapshot()[cp.STAGE_GEMMA]
    assert snap["worker_count"] == 3
    assert snap["gpu_ids"] == [0, 1, 2]
    assert snap["load_counts"] == [1, 1, 1]  # one load per worker, regardless of cell count
    assert sum(snap["cells_processed"]) == 9
    assert all(snap["cells_processed"]) and all(c == 3 for c in snap["cells_processed"])
    sr.unload_or_retain_stage_model(cp.STAGE_GEMMA)


# ---------------------------------------------------------------------------
# S01.09 — canonical artifact paths per stage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stage,filename",
    [
        (cp.STAGE_GEMMA, "intermediate_prompt.txt"),
        (cp.STAGE_FLUX, "result.png"),
        (cp.STAGE_QWEN, "eval.json"),
    ],
)
def test_canonical_artifact_paths_per_stage(tmp_path, stage, filename):
    if stage == cp.STAGE_QWEN:
        executor = _qwen_executor(
            {f"SAMPLE{i:03d}": {"raw_bytes": b'{"verdict":"yes"}',
                                "parse_status": "ok",
                                "parsed": {"verdict": "yes"}}
             for i in range(2)}
        )
    else:
        executor = _ok_executor(lambda c: f"{stage}-bytes".encode())
    pool = sr.load_stage_model(stage, [0], cell_executor=executor)
    dispatch = sr.StageDispatch(
        match_fields=_make_match(stage),
        attempt_id="20260426T010005Z-cafe",
        lifecycle=sr.LifecycleEcho("cold", "disk_unloaded"),
        dispatch_cells=_cells(2),
        artifact_root=tmp_path,
        gpu_ids=[0],
    )
    manifest = sr.run_stage_cells(pool, dispatch)
    for cell in manifest["cells"]:
        assert cell["artifact_relpath"].endswith(f"/{filename}")
        assert (tmp_path / cell["artifact_relpath"]).is_file()
    sr.unload_or_retain_stage_model(stage)


# ---------------------------------------------------------------------------
# Empty dispatch + lifecycle matrix safety
# ---------------------------------------------------------------------------


def test_empty_dispatch_rejected_by_runner(tmp_path):
    pool = sr.load_stage_model(cp.STAGE_GEMMA, [0], cell_executor=_ok_executor(lambda c: b"x"))
    dispatch = sr.StageDispatch(
        match_fields=_make_match(cp.STAGE_GEMMA),
        attempt_id="20260426T010005Z-cafe",
        lifecycle=sr.LifecycleEcho("cold", "disk_unloaded"),
        dispatch_cells=[],
        artifact_root=tmp_path,
        gpu_ids=[0],
    )
    with pytest.raises(ValueError, match="dispatch_cells is empty"):
        sr.run_stage_cells(pool, dispatch)
    sr.unload_or_retain_stage_model(cp.STAGE_GEMMA)


def test_lifecycle_matrix_violation_rejected(tmp_path):
    pool = sr.load_stage_model(cp.STAGE_GEMMA, [0], cell_executor=_ok_executor(lambda c: b"x"))
    dispatch = sr.StageDispatch(
        match_fields=_make_match(cp.STAGE_GEMMA),
        attempt_id="20260426T010005Z-cafe",
        lifecycle=sr.LifecycleEcho("cold", "cpu_prefetched"),  # invalid for cold
        dispatch_cells=_cells(1),
        artifact_root=tmp_path,
        gpu_ids=[0],
    )
    with pytest.raises(ValueError):
        sr.run_stage_cells(pool, dispatch)
    sr.unload_or_retain_stage_model(cp.STAGE_GEMMA)


def test_per_attempt_manifest_payload_is_canonical_json(tmp_path):
    pool = sr.load_stage_model(cp.STAGE_GEMMA, [0], cell_executor=_ok_executor(lambda c: b"x"))
    dispatch = sr.StageDispatch(
        match_fields=_make_match(cp.STAGE_GEMMA, round_id=7),
        attempt_id="20260426T010005Z-cafe",
        lifecycle=sr.LifecycleEcho("cold", "disk_unloaded"),
        dispatch_cells=_cells(1),
        artifact_root=tmp_path,
        gpu_ids=[0],
    )
    sr.run_stage_cells(pool, dispatch)
    attempt_path = tmp_path / cp.stage_manifest_attempt_path(
        dispatch.match_fields.run_id, "gemma", 7, dispatch.attempt_id
    )
    raw = attempt_path.read_bytes()
    # canonical_json is sorted-key, no whitespace, UTF-8 — re-parse must succeed
    payload = json.loads(raw)
    assert payload["schema_version"] == "v0.2.2"
    assert payload["round_id"] == 7
    sr.unload_or_retain_stage_model(cp.STAGE_GEMMA)
