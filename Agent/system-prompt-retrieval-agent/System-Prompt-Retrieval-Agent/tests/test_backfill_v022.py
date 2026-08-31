"""Tests for scripts/backfill_v022_aggregates.py — V0.2.2 in-place
re-aggregation of sub-scores from on-disk eval.json files."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from system_prompt_retrieval_agent.remote._vendored import canonical_paths as cp


def _load_backfill_module():
    repo_root = Path(__file__).resolve().parents[2]
    src = repo_root / "System-Prompt-Retrieval-Agent" / "scripts" / "backfill_v022_aggregates.py"
    spec = importlib.util.spec_from_file_location("backfill_v022_aggregates", src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("backfill_v022_aggregates", mod)
    spec.loader.exec_module(mod)
    return mod


def _seed_cell(artifact_root: Path, run_id: str, round_id: int,
               pid: str, upid: str, sid: str, eval_payload: dict):
    for stage, body in (
        ("gemma", b"intermediate prompt text"),
        ("flux", b"\x89PNG-fake"),
    ):
        p = artifact_root / cp.cell_artifact_path(run_id, stage, round_id, pid, upid, sid)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(body)
    qp = artifact_root / cp.cell_artifact_path(run_id, "qwen", round_id, pid, upid, sid)
    qp.parent.mkdir(parents=True, exist_ok=True)
    qp.write_text(json.dumps(eval_payload), encoding="utf-8")


def test_backfill_idempotent(tmp_path):
    backfill = _load_backfill_module()

    run_id = "20260427T020000Z-cafebabe"
    artifact_root = tmp_path / "live"
    run_root = tmp_path / "live" / "runs"
    memory_root = tmp_path / "mem"
    round_dir = run_root / run_id / "rounds" / "round_001"
    round_dir.mkdir(parents=True, exist_ok=True)

    samples_by_cat = {
        "dress": ["dress__a__01", "dress__a__02"],
        "lower": ["lower__b__01"],
        "upper": ["upper__c__01"],
    }
    for cat, sids in samples_by_cat.items():
        for sid in sids:
            _seed_cell(
                artifact_root, run_id, 1, "PP1", "UP_FIXED", sid,
                {"qwen_pass_rate": 0.8, "edit_correctness": 0.7,
                 "garment_transfer_correctness": 0.65, "preservation": 0.85,
                 "artifact_penalty": 0.05},
            )

    weights = {"qwen_pass_rate": 0.40, "edit_correctness": 0.20,
               "garment_transfer_correctness": 0.15, "preservation": 0.15,
               "artifact_penalty": -0.10}

    stats1 = backfill.backfill_round(
        artifact_root=artifact_root, run_root=run_root,
        memory_root=memory_root, run_id=run_id, round_id=1,
        round_dir=round_dir, weights=weights,
    )
    assert stats1["n_pairs"] == 1
    assert stats1["n_cells"] == 4

    ranking_path = round_dir / "scoring" / "ranking_v022.json"
    failed_cells_path = round_dir / "failures" / "failed_cells.json"
    long_mem_path = memory_root / "long_memory.csv"
    assert ranking_path.is_file()
    assert failed_cells_path.is_file()
    assert long_mem_path.is_file()

    ranking_bytes_1 = ranking_path.read_bytes()
    ranking_payload_1 = json.loads(ranking_bytes_1)
    r0 = ranking_payload_1[0]
    assert r0["mean_qwen_pass_rate"] == pytest.approx(0.8)
    # Per-category direct means (NOT routed through qwen_status).
    assert r0["per_category"]["dress"]["qwen_pass_rate"] == pytest.approx(0.8)
    assert r0["per_category"]["dress"]["n_cells"] == 2

    rows1 = list(csv.DictReader(long_mem_path.open()))
    live1 = [r for r in rows1 if not r.get("archived_by_run_id")]
    assert len(live1) == 1
    assert live1[0]["qwen_pass_rate"] not in (None, "")
    assert live1[0]["dress_weighted_score"] not in (None, "")
    assert live1[0]["missing_score_reason"] in (None, "")
    assert live1[0]["dress_missing_score_reason"] in (None, "")

    # --- second pass: idempotence ---
    stats2 = backfill.backfill_round(
        artifact_root=artifact_root, run_root=run_root,
        memory_root=memory_root, run_id=run_id, round_id=1,
        round_dir=round_dir, weights=weights,
    )
    assert stats2 == stats1
    assert ranking_path.read_bytes() == ranking_bytes_1

    rows2 = list(csv.DictReader(long_mem_path.open()))
    live2 = [r for r in rows2 if not r.get("archived_by_run_id")]
    archived2 = [r for r in rows2 if r.get("archived_by_run_id")]
    # Upsert key (run_id, round_id, prompt_pair_id): exactly one live row,
    # the prior row archived (not duplicated).
    assert len(live2) == 1
    assert len(archived2) == 1
    assert archived2[0]["archived_by_run_id"] == run_id


def test_backfill_handles_legacy_sample_id_without_double_underscore(tmp_path):
    backfill = _load_backfill_module()
    run_id = "20260427T030000Z-deadbeef"
    artifact_root = tmp_path / "live"
    run_root = tmp_path / "live" / "runs"
    memory_root = tmp_path / "mem"
    round_dir = run_root / run_id / "rounds" / "round_001"
    round_dir.mkdir(parents=True, exist_ok=True)

    _seed_cell(artifact_root, run_id, 1, "PP1", "UP", "legacy_id",
               {"qwen_pass_rate": 0.5, "edit_correctness": 0.5,
                "garment_transfer_correctness": 0.5, "preservation": 0.5,
                "artifact_penalty": 0.1})

    weights = {"qwen_pass_rate": 0.40, "edit_correctness": 0.20,
               "garment_transfer_correctness": 0.15, "preservation": 0.15,
               "artifact_penalty": -0.10}
    stats = backfill.backfill_round(
        artifact_root=artifact_root, run_root=run_root,
        memory_root=memory_root, run_id=run_id, round_id=1,
        round_dir=round_dir, weights=weights,
    )
    # The legacy cell still contributes to top-level means, but no category.
    assert stats["n_cells"] == 1
    payload = json.loads((round_dir / "scoring" / "ranking_v022.json").read_text())
    assert payload[0]["mean_qwen_pass_rate"] == pytest.approx(0.5)
    for cat in ("dress", "lower", "upper"):
        assert payload[0]["per_category"][cat]["n_cells"] == 0
        assert payload[0]["per_category"][cat]["missing_score_reason"] == "no_cells_in_category"
