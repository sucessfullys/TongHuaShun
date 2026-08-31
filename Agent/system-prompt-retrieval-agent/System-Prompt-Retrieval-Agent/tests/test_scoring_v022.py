"""S06 tests for V0.2.2 scoring wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from system_prompt_retrieval_agent.remote._vendored import canonical_paths as cp
from system_prompt_retrieval_agent.scoring_v022 import (
    EvalCell,
    FailedCellPolicy,
    NoRankablePairsError,
    ScoringInputMissing,
    apply_failed_cell_policy,
    assert_some_pairs_rankable,
    build_evaluation_cells,
    partition_pairs_by_scoring_input,
    reject_empty_evaluation_per_pair,
)


_RID = "20260426T010000Z-deadbeef"


def _seed(artifact_root: Path, run_id: str, round_id: int, stage: str,
          pid: str, upid: str, sid: str, payload: bytes) -> str:
    relpath = cp.cell_artifact_path(run_id, stage, round_id, pid, upid, sid)
    full = artifact_root / relpath
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_bytes(payload)
    return relpath


def _merged(stage, cells_records):
    return {
        "schema_version": cp.SCHEMA_VERSION, "run_id": _RID, "round_id": 0,
        "stage": stage, "config_hash": "c" * 64,
        "user_prompt_corpus_id": "v0", "user_prompt_corpus_hash": "u" * 64,
        "prompt_pair_corpus_hash": "p" * 64, "sample_corpus_hash": "s" * 64,
        "cells": cells_records,
    }


def _ok(pid, upid, sid, status="ok"):
    return {"prompt_pair_id": pid, "user_prompt_id": upid, "sample_id": sid,
            "status": status, "artifact_relpath": "x",
            "artifact_size_bytes": 1, "artifact_sha256": "a" * 64}


# ---------------------------------------------------------------------------
# S06.01 — eval cell construction
# ---------------------------------------------------------------------------


def test_build_evaluation_cells_intersects_three_stages(tmp_path):
    _seed(tmp_path, _RID, 0, "gemma", "PP1", "UP1", "S1", b"prompt-text")
    _seed(tmp_path, _RID, 0, "flux", "PP1", "UP1", "S1", b"\x89PNG-fake")
    _seed(tmp_path, _RID, 0, "qwen", "PP1", "UP1", "S1",
          b'{"verdict":"yes","score":0.9}')
    eval_cells = build_evaluation_cells(
        artifact_root=tmp_path, run_id=_RID, round_id=0,
        qwen_merged=_merged("qwen", [_ok("PP1", "UP1", "S1")]),
        flux_merged=_merged("flux", [_ok("PP1", "UP1", "S1")]),
        gemma_merged=_merged("gemma", [_ok("PP1", "UP1", "S1")]),
        sample_lookup={"S1": {"model_image_path": "/m.png", "cloth_image_path": "/c.png"}},
    )
    assert len(eval_cells) == 1
    cell = eval_cells[0]
    assert cell.intermediate_prompt == "prompt-text"
    assert cell.generated_image_path.endswith("/result.png")
    assert cell.qwen_eval_json_path.endswith("/eval.json")
    assert cell.model_image_path == "/m.png"


def test_build_evaluation_cells_drops_pair_missing_one_stage(tmp_path):
    _seed(tmp_path, _RID, 0, "gemma", "PP1", "UP1", "S1", b"x")
    _seed(tmp_path, _RID, 0, "flux", "PP1", "UP1", "S1", b"x")
    # Qwen missing for this cell
    eval_cells = build_evaluation_cells(
        artifact_root=tmp_path, run_id=_RID, round_id=0,
        qwen_merged=_merged("qwen", []),
        flux_merged=_merged("flux", [_ok("PP1", "UP1", "S1")]),
        gemma_merged=_merged("gemma", [_ok("PP1", "UP1", "S1")]),
        sample_lookup={"S1": {}},
    )
    assert eval_cells == []


def test_build_evaluation_cells_skips_unsuccessful_status(tmp_path):
    _seed(tmp_path, _RID, 0, "gemma", "PP1", "UP1", "S1", b"x")
    _seed(tmp_path, _RID, 0, "flux", "PP1", "UP1", "S1", b"x")
    _seed(tmp_path, _RID, 0, "qwen", "PP1", "UP1", "S1", b"x")
    eval_cells = build_evaluation_cells(
        artifact_root=tmp_path, run_id=_RID, round_id=0,
        qwen_merged=_merged("qwen", [_ok("PP1", "UP1", "S1", status="parse_failed")]),
        flux_merged=_merged("flux", [_ok("PP1", "UP1", "S1")]),
        gemma_merged=_merged("gemma", [_ok("PP1", "UP1", "S1")]),
        sample_lookup={"S1": {}},
    )
    assert eval_cells == []


# ---------------------------------------------------------------------------
# S06.02 — empty per-pair eval rejected
# ---------------------------------------------------------------------------


def test_qwen_surviving_pair_with_no_eval_cells_aborts():
    with pytest.raises(ScoringInputMissing, match="zero evaluation cells"):
        reject_empty_evaluation_per_pair(
            eval_cells=[],
            qwen_surviving_pairs=["PP1"],
        )


def test_qwen_surviving_pair_with_eval_cells_groups_by_pair():
    cells = [
        EvalCell("PP1", "UP1", "S1", "p", "/g.png", "/q.json", "/m.png", "/c.png"),
        EvalCell("PP1", "UP1", "S2", "p", "/g.png", "/q.json", "/m.png", "/c.png"),
        EvalCell("PP2", "UP1", "S1", "p", "/g.png", "/q.json", "/m.png", "/c.png"),
    ]
    out = reject_empty_evaluation_per_pair(cells, qwen_surviving_pairs=["PP1", "PP2"])
    assert len(out["PP1"]) == 2 and len(out["PP2"]) == 1


# ---------------------------------------------------------------------------
# S06.05 — failed-cell policy
# ---------------------------------------------------------------------------


def test_failed_cell_policy_skip_drops_failed_results():
    results = [
        {"sample_id": "S1", "score": 0.8},
        {"sample_id": "S2", "error": "vlm_timeout"},
        {"sample_id": "S3", "status": "error"},
    ]
    out = apply_failed_cell_policy(results, FailedCellPolicy(skip=True))
    assert len(out) == 1 and out[0]["sample_id"] == "S1"


def test_failed_cell_policy_penalty_replaces_score():
    results = [{"sample_id": "S2", "error": "vlm_timeout", "score": 0.9}]
    out = apply_failed_cell_policy(results, FailedCellPolicy(skip=False, penalty_score=0.0))
    assert out[0]["score"] == 0.0


# ---------------------------------------------------------------------------
# S06.09 — failed_pairs[] partition + abort
# ---------------------------------------------------------------------------


def test_partition_pairs_by_scoring_input():
    cells = [EvalCell("PP1", "UP1", "S1", "", "", "", "", "")]
    grouped = {"PP1": cells, "PP2": []}
    partition = partition_pairs_by_scoring_input(
        qwen_surviving_pairs=["PP1", "PP2"],
        eval_cells_by_pair=grouped,
    )
    assert partition.rankable_pairs == ["PP1"]
    assert partition.failed_pairs == [
        {"prompt_pair_id": "PP2", "failure_reason": "missing_scoring_input"}
    ]


def test_assert_some_pairs_rankable_aborts_when_empty():
    from system_prompt_retrieval_agent.scoring_v022 import ScoringPartition
    p = ScoringPartition(rankable_pairs=[], failed_pairs=[
        {"prompt_pair_id": "PP1", "failure_reason": "missing_scoring_input"}
    ])
    with pytest.raises(NoRankablePairsError, match="no rankable pairs"):
        assert_some_pairs_rankable(p)


def test_assert_some_pairs_rankable_passes_when_nonempty():
    from system_prompt_retrieval_agent.scoring_v022 import ScoringPartition
    p = ScoringPartition(rankable_pairs=["PP1"], failed_pairs=[])
    assert_some_pairs_rankable(p)  # no raise


# ---------------------------------------------------------------------------
# Sub-score / per-category propagation (post-bugfix coverage)
# ---------------------------------------------------------------------------

from system_prompt_retrieval_agent.scoring_v022 import (  # noqa: E402
    build_pair_ranking_rows,
)


def _cell(pid, upid, sid):
    return EvalCell(
        prompt_pair_id=pid, user_prompt_id=upid, sample_id=sid,
        intermediate_prompt="x", generated_image_path="x",
        qwen_eval_json_path="x", model_image_path="x", cloth_image_path="x",
    )


def test_pair_ranking_row_propagates_subscore_means():
    cells = {
        "PP1": [
            _cell("PP1", "UP", "dress__a__01"),
            _cell("PP1", "UP", "dress__a__02"),
            _cell("PP1", "UP", "lower__b__01"),
        ]
    }
    results = {
        "PP1": [
            {"user_prompt_id": "UP", "sample_id": "dress__a__01", "score": 0.7,
             "qwen_pass_rate": 0.80, "edit_correctness": 0.70,
             "garment_transfer_correctness": 0.65, "preservation": 0.85,
             "artifact_penalty": 0.05},
            {"user_prompt_id": "UP", "sample_id": "dress__a__02", "score": 0.6,
             "qwen_pass_rate": 0.60, "edit_correctness": 0.50,
             "garment_transfer_correctness": 0.55, "preservation": 0.75,
             "artifact_penalty": 0.10},
            {"user_prompt_id": "UP", "sample_id": "lower__b__01", "score": 0.9,
             "qwen_pass_rate": 1.00, "edit_correctness": 0.90,
             "garment_transfer_correctness": 0.95, "preservation": 0.95,
             "artifact_penalty": 0.00},
        ]
    }
    rows = build_pair_ranking_rows(
        run_id=_RID, round_id=1,
        eval_cells_by_pair=cells, cell_results_by_pair=results,
    )
    assert len(rows) == 1
    r = rows[0]
    assert r.mean_qwen_pass_rate == pytest.approx((0.80 + 0.60 + 1.00) / 3)
    assert r.mean_edit_correctness == pytest.approx((0.70 + 0.50 + 0.90) / 3)
    assert r.mean_garment_transfer_correctness == pytest.approx((0.65 + 0.55 + 0.95) / 3)
    assert r.mean_preservation == pytest.approx((0.85 + 0.75 + 0.95) / 3)
    assert r.mean_artifact_penalty == pytest.approx((0.05 + 0.10 + 0.00) / 3)
    expected_total = (
        r.mean_qwen_pass_rate * 0.40
        + r.mean_edit_correctness * 0.20
        + r.mean_garment_transfer_correctness * 0.15
        + r.mean_preservation * 0.15
        + r.mean_artifact_penalty * (-0.10)
    )
    assert r.total_score == pytest.approx(expected_total)
    d = r.as_dict()
    for k in ("mean_qwen_pass_rate", "mean_edit_correctness",
              "mean_garment_transfer_correctness", "mean_preservation",
              "mean_artifact_penalty", "total_score", "per_category",
              "worst_cells"):
        assert k in d


def test_pair_ranking_row_per_category_bucketing_preserves_real_qwen():
    cells = {
        "PP1": [
            _cell("PP1", "UP", "dress__a__01"),
            _cell("PP1", "UP", "lower__b__01"),
            _cell("PP1", "UP", "upper__c__01"),
            _cell("PP1", "UP", "legacy_id_no_double_underscore"),
        ]
    }
    results = {
        "PP1": [
            {"user_prompt_id": "UP", "sample_id": "dress__a__01", "score": 0.7,
             "qwen_pass_rate": 0.80, "edit_correctness": 0.70,
             "garment_transfer_correctness": 0.65, "preservation": 0.85,
             "artifact_penalty": 0.05},
            {"user_prompt_id": "UP", "sample_id": "lower__b__01", "score": 0.5,
             "qwen_pass_rate": 0.50, "edit_correctness": 0.40,
             "garment_transfer_correctness": 0.45, "preservation": 0.60,
             "artifact_penalty": 0.20},
            {"user_prompt_id": "UP", "sample_id": "upper__c__01", "score": 0.9,
             "qwen_pass_rate": 1.00, "edit_correctness": 0.95,
             "garment_transfer_correctness": 0.90, "preservation": 0.90,
             "artifact_penalty": 0.00},
            {"user_prompt_id": "UP", "sample_id": "legacy_id_no_double_underscore",
             "score": 0.3,
             "qwen_pass_rate": 0.30, "edit_correctness": 0.30,
             "garment_transfer_correctness": 0.30, "preservation": 0.30,
             "artifact_penalty": 0.30},
        ]
    }
    rows = build_pair_ranking_rows(
        run_id=_RID, round_id=1,
        eval_cells_by_pair=cells, cell_results_by_pair=results,
    )
    r = rows[0]
    # All 4 cells contribute to top-level pair_overall + means.
    assert r.n_cells == 4
    assert r.mean_qwen_pass_rate == pytest.approx((0.80 + 0.50 + 1.00 + 0.30) / 4)
    # Per-category: each bucket has exactly 1 cell, malformed cell excluded.
    assert r.per_category["dress"]["n_cells"] == 1
    assert r.per_category["lower"]["n_cells"] == 1
    assert r.per_category["upper"]["n_cells"] == 1
    # CRITICAL: dress qwen_pass_rate must equal the raw 0.80, NOT 1.0
    # (would happen if we routed through qwen_status="yes" indirection).
    assert r.per_category["dress"]["qwen_pass_rate"] == pytest.approx(0.80)
    assert r.per_category["lower"]["qwen_pass_rate"] == pytest.approx(0.50)
    assert r.per_category["upper"]["qwen_pass_rate"] == pytest.approx(1.00)
    # Each category gets a weighted score derived from the same weights.
    for cat in ("dress", "lower", "upper"):
        assert r.per_category[cat]["weighted_score"] is not None
        assert r.per_category[cat]["missing_score_reason"] is None


def test_per_category_emits_no_cells_in_category_when_empty():
    cells = {"PP1": [_cell("PP1", "UP", "dress__a__01")]}
    results = {
        "PP1": [{
            "user_prompt_id": "UP", "sample_id": "dress__a__01", "score": 0.7,
            "qwen_pass_rate": 0.80, "edit_correctness": 0.70,
            "garment_transfer_correctness": 0.65, "preservation": 0.85,
            "artifact_penalty": 0.05,
        }],
    }
    rows = build_pair_ranking_rows(
        run_id=_RID, round_id=1,
        eval_cells_by_pair=cells, cell_results_by_pair=results,
    )
    r = rows[0]
    assert r.per_category["lower"] == {
        "n_cells": 0, "missing_score_reason": "no_cells_in_category",
    }
    assert r.per_category["upper"] == {
        "n_cells": 0, "missing_score_reason": "no_cells_in_category",
    }


def test_worst_cells_emitted_unconditionally_and_capped():
    # 15 cells, all with overall above 0.5 — no hard threshold should apply.
    n = 15
    cells = {"PP1": [_cell("PP1", "UP", f"dress__a__{i:02d}") for i in range(n)]}
    results = {
        "PP1": [
            {"user_prompt_id": "UP", "sample_id": f"dress__a__{i:02d}",
             "score": 0.55 + 0.02 * i,  # 0.55..0.83
             "qwen_pass_rate": 0.55 + 0.02 * i,
             "edit_correctness": 0.6, "garment_transfer_correctness": 0.6,
             "preservation": 0.6, "artifact_penalty": 0.05}
            for i in range(n)
        ],
    }
    rows = build_pair_ranking_rows(
        run_id=_RID, round_id=1,
        eval_cells_by_pair=cells, cell_results_by_pair=results,
    )
    r = rows[0]
    assert len(r.worst_cells) == 10
    overalls = [c["overall"] for c in r.worst_cells]
    assert overalls == sorted(overalls)
    # Bottom-N policy: smallest overall (0.55) is included even though > 0.5.
    assert r.worst_cells[0]["overall"] == pytest.approx(0.55)
    assert r.worst_cells[0]["category"] == "dress"
