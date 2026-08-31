"""Tests for the Stage 8 review-model adapter (era.orchestration.review_adapter).

These prove the web app's input is normalized correctly for *any* iteration —
an editing task, a generation task, an iteration whose agent-authored
scores.jsonl is malformed, and one with missing image files — not just the demo.
"""

from __future__ import annotations

import json
from pathlib import Path

from era.config import ERAConfig
from era.orchestration.review_adapter import (
    build_and_write_review_model,
    build_review_model_json,
    derive_human_labels_from_model,
    load_review_model,
    review_model_path,
)
from test_human_feedback import make_feedback_workspace


def _build(ws) -> dict:
    cfg = ERAConfig.from_yaml(ws.root / "config.yaml")
    return build_review_model_json(ws.root, ws.root / "iter_001", cfg)


def test_review_model_editing_shape(tmp_path: Path):
    ws, _ = make_feedback_workspace(tmp_path)
    rm = _build(ws)
    assert rm["schema_version"] and rm["generated_at"]
    assert rm["task_family"] == "editing"
    assert len(rm["configs"]) == 2
    # every config carries a task-agnostic render hint
    assert all("score_display" in c for c in rm["configs"])
    sample = rm["samples"][0]
    assert "cells" in sample and "family_b_rankings" in sample
    # image availability is annotated for the UI
    assert all("output_available" in cell for cell in sample["cells"])
    assert rm["adapter"]["deterministic_ok"] is True
    assert rm["adapter"]["subagent_ran"] is False


def test_review_model_generation_task(tmp_path: Path):
    ws, _ = make_feedback_workspace(
        tmp_path, task_family="generation", input_roles={},
        sample_metadata={"prompt": "a red fox in deep snow"})
    rm = _build(ws)
    assert rm["task_family"] == "generation"
    sample = rm["samples"][0]
    assert sample["input"].get("prompt") == "a red fox in deep snow"
    assert "images" not in sample["input"]


def test_review_model_tolerates_malformed_scores(tmp_path: Path):
    """A runner that emits sample_id / a string score is still rendered."""
    ws, _ = make_feedback_workspace(tmp_path)
    scores = (ws.root / "iter_001" / "experiments" / "results" / "full"
              / "vlm-pointwise-72b" / "scores.jsonl")
    rows = [
        {"sample_id": "sample_001", "method_id": "method_a", "score": "0.8",
         "sub_scores": {"quality": 4}, "scope": "pointwise", "ok": True},
        {"sample_id": "sample_001", "method_id": "method_b", "score": "bad",
         "sub_scores": {}, "scope": "pointwise", "ok": True},
    ]
    scores.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                      encoding="utf-8")
    rm = _build(ws)
    # the non-numeric score is reported as a warning, not silently dropped
    assert any("non-numeric" in w for w in rm["warnings"])
    # the sample_id row was still picked up, and the string "0.8" still renders
    sample = next(s for s in rm["samples"] if s["sample_key"] == "sample_001")
    cell = next(c for c in sample["cells"] if c["method_id"] == "method_a")
    judge = next(j for j in cell["judges"]
                 if j["combination_id"] == "vlm-pointwise-72b")
    assert "quality" in judge["display"]


def test_review_model_flags_missing_images(tmp_path: Path):
    ws, _ = make_feedback_workspace(tmp_path)
    (tmp_path / "dataset" / "method_a" / "sample_001" / "out.png").unlink()
    rm = _build(ws)
    assert any("missing on disk" in w for w in rm["warnings"])
    sample = next(s for s in rm["samples"] if s["sample_key"] == "sample_001")
    cell = next(c for c in sample["cells"] if c["method_id"] == "method_a")
    assert cell["output_available"] is False


def test_review_model_normalizes_prefixed_sample_keys(tmp_path: Path):
    """Prefixed ``"<method_id>/sample_NNN"`` keys must resolve images cleanly.

    The fixture lays images at the unprefixed path
    ``<method_base>/sample_NNN/<file>`` (the on-disk reality). With the
    deterministic strip, ``warnings`` must carry no missing-image entry and
    every cell's ``output_available`` must be ``True``.
    """
    ws, iter_dir = make_feedback_workspace(tmp_path)
    results = iter_dir / "experiments" / "results" / "full"
    for cid, scope, base in (("vlm-pointwise-72b", "pointwise-1to5", 0.7),
                             ("region-metric-baseline", "region", -0.3)):
        rows = []
        for sk in ("sample_001", "sample_002"):
            for i, mid in enumerate(("method_a", "method_b")):
                rows.append(json.dumps({
                    "sample_key": f"{mid}/{sk}", "method_id": mid,
                    "score": base + 0.1 * i, "sub_scores": {}, "scope": scope,
                    "ok": True,
                }))
        (results / cid / "scores.jsonl").write_text(
            "\n".join(rows) + "\n", encoding="utf-8")

    rm = _build(ws)
    assert not any("missing on disk" in w for w in rm["warnings"])
    assert [s["sample_key"] for s in rm["samples"]] == ["sample_001", "sample_002"]
    for sample in rm["samples"]:
        for cell in sample["cells"]:
            assert cell["output_available"] is True


def test_build_write_and_load_roundtrip(tmp_path: Path):
    ws, _ = make_feedback_workspace(tmp_path)
    result = build_and_write_review_model(str(ws.root), iteration=1)
    assert result["status"] == "ok"
    iter_dir = ws.root / "iter_001"
    assert review_model_path(iter_dir).is_file()
    rm = load_review_model(iter_dir)
    assert rm is not None and rm["iteration_dir"] == "iter_001"


def test_load_review_model_absent(tmp_path: Path):
    ws, _ = make_feedback_workspace(tmp_path)
    assert load_review_model(ws.root / "iter_001") is None


def test_derive_human_labels_from_model(tmp_path: Path):
    ws, _ = make_feedback_workspace(tmp_path)
    rm = _build(ws)
    feedback = {"item_marks": [], "comparison_marks": [], "general_feedback": ""}
    labels = derive_human_labels_from_model(rm, feedback)
    assert labels["default_rule"] == "unmarked_is_correct"
    # 1 flagging config x 2 samples x 2 methods = 4 cells, all default-endorsed
    assert len(labels["labels"]) == 4
    assert all(lbl["human_verdict"] == "correct" for lbl in labels["labels"])
    # 1 comparison config x 2 samples = 2 rankings
    assert len(labels["comparison_labels"]) == 2
