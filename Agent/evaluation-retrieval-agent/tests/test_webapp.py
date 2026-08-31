"""Tests for the ERA Stage 8 human-feedback web app (FastAPI)."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from era.webapp.app import build_app
from test_human_feedback import make_feedback_workspace


def _client(tmp_path: Path) -> TestClient:
    ws, _ = make_feedback_workspace(tmp_path)
    return TestClient(build_app(str(ws.root), iteration=1))


def test_health(tmp_path: Path):
    res = _client(tmp_path).get("/api/health")
    assert res.status_code == 200
    assert res.json()["mode"] == "full"


def test_review_payload_shape(tmp_path: Path):
    data = _client(tmp_path).get("/api/review").json()
    assert data["task_family"] == "editing"
    assert len(data["configs"]) == 2
    # every config carries a plain-language description
    assert all(c["description"] for c in data["configs"])
    sample = data["samples"][0]
    assert "input" in sample and "cells" in sample
    # Family-A/hybrid render as per-cell judges; Family-B as rankings
    assert sample["cells"][0]["judges"][0]["combination_id"] == "vlm-pointwise-72b"
    assert sample["family_b_rankings"][0]["combination_id"] == "region-metric-baseline"
    assert len(sample["family_b_rankings"][0]["ranked_method_ids"]) == 2


def test_review_payload_strips_method_prefix_from_sample_key(tmp_path: Path):
    """A runner that prefixed sample_key with the method dir still merges cells.

    Real Stage 6 runners have shipped rows like ``"method_a/sample_001"`` —
    the build must strip the ``"<method_id>/"`` prefix so cells across methods
    group into one sample (not N method-private samples) and the image
    resolver finds files under ``<method_base>/sample_NNN/``.
    """
    ws, iter_dir = make_feedback_workspace(tmp_path)
    results = iter_dir / "experiments" / "results" / "full"
    for cid, scope, base in (("vlm-pointwise-72b", "pointwise-1to5", 0.7),
                             ("region-metric-baseline", "region", -0.3)):
        rows = []
        for sk in ("sample_001", "sample_002"):
            for i, mid in enumerate(("method_a", "method_b")):
                rows.append(json.dumps({
                    # prefixed sample_key — the bug we're fixing
                    "sample_key": f"{mid}/{sk}", "method_id": mid,
                    "score": base + 0.1 * i, "sub_scores": {}, "scope": scope,
                    "ok": True,
                }))
        (results / cid / "scores.jsonl").write_text(
            "\n".join(rows) + "\n", encoding="utf-8")

    data = TestClient(build_app(str(ws.root), iteration=1)).get("/api/review").json()
    # 2 samples (sample_001, sample_002), not 4 (method_a/sample_001, …).
    assert [s["sample_key"] for s in data["samples"]] == ["sample_001", "sample_002"]
    sample = data["samples"][0]
    # cells from both methods merged into the one sample
    assert {c["method_id"] for c in sample["cells"]} == {"method_a", "method_b"}
    # Family-B aggregates per sample as one ranking of both methods
    assert len(sample["family_b_rankings"][0]["ranked_method_ids"]) == 2


def test_review_payload_generation_prompt(tmp_path: Path):
    """A generation workspace surfaces the per-sample prompt, no input images."""
    ws, _ = make_feedback_workspace(
        tmp_path, task_family="generation", input_roles={},
        sample_metadata={"prompt": "a red fox in deep snow"})
    data = TestClient(build_app(str(ws.root), iteration=1)).get("/api/review").json()
    assert data["task_family"] == "generation"
    sample = data["samples"][0]
    assert sample["input"]["prompt"] == "a red fox in deep snow"
    assert "images" not in sample["input"]


def test_review_payload_editing_instruction(tmp_path: Path):
    """An editing sample with an instruction surfaces both the text and images."""
    ws, _ = make_feedback_workspace(
        tmp_path, sample_metadata={"instruction": "make the sky stormy"})
    data = TestClient(build_app(str(ws.root), iteration=1)).get("/api/review").json()
    sample = data["samples"][0]
    assert sample["input"]["prompt"] == "make the sky stormy"
    assert len(sample["input"]["images"]) == 2


def test_image_traversal_rejected(tmp_path: Path):
    client = _client(tmp_path)
    res = client.get("/api/image", params={
        "method": "method_a", "sample": "../../etc", "role": "output"})
    assert res.status_code == 404


def test_image_served(tmp_path: Path):
    client = _client(tmp_path)
    res = client.get("/api/image", params={
        "method": "method_a", "sample": "sample_001", "role": "input_cloth"})
    assert res.status_code == 200


def test_item_and_comparison_then_finalize(tmp_path: Path):
    ws, iter_dir = make_feedback_workspace(tmp_path)
    client = TestClient(build_app(str(ws.root), iteration=1))

    r = client.put("/api/feedback/item", json={
        "sample_key": "sample_001", "method_id": "method_a",
        "combination_id": "vlm-pointwise-72b", "label": "wrong",
        "comment": "bad call"})
    assert r.status_code == 200
    assert len(r.json()["feedback"]["item_marks"]) == 1

    r = client.put("/api/feedback/comparison", json={
        "sample_key": "sample_001", "combination_id": "region-metric-baseline",
        "label": "wrong"})
    assert r.status_code == 200
    assert len(r.json()["feedback"]["comparison_marks"]) == 1

    r = client.post("/api/feedback/finalize", json={})
    assert r.status_code == 200
    labels = iter_dir / "human" / "human_labels.json"
    assert labels.is_file()
    payload = r.json()
    assert payload["labels"]["flagging_label_count"] == 4
    assert payload["labels"]["comparison_label_count"] == 2


def test_finalize_twice_conflicts(tmp_path: Path):
    client = _client(tmp_path)
    assert client.post("/api/feedback/finalize", json={}).status_code == 200
    assert client.post("/api/feedback/finalize", json={}).status_code == 409
    # an explicit reopen is accepted
    assert client.post(
        "/api/feedback/finalize", json={"reopen": True}).status_code == 200


def test_overview(tmp_path: Path):
    data = _client(tmp_path).get("/api/overview").json()
    assert data["summary"]["complete"] is True


def test_review_served_from_normalized_model(tmp_path: Path):
    """When review_model.json exists, /api/review serves the normalized artifact."""
    from era.orchestration.review_adapter import build_and_write_review_model

    ws, _ = make_feedback_workspace(tmp_path)
    build_and_write_review_model(str(ws.root), iteration=1)
    data = TestClient(build_app(str(ws.root), iteration=1)).get("/api/review").json()
    # the normalized artifact carries schema_version + warnings; the raw
    # fallback payload does not — so this confirms the adapter path was used
    assert data.get("schema_version")
    assert "warnings" in data and "adapter" in data
    assert "feedback" in data and "samples" in data
