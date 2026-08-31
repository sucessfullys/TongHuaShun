"""Tests for ERA Phase C-2 — pass/recall auto-validation gate.

Cover the deterministic backbone (``era.orchestration.auto_validate``)
end-to-end. The sub-agent itself is not unit-tested (it's an LLM
call); we stub its judgments on disk and exercise prepare → finalize
round-trip.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import yaml

from era.cli import main
from era.orchestration.auto_validate import (
    aggregate_judgments,
    build_batches,
    load_full_annotations,
    read_result,
    AUTO_VALIDATE_REL,
    INPUTS_REL,
    JUDGMENTS_REL,
    RESULT_REL,
)
from era.workspace import Workspace


# ---- fixtures -----------------------------------------------------------

def _scaffold_dataset(tmp_path: Path) -> Path:
    """Build a minimal dataset with three samples + an annotations dir."""
    data_root = tmp_path / "data"
    method_a = data_root / "method_a"
    method_b = data_root / "method_b"
    method_a.mkdir(parents=True)
    method_b.mkdir(parents=True)
    for sample_dir in (method_a / "s1", method_a / "s2", method_a / "s3",
                       method_b / "s1", method_b / "s2", method_b / "s3"):
        sample_dir.mkdir(parents=True)
        (sample_dir / "out.png").write_bytes(b"\x89PNG\r\n")
    return data_root


def _write_annotation(
    data_root: Path, sample_key: str, per_method: dict[str, str],
) -> Path:
    """Write a central annotation file (mirroring era.annotate.store)."""
    path = data_root / "annotations" / f"{sample_key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "2.0",
        "sample_key": sample_key,
        "per_method": per_method,
        "created_at": "2026-05-27T00:00:00+00:00",
        "updated_at": "2026-05-27T00:00:00+00:00",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _make_workspace_with_min_passing(
    base: Path, data_root: Path, *,
    name: str,
    chosen_configs: list[dict],
    min_passing: int,
    sample_count: int,
) -> Workspace:
    """Build a minimal workspace at iter_001 with config.yaml + brief,
    parameterised by Phase C-2.5 ``auto_validate_min_passing``.

    Shared scaffolding for the two auto-validate test fixtures:
    legacy "any 1 passes" semantics (``min_passing=1``) and the
    C-2.5 M-threshold default (``min_passing=3``).
    """
    ws = Workspace(base, name)
    ws.scaffold()
    ws.create_iteration(1)
    ws.set_current(1)
    ws.write_status({"project_name": name, "stage": "full_experiment",
                     "stage_index": 6, "iteration": 1,
                     "run_state": "running"})
    (ws.root / "config.yaml").write_text(yaml.safe_dump({
        "project_name": name, "task_family": "editing", "task_adapter": "x",
        "hardware": {"visible_gpu_ids": [0], "max_gpus_per_run": 1,
                     "per_gpu_memory_gb": 10.0},
        "data": {"data_root": str(data_root), "layout": "per_sample_dirs",
                 "methods": [
                    {"method_id": "method_a",
                     "path": str(data_root / "method_a"),
                     "output_file": "out.png"},
                    {"method_id": "method_b",
                     "path": str(data_root / "method_b"),
                     "output_file": "out.png"},
                 ],
                 "sample_glob": "*", "sample_count": sample_count,
                 "input_roles": {"source": "src.png"},
                 "sample_key": "relpath"},
        "experiment": {
            "auto_validate_pass_threshold": 0.70,
            "auto_validate_recall_threshold": 0.60,
            "auto_validate_min_samples": 2,  # low for tests
            "auto_validate_min_passing": min_passing,
        },
    }), encoding="utf-8")
    iter_dir = ws.iter_path()
    brief_path = iter_dir / "design" / "experiment_brief.json"
    brief_path.parent.mkdir(parents=True, exist_ok=True)
    brief_path.write_text(json.dumps({
        "schema_version": 1,
        "evaluation_goal": "test",
        "candidate_configs": chosen_configs,
        "validation": {"sample_size": 50},
        "pilot": {"sample_count": 5},
    }), encoding="utf-8")
    return ws


def _make_workspace(
    base: Path, data_root: Path, *,
    name: str = "av-demo",
    chosen_configs: list[dict] | None = None,
) -> Workspace:
    """Legacy adapter (Phase C-2.5 ``min_passing=1`` — preserves the
    "any 1 passes" semantics existing tests built on)."""
    if chosen_configs is None:
        chosen_configs = [
            {"combination_id": "cfg-pass", "family": "A",
             "slot": "vlm", "hypothesis_id": "h1", "scope": "whole"},
            {"combination_id": "cfg-fail", "family": "A",
             "slot": "vlm", "hypothesis_id": "h2", "scope": "whole"},
        ]
    return _make_workspace_with_min_passing(
        base, data_root,
        name=name, chosen_configs=chosen_configs,
        min_passing=1, sample_count=3,
    )


def _write_scores(
    iter_dir: Path, combination_id: str, mode: str,
    rows: list[dict],
) -> Path:
    """Write a scores.jsonl for one config in one mode."""
    cdir = iter_dir / "experiments" / "results" / mode / combination_id
    cdir.mkdir(parents=True, exist_ok=True)
    path = cdir / "scores.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def _write_judgments(
    iter_dir: Path, combination_id: str, method_id: str,
    judgments: list[dict],
) -> Path:
    """Stub a sub-agent's judgments output."""
    path = iter_dir / JUDGMENTS_REL / f"{combination_id}__{method_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": 1,
        "combination_id": combination_id,
        "method_id": method_id,
        "judgments": judgments,
    }), encoding="utf-8")
    return path


# ---- load_full_annotations ---------------------------------------------

def test_load_full_annotations_walks_all_files(tmp_path: Path):
    data_root = _scaffold_dataset(tmp_path)
    _write_annotation(data_root, "s1", {"method_a": "logo blurred",
                                        "method_b": ""})
    _write_annotation(data_root, "s2", {"method_a": "",
                                        "method_b": "color too white"})
    rows = load_full_annotations(data_root)
    by_key = {r["sample_key"]: r["per_method"] for r in rows}
    assert set(by_key) == {"s1", "s2"}
    assert by_key["s1"]["method_a"] == "logo blurred"
    assert by_key["s1"]["method_b"] == ""
    assert by_key["s2"]["method_b"] == "color too white"


def test_load_full_annotations_handles_missing_dir(tmp_path: Path):
    assert load_full_annotations(tmp_path) == []


# ---- build_batches ------------------------------------------------------

def test_build_batches_emits_one_per_config_method_pair(tmp_path: Path):
    data_root = _scaffold_dataset(tmp_path)
    _write_annotation(data_root, "s1", {"method_a": "logo blurred",
                                        "method_b": ""})
    _write_annotation(data_root, "s2", {"method_a": "",
                                        "method_b": "color too white"})
    ws = _make_workspace(tmp_path, data_root)
    iter_dir = ws.iter_path()
    # Score both configs on both methods on both annotated samples.
    for cid in ("cfg-pass", "cfg-fail"):
        rows = [
            {"sample_key": "s1", "method_id": "method_a",
             "score": 0.9, "sub_scores": {"cloth": 1, "model": 1},
             "scope": "whole", "ok": True},
            {"sample_key": "s1", "method_id": "method_b",
             "score": 0.9, "sub_scores": {"cloth": 1, "model": 1},
             "scope": "whole", "ok": True},
            {"sample_key": "s2", "method_id": "method_a",
             "score": 0.9, "sub_scores": {"cloth": 1, "model": 1},
             "scope": "whole", "ok": True},
            {"sample_key": "s2", "method_id": "method_b",
             "score": 0.9, "sub_scores": {"cloth": 1, "model": 1},
             "scope": "whole", "ok": True},
        ]
        _write_scores(iter_dir, cid, "annotated", rows)

    out = build_batches(str(ws.root))
    assert out["status"] == "ok"
    assert out["skipped_for_min_samples"] is False
    # 2 configs × 2 methods = 4 batches
    assert len(out["batches"]) == 4
    by_pair = {(b["combination_id"], b["method_id"]): b for b in out["batches"]}
    assert set(by_pair) == {
        ("cfg-pass", "method_a"), ("cfg-pass", "method_b"),
        ("cfg-fail", "method_a"), ("cfg-fail", "method_b"),
    }
    # Each batch carries both s1 + s2 (every annotated sample × method).
    for batch in out["batches"]:
        assert batch["sample_count"] == 2
        payload = json.loads(Path(batch["input_path"]).read_text())
        assert payload["combination_id"] == batch["combination_id"]
        assert payload["method_id"] == batch["method_id"]
        assert len(payload["samples"]) == 2
        for s in payload["samples"]:
            assert s["sample_key"] in {"s1", "s2"}
            assert "operator_annotation" in s


def test_build_batches_skips_below_min_samples(tmp_path: Path):
    data_root = _scaffold_dataset(tmp_path)
    _write_annotation(data_root, "s1", {"method_a": "logo blurred"})
    # Only 1 sample annotated, but min_samples=2 in the fixture config.
    ws = _make_workspace(tmp_path, data_root)
    out = build_batches(str(ws.root))
    assert out["status"] == "ok"
    assert out["skipped_for_min_samples"] is True
    assert out["batches"] == []
    assert out["annotated_sample_count"] == 1


def test_build_batches_rejects_non_workspace(tmp_path: Path):
    out = build_batches(tmp_path / "nope")
    assert out["error"] == "not_a_workspace"


def test_build_batches_carries_operator_annotation_text(tmp_path: Path):
    """The free-text note must land in the batch — the sub-agent reads it."""
    data_root = _scaffold_dataset(tmp_path)
    _write_annotation(data_root, "s1", {"method_a": "logo blurred"})
    _write_annotation(data_root, "s2", {"method_a": "color too white"})
    ws = _make_workspace(tmp_path, data_root)
    iter_dir = ws.iter_path()
    _write_scores(iter_dir, "cfg-pass", "annotated", [
        {"sample_key": "s1", "method_id": "method_a",
         "score": 0.5, "sub_scores": {"cloth": 0}, "scope": "whole", "ok": True},
        {"sample_key": "s2", "method_id": "method_a",
         "score": 0.7, "sub_scores": {}, "scope": "whole", "ok": True},
    ])
    _write_scores(iter_dir, "cfg-fail", "annotated", [
        {"sample_key": "s1", "method_id": "method_a", "score": 1.0,
         "sub_scores": {}, "scope": "whole", "ok": True},
        {"sample_key": "s2", "method_id": "method_a", "score": 1.0,
         "sub_scores": {}, "scope": "whole", "ok": True},
    ])
    out = build_batches(str(ws.root))
    batch = next(b for b in out["batches"]
                 if b["combination_id"] == "cfg-pass"
                 and b["method_id"] == "method_a")
    payload = json.loads(Path(batch["input_path"]).read_text())
    notes = {s["sample_key"]: s["operator_annotation"]
             for s in payload["samples"]}
    assert notes["s1"] == "logo blurred"
    assert notes["s2"] == "color too white"


def test_build_batches_remaps_dataset_method_keys_to_method_id(tmp_path: Path):
    """Annotations labelled by the dataset method name (the lowercased
    result-dir basename, e.g. ``tryon_results``) must still join to the
    config ``method_id`` (e.g. ``flux2klein``). Without the remap the join
    silently misses and every operator_annotation reads empty — the bug
    that made the Phase C-2 gate a no-op. Also asserts a 'Good' verdict is
    normalized to '' at the batch (so the sub-agent treats it as GOOD)."""
    data_root = tmp_path / "data"
    method_dir = data_root / "TryOn_results"   # basename -> 'tryon_results'
    for sk in ("s1", "s2"):
        (method_dir / sk).mkdir(parents=True)
        (method_dir / sk / "out.png").write_bytes(b"\x89PNG\r\n")
    # Annotation per_method is keyed by the DATASET name, not the method_id.
    _write_annotation(data_root, "s1", {"tryon_results": "color too white"})
    _write_annotation(data_root, "s2", {"tryon_results": "Good"})
    ws = _make_workspace_with_min_passing(
        tmp_path, data_root, name="av-remap",
        chosen_configs=[{"combination_id": "cfg", "family": "A",
                         "slot": "vlm", "hypothesis_id": "h1", "scope": "whole"}],
        min_passing=1, sample_count=2,
    )
    # Override config method_id != dataset basename.
    cfg_path = ws.root / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["data"]["methods"] = [{
        "method_id": "flux2klein",
        "path": str(method_dir),
        "output_file": "out.png",
    }]
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    iter_dir = ws.iter_path()
    _write_scores(iter_dir, "cfg", "annotated", [
        {"sample_key": "s1", "method_id": "flux2klein", "score": 1.0,
         "sub_scores": {}, "scope": "whole", "ok": True},
        {"sample_key": "s2", "method_id": "flux2klein", "score": 1.0,
         "sub_scores": {}, "scope": "whole", "ok": True},
    ])
    out = build_batches(str(ws.root))
    batch = next(b for b in out["batches"] if b["method_id"] == "flux2klein")
    payload = json.loads(Path(batch["input_path"]).read_text())
    notes = {s["sample_key"]: s["operator_annotation"] for s in payload["samples"]}
    assert notes["s1"] == "color too white"   # remapped from 'tryon_results'
    assert notes["s2"] == ""                   # 'Good' normalized to GOOD


# ---- aggregate_judgments -----------------------------------------------

def _setup_two_configs_with_scores(tmp_path: Path) -> Workspace:
    """Fixture: 2 configs × 1 method × 4 annotated samples."""
    data_root = _scaffold_dataset(tmp_path)
    # Add s4 to the dataset for variety
    (data_root / "method_a" / "s4").mkdir()
    (data_root / "method_a" / "s4" / "out.png").write_bytes(b"\x89PNG\r\n")
    (data_root / "method_b" / "s4").mkdir()
    (data_root / "method_b" / "s4" / "out.png").write_bytes(b"\x89PNG\r\n")
    # 4 annotated samples: 2 flagged bad for method_a, 2 left good
    _write_annotation(data_root, "s1", {"method_a": "logo blurred"})
    _write_annotation(data_root, "s2", {"method_a": "color wrong"})
    _write_annotation(data_root, "s3", {"method_a": "", "method_b": "filler"})
    _write_annotation(data_root, "s4", {"method_a": "", "method_b": "filler"})
    # s3/s4 carry no flag for method_a — operator labels GOOD for method_a.

    ws = _make_workspace(tmp_path, data_root)
    iter_dir = ws.iter_path()
    for cid in ("cfg-pass", "cfg-fail"):
        rows = [
            {"sample_key": sk, "method_id": "method_a", "score": 0.5,
             "sub_scores": {}, "scope": "whole", "ok": True}
            for sk in ("s1", "s2", "s3", "s4")
        ]
        _write_scores(iter_dir, cid, "annotated", rows)
    build_batches(str(ws.root))
    return ws


def test_aggregate_judgments_passes_a_high_agreement_config(tmp_path: Path):
    ws = _setup_two_configs_with_scores(tmp_path)
    iter_dir = ws.iter_path()
    # cfg-pass: agrees on all 4 samples (pass_rate=1.0, recall=1.0)
    _write_judgments(iter_dir, "cfg-pass", "method_a", [
        {"sample_key": "s1", "agree": True, "rationale": "caught it"},
        {"sample_key": "s2", "agree": True, "rationale": "caught it"},
        {"sample_key": "s3", "agree": True, "rationale": "clean OK"},
        {"sample_key": "s4", "agree": True, "rationale": "clean OK"},
    ])
    # cfg-fail: disagrees on the 2 flagged samples (pass_rate=0.5, recall=0)
    _write_judgments(iter_dir, "cfg-fail", "method_a", [
        {"sample_key": "s1", "agree": False, "rationale": "missed it"},
        {"sample_key": "s2", "agree": False, "rationale": "missed it"},
        {"sample_key": "s3", "agree": True, "rationale": "OK"},
        {"sample_key": "s4", "agree": True, "rationale": "OK"},
    ])
    out = aggregate_judgments(str(ws.root))
    assert out["status"] == "ok"
    assert out["any_passed"] is True
    assert out["passing_configs"] == ["cfg-pass"]
    assert out["failing_configs"] == ["cfg-fail"]
    by_id = {row["combination_id"]: row for row in out["per_config"]}
    assert by_id["cfg-pass"]["pass_rate"] == 1.0
    assert by_id["cfg-pass"]["recall_rate"] == 1.0
    assert by_id["cfg-pass"]["passed"] is True
    assert by_id["cfg-fail"]["pass_rate"] == 0.5
    assert by_id["cfg-fail"]["recall_rate"] == 0.0
    assert by_id["cfg-fail"]["passed"] is False
    # result.json written
    result_on_disk = json.loads(
        (iter_dir / RESULT_REL).read_text(encoding="utf-8"))
    assert result_on_disk["any_passed"] is True


def test_aggregate_judgments_all_fail(tmp_path: Path):
    """Every config below thresholds → any_passed: false."""
    ws = _setup_two_configs_with_scores(tmp_path)
    iter_dir = ws.iter_path()
    for cid in ("cfg-pass", "cfg-fail"):
        _write_judgments(iter_dir, cid, "method_a", [
            {"sample_key": "s1", "agree": False, "rationale": "missed"},
            {"sample_key": "s2", "agree": False, "rationale": "missed"},
            {"sample_key": "s3", "agree": False, "rationale": "false alarm"},
            {"sample_key": "s4", "agree": False, "rationale": "false alarm"},
        ])
    out = aggregate_judgments(str(ws.root))
    assert out["any_passed"] is False
    assert set(out["failing_configs"]) == {"cfg-pass", "cfg-fail"}
    assert out["passing_configs"] == []


def _setup_three_configs_with_scores(tmp_path: Path) -> Workspace:
    """Phase C-2.5 fixture: 3 configs × 1 method × 4 annotated samples,
    pinned with ``auto_validate_min_passing=3`` to exercise the
    M-threshold semantics."""
    data_root = _scaffold_dataset(tmp_path)
    # Extend the 3-sample dataset scaffolded by _scaffold_dataset with a
    # 4th sample so the M=3 fixture has more coverage.
    (data_root / "method_a" / "s4").mkdir()
    (data_root / "method_a" / "s4" / "out.png").write_bytes(b"\x89PNG\r\n")
    (data_root / "method_b" / "s4").mkdir()
    (data_root / "method_b" / "s4" / "out.png").write_bytes(b"\x89PNG\r\n")
    _write_annotation(data_root, "s1", {"method_a": "logo blurred"})
    _write_annotation(data_root, "s2", {"method_a": "color wrong"})
    _write_annotation(data_root, "s3", {"method_a": "", "method_b": "filler"})
    _write_annotation(data_root, "s4", {"method_a": "", "method_b": "filler"})

    chosen_configs = [
        {"combination_id": f"cfg-{i}", "family": "A",
         "slot": "vlm", "hypothesis_id": f"h{i}", "scope": "whole"}
        for i in range(1, 4)
    ]
    ws = _make_workspace_with_min_passing(
        tmp_path, data_root,
        name="av-m3", chosen_configs=chosen_configs,
        min_passing=3, sample_count=4,
    )
    iter_dir = ws.iter_path()
    for cid in ("cfg-1", "cfg-2", "cfg-3"):
        rows = [
            {"sample_key": sk, "method_id": "method_a", "score": 0.5,
             "sub_scores": {}, "scope": "whole", "ok": True}
            for sk in ("s1", "s2", "s3", "s4")
        ]
        _write_scores(iter_dir, cid, "annotated", rows)
    build_batches(str(ws.root))
    return ws


def test_aggregate_judgments_m_threshold_partial_pass(tmp_path: Path):
    """Phase C-2.5: 2/3 configs clear pass/recall, min_passing=3 →
    any_passed: false (the iter should auto-revise under the new rule
    instead of advancing on partial-pass)."""
    ws = _setup_three_configs_with_scores(tmp_path)
    iter_dir = ws.iter_path()
    # cfg-1 + cfg-2 pass (all agree), cfg-3 fails on the 2 flagged
    for cid in ("cfg-1", "cfg-2"):
        _write_judgments(iter_dir, cid, "method_a", [
            {"sample_key": "s1", "agree": True, "rationale": "caught"},
            {"sample_key": "s2", "agree": True, "rationale": "caught"},
            {"sample_key": "s3", "agree": True, "rationale": "OK"},
            {"sample_key": "s4", "agree": True, "rationale": "OK"},
        ])
    _write_judgments(iter_dir, "cfg-3", "method_a", [
        {"sample_key": "s1", "agree": False, "rationale": "missed"},
        {"sample_key": "s2", "agree": False, "rationale": "missed"},
        {"sample_key": "s3", "agree": True, "rationale": "OK"},
        {"sample_key": "s4", "agree": True, "rationale": "OK"},
    ])
    out = aggregate_judgments(str(ws.root))
    assert out["status"] == "ok"
    assert out["passing_count"] == 2
    assert out["min_passing"] == 3
    # any_passed redefined as passing_count >= min_passing → false here
    assert out["any_passed"] is False
    assert set(out["passing_configs"]) == {"cfg-1", "cfg-2"}
    assert out["failing_configs"] == ["cfg-3"]


def test_aggregate_judgments_m_threshold_meets_floor(tmp_path: Path):
    """Phase C-2.5: 3/3 configs pass, min_passing=3 → any_passed: true."""
    ws = _setup_three_configs_with_scores(tmp_path)
    iter_dir = ws.iter_path()
    for cid in ("cfg-1", "cfg-2", "cfg-3"):
        _write_judgments(iter_dir, cid, "method_a", [
            {"sample_key": "s1", "agree": True, "rationale": "caught"},
            {"sample_key": "s2", "agree": True, "rationale": "caught"},
            {"sample_key": "s3", "agree": True, "rationale": "OK"},
            {"sample_key": "s4", "agree": True, "rationale": "OK"},
        ])
    out = aggregate_judgments(str(ws.root))
    assert out["status"] == "ok"
    assert out["passing_count"] == 3
    assert out["min_passing"] == 3
    assert out["any_passed"] is True
    assert set(out["passing_configs"]) == {"cfg-1", "cfg-2", "cfg-3"}
    assert out["failing_configs"] == []


def test_aggregate_judgments_missing_file(tmp_path: Path):
    """A missing judgments file surfaces as missing_judgments error."""
    ws = _setup_two_configs_with_scores(tmp_path)
    iter_dir = ws.iter_path()
    _write_judgments(iter_dir, "cfg-pass", "method_a", [
        {"sample_key": "s1", "agree": True, "rationale": "ok"},
    ])
    # cfg-fail judgments not written
    out = aggregate_judgments(str(ws.root))
    assert out["error"] == "missing_judgments"
    missing_pairs = {(m["combination_id"], m["method_id"]) for m in out["missing"]}
    assert ("cfg-fail", "method_a") in missing_pairs


def test_aggregate_judgments_below_min_samples_marks_all_pass(tmp_path: Path):
    """Floor case: too few annotations → all configs proceed (no gate)."""
    data_root = _scaffold_dataset(tmp_path)
    _write_annotation(data_root, "s1", {"method_a": "only one"})
    ws = _make_workspace(tmp_path, data_root)  # min_samples=2 by default
    out = aggregate_judgments(str(ws.root))
    assert out["status"] == "ok"
    assert out["skipped_for_min_samples"] is True
    assert out["any_passed"] is True
    assert "cfg-pass" in out["passing_configs"]
    assert "cfg-fail" in out["passing_configs"]


def test_aggregate_judgments_recall_vacuous_when_no_operator_flag(tmp_path: Path):
    """When the operator flagged NOTHING for this method on annotated
    samples, recall is vacuously 1.0 — only pass_rate gates."""
    data_root = _scaffold_dataset(tmp_path)
    # Two annotated samples but only method_b is flagged on them.
    _write_annotation(data_root, "s1", {"method_a": "", "method_b": "x"})
    _write_annotation(data_root, "s2", {"method_a": "", "method_b": "y"})
    ws = _make_workspace(tmp_path, data_root)
    iter_dir = ws.iter_path()
    _write_scores(iter_dir, "cfg-pass", "annotated", [
        {"sample_key": "s1", "method_id": "method_a", "score": 1,
         "sub_scores": {}, "scope": "whole", "ok": True},
        {"sample_key": "s2", "method_id": "method_a", "score": 1,
         "sub_scores": {}, "scope": "whole", "ok": True},
    ])
    _write_scores(iter_dir, "cfg-fail", "annotated", [
        {"sample_key": "s1", "method_id": "method_a", "score": 1,
         "sub_scores": {}, "scope": "whole", "ok": True},
        {"sample_key": "s2", "method_id": "method_a", "score": 1,
         "sub_scores": {}, "scope": "whole", "ok": True},
    ])
    build_batches(str(ws.root))
    # 100% agreement on operator-good samples
    for cid in ("cfg-pass", "cfg-fail"):
        _write_judgments(iter_dir, cid, "method_a", [
            {"sample_key": "s1", "agree": True, "rationale": "ok"},
            {"sample_key": "s2", "agree": True, "rationale": "ok"},
        ])
    out = aggregate_judgments(str(ws.root))
    by_id = {row["combination_id"]: row for row in out["per_config"]}
    # operator_flagged=0 → recall is 1.0 vacuously
    assert by_id["cfg-pass"]["operator_flagged"] == 0
    assert by_id["cfg-pass"]["recall_rate"] == 1.0
    assert by_id["cfg-pass"]["passed"] is True


# ---- Phase C-2 scope-gating (Family B) ---------------------------------

def _write_hypotheses(iter_dir: Path, mapping: dict[str, str]) -> Path:
    """Stub a design/hypotheses.md so hypothesis_text resolves."""
    path = iter_dir / "design" / "hypotheses.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(f"## {hid}: {text}" for hid, text in mapping.items()) + "\n",
        encoding="utf-8",
    )
    return path


def test_build_batches_stamps_scope_gating_and_evaluation_target(tmp_path: Path):
    """Family-B batches carry scope_gating_enabled: true + an
    evaluation_target with the config's measured dimension (hypothesis_text
    resolved from hypotheses.md); Family-A batches carry false."""
    data_root = _scaffold_dataset(tmp_path)
    _write_annotation(data_root, "s1", {"method_a": "cloth not changed"})
    _write_annotation(data_root, "s2", {"method_a": "color wrong"})
    chosen = [
        {"combination_id": "cfg-b", "family": "B", "slot": "metric_baseline",
         "hypothesis_id": "H3", "scope": "garment-region",
         "metric_subfamily": "model-preservation-region-mask-pack-v1"},
        {"combination_id": "cfg-a", "family": "A", "slot": "vlm_scale",
         "hypothesis_id": "H1", "scope": "whole",
         "judge": "qwen", "prompt": "p"},
    ]
    ws = _make_workspace(tmp_path, data_root, chosen_configs=chosen)
    iter_dir = ws.iter_path()
    _write_hypotheses(iter_dir, {
        "H3": "the person's pose and position are preserved",
        "H1": "garment fidelity is high",
    })
    for cid in ("cfg-b", "cfg-a"):
        _write_scores(iter_dir, cid, "annotated", [
            {"sample_key": "s1", "method_id": "method_a", "score": 0.9,
             "sub_scores": {}, "scope": "whole", "ok": True},
            {"sample_key": "s2", "method_id": "method_a", "score": 0.9,
             "sub_scores": {}, "scope": "whole", "ok": True},
        ])
    out = build_batches(str(ws.root))
    assert out["status"] == "ok"
    by_cid = {b["combination_id"]: b for b in out["batches"]}

    pb = json.loads(Path(by_cid["cfg-b"]["input_path"]).read_text())
    assert pb["scope_gating_enabled"] is True
    et = pb["evaluation_target"]
    assert et["family"] == "B"
    assert et["scope"] == "garment-region"
    assert et["metric_subfamily"] == "model-preservation-region-mask-pack-v1"
    assert et["hypothesis_text"] == "the person's pose and position are preserved"

    pa = json.loads(Path(by_cid["cfg-a"]["input_path"]).read_text())
    assert pa["scope_gating_enabled"] is False
    assert pa["evaluation_target"]["judge"] == "qwen"
    assert pa["evaluation_target"]["hypothesis_text"] == "garment fidelity is high"


def test_aggregate_judgments_out_of_scope_excluded_from_recall(tmp_path: Path):
    """An ``applicable: false`` operator-flagged sample is excluded from
    the recall denominator (it was never a test of this config) and
    counts as agreed for pass_rate. Without the exclusion, the method
    would wrongly get recall credit for an out-of-scope 'catch'."""
    ws = _setup_two_configs_with_scores(tmp_path)
    iter_dir = ws.iter_path()
    # cfg-pass on method_a: s1+s2 are operator-flagged; s3+s4 good.
    #   s1 → in-scope, method MISSED it (agree False, applicable True)
    #   s2 → OUT of scope (agree True, applicable False) — excluded from recall
    #   s3,s4 → clean, agree True
    _write_judgments(iter_dir, "cfg-pass", "method_a", [
        {"sample_key": "s1", "agree": False, "applicable": True,
         "rationale": "in-scope defect missed"},
        {"sample_key": "s2", "agree": True, "applicable": False,
         "rationale": "note out of this metric's dimension"},
        {"sample_key": "s3", "agree": True, "applicable": True,
         "rationale": "clean OK"},
        {"sample_key": "s4", "agree": True, "applicable": True,
         "rationale": "clean OK"},
    ])
    # cfg-fail: simple all-agree (not the focus of this test).
    _write_judgments(iter_dir, "cfg-fail", "method_a", [
        {"sample_key": sk, "agree": True, "rationale": "ok"}
        for sk in ("s1", "s2", "s3", "s4")
    ])
    out = aggregate_judgments(str(ws.root))
    assert out["status"] == "ok"
    row = next(r for r in out["per_config"] if r["combination_id"] == "cfg-pass")
    assert row["operator_flagged"] == 2
    assert row["in_scope_flagged"] == 1          # s2 excluded
    assert row["out_of_scope_flagged"] == 1      # s2 tallied here
    assert row["method_caught_flagged"] == 0     # missed the one in-scope flag
    # recall over in-scope flags only: 0/1 = 0.0 (NOT 1/2 = 0.5)
    assert row["recall_rate"] == 0.0
    # pass_rate counts the out-of-scope sample as agreed: 3/4 = 0.75
    assert row["pass_rate"] == 0.75
    # per_method mirrors the split
    pm = row["per_method"][0]
    assert pm["in_scope_flagged"] == 1
    assert pm["out_of_scope_flagged"] == 1


def test_aggregate_judgments_applicable_defaults_true(tmp_path: Path):
    """Judgments with no ``applicable`` key behave exactly as before:
    every flagged sample is in-scope, out_of_scope_flagged is 0."""
    ws = _setup_two_configs_with_scores(tmp_path)
    iter_dir = ws.iter_path()
    for cid in ("cfg-pass", "cfg-fail"):
        _write_judgments(iter_dir, cid, "method_a", [
            {"sample_key": "s1", "agree": True, "rationale": "caught"},
            {"sample_key": "s2", "agree": True, "rationale": "caught"},
            {"sample_key": "s3", "agree": True, "rationale": "OK"},
            {"sample_key": "s4", "agree": True, "rationale": "OK"},
        ])
    out = aggregate_judgments(str(ws.root))
    row = next(r for r in out["per_config"] if r["combination_id"] == "cfg-pass")
    assert row["out_of_scope_flagged"] == 0
    assert row["in_scope_flagged"] == row["operator_flagged"] == 2
    assert row["recall_rate"] == 1.0
    assert row["pass_rate"] == 1.0


# ---- read_result -------------------------------------------------------

def test_read_result_returns_none_when_missing(tmp_path: Path):
    assert read_result(tmp_path) is None


def test_read_result_returns_dict_when_present(tmp_path: Path):
    ws = _setup_two_configs_with_scores(tmp_path)
    iter_dir = ws.iter_path()
    for cid in ("cfg-pass", "cfg-fail"):
        _write_judgments(iter_dir, cid, "method_a", [
            {"sample_key": sk, "agree": True, "rationale": ""}
            for sk in ("s1", "s2", "s3", "s4")
        ])
    aggregate_judgments(str(ws.root))
    result = read_result(iter_dir)
    assert isinstance(result, dict)
    assert result["any_passed"] is True


# ---- CLI round-trip -----------------------------------------------------

def test_cli_auto_validate_prepare_finalize_round_trip(
    tmp_path: Path, monkeypatch, capsys,
):
    """End-to-end CLI: prepare emits batches, stub judgments, finalize
    aggregates them into result.json."""
    ws = _setup_two_configs_with_scores(tmp_path)
    iter_dir = ws.iter_path()

    # prepare
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "mode": "annotated"})))
    rc = main(["auto-validate-prepare"])
    prep = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert prep["status"] == "ok"
    assert len(prep["batches"]) == 2  # 2 configs × 1 annotated method

    # stub judgments: cfg-pass agrees on all 4, cfg-fail disagrees on flagged
    for batch in prep["batches"]:
        cid = batch["combination_id"]
        judgments = [
            {"sample_key": sk,
             "agree": (cid == "cfg-pass") or sk in ("s3", "s4"),
             "rationale": "test"}
            for sk in ("s1", "s2", "s3", "s4")
        ]
        Path(batch["output_path"]).write_text(json.dumps({
            "schema_version": 1, "combination_id": cid,
            "method_id": batch["method_id"], "judgments": judgments,
        }), encoding="utf-8")

    # finalize
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root)})))
    rc = main(["auto-validate-finalize"])
    final = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert final["any_passed"] is True
    assert final["passing_configs"] == ["cfg-pass"]
    assert final["failing_configs"] == ["cfg-fail"]
    # result.json on disk
    assert (iter_dir / RESULT_REL).is_file()


def test_cli_auto_validate_prepare_missing_workspace_path(
    monkeypatch, capsys,
):
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
    rc = main(["auto-validate-prepare"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["error"] == "missing_workspace_path"


def test_cli_auto_validate_finalize_missing_workspace_path(
    monkeypatch, capsys,
):
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
    rc = main(["auto-validate-finalize"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["error"] == "missing_workspace_path"


# ---- list-annotations (Phase C-2.1) ------------------------------------

def test_cli_list_annotations_returns_sorted_keys(
    tmp_path: Path, monkeypatch, capsys,
):
    """Stage 5 reads the FULL sorted annotated sample list from this CLI
    to stamp samples_subset on annotated-mode tasks."""
    data_root = _scaffold_dataset(tmp_path)
    # Annotate 3 samples; only 2 carry non-empty notes.
    _write_annotation(data_root, "s2", {"method_a": "logo blurred"})
    _write_annotation(data_root, "s1", {"method_a": "color wrong",
                                        "method_b": "fold issue"})
    _write_annotation(data_root, "s3", {"method_a": ""})  # blank — skipped
    ws = _make_workspace(tmp_path, data_root)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root)})))
    rc = main(["list-annotations"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["status"] == "ok"
    # Sorted, only samples with at least one non-empty note.
    assert payload["sample_keys"] == ["s1", "s2"]
    assert payload["count"] == 2
    # Method coverage: method_a flagged on both, method_b only on s1.
    assert payload["method_coverage"] == {"method_a": 2, "method_b": 1}
    assert payload["data_root"] == str(data_root)


def test_cli_list_annotations_handles_empty_dataset(
    tmp_path: Path, monkeypatch, capsys,
):
    """No annotations dir → count: 0, sample_keys: [] (not an error —
    the C-2 fall-through path)."""
    data_root = _scaffold_dataset(tmp_path)  # no annotations written
    ws = _make_workspace(tmp_path, data_root)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root)})))
    rc = main(["list-annotations"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["status"] == "ok"
    assert payload["sample_keys"] == []
    assert payload["count"] == 0
    assert payload["method_coverage"] == {}


def test_cli_list_annotations_missing_workspace_path(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
    rc = main(["list-annotations"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["error"] == "missing_workspace_path"


# ---- Phase C-2.4: fail-loud when annotated round didn't run ------------

def _write_task_plan(iter_dir: Path, has_annotated: bool) -> Path:
    """Stub an iter_NNN/experiments/plans/task_plan.json. Set
    has_annotated=False to mimic the legacy v0.1.6 plan that triggered
    the user's false-negative."""
    plan_path = iter_dir / "experiments" / "plans" / "task_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    tasks = [
        {"id": "eval-cfg-pass-pilot", "type": "eval", "family": "A",
         "mode": "pilot", "combination_id": "cfg-pass"},
        {"id": "eval-cfg-pass-full", "type": "eval", "family": "A",
         "mode": "full", "combination_id": "cfg-pass"},
        {"id": "eval-cfg-fail-pilot", "type": "eval", "family": "A",
         "mode": "pilot", "combination_id": "cfg-fail"},
        {"id": "eval-cfg-fail-full", "type": "eval", "family": "A",
         "mode": "full", "combination_id": "cfg-fail"},
    ]
    if has_annotated:
        tasks.extend([
            {"id": "eval-cfg-pass-annot", "type": "eval", "family": "A",
             "mode": "annotated", "combination_id": "cfg-pass"},
            {"id": "eval-cfg-fail-annot", "type": "eval", "family": "A",
             "mode": "annotated", "combination_id": "cfg-fail"},
        ])
    plan_path.write_text(json.dumps({
        "schema_version": 1, "iteration": 1, "evaluation_goal": "g",
        "gpu_pool": 4, "tasks": tasks,
    }), encoding="utf-8")
    return plan_path


def test_build_batches_returns_error_when_plan_lacks_annotated_tasks(
    tmp_path: Path,
):
    """The user's exact bug: ≥ min_samples annotations exist, but the
    task plan has no annotated-mode tasks (legacy v0.1.6 plan).
    build_batches must return a distinct error, NOT silently empty
    batches that finalize would treat as all-fail."""
    data_root = _scaffold_dataset(tmp_path)
    # 3 annotated samples; min_samples is 2 in this fixture so the gate
    # is supposed to run.
    _write_annotation(data_root, "s1", {"method_a": "logo blurred"})
    _write_annotation(data_root, "s2", {"method_a": "color wrong"})
    _write_annotation(data_root, "s3", {"method_a": "filler"})
    ws = _make_workspace(tmp_path, data_root)
    _write_task_plan(ws.iter_path(), has_annotated=False)

    out = build_batches(str(ws.root))
    assert out["error"] == "no_annotated_scores"
    assert out["missing_reason"] == "no_annotated_tasks_in_plan"
    assert out["plan_has_annotated"] is False
    assert out["annotated_sample_count"] >= 2


def test_build_batches_distinguishes_didnt_run_from_missing_in_plan(
    tmp_path: Path,
):
    """When the plan HAS annotated tasks but no scores.jsonl exists, the
    missing_reason should distinguish 'annotated_round_didnt_run' from
    'no_annotated_tasks_in_plan' so Stage 6 routes correctly."""
    data_root = _scaffold_dataset(tmp_path)
    _write_annotation(data_root, "s1", {"method_a": "logo blurred"})
    _write_annotation(data_root, "s2", {"method_a": "color wrong"})
    _write_annotation(data_root, "s3", {"method_a": "filler"})
    ws = _make_workspace(tmp_path, data_root)
    _write_task_plan(ws.iter_path(), has_annotated=True)
    # No scores.jsonl written — annotated round was planned but didn't run.

    out = build_batches(str(ws.root))
    assert out["error"] == "no_annotated_scores"
    assert out["missing_reason"] == "annotated_round_didnt_run"
    assert out["plan_has_annotated"] is True


def test_aggregate_judgments_returns_error_on_missing_scores(tmp_path: Path):
    """aggregate_judgments must also surface the no_annotated_scores
    error directly (the Stage 6 prompt's defensive backstop), not
    return all-fail per_config rows."""
    data_root = _scaffold_dataset(tmp_path)
    _write_annotation(data_root, "s1", {"method_a": "logo blurred"})
    _write_annotation(data_root, "s2", {"method_a": "color wrong"})
    _write_annotation(data_root, "s3", {"method_a": "filler"})
    ws = _make_workspace(tmp_path, data_root)
    _write_task_plan(ws.iter_path(), has_annotated=False)
    # Note: build_batches was NOT called, so inputs/ doesn't exist —
    # this is the "finalize called blind" case. Should still detect.

    out = aggregate_judgments(str(ws.root))
    assert out["error"] == "no_annotated_scores"
    assert out["missing_reason"] == "no_annotated_tasks_in_plan"


def test_aggregate_judgments_below_min_samples_still_skips(tmp_path: Path):
    """The min-samples skip path is unaffected by the new fail-loud
    branch — datasets below the floor still get any_passed: true
    (no gate runs)."""
    data_root = _scaffold_dataset(tmp_path)
    # Only 1 annotated sample; default min_samples is 2 in this fixture.
    _write_annotation(data_root, "s1", {"method_a": "single note"})
    ws = _make_workspace(tmp_path, data_root)

    out = aggregate_judgments(str(ws.root))
    assert out.get("status") == "ok"
    assert out["skipped_for_min_samples"] is True
    assert out["any_passed"] is True


def test_plan_has_annotated_tasks_helper(tmp_path: Path):
    """Direct unit test of the helper."""
    from era.orchestration.auto_validate import (
        _load_task_plan,
        _plan_has_annotated_tasks,
    )
    iter_dir = tmp_path / "iter_001"
    iter_dir.mkdir()
    # No file → False
    assert _load_task_plan(iter_dir) is None
    assert _plan_has_annotated_tasks(None) is False
    # Plan without annotated → False
    _write_task_plan(iter_dir, has_annotated=False)
    plan = _load_task_plan(iter_dir)
    assert _plan_has_annotated_tasks(plan) is False
    # Plan with annotated → True
    _write_task_plan(iter_dir, has_annotated=True)
    plan = _load_task_plan(iter_dir)
    assert _plan_has_annotated_tasks(plan) is True
