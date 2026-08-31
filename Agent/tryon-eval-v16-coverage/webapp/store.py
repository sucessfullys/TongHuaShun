"""Feedback persistence — ``<run_dir>/human/feedback.json``.

``feedback.json`` is sparse: it stores only deviations from the default
(flagging results default to *endorsed*). The full ground-truth label set is
materialized into ``human_labels.json`` at finalize.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from evaluator.review_model import write_json_atomic

SCHEMA_VERSION = "1.0"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def feedback_path(run_dir: Path) -> Path:
    return Path(run_dir) / "human" / "feedback.json"


def human_labels_path(run_dir: Path) -> Path:
    return Path(run_dir) / "human" / "human_labels.json"


def default_feedback(iteration: int, run_dir_name: str, mode: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "iteration": iteration,
        "iteration_dir": run_dir_name,
        "mode": mode,
        "status": "draft",
        "reviewer": "",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "finalized_at": None,
        "item_marks": [],
        "comparison_marks": [],
        "general_feedback": "",
    }


def read_feedback(run_dir: Path, iteration: int, mode: str) -> dict:
    """Load ``feedback.json``, or a fresh draft when it does not exist yet."""
    path = feedback_path(run_dir)
    if path.is_file():
        try:
            fb = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            fb = {}
        if isinstance(fb, dict) and fb.get("schema_version"):
            fb.setdefault("item_marks", [])
            fb.setdefault("comparison_marks", [])
            fb.setdefault("general_feedback", "")
            return fb
    return default_feedback(iteration, Path(run_dir).name, mode)


def write_feedback(run_dir: Path, fb: dict) -> dict:
    """Persist ``feedback.json`` atomically, refreshing ``updated_at``."""
    fb["updated_at"] = _now_iso()
    write_json_atomic(feedback_path(run_dir), fb)
    return fb


def set_item_mark(
    fb: dict, sample_key: str, method_id: str, combination_id: str,
    label: str, comment: str = "",
) -> dict:
    """Flag (``label="wrong"``) or clear (``label="ok"``) one flagging cell."""
    marks = [
        m for m in fb.get("item_marks", [])
        if (m.get("sample_key"), m.get("method_id"), m.get("combination_id"))
        != (sample_key, method_id, combination_id)
    ]
    if label == "wrong":
        marks.append({
            "sample_key": sample_key, "method_id": method_id,
            "combination_id": combination_id, "label": "wrong",
            "comment": comment or "",
        })
    fb["item_marks"] = marks
    return fb


def set_comparison_mark(
    fb: dict, sample_key: str, combination_id: str,
    label: str, comment: str = "",
) -> dict:
    """Mark (``label="wrong"``) or clear one comparison ranking."""
    marks = [
        m for m in fb.get("comparison_marks", [])
        if (m.get("sample_key"), m.get("combination_id"))
        != (sample_key, combination_id)
    ]
    if label == "wrong":
        marks.append({
            "sample_key": sample_key, "combination_id": combination_id,
            "label": "wrong", "comment": comment or "",
        })
    fb["comparison_marks"] = marks
    return fb


def set_general(fb: dict, text: str) -> dict:
    fb["general_feedback"] = text or ""
    return fb
