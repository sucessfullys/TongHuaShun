"""Tests for the Stage 0 annotation probe (era/probe/annotations.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from era.probe.annotations import probe_annotations


def _write_central_annotation(
    dataset_root: Path, sample_key: str, per_method: dict[str, str],
) -> None:
    """Drop one central annotation file (matches era.annotate.store)."""
    path = dataset_root / "annotations" / f"{sample_key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": "2.0",
        "sample_key": sample_key,
        "per_method": per_method,
        "created_at": "2026-05-27T01:00:00+00:00",
        "updated_at": "2026-05-27T01:00:00+00:00",
    }), encoding="utf-8")


def _write_per_method_copy(
    method_dir: Path, sample_key: str, method_id: str, text: str,
) -> None:
    """Drop one per-method annotation.json copy."""
    path = method_dir / sample_key / "annotation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": "2.0",
        "sample_key": sample_key,
        "method_id": method_id,
        "annotation": text,
        "created_at": "2026-05-27T01:00:00+00:00",
        "updated_at": "2026-05-27T01:00:00+00:00",
    }), encoding="utf-8")


def _scaffold_dataset_with_annotations(root: Path) -> dict[str, Path]:
    """Two methods, three samples each. Populates central + per-method
    files for the first two samples; leaves the third unannotated."""
    method_paths: dict[str, Path] = {}
    for method in ("method_a", "method_b"):
        for sk in ("s1", "s2", "s3"):
            (root / method / sk).mkdir(parents=True, exist_ok=True)
        method_paths[method] = root / method

    # Central annotations: s1 has both methods, s2 has only method_a.
    _write_central_annotation(root, "s1", {
        "method_a": "A's verdict on s1",
        "method_b": "B's verdict on s1",
    })
    _write_central_annotation(root, "s2", {
        "method_a": "A's verdict on s2",
    })

    # Per-method copies — three (mirror of the central state)
    _write_per_method_copy(root / "method_a", "s1", "method_a", "A's verdict on s1")
    _write_per_method_copy(root / "method_b", "s1", "method_b", "B's verdict on s1")
    _write_per_method_copy(root / "method_a", "s2", "method_a", "A's verdict on s2")
    return method_paths


# ---- happy path -------------------------------------------------------

def test_probe_reports_counts(tmp_path: Path):
    mp = _scaffold_dataset_with_annotations(tmp_path)
    out = probe_annotations(tmp_path, mp)
    assert out["central_count"] == 2
    assert out["per_method_count"] == {"method_a": 2, "method_b": 1}
    assert out["per_method_count_total"] == 3
    assert out["samples_with_any_note"] == ["s1", "s2"]


def test_probe_method_coverage(tmp_path: Path):
    """method_coverage = how many central samples each method appears in."""
    mp = _scaffold_dataset_with_annotations(tmp_path)
    out = probe_annotations(tmp_path, mp)
    assert out["method_coverage"] == {"method_a": 2, "method_b": 1}


def test_probe_reports_in_sync_when_counts_match(tmp_path: Path):
    """Sync = sum(per_method_count) == sum(method_coverage)."""
    mp = _scaffold_dataset_with_annotations(tmp_path)
    out = probe_annotations(tmp_path, mp)
    assert out["annotators_in_sync"] is True


def test_probe_detects_sync_drift(tmp_path: Path):
    """Central file exists but per-method copy is missing → out of sync."""
    mp = _scaffold_dataset_with_annotations(tmp_path)
    # Delete one per-method copy without touching central
    (tmp_path / "method_a" / "s1" / "annotation.json").unlink()
    out = probe_annotations(tmp_path, mp)
    assert out["annotators_in_sync"] is False


def test_probe_sample_notes_preview(tmp_path: Path):
    """Preview holds verbatim per-method dicts for up to 5 samples."""
    mp = _scaffold_dataset_with_annotations(tmp_path)
    out = probe_annotations(tmp_path, mp)
    assert len(out["sample_notes_preview"]) == 2
    assert out["sample_notes_preview"][0]["sample_key"] == "s1"
    assert out["sample_notes_preview"][0]["per_method"] == {
        "method_a": "A's verdict on s1",
        "method_b": "B's verdict on s1",
    }


# ---- refusal / empty cases --------------------------------------------

def test_probe_returns_zeros_when_no_annotations(tmp_path: Path):
    """A fresh dataset (no annotations dir) returns zero counts, no error."""
    method_paths = {"only_method": tmp_path / "only_method"}
    (tmp_path / "only_method").mkdir()
    out = probe_annotations(tmp_path, method_paths)
    assert out["central_count"] == 0
    assert out["per_method_count"] == {"only_method": 0}
    assert out["samples_with_any_note"] == []


def test_probe_works_without_method_paths(tmp_path: Path):
    """When method_paths is None, only the central count is reported."""
    _scaffold_dataset_with_annotations(tmp_path)
    out = probe_annotations(tmp_path)
    assert out["central_count"] == 2
    assert out["per_method_count"] == {}
    assert out["per_method_count_total"] == 0
    # in_sync defaults to True when there's nothing to compare against
    assert out["annotators_in_sync"] is True


def test_probe_skips_empty_per_method_dicts(tmp_path: Path):
    """A central file whose per_method dict has no non-empty slots is
    treated as 'not annotated' for counting purposes."""
    method_paths = {"method_a": tmp_path / "method_a"}
    (tmp_path / "method_a" / "s1").mkdir(parents=True)
    _write_central_annotation(tmp_path, "s1", {"method_a": ""})
    out = probe_annotations(tmp_path, method_paths)
    assert out["central_count"] == 0


def test_probe_skips_malformed_central_file(tmp_path: Path):
    """Non-JSON central file is silently skipped."""
    (tmp_path / "annotations").mkdir()
    (tmp_path / "annotations" / "bad.json").write_text("not valid {{{", encoding="utf-8")
    out = probe_annotations(tmp_path)
    assert out["central_count"] == 0


def test_probe_skips_dotfile_cache_files(tmp_path: Path):
    """Files starting with __ (theme/verdict caches) are skipped."""
    (tmp_path / "annotations").mkdir()
    cache = tmp_path / "annotations" / "__themes.json"
    cache.write_text(json.dumps({"themes": []}), encoding="utf-8")
    out = probe_annotations(tmp_path)
    assert out["central_count"] == 0


# ---- L2: samples_with_any_note truncation ----------------------------

def test_probe_caps_samples_with_any_note(tmp_path: Path):
    """L2 — when more than SAMPLES_WITH_NOTE_CAP samples are annotated,
    the on-disk list is truncated but the totals + a truncation flag
    surface the full count. Prevents bloating probe/annotations.json
    on huge datasets."""
    from era.probe.annotations import SAMPLES_WITH_NOTE_CAP

    # Scaffold (cap + 50) samples with one annotated method each.
    n = SAMPLES_WITH_NOTE_CAP + 50
    (tmp_path / "method_a").mkdir()
    for i in range(n):
        sk = f"sample_{i:04d}"
        (tmp_path / "method_a" / sk).mkdir()
        _write_central_annotation(tmp_path, sk, {"method_a": f"note {i}"})

    out = probe_annotations(tmp_path, {"method_a": tmp_path / "method_a"})
    assert out["central_count"] == n
    assert out["samples_with_any_note_total"] == n
    assert out["samples_with_any_note_truncated"] is True
    # Truncated to the cap
    assert len(out["samples_with_any_note"]) == SAMPLES_WITH_NOTE_CAP


def test_probe_does_not_truncate_under_cap(tmp_path: Path):
    """When sample count is below the cap, the list isn't truncated +
    the flag is False."""
    mp = _scaffold_dataset_with_annotations(tmp_path)
    out = probe_annotations(tmp_path, mp)
    assert out["samples_with_any_note_truncated"] is False
    assert out["samples_with_any_note_total"] == out["central_count"]


def test_probe_empty_dataset_has_truncation_fields(tmp_path: Path):
    """The empty-dataset path also carries the truncation fields (zeros)."""
    method_paths = {"only_method": tmp_path / "only_method"}
    (tmp_path / "only_method").mkdir()
    out = probe_annotations(tmp_path, method_paths)
    assert out["samples_with_any_note_total"] == 0
    assert out["samples_with_any_note_truncated"] is False
