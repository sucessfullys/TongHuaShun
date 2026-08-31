"""Tests for per-(sample, method) annotation persistence (era/annotate/store.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from era.annotate.store import (
    ANNOTATIONS_DIR,
    PER_METHOD_FILENAME,
    annotated_method_count,
    annotation_path,
    default_annotation,
    has_any_annotation,
    mirror_central_to_per_method,
    per_method_annotation_path,
    read_annotation,
    write_per_method_annotation,
)


# ---- annotation_path -----------------------------------------------------

def test_annotation_path_nests_under_annotations_dir(tmp_path: Path):
    p = annotation_path(tmp_path, "dress/dress_complex/sample_001")
    expected = (tmp_path / ANNOTATIONS_DIR
                / "dress/dress_complex/sample_001.json").resolve()
    assert p == expected


def test_annotation_path_handles_flat_sample_key(tmp_path: Path):
    """Single-segment sample_keys still nest under annotations/."""
    p = annotation_path(tmp_path, "sample_001")
    assert p == (tmp_path / ANNOTATIONS_DIR / "sample_001.json").resolve()


def test_annotation_path_rejects_dotdot(tmp_path: Path):
    with pytest.raises(ValueError):
        annotation_path(tmp_path, "a/../etc/passwd")
    with pytest.raises(ValueError):
        annotation_path(tmp_path, "..")


def test_annotation_path_rejects_absolute(tmp_path: Path):
    with pytest.raises(ValueError):
        annotation_path(tmp_path, "/etc/passwd")


def test_annotation_path_rejects_empty(tmp_path: Path):
    with pytest.raises(ValueError):
        annotation_path(tmp_path, "")


def test_annotation_path_rejects_hidden_segment(tmp_path: Path):
    with pytest.raises(ValueError):
        annotation_path(tmp_path, ".hidden/sample")


def test_annotation_path_rejects_empty_segment(tmp_path: Path):
    with pytest.raises(ValueError):
        annotation_path(tmp_path, "a//b")


# ---- read / write per-method --------------------------------------------

def test_read_missing_returns_default(tmp_path: Path):
    rec = read_annotation(tmp_path, "sample_001")
    assert rec == default_annotation("sample_001")
    assert rec["per_method"] == {}


def test_write_single_method_creates_file_and_roundtrips(tmp_path: Path):
    rec = write_per_method_annotation(
        tmp_path, "dress/sample_001", "method_a",
        "logo is hard to keep")
    assert rec["per_method"] == {"method_a": "logo is hard to keep"}
    assert rec["created_at"] is not None
    # Nested dirs created
    expected = annotation_path(tmp_path, "dress/sample_001")
    assert expected.is_file()
    assert expected.parent.is_dir()    # annotations/dress/
    reread = read_annotation(tmp_path, "dress/sample_001")
    assert reread["per_method"] == {"method_a": "logo is hard to keep"}


def test_write_second_method_preserves_first(tmp_path: Path):
    """Editing method_b must not clobber method_a's existing slot."""
    write_per_method_annotation(tmp_path, "s1", "method_a", "A note")
    rec = write_per_method_annotation(tmp_path, "s1", "method_b", "B note")
    assert rec["per_method"] == {"method_a": "A note", "method_b": "B note"}


def test_write_preserves_created_at_on_update(tmp_path: Path):
    first = write_per_method_annotation(tmp_path, "s1", "method_a", "v1")
    second = write_per_method_annotation(tmp_path, "s1", "method_a", "v2")
    assert first["created_at"] == second["created_at"]
    assert second["updated_at"] >= first["updated_at"]
    assert second["per_method"] == {"method_a": "v2"}


def test_clear_one_method_keeps_siblings(tmp_path: Path):
    """Clearing method_a leaves method_b intact."""
    write_per_method_annotation(tmp_path, "s1", "method_a", "A note")
    write_per_method_annotation(tmp_path, "s1", "method_b", "B note")
    rec = write_per_method_annotation(tmp_path, "s1", "method_a", "")
    assert rec["per_method"] == {"method_b": "B note"}


def test_clearing_last_slot_deletes_file(tmp_path: Path):
    """When the last non-empty slot is cleared, the file is removed."""
    write_per_method_annotation(tmp_path, "s1", "method_a", "A note")
    path = annotation_path(tmp_path, "s1")
    assert path.is_file()
    rec = write_per_method_annotation(tmp_path, "s1", "method_a", "")
    assert not path.is_file()
    assert rec == default_annotation("s1")


def test_whitespace_only_text_clears_slot(tmp_path: Path):
    write_per_method_annotation(tmp_path, "s1", "method_a", "real")
    rec = write_per_method_annotation(tmp_path, "s1", "method_a", "   \n  ")
    assert rec["per_method"] == {}


def test_per_sample_isolation(tmp_path: Path):
    """Writing one sample's annotation must not touch another sample."""
    write_per_method_annotation(tmp_path, "s1", "method_a", "note for 1")
    write_per_method_annotation(tmp_path, "s2", "method_a", "note for 2")
    a = read_annotation(tmp_path, "s1")
    b = read_annotation(tmp_path, "s2")
    assert a["per_method"] == {"method_a": "note for 1"}
    assert b["per_method"] == {"method_a": "note for 2"}


def test_atomic_write_leaves_no_tmp(tmp_path: Path):
    write_per_method_annotation(tmp_path, "s1", "method_a", "x")
    leftovers = list((tmp_path / ANNOTATIONS_DIR).rglob("*.tmp"))
    assert leftovers == []


def test_read_handles_malformed_json(tmp_path: Path):
    path = annotation_path(tmp_path, "s1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not valid {{{", encoding="utf-8")
    rec = read_annotation(tmp_path, "s1")
    assert rec["per_method"] == {}


def test_read_migrates_legacy_v1_shape(tmp_path: Path):
    """A v1 file with the old ``annotation: str`` shape is silently dropped
    (no per_method block) — the new schema is the only one we honor."""
    path = annotation_path(tmp_path, "s1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": "1.0",
        "sample_key": "s1",
        "annotation": "old shape",
    }), encoding="utf-8")
    rec = read_annotation(tmp_path, "s1")
    assert rec["per_method"] == {}


# ---- helpers -------------------------------------------------------------

def test_has_any_annotation(tmp_path: Path):
    assert has_any_annotation(tmp_path, "s1") is False
    write_per_method_annotation(tmp_path, "s1", "method_a", "")
    assert has_any_annotation(tmp_path, "s1") is False
    write_per_method_annotation(tmp_path, "s1", "method_a", "real")
    assert has_any_annotation(tmp_path, "s1") is True


def test_annotated_method_count(tmp_path: Path):
    assert annotated_method_count(tmp_path, "s1") == 0
    write_per_method_annotation(tmp_path, "s1", "method_a", "A")
    assert annotated_method_count(tmp_path, "s1") == 1
    write_per_method_annotation(tmp_path, "s1", "method_b", "B")
    assert annotated_method_count(tmp_path, "s1") == 2
    write_per_method_annotation(tmp_path, "s1", "method_a", "")
    assert annotated_method_count(tmp_path, "s1") == 1


def test_has_any_annotation_handles_bad_key_gracefully(tmp_path: Path):
    """A bad key (path traversal) returns False instead of raising."""
    assert has_any_annotation(tmp_path, "../etc/passwd") is False


def test_slash_bearing_sample_key_writes_nested_dirs(tmp_path: Path):
    """Confirm the file lands at annotations/<a>/<b>/<c>.json (mirrors
    dataset structure)."""
    write_per_method_annotation(
        tmp_path, "dress/dress_white/dress_white01", "method_a", "note")
    expected = (tmp_path / ANNOTATIONS_DIR / "dress" / "dress_white"
                / "dress_white01.json")
    assert expected.is_file()
    body = json.loads(expected.read_text(encoding="utf-8"))
    assert body["sample_key"] == "dress/dress_white/dress_white01"
    assert body["per_method"] == {"method_a": "note"}


# ---- per-method mirror copies -------------------------------------------

def _build_method_dirs(root: Path, methods: list[str],
                       samples: list[str]) -> dict[str, Path]:
    """Scaffold <root>/<method>/<sample>/ for each (method, sample) pair so
    the per-method copy has somewhere to land. Returns method_paths dict."""
    method_paths: dict[str, Path] = {}
    for method in methods:
        mdir = root / method
        for sk in samples:
            (mdir / sk).mkdir(parents=True, exist_ok=True)
        method_paths[method] = mdir
    return method_paths


def test_write_mirrors_to_method_dir(tmp_path: Path):
    """Saving a note writes the central file AND a per-method copy at
    <method>/<sample>/annotation.json containing only that method's text."""
    mp = _build_method_dirs(tmp_path, ["m_a", "m_b"], ["sample_001"])
    write_per_method_annotation(
        tmp_path, "sample_001", "m_a", "logo is hard to keep",
        method_paths=mp,
    )
    # Central file still authoritative
    central = annotation_path(tmp_path, "sample_001")
    assert central.is_file()
    # Per-method copy lands next to the (would-be) images
    copy = per_method_annotation_path(mp["m_a"], "sample_001")
    assert copy.is_file()
    body = json.loads(copy.read_text(encoding="utf-8"))
    assert body["sample_key"] == "sample_001"
    assert body["method_id"] == "m_a"
    assert body["annotation"] == "logo is hard to keep"
    assert body["created_at"] and body["updated_at"]
    # The OTHER method got nothing (since we only saved m_a's slot)
    other = per_method_annotation_path(mp["m_b"], "sample_001")
    assert not other.is_file()


def test_clear_one_slot_removes_only_that_per_method_copy(tmp_path: Path):
    """Clearing m_a's slot deletes only <m_a>/<sample>/annotation.json;
    m_b's copy stays put."""
    mp = _build_method_dirs(tmp_path, ["m_a", "m_b"], ["s1"])
    write_per_method_annotation(tmp_path, "s1", "m_a", "A", method_paths=mp)
    write_per_method_annotation(tmp_path, "s1", "m_b", "B", method_paths=mp)
    write_per_method_annotation(tmp_path, "s1", "m_a", "", method_paths=mp)

    assert not per_method_annotation_path(mp["m_a"], "s1").is_file()
    assert per_method_annotation_path(mp["m_b"], "s1").is_file()
    body = json.loads(
        per_method_annotation_path(mp["m_b"], "s1").read_text(encoding="utf-8"))
    assert body["annotation"] == "B"


def test_mirror_skips_method_without_sample_dir(tmp_path: Path):
    """A method that doesn't have this sample (no sample dir) is silently
    skipped — no error, no spurious file creation."""
    mp = _build_method_dirs(tmp_path, ["m_a"], ["s1"])
    # m_b is registered but has NO sample dir for s1
    mp["m_b"] = tmp_path / "m_b"
    mp["m_b"].mkdir()
    write_per_method_annotation(tmp_path, "s1", "m_a", "A", method_paths=mp)
    write_per_method_annotation(tmp_path, "s1", "m_b", "B", method_paths=mp)
    # m_a's copy lands
    assert per_method_annotation_path(mp["m_a"], "s1").is_file()
    # m_b has no s1/ subdir, so no copy was written
    assert not (mp["m_b"] / "s1" / PER_METHOD_FILENAME).is_file()
    # But the central file still records both slots
    assert sorted(read_annotation(tmp_path, "s1")["per_method"]) == [
        "m_a", "m_b"]


def test_per_method_copy_updates_in_place(tmp_path: Path):
    """Re-saving a slot rewrites the per-method copy (no stale content)."""
    mp = _build_method_dirs(tmp_path, ["m_a"], ["s1"])
    write_per_method_annotation(tmp_path, "s1", "m_a", "v1", method_paths=mp)
    write_per_method_annotation(tmp_path, "s1", "m_a", "v2", method_paths=mp)
    body = json.loads(
        per_method_annotation_path(mp["m_a"], "s1").read_text("utf-8"))
    assert body["annotation"] == "v2"


def test_write_without_method_paths_does_not_mirror(tmp_path: Path):
    """When method_paths is None (legacy call sites), only the central
    file is written — no per-method copies."""
    mp = _build_method_dirs(tmp_path, ["m_a"], ["s1"])
    write_per_method_annotation(tmp_path, "s1", "m_a", "note")  # no mp
    assert annotation_path(tmp_path, "s1").is_file()
    assert not per_method_annotation_path(mp["m_a"], "s1").is_file()


# ---- mirror_central_to_per_method backfill ------------------------------

def test_mirror_backfills_existing_central_files(tmp_path: Path):
    """The backfill helper walks <dataset>/annotations/ and writes a
    per-method copy for every (sample, method) slot it finds."""
    mp = _build_method_dirs(tmp_path, ["m_a", "m_b"], ["s1", "s2"])
    # Write the central files directly (simulating an operator who already
    # annotated before the mirroring feature shipped).
    for sk in ("s1", "s2"):
        path = annotation_path(tmp_path, sk)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "schema_version": "2.0",
            "sample_key": sk,
            "per_method": {"m_a": f"A-{sk}", "m_b": f"B-{sk}"},
            "created_at": "2026-05-26T12:00:00+00:00",
            "updated_at": "2026-05-26T12:00:00+00:00",
        }), encoding="utf-8")
    summary = mirror_central_to_per_method(tmp_path, mp)
    assert summary["scanned"] == 2     # two central files
    assert summary["written"] == 4     # 2 samples × 2 methods
    assert summary["skipped_no_dir"] == 0

    # Spot-check the resulting per-method copies
    for sk in ("s1", "s2"):
        for method in ("m_a", "m_b"):
            body = json.loads(
                per_method_annotation_path(mp[method], sk).read_text("utf-8"))
            assert body["method_id"] == method
            assert body["annotation"] == f"{method.upper()[-1]}-{sk}"


def test_mirror_clears_stale_per_method_copy(tmp_path: Path):
    """If a central file no longer carries m_a (operator cleared the slot
    out-of-band) but a per-method copy lingers, mirror should delete it."""
    mp = _build_method_dirs(tmp_path, ["m_a", "m_b"], ["s1"])
    # Pre-existing stale per-method copy for m_a
    stale = per_method_annotation_path(mp["m_a"], "s1")
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text(json.dumps({
        "schema_version": "2.0", "sample_key": "s1", "method_id": "m_a",
        "annotation": "stale note",
        "created_at": "x", "updated_at": "x",
    }), encoding="utf-8")
    # Central file only has m_b populated (m_a was cleared)
    central = annotation_path(tmp_path, "s1")
    central.parent.mkdir(parents=True, exist_ok=True)
    central.write_text(json.dumps({
        "schema_version": "2.0", "sample_key": "s1",
        "per_method": {"m_b": "B"},
        "created_at": "x", "updated_at": "x",
    }), encoding="utf-8")
    summary = mirror_central_to_per_method(tmp_path, mp)
    assert summary["deleted"] >= 1     # the stale m_a copy
    assert not stale.is_file()
    # And m_b's copy was written
    assert per_method_annotation_path(mp["m_b"], "s1").is_file()


def test_mirror_empty_annotations_dir_is_noop(tmp_path: Path):
    """No central files → mirror reports zero work, no error."""
    mp = _build_method_dirs(tmp_path, ["m_a"], ["s1"])
    summary = mirror_central_to_per_method(tmp_path, mp)
    assert summary == {"scanned": 0, "written": 0,
                       "skipped_no_dir": 0, "deleted": 0}


def test_mirror_skips_malformed_central_file(tmp_path: Path):
    """A non-JSON central file is skipped silently rather than crashing."""
    mp = _build_method_dirs(tmp_path, ["m_a"], ["s1"])
    bad = annotation_path(tmp_path, "s1")
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("not valid json {{{", encoding="utf-8")
    summary = mirror_central_to_per_method(tmp_path, mp)
    assert summary["scanned"] == 0
    assert summary["written"] == 0


def test_per_method_annotation_path_rejects_traversal(tmp_path: Path):
    """Sample_key with .. segments is rejected (same guard as central path)."""
    method_root = tmp_path / "m_a"
    method_root.mkdir()
    with pytest.raises(ValueError):
        per_method_annotation_path(method_root, "../../etc/passwd")
