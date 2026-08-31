"""Tests for the probe-driven annotation walker (era/annotate/data.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from era.annotate.data import (
    FLAT_METHOD_ID,
    OUTPUT_ROLE,
    detect_flat_layout,
    walk_dataset,
)


# --- fixture: a multi-method, depth-3 dataset that mirrors the real one -----

def _make_real_shape_dataset(root: Path) -> None:
    """Two methods × two categories × two backgrounds × three samples, each
    with per-method output filenames (the real dataset has e.g.
    tryon_result_flux2klein.png for method 1 — *not* a generic name)."""
    methods_outputs = {
        "method_a": "tryon_result_method_a.png",
        "method_b": "tryon_result_method_b.png",
    }
    for method, output_file in methods_outputs.items():
        for category in ("dress", "upper"):
            for bg in (f"{category}_white", f"{category}_complex"):
                for n in (1, 2, 3):
                    sample_dir = root / method / category / bg / f"{bg}{n:02d}"
                    sample_dir.mkdir(parents=True)
                    (sample_dir / "input_cloth.png").write_bytes(
                        f"PNG-cloth-{category}-{bg}{n}".encode())
                    (sample_dir / "input_model.png").write_bytes(
                        f"PNG-model-{category}-{bg}{n}".encode())
                    (sample_dir / output_file).write_bytes(
                        f"PNG-{method}-{category}-{bg}{n}".encode())


def test_walk_dataset_finds_real_shape(tmp_path: Path):
    _make_real_shape_dataset(tmp_path)
    ds = walk_dataset(tmp_path)
    assert ds.methods == ["method_a", "method_b"]
    # Sample keys carry slashes (depth-3 relpath under each method dir)
    assert len(ds.sample_keys) == 12      # 2 categories × 2 bgs × 3 samples
    # Same sample key appears under both methods (apples-to-apples)
    for sk in ds.sample_keys:
        assert ds.methods_for_sample(sk) == ["method_a", "method_b"]
        assert "/" in sk


def test_walk_dataset_resolves_per_method_output_filenames(tmp_path: Path):
    """Method A uses tryon_result_method_a.png; Method B uses _method_b.
    The walker must pick the right one per method."""
    _make_real_shape_dataset(tmp_path)
    ds = walk_dataset(tmp_path)
    sk = ds.sample_keys[0]
    a = ds.image_path_for("method_a", sk, OUTPUT_ROLE)
    b = ds.image_path_for("method_b", sk, OUTPUT_ROLE)
    assert a is not None and a.name == "tryon_result_method_a.png"
    assert b is not None and b.name == "tryon_result_method_b.png"


def test_walk_dataset_resolves_input_roles(tmp_path: Path):
    _make_real_shape_dataset(tmp_path)
    ds = walk_dataset(tmp_path)
    sk = ds.sample_keys[0]
    # The probe should have detected both input filenames as input_roles
    assert "input_cloth" in ds.input_roles
    assert "input_model" in ds.input_roles
    cloth = ds.image_path_for("method_a", sk, "input_cloth")
    model = ds.image_path_for("method_a", sk, "input_model")
    assert cloth is not None and cloth.name == "input_cloth.png"
    assert model is not None and model.name == "input_model.png"


def test_walk_dataset_path_traversal_guard(tmp_path: Path):
    """A maliciously-shaped sample_key (e.g. '../..') must resolve to None."""
    _make_real_shape_dataset(tmp_path)
    ds = walk_dataset(tmp_path)
    assert ds.image_path_for(
        "method_a", "../../../etc/passwd", "input_cloth") is None


def test_walk_dataset_unknown_method_returns_none(tmp_path: Path):
    _make_real_shape_dataset(tmp_path)
    ds = walk_dataset(tmp_path)
    sk = ds.sample_keys[0]
    assert ds.image_path_for("method_z", sk, OUTPUT_ROLE) is None


def test_walk_dataset_unknown_role_returns_none(tmp_path: Path):
    _make_real_shape_dataset(tmp_path)
    ds = walk_dataset(tmp_path)
    sk = ds.sample_keys[0]
    assert ds.image_path_for("method_a", sk, "bogus_role") is None


def test_walk_dataset_all_roles_includes_output(tmp_path: Path):
    _make_real_shape_dataset(tmp_path)
    ds = walk_dataset(tmp_path)
    assert ds.output_role == OUTPUT_ROLE
    assert OUTPUT_ROLE in ds.all_roles
    # Inputs are sorted, output last
    assert ds.all_roles[-1] == OUTPUT_ROLE


def test_walk_dataset_carries_probe_summary(tmp_path: Path):
    _make_real_shape_dataset(tmp_path)
    ds = walk_dataset(tmp_path)
    assert ds.probe_summary["layout"] == "per_sample_dirs"
    assert ds.probe_summary["sample_glob"]      # populated


def test_walk_dataset_empty_root(tmp_path: Path):
    """An empty (or non-method-shaped) root yields zero methods but does
    not raise — the web app surfaces this via warnings."""
    ds = walk_dataset(tmp_path)
    assert ds.methods == []
    assert ds.sample_keys == []


def test_walk_dataset_missing_root_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        walk_dataset(tmp_path / "does-not-exist")


# --- additional fixture: shallow single-method layout ---------------------

def test_walk_dataset_handles_shallow_two_method_layout(tmp_path: Path):
    """A shallow two-method layout (samples are direct children of each
    method dir, no nested categories) is recognized by the probe as a
    multi-method dataset with single-segment sample_keys."""
    for method in ("m_alpha", "m_beta"):
        for sk in ("s001", "s002"):
            sample_dir = tmp_path / method / sk
            sample_dir.mkdir(parents=True)
            (sample_dir / "input_cloth.png").write_bytes(b"cloth")
            (sample_dir / "input_model.png").write_bytes(b"model")
            (sample_dir / "tryon_result.png").write_bytes(b"result")
    ds = walk_dataset(tmp_path)
    assert ds.methods == ["m_alpha", "m_beta"]
    # Single-segment sample_keys — no slashes when samples are direct children.
    assert ds.sample_keys == ["s001", "s002"]
    # Same sample appears under both methods
    for sk in ds.sample_keys:
        assert ds.methods_for_sample(sk) == ["m_alpha", "m_beta"]


# --- flat-image layout (per-image notes mode) ------------------------------

def _make_bucketed_flat_dataset(root: Path) -> None:
    """A directory of standalone images bucketed one level deep (good/bad),
    like the real ``cases/`` poster dataset — no per-sample dirs, no
    methods, mixed extensions."""
    for name in ("img_a.png", "img_b.jpg", "shared.png"):
        (root / "good").mkdir(parents=True, exist_ok=True)
        (root / "good" / name).write_bytes(f"good-{name}".encode())
    for name in ("img_c.png", "shared.png", "img_d.webp"):
        (root / "bad").mkdir(parents=True, exist_ok=True)
        (root / "bad" / name).write_bytes(f"bad-{name}".encode())


def test_detect_flat_layout_bucketed(tmp_path: Path):
    _make_bucketed_flat_dataset(tmp_path)
    buckets = detect_flat_layout(tmp_path)
    assert buckets is not None
    assert set(buckets) == {"good", "bad"}
    assert len(buckets["good"]) == 3 and len(buckets["bad"]) == 3


def test_detect_flat_layout_single_bucket(tmp_path: Path):
    """Images directly under root (no subdirs) → one unnamed bucket."""
    for name in ("a.png", "b.jpg", "c.webp"):
        (tmp_path / name).write_bytes(name.encode())
    buckets = detect_flat_layout(tmp_path)
    assert buckets is not None
    assert set(buckets) == {""}
    assert len(buckets[""]) == 3


def test_detect_flat_layout_rejects_nested(tmp_path: Path):
    """A normal <method>/<sample_dir>/<images> layout is NOT flat."""
    _make_real_shape_dataset(tmp_path)
    assert detect_flat_layout(tmp_path) is None


def test_detect_flat_layout_ignores_annotations_dir(tmp_path: Path):
    """The app's own annotations/ subdir must not break flat detection."""
    _make_bucketed_flat_dataset(tmp_path)
    (tmp_path / "annotations" / "good").mkdir(parents=True)
    (tmp_path / "annotations" / "good" / "img_a.png.json").write_text("{}")
    buckets = detect_flat_layout(tmp_path)
    assert buckets is not None
    assert set(buckets) == {"good", "bad"}


def test_walk_dataset_flat_bucketed(tmp_path: Path):
    _make_bucketed_flat_dataset(tmp_path)
    ds = walk_dataset(tmp_path)
    assert ds.methods == [FLAT_METHOD_ID]
    assert ds.probe_summary["layout"] == "flat_files"
    assert ds.probe_summary["confidence"] == "high"
    assert ds.input_roles == {}
    # 3 good + 3 bad = 6 samples; bucket preserved in the sample_key.
    assert len(ds.sample_keys) == 6
    assert "good/img_a.png" in ds.sample_keys
    assert "bad/img_d.webp" in ds.sample_keys
    # Same filename in both buckets stays distinct (bucket-prefixed key).
    assert "good/shared.png" in ds.sample_keys
    assert "bad/shared.png" in ds.sample_keys
    for sk in ds.sample_keys:
        assert ds.methods_for_sample(sk) == [FLAT_METHOD_ID]


def test_walk_dataset_flat_resolves_output_image(tmp_path: Path):
    _make_bucketed_flat_dataset(tmp_path)
    ds = walk_dataset(tmp_path)
    p = ds.image_path_for(FLAT_METHOD_ID, "good/img_a.png", OUTPUT_ROLE)
    assert p is not None and p.name == "img_a.png"
    assert p.read_bytes() == b"good-img_a.png"
    # The shared filename resolves to the right bucket's file.
    bad_shared = ds.image_path_for(FLAT_METHOD_ID, "bad/shared.png", OUTPUT_ROLE)
    assert bad_shared is not None
    assert bad_shared.read_bytes() == b"bad-shared.png"


def test_walk_dataset_flat_non_output_role_is_none(tmp_path: Path):
    _make_bucketed_flat_dataset(tmp_path)
    ds = walk_dataset(tmp_path)
    assert ds.image_path_for(
        FLAT_METHOD_ID, "good/img_a.png", "input_cloth") is None


def test_walk_dataset_flat_unknown_sample_is_none(tmp_path: Path):
    _make_bucketed_flat_dataset(tmp_path)
    ds = walk_dataset(tmp_path)
    assert ds.image_path_for(
        FLAT_METHOD_ID, "good/nope.png", OUTPUT_ROLE) is None


def test_walk_dataset_flat_traversal_guard(tmp_path: Path):
    """A crafted sample_key not in the flat map resolves to None (the map is
    the allowlist — only real images can be addressed)."""
    _make_bucketed_flat_dataset(tmp_path)
    ds = walk_dataset(tmp_path)
    assert ds.image_path_for(
        FLAT_METHOD_ID, "../../etc/passwd", OUTPUT_ROLE) is None
