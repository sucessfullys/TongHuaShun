"""Tests for pipelines.walker.walk_dataset."""

from __future__ import annotations

import os
import tempfile
import hashlib

import pytest

from pipelines.walker import walk_dataset


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_dataset(base: str, structure: dict) -> None:
    """Create a minimal mock dataset under *base*.

    ``structure`` maps  category -> [sample_name, ...]  to create.
    Each sample directory receives ``model.jpg`` and ``cloth.jpg`` placeholder
    files so that the default pair-pattern matches.
    """
    for category, samples in structure.items():
        for sample in samples:
            sample_dir = os.path.join(base, category, sample)
            os.makedirs(sample_dir, exist_ok=True)
            # Write distinct content so SHA-256 values differ across samples.
            for fname in ("model.jpg", "cloth.jpg"):
                fpath = os.path.join(sample_dir, fname)
                with open(fpath, "wb") as fh:
                    fh.write(f"{category}/{sample}/{fname}".encode())


DEFAULT_PAIR_PATTERNS = [{"model": "model.jpg", "cloth": "cloth.jpg"}]


# ── fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def dataset_dir():
    """Yield the path to a temporary dataset with 3 categories × 2 samples."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_dataset(
            tmpdir,
            {
                "dress": ["0001", "0002"],
                "jacket": ["0001", "0002"],
                "shirt": ["0001", "0002"],
            },
        )
        yield tmpdir


# ── tests ──────────────────────────────────────────────────────────────────────


class TestWalkDataset:
    def test_correct_count(self, dataset_dir):
        """3 categories × 2 samples = 6 records."""
        records = walk_dataset(dataset_dir, ["dress", "jacket", "shirt"], DEFAULT_PAIR_PATTERNS)
        assert len(records) == 6

    def test_sequential_index(self, dataset_dir):
        """Indices must be 0..N-1 after sorting."""
        records = walk_dataset(dataset_dir, ["dress", "jacket", "shirt"], DEFAULT_PAIR_PATTERNS)
        indices = [r["index"] for r in records]
        assert indices == list(range(6))

    def test_sort_order(self, dataset_dir):
        """Records must be sorted by category then sample name."""
        records = walk_dataset(dataset_dir, ["dress", "jacket", "shirt"], DEFAULT_PAIR_PATTERNS)
        sample_ids = [r["sample_id"] for r in records]
        assert sample_ids == [
            "dress/0001",
            "dress/0002",
            "jacket/0001",
            "jacket/0002",
            "shirt/0001",
            "shirt/0002",
        ]

    def test_required_fields_present(self, dataset_dir):
        """Every record must carry all SampleRecord-compatible keys."""
        required = {
            "index",
            "sample_id",
            "category",
            "source_root",
            "relpath",
            "pair_pattern",
            "model_image",
            "cloth_image",
            "output_relpath",
            "model_image_sha256",
            "cloth_image_sha256",
            "model_image_mtime",
        }
        records = walk_dataset(dataset_dir, ["dress", "jacket", "shirt"], DEFAULT_PAIR_PATTERNS)
        for record in records:
            missing = required - record.keys()
            assert not missing, f"Record missing keys: {missing}"

    def test_sample_id_format(self, dataset_dir):
        """sample_id must be '<category>/<sample_name>'."""
        records = walk_dataset(dataset_dir, ["dress"], DEFAULT_PAIR_PATTERNS)
        for record in records:
            assert record["sample_id"] == f"{record['category']}/{os.path.basename(record['relpath'])}"

    def test_source_root_stored(self, dataset_dir):
        """source_root field must equal the argument passed in."""
        records = walk_dataset(dataset_dir, ["dress"], DEFAULT_PAIR_PATTERNS)
        for record in records:
            assert record["source_root"] == dataset_dir

    def test_sha256_correct(self, dataset_dir):
        """SHA-256 values must match the file contents written in the fixture."""
        records = walk_dataset(dataset_dir, ["dress"], DEFAULT_PAIR_PATTERNS)
        for record in records:
            model_abs = os.path.join(dataset_dir, record["model_image"])
            cloth_abs = os.path.join(dataset_dir, record["cloth_image"])
            expected_model_sha = hashlib.sha256(open(model_abs, "rb").read()).hexdigest()
            expected_cloth_sha = hashlib.sha256(open(cloth_abs, "rb").read()).hexdigest()
            assert record["model_image_sha256"] == expected_model_sha
            assert record["cloth_image_sha256"] == expected_cloth_sha

    def test_mtime_is_float(self, dataset_dir):
        """model_image_mtime must be a float (seconds since epoch)."""
        records = walk_dataset(dataset_dir, ["dress"], DEFAULT_PAIR_PATTERNS)
        for record in records:
            assert isinstance(record["model_image_mtime"], float)

    def test_pair_pattern_recorded(self, dataset_dir):
        """pair_pattern must equal the matched pattern dict."""
        records = walk_dataset(dataset_dir, ["dress"], DEFAULT_PAIR_PATTERNS)
        for record in records:
            assert record["pair_pattern"] == DEFAULT_PAIR_PATTERNS[0]

    def test_image_paths_relative_to_source_root(self, dataset_dir):
        """model_image and cloth_image must be relative to source_root."""
        records = walk_dataset(dataset_dir, ["dress"], DEFAULT_PAIR_PATTERNS)
        for record in records:
            # Joining source_root with model_image must yield a real file.
            assert os.path.isfile(os.path.join(dataset_dir, record["model_image"]))
            assert os.path.isfile(os.path.join(dataset_dir, record["cloth_image"]))

    def test_empty_source_root_returns_empty(self):
        """Non-existent source_root must return [] without raising."""
        result = walk_dataset("/does/not/exist", ["dress"], DEFAULT_PAIR_PATTERNS)
        assert result == []

    def test_empty_directory_returns_empty(self):
        """An existing but empty source_root must return []."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = walk_dataset(tmpdir, ["dress"], DEFAULT_PAIR_PATTERNS)
            assert result == []

    def test_missing_category_skipped(self, dataset_dir):
        """Categories absent from source_root must be silently skipped."""
        records = walk_dataset(
            dataset_dir,
            ["dress", "nonexistent_category"],
            DEFAULT_PAIR_PATTERNS,
        )
        # Only 'dress' exists → 2 records, no error.
        assert len(records) == 2
        for record in records:
            assert record["category"] == "dress"

    def test_only_requested_categories_included(self, dataset_dir):
        """Passing a subset of categories returns only those categories."""
        records = walk_dataset(dataset_dir, ["jacket"], DEFAULT_PAIR_PATTERNS)
        assert len(records) == 2
        for record in records:
            assert record["category"] == "jacket"

    def test_sample_missing_pair_file_skipped(self, dataset_dir):
        """A sample directory that lacks the cloth image must be skipped."""
        # Remove cloth.jpg from dress/0001
        os.remove(os.path.join(dataset_dir, "dress", "0001", "cloth.jpg"))
        records = walk_dataset(dataset_dir, ["dress"], DEFAULT_PAIR_PATTERNS)
        # dress/0002 remains; dress/0001 is skipped.
        assert len(records) == 1
        assert records[0]["sample_id"] == "dress/0002"

    def test_multiple_pair_patterns_first_match_wins(self):
        """When multiple patterns are given, the first matching one is recorded."""
        alt_patterns = [
            {"model": "front.jpg", "cloth": "top.jpg"},
            {"model": "model.jpg", "cloth": "cloth.jpg"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            # Only the second pattern's files exist.
            sample_dir = os.path.join(tmpdir, "dress", "0001")
            os.makedirs(sample_dir)
            for fname in ("model.jpg", "cloth.jpg"):
                with open(os.path.join(sample_dir, fname), "wb") as fh:
                    fh.write(fname.encode())
            records = walk_dataset(tmpdir, ["dress"], alt_patterns)
            assert len(records) == 1
            assert records[0]["pair_pattern"] == alt_patterns[1]

    def test_category_sort_respects_passed_order(self, dataset_dir):
        """Sort is by category string value, not by position in categories list."""
        # Passing categories in reverse order; output should still be alpha-sorted.
        records = walk_dataset(
            dataset_dir, ["shirt", "jacket", "dress"], DEFAULT_PAIR_PATTERNS
        )
        categories_seen = [r["category"] for r in records]
        assert categories_seen == sorted(categories_seen)
