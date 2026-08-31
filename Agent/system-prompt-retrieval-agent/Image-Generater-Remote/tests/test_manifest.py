"""Tests for server/manifest.py — deterministic manifest builder with SHA-256."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from server.manifest import (
    build_manifest,
    compute_file_sha256,
    verify_manifest,
)


# ── Sample data fixture ────────────────────────────────────────────────────

SAMPLE_ITEMS = [
    {
        "index": 0,
        "sample_id": "s0001",
        "category": "upper",
        "source_root": "/data/src",
        "relpath": "upper/pair_001",
        "pair_pattern": {"model": "model.jpg", "cloth": "cloth.jpg"},
        "model_image": "upper/pair_001/model.jpg",
        "cloth_image": "upper/pair_001/cloth.jpg",
        "output_relpath": "upper/pair_001",
        "model_image_sha256": "abc123",
        "cloth_image_sha256": "def456",
        "model_image_mtime": 1700000000.0,
    },
    {
        "index": 1,
        "sample_id": "s0002",
        "category": "lower",
        "source_root": "/data/src",
        "relpath": "lower/pair_002",
        "pair_pattern": {"model": "model.jpg", "cloth": "cloth.jpg"},
        "model_image": "lower/pair_002/model.jpg",
        "cloth_image": "lower/pair_002/cloth.jpg",
        "output_relpath": "lower/pair_002",
        "model_image_sha256": "111aaa",
        "cloth_image_sha256": "222bbb",
        "model_image_mtime": 1700001000.0,
    },
]


# ── compute_file_sha256 ────────────────────────────────────────────────────


class TestComputeFileSha256:
    def test_known_hash(self, tmp_path):
        f = tmp_path / "hello.txt"
        content = b"hello world\n"
        f.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert compute_file_sha256(str(f)) == expected

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        expected = hashlib.sha256(b"").hexdigest()
        assert compute_file_sha256(str(f)) == expected

    def test_binary_file(self, tmp_path):
        data = bytes(range(256)) * 1000  # 256 KB of binary
        f = tmp_path / "binary.bin"
        f.write_bytes(data)
        expected = hashlib.sha256(data).hexdigest()
        assert compute_file_sha256(str(f)) == expected

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError)):
            compute_file_sha256(str(tmp_path / "nonexistent.txt"))

    def test_returns_lowercase_hex(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_bytes(b"test")
        result = compute_file_sha256(str(f))
        assert result == result.lower()
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)


# ── build_manifest ─────────────────────────────────────────────────────────


class TestBuildManifest:
    def test_returns_required_keys(self):
        manifest = build_manifest("/data/src", SAMPLE_ITEMS)
        assert "manifest_hash" in manifest
        assert "source_root" in manifest
        assert "created_at" in manifest
        assert "items" in manifest

    def test_source_root_stored_verbatim(self):
        manifest = build_manifest("/data/src", SAMPLE_ITEMS)
        assert manifest["source_root"] == "/data/src"

    def test_items_preserved(self):
        manifest = build_manifest("/data/src", SAMPLE_ITEMS)
        assert manifest["items"] == SAMPLE_ITEMS

    def test_manifest_hash_is_sha256_hex(self):
        manifest = build_manifest("/data/src", SAMPLE_ITEMS)
        h = manifest["manifest_hash"]
        assert isinstance(h, str)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic_same_input(self):
        m1 = build_manifest("/data/src", SAMPLE_ITEMS)
        m2 = build_manifest("/data/src", SAMPLE_ITEMS)
        assert m1["manifest_hash"] == m2["manifest_hash"]

    def test_deterministic_independent_of_item_dict_construction_order(self):
        """Two item lists with the same data but different key insertion
        order must produce the same hash (sort_keys=True guarantees this)."""
        items_a = [{"index": 0, "sample_id": "x", "category": "upper"}]
        # Reverse key insertion order
        items_b = [{"category": "upper", "sample_id": "x", "index": 0}]
        m_a = build_manifest("/src", items_a)
        m_b = build_manifest("/src", items_b)
        assert m_a["manifest_hash"] == m_b["manifest_hash"]

    def test_different_items_produce_different_hash(self):
        items_alt = copy.deepcopy(SAMPLE_ITEMS)
        items_alt[0]["sample_id"] = "TAMPERED"
        m1 = build_manifest("/data/src", SAMPLE_ITEMS)
        m2 = build_manifest("/data/src", items_alt)
        assert m1["manifest_hash"] != m2["manifest_hash"]

    def test_hash_matches_canonical_json_sha256(self):
        """The hash must equal SHA-256 of the canonical JSON bytes."""
        manifest = build_manifest("/data/src", SAMPLE_ITEMS)
        canonical = json.dumps(
            SAMPLE_ITEMS, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        expected_hash = hashlib.sha256(canonical).hexdigest()
        assert manifest["manifest_hash"] == expected_hash

    def test_empty_items_list(self):
        manifest = build_manifest("/data/src", [])
        assert isinstance(manifest["manifest_hash"], str)
        assert len(manifest["manifest_hash"]) == 64

    def test_does_not_mutate_input(self):
        items_copy = copy.deepcopy(SAMPLE_ITEMS)
        build_manifest("/data/src", SAMPLE_ITEMS)
        assert SAMPLE_ITEMS == items_copy


# ── verify_manifest ────────────────────────────────────────────────────────


class TestVerifyManifest:
    def test_valid_manifest_returns_true(self):
        manifest = build_manifest("/data/src", SAMPLE_ITEMS)
        assert verify_manifest(manifest["manifest_hash"], SAMPLE_ITEMS) is True

    def test_tampered_item_returns_false(self):
        manifest = build_manifest("/data/src", SAMPLE_ITEMS)
        tampered = copy.deepcopy(SAMPLE_ITEMS)
        tampered[0]["sample_id"] = "HACKED"
        assert verify_manifest(manifest["manifest_hash"], tampered) is False

    def test_extra_item_returns_false(self):
        manifest = build_manifest("/data/src", SAMPLE_ITEMS)
        extra = copy.deepcopy(SAMPLE_ITEMS) + [{"index": 99, "sample_id": "extra"}]
        assert verify_manifest(manifest["manifest_hash"], extra) is False

    def test_missing_item_returns_false(self):
        manifest = build_manifest("/data/src", SAMPLE_ITEMS)
        fewer = copy.deepcopy(SAMPLE_ITEMS[:1])
        assert verify_manifest(manifest["manifest_hash"], fewer) is False

    def test_wrong_hash_string_returns_false(self):
        assert verify_manifest("0" * 64, SAMPLE_ITEMS) is False

    def test_empty_hash_returns_false(self):
        assert verify_manifest("", SAMPLE_ITEMS) is False

    def test_empty_items_round_trip(self):
        manifest = build_manifest("/data/src", [])
        assert verify_manifest(manifest["manifest_hash"], []) is True

    def test_roundtrip_determinism(self):
        """build then verify must succeed even when called in different Python
        scopes (no shared state between build and verify)."""
        h = build_manifest("/root", SAMPLE_ITEMS)["manifest_hash"]
        # Reconstruct items from scratch (simulate loading from disk)
        items_reloaded = copy.deepcopy(SAMPLE_ITEMS)
        assert verify_manifest(h, items_reloaded) is True
