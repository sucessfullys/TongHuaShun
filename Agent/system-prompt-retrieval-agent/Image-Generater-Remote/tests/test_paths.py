"""Tests for server/paths.py — path validation and mirroring utilities."""

from __future__ import annotations

import os

import pytest

from server.paths import (
    ensure_output_dir,
    is_path_allowed,
    mirror_path,
    validate_path_exists,
)


# ── is_path_allowed ────────────────────────────────────────────────────────


class TestIsPathAllowed:
    def test_path_directly_inside_root(self, tmp_path):
        root = str(tmp_path)
        candidate = str(tmp_path / "file.txt")
        assert is_path_allowed(candidate, [root]) is True

    def test_path_deeply_nested(self, tmp_path):
        root = str(tmp_path)
        candidate = str(tmp_path / "a" / "b" / "c" / "file.jpg")
        assert is_path_allowed(candidate, [root]) is True

    def test_path_equal_to_root(self, tmp_path):
        root = str(tmp_path)
        assert is_path_allowed(root, [root]) is True

    def test_path_outside_all_roots(self, tmp_path):
        root = str(tmp_path / "allowed")
        candidate = str(tmp_path / "other" / "file.txt")
        assert is_path_allowed(candidate, [root]) is False

    def test_empty_allowed_roots(self, tmp_path):
        candidate = str(tmp_path / "file.txt")
        assert is_path_allowed(candidate, []) is False

    def test_traversal_attack_dotdot(self, tmp_path):
        """../../etc/passwd style should be rejected if it escapes the root."""
        allowed = str(tmp_path / "safe")
        # Build a path that *looks* like it starts with the safe dir but
        # uses '..' to escape it.
        attack = str(tmp_path / "safe" / ".." / ".." / "etc" / "passwd")
        assert is_path_allowed(attack, [allowed]) is False

    def test_traversal_attack_prefix_trick(self, tmp_path):
        """A path like /allowed_extra should not match /allowed root."""
        sibling = str(tmp_path / "allowed_extra" / "secret.txt")
        root = str(tmp_path / "allowed")
        assert is_path_allowed(sibling, [root]) is False

    def test_multiple_roots_first_matches(self, tmp_path):
        root_a = str(tmp_path / "a")
        root_b = str(tmp_path / "b")
        candidate = str(tmp_path / "a" / "data.txt")
        assert is_path_allowed(candidate, [root_a, root_b]) is True

    def test_multiple_roots_second_matches(self, tmp_path):
        root_a = str(tmp_path / "a")
        root_b = str(tmp_path / "b")
        candidate = str(tmp_path / "b" / "data.txt")
        assert is_path_allowed(candidate, [root_a, root_b]) is True

    def test_symlink_resolved(self, tmp_path):
        """A symlink pointing outside the allowed root must be rejected."""
        outside = tmp_path / "outside"
        outside.mkdir()
        secret = outside / "secret.txt"
        secret.write_text("secret")

        allowed = tmp_path / "allowed"
        allowed.mkdir()
        link = allowed / "link.txt"
        link.symlink_to(secret)

        # The symlink is inside `allowed/` but resolves to `outside/`
        assert is_path_allowed(str(link), [str(allowed)]) is False


# ── validate_path_exists ────────────────────────────────────────────────────


class TestValidatePathExists:
    def test_valid_existing_path(self, tmp_path):
        root = str(tmp_path)
        f = tmp_path / "sample.jpg"
        f.write_text("data")
        result = validate_path_exists(str(f), [root])
        assert result == os.path.realpath(str(f))

    def test_returns_realpath(self, tmp_path):
        root = str(tmp_path)
        f = tmp_path / "img.jpg"
        f.write_text("x")
        result = validate_path_exists(str(f), [root])
        assert result == os.path.realpath(str(f))

    def test_raises_path_not_found_for_missing(self, tmp_path):
        root = str(tmp_path)
        missing = str(tmp_path / "nonexistent.jpg")
        with pytest.raises(ValueError, match="path_not_found"):
            validate_path_exists(missing, [root])

    def test_raises_path_not_allowed_for_outside(self, tmp_path):
        allowed = str(tmp_path / "safe")
        (tmp_path / "safe").mkdir()
        outside = tmp_path / "other"
        outside.mkdir()
        f = outside / "file.txt"
        f.write_text("x")
        with pytest.raises(ValueError, match="path_not_allowed"):
            validate_path_exists(str(f), [allowed])

    def test_existing_path_outside_returns_not_allowed(self, tmp_path):
        """Existence check must happen before allowlist check — but both
        errors must be distinguishable."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        f = outside_dir / "file.txt"
        f.write_text("hi")
        with pytest.raises(ValueError) as exc_info:
            validate_path_exists(str(f), [str(allowed_dir)])
        assert "path_not_allowed" in str(exc_info.value)


# ── mirror_path ────────────────────────────────────────────────────────────


class TestMirrorPath:
    def test_basic_mirror(self, tmp_path):
        src_root = str(tmp_path / "src")
        out_root = str(tmp_path / "out")
        result = mirror_path(src_root, "category/img.jpg", out_root)
        assert result == os.path.join(out_root, "category", "img.jpg")

    def test_parent_directory_created(self, tmp_path):
        src_root = str(tmp_path / "src")
        out_root = str(tmp_path / "out")
        mirror_path(src_root, "a/b/c/img.jpg", out_root)
        expected_parent = os.path.join(out_root, "a", "b", "c")
        assert os.path.isdir(expected_parent)

    def test_flat_relpath(self, tmp_path):
        src_root = str(tmp_path / "src")
        out_root = str(tmp_path / "out")
        result = mirror_path(src_root, "img.jpg", out_root)
        assert result == os.path.join(out_root, "img.jpg")
        # Parent dir (out_root itself) should be created
        assert os.path.isdir(out_root)

    def test_leading_separator_stripped(self, tmp_path):
        src_root = str(tmp_path / "src")
        out_root = str(tmp_path / "out")
        result = mirror_path(src_root, "/category/img.jpg", out_root)
        assert result == os.path.join(out_root, "category", "img.jpg")

    def test_idempotent_directory_creation(self, tmp_path):
        """Calling mirror_path twice must not raise."""
        src_root = str(tmp_path / "src")
        out_root = str(tmp_path / "out")
        mirror_path(src_root, "cat/img.jpg", out_root)
        mirror_path(src_root, "cat/img2.jpg", out_root)
        assert os.path.isdir(os.path.join(out_root, "cat"))


# ── ensure_output_dir ──────────────────────────────────────────────────────


class TestEnsureOutputDir:
    def test_creates_directory(self, tmp_path):
        out_root = str(tmp_path / "out")
        result = ensure_output_dir(out_root, "sub/dir/file.jpg")
        expected = os.path.join(out_root, "sub", "dir")
        assert result == expected
        assert os.path.isdir(expected)

    def test_returns_correct_path(self, tmp_path):
        out_root = str(tmp_path / "out")
        result = ensure_output_dir(out_root, "a/b/c/img.png")
        assert result == os.path.join(out_root, "a", "b", "c")

    def test_flat_relpath_returns_output_root(self, tmp_path):
        out_root = str(tmp_path / "out")
        result = ensure_output_dir(out_root, "img.png")
        assert result == out_root
        assert os.path.isdir(out_root)

    def test_idempotent(self, tmp_path):
        out_root = str(tmp_path / "out")
        ensure_output_dir(out_root, "sub/img.jpg")
        # Second call must not raise
        result = ensure_output_dir(out_root, "sub/img2.jpg")
        assert os.path.isdir(result)
