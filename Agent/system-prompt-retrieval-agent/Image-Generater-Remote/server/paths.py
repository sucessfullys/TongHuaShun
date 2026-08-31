"""Path validation and mirroring utilities for the remote image service.

All path operations resolve symlinks via os.path.realpath() to prevent
directory traversal attacks before any existence or allowlist check.
"""

from __future__ import annotations

import os


def is_path_allowed(path: str, allowed_roots: list[str]) -> bool:
    """Return True if *path* is under any of the *allowed_roots*.

    Both the candidate path and every allowed root are resolved through
    ``os.path.realpath()`` so that symlinks and ``..`` components cannot
    be used to escape the allowlist.

    Args:
        path: Candidate filesystem path (need not exist).
        allowed_roots: List of directory paths that define the allowlist.

    Returns:
        True when the resolved path starts with at least one resolved root,
        False otherwise (including when *allowed_roots* is empty).
    """
    if not allowed_roots:
        return False

    resolved = os.path.realpath(path)

    for root in allowed_roots:
        resolved_root = os.path.realpath(root)
        # Ensure the root ends with a separator so that a path like
        # /allowed_rootX does not incorrectly match /allowed_root.
        if not resolved_root.endswith(os.sep):
            resolved_root_with_sep = resolved_root + os.sep
        else:
            resolved_root_with_sep = resolved_root

        if resolved == os.path.realpath(root) or resolved.startswith(resolved_root_with_sep):
            return True

    return False


def validate_path_exists(path: str, allowed_roots: list[str]) -> str:
    """Validate that *path* exists and is under *allowed_roots*.

    Raises:
        ValueError: With error code ``path_not_found`` when the path does not
            exist on the filesystem.
        ValueError: With error code ``path_not_allowed`` when the path exists
            but falls outside every allowed root.

    Returns:
        The resolved (realpath) version of *path* on success.
    """
    resolved = os.path.realpath(path)

    if not os.path.exists(resolved):
        raise ValueError(
            f"path_not_found: '{path}' does not exist (resolved: '{resolved}')"
        )

    if not is_path_allowed(path, allowed_roots):
        raise ValueError(
            f"path_not_allowed: '{path}' is not under any allowed root "
            f"(resolved: '{resolved}', roots: {allowed_roots})"
        )

    return resolved


def mirror_path(source_root: str, relpath: str, output_root: str) -> str:
    """Compute the mirrored output path and create its parent directories.

    Given ``source_root`` and a relative path ``relpath`` (which may already
    appear under ``source_root`` or may be a bare relative path), compute the
    equivalent path under ``output_root`` and ensure the parent directory
    exists.

    Examples::

        mirror_path("/data/src", "category/img.jpg", "/data/out")
        # -> "/data/out/category/img.jpg"  (creates /data/out/category/)

    Args:
        source_root: Absolute path of the dataset source root (not used for
            the actual mirroring calculation, kept for API symmetry and future
            validation use).
        relpath: Relative path of the file within *source_root*.
        output_root: Absolute path of the output root directory.

    Returns:
        The full mirrored output path (parent directories are created).
    """
    # Strip any leading separator so os.path.join behaves correctly.
    clean_relpath = relpath.lstrip(os.sep)
    output_path = os.path.join(output_root, clean_relpath)
    parent = os.path.dirname(output_path)
    os.makedirs(parent, exist_ok=True)
    return output_path


def ensure_output_dir(output_root: str, relpath: str) -> str:
    """Create the output sub-directory for *relpath* and return its path.

    The returned directory is ``output_root / dirname(relpath)``.  All
    intermediate directories are created with ``exist_ok=True``.

    Args:
        output_root: Root of the output tree.
        relpath: Relative path whose directory component is appended to
            *output_root*.

    Returns:
        The created (or already-existing) directory path.
    """
    clean_relpath = relpath.lstrip(os.sep)
    sub_dir = os.path.join(output_root, os.path.dirname(clean_relpath))
    os.makedirs(sub_dir, exist_ok=True)
    return sub_dir
