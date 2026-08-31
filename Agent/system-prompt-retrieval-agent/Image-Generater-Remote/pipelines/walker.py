"""Dataset directory walker — discovers image pairs and returns SampleRecords.

Public API
----------
walk_dataset(source_root, categories, pair_patterns) -> list[dict]
    Walk source_root/{category}/ for each category, find sample subdirectories
    that contain at least one complete pair described by pair_patterns, compute
    SHA-256 hashes and mtime, and return a sorted list of SampleRecord-compatible
    dicts (sorted by category, then lexicographic sample-directory name).
    Index is assigned sequentially starting from 0.
    Returns an empty list if source_root does not exist.
"""

from __future__ import annotations

import hashlib
import os
from typing import Optional


# ── helpers ────────────────────────────────────────────────────────────────────


def _sha256_file(path: str, chunk_size: int = 65536) -> str:
    """Return the hex SHA-256 digest of a file read in 64 KB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _match_pair_pattern(
    sample_dir: str, pair_patterns: list[dict[str, str]]
) -> Optional[dict[str, str]]:
    """Return the first pair_pattern whose files both exist in sample_dir.

    Parameters
    ----------
    sample_dir:
        Absolute path to the sample subdirectory.
    pair_patterns:
        Each entry must have at least ``model`` and ``cloth`` keys whose values
        are file names relative to the sample directory.

    Returns
    -------
    The matched pattern dict, or ``None`` if no pattern matches.
    """
    for pattern in pair_patterns:
        model_file = pattern.get("model", "")
        cloth_file = pattern.get("cloth", "")
        if model_file and cloth_file:
            if os.path.isfile(os.path.join(sample_dir, model_file)) and os.path.isfile(
                os.path.join(sample_dir, cloth_file)
            ):
                return pattern
    return None


# ── public API ─────────────────────────────────────────────────────────────────


def walk_dataset(
    source_root: str,
    categories: list[str],
    pair_patterns: list[dict[str, str]],
) -> list[dict]:
    """Discover image pairs from a dataset directory.

    Parameters
    ----------
    source_root:
        Root directory that contains one sub-directory per category.
    categories:
        Ordered list of category names to scan.  Categories that do not exist
        under *source_root* are silently skipped.
    pair_patterns:
        List of ``{"model": "<filename>", "cloth": "<filename>"}`` dicts.  A
        sample directory is valid when it contains both files from at least one
        entry.  The first matching pattern is recorded.

    Returns
    -------
    A list of SampleRecord-compatible dicts, sorted by (category, sample_name),
    with ``index`` assigned sequentially starting from 0.  Returns ``[]`` when
    *source_root* does not exist.
    """
    if not os.path.isdir(source_root):
        return []

    records: list[dict] = []

    for category in categories:
        category_dir = os.path.join(source_root, category)
        if not os.path.isdir(category_dir):
            continue

        # Collect and sort sample sub-directories lexicographically.
        try:
            entries = sorted(os.listdir(category_dir))
        except OSError:
            continue

        for entry in entries:
            sample_dir = os.path.join(category_dir, entry)
            if not os.path.isdir(sample_dir):
                continue

            matched = _match_pair_pattern(sample_dir, pair_patterns)
            if matched is None:
                continue

            model_file = matched["model"]
            cloth_file = matched["cloth"]

            model_abs = os.path.join(sample_dir, model_file)
            cloth_abs = os.path.join(sample_dir, cloth_file)

            # Paths relative to source_root (forward slash for portability).
            relpath = os.path.join(category, entry)
            model_rel = os.path.join(relpath, model_file)
            cloth_rel = os.path.join(relpath, cloth_file)
            output_relpath = relpath  # caller fills in the actual output sub-path

            records.append(
                {
                    # index is filled in below after sorting
                    "index": -1,
                    "sample_id": f"{category}/{entry}",
                    "category": category,
                    "source_root": source_root,
                    "relpath": relpath,
                    "pair_pattern": matched,
                    "model_image": model_rel,
                    "cloth_image": cloth_rel,
                    "output_relpath": output_relpath,
                    "model_image_sha256": _sha256_file(model_abs),
                    "cloth_image_sha256": _sha256_file(cloth_abs),
                    "model_image_mtime": os.path.getmtime(model_abs),
                }
            )

    # Sort by (category, sample directory name) — categories are already
    # enumerated in order; within a category entries were already sorted, so the
    # combined list is already in the right order.  Re-sort for safety in case
    # the caller passes an unordered categories list.
    records.sort(key=lambda r: (r["category"], r["relpath"]))

    # Assign sequential indices after sorting.
    for idx, record in enumerate(records):
        record["index"] = idx

    return records
