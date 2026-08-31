"""Deterministic manifest builder with SHA-256 integrity verification.

All hashing is deterministic: ``build_manifest`` and ``verify_manifest``
produce identical results for identical inputs regardless of Python runtime
or platform.

Schemas note: functions accept and return plain ``dict`` objects so that this
module can be imported independently of Pydantic (useful in test and CLI
contexts).  Callers that hold ``InputManifest`` / ``SampleRecord`` Pydantic
models should call ``.model_dump()`` before passing them in.
"""

from __future__ import annotations

import hashlib
import json


# ── Low-level hash helpers ─────────────────────────────────────────────────


def compute_file_sha256(path: str) -> str:
    """Return the SHA-256 hex digest of the file at *path*.

    The file is read in 64 KiB chunks to avoid loading large files into
    memory all at once.

    Args:
        path: Absolute or relative path to the file.

    Returns:
        Lower-case hexadecimal SHA-256 digest string (64 characters).

    Raises:
        FileNotFoundError: If *path* does not exist.
        OSError: For any other I/O error.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json(items: list[dict]) -> bytes:
    """Serialise *items* to a canonical, deterministic JSON byte string.

    Uses ``sort_keys=True`` and no extra whitespace so that the output is
    identical for the same logical data regardless of insertion order or
    formatting preferences.
    """
    return json.dumps(items, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _hash_items(items: list[dict]) -> str:
    """Return the SHA-256 hex digest of the canonical JSON of *items*."""
    return hashlib.sha256(_canonical_json(items)).hexdigest()


# ── Public API ─────────────────────────────────────────────────────────────


def build_manifest(source_root: str, items: list[dict]) -> dict:
    """Build an ``InputManifest``-compatible dict from *items*.

    The ``manifest_hash`` field is the SHA-256 digest over the canonical JSON
    representation of *items* (keys sorted, no whitespace).  This is
    deterministic: identical *items* always produce the same hash.

    Args:
        source_root: Dataset source root path (stored verbatim in the manifest).
        items: List of ``SampleRecord``-compatible dicts.

    Returns:
        A plain dict matching the ``InputManifest`` schema::

            {
                "manifest_hash": "<sha256-hex>",
                "source_root": "<source_root>",
                "created_at": "",   # left blank; caller may fill in
                "items": [...]
            }
    """
    manifest_hash = _hash_items(items)
    return {
        "manifest_hash": manifest_hash,
        "source_root": source_root,
        "created_at": "",
        "items": list(items),
    }


def verify_manifest(manifest_hash: str, items: list[dict]) -> bool:
    """Return ``True`` when *items* hash to *manifest_hash*.

    Recomputes the SHA-256 over the canonical JSON of *items* using the same
    algorithm as :func:`build_manifest` and compares the result to
    *manifest_hash* using a constant-time comparison to avoid timing attacks.

    Args:
        manifest_hash: Expected hex-digest string (as stored in the manifest).
        items: List of ``SampleRecord``-compatible dicts.

    Returns:
        ``True`` if the recomputed hash matches *manifest_hash*, ``False``
        otherwise (including when *manifest_hash* has the wrong length/format).
    """
    computed = _hash_items(items)
    # hmac.compare_digest requires strings of equal length; fall back to simple
    # equality which is fine here because we do not need timing safety against
    # local-only callers — but use compare_digest for correctness.
    import hmac as _hmac
    try:
        return _hmac.compare_digest(computed, manifest_hash)
    except TypeError:
        return False
