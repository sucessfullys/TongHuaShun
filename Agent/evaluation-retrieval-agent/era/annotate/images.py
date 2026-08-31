"""Resolve and serve dataset images for the annotation web app.

Delegates path resolution to :meth:`Dataset.image_path_for` (which carries
the probe-discovered per-method output filenames and per-role input
filenames, plus a path-traversal guard). This module keeps the LRU byte
cache + media-type lookup separately so the FastAPI handler stays a
two-line wrapper.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

from .data import Dataset

_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}

# LRU cache of (bytes, media_type), capped by total size so a huge dataset
# cannot exhaust RAM. 1 GB is plenty for a typical annotation pass.
_CACHE_MAX_BYTES = 1_000_000_000
_cache: "OrderedDict[str, tuple[bytes, str]]" = OrderedDict()
_cache_bytes = 0
_cache_lock = threading.Lock()


def _media_type(path: Path) -> str:
    return _MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")


def resolve_image(
    dataset: Dataset, method_id: str, sample_key: str, role: str,
) -> Path | None:
    """Return the on-disk path, or ``None`` when not servable.

    Thin wrapper over :meth:`Dataset.image_path_for` — kept as a separate
    symbol so the FastAPI handler can be unit-tested by monkeypatching it.
    """
    return dataset.image_path_for(method_id, sample_key, role)


def load_image(
    dataset: Dataset, method_id: str, sample_key: str, role: str,
) -> tuple[bytes, str] | None:
    """Return ``(bytes, media_type)`` for one image, through the LRU cache."""
    path = resolve_image(dataset, method_id, sample_key, role)
    if path is None:
        return None
    key = str(path)

    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return hit

    try:
        data = path.read_bytes()
    except OSError:
        return None
    entry = (data, _media_type(path))

    with _cache_lock:
        global _cache_bytes
        if key not in _cache:
            _cache[key] = entry
            _cache_bytes += len(data)
            while _cache_bytes > _CACHE_MAX_BYTES and len(_cache) > 1:
                _, (evicted, _m) = _cache.popitem(last=False)
                _cache_bytes -= len(evicted)
        else:
            _cache.move_to_end(key)
    return entry


def cache_stats() -> dict:
    """Snapshot of the image cache — for /api/health diagnostics."""
    with _cache_lock:
        return {"images": len(_cache), "bytes": _cache_bytes}
