"""Image resolution + an in-memory cache for the review web app.

Every image path comes straight from the run's ``detection.json`` ``samples``
map — there is no path recomputation, so co-located and split-roots datasets
serve identically. Image bytes are cached in RAM so re-visiting a sample never
re-reads the (slow network) mount.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

# The output role — resolves to a method's generated image rather than an input.
OUTPUT_ROLE = "output"

_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}

_CACHE_MAX_BYTES = 2_000_000_000
_cache: "OrderedDict[str, tuple[bytes, str]]" = OrderedDict()
_cache_bytes = 0
_cache_lock = threading.Lock()


def build_sample_index(detection: dict) -> dict:
    """``{sample_key: sample_entry}`` for O(1) lookup."""
    return {s["sample_key"]: s for s in detection.get("samples", [])}


def resolve_image(
    sample_index: dict, method_id: str, sample_key: str, role: str,
) -> Path | None:
    """Resolve one image to an absolute path, or ``None`` if not servable.

    Only paths recorded in ``detection.json`` are ever served — a request for
    an unknown (method, sample, role) is a 404, so no path traversal exists.
    """
    entry = sample_index.get(sample_key)
    if not entry:
        return None
    method = (entry.get("methods") or {}).get(method_id)
    if role == OUTPUT_ROLE:
        path = (method or {}).get("output")
    else:
        # inputs are shared across methods; fall back to the sample-level copy
        path = ((method or {}).get("inputs") or {}).get(role) \
            or (entry.get("inputs") or {}).get(role)
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return None
    return candidate


def _media_type(path: Path) -> str:
    return _MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")


def load_image(
    sample_index: dict, method_id: str, sample_key: str, role: str,
) -> tuple[bytes, str] | None:
    """Return ``(bytes, media_type)`` for one image, reading through the cache."""
    path = resolve_image(sample_index, method_id, sample_key, role)
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


def prewarm(sample_index: dict, detection: dict, sample_keys: list) -> int:
    """Eagerly load every reviewed image into the cache (best-effort)."""
    roles = [OUTPUT_ROLE, *detection.get("input_roles", {})]
    loaded = 0
    for sample_key in sample_keys:
        entry = sample_index.get(sample_key) or {}
        for method_id in entry.get("methods", {}):
            for role in roles:
                if load_image(sample_index, method_id, sample_key, role) is not None:
                    loaded += 1
    return loaded


def cache_stats() -> dict:
    with _cache_lock:
        return {"images": len(_cache), "bytes": _cache_bytes}
