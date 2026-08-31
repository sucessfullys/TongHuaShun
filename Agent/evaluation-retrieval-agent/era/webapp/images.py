"""Image resolution + an in-memory cache for the Stage 8 web app.

Dataset images live *outside* the workspace, on the dataset mount named by
``config.yaml`` ``data.methods[].path`` — typically a slow network filesystem.
The web app must never serve a file outside a declared method root (every
resolved path is checked back against it), and it caches image bytes in RAM so
re-visiting a sample never re-reads the network mount.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

from ..config import ERAConfig

# The output role — resolves to a method's `output_file` rather than an input.
OUTPUT_ROLE = "output"

_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}

# In-memory LRU cache of raw image bytes, capped by total size so a huge
# dataset cannot exhaust RAM. The box has plenty of memory; ~2 GB is generous.
_CACHE_MAX_BYTES = 2_000_000_000
_cache: "OrderedDict[str, tuple[bytes, str]]" = OrderedDict()
_cache_bytes = 0
_cache_lock = threading.Lock()


def resolve_image(
    cfg: ERAConfig, method_id: str, sample_key: str, role: str,
) -> Path | None:
    """Resolve one image to an absolute path, or ``None`` if it is not servable.

    ``role`` is ``"output"`` (the method's generated/edited image) or an input
    role from ``config.yaml`` ``data.input_roles``. Returns ``None`` for an
    unknown method/role, a path-traversal attempt, or a missing file.
    """
    method = next(
        (m for m in cfg.data.methods if m.get("method_id") == method_id), None
    )
    if not method or not method.get("path"):
        return None
    base = Path(method["path"]).resolve()

    if role == OUTPUT_ROLE:
        filename = method.get("output_file") or ""
    else:
        filename = cfg.data.input_roles.get(role) or ""
    if not filename:
        return None

    candidate = (base / sample_key / filename).resolve()
    # Path-traversal guard: the resolved path must stay strictly under `base`.
    if base != candidate and base not in candidate.parents:
        return None
    if not candidate.is_file():
        return None
    return candidate


def _media_type(path: Path) -> str:
    return _MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")


def load_image(
    cfg: ERAConfig, method_id: str, sample_key: str, role: str,
) -> tuple[bytes, str] | None:
    """Return ``(bytes, media_type)`` for one image, reading through the cache.

    First request for an image reads it off the (network) mount and stores it;
    every later request is served from RAM. Returns ``None`` when the image is
    not servable.
    """
    path = resolve_image(cfg, method_id, sample_key, role)
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


def prewarm(cfg: ERAConfig, sample_keys: list[str]) -> int:
    """Eagerly load every image of an iteration into the cache (best-effort).

    Runs in a background thread at server start so the first review is already
    fast. Returns the number of images loaded.
    """
    roles = [OUTPUT_ROLE, *cfg.data.input_roles]
    loaded = 0
    for sample_key in sample_keys:
        for method in cfg.data.methods:
            for role in roles:
                if load_image(cfg, method.get("method_id", ""),
                               sample_key, role) is not None:
                    loaded += 1
    return loaded


def cache_stats() -> dict:
    """Snapshot of the image cache — for diagnostics."""
    with _cache_lock:
        return {"images": len(_cache), "bytes": _cache_bytes}
