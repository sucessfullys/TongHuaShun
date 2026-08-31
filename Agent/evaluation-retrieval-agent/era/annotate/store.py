"""Per-(sample, method) annotation persistence.

Each annotation file lives at ``<dataset_root>/annotations/<sample_key>.json``.
The ``annotations/`` subdirectory mirrors the dataset's own directory shape,
so a sample whose key is ``dress/dress_complex_background/sample_001`` lands
at ``<dataset>/annotations/dress/dress_complex_background/sample_001.json``.
This keeps annotation files out of the method dirs (no per-method privilege),
keeps the dataset root from being cluttered with hundreds of flat files, and
lets Stage 7 / Stage 9 walk a single subtree to find every annotation.

JSON shape (``schema_version: "2.0"``)::

    {
      "schema_version": "2.0",
      "sample_key": "dress/dress_complex_background/sample_001",
      "per_method": {
        "tryon_results": "the logo is blurred",
        "tryon_results_pe": "color preserved but sleeve warped"
      },
      "created_at": "2026-05-26T12:34:56+00:00",
      "updated_at": "2026-05-26T12:34:56+00:00"
    }

Writing an empty string for a method clears that method's slot. When the
last non-empty slot is cleared the file is deleted entirely (no hollow
records left behind for Stage 7 to chase).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..orchestration.experiment_results import write_json_atomic

SCHEMA_VERSION = "2.0"
ANNOTATIONS_DIR = "annotations"

# Filename of the per-method copy that lands inside each sample's image
# directory (alongside the tryon_result image). The central file at
# ``<dataset>/annotations/<sample_key>.json`` stays the source of truth;
# per-method copies are derived so any agent reading
# ``<method>/<sample_key>/`` finds the relevant note right next to the
# image without consulting a separate annotations subtree.
PER_METHOD_FILENAME = "annotation.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize_sample_key(sample_key: str) -> str:
    """Reject dangerous sample keys; normalize separators to forward slashes.

    The walker only ever produces sample keys that are POSIX-style relpaths
    of real on-disk directories under a method dir, so they cannot escape.
    But annotations also flow via the FastAPI URL, so we re-validate here
    against ``..`` segments, absolute paths, NUL bytes, leading dots, and
    Windows-style backslashes.
    """
    if not sample_key:
        raise ValueError("sample_key may not be empty")
    if "\x00" in sample_key:
        raise ValueError(f"sample_key contains NUL: {sample_key!r}")
    if sample_key.startswith("/") or sample_key.startswith("\\"):
        raise ValueError(f"sample_key may not be absolute: {sample_key!r}")
    normalized = sample_key.replace("\\", "/")
    parts = normalized.split("/")
    for part in parts:
        if not part:
            raise ValueError(f"sample_key has empty segment: {sample_key!r}")
        if part == "..":
            raise ValueError(
                f"sample_key may not contain '..' segments: {sample_key!r}")
        if part.startswith("."):
            raise ValueError(
                f"sample_key segment may not start with '.': {sample_key!r}")
    return normalized


def annotation_path(dataset_root: Path, sample_key: str) -> Path:
    """Filesystem path of one sample's annotation file.

    The returned path is guaranteed to be inside
    ``<dataset_root>/annotations/`` — calls that resolve outside raise
    :class:`ValueError`.
    """
    normalized = _normalize_sample_key(sample_key)
    annotations_dir = (Path(dataset_root) / ANNOTATIONS_DIR).resolve()
    candidate = (annotations_dir / f"{normalized}.json").resolve()
    # Defense in depth — even after the segment-level guard above, re-check
    # the resolved path is under annotations_dir.
    if (annotations_dir != candidate
            and annotations_dir not in candidate.parents):
        raise ValueError(
            f"resolved annotation path escapes annotations dir: "
            f"{candidate!r}"
        )
    return candidate


def default_annotation(sample_key: str) -> dict:
    """The shape returned when no annotation file exists for a sample."""
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_key": sample_key,
        "per_method": {},
        "created_at": None,
        "updated_at": None,
    }


def read_annotation(dataset_root: Path, sample_key: str) -> dict:
    """Load one sample's annotation; fresh default when absent or malformed."""
    path = annotation_path(dataset_root, sample_key)
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            # Best-effort migration: v1 ``annotation: str`` style is silently
            # dropped (the new schema is per-method). Legitimate v2 reads
            # get the per_method dict filled out.
            per_method = data.get("per_method")
            if not isinstance(per_method, dict):
                per_method = {}
            return {
                "schema_version": SCHEMA_VERSION,
                "sample_key": data.get("sample_key") or sample_key,
                "per_method": {
                    str(k): str(v) for k, v in per_method.items()
                    if isinstance(v, str)
                },
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
            }
    return default_annotation(sample_key)


def per_method_annotation_path(method_path: Path, sample_key: str) -> Path:
    """Filesystem path of one method's annotation copy.

    Lands at ``<method_path>/<sample_key>/annotation.json`` so an agent
    walking the dataset finds the note right next to the corresponding
    ``tryon_result_*.png``. The sample-key guard from
    :func:`_normalize_sample_key` still applies — the resolved path must
    stay strictly under the method directory.
    """
    normalized = _normalize_sample_key(sample_key)
    method_root = Path(method_path).resolve()
    candidate = (method_root / normalized / PER_METHOD_FILENAME).resolve()
    if (method_root != candidate
            and method_root not in candidate.parents):
        raise ValueError(
            f"resolved per-method annotation path escapes method dir: "
            f"{candidate!r}"
        )
    return candidate


def _write_per_method_copy(
    method_path: Path, sample_key: str, method_id: str,
    text: str, created_at: str | None, updated_at: str,
) -> str:
    """Write or delete one ``<method>/<sample_key>/annotation.json`` copy.

    Returns ``"written"`` / ``"deleted"`` / ``"skipped_no_dir"``. The sample
    dir must already exist (the walker discovered it); we never create a
    sample dir from scratch — that would change the dataset shape. A
    missing sample dir is silently skipped (it's the normal "method does
    not contain this sample" case).
    """
    cleaned = (text or "").strip()
    try:
        path = per_method_annotation_path(method_path, sample_key)
    except ValueError:
        return "skipped_no_dir"
    sample_dir = path.parent
    if not sample_dir.is_dir():
        # Method doesn't have this sample → no place to drop the copy.
        # Also clean up a stale file if one was somehow left behind.
        if path.is_file():
            path.unlink(missing_ok=True)
        return "skipped_no_dir"
    if not cleaned:
        if path.is_file():
            path.unlink(missing_ok=True)
        return "deleted"
    record = {
        "schema_version": SCHEMA_VERSION,
        "sample_key": sample_key,
        "method_id": method_id,
        "annotation": text,
        "created_at": created_at or updated_at,
        "updated_at": updated_at,
    }
    write_json_atomic(path, record)
    return "written"


def write_per_method_annotation(
    dataset_root: Path, sample_key: str, method_id: str, text: str,
    *,
    method_paths: dict[str, Path] | None = None,
) -> dict:
    """Set or clear one method's slot in a sample's annotation file.

    - ``text`` is a non-empty string → set ``per_method[method_id] = text``.
    - ``text`` is blank/whitespace → clear ``per_method[method_id]``. When
      the resulting ``per_method`` is empty, the central file is deleted
      entirely (no hollow records).
    - All other slots are left untouched (so concurrent edits on
      different methods of the same sample don't clobber each other).

    When ``method_paths`` is supplied (the FastAPI app passes
    :attr:`Dataset.method_paths`), this *also* mirrors the change to the
    per-method copy at
    ``<method_paths[method_id]>/<sample_key>/annotation.json`` so an
    agent walking the dataset finds the note next to the corresponding
    output image. Mirror writes are best-effort — a write failure (e.g.
    a read-only mount, a missing sample dir) is recorded but does not
    fail the central write.
    """
    if not method_id:
        raise ValueError("method_id may not be empty")
    path = annotation_path(dataset_root, sample_key)
    existing = read_annotation(dataset_root, sample_key)
    per_method = dict(existing.get("per_method") or {})

    cleaned = (text or "").strip()
    if cleaned:
        per_method[method_id] = text
    else:
        per_method.pop(method_id, None)

    now = _now_iso()
    created_at = existing.get("created_at") or now

    if not per_method:
        if path.is_file():
            path.unlink()
        record = default_annotation(sample_key)
    else:
        record = {
            "schema_version": SCHEMA_VERSION,
            "sample_key": sample_key,
            "per_method": per_method,
            "created_at": created_at,
            "updated_at": now,
        }
        write_json_atomic(path, record)

    # Mirror this one method's slot to its per-method copy. Best-effort;
    # never let a mirror failure block the central write.
    if method_paths is not None:
        target = method_paths.get(method_id)
        if target is not None:
            try:
                _write_per_method_copy(
                    target, sample_key, method_id,
                    cleaned and text or "",
                    created_at, now,
                )
            except OSError:
                pass

    return record


def has_any_annotation(dataset_root: Path, sample_key: str) -> bool:
    """True iff any method's slot in this sample's annotation is non-empty."""
    try:
        path = annotation_path(dataset_root, sample_key)
    except ValueError:
        return False
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    per_method = (data.get("per_method") if isinstance(data, dict) else None)
    if not isinstance(per_method, dict):
        return False
    return any(
        isinstance(v, str) and v.strip() for v in per_method.values()
    )


def mirror_central_to_per_method(
    dataset_root: Path, method_paths: dict[str, Path],
) -> dict:
    """Walk every central annotation file and write per-method copies.

    Idempotent — re-running with no changes is a no-op for any sample
    whose per-method copies already match the central record. Used at
    server startup to backfill existing operator annotations the first
    time this feature ships, and also driveable from the CLI via
    ``era.cli annotate-mirror`` if the operator wants a one-shot sync.

    Returns a summary::

        {
          "scanned": int,           # central files inspected
          "written": int,           # per-method copies created or updated
          "skipped_no_dir": int,    # sample missing from a method dir
          "deleted": int,           # stale per-method copies removed
        }
    """
    annotations_dir = (Path(dataset_root) / ANNOTATIONS_DIR).resolve()
    if not annotations_dir.is_dir():
        return {"scanned": 0, "written": 0, "skipped_no_dir": 0, "deleted": 0}

    scanned = written = skipped = deleted = 0
    for path in sorted(annotations_dir.rglob("*.json")):
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        sample_key = data.get("sample_key")
        per_method = data.get("per_method")
        if not sample_key or not isinstance(per_method, dict):
            continue
        scanned += 1
        created_at = data.get("created_at")
        updated_at = data.get("updated_at") or _now_iso()
        # For every method we know about — even ones not in this sample's
        # per_method dict — make sure the per-method copy is in sync. That
        # way a method whose slot was *cleared* gets its stale copy removed.
        for method_id, method_root in method_paths.items():
            text = per_method.get(method_id) or ""
            try:
                outcome = _write_per_method_copy(
                    method_root, sample_key, method_id, text,
                    created_at, updated_at,
                )
            except (OSError, ValueError):
                continue
            if outcome == "written":
                written += 1
            elif outcome == "deleted":
                deleted += 1
            elif outcome == "skipped_no_dir":
                skipped += 1
    return {
        "scanned": scanned,
        "written": written,
        "skipped_no_dir": skipped,
        "deleted": deleted,
    }


def annotated_method_count(
    dataset_root: Path, sample_key: str,
) -> int:
    """How many of a sample's methods carry a non-empty annotation."""
    try:
        path = annotation_path(dataset_root, sample_key)
    except ValueError:
        return 0
    if not path.is_file():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    per_method = (data.get("per_method") if isinstance(data, dict) else None)
    if not isinstance(per_method, dict):
        return 0
    return sum(
        1 for v in per_method.values()
        if isinstance(v, str) and v.strip()
    )
