"""Shared cross-project serving-recipe memory.

Captures lessons from a successful Stage 6 judge serve (install commands,
working launch flags, runner template snippets, known quirks) so the
*next* project's Stage 5-6 doesn't re-derive them. Storage is per-user,
cross-project, at:

    ~/.era/memory/serving_recipes/<model_slug>__<backend>.json

The directory is namespaced under ``~/.era/`` so it never collides with
Claude Code's per-project memory at
``~/.claude/projects/<slug>/memory/``. Phase A ships the passive read/
write API; Phase C wires the active capture into Stage 6's serve-task
completion path.

Schema v1 is documented in the approved plan; see :data:`SCHEMA_VERSION`
and the ``REQUIRED_TOP_KEYS`` guard in :func:`write_recipe`.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from .experiment_results import write_json_atomic

SCHEMA_VERSION = "1.0"

# Backends a recipe may carry — mirrors ``serving.backend`` valid values in
# ``ERAConfig``. Catches typos at write time.
KNOWN_BACKENDS = ("ms_swift", "vllm", "lmdeploy")

# Top-level keys ``write_recipe`` requires before persisting a recipe.
# Optional sub-fields under ``install`` / ``launch`` are validated soft —
# missing them is allowed (a partial recipe is better than no recipe).
REQUIRED_TOP_KEYS = ("model_id", "backend")

# Filename slug separator: ``<model_slug>__<backend>.json``.
SLUG_SEP = "__"


def _now_iso() -> str:
    # Millisecond precision so ``list_recipes()``'s order-by-``last_validated``
    # sort is stable across writes that arrive in the same wall-clock second
    # (which is normal on a fast machine doing back-to-back writes).
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def memory_root() -> Path:
    """The on-disk root of the shared serving-memory store.

    Override at runtime by setting ``ERA_MEMORY_DIR=/abs/path`` — tests
    use this to redirect to a tmpdir without touching the user's real
    ``~/.era/`` tree.
    """
    override = os.environ.get("ERA_MEMORY_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".era" / "memory").resolve()


def serving_recipes_dir() -> Path:
    """The directory holding one ``.json`` per (model, backend) pair."""
    d = memory_root() / "serving_recipes"
    return d


def _ident(text: str) -> str:
    """Normalize a model id into a filesystem-safe slug (matches the
    probe's ``_ident`` convention in :mod:`era.probe.data`)."""
    ident = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").lower()).strip("_")
    return ident or "model"


def _model_slug(model_id: str) -> str:
    return _ident(model_id)


def _recipe_slug(model_id: str, backend: str) -> str:
    return f"{_model_slug(model_id)}{SLUG_SEP}{_ident(backend)}"


def _recipe_path(model_id: str, backend: str) -> Path:
    return serving_recipes_dir() / f"{_recipe_slug(model_id, backend)}.json"


# ---- read / write API ----------------------------------------------------

def read_recipe(model_id: str, backend: str) -> dict | None:
    """Look up the recipe for ``(model_id, backend)``; ``None`` when absent.

    Lookup is by exact ``model_id`` (case-insensitive via the slug
    normalization). Returns the full recipe dict including the
    ``last_validated`` timestamp.
    """
    if not model_id or not backend:
        return None
    path = _recipe_path(model_id, backend)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def write_recipe(
    recipe: dict, *, overwrite: bool = True,
    captured_by_workspace: str | None = None,
    captured_by_iteration: int | None = None,
) -> dict:
    """Persist a recipe atomically; return the written record.

    ``recipe`` must carry ``model_id`` and ``backend``. The function
    fills in ``schema_version``, ``model_slug``, ``captured_at``,
    ``last_validated``, and ``captured_by_workspace`` /
    ``captured_by_iteration`` when supplied.

    Existing recipes are overwritten by default (so re-running a
    successful serve refreshes ``last_validated``). Set
    ``overwrite=False`` to refuse re-writes — returns the existing
    record instead.
    """
    if not isinstance(recipe, dict):
        raise ValueError("recipe must be a dict")
    for key in REQUIRED_TOP_KEYS:
        if not recipe.get(key):
            raise ValueError(f"recipe missing required key: {key!r}")

    model_id = str(recipe["model_id"])
    backend = str(recipe["backend"])
    path = _recipe_path(model_id, backend)

    existing = read_recipe(model_id, backend) if path.is_file() else None
    if existing is not None and not overwrite:
        return existing

    now = _now_iso()
    record = {
        "schema_version": SCHEMA_VERSION,
        "model_id": model_id,
        "model_slug": _model_slug(model_id),
        "backend": backend,
        "captured_at": (existing or {}).get("captured_at") or now,
        "last_validated": now,
        **{
            k: v for k, v in recipe.items()
            if k not in {"schema_version", "model_slug",
                         "captured_at", "last_validated",
                         "unknown_backend"}
        },
    }
    # Flag unknown backends on the record so Stage 6's reader (and the CLI
    # ``list`` view) can surface a warning. Operators with custom serving
    # stacks can still write recipes; they just carry the flag.
    if backend not in KNOWN_BACKENDS:
        record["unknown_backend"] = True
    # Audit fields: preserve from the existing record when the caller didn't
    # supply them on the rewrite. Without this fallback, a second project
    # re-validating an existing recipe via ``write_recipe(recipe)`` (no
    # kwargs) would silently lose the original captured_by_workspace /
    # captured_by_iteration provenance.
    if captured_by_workspace is None and existing is not None:
        captured_by_workspace = existing.get("captured_by_workspace")
    if captured_by_iteration is None and existing is not None:
        captured_by_iteration = existing.get("captured_by_iteration")
    if captured_by_workspace is not None:
        record["captured_by_workspace"] = captured_by_workspace
    if captured_by_iteration is not None:
        record["captured_by_iteration"] = captured_by_iteration

    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, record)
    return record


def list_recipes() -> list[dict]:
    """A summary row per stored recipe, sorted by ``last_validated`` desc.

    Each row: ``{slug, model_id, backend, captured_at, last_validated,
    captured_by_workspace?}``. Use to drive the CLI ``list`` verb +
    Stage 5-6's "is there a recipe for this judge?" lookup.
    """
    d = serving_recipes_dir()
    if not d.is_dir():
        return []
    rows: list[dict] = []
    for path in d.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        rows.append({
            "slug": path.stem,
            "model_id": data.get("model_id"),
            "backend": data.get("backend"),
            "captured_at": data.get("captured_at"),
            "last_validated": data.get("last_validated"),
            "captured_by_workspace": data.get("captured_by_workspace"),
        })
    rows.sort(
        key=lambda r: (r.get("last_validated") or "", r.get("slug") or ""),
        reverse=True,
    )
    return rows


def forget_recipe(model_id: str, backend: str) -> dict:
    """Delete the recipe file for ``(model_id, backend)``.

    Returns ``{"status": "ok", "slug": ..., "deleted": bool}``.
    Missing files are a no-op (``deleted: False``).
    """
    if not model_id or not backend:
        raise ValueError("model_id and backend are required")
    path = _recipe_path(model_id, backend)
    slug = _recipe_slug(model_id, backend)
    if not path.is_file():
        return {"status": "ok", "slug": slug, "deleted": False}
    path.unlink()
    return {"status": "ok", "slug": slug, "deleted": True}
