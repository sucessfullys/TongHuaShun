"""Phase C-2.3 — pick the iter's random sample window from the full dataset.

The Stage 5 planner calls :func:`pick_random_window` (via the
``era.cli sample-window`` CLI) to choose N sample_keys for the current
iter's full-mode round. The same list is stamped as ``samples_subset``
on every full-mode eval task so all methods score the same shuffled
subset (apples-to-apples within an iter, different sample set per iter
for broader dataset coverage across iters).

The selection is **deterministically random**: the seed defaults to
``sha256(project_name:iteration)`` truncated to 32 bits, so re-running
the same iter (e.g. after auto-revise scaffolds a new task plan)
produces the same sample list. Operators can pin an explicit ``seed``
to reproduce another workspace's choice.

The annotated round is unchanged — it uses the operator's annotated
sample_keys (Phase C-2). Pilot mode keeps its ``sorted(glob)[:N]``
fallback (small, stable smoke pass; not the user's target).
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any

from ..config import ERAConfig
from ..workspace import Workspace, resolve_workspace_root


SCHEMA_VERSION = 1


def _sha256_seed(project_name: str, iteration: int) -> int:
    """Stable 32-bit seed from (project_name, iteration).

    Python's built-in ``hash()`` is salted per-interpreter; sha256 is
    stable across invocations, which is what we need for reproducibility.
    """
    payload = f"{project_name}:{iteration}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big")


def _enumerate_sample_keys(method_path: Path, sample_glob: str) -> list[str]:
    """Enumerate every leaf sample dir under ``method_path`` via the
    operator-pinned ``sample_glob``.

    Returns sample_keys as POSIX-style relpaths of the matching dirs
    relative to ``method_path``. The glob may be ``"*"`` (flat layout
    — direct children) or ``"*/*"`` etc. for nested layouts.
    """
    if not method_path.is_dir():
        return []
    glob = sample_glob or "*"
    keys: list[str] = []
    for path in method_path.glob(glob):
        if not path.is_dir():
            continue
        rel = path.relative_to(method_path)
        keys.append(rel.as_posix())
    return keys


def pick_random_window(
    workspace_path: str | Path,
    *,
    n: int | None = None,
    iteration: int | None = None,
    seed: int | None = None,
) -> dict:
    """Pick N sample_keys randomly from the full dataset for one iter.

    Defaults:
    - ``iteration``: active iter from ``status.json`` (fallback to 1).
    - ``n``: ``cfg.effective_iter_sample_count()`` (the operator's
      per-iter cap clamped to ``data.sample_count``).
    - ``seed``: ``sha256(project_name:iteration)[:4]`` → deterministic
      per (workspace, iteration). Re-running the same iter produces
      the same set.

    Returns:
        {status: "ok", sample_keys: [sorted picked keys],
         seed: int, iteration: int,
         count: int, total: int,
         source: "random" | "all" (when n >= total),
         project_name: str}

    On error returns ``{error: ..., message: ...}``.
    """
    root = resolve_workspace_root(workspace_path)
    ws = Workspace(root.parent, root.name)
    if not ws.exists():
        return {"error": "not_a_workspace",
                "message": f"{root} has no status.json"}

    cfg = ERAConfig.from_yaml(root / "config.yaml")
    if not cfg.data.methods:
        return {"error": "no_methods",
                "message": "config.yaml's data.methods is empty"}

    if iteration is None:
        try:
            status = ws.read_status()
            iteration = int(status.get("iteration", 1))
        except (OSError, ValueError):
            iteration = 1
    if iteration < 1:
        return {"error": "bad_iteration",
                "message": f"iteration must be >= 1, got {iteration}"}

    if n is None:
        n = cfg.effective_iter_sample_count()
    if n is None or n < 1:
        return {"error": "bad_n",
                "message": f"n must be >= 1, got {n!r}"}

    project_name = cfg.project_name or root.name
    if seed is None:
        seed_value = _sha256_seed(project_name, iteration)
    else:
        seed_value = int(seed)

    # Enumerate from method 0 — Stage 0's data probe asserts every method
    # shares the same per_sample_dirs structure, so the first method is
    # the canonical source. (A future enhancement could intersect across
    # methods to handle drift, but the data contract today is "mirror".)
    first_method = cfg.data.methods[0]
    method_path = Path(first_method.get("path") or "")
    sample_glob = cfg.data.sample_glob or "*"
    all_keys = sorted(_enumerate_sample_keys(method_path, sample_glob))
    total = len(all_keys)

    if total == 0:
        return {"error": "no_samples",
                "message": (
                    f"no samples found under {method_path} matching "
                    f"glob {sample_glob!r}"
                ),
                "method_path": str(method_path)}

    if n >= total:
        picked = list(all_keys)
        source: str = "all"
    else:
        rng = random.Random(seed_value)
        picked = rng.sample(all_keys, n)
        source = "random"

    picked.sort()
    return {
        "status": "ok",
        "schema_version": SCHEMA_VERSION,
        "sample_keys": picked,
        "seed": seed_value,
        "iteration": iteration,
        "count": len(picked),
        "total": total,
        "source": source,
        "project_name": project_name,
    }
