"""``Workspace`` — scaffolds and tracks an ERA project workspace.

The workspace splits into **global** content (Stage 0 init + Stage 1 research,
done once) and **per-iteration** content (``iter_NNN/``, Stages 2-10). A
``current`` symlink points at the active iteration. Stage 0 creates iteration 1.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

# Global (workspace-level) directories — created once.
GLOBAL_DIRS = [
    "probe",
    "research",
    "shared",
    "logs/iterations",
]

# Per-iteration directories — created inside every iter_NNN/.
ITER_DIRS = [
    "design/candidates",
    "design/reviews",
    "experiments/plans",
    "experiments/configs",
    "experiments/results",
    "experiments/logs",
    "serving",
    "comparison",
    "human",
    "react",
    "deliverable",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_workspace_root(workspace_path: str | Path) -> Path:
    """Normalize a path to a workspace's stable project root.

    Accepts either the workspace root itself or its ``current`` iteration
    pointer (the ``current`` symlink or ``current.txt``) and returns the
    resolved root. The orchestration entry points share this so a caller may
    pass whichever workspace path it happens to hold.
    """
    root = Path(workspace_path).expanduser()
    if root.name == "current" and (root.parent / "status.json").exists():
        root = root.parent
    return root.resolve()


class Workspace:
    """Filesystem handle for a single ERA project workspace."""

    def __init__(self, base_dir: str | Path, project_name: str):
        self.name = project_name
        self.root = Path(base_dir) / project_name

    # ---- existence -------------------------------------------------------

    def exists(self) -> bool:
        """True if an initialized workspace already lives here."""
        return (self.root / "status.json").exists()

    # ---- scaffolding -----------------------------------------------------

    def scaffold(self) -> None:
        """Create the workspace root and all global directories."""
        self.root.mkdir(parents=True, exist_ok=True)
        for rel in GLOBAL_DIRS:
            (self.root / rel).mkdir(parents=True, exist_ok=True)

    def create_iteration(
        self, n: int, *, parent_feedback: dict | None = None,
    ) -> Path:
        """Create ``iter_NNN/`` with its sub-tree and ``iteration.json``.

        ``parent_feedback`` (Stage 9 carry-forward) is recorded verbatim into
        ``iteration.json`` when given — it typically carries workspace-relative
        paths to the prior iter's ``human/human_labels.json``,
        ``react/evolution_state.json``, and (optionally)
        ``react/literature_update_brief.md``. Stages 2 and 7 read this field on
        the new iteration; absent it (iter_001), they fall back to single-iter
        behavior.
        """
        iter_dir = self.root / f"iter_{n:03d}"
        for rel in ITER_DIRS:
            (iter_dir / rel).mkdir(parents=True, exist_ok=True)
        iter_json = iter_dir / "iteration.json"
        if not iter_json.exists():
            # Lifecycle stage lives in status.json (the single source of truth);
            # iteration.json holds only per-iteration metadata.
            _write_json_atomic(iter_json, {
                "iteration": n,
                "created_at": _now_iso(),
                "parent_feedback": parent_feedback,
            })
        elif parent_feedback is not None:
            existing = json.loads(iter_json.read_text(encoding="utf-8"))
            if existing.get("parent_feedback") is None:
                existing["parent_feedback"] = parent_feedback
                _write_json_atomic(iter_json, existing)
        return iter_dir

    def set_current(self, n: int) -> None:
        """Point ``current`` at ``iter_NNN`` atomically.

        Falls back to a ``current.txt`` pointer file if the filesystem
        rejects symlinks.
        """
        iter_name = f"iter_{n:03d}"
        link = self.root / "current"
        tmp = self.root / ".current.tmp"
        try:
            if tmp.is_symlink() or tmp.exists():
                tmp.unlink()
            tmp.symlink_to(iter_name)
            os.replace(tmp, link)
        except (OSError, NotImplementedError):
            if tmp.is_symlink() or tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            if link.is_symlink() or link.exists():
                try:
                    link.unlink()
                except OSError:
                    pass
            (self.root / "current.txt").write_text(
                iter_name + "\n", encoding="utf-8"
            )

    def iter_path(self) -> Path:
        """Resolve the active iteration directory."""
        link = self.root / "current"
        if link.is_symlink():
            return self.root / os.readlink(link)
        ptr = self.root / "current.txt"
        if ptr.exists():
            return self.root / ptr.read_text(encoding="utf-8").strip()
        return self.root / "iter_001"

    # ---- file writers ----------------------------------------------------

    def write_file(self, rel: str, content: str) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def write_json(self, rel: str, data: dict) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(path, data)

    # ---- status ----------------------------------------------------------

    def read_status(self) -> dict:
        path = self.root / "status.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {}

    def write_status(self, status: dict) -> None:
        _write_json_atomic(self.root / "status.json", status)

    def update_stage(self, stage: str) -> None:
        status = self.read_status()
        status["stage"] = stage
        status["updated_at"] = _now_iso()
        self.write_status(status)


def _write_json_atomic(path: Path, data: dict) -> None:
    """Write JSON via a temp file + ``os.replace`` so readers never see a
    partial file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)
