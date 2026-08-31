"""Filesystem path resolution for the ERA repo."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """ERA repo root — the parent of the ``era/`` package directory.

    The repo root contains ``plugin/``, ``era/`` and ``knowledge/``.
    """
    return Path(__file__).resolve().parents[1]


def workspaces_dir() -> Path:
    """Parent directory under which all project workspaces are created."""
    return repo_root() / "workspaces"


def era_state_dir() -> Path:
    """Cross-workspace runtime-state root for the ERA repo (``.era/``).

    Holds the GPU-lease file the Stage 6 scheduler shares across every workspace
    under one repo. Honors the ``ERA_STATE_DIR`` env var so tests redirect it to
    a temp directory. The directory is git-ignored and machine-local.
    """
    override = os.environ.get("ERA_STATE_DIR")
    if override:
        return Path(override).expanduser()
    return repo_root() / ".era"


def prompts_dir() -> Path:
    """Directory holding the ERA runtime prompts/templates.

    Lives under ``docs/`` (git-tracked, shipped with the code), flat and
    unversioned. ``knowledge/`` is reserved for coding-task prompts/plans, not
    ERA runtime files.
    """
    return repo_root() / "docs" / "prompts"
