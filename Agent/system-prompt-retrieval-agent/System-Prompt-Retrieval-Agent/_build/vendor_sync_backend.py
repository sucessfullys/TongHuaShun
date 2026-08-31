"""PEP 517 build backend that hard-gates byte-equality of the vendored
``canonical_paths.py`` before any wheel or sdist is produced.

This module wraps ``setuptools.build_meta`` and inserts a vendor-sync
``--check`` step at the start of every PEP 517 entry point. If the local
vendored copy under
``src/system_prompt_retrieval_agent/remote/_vendored/canonical_paths.py``
does not match the remote source at
``Image-Generater-Remote/server/canonical_paths.py`` byte-for-byte, the
build aborts with ``SystemExit(2)`` (or ``3`` if the master is missing).
The aborts happen **before** any package files are installed, so
``pip install`` fails loudly and atomically.

Wired into ``pyproject.toml`` via::

    [build-system]
    build-backend = "vendor_sync_backend"
    backend-path = ["_build"]
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from setuptools import build_meta as _orig

# Re-export setuptools' build-meta surface that we don't override.
__all__ = [
    "get_requires_for_build_wheel",
    "get_requires_for_build_sdist",
    "get_requires_for_build_editable",
    "prepare_metadata_for_build_wheel",
    "prepare_metadata_for_build_editable",
    "build_wheel",
    "build_sdist",
    "build_editable",
]

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent  # System-Prompt-Retrieval-Agent/
_REPO_ROOT = _PROJECT_ROOT.parent  # outer repo root containing Image-Generater-Remote/
_SCRIPT = _PROJECT_ROOT / "scripts" / "sync_vendored_canonical_paths.py"
_MASTER = _REPO_ROOT / "Image-Generater-Remote" / "server" / "canonical_paths.py"
_VENDORED = (
    _PROJECT_ROOT
    / "src"
    / "system_prompt_retrieval_agent"
    / "remote"
    / "_vendored"
    / "canonical_paths.py"
)
_PROVENANCE = _VENDORED.parent / "_canonical_paths_provenance.json"


def _verify_provenance_only() -> None:
    """Sdist install path: master absent, but vendored copy + provenance JSON
    must still be self-consistent (ships inside the sdist).
    """
    if not _VENDORED.is_file():
        sys.stderr.write(
            f"ABORT: vendored canonical_paths.py missing at {_VENDORED}; "
            "install/build refused before any files were written\n"
        )
        raise SystemExit(2)
    if not _PROVENANCE.is_file():
        sys.stderr.write(
            f"ABORT: provenance JSON missing at {_PROVENANCE}; "
            "install/build refused before any files were written\n"
        )
        raise SystemExit(2)
    try:
        prov = json.loads(_PROVENANCE.read_text(encoding="utf-8"))
    except Exception as exc:
        sys.stderr.write(
            f"ABORT: provenance JSON unparseable ({exc}); "
            "install/build refused before any files were written\n"
        )
        raise SystemExit(2)
    actual = hashlib.sha256(_VENDORED.read_bytes()).hexdigest()
    expected = prov.get("sha256")
    if actual != expected:
        sys.stderr.write(
            "ABORT: vendored canonical_paths.py sha256 does not match "
            f"recorded provenance sha256 ({actual} != {expected}); "
            "install/build refused before any files were written\n"
        )
        raise SystemExit(2)


def _run_vendor_sync_check() -> None:
    if os.environ.get("SPRA_SKIP_VENDOR_SYNC_CHECK") == "1":
        # Escape hatch for the rare case of building from a sdist that has
        # already been verified upstream. CI must not set this.
        return
    if not _MASTER.is_file():
        # Sdist install path — master is not shipped. Verify provenance
        # self-consistency instead.
        _verify_provenance_only()
        return
    if not _SCRIPT.is_file():
        sys.stderr.write(
            f"ABORT: vendor-sync script missing at {_SCRIPT}; "
            "refusing to build\n"
        )
        raise SystemExit(2)
    rc = subprocess.call(
        [sys.executable, str(_SCRIPT), "--check", "--repo-root", str(_REPO_ROOT)]
    )
    if rc != 0:
        sys.stderr.write(
            f"ABORT: vendored canonical_paths.py failed byte-equality check "
            f"(exit {rc}); install/build refused before any files were written\n"
        )
        raise SystemExit(rc)


def get_requires_for_build_wheel(config_settings=None):
    _run_vendor_sync_check()
    return _orig.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_sdist(config_settings=None):
    _run_vendor_sync_check()
    return _orig.get_requires_for_build_sdist(config_settings)


def get_requires_for_build_editable(config_settings=None):
    _run_vendor_sync_check()
    return _orig.get_requires_for_build_editable(config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    _run_vendor_sync_check()
    return _orig.prepare_metadata_for_build_wheel(
        metadata_directory, config_settings
    )


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    _run_vendor_sync_check()
    return _orig.prepare_metadata_for_build_editable(
        metadata_directory, config_settings
    )


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _run_vendor_sync_check()
    return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    _run_vendor_sync_check()
    return _orig.build_sdist(sdist_directory, config_settings)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    _run_vendor_sync_check()
    return _orig.build_editable(
        wheel_directory, config_settings, metadata_directory
    )
