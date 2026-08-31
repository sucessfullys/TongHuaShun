"""Deterministic vendor-sync for ``canonical_paths.py``.

Master source: ``Image-Generater-Remote/server/canonical_paths.py``
Vendored copy: ``System-Prompt-Retrieval-Agent/src/system_prompt_retrieval_agent/remote/_vendored/canonical_paths.py``

Modes:
- ``--check`` : verify the vendored copy is byte-identical to the master.
  Exit status 0 on match, 2 on mismatch, 3 on missing source.
  Used by CI and by deploy/install hard-gates.
- (default)   : copy master → vendored, write provenance JSON.

Usage:
    python scripts/sync_vendored_canonical_paths.py
    python scripts/sync_vendored_canonical_paths.py --check
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MASTER_REL = "Image-Generater-Remote/server/canonical_paths.py"
VENDORED_REL = (
    "System-Prompt-Retrieval-Agent/src/system_prompt_retrieval_agent/"
    "remote/_vendored/canonical_paths.py"
)
PROVENANCE_REL = (
    "System-Prompt-Retrieval-Agent/src/system_prompt_retrieval_agent/"
    "remote/_vendored/_canonical_paths_provenance.json"
)


def _read(path: Path) -> bytes:
    with path.open("rb") as fh:
        return fh.read()


def _utc_now() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_mtime(path: Path) -> str:
    return _dt.datetime.fromtimestamp(
        path.stat().st_mtime, tz=_dt.timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_provenance(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def cmd_check(master: Path, vendored: Path, provenance: Path) -> int:
    if not master.is_file():
        print(f"VENDOR-SYNC: master missing: {master}", file=sys.stderr)
        return 3
    if not vendored.is_file():
        print(f"VENDOR-SYNC: vendored copy missing: {vendored}", file=sys.stderr)
        return 2
    master_bytes = _read(master)
    vendored_bytes = _read(vendored)
    if master_bytes != vendored_bytes:
        master_sha = hashlib.sha256(master_bytes).hexdigest()
        vendored_sha = hashlib.sha256(vendored_bytes).hexdigest()
        print(
            "VENDOR-SYNC: byte-equality FAILED.\n"
            f"  master sha256:   {master_sha}\n"
            f"  vendored sha256: {vendored_sha}\n"
            f"  fix: python {Path(__file__).name}",
            file=sys.stderr,
        )
        return 2
    prov = _load_provenance(provenance)
    if prov is None:
        print(
            f"VENDOR-SYNC: provenance missing or unparseable: {provenance}",
            file=sys.stderr,
        )
        return 2
    expected_sha = hashlib.sha256(master_bytes).hexdigest()
    if prov.get("sha256") != expected_sha:
        print(
            "VENDOR-SYNC: provenance sha256 mismatch.\n"
            f"  expected: {expected_sha}\n"
            f"  recorded: {prov.get('sha256')}",
            file=sys.stderr,
        )
        return 2
    print("VENDOR-SYNC: byte-identical and provenance current.")
    return 0


def cmd_sync(master: Path, vendored: Path, provenance: Path) -> int:
    if not master.is_file():
        print(f"VENDOR-SYNC: master missing: {master}", file=sys.stderr)
        return 3
    master_bytes = _read(master)
    vendored.parent.mkdir(parents=True, exist_ok=True)
    tmp = vendored.with_suffix(vendored.suffix + ".tmp")
    tmp.write_bytes(master_bytes)
    os.replace(tmp, vendored)
    prov = {
        "source_path": MASTER_REL,
        "sha256": hashlib.sha256(master_bytes).hexdigest(),
        "size_bytes": len(master_bytes),
        "source_mtime_utc": _utc_mtime(master),
        "sync_timestamp_utc": _utc_now(),
    }
    provenance.write_text(
        json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"VENDOR-SYNC: synced {master} -> {vendored}")
    print(f"VENDOR-SYNC: sha256 {prov['sha256']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true", help="verify only")
    p.add_argument(
        "--repo-root",
        default=str(REPO_ROOT),
        help="repository root (default: derived from script path)",
    )
    args = p.parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    master = repo_root / MASTER_REL
    vendored = repo_root / VENDORED_REL
    provenance = repo_root / PROVENANCE_REL
    if args.check:
        return cmd_check(master, vendored, provenance)
    return cmd_sync(master, vendored, provenance)


if __name__ == "__main__":
    sys.exit(main())
