"""rsync-based copy-back helper for post-stage output retrieval."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Optional


class CopybackError(RuntimeError):
    """Raised when rsync copyback fails after retries or required file is absent."""


def rsync_copyback(
    remote_path: str,
    local_path: Path,
    *,
    ssh_alias: str = "3h100",
    required_file: str = "stage3_qwen/stage_summary.json",
    max_retries: int = 1,
    subprocess_run: Optional[Callable] = None,
) -> None:
    """Copy remote output directory to *local_path* via rsync.

    After a successful rsync, the function checks that
    ``local_path / required_file`` exists.  On a missing file it retries the
    rsync once (``max_retries=1`` by default); a second miss raises
    :exc:`CopybackError`.

    Parameters
    ----------
    remote_path:
        Absolute path on the remote host (no ``user@host:`` prefix — the
        ``ssh_alias`` is used automatically).
    local_path:
        Local directory where files will be written.  Created if absent.
    ssh_alias:
        SSH host alias from ``~/.ssh/config`` (default ``"3h100"``).
    required_file:
        Relative path (within *local_path*) that must exist after a
        successful copy.
    max_retries:
        Number of extra attempts on a missing *required_file* (default 1).
    subprocess_run:
        Injectable replacement for ``subprocess.run`` (default).  Useful in
        tests to avoid real rsync calls.

    Raises
    ------
    CopybackError
        If rsync exits non-zero or if *required_file* is absent after all
        retry attempts.
    """
    _run = subprocess_run if subprocess_run is not None else subprocess.run

    local_path.mkdir(parents=True, exist_ok=True)
    remote_src = f"{ssh_alias}:{remote_path}/"
    local_dst = str(local_path) + "/"

    def _do_rsync() -> None:
        # Try rsync first; fall back to `scp -r` if rsync is unavailable on
        # remote (V0.1.1 environment ships without rsync).
        result = _run(
            ["rsync", "-a", "--delete", remote_src, local_dst],
            capture_output=True,
        )
        if result.returncode == 0:
            return
        stderr = getattr(result, "stderr", b"")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        if "rsync: command not found" in stderr or "rsync: not found" in stderr or "rsync(" in stderr:
            scp_result = _run(
                ["scp", "-q", "-r", f"{ssh_alias}:{remote_path}/.", local_dst],
                capture_output=True,
            )
            if scp_result.returncode == 0:
                return
            scp_err = getattr(scp_result, "stderr", b"")
            if isinstance(scp_err, bytes):
                scp_err = scp_err.decode(errors="replace")
            raise CopybackError(
                f"rsync absent and scp fallback failed (code {scp_result.returncode}): {scp_err[:400]}"
            )
        raise CopybackError(
            f"rsync exited with code {result.returncode}: {stderr[:400]}"
        )

    attempts = 0
    while True:
        _do_rsync()
        # If required_file is empty, treat presence of any file under local_path as success.
        if not required_file:
            if any(local_path.rglob("*")):
                return
        else:
            target = local_path / required_file
            if target.exists():
                return
        attempts += 1
        if attempts > max_retries:
            raise CopybackError(
                f"Required file missing after {attempts} rsync attempt(s): {(local_path / required_file) if required_file else local_path}"
            )
        # Loop to retry.
