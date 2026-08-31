"""SSH tunnel context-manager for forwarding a remote controller port locally."""
from __future__ import annotations

import subprocess
import time
from contextlib import contextmanager
from typing import Callable, Iterator, Optional

import httpx


class TunnelError(RuntimeError):
    """Raised when the SSH tunnel cannot be established or health check fails."""


@contextmanager
def open_tunnel(
    alias: str = "3h100",
    local_port: int = 17700,
    remote_port: int = 17700,
    reconnect_max: int = 5,
    health_timeout_s: float = 30.0,
    health_url_template: str = "http://127.0.0.1:{port}/health",
    popen: Optional[Callable] = None,
    sleep: Optional[Callable] = None,
    http_get: Optional[Callable] = None,
) -> Iterator[None]:
    """Open an SSH port-forward tunnel and poll health until ready.

    Parameters
    ----------
    alias:
        SSH host alias (must be configured in ~/.ssh/config).
    local_port:
        Local port to bind.
    remote_port:
        Remote port to forward to.
    reconnect_max:
        Maximum reconnect attempts before raising TunnelError.  NOTE: this
        counter applies to the initial bring-up phase, NOT to the caller's
        retry loop — the caller retries by re-entering the context manager.
    health_timeout_s:
        Total seconds to wait for the health endpoint to return HTTP 200.
    health_url_template:
        URL template for the health check; ``{port}`` is substituted.
    popen:
        Injectable replacement for ``subprocess.Popen`` (default).
    sleep:
        Injectable replacement for ``time.sleep`` (default).
    http_get:
        Injectable replacement for ``httpx.get`` (default).

    Yields
    ------
    None
        Yields once the tunnel is healthy.

    Raises
    ------
    TunnelError
        If the health endpoint never returns 200 within the timeout.
    """
    _popen = popen if popen is not None else subprocess.Popen
    _sleep = sleep if sleep is not None else time.sleep
    _http_get = http_get if http_get is not None else httpx.get

    health_url = health_url_template.format(port=local_port)

    cmd = [
        "ssh",
        "-N",
        "-L",
        f"{local_port}:127.0.0.1:{remote_port}",
        alias,
    ]

    proc = _popen(cmd, start_new_session=True)
    try:
        # Poll health endpoint until 200 or timeout.
        deadline = time.monotonic() + health_timeout_s
        healthy = False
        poll_interval = 0.5
        while time.monotonic() < deadline:
            try:
                resp = _http_get(health_url, timeout=2.0)
                if resp.status_code == 200:
                    healthy = True
                    break
            except Exception:
                pass
            _sleep(poll_interval)

        if not healthy:
            raise TunnelError(
                f"SSH tunnel health check failed after {health_timeout_s}s "
                f"(url={health_url})"
            )

        yield

    finally:
        # Terminate the SSH process gracefully then force-kill after 5s.
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:
            pass
