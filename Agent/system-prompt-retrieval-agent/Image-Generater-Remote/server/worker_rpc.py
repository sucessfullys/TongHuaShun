"""JSON-line RPC helpers for supervisor <-> backend worker communication.

Protocol overview
-----------------
Supervisor -> Worker (stdin of child):
    {"type": "load", "model_name": "gemma4", "gpu_id": 0, "config": {...}}
    {"type": "infer", "request_id": "abc123", "payload": {...}}
    {"type": "unload"}
    {"type": "exit"}

Worker -> Supervisor (stdout of child):
    {"type": "ready", "model_name": "gemma4", "load_latency_ms": 1234}
    {"type": "result", "request_id": "abc123", "ok": true, "payload": {...}}
    {"type": "error", "request_id": "abc123", "error_code": "oom", "message": "..."}
    {"type": "status", "message": "..."}
    {"type": "exiting"}
"""

from __future__ import annotations

import asyncio
import json
import os
import select
import signal
import subprocess
import sys
import time
from typing import IO, Optional


# ── Low-level stream helpers ───────────────────────────────────────────────────

def send_message(stream: IO, msg: dict) -> None:
    """Write a single JSON line to *stream* and flush immediately.

    Args:
        stream: Any writable file-like object (subprocess stdin, sys.stdout, …).
        msg:    Dict to serialise as JSON.

    Raises:
        BrokenPipeError: If the receiving end has closed.
        ValueError:      If *msg* cannot be serialised to JSON.
    """
    line = json.dumps(msg, ensure_ascii=False) + "\n"
    stream.write(line)
    stream.flush()


def recv_message(stream: IO, timeout_s: Optional[float] = None) -> Optional[dict]:
    """Read one JSON line from *stream*.

    Returns ``None`` on EOF or on timeout expiry.  Raises ``json.JSONDecodeError``
    if the line cannot be decoded (the caller should decide how to handle that).

    Args:
        stream:    Any readable file-like object.
        timeout_s: If given and > 0 uses ``select`` to wait at most this many
                   seconds.  A value of 0 performs a non-blocking poll.  ``None``
                   means block indefinitely.  On platforms that do not support
                   ``select`` on files (e.g. Windows with non-socket streams) the
                   timeout is silently ignored and the call blocks.
    """
    if timeout_s is not None:
        # select() only works on real file descriptors on Unix.
        try:
            fd = stream.fileno()
            ready, _, _ = select.select([fd], [], [], timeout_s)
            if not ready:
                return None  # timeout
        except (AttributeError, ValueError, select.error, OSError):
            # Stream doesn't have a real fd (e.g. io.StringIO in tests) —
            # fall through to blocking read.
            pass

    line = stream.readline()
    if not line:
        return None  # EOF
    line = line.strip()
    if not line:
        return None
    return json.loads(line)


# ── Async variant ──────────────────────────────────────────────────────────────

async def async_recv_message(
    reader: asyncio.StreamReader,
    timeout_s: Optional[float] = None,
) -> Optional[dict]:
    """Async version of ``recv_message`` using an ``asyncio.StreamReader``.

    Returns ``None`` on EOF, timeout, or connection-reset.

    Args:
        reader:    ``asyncio.StreamReader`` attached to the worker's stdout.
        timeout_s: Maximum seconds to wait.  ``None`` means wait indefinitely.
    """
    try:
        coro = reader.readline()
        if timeout_s is not None:
            line_bytes = await asyncio.wait_for(coro, timeout=timeout_s)
        else:
            line_bytes = await coro
    except asyncio.TimeoutError:
        return None
    except (asyncio.IncompleteReadError, ConnectionResetError):
        return None

    if not line_bytes:
        return None  # EOF
    line = line_bytes.decode("utf-8", errors="replace").strip()
    if not line:
        return None
    return json.loads(line)


# ── WorkerConnection ───────────────────────────────────────────────────────────

class WorkerConnection:
    """Manages JSON-line RPC communication with a spawned child process.

    The child process must:
    - Read JSON-line commands from its *stdin*.
    - Write JSON-line responses to its *stdout*.

    Typical usage::

        proc = subprocess.Popen(
            [sys.executable, "-m", "server.backend_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        conn = WorkerConnection(proc)
        conn.send({"type": "load", "model_name": "gemma4", "gpu_id": 0, "config": {}})
        resp = conn.recv(timeout_s=120)
    """

    def __init__(self, process: subprocess.Popen) -> None:
        """Store the process handle and validate that the streams are open.

        Args:
            process: A ``subprocess.Popen`` instance created with
                     ``stdin=subprocess.PIPE`` and ``stdout=subprocess.PIPE``.

        Raises:
            ValueError: If *stdin* or *stdout* are not available on *process*.
        """
        if process.stdin is None or process.stdout is None:
            raise ValueError(
                "WorkerConnection requires process.stdin and process.stdout. "
                "Create the Popen with stdin=PIPE and stdout=PIPE."
            )
        self._process = process

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def pid(self) -> int:
        """Return the PID of the child process."""
        return self._process.pid

    # ── Communication ─────────────────────────────────────────────────────────

    def send(self, msg: dict) -> None:
        """Send a JSON-line message to the worker's stdin.

        Args:
            msg: Dict to serialise and send.

        Raises:
            BrokenPipeError: If the worker has already closed its stdin.
        """
        send_message(self._process.stdin, msg)

    def recv(self, timeout_s: Optional[float] = None) -> Optional[dict]:
        """Receive one JSON-line message from the worker's stdout.

        Args:
            timeout_s: Maximum seconds to wait.  ``None`` blocks indefinitely.

        Returns:
            Parsed dict, or ``None`` on EOF / timeout.
        """
        return recv_message(self._process.stdout, timeout_s=timeout_s)

    # ── Process management ────────────────────────────────────────────────────

    def is_alive(self) -> bool:
        """Return ``True`` if the child process is still running."""
        return self._process.poll() is None

    def terminate(self, grace_s: float = 30) -> int:
        """Gracefully shut down the worker and return its exit code.

        Shutdown sequence:

        1. Send ``{"type": "exit"}`` to the worker's stdin.
        2. Wait up to *grace_s* seconds for the process to exit naturally.
        3. If still alive, send ``SIGTERM`` and wait up to 5 seconds.
        4. If still alive, send ``SIGKILL``.
        5. ``wait()`` and return the final exit code.

        Args:
            grace_s: Seconds to wait after sending the exit message before
                     resorting to SIGTERM.

        Returns:
            The process exit code (0 = clean exit).
        """
        # Step 1: polite exit message.
        try:
            self.send({"type": "exit"})
        except (BrokenPipeError, OSError):
            pass  # Worker stdin already closed — process likely dead.

        # Close stdin so the worker's readline() gets EOF if it ignores "exit".
        try:
            self._process.stdin.close()
        except OSError:
            pass

        # Step 2: wait grace_s for natural exit.
        deadline = time.monotonic() + grace_s
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                return self._process.returncode
            time.sleep(0.1)

        # Step 3: SIGTERM + 5 s.
        try:
            os.kill(self.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        else:
            term_deadline = time.monotonic() + 5.0
            while time.monotonic() < term_deadline:
                if self._process.poll() is not None:
                    return self._process.returncode
                time.sleep(0.1)

        # Step 4: SIGKILL.
        try:
            os.kill(self.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

        # Step 5: final wait.
        return self._process.wait()
