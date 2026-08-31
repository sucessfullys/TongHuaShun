"""Remote client package for the system-prompt retrieval agent.

Public API
----------
RemoteControllerClient
    Async HTTP client for the remote controller REST API.
open_tunnel
    SSH port-forward tunnel context manager.
assert_all_workers_done
    Non-negotiable all-workers-done barrier.
BarrierViolation
    Raised by :func:`assert_all_workers_done` on barrier violations.
CopybackError
    Raised by :func:`rsync_copyback` on copy-back failures.
rsync_copyback
    Copy remote output directory to a local path via rsync.
MANIFEST_VALIDATION_ERRORS
    List of canonical error codes used by :func:`validate_manifest`.
"""
from system_prompt_retrieval_agent.remote.barrier import (
    BarrierViolation,
    assert_all_workers_done,
)
from system_prompt_retrieval_agent.remote.client import RemoteControllerClient
from system_prompt_retrieval_agent.remote.copyback import CopybackError, rsync_copyback
from system_prompt_retrieval_agent.remote.manifest_validator import (
    MANIFEST_VALIDATION_ERRORS,
    ManifestValidationError,
    validate_manifest,
)
from system_prompt_retrieval_agent.remote.tunnel import TunnelError, open_tunnel

__all__ = [
    "RemoteControllerClient",
    "open_tunnel",
    "TunnelError",
    "assert_all_workers_done",
    "BarrierViolation",
    "CopybackError",
    "rsync_copyback",
    "MANIFEST_VALIDATION_ERRORS",
    "ManifestValidationError",
    "validate_manifest",
]
