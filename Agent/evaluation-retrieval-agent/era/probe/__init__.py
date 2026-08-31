"""Deterministic environment probes for ERA Stage 0.

Each probe is read-only, never raises, and returns a plain ``dict`` so the
result can be serialized to JSON and handed back to the init slash-command.
"""

from .checkpoints import probe_checkpoints
from .credentials import probe_credentials
from .data import probe_data_roots
from .gpu import probe_gpus

__all__ = [
    "probe_gpus",
    "probe_data_roots",
    "probe_checkpoints",
    "probe_credentials",
]
