"""YAML config loader with required-section validation and typed accessors."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml


REQUIRED_SECTIONS = ["server", "ports", "gpu", "paths", "models"]

_loaded: dict[str, Any] = {}


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load and validate config, caching the result."""
    global _loaded
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}")
    missing = [s for s in REQUIRED_SECTIONS if s not in data]
    if missing:
        raise ValueError(f"Config missing required sections: {missing}")
    _loaded = data
    return data


def get_config() -> dict[str, Any]:
    """Return the cached config; raises if not yet loaded."""
    if not _loaded:
        raise RuntimeError("Config not loaded — call load_config() first")
    return _loaded


def deep_get(cfg: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Navigate nested dicts with dotted keys. Returns default if any key is missing."""
    keys = dotted_key.split(".")
    current = cfg
    for k in keys:
        if not isinstance(current, dict) or k not in current:
            return default
        current = current[k]
    return current


# ── Typed accessors ────────────────────────────────────────────────

def server_host(cfg: dict[str, Any]) -> str:
    return str(deep_get(cfg, "server.host", "127.0.0.1"))


def server_port_for_gpu(cfg: dict[str, Any], gpu_id: int) -> int:
    mapping = deep_get(cfg, "gpu.per_host_gpu", {})
    for port_str, gid in mapping.items():
        if int(gid) == gpu_id:
            return int(port_str)
    raise ValueError(f"No port mapped to gpu_id={gpu_id}")


def port_to_gpu(cfg: dict[str, Any], port: int) -> int:
    mapping = deep_get(cfg, "gpu.per_host_gpu", {})
    gid = mapping.get(port, mapping.get(str(port)))
    if gid is None:
        raise ValueError(f"No GPU mapped to port={port}")
    return int(gid)


def all_host_ports(cfg: dict[str, Any]) -> list[int]:
    return [
        int(deep_get(cfg, f"ports.host_gpu{i}", 17610 + i))
        for i in range(3)
    ]


def model_config(cfg: dict[str, Any], model_name: str) -> dict[str, Any]:
    mc = deep_get(cfg, f"models.{model_name}")
    if mc is None:
        raise KeyError(f"No model config for '{model_name}'")
    return dict(mc)


def deployment_mode(cfg: dict[str, Any], model_name: str) -> str:
    return str(deep_get(cfg, f"models.{model_name}.deployment_mode", "single_gpu_replicated"))


def memory_ceiling_pct(cfg: dict[str, Any]) -> int:
    return int(deep_get(cfg, "gpu.memory_ceiling_pct", 90))


def post_unload_min_free_gib(cfg: dict[str, Any]) -> float:
    return float(deep_get(cfg, "gpu.post_unload_min_free_gib", 70))


def unload_grace_s(cfg: dict[str, Any]) -> int:
    return int(deep_get(cfg, "gpu.unload_grace_s", 30))


def output_root(cfg: dict[str, Any]) -> str:
    return str(deep_get(cfg, "paths.output_root", "outputs/v01"))


def allowed_read_roots(cfg: dict[str, Any]) -> list[str]:
    return list(deep_get(cfg, "paths.allowed_read_roots", []))


def callback_url(cfg: dict[str, Any]) -> str:
    return str(deep_get(cfg, "callback.default_url", "http://127.0.0.1:17700/status"))


def callback_required(cfg: dict[str, Any]) -> bool:
    return bool(deep_get(cfg, "callback.callback_required", True))


def callback_timeout(cfg: dict[str, Any]) -> int:
    return int(deep_get(cfg, "callback.timeout_s", 10))


def callback_retries(cfg: dict[str, Any]) -> int:
    return int(deep_get(cfg, "callback.retries", 3))


def callback_backoff(cfg: dict[str, Any]) -> list[float]:
    return list(deep_get(cfg, "callback.backoff_s", [0.5, 2.0, 8.0]))


def log_level(cfg: dict[str, Any]) -> str:
    return str(deep_get(cfg, "logging.level", "INFO"))


def log_format(cfg: dict[str, Any]) -> str:
    return str(deep_get(cfg, "logging.format", "jsonl"))


def correlation_fields(cfg: dict[str, Any]) -> list[str]:
    return list(deep_get(cfg, "logging.correlation_fields", [
        "run_id", "stage_id", "cell_id", "shard_id",
        "sample_id", "host_port", "model_name",
    ]))


def gpu_preservation_pattern(cfg: dict[str, Any]) -> str:
    return str(deep_get(cfg, "gpu.gpu_preservation_process_pattern", "NoGPUAlarmNew.py"))


def kill_gpu_preservation_on_load(cfg: dict[str, Any]) -> bool:
    return bool(deep_get(cfg, "gpu.kill_gpu_preservation_on_load", True))


def negative_prompt_targets(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return list(deep_get(cfg, "prompts.negative_prompt_targets", []))


def system_prompt_targets(cfg: dict[str, Any]) -> list[str]:
    return list(deep_get(cfg, "prompts.system_prompt_targets", []))


def image_pair_patterns(cfg: dict[str, Any]) -> list[dict[str, str]]:
    return list(deep_get(cfg, "data.image_pair_patterns", []))


def categories(cfg: dict[str, Any]) -> list[str]:
    return list(deep_get(cfg, "data.categories", ["dress", "lower", "upper"]))
