#!/usr/bin/env python3
"""check_model_assets.py — Validate model/adapter/dataset/output/allowed-read paths from config.

Usage:
    python scripts/check_model_assets.py [--config CONFIG] [--local-only]

    --config PATH    Path to config YAML file (default: config.yaml)
    --local-only     Skip remote path checks; only verify config is parseable
                     and has required top-level sections. Exits 0 with warnings.
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml is not installed. Run: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


REQUIRED_SECTIONS = ("models", "paths", "gpu", "server", "ports")

# Paths to validate when NOT in --local-only mode.
# Each entry: (dot-notation key, "dir" | "file" | "parent_must_exist")
REMOTE_PATH_CHECKS = [
    ("models.gemma4.path",           "dir"),
    ("models.flux2_klein.path",      "dir"),
    ("models.qwen3_vl.path",         "dir"),
    ("models.qwen3_vl.adapter_path", "dir"),
    ("paths.dataset_root",           "dir"),
    ("paths.output_root",            "parent_must_exist"),
]

# Each entry in paths.allowed_read_roots is validated as a directory.
ALLOWED_READ_ROOTS_KEY = "paths.allowed_read_roots"


def _deep_get(mapping: dict, dotted_key: str):
    """Return the value at a dotted key path, or raise KeyError."""
    parts = dotted_key.split(".")
    node = mapping
    for part in parts:
        if not isinstance(node, dict):
            raise KeyError(f"Expected a mapping at '{part}' in key '{dotted_key}'")
        if part not in node:
            raise KeyError(f"Key '{part}' not found (full path: '{dotted_key}')")
        node = node[part]
    return node


def load_config(config_path: str) -> dict:
    """Load and return the YAML config, raising on parse error."""
    path = Path(config_path)
    if not path.exists():
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with path.open("r", encoding="utf-8") as fh:
        try:
            data = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            print(f"ERROR: Failed to parse config YAML: {exc}", file=sys.stderr)
            sys.exit(1)
    if not isinstance(data, dict):
        print("ERROR: Config file did not parse to a mapping.", file=sys.stderr)
        sys.exit(1)
    return data


def check_required_sections(cfg: dict) -> list[str]:
    """Return list of missing required top-level sections."""
    return [sec for sec in REQUIRED_SECTIONS if sec not in cfg]


def check_path(label: str, raw_value, check_type: str) -> tuple[bool, str]:
    """
    Validate a single path entry.

    check_type:
        "dir"               — must exist and be a directory
        "file"              — must exist and be a file
        "parent_must_exist" — parent directory must exist (path itself may not yet)

    Returns (passed: bool, message: str).
    """
    if raw_value is None:
        return False, f"  FAIL  [{label}]: value is null/missing"

    p = Path(str(raw_value))

    if check_type == "dir":
        if p.is_dir():
            return True, f"  PASS  [{label}]: directory exists: {p}"
        elif p.exists():
            return False, f"  FAIL  [{label}]: path exists but is not a directory: {p}"
        else:
            return False, f"  FAIL  [{label}]: directory not found: {p}"

    elif check_type == "file":
        if p.is_file():
            return True, f"  PASS  [{label}]: file exists: {p}"
        elif p.exists():
            return False, f"  FAIL  [{label}]: path exists but is not a file: {p}"
        else:
            return False, f"  FAIL  [{label}]: file not found: {p}"

    elif check_type == "parent_must_exist":
        parent = p.parent
        if parent.is_dir():
            return True, f"  PASS  [{label}]: parent directory exists: {parent}  (output path: {p})"
        else:
            return False, f"  FAIL  [{label}]: parent directory not found: {parent}  (output path: {p})"

    else:
        return False, f"  FAIL  [{label}]: unknown check_type '{check_type}'"


def run_local_only(cfg: dict, config_path: str):
    """Perform local-only validation: config parseable + required sections present."""
    print(f"Config: {config_path}")
    print("Local-only mode: skipping remote path checks")
    print()

    missing = check_required_sections(cfg)
    if missing:
        print("FAIL — The following required top-level sections are absent:")
        for sec in missing:
            print(f"  FAIL  [section '{sec}']: not found in config")
        sys.exit(1)

    print("Required sections present:", ", ".join(REQUIRED_SECTIONS))
    print()
    print("Result: PASSED (local-only checks)")
    sys.exit(0)


def run_full(cfg: dict, config_path: str):
    """Perform full remote-path validation."""
    print(f"Config: {config_path}")
    print("Full path validation (remote mount checks enabled)")
    print()

    failures = []
    passes = []

    # Fixed path checks
    for dotted_key, check_type in REMOTE_PATH_CHECKS:
        label = dotted_key
        try:
            value = _deep_get(cfg, dotted_key)
        except KeyError as exc:
            failures.append(f"  FAIL  [{label}]: key not found in config — {exc}")
            continue
        passed, msg = check_path(label, value, check_type)
        (passes if passed else failures).append(msg)

    # allowed_read_roots — list of directories
    try:
        roots = _deep_get(cfg, ALLOWED_READ_ROOTS_KEY)
    except KeyError as exc:
        failures.append(f"  FAIL  [{ALLOWED_READ_ROOTS_KEY}]: key not found — {exc}")
        roots = []

    if roots is None:
        failures.append(f"  FAIL  [{ALLOWED_READ_ROOTS_KEY}]: value is null")
    elif not isinstance(roots, list):
        failures.append(f"  FAIL  [{ALLOWED_READ_ROOTS_KEY}]: expected a list, got {type(roots).__name__}")
    else:
        for i, entry in enumerate(roots):
            label = f"{ALLOWED_READ_ROOTS_KEY}[{i}]"
            passed, msg = check_path(label, entry, "dir")
            (passes if passed else failures).append(msg)

    # Print results
    if passes:
        print("Passed:")
        for msg in passes:
            print(msg)
        print()

    if failures:
        print("Failed:")
        for msg in failures:
            print(msg)
        print()
        total = len(passes) + len(failures)
        print(f"Result: FAILED — {len(failures)} of {total} checks failed")
        sys.exit(1)

    print(f"Result: PASSED — all {len(passes)} checks passed")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Validate model/adapter/dataset/output/allowed-read paths from config."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Skip remote path checks; only verify config parseability and required sections",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.local_only:
        run_local_only(cfg, args.config)
    else:
        run_full(cfg, args.config)


if __name__ == "__main__":
    main()
