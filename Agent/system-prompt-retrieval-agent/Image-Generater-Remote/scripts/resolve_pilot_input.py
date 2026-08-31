#!/usr/bin/env python3
"""resolve_pilot_input.py — Probe paths.pilot_input_candidates in order and pick the first valid one.

Usage:
    python scripts/resolve_pilot_input.py [--config CONFIG] [--dry-run] [--output PATH]

    --config PATH    Path to config YAML file (default: config.yaml)
    --dry-run        Print the ordered candidate list and exit 0 without probing
    --output PATH    Path for the resolved output JSON (default: resolved_pilot_input.json)

A candidate is valid if:
  1. It is a readable, non-empty directory.
  2. It contains at least one of {dress, lower, upper}/ subdirectories.
  3. That subdirectory contains at least 1 sample directory with a valid image pair
     matching any pattern from data.image_pair_patterns.
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml is not installed. Run: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


KNOWN_CATEGORIES = ("dress", "lower", "upper")


def load_config(config_path: str) -> dict:
    """Load and return the YAML config."""
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


def get_candidates(cfg: dict) -> list[str]:
    """Extract the pilot_input_candidates list from config."""
    try:
        candidates = _deep_get(cfg, "paths.pilot_input_candidates")
    except KeyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    if candidates is None:
        print("ERROR: paths.pilot_input_candidates is null in config.", file=sys.stderr)
        sys.exit(1)
    if not isinstance(candidates, list):
        print(
            f"ERROR: paths.pilot_input_candidates must be a list, got {type(candidates).__name__}.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not candidates:
        print("ERROR: paths.pilot_input_candidates is empty.", file=sys.stderr)
        sys.exit(1)
    return [str(c) for c in candidates]


def get_image_pair_patterns(cfg: dict) -> list[dict]:
    """Extract data.image_pair_patterns from config, with a sensible default."""
    try:
        patterns = _deep_get(cfg, "data.image_pair_patterns")
    except KeyError:
        # Fallback to spec defaults if key absent
        patterns = [
            {"model": "input_model.png", "cloth": "input_cloth.png"},
            {"model": "model.png", "cloth": "cloth.png"},
        ]
    if not isinstance(patterns, list) or not patterns:
        patterns = [
            {"model": "input_model.png", "cloth": "input_cloth.png"},
            {"model": "model.png", "cloth": "cloth.png"},
        ]
    return patterns


def _sample_dir_has_valid_pair(sample_dir: Path, patterns: list[dict]) -> bool:
    """Return True if sample_dir contains both files from at least one pattern."""
    for pattern in patterns:
        files_needed = list(pattern.values())
        if all((sample_dir / fname).is_file() for fname in files_needed):
            return True
    return False


def _category_has_valid_sample(category_dir: Path, patterns: list[dict]) -> tuple[bool, int]:
    """
    Return (has_valid_sample, sample_count) where sample_count is the number
    of sample directories (whether valid or not) found immediately under category_dir.
    """
    if not category_dir.is_dir():
        return False, 0
    try:
        entries = list(category_dir.iterdir())
    except PermissionError:
        return False, 0
    sample_dirs = [e for e in entries if e.is_dir()]
    for sample_dir in sample_dirs:
        if _sample_dir_has_valid_pair(sample_dir, patterns):
            return True, len(sample_dirs)
    return False, len(sample_dirs)


def probe_candidate(candidate_path: str, patterns: list[dict]) -> dict:
    """
    Probe a single candidate.

    Returns a result dict with keys:
        path, is_valid, reason, categories_found, sample_count
    """
    p = Path(candidate_path)

    # Check 1: readable, non-empty directory
    if not p.exists():
        return {
            "path": candidate_path,
            "is_valid": False,
            "reason": "path does not exist",
            "categories_found": [],
            "sample_count": 0,
        }
    if not p.is_dir():
        return {
            "path": candidate_path,
            "is_valid": False,
            "reason": "path is not a directory",
            "categories_found": [],
            "sample_count": 0,
        }
    try:
        entries = list(p.iterdir())
    except PermissionError:
        return {
            "path": candidate_path,
            "is_valid": False,
            "reason": "permission denied reading directory",
            "categories_found": [],
            "sample_count": 0,
        }
    if not entries:
        return {
            "path": candidate_path,
            "is_valid": False,
            "reason": "directory is empty",
            "categories_found": [],
            "sample_count": 0,
        }

    # Check 2 & 3: at least one category subdir with a valid image pair
    valid_categories = []
    total_sample_count = 0

    for category in KNOWN_CATEGORIES:
        cat_dir = p / category
        has_valid, sample_count = _category_has_valid_sample(cat_dir, patterns)
        if has_valid:
            valid_categories.append(category)
            total_sample_count += sample_count
        elif sample_count > 0:
            # Category dir exists but no valid pairs found
            total_sample_count += sample_count

    if not valid_categories:
        # Determine a helpful reason
        present_cats = [cat for cat in KNOWN_CATEGORIES if (p / cat).is_dir()]
        if not present_cats:
            reason = f"no category subdirectories ({', '.join(KNOWN_CATEGORIES)}) found"
        else:
            reason = (
                f"category dirs found ({', '.join(present_cats)}) "
                "but none contained a valid image pair"
            )
        return {
            "path": candidate_path,
            "is_valid": False,
            "reason": reason,
            "categories_found": [],
            "sample_count": total_sample_count,
        }

    return {
        "path": candidate_path,
        "is_valid": True,
        "reason": f"valid candidate with categories: {', '.join(valid_categories)}",
        "categories_found": valid_categories,
        "sample_count": total_sample_count,
    }


def run_dry_run(candidates: list[str]):
    """Print the ordered candidate list and exit 0."""
    print("Pilot input candidates (in probe order):")
    for i, c in enumerate(candidates, start=1):
        print(f"  {i}. {c}")
    sys.exit(0)


def run_probe(candidates: list[str], patterns: list[dict], output_path: str):
    """Probe candidates in order, write result JSON, exit 0 on success or non-zero on failure."""
    tried = []
    picked_result = None

    for rank, candidate in enumerate(candidates, start=1):
        result = probe_candidate(candidate, patterns)

        if result["is_valid"]:
            tried.append({
                "path": candidate,
                "status": "picked",
                "reason": result["reason"],
            })
            # Mark any remaining candidates as not probed
            for remaining in candidates[rank:]:
                tried.append({
                    "path": remaining,
                    "status": "skipped",
                    "reason": "not probed (earlier candidate succeeded)",
                })
            picked_result = result
            picked_result["candidate_rank"] = rank
            break
        else:
            tried.append({
                "path": candidate,
                "status": "failed",
                "reason": result["reason"],
            })

    if picked_result is None:
        # All candidates failed
        print("ERROR: No valid pilot input candidate found.", file=sys.stderr)
        print("Tried paths:", file=sys.stderr)
        for entry in tried:
            print(f"  [{entry['status']}] {entry['path']}: {entry['reason']}", file=sys.stderr)
        sys.exit(1)

    output = {
        "picked": picked_result["path"],
        "candidate_rank": picked_result["candidate_rank"],
        "tried": tried,
        "categories_found": picked_result["categories_found"],
        "sample_count": picked_result["sample_count"],
    }

    out_path = Path(output_path)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2)
            fh.write("\n")
    except OSError as exc:
        print(f"ERROR: Could not write output file '{output_path}': {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Picked: {picked_result['path']}")
    print(f"  Candidate rank: {picked_result['candidate_rank']}")
    print(f"  Categories found: {', '.join(picked_result['categories_found'])}")
    print(f"  Sample count: {picked_result['sample_count']}")
    print(f"  Written: {output_path}")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Probe paths.pilot_input_candidates in order and pick the first valid one."
        )
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the ordered candidate list and exit 0 without probing",
    )
    parser.add_argument(
        "--output",
        default="resolved_pilot_input.json",
        help="Path for the resolved output JSON (default: resolved_pilot_input.json)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    candidates = get_candidates(cfg)

    if args.dry_run:
        run_dry_run(candidates)
    else:
        patterns = get_image_pair_patterns(cfg)
        run_probe(candidates, patterns, args.output)


if __name__ == "__main__":
    main()
