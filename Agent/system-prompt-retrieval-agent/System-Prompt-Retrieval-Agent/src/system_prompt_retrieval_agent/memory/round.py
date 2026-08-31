"""round.py — Per-round directory tree management."""
from __future__ import annotations

from pathlib import Path


def make_round_tree(rounds_root: Path, round_id: int) -> dict[str, Path]:
    """Create the per-round directory tree and return a mapping of name -> Path.

    Tree structure::

        rounds_root/
          round_{round_id:03d}/
            remote_manifests/
            api_eval/
            scoring/
            failures/
            visualizations/

    Keys returned:
        root, prompt_pairs_json, prompt_generation_context_json,
        prompt_generation_response_json, remote_jobs_jsonl,
        remote_manifests, remote_outputs_index_jsonl,
        api_eval, scoring, failures, visualizations,
        round_summary_md
    """
    rounds_root = Path(rounds_root)
    root = rounds_root / f"round_{round_id:03d}"

    subdirs = {
        "remote_manifests": root / "remote_manifests",
        "api_eval": root / "api_eval",
        "scoring": root / "scoring",
        "failures": root / "failures",
        "visualizations": root / "visualizations",
    }

    # Create all directories
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)

    tree: dict[str, Path] = {
        "root": root,
        "prompt_pairs_json": root / "prompt_pairs.json",
        "prompt_generation_context_json": root / "prompt_generation_context.json",
        "prompt_generation_response_json": root / "prompt_generation_response.json",
        "remote_jobs_jsonl": root / "remote_jobs.jsonl",
        "remote_outputs_index_jsonl": root / "remote_outputs_index.jsonl",
        "round_summary_md": root / "round_summary.md",
        **subdirs,
    }
    return tree
