"""Dataset discovery and sharding for the workflow pipeline.

Discovers image samples from the dataset directory, splits them into
disjoint shards for multi-GPU processing, and builds stage-specific
input lists.
"""

from __future__ import annotations

import json
import os
from typing import Any


def discover_samples(
    dataset_root: str,
    categories: list[str] = ("dress", "lower", "upper"),
    pair_patterns: list[dict[str, str]] = (
        {"model": "model.png", "cloth": "cloth.png"},
        {"model": "input_model.png", "cloth": "input_cloth.png"},
    ),
) -> list[dict[str, Any]]:
    """Discover all valid image pair samples in the dataset.

    Walks {dataset_root}/{category}/{subcategory}/{sample}/ and looks for
    files matching any pair pattern.

    Returns:
        Sorted list of sample dicts with keys:
            index, sample_id, category, model_path, cloth_path, pair_pattern
    """
    samples = []
    idx = 0

    for cat in categories:
        cat_dir = os.path.join(dataset_root, cat)
        if not os.path.isdir(cat_dir):
            continue

        for subcat in sorted(os.listdir(cat_dir)):
            subcat_dir = os.path.join(cat_dir, subcat)
            if not os.path.isdir(subcat_dir):
                continue

            for sample_name in sorted(os.listdir(subcat_dir)):
                sample_dir = os.path.join(subcat_dir, sample_name)
                if not os.path.isdir(sample_dir):
                    continue

                # Try each pair pattern
                for pattern in pair_patterns:
                    model_path = os.path.join(sample_dir, pattern["model"])
                    cloth_path = os.path.join(sample_dir, pattern["cloth"])
                    if os.path.exists(model_path) and os.path.exists(cloth_path):
                        samples.append({
                            "index": idx,
                            "sample_id": f"{cat}/{subcat}/{sample_name}",
                            "category": cat,
                            "model_path": model_path,
                            "cloth_path": cloth_path,
                            "pair_pattern": pattern,
                        })
                        idx += 1
                        break  # First matching pattern wins

    return samples


def split_into_shards(
    samples: list[dict],
    n_shards: int,
) -> list[list[dict]]:
    """Split samples into n_shards disjoint lists via round-robin.

    sample[i] goes to shard[i % n_shards]. No duplication.
    """
    shards: list[list[dict]] = [[] for _ in range(n_shards)]
    for i, sample in enumerate(samples):
        shards[i % n_shards].append(sample)
    return shards


def build_gemma_inputs(
    shard: list[dict],
    system_prompt: str,
    user_instruction: str = "the model in image 1 wears the clothing from image 2",
) -> list[dict]:
    """Build Stage 1 (Gemma) input items from a shard."""
    return [
        {
            "sample_id": s["sample_id"],
            "model_path": s["model_path"],
            "cloth_path": s["cloth_path"],
            "system_prompt": system_prompt,
            "user_instruction": user_instruction,
        }
        for s in shard
    ]


def build_flux_inputs(
    shard: list[dict],
    gemma_results: dict[str, dict],
    output_root: str,
    prompt_id: str,
    negative_prompt: str | None = None,
) -> list[dict]:
    """Build Stage 2 (FLUX) input items from Gemma results.

    Args:
        shard: Original sample dicts for this GPU's shard.
        gemma_results: {sample_id: {intermediate_prompt, status}} from Stage 1.
        output_root: Root output directory.
        prompt_id: System prompt identifier (for path organization).

    Returns:
        List of FLUX work items (only includes samples that succeeded in Stage 1).
    """
    items = []
    for s in shard:
        sid = s["sample_id"]
        gemma = gemma_results.get(sid)
        if not gemma or gemma.get("status") not in ("ok", "skipped"):
            continue  # Skip failed Gemma samples (but include resumed/skipped ones)

        output_dir = os.path.join(output_root, prompt_id, "stage2_flux", "images", sid)
        items.append({
            "sample_id": sid,
            "model_path": s["model_path"],
            "cloth_path": s["cloth_path"],
            "intermediate_prompt": gemma["intermediate_prompt"],
            "negative_prompt": negative_prompt,
            "output_dir": output_dir,
            "output_path": os.path.join(output_dir, "tryon_result.png"),
        })
    return items


def build_qwen_inputs(
    shard: list[dict],
    flux_results: dict[str, dict],
) -> list[dict]:
    """Build Stage 3 (Qwen) input items from FLUX results.

    Returns:
        List of Qwen eval items (only includes samples that succeeded in Stage 2).
    """
    items = []
    for s in shard:
        sid = s["sample_id"]
        flux = flux_results.get(sid)
        if not flux or flux.get("status") not in ("ok", "skipped"):
            continue  # Skip failed FLUX samples (but include resumed/skipped ones)

        items.append({
            "sample_id": sid,
            "model_path": s["model_path"],
            "cloth_path": s["cloth_path"],
            "result_image_path": flux["output_path"],
        })
    return items


def write_shard_file(items: list[dict], path: str) -> None:
    """Write shard items to a JSON file for subprocess consumption."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)


def read_stage_results(path: str) -> dict[str, dict]:
    """Read stage_results.jsonl into a {sample_id: result} dict."""
    results = {}
    if not os.path.exists(path):
        return results
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            results[item["sample_id"]] = item
    return results


def write_stage_result(f, result: dict) -> None:
    """Append a single result to a stage_results.jsonl file handle."""
    f.write(json.dumps(result, ensure_ascii=False) + "\n")
    f.flush()
