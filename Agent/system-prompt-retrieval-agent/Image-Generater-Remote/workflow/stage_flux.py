#!/usr/bin/env python3
"""Stage 2 — FLUX image generation worker.

Runs as a SUBPROCESS (not in the main orchestrator process) for clean
VRAM isolation. Each subprocess loads FLUX onto its assigned GPU,
processes its shard, and exits — OS reclaims all GPU memory.

Usage (called by gpu_pool.start_flux_workers):
    CUDA_VISIBLE_DEVICES=0 python workflow/stage_flux.py \
        --input shard_0.json --output results_0.json \
        --model-path /path/to/FLUX --config '{...}'
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FLUX-%(process)d] %(levelname)s %(message)s",
)
log = logging.getLogger("stage_flux")


def normalize_negative_prompt(value):
    """Normalize negative prompt: None/""/"None"/"none"/whitespace → Python None."""
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    s = value.strip()
    if s == "" or s.lower() == "none":
        return None
    return s


def process_shard(
    items: list[dict],
    model_path: str,
    config: dict,
) -> list[dict]:
    """Load FLUX and process all items in this shard.

    Args:
        items: List of work items with keys:
            sample_id, intermediate_prompt, model_path, cloth_path, output_dir, output_path
        model_path: Path to FLUX model weights.
        config: FLUX params (height, width, num_inference_steps, guidance_scale, seed).

    Returns:
        List of result dicts.
    """
    import torch
    from PIL import Image

    # Add parent dir to path for imports
    parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from workflow.image_preprocess import resize_model_image, resize_cloth_image

    # Load FLUX pipeline
    log.info("Loading FLUX pipeline from %s...", model_path)
    t0 = time.time()

    from diffusers import Flux2KleinPipeline
    pipeline = Flux2KleinPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipeline.to("cuda:0")  # CUDA_VISIBLE_DEVICES already set; device 0 = assigned GPU
    load_time = time.time() - t0
    log.info("FLUX loaded in %.1fs", load_time)

    # Introspect pipeline signature for negative_prompt support
    import inspect
    pipe_sig = inspect.signature(pipeline.__call__)
    supports_neg_prompt = "negative_prompt" in pipe_sig.parameters
    if not supports_neg_prompt:
        log.info("Pipeline does NOT support negative_prompt kwarg — will skip it")
    else:
        log.info("Pipeline supports negative_prompt kwarg")

    height = config.get("height", 1376)
    width = config.get("width", 768)
    num_steps = config.get("num_inference_steps", 4)
    guidance = config.get("guidance_scale", 1.0)
    seed = config.get("seed", 0)

    results = []

    for idx, item in enumerate(items):
        sample_id = item["sample_id"]
        output_path = item["output_path"]

        # Skip if already exists
        if os.path.exists(output_path):
            log.info("[%d/%d] SKIP %s (exists)", idx + 1, len(items), sample_id)
            results.append({"sample_id": sample_id, "output_path": output_path, "status": "skipped"})
            continue

        log.info("[%d/%d] Processing %s", idx + 1, len(items), sample_id)

        try:
            # Preprocess images
            img_model = Image.open(item["model_path"])
            img_model = resize_model_image(img_model, height, width)
            img_cloth = Image.open(item["cloth_path"])
            img_cloth = resize_cloth_image(img_cloth, height, width)

            w, h = img_model.size

            # Normalize negative prompt
            neg = normalize_negative_prompt(item.get("negative_prompt"))

            # Generate
            t0 = time.time()
            pipe_kwargs = dict(
                image=[img_model, img_cloth],
                prompt=item["intermediate_prompt"],
                height=h,
                width=w,
                guidance_scale=guidance,
                num_inference_steps=num_steps,
                generator=torch.Generator(device="cuda:0").manual_seed(seed),
            )
            # Only pass negative_prompt if pipeline supports it AND value is non-None
            neg_applied = False
            if neg is not None and supports_neg_prompt:
                pipe_kwargs["negative_prompt"] = neg
                neg_applied = True

            result_img = pipeline(**pipe_kwargs).images[0]
            flux_time = time.time() - t0

            # Save
            os.makedirs(item["output_dir"], exist_ok=True)
            result_img.save(output_path)

            # Save preprocessed inputs alongside result
            img_model.save(os.path.join(item["output_dir"], "input_model.png"))
            img_cloth.save(os.path.join(item["output_dir"], "input_cloth.png"))

            # Save generation metadata
            with open(os.path.join(item["output_dir"], "generation.json"), "w") as f:
                json.dump({
                    "intermediate_prompt": item["intermediate_prompt"],
                    "negative_prompt_applied": neg_applied,
                    "negative_prompt_normalized": neg,
                    "latency_s": round(flux_time, 2),
                    "seed": seed,
                    "height": h,
                    "width": w,
                    "num_inference_steps": num_steps,
                }, f, indent=2)

            log.info("  FLUX (%.1fs): saved %s", flux_time, output_path)
            results.append({
                "sample_id": sample_id,
                "output_path": output_path,
                "latency_s": round(flux_time, 2),
                "status": "ok",
            })

        except Exception as e:
            log.error("  ERROR on %s: %s", sample_id, e)
            traceback.print_exc()
            results.append({"sample_id": sample_id, "status": "error", "error": str(e)})

    ok_count = sum(1 for r in results if r["status"] in ("ok", "skipped"))
    log.info("FLUX shard complete: %d/%d ok", ok_count, len(items))
    return results


def main():
    parser = argparse.ArgumentParser(description="FLUX shard worker (subprocess)")
    parser.add_argument("--input", required=True, help="JSON file with shard items")
    parser.add_argument("--output", required=True, help="JSON file for results")
    parser.add_argument("--model-path", required=True, help="FLUX model path")
    parser.add_argument("--config", default="{}", help="JSON string with FLUX params")
    args = parser.parse_args()

    with open(args.input) as f:
        items = json.load(f)

    config = json.loads(args.config)
    results = process_shard(items, args.model_path, config)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    failed = sum(1 for r in results if r["status"] == "error")
    sys.exit(1 if failed == len(results) else 0)


if __name__ == "__main__":
    main()
