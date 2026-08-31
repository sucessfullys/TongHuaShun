#!/usr/bin/env python3
"""Staged workflow pipeline: Gemma → FLUX → Qwen across 3×H100 GPUs.

Each stage returns (merged_results, manifest). The manifest contains
counts and error details for the controller response.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from workflow.gpu_pool import (
    GPU_IDS, kill_gpu_hogs, start_vllm_servers, stop_servers,
    start_flux_workers, wait_for_flux_workers, verify_gpus_free,
)
from workflow.data_shard import (
    discover_samples, split_into_shards,
    build_gemma_inputs, build_flux_inputs, build_qwen_inputs,
    write_shard_file,
)
from workflow.stage_gemma import run_gemma_shard
from workflow.stage_qwen import run_qwen_shard

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("workflow_pipeline.log"),
    ],
)
log = logging.getLogger("pipeline")


# ── Barrier verification ──────────────────────────────────────────

def _verify_stage_barrier(
    stage_name: str,
    merged: dict[str, dict],
    expected_total: int,
    gpu_ids: list[int],
    check_vram: bool = True,
) -> dict:
    """Post-stage verification. Returns manifest dict."""
    ok = sum(1 for r in merged.values() if r.get("status") in ("ok", "skipped"))
    errors = sum(1 for r in merged.values() if r.get("status") == "error")
    error_samples = [r["sample_id"] for r in merged.values() if r.get("status") == "error"]

    if len(merged) != expected_total:
        log.warning("BARRIER %s: got %d results, expected %d",
                    stage_name, len(merged), expected_total)

    if check_vram and not verify_gpus_free(gpu_ids, min_free_gib=40.0):
        log.warning("BARRIER %s: GPUs not fully free after unload", stage_name)

    manifest = {
        "stage": stage_name,
        "total": len(merged),
        "ok": ok,
        "errors": errors,
        "error_samples": error_samples,
    }
    log.info("BARRIER %s: %d ok, %d errors out of %d", stage_name, ok, errors, len(merged))
    return manifest


# ── Stage 1: Gemma ────────────────────────────────────────────────

async def run_stage1_gemma(
    samples: list[dict],
    system_prompt: str,
    prompt_id: str,
    output_root: str,
    gemma_model_path: str,
    gpu_ids: list[int],
    ports: list[int],
    gemma_params: dict | None = None,
) -> tuple[dict[str, dict], dict]:
    """Returns (merged_results, manifest)."""
    log.info("=" * 60)
    log.info("STAGE 1: GEMMA — %s (%d samples, %d GPUs)", prompt_id, len(samples), len(gpu_ids))
    log.info("=" * 60)

    kill_gpu_hogs()
    time.sleep(3)
    if not verify_gpus_free(gpu_ids, min_free_gib=40.0):
        log.warning("GPUs not fully free before Gemma load")

    procs = start_vllm_servers(
        gemma_model_path, ports, gpu_ids,
        gpu_memory_utilization=0.95, max_model_len=8192,
    )

    try:
        shards = split_into_shards(samples, len(gpu_ids))
        prompt_output_dir = os.path.join(output_root, prompt_id, "stage1_gemma", "prompts")
        results_dir = os.path.join(output_root, prompt_id, "stage1_gemma")
        os.makedirs(results_dir, exist_ok=True)

        shard_inputs = [build_gemma_inputs(shard, system_prompt) for shard in shards]

        tasks = []
        for gpu_id, port, items in zip(gpu_ids, ports, shard_inputs):
            results_file = os.path.join(results_dir, f"results_gpu{gpu_id}.jsonl")
            tasks.append(run_gemma_shard(
                items=items, port=port, gpu_id=gpu_id,
                output_dir=prompt_output_dir, results_file=results_file,
                gemma_params=gemma_params,
            ))

        t0 = time.time()
        all_results = await asyncio.gather(*tasks)
        stage_time = time.time() - t0

        merged = {}
        for shard_results in all_results:
            for r in shard_results:
                merged[r["sample_id"]] = r

        with open(os.path.join(results_dir, "stage_summary.json"), "w") as f:
            json.dump({"stage": "gemma", "prompt_id": prompt_id,
                        "latency_s": round(stage_time, 1),
                        "timestamp": datetime.now().isoformat()}, f, indent=2)

        log.info("STAGE 1 COMPLETE: %.1fs", stage_time)

    finally:
        log.info("Unloading Gemma from all GPUs...")
        stop_servers(procs)
        time.sleep(5)

    manifest = _verify_stage_barrier("gemma", merged, len(samples), gpu_ids)
    manifest["output_dir"] = prompt_output_dir
    return merged, manifest


# ── Stage 2: FLUX ─────────────────────────────────────────────────

async def run_stage2_flux(
    samples: list[dict],
    gemma_results: dict[str, dict],
    prompt_id: str,
    output_root: str,
    flux_model_path: str,
    gpu_ids: list[int],
    flux_config: dict | None = None,
    negative_prompt: str | None = None,
) -> tuple[dict[str, dict], dict]:
    """Returns (merged_results, manifest)."""
    log.info("=" * 60)
    log.info("STAGE 2: FLUX — %s (%d samples, %d GPUs)", prompt_id, len(samples), len(gpu_ids))
    log.info("=" * 60)

    kill_gpu_hogs()
    time.sleep(3)
    if not verify_gpus_free(gpu_ids, min_free_gib=40.0):
        log.warning("GPUs not fully free before FLUX load")

    config = flux_config or {}
    shards = split_into_shards(samples, len(gpu_ids))

    stage_dir = os.path.join(output_root, prompt_id, "stage2_flux")
    os.makedirs(stage_dir, exist_ok=True)

    shard_input_files = []
    shard_output_files = []

    for gpu_id, shard in zip(gpu_ids, shards):
        flux_items = build_flux_inputs(shard, gemma_results, output_root, prompt_id,
                                        negative_prompt=negative_prompt)
        input_file = os.path.join(stage_dir, f"shard_input_gpu{gpu_id}.json")
        output_file = os.path.join(stage_dir, f"shard_output_gpu{gpu_id}.json")
        write_shard_file(flux_items, input_file)
        shard_input_files.append(input_file)
        shard_output_files.append(output_file)
        log.info("GPU %d: %d items", gpu_id, len(flux_items))

    t0 = time.time()
    procs = start_flux_workers(flux_model_path, gpu_ids, shard_input_files, shard_output_files, config)
    exit_codes = wait_for_flux_workers(procs, timeout_s=7200)
    stage_time = time.time() - t0

    merged = {}
    for gpu_id, output_file in zip(gpu_ids, shard_output_files):
        if os.path.exists(output_file):
            with open(output_file) as f:
                for r in json.load(f):
                    merged[r["sample_id"]] = r
        else:
            log.error("GPU %d output file missing: %s", gpu_id, output_file)

    with open(os.path.join(stage_dir, "stage_summary.json"), "w") as f:
        json.dump({"stage": "flux", "prompt_id": prompt_id,
                    "exit_codes": exit_codes, "latency_s": round(stage_time, 1),
                    "timestamp": datetime.now().isoformat()}, f, indent=2)

    log.info("STAGE 2 COMPLETE: %.1fs", stage_time)
    time.sleep(3)  # FLUX subprocesses already exited — VRAM freed by OS

    manifest = _verify_stage_barrier("flux", merged, len(samples), gpu_ids, check_vram=False)
    manifest["output_dir"] = os.path.join(stage_dir, "images")
    return merged, manifest


# ── Stage 3: Qwen ─────────────────────────────────────────────────

async def run_stage3_qwen(
    samples: list[dict],
    flux_results: dict[str, dict],
    prompt_id: str,
    output_root: str,
    qwen_model_path: str,
    gpu_ids: list[int],
    ports: list[int],
) -> tuple[dict, dict]:
    """Returns (eval_summary, manifest)."""
    log.info("=" * 60)
    log.info("STAGE 3: QWEN — %s (%d samples, %d GPUs)", prompt_id, len(samples), len(gpu_ids))
    log.info("=" * 60)

    kill_gpu_hogs()
    time.sleep(3)
    if not verify_gpus_free(gpu_ids, min_free_gib=40.0):
        log.warning("GPUs not fully free before Qwen load")

    procs = start_vllm_servers(
        qwen_model_path, ports, gpu_ids,
        gpu_memory_utilization=0.90, max_model_len=32768,
        extra_args=["--limit-mm-per-prompt", '{"image": 3}'],
    )

    try:
        shards = split_into_shards(samples, len(gpu_ids))
        eval_output_dir = os.path.join(output_root, prompt_id, "stage3_qwen", "evals")
        results_dir = os.path.join(output_root, prompt_id, "stage3_qwen")
        os.makedirs(results_dir, exist_ok=True)

        shard_inputs = [build_qwen_inputs(shard, flux_results) for shard in shards]

        tasks = []
        for gpu_id, port, items in zip(gpu_ids, ports, shard_inputs):
            results_file = os.path.join(results_dir, f"results_gpu{gpu_id}.jsonl")
            tasks.append(run_qwen_shard(
                items=items, port=port, gpu_id=gpu_id,
                output_dir=eval_output_dir, results_file=results_file,
            ))

        t0 = time.time()
        all_results = await asyncio.gather(*tasks)
        stage_time = time.time() - t0

        # Aggregate
        yes_count = no_count = parse_fail = total_evaluated = 0
        merged = {}
        for shard_results in all_results:
            for r in shard_results:
                merged[r["sample_id"]] = r
                if r.get("status") not in ("ok", "skipped"):
                    continue
                total_evaluated += 1
                parsed = r.get("parsed")
                if parsed == "yes":
                    yes_count += 1
                elif parsed == "no":
                    no_count += 1
                else:
                    parse_fail += 1

        valid = yes_count + no_count
        pass_rate = yes_count / valid if valid > 0 else 0.0

        eval_summary = {
            "prompt_id": prompt_id,
            "total_evaluated": total_evaluated,
            "yes": yes_count, "no": no_count, "parse_fail": parse_fail,
            "pass_rate": round(pass_rate, 4),
            "pass_rate_pct": f"{pass_rate * 100:.1f}%",
            "latency_s": round(stage_time, 1),
            "timestamp": datetime.now().isoformat(),
        }

        with open(os.path.join(results_dir, "stage_summary.json"), "w") as f:
            json.dump(eval_summary, f, indent=2)
        with open(os.path.join(output_root, prompt_id, "eval_summary.json"), "w") as f:
            json.dump(eval_summary, f, indent=2)

        log.info("STAGE 3 COMPLETE: pass_rate=%s, %.1fs", eval_summary["pass_rate_pct"], stage_time)

    finally:
        log.info("Unloading Qwen from all GPUs...")
        stop_servers(procs)
        time.sleep(5)

    manifest = _verify_stage_barrier("qwen", merged, len(samples), gpu_ids)
    manifest["output_dir"] = eval_output_dir
    return eval_summary, manifest


# ── CLI (direct execution without controller) ─────────────────────

async def run_pipeline(config, dataset_root, output_root, prompt_ids, limit=0):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from agent_sim.prompts import SYSTEM_PROMPTS
    except ImportError:
        sys.path.insert(0, os.path.join(_project_root, "..", "eval_run"))
        from prompts_data import system_prompt_for_tryon_v5, system_prompt_for_tryon_v5_new
        SYSTEM_PROMPTS = {"system_prompt_for_tryon_v5": system_prompt_for_tryon_v5,
                          "system_prompt_for_tryon_v5_new": system_prompt_for_tryon_v5_new}

    gemma_path = config.get("models", {}).get("gemma4", {}).get("path", "")
    flux_path = config.get("models", {}).get("flux2_klein", {}).get("path", "")
    qwen_path = config.get("models", {}).get("qwen3_vl", {}).get("path", "")
    gpu_ids = [int(g) for g in config.get("gpu", {}).get("visible_devices", "0,1,2").split(",")]

    gemma_ports = config.get("workflow", {}).get("gemma_ports", [8431, 8432, 8433])
    qwen_ports = config.get("workflow", {}).get("qwen_ports", [8434, 8435, 8436])

    flux_config = {k: config.get("models", {}).get("flux2_klein", {}).get(k, d)
                   for k, d in [("height", 1376), ("width", 768), ("num_inference_steps", 4),
                                ("guidance_scale", 1.0), ("seed", 0)]}

    neg_targets = config.get("prompts", {}).get("negative_prompt_targets", [])
    neg_text = next((nt["text"] for nt in neg_targets if nt.get("text")), None)

    samples = discover_samples(dataset_root,
                                config.get("data", {}).get("categories", ["dress", "lower", "upper"]),
                                config.get("data", {}).get("image_pair_patterns", [
                                    {"model": "model.png", "cloth": "cloth.png"},
                                    {"model": "input_model.png", "cloth": "input_cloth.png"}]))
    if limit > 0:
        samples = samples[:limit]
    if not samples:
        log.error("No samples found!"); sys.exit(1)

    os.makedirs(output_root, exist_ok=True)
    all_summaries = []
    for prompt_id in prompt_ids:
        prompt_text = SYSTEM_PROMPTS.get(prompt_id)
        if not prompt_text:
            continue
        gemma_results, _ = await run_stage1_gemma(samples, prompt_text, prompt_id, output_root,
                                                    gemma_path, gpu_ids, gemma_ports[:len(gpu_ids)])
        flux_results, _ = await run_stage2_flux(samples, gemma_results, prompt_id, output_root,
                                                 flux_path, gpu_ids, flux_config, neg_text)
        eval_summary, _ = await run_stage3_qwen(samples, flux_results, prompt_id, output_root,
                                                  qwen_path, gpu_ids, qwen_ports[:len(gpu_ids)])
        all_summaries.append(eval_summary)

    with open(os.path.join(output_root, "final_summary.json"), "w") as f:
        json.dump({"results": all_summaries}, f, indent=2)
    for s in all_summaries:
        log.info("  %s: pass_rate=%s", s["prompt_id"], s["pass_rate_pct"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--prompts", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)
    output_root = args.output or config.get("paths", {}).get("output_root", "outputs/v02")
    prompt_ids = args.prompts or config.get("prompts", {}).get("system_prompt_targets",
                    ["system_prompt_for_tryon_v5", "system_prompt_for_tryon_v5_new"])
    os.environ["LOG_DIR"] = output_root
    asyncio.run(run_pipeline(config, args.dataset, output_root, prompt_ids, args.limit))

if __name__ == "__main__":
    main()
