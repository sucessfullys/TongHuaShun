"""Simulated agent — drives Gemma -> FLUX -> Qwen pipeline end-to-end.

Runs on the remote 3H100 server, targeting the 3 supervisors directly.

Usage:
    python agent_sim/run_pipeline.py --config config.yaml [--limit N] [--run-id ID]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

import httpx

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.config import load_config, deep_get, model_config, all_host_ports
from server.manifest import build_manifest
from pipelines.walker import walk_dataset
from pipelines.sharding import split_indices
from pipelines.records import (
    write_intermediate_prompt, write_generation_record,
    write_eval_record, write_run_record,
    write_stage_record, write_eval_summary,
)
from agent_sim.prompts import SYSTEM_PROMPTS, DEFAULT_USER_INSTRUCTION

logging.basicConfig(level=logging.INFO, format="%(asctime)s [pipeline] %(levelname)s %(message)s")
log = logging.getLogger("pipeline")


# ── HTTP helpers ───────────────────────────────────────────────────

async def post_json(url: str, data: dict, timeout_s: int = 1800) -> dict:
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.post(url, json=data)
        resp.raise_for_status()
        return resp.json()


async def load_model(hosts: list[str], model_name: str) -> list[dict]:
    """Load a model on all specified hosts."""
    results = []
    for host in hosts:
        log.info("Loading %s on %s", model_name, host)
        r = await post_json(f"{host}/load", {"model_name": model_name})
        results.append(r)
        log.info("Loaded on %s: %s", host, r.get("status"))
    return results


async def unload_model(hosts: list[str]) -> list[dict]:
    """Unload from all hosts."""
    results = []
    for host in hosts:
        log.info("Unloading from %s", host)
        r = await post_json(f"{host}/unload", {})
        results.append(r)
        log.info("Unloaded from %s: mem %.1f→%.1f GiB",
                 host, r.get("mem_before_gib", 0), r.get("mem_after_gib", 0))
    return results


async def infer_shard(host: str, model_name: str, items: list[dict],
                      envelope: dict, timeout_s: int = 1800) -> dict:
    """Send /infer/list to one host with its shard items."""
    payload = {**envelope, "items": items}
    return await post_json(f"{host}/infer/list", payload, timeout_s=timeout_s)


async def run_stage(stage_id: str, model_name: str, hosts: list[str],
                    shards: list[list[dict]], envelope: dict,
                    timeout_s: int = 1800) -> dict:
    """Run one pipeline stage across all hosts. No return_exceptions."""
    log.info("Stage %s: dispatching to %d hosts", stage_id, len(hosts))
    t0 = time.monotonic()

    coros = [
        infer_shard(hosts[i], model_name, shards[i], envelope, timeout_s)
        for i in range(len(hosts))
    ]
    # asyncio.gather WITHOUT return_exceptions=True — any failure raises
    results = await asyncio.gather(*coros)

    total_ok = sum(r.get("ok_count", 0) for r in results)
    total_err = sum(r.get("error_count", 0) for r in results)
    total = sum(r.get("total", 0) for r in results)
    elapsed_ms = (time.monotonic() - t0) * 1000

    stage_result = {
        "stage_id": stage_id,
        "overall_ok": all(r.get("ok", False) for r in results),
        "expected_host_count": len(hosts),
        "actual_host_count": len(results),
        "total_samples": total,
        "ok_count": total_ok,
        "error_count": total_err,
        "hosts": [{"host": hosts[i], "result": results[i]} for i in range(len(results))],
        "latency_ms": round(elapsed_ms, 1),
    }

    if not stage_result["overall_ok"] or total_err > 0:
        raise RuntimeError(f"Stage {stage_id} failed: ok_count={total_ok}, errors={total_err}")

    log.info("Stage %s complete: %d samples, %.1fs", stage_id, total, elapsed_ms / 1000)
    return stage_result


# ── Main pipeline ──────────────────────────────────────────────────

async def run_pipeline(cfg: dict, run_id: str, limit: int = 0):
    """Run the full pipeline for all cells."""

    base_url = f"http://{deep_get(cfg, 'server.host', '127.0.0.1')}"
    ports = all_host_ports(cfg)
    hosts = [f"{base_url}:{p}" for p in ports]
    output_root = deep_get(cfg, "paths.output_root", "outputs/v01")
    run_dir = os.path.join(output_root, "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Save config snapshot
    import yaml
    with open(os.path.join(run_dir, "config_snapshot.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Check model assets
    log.info("Checking model assets...")
    # (resolve_pilot_input should have been run already)

    # Build walker input
    pilot_json_path = "resolved_pilot_input.json"
    if os.path.exists(pilot_json_path):
        with open(pilot_json_path) as f:
            pilot_info = json.load(f)
        source_root = pilot_info["picked"]
    else:
        candidates = deep_get(cfg, "paths.pilot_input_candidates", [])
        source_root = candidates[0] if candidates else ""
        log.warning("resolved_pilot_input.json not found, using first candidate: %s", source_root)

    categories = deep_get(cfg, "data.categories", ["dress", "lower", "upper"])
    pair_patterns = deep_get(cfg, "data.image_pair_patterns", [])

    log.info("Walking dataset from %s", source_root)
    samples = walk_dataset(source_root, categories, pair_patterns)
    if limit > 0:
        samples = samples[:limit]
    log.info("Found %d samples", len(samples))

    if not samples:
        log.error("No samples found — aborting")
        sys.exit(1)

    # Build manifest
    manifest = build_manifest(source_root, samples)
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    with open(os.path.join(run_dir, "input_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(run_dir, "input_manifest.sha256"), "w") as f:
        f.write(manifest["manifest_hash"])

    # Ports manifest
    with open(os.path.join(run_dir, "ports_manifest.json"), "w") as f:
        json.dump({"ports": ports, "hosts": [str(h) for h in hosts]}, f, indent=2)

    # Determine host strategy per model
    def get_stage_hosts(model_name: str) -> list[str]:
        dm = deep_get(cfg, f"models.{model_name}.deployment_mode", "single_gpu_replicated")
        if dm == "tensor_parallel_single_service":
            return [hosts[0]]  # Only GPU0
        return hosts

    # Shard samples
    def shard_for_hosts(stage_hosts: list[str]) -> list[list[dict]]:
        n_shards = len(stage_hosts)
        indices = split_indices(len(samples), n_shards)
        return [[samples[i] for i in shard] for shard in indices]

    # System prompts and negative prompts from config
    sys_prompt_ids = deep_get(cfg, "prompts.system_prompt_targets", [])
    neg_prompts = deep_get(cfg, "prompts.negative_prompt_targets", [])
    user_instruction = deep_get(cfg, "prompts.default_user_instruction", DEFAULT_USER_INSTRUCTION)
    callback_url = deep_get(cfg, "callback.default_url", "")

    eval_cells = []

    for sys_prompt_id in sys_prompt_ids:
        sys_prompt_text = SYSTEM_PROMPTS.get(sys_prompt_id, "")
        if not sys_prompt_text:
            log.error("System prompt %s not found in SYSTEM_PROMPTS", sys_prompt_id)
            continue

        for neg_entry in neg_prompts:
            neg_id = neg_entry["id"]
            neg_text = neg_entry.get("text")
            cell_id = f"{sys_prompt_id}__{neg_id}"
            cell_dir = os.path.join(run_dir, "samples", sys_prompt_id, neg_id)

            log.info("=== Cell: %s ===", cell_id)

            envelope = {
                "run_id": run_id,
                "cell_id": cell_id,
                "manifest_hash": manifest["manifest_hash"],
                "callback_url": callback_url,
            }

            # ── GEMMA STAGE ──
            gemma_hosts = get_stage_hosts("gemma4")
            gemma_shards = shard_for_hosts(gemma_hosts)

            try:
                await load_model(gemma_hosts, "gemma4")
                # F10g: removed dead gemma_items list (S6 fix)
                gemma_shard_payloads = []
                for shard in gemma_shards:
                    items = [{
                        "system_prompt": sys_prompt_text,
                        "user_instruction": user_instruction,
                        "model_image_path": os.path.join(source_root, s["model_image"]),
                        "cloth_image_path": os.path.join(source_root, s["cloth_image"]),
                        "sample_id": s["sample_id"],
                        "index": s["index"],
                    } for s in shard]
                    gemma_shard_payloads.append(items)

                gemma_result = await run_stage(
                    "gemma", "gemma4", gemma_hosts,
                    gemma_shard_payloads, {**envelope, "stage_id": "gemma"},
                )
                write_stage_record(run_dir, "gemma", gemma_result)

                # Collect intermediate prompts from results
                intermediate_prompts = {}
                for host_data in gemma_result["hosts"]:
                    for item in host_data["result"].get("items", []):
                        sid = item.get("sample_id", "")
                        result_data = item.get("result", {})
                        prompt_text = result_data.get("intermediate_prompt", "")
                        intermediate_prompts[sid] = prompt_text
                        sample_dir = os.path.join(cell_dir, sid)
                        write_intermediate_prompt(sample_dir, sid, prompt_text, result_data)

                await unload_model(gemma_hosts)

            except Exception as e:
                log.error("Gemma stage failed: %s", e)
                try:
                    await unload_model(gemma_hosts)
                except Exception:
                    pass
                write_stage_record(run_dir, "gemma", {
                    "stage_id": "gemma", "overall_ok": False,
                    "error": str(e), "cell_id": cell_id,
                })
                sys.exit(1)

            # ── FLUX STAGE ──
            flux_hosts = get_stage_hosts("flux2_klein")
            flux_shard_items = shard_for_hosts(flux_hosts)

            try:
                await load_model(flux_hosts, "flux2_klein")
                flux_shard_payloads = []
                for shard in flux_shard_items:
                    items = [{
                        "intermediate_prompt": intermediate_prompts.get(s["sample_id"], ""),
                        "negative_prompt": neg_text,
                        "model_image_path": os.path.join(source_root, s["model_image"]),
                        "cloth_image_path": os.path.join(source_root, s["cloth_image"]),
                        # F7: unique output_relpath per sample to prevent collisions
                        "output_relpath": f"runs/{run_id}/samples/{sys_prompt_id}/{neg_id}/{s['sample_id']}/tryon_result_flux2klein_agent.png",
                        "sample_id": s["sample_id"],
                        "index": s["index"],
                    } for s in shard]
                    flux_shard_payloads.append(items)

                flux_result = await run_stage(
                    "flux", "flux2_klein", flux_hosts,
                    flux_shard_payloads, {**envelope, "stage_id": "flux"},
                )
                write_stage_record(run_dir, "flux", flux_result)

                # Collect output paths
                flux_outputs = {}
                for host_data in flux_result["hosts"]:
                    for item in host_data["result"].get("items", []):
                        sid = item.get("sample_id", "")
                        result_data = item.get("result", {})
                        flux_outputs[sid] = result_data.get("output_path", "")
                        sample_dir = os.path.join(cell_dir, sid)
                        write_generation_record(sample_dir, sid, {
                            "negative_prompt_applied": result_data.get("negative_prompt_applied", False),
                            "negative_prompt_normalized": result_data.get("negative_prompt_normalized"),
                            "seed_used": result_data.get("seed_used", 0),
                            **result_data,
                        })

                await unload_model(flux_hosts)

            except Exception as e:
                log.error("FLUX stage failed: %s", e)
                try:
                    await unload_model(flux_hosts)
                except Exception:
                    pass
                write_stage_record(run_dir, "flux", {
                    "stage_id": "flux", "overall_ok": False,
                    "error": str(e), "cell_id": cell_id,
                })
                sys.exit(1)

            # ── QWEN STAGE ──
            qwen_hosts = get_stage_hosts("qwen3_vl")
            qwen_shard_items = shard_for_hosts(qwen_hosts)

            try:
                await load_model(qwen_hosts, "qwen3_vl")
                qwen_shard_payloads = []
                for shard in qwen_shard_items:
                    items = [{
                        "model_image_path": os.path.join(source_root, s["model_image"]),
                        "cloth_image_path": os.path.join(source_root, s["cloth_image"]),
                        "result_image_path": flux_outputs.get(s["sample_id"], ""),
                        "eval_instruction": "Evaluate the virtual try-on result.",
                        "sample_id": s["sample_id"],
                        "index": s["index"],
                    } for s in shard]
                    qwen_shard_payloads.append(items)

                qwen_result = await run_stage(
                    "qwen", "qwen3_vl", qwen_hosts,
                    qwen_shard_payloads, {**envelope, "stage_id": "qwen"},
                )
                write_stage_record(run_dir, "qwen", qwen_result)

                # Collect eval results for summary
                cell_total = 0
                cell_yes = 0
                cell_no = 0
                cell_parse_fail = 0

                for host_data in qwen_result["hosts"]:
                    for item in host_data["result"].get("items", []):
                        sid = item.get("sample_id", "")
                        result_data = item.get("result", {})
                        sample_dir = os.path.join(cell_dir, sid)
                        write_eval_record(sample_dir, sid, {
                            "raw_response": result_data.get("raw_response", ""),
                            "parsed": result_data.get("parsed"),
                            "parse_status": result_data.get("parse_status", ""),
                        })
                        write_run_record(sample_dir, sid, {
                            "run_id": run_id,
                            "cell_id": cell_id,
                            "sample_id": sid,
                            "system_prompt_id": sys_prompt_id,
                            "negative_prompt_id": neg_id,
                        })

                        cell_total += 1
                        parsed = result_data.get("parsed")
                        if parsed == "yes":
                            cell_yes += 1
                        elif parsed == "no":
                            cell_no += 1
                        else:
                            cell_parse_fail += 1

                await unload_model(qwen_hosts)

            except Exception as e:
                log.error("Qwen stage failed: %s", e)
                try:
                    await unload_model(qwen_hosts)
                except Exception:
                    pass
                write_stage_record(run_dir, "qwen", {
                    "stage_id": "qwen", "overall_ok": False,
                    "error": str(e), "cell_id": cell_id,
                })
                sys.exit(1)

            # Cell summary
            denom = cell_total - cell_parse_fail
            yes_ratio = cell_yes / denom if denom > 0 else None
            eval_cells.append({
                "system_prompt_id": sys_prompt_id,
                "negative_prompt_id": neg_id,
                "cell_id": cell_id,
                "total": cell_total,
                "yes": cell_yes,
                "no": cell_no,
                "parse_fail": cell_parse_fail,
                "yes_ratio": yes_ratio,
            })
            log.info("Cell %s: total=%d yes=%d no=%d parse_fail=%d yes_ratio=%s",
                     cell_id, cell_total, cell_yes, cell_no, cell_parse_fail,
                     f"{yes_ratio:.2%}" if yes_ratio is not None else "N/A")

    # Write eval summary
    write_eval_summary(run_dir, eval_cells)
    log.info("Pipeline complete. %d cells processed.", len(eval_cells))
    log.info("Eval summary written to %s/eval_summary.json", run_dir)

    return eval_cells


# ── CLI ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="V0.1 Simulated Agent Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--run-id", default=None, help="Run ID (auto-generated if omitted)")
    parser.add_argument("--limit", type=int, default=0, help="Limit samples (0=all)")
    parser.add_argument("--image-root", default=None, help="Override pilot image root")
    parser.add_argument("--system-prompts", nargs="*", default=None, help="Override system prompt IDs")
    parser.add_argument("--negative-prompts", nargs="*", default=None, help="Override negative prompt IDs")
    parser.add_argument("--resume", action="store_true", help="Resume from partial run")
    args = parser.parse_args()

    cfg = load_config(args.config)

    run_id = args.run_id or f"v01_pilot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if args.system_prompts:
        cfg.setdefault("prompts", {})["system_prompt_targets"] = args.system_prompts
    if args.negative_prompts:
        # Rebuild neg prompt entries
        cfg.setdefault("prompts", {})["negative_prompt_targets"] = [
            {"id": nid, "text": None if nid == "neg_none" else nid}
            for nid in args.negative_prompts
        ]

    asyncio.run(run_pipeline(cfg, run_id, limit=args.limit))


if __name__ == "__main__":
    main()
