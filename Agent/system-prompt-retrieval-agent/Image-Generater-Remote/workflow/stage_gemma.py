"""Stage 1 — Gemma prompt generation worker.

Processes a shard of samples against one vLLM server on one GPU.
Called by the orchestrator as an async task; one task per GPU.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import traceback
from typing import Any

from PIL import Image

from workflow.image_preprocess import resize_model_image, resize_cloth_image, to_base64_url

log = logging.getLogger("stage_gemma")


def _run_gemma_shard_sync(
    items: list[dict],
    port: int,
    gpu_id: int,
    output_dir: str,
    results_file: str,
    target_height: int = 1376,
    target_width: int = 768,
    gemma_params: dict[str, Any] | None = None,
) -> list[dict]:
    """Synchronous implementation — runs in a thread via asyncio.to_thread."""
    from openai import OpenAI

    params = gemma_params or {}
    temperature = params.get("temperature", 0.1)
    top_p = params.get("top_p", 0.5)
    presence_penalty = params.get("presence_penalty", 1.5)
    top_k = params.get("top_k", 20)

    client = OpenAI(api_key="EMPTY", base_url=f"http://127.0.0.1:{port}/v1")

    models = client.models.list()
    model_name = models.data[0].id if models.data else "unknown"
    log.info("[GPU %d] Gemma model: %s, processing %d items", gpu_id, model_name, len(items))

    results = []
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    with open(results_file, "w") as rf:
        for idx, item in enumerate(items):
            sample_id = item["sample_id"]
            sample_out = os.path.join(output_dir, sample_id)
            prompt_txt_path = os.path.join(sample_out, "intermediate_prompt.txt")

            if os.path.exists(prompt_txt_path):
                try:
                    with open(prompt_txt_path) as f:
                        existing_prompt = f.read().strip()
                    result = {"sample_id": sample_id, "intermediate_prompt": existing_prompt, "status": "skipped"}
                    results.append(result)
                    rf.write(json.dumps(result, ensure_ascii=False) + "\n")
                    rf.flush()
                    continue
                except Exception:
                    pass

            log.info("[GPU %d] [%d/%d] %s", gpu_id, idx + 1, len(items), sample_id)

            try:
                img_model = Image.open(item["model_path"])
                img_model = resize_model_image(img_model, target_height, target_width)
                img_cloth = Image.open(item["cloth_path"])
                img_cloth = resize_cloth_image(img_cloth, target_height, target_width)

                url_model = to_base64_url(img_model)
                url_cloth = to_base64_url(img_cloth)

                t0 = time.time()
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": item["system_prompt"]},
                        {"role": "user", "content": [
                            {"type": "text", "text": "Image 1: Model Image - Shows the model wearing current clothing"},
                            {"type": "image_url", "image_url": {"url": url_model}},
                            {"type": "text", "text": "Image 2: Clothing Image - Shows the garment to be tried on"},
                            {"type": "image_url", "image_url": {"url": url_cloth}},
                            {"type": "text", "text": item["user_instruction"]},
                        ]},
                    ],
                    temperature=temperature,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    extra_body={"top_k": top_k, "chat_template_kwargs": {"enable_thinking": False}},
                )
                latency = time.time() - t0
                intermediate_prompt = response.choices[0].message.content

                os.makedirs(sample_out, exist_ok=True)
                with open(prompt_txt_path, "w") as f:
                    f.write(intermediate_prompt)
                with open(os.path.join(sample_out, "intermediate_prompt.json"), "w") as f:
                    json.dump({"prompt": intermediate_prompt, "latency_s": round(latency, 2)}, f, indent=2)

                log.info("[GPU %d]   Gemma (%.1fs): %s", gpu_id, latency, intermediate_prompt[:60])
                result = {
                    "sample_id": sample_id,
                    "intermediate_prompt": intermediate_prompt,
                    "latency_s": round(latency, 2),
                    "status": "ok",
                }

            except Exception as e:
                log.error("[GPU %d]   ERROR on %s: %s", gpu_id, sample_id, e)
                traceback.print_exc()
                result = {"sample_id": sample_id, "status": "error", "error": str(e)}

            results.append(result)
            rf.write(json.dumps(result, ensure_ascii=False) + "\n")
            rf.flush()

    ok_count = sum(1 for r in results if r["status"] in ("ok", "skipped"))
    log.info("[GPU %d] Gemma shard complete: %d/%d ok", gpu_id, ok_count, len(items))
    return results


async def run_gemma_shard(
    items: list[dict],
    port: int,
    gpu_id: int,
    output_dir: str,
    results_file: str,
    target_height: int = 1376,
    target_width: int = 768,
    gemma_params: dict[str, Any] | None = None,
) -> list[dict]:
    """Async wrapper — runs sync shard work in a thread so asyncio.gather
    achieves true parallelism across GPUs."""
    return await asyncio.to_thread(
        _run_gemma_shard_sync,
        items, port, gpu_id, output_dir, results_file,
        target_height, target_width, gemma_params,
    )
