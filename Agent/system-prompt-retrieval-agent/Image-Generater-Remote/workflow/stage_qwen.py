"""Stage 3 — Qwen evaluation worker.

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

from workflow.image_preprocess import to_base64_url

log = logging.getLogger("stage_qwen")

EVAL_INSTRUCTION = """Please compare three images:

1. Image 1: Original person image showing appearance and pose.
2. Image 2: Target clothing image showing garment details.
3. Image 3: Final result showing person wearing target clothing.

Tasks:
- Judge if Image 2's clothing is accurately worn by Image 1's person.
- Ensure non-clothing areas (hair, face, background) remain unchanged between Image 1 and Image 3.
- Judge if a valid try-on result was generated.

Output:
- If Image 2's clothing is perfectly worn and Image 3 has no non-clothing changes, output "yes".
- Otherwise, output "no".
"""


def _parse_yes_no(raw: str) -> str:
    """Parse yes/no from raw VLM response."""
    if not raw or not raw.strip():
        return "parse_fail_empty"
    text = raw.strip().lower()
    # Direct match
    if text in ("yes", "y"):
        return "yes"
    if text in ("no", "n"):
        return "no"
    # First word
    first_word = text.split()[0].strip(".,!?;:")
    if first_word in ("yes", "y", "true", "1"):
        return "yes"
    if first_word in ("no", "n", "false", "0"):
        return "no"
    # Contains check (only if unambiguous)
    has_yes = "yes" in text
    has_no = "no" in text
    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"
    if has_yes and has_no:
        return "parse_fail_multiline"
    return "parse_fail_unknown"


def _run_qwen_shard_sync(
    items: list[dict],
    port: int,
    gpu_id: int,
    output_dir: str,
    results_file: str,
) -> list[dict]:
    """Synchronous implementation — runs in a thread via asyncio.to_thread."""
    from openai import OpenAI

    client = OpenAI(api_key="EMPTY", base_url=f"http://127.0.0.1:{port}/v1")
    models = client.models.list()
    model_name = models.data[0].id if models.data else "unknown"
    log.info("[GPU %d] Qwen model: %s, evaluating %d items", gpu_id, model_name, len(items))

    results = []
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    with open(results_file, "w") as rf:
        for idx, item in enumerate(items):
            sample_id = item["sample_id"]
            eval_path = os.path.join(output_dir, sample_id, "eval.json")

            if os.path.exists(eval_path):
                try:
                    with open(eval_path) as f:
                        existing = json.load(f)
                    result = {
                        "sample_id": sample_id,
                        "raw_response": existing.get("raw_response", ""),
                        "parsed": existing.get("parsed", ""),
                        "parse_status": existing.get("parse_status", ""),
                        "status": "skipped",
                    }
                    results.append(result)
                    rf.write(json.dumps(result, ensure_ascii=False) + "\n")
                    rf.flush()
                    continue
                except Exception:
                    pass

            log.info("[GPU %d] [%d/%d] %s", gpu_id, idx + 1, len(items), sample_id)

            try:
                # Downscale images for eval to fit within VLM context window.
                # Full-res 768x1376 × 3 images ≈ 33K tokens; half-res fits in 32K.
                def _load_and_downscale(path, max_dim=800):
                    img = Image.open(path)
                    w, h = img.size
                    if max(w, h) > max_dim:
                        scale = max_dim / max(w, h)
                        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                    return img

                img_model = _load_and_downscale(item["model_path"])
                img_cloth = _load_and_downscale(item["cloth_path"])
                img_result = _load_and_downscale(item["result_image_path"])

                url_model = to_base64_url(img_model)
                url_cloth = to_base64_url(img_cloth)
                url_result = to_base64_url(img_result)

                t0 = time.time()
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": url_model}},
                            {"type": "image_url", "image_url": {"url": url_cloth}},
                            {"type": "image_url", "image_url": {"url": url_result}},
                            {"type": "text", "text": EVAL_INSTRUCTION},
                        ]},
                    ],
                    temperature=0.0,
                    max_tokens=2048,
                )
                latency = time.time() - t0
                raw_response = response.choices[0].message.content

                parsed = _parse_yes_no(raw_response)
                parse_status = f"matched_{parsed}" if parsed in ("yes", "no") else parsed

                log.info("[GPU %d]   Qwen (%.1fs): %s → %s", gpu_id, latency, raw_response[:50], parsed)

                eval_dir = os.path.join(output_dir, sample_id)
                os.makedirs(eval_dir, exist_ok=True)
                eval_data = {
                    "sample_id": sample_id,
                    "raw_response": raw_response,
                    "parsed": parsed if parsed in ("yes", "no") else None,
                    "parse_status": parse_status,
                    "latency_s": round(latency, 2),
                }
                with open(eval_path, "w") as f:
                    json.dump(eval_data, f, indent=2, ensure_ascii=False)

                result = {**eval_data, "status": "ok"}

            except Exception as e:
                log.error("[GPU %d]   ERROR on %s: %s", gpu_id, sample_id, e)
                traceback.print_exc()
                result = {"sample_id": sample_id, "status": "error", "error": str(e)}

            results.append(result)
            rf.write(json.dumps(result, ensure_ascii=False) + "\n")
            rf.flush()

    ok_count = sum(1 for r in results if r["status"] in ("ok", "skipped"))
    log.info("[GPU %d] Qwen shard complete: %d/%d ok", gpu_id, ok_count, len(items))
    return results


async def run_qwen_shard(
    items: list[dict],
    port: int,
    gpu_id: int,
    output_dir: str,
    results_file: str,
) -> list[dict]:
    """Async wrapper — runs sync shard work in a thread so asyncio.gather
    achieves true parallelism across GPUs."""
    return await asyncio.to_thread(
        _run_qwen_shard_sync,
        items, port, gpu_id, output_dir, results_file,
    )
