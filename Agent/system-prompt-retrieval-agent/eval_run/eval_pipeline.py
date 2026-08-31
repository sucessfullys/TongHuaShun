#!/usr/bin/env python3
"""
Full evaluation pipeline: Gemma prompt generation → FLUX image generation → Qwen evaluation.

Processes the entire VirtualTryOn_whq_test500 dataset (513 samples) with two system prompts.
Runs entirely on the 3H100 remote server.

Usage:
    python eval_pipeline.py --stage generate  # Run Gemma + FLUX
    python eval_pipeline.py --stage evaluate  # Run Qwen evaluation
    python eval_pipeline.py --stage all       # Run everything
"""

import argparse
import json
import logging
import os
import sys
import time
import glob
import traceback
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("eval_pipeline.log")]
)
log = logging.getLogger("eval_pipeline")

# ── Configuration ──────────────────────────────────────────────────

DATASET_ROOT = "/mnt/TryOn_Data/VirtualTryOn_whq_test500/test_data"
OUTPUT_ROOT = "/mnt/image-edit/datasets/xywang/code/System-Prompt-Retrieval-Agent/eval_results"
CATEGORIES = ["dress", "lower", "upper"]

GEMMA_MODEL_PATH = "/mnt/image-edit/models/Google/Gemma-4-31B-it"
FLUX_MODEL_PATH = "/mnt/image-edit/models/black-forest-labs/FLUX.2-klein-9B/"
QWEN_MODEL_PATH = "/mnt/image-edit/models/Qwen/Qwen3-VL-8B-Instruct/"

GEMMA_PORT = 8431
QWEN_PORT = 8432

TARGET_HEIGHT = 1376
TARGET_WIDTH = 768

# System prompts (embedded directly to avoid import issues)
SYSTEM_PROMPTS = {}  # Populated at runtime from prompts_data.py


# ── Image preprocessing (from temp_wxy/utils_tryon.py) ─────────────

def convert_transparent_to_white(image):
    from PIL import Image
    if image.mode == 'RGBA':
        white_bg = Image.new('RGB', image.size, (255, 255, 255))
        white_bg.paste(image, mask=image.split()[3])
        return white_bg
    elif image.mode == 'P':
        rgba_img = image.convert('RGBA')
        white_bg = Image.new('RGB', rgba_img.size, (255, 255, 255))
        white_bg.paste(rgba_img, mask=rgba_img.split()[3])
        return white_bg
    return image.convert('RGB')


def fit_image_to_canvas(input_img, target_width=1360, target_height=2048):
    from PIL import Image
    original_width, original_height = input_img.size
    original_ratio = original_width / original_height
    target_ratio = target_width / target_height
    canvas = Image.new('RGB', (target_width, target_height), (255, 255, 255))
    if original_ratio > target_ratio:
        scale_factor = target_width / original_width
        new_width = target_width
        new_height = int(original_height * scale_factor)
        resized_image = input_img.resize((new_width, new_height), Image.LANCZOS)
        padding_top = (target_height - new_height) // 2
        canvas.paste(resized_image, (0, padding_top))
    else:
        scale_factor = target_height / original_height
        new_height = target_height
        new_width = int(original_width * scale_factor)
        resized_image = input_img.resize((new_width, new_height), Image.LANCZOS)
        padding_left = (target_width - new_width) // 2
        canvas.paste(resized_image, (padding_left, 0))
    return canvas


def center_crop_new(img, target_width, target_height):
    from PIL import Image
    original_width, original_height = img.size
    if original_width > target_width:
        x_start = (original_width - target_width) // 2
        x_end = x_start + target_width
        cropped = img.crop((x_start, 0, x_end, target_height))
    else:
        cropped = fit_image_to_canvas(img, target_width, target_height)
    return cropped


def resize_canvas(img, target_height, target_width):
    from PIL import Image, ImageOps
    image_Img = ImageOps.exif_transpose(img)
    if image_Img.mode == 'RGBA':
        image_Img = convert_transparent_to_white(image_Img)
    width_percent = target_height / float(image_Img.size[1])
    new_width = int(float(image_Img.size[0]) * width_percent)
    img_resized = image_Img.resize((new_width, target_height), Image.LANCZOS)
    cropped_Img = center_crop_new(img_resized, target_width, target_height)
    return cropped_Img


def resize_canvas_dd(img, target_height, target_width):
    from PIL import Image, ImageOps
    image_Img = ImageOps.exif_transpose(img)
    if image_Img.mode == 'RGBA':
        image_Img = convert_transparent_to_white(image_Img)
    original_width, original_height = image_Img.size
    ratio = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    img_resized = image_Img.resize((new_width, new_height), Image.LANCZOS)
    background = Image.new('RGB', (target_width, target_height), (255, 255, 255))
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    background.paste(img_resized, (x_offset, y_offset))
    return background


def base64_conversion(pil_img):
    from io import BytesIO
    import base64
    buffer = BytesIO()
    fmt = pil_img.format or "PNG"
    if fmt.upper() == "JPEG" and pil_img.mode in ("RGBA", "P"):
        pil_img = pil_img.convert("RGB")
    pil_img.save(buffer, format=fmt, quality=95)
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{img_base64}"


# ── Dataset discovery ──────────────────────────────────────────────

def discover_samples(dataset_root):
    """Find all (model.png, cloth.png) pairs in the dataset."""
    samples = []
    for cat in CATEGORIES:
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
                model_path = os.path.join(sample_dir, "model.png")
                cloth_path = os.path.join(sample_dir, "cloth.png")
                if os.path.exists(model_path) and os.path.exists(cloth_path):
                    samples.append({
                        "sample_id": f"{cat}/{subcat}/{sample_name}",
                        "category": cat,
                        "model_path": model_path,
                        "cloth_path": cloth_path,
                    })
    return samples


# ── Stage 1: Gemma + FLUX generation ──────────────────────────────

def run_generation(prompt_id, prompt_text, samples, output_root, flux_pipeline, gemma_client, gemma_model_name):
    """Generate intermediate prompts and try-on images for one system prompt."""
    from PIL import Image
    import torch

    prompt_dir = os.path.join(output_root, prompt_id)
    os.makedirs(prompt_dir, exist_ok=True)

    results = []
    total = len(samples)

    for idx, sample in enumerate(samples):
        sample_id = sample["sample_id"]
        sample_out_dir = os.path.join(prompt_dir, sample_id)
        result_path = os.path.join(sample_out_dir, "tryon_result_flux2klein_agent.png")

        # Skip if already exists
        if os.path.exists(result_path):
            log.info("[%d/%d] SKIP %s (exists)", idx + 1, total, sample_id)
            # Load existing result for eval
            try:
                with open(os.path.join(sample_out_dir, "intermediate_prompt.txt")) as f:
                    prompt_text_saved = f.read().strip()
                results.append({
                    "sample_id": sample_id,
                    "status": "skipped",
                    "intermediate_prompt": prompt_text_saved,
                    "result_path": result_path,
                })
            except Exception:
                results.append({"sample_id": sample_id, "status": "skipped", "result_path": result_path})
            continue

        os.makedirs(sample_out_dir, exist_ok=True)
        log.info("[%d/%d] Processing %s with %s", idx + 1, total, sample_id, prompt_id)

        try:
            # Load and preprocess images
            img_model = Image.open(sample["model_path"])
            img_model = resize_canvas(img_model, TARGET_HEIGHT, TARGET_WIDTH)
            img_cloth = Image.open(sample["cloth_path"])
            img_cloth = resize_canvas_dd(img_cloth, TARGET_HEIGHT, TARGET_WIDTH)

            width, height = img_model.size

            # Save preprocessed inputs
            img_model.save(os.path.join(sample_out_dir, "input_model.png"))
            img_cloth.save(os.path.join(sample_out_dir, "input_cloth.png"))

            # Step 1: Gemma — generate intermediate prompt
            url_model = base64_conversion(img_model)
            url_cloth = base64_conversion(img_cloth)

            user_instruction = "the model in image 1 wears the clothing from image 2"

            t0 = time.time()
            response = gemma_client.chat.completions.create(
                model=gemma_model_name,
                messages=[
                    {"role": "system", "content": prompt_text},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Image 1: Model Image - Shows the model wearing current clothing"},
                        {"type": "image_url", "image_url": {"url": url_model}},
                        {"type": "text", "text": "Image 2: Clothing Image - Shows the garment to be tried on"},
                        {"type": "image_url", "image_url": {"url": url_cloth}},
                        {"type": "text", "text": user_instruction},
                    ]}
                ],
                temperature=0.1,
                top_p=0.5,
                presence_penalty=1.5,
                extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
            )
            gemma_time = time.time() - t0
            intermediate_prompt = response.choices[0].message.content
            log.info("  Gemma (%.1fs): %s", gemma_time, intermediate_prompt[:80])

            # Save intermediate prompt
            with open(os.path.join(sample_out_dir, "intermediate_prompt.txt"), "w") as f:
                f.write(intermediate_prompt)
            with open(os.path.join(sample_out_dir, "intermediate_prompt.json"), "w") as f:
                json.dump({"prompt": intermediate_prompt, "latency_s": gemma_time}, f, indent=2)

            # Step 2: FLUX — generate try-on image
            t0 = time.time()
            flux_result = flux_pipeline(
                image=[img_model, img_cloth],
                prompt=intermediate_prompt,
                height=height,
                width=width,
                guidance_scale=1.0,
                num_inference_steps=4,
                generator=torch.Generator(device=flux_pipeline.device).manual_seed(0),
            ).images[0]
            flux_time = time.time() - t0
            log.info("  FLUX  (%.1fs): generated", flux_time)

            flux_result.save(result_path)

            # Save generation record
            with open(os.path.join(sample_out_dir, "generation.json"), "w") as f:
                json.dump({
                    "intermediate_prompt": intermediate_prompt,
                    "latency_gemma_s": round(gemma_time, 2),
                    "latency_flux_s": round(flux_time, 2),
                    "seed": 0,
                    "height": height,
                    "width": width,
                }, f, indent=2)

            results.append({
                "sample_id": sample_id,
                "status": "ok",
                "intermediate_prompt": intermediate_prompt,
                "result_path": result_path,
                "gemma_time": gemma_time,
                "flux_time": flux_time,
            })

        except Exception as e:
            log.error("  ERROR: %s", e)
            traceback.print_exc()
            results.append({
                "sample_id": sample_id,
                "status": "error",
                "error": str(e),
            })

    # Save generation summary
    ok_count = sum(1 for r in results if r["status"] in ("ok", "skipped"))
    summary = {
        "prompt_id": prompt_id,
        "total": len(samples),
        "ok": ok_count,
        "errors": len(samples) - ok_count,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(prompt_dir, "generation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Generation summary for %s: %d/%d ok", prompt_id, ok_count, len(samples))

    return results


# ── Stage 2: Qwen evaluation ──────────────────────────────────────

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


def run_evaluation(prompt_id, output_root, qwen_client, qwen_model_name):
    """Evaluate all generated results for one prompt using Qwen."""

    prompt_dir = os.path.join(output_root, prompt_id)
    eval_results = []
    total = 0
    yes_count = 0
    no_count = 0
    error_count = 0

    # Find all generated results
    for cat in CATEGORIES:
        cat_dir = os.path.join(prompt_dir, cat)
        if not os.path.isdir(cat_dir):
            continue
        for subcat in sorted(os.listdir(cat_dir)):
            subcat_dir = os.path.join(cat_dir, subcat)
            if not os.path.isdir(subcat_dir):
                continue
            for sample_name in sorted(os.listdir(subcat_dir)):
                sample_dir = os.path.join(subcat_dir, sample_name)
                result_path = os.path.join(sample_dir, "tryon_result_flux2klein_agent.png")
                model_path = os.path.join(sample_dir, "input_model.png")
                cloth_path = os.path.join(sample_dir, "input_cloth.png")

                if not all(os.path.exists(p) for p in [result_path, model_path, cloth_path]):
                    continue

                eval_json_path = os.path.join(sample_dir, "eval.json")
                if os.path.exists(eval_json_path):
                    # Load existing eval
                    try:
                        with open(eval_json_path) as f:
                            existing = json.load(f)
                        total += 1
                        parsed = existing.get("parsed", "").lower().strip()
                        if parsed == "yes":
                            yes_count += 1
                        elif parsed == "no":
                            no_count += 1
                        else:
                            error_count += 1
                        eval_results.append(existing)
                        continue
                    except Exception:
                        pass

                sample_id = f"{cat}/{subcat}/{sample_name}"
                total += 1
                log.info("[eval %d] %s/%s", total, prompt_id, sample_id)

                try:
                    from PIL import Image
                    # Build merged reference image (model + cloth side by side)
                    img_model = Image.open(model_path)
                    img_cloth = Image.open(cloth_path)
                    img_result = Image.open(result_path)

                    url_model = base64_conversion(img_model)
                    url_cloth = base64_conversion(img_cloth)
                    url_result = base64_conversion(img_result)

                    t0 = time.time()
                    response = qwen_client.chat.completions.create(
                        model=qwen_model_name,
                        messages=[
                            {"role": "user", "content": [
                                {"type": "image_url", "image_url": {"url": url_model}},
                                {"type": "image_url", "image_url": {"url": url_cloth}},
                                {"type": "image_url", "image_url": {"url": url_result}},
                                {"type": "text", "text": EVAL_INSTRUCTION},
                            ]}
                        ],
                        temperature=0.0,
                        max_tokens=2048,
                    )
                    eval_time = time.time() - t0

                    raw_response = response.choices[0].message.content
                    parsed = _parse_yes_no(raw_response)

                    if parsed == "yes":
                        yes_count += 1
                    elif parsed == "no":
                        no_count += 1
                    else:
                        error_count += 1

                    eval_data = {
                        "sample_id": sample_id,
                        "raw_response": raw_response,
                        "parsed": parsed,
                        "latency_s": round(eval_time, 2),
                    }
                    eval_results.append(eval_data)

                    with open(eval_json_path, "w") as f:
                        json.dump(eval_data, f, indent=2, ensure_ascii=False)

                    log.info("  Qwen (%.1fs): %s → %s", eval_time, raw_response[:60], parsed)

                except Exception as e:
                    log.error("  Eval ERROR: %s", e)
                    error_count += 1
                    eval_results.append({
                        "sample_id": sample_id,
                        "raw_response": "",
                        "parsed": "error",
                        "error": str(e),
                    })

    # Compute pass rate
    valid = yes_count + no_count
    pass_rate = yes_count / valid if valid > 0 else 0.0

    summary = {
        "prompt_id": prompt_id,
        "total_evaluated": total,
        "yes": yes_count,
        "no": no_count,
        "parse_errors": error_count,
        "pass_rate": round(pass_rate, 4),
        "pass_rate_pct": f"{pass_rate * 100:.1f}%",
        "timestamp": datetime.now().isoformat(),
    }

    with open(os.path.join(prompt_dir, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log.info("Eval summary for %s: pass_rate=%.1f%% (%d yes / %d valid, %d errors)",
             prompt_id, pass_rate * 100, yes_count, valid, error_count)

    return summary


def _parse_yes_no(raw):
    """Parse yes/no from raw VLM response."""
    if not raw or not raw.strip():
        return "error"
    text = raw.strip().lower()
    # Direct match
    if text in ("yes", "y"):
        return "yes"
    if text in ("no", "n"):
        return "no"
    # Check first word
    first_word = text.split()[0].strip(".,!?;:")
    if first_word in ("yes", "y", "true", "1"):
        return "yes"
    if first_word in ("no", "n", "false", "0"):
        return "no"
    # Check if response contains yes/no
    if "yes" in text and "no" not in text:
        return "yes"
    if "no" in text and "yes" not in text:
        return "no"
    return "unknown"


# ── Server management ─────────────────────────────────────────────

def start_vllm_server(model_path, port, gpu_id, extra_args=None):
    """Start a vLLM OpenAI-compatible server in the background."""
    import subprocess
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--host", "127.0.0.1",
        "--trust-remote-code",
        "--gpu-memory-utilization", "0.95",
        "--max-model-len", "8192",
    ]
    if extra_args:
        cmd.extend(extra_args)

    log.info("Starting vLLM server: %s on GPU %d, port %d", os.path.basename(model_path), gpu_id, port)
    proc = subprocess.Popen(
        cmd, env=env,
        stdout=open(f"vllm_{port}.log", "w"),
        stderr=subprocess.STDOUT,
    )
    log.info("vLLM server PID=%d, waiting for readiness...", proc.pid)

    # Wait for server to be ready
    import urllib.request
    for attempt in range(180):  # up to 15 minutes
        time.sleep(5)
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    log.info("vLLM server ready on port %d (attempt %d)", port, attempt + 1)
                    return proc
        except Exception:
            pass
        if proc.poll() is not None:
            log.error("vLLM server died with code %d", proc.returncode)
            return None

    log.error("vLLM server did not become ready in time")
    proc.terminate()
    return None


def stop_server(proc):
    """Stop a background server process."""
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log.info("Server PID=%d stopped", proc.pid)


def kill_nogpualarm():
    """Kill NoGPUAlarm processes to free GPU memory."""
    import subprocess
    try:
        subprocess.run(["pkill", "-f", "NoGPUAlarmNew.py"], capture_output=True, timeout=10)
        time.sleep(2)
        subprocess.run(["pkill", "-9", "-f", "NoGPUAlarmNew.py"], capture_output=True, timeout=10)
        log.info("NoGPUAlarm killed")
    except Exception as e:
        log.warning("Failed to kill NoGPUAlarm: %s", e)


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="V0.1 Full Evaluation Pipeline")
    parser.add_argument("--stage", choices=["generate", "evaluate", "all"], default="all")
    parser.add_argument("--prompt", choices=["v5", "v5_new", "both"], default="both")
    parser.add_argument("--limit", type=int, default=0, help="Limit samples (0=all)")
    parser.add_argument("--output", default=OUTPUT_ROOT, help="Output root directory")
    parser.add_argument("--gemma-gpu", type=int, default=0)
    parser.add_argument("--flux-gpu", type=int, default=1)
    parser.add_argument("--qwen-gpu", type=int, default=0)
    args = parser.parse_args()

    output_root = args.output
    os.makedirs(output_root, exist_ok=True)

    # Load prompts
    from prompts_data import system_prompt_for_tryon_v5, system_prompt_for_tryon_v5_new
    prompts = {}
    if args.prompt in ("v5", "both"):
        prompts["system_prompt_for_tryon_v5"] = system_prompt_for_tryon_v5
    if args.prompt in ("v5_new", "both"):
        prompts["system_prompt_for_tryon_v5_new"] = system_prompt_for_tryon_v5_new

    # Discover samples
    log.info("Discovering samples from %s", DATASET_ROOT)
    samples = discover_samples(DATASET_ROOT)
    if args.limit > 0:
        samples = samples[:args.limit]
    log.info("Found %d samples", len(samples))

    if not samples:
        log.error("No samples found!")
        sys.exit(1)

    # Save sample manifest
    with open(os.path.join(output_root, "sample_manifest.json"), "w") as f:
        json.dump({"total": len(samples), "samples": [s["sample_id"] for s in samples]}, f, indent=2)

    if args.stage in ("generate", "all"):
        # Kill GPU hog processes
        kill_nogpualarm()
        time.sleep(3)

        # Start Gemma vLLM server
        gemma_proc = start_vllm_server(GEMMA_MODEL_PATH, GEMMA_PORT, args.gemma_gpu)
        if gemma_proc is None:
            log.error("Failed to start Gemma server")
            sys.exit(1)

        try:
            # Initialize FLUX pipeline
            log.info("Loading FLUX pipeline on GPU %d...", args.flux_gpu)
            import torch
            from diffusers import Flux2KleinPipeline

            flux_pipeline = Flux2KleinPipeline.from_pretrained(
                FLUX_MODEL_PATH, torch_dtype=torch.bfloat16
            )
            flux_pipeline.to(f"cuda:{args.flux_gpu}")
            log.info("FLUX pipeline loaded")

            # Initialize Gemma client
            from openai import OpenAI
            gemma_client = OpenAI(api_key="EMPTY", base_url=f"http://127.0.0.1:{GEMMA_PORT}/v1")
            # Get actual model name from server
            models = gemma_client.models.list()
            gemma_model_name = models.data[0].id if models.data else GEMMA_MODEL_PATH
            log.info("Gemma model name: %s", gemma_model_name)

            # Run generation for each prompt
            for prompt_id, prompt_text in prompts.items():
                log.info("=" * 60)
                log.info("GENERATING with prompt: %s", prompt_id)
                log.info("=" * 60)
                run_generation(prompt_id, prompt_text, samples, output_root,
                              flux_pipeline, gemma_client, gemma_model_name)

            # Cleanup FLUX
            del flux_pipeline
            torch.cuda.empty_cache()
            log.info("FLUX pipeline unloaded")

        finally:
            stop_server(gemma_proc)
            log.info("Gemma server stopped")

    if args.stage in ("evaluate", "all"):
        # Start Qwen vLLM server
        kill_nogpualarm()
        time.sleep(3)

        qwen_proc = start_vllm_server(
            QWEN_MODEL_PATH, QWEN_PORT, args.qwen_gpu,
            extra_args=["--gpu-memory-utilization", "0.90", "--max-model-len", "8192",
                        "--limit-mm-per-prompt", "image=3"],
        )
        if qwen_proc is None:
            log.error("Failed to start Qwen server")
            sys.exit(1)

        try:
            from openai import OpenAI
            qwen_client = OpenAI(api_key="EMPTY", base_url=f"http://127.0.0.1:{QWEN_PORT}/v1")
            models = qwen_client.models.list()
            qwen_model_name = models.data[0].id if models.data else QWEN_MODEL_PATH
            log.info("Qwen model name: %s", qwen_model_name)

            all_summaries = []
            for prompt_id in prompts:
                log.info("=" * 60)
                log.info("EVALUATING prompt: %s", prompt_id)
                log.info("=" * 60)
                summary = run_evaluation(prompt_id, output_root, qwen_client, qwen_model_name)
                all_summaries.append(summary)

            # Final combined summary
            final = {
                "run_timestamp": datetime.now().isoformat(),
                "dataset": DATASET_ROOT,
                "total_samples": len(samples),
                "prompts_evaluated": list(prompts.keys()),
                "results": all_summaries,
            }
            with open(os.path.join(output_root, "final_summary.json"), "w") as f:
                json.dump(final, f, indent=2)

            log.info("=" * 60)
            log.info("FINAL RESULTS")
            log.info("=" * 60)
            for s in all_summaries:
                log.info("  %s: pass_rate=%s (%d yes / %d total)",
                         s["prompt_id"], s["pass_rate_pct"], s["yes"], s["total_evaluated"])

        finally:
            stop_server(qwen_proc)
            log.info("Qwen server stopped")

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
