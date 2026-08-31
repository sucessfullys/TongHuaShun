#!/usr/bin/env python3
"""
FLUX.2-klein-9B + DreamFace LoRA batch image-to-image inference script.

Loads the FLUX.2-klein-9B base model, merges the DreamFace LoRA weights in
memory (without modifying the base model directory). Reads reference images
and text prompts from a CSV file, and generates edited images for each row.

Task: home_img (reference face) + replaced_prompt (editing instruction)
      → output image (DreamFace-stylized edit)

Usage (defaults):
    python infer_flux2_lora.py

Usage (custom):
    python infer_flux2_lora.py \
        --base_model /path/to/FLUX.2-klein-9B \
        --lora_path /path/to/diffusers_lora.safetensors \
        --csv_path /path/to/csv-fluxout.csv \
        --output_dir /path/to/output \
        --reference_image_column home_img \
        --prompt_column replaced_prompt \
        --num_inference_steps 4 \
        --guidance_scale 1.0 \
        --height 1024 --width 1024 \
        --max_samples 5 \
        --seed 42
"""

import argparse
import csv
import json
import os
import sys
import time
import traceback
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import torch
from PIL import Image
from tqdm import tqdm

# Optional: accelerate image downloads
try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


# ---------------------------------------------------------------------------
# Default paths (matching the project directory layout)
# ---------------------------------------------------------------------------
DEFAULT_BASE_MODEL = "/mnt/data/image-edit/datasets/shensheng/models/black-forest-labs/FLUX.2-klein-9B"
DEFAULT_LORA_PATH = "/mnt/data/image-edit/datasets/shensheng/models/hithink-image-labs/DreamFace_lora/v2.1/diffusers_lora.safetensors"
DEFAULT_CSV_PATH = "/mnt/data/image-edit/datasets/shensheng/code/stable/DreamFaceOmini/exp_out/csv-fluxout-1k/csv-fluxout.csv"
DEFAULT_OUTPUT_DIR = "/mnt/image-edit/datasets/duanyufa/task_shengsheng/output/lora_flux_9B_1440x1080"
DEFAULT_REF_IMAGE_DIR = "/mnt/image-edit/datasets/duanyufa/task_shengsheng/Ref_image"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _detect_prompt_column(columns: list[str]) -> str:
    """
    Heuristic: find the most likely prompt column from CSV headers.
    Priority: exact match named columns → substring matches.
    """
    candidates = [
        "replaced_prompt",   # <-- this is what the target CSV uses
        "prompt",
        "text",
        "caption",
        "description",
        "input",
        "instruction",
    ]
    lowered = {c.lower().strip(): c for c in columns}

    for cand in candidates:
        if cand in columns:
            return cand
        if cand in lowered:
            return lowered[cand]

    # Fallback: first column whose name suggests it contains long text
    for col in columns:
        low = col.lower()
        if any(kw in low for kw in ["prompt", "text", "caption", "desc", "input"]):
            return col

    return None


def _safe_mkdir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _load_reference_image(src: str) -> Image.Image | None:
    """
    Load a reference image from a URL or local file path.
    Returns a PIL Image in RGB mode, or None on failure.
    """
    if not src or not src.strip():
        return None

    src = src.strip()

    # --- Local file path ---
    if not urlparse(src).scheme or urlparse(src).scheme == "":
        path = Path(src)
        if path.is_file():
            try:
                return Image.open(path).convert("RGB")
            except Exception:
                return None
        return None

    # --- Remote URL ---
    if _HAS_REQUESTS:
        try:
            resp = requests.get(src, timeout=30)
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception:
            return None

    # Fallback: try urllib
    try:
        from urllib.request import urlopen
        with urlopen(src, timeout=30) as resp:  # type: ignore[no-any]
            return Image.open(BytesIO(resp.read())).convert("RGB")
    except Exception:
        return None


def _resolve_reference_image(
    row: dict,
    ref_image_dir: str | None,
    reference_image_column: str,
) -> Image.Image | None:
    """
    Resolve a reference image for a CSV row.

    Priority:
    1. If ref_image_dir is set, look for '{id}.png' (or .jpg/.webp) in that dir
    2. Otherwise, use the URL/path from `reference_image_column`
    """
    row_id = row.get("id", "").strip()

    # --- Try local directory first (fastest, most reliable) ---
    if ref_image_dir and row_id:
        ref_dir = Path(ref_image_dir)
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            candidate = ref_dir / f"{row_id}{ext}"
            if candidate.is_file():
                try:
                    return Image.open(candidate).convert("RGB")
                except Exception:
                    continue

    # --- Fall back to CSV column (URL or path) ---
    src = row.get(reference_image_column, "").strip()
    if src:
        return _load_reference_image(src)

    return None


def _build_image_filename(row_index: int, row: dict, output_dir: str) -> str:
    """
    Build a stable output filename.
    Priority: 'id' field → 'file_name' field → zero-padded index.
    """
    row_id = row.get("id", None)
    file_name = row.get("file_name", None)

    if row_id:
        base = str(row_id)
    elif file_name:
        base = str(file_name)
    else:
        base = f"{row_index:06d}"

    # Ensure .png extension
    if not base.lower().endswith(".png"):
        base = base + ".png"

    return os.path.join(output_dir, base)


# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------
def load_pipeline(
    base_model: str,
    lora_path: str,
    device: str,
    dtype: torch.dtype,
) -> tuple:
    """
    Load the FLUX.2-klein-9B pipeline and apply LoRA weights in memory.

    Returns (pipe, pipeline_name_str).
    """
    from diffusers import Flux2KleinPipeline

    print(f"[1/4] Loading Flux2KleinPipeline from {base_model} ...")
    pipe = Flux2KleinPipeline.from_pretrained(
        base_model,
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)
    print("       Pipeline loaded.")

    # Enable CPU offload (official recommended, saves VRAM)
    # print(f"[2/4] Enabling model CPU offload ...")
    # # pipe.enable_model_cpu_offload()
    # print("       Done.")

    # Load LoRA (in memory — does NOT modify base model files)
    print(f"[3/4] Loading DreamFace LoRA from {lora_path} ...")
    pipe.load_lora_weights(lora_path)
    print("       LoRA weights loaded in memory.")

    # Optionally fuse LoRA for faster inference
    print(f"[4/4] Fusing LoRA into model weights (in-memory) ...")
    pipe.fuse_lora()
    print("       LoRA fused.")

    return pipe


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------
def read_csv_rows(csv_path: str, max_samples: int | None) -> tuple[list[dict], list[str]]:
    """
    Read CSV and return (rows, columns).
    rows are dicts keyed by CSV header.
    """
    # Try UTF-8 first, fallback to UTF-8-BOM / utf-8-sig
    rows = []
    columns = []

    for encoding in ["utf-8-sig", "utf-8", "gbk", "gb2312", "latin-1"]:
        try:
            with open(csv_path, "r", encoding=encoding) as f:
                reader = csv.DictReader(f)
                columns = reader.fieldnames or []
                for row in reader:
                    rows.append(row)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    else:
        raise ValueError(f"Failed to decode CSV file: {csv_path}")

    if max_samples is not None and max_samples > 0:
        rows = rows[:max_samples]

    return rows, columns


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference(
    pipe,
    rows: list[dict],
    prompt_column: str,
    reference_image_column: str,
    ref_image_dir: str | None,
    output_dir: str,
    seed: int | None,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
) -> list[dict]:
    """
    Iterate over rows, load reference image + prompt, run i2i inference,
    save images, return results list.
    """
    _safe_mkdir(output_dir)
    results: list[dict] = []

    for idx, row in enumerate(tqdm(rows, desc="Generating", unit="sample")):
        prompt = row.get(prompt_column, "")
        ref_src = row.get(reference_image_column, "")

        result_entry = {
            "row_index": idx,
            "csv_id": row.get("id", None),
            "reference_image": ref_src,
            "prompt": prompt,
            "output_path": None,
            "seed": seed,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
            "inference_time_s": None,
            "success": False,
            "error": None,
        }

        # --- Validate inputs ---
        if not prompt or not prompt.strip():
            result_entry["error"] = "Empty or missing prompt"
            results.append(result_entry)
            tqdm.write(f"  [{idx}] SKIP: empty prompt")
            continue

        # --- Load reference image (local dir first, then URL fallback) ---
        ref_image = _resolve_reference_image(row, ref_image_dir, reference_image_column)
        if ref_image is None:
            result_entry["error"] = f"Failed to load reference image: {ref_src}"
            results.append(result_entry)
            tqdm.write(f"  [{idx}] FAILED: cannot load reference image (id={row.get('id', '?')})")
            continue

        try:
            row_seed = seed + idx if seed is not None else None
            row_generator = (
                torch.Generator(device=pipe.device).manual_seed(row_seed)
                if row_seed is not None
                else None
            )

            # image-to-image: pass reference image via `image` parameter
            t0 = time.time()
            image = pipe(
                prompt=prompt,
                image=ref_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=row_generator,
            ).images[0]
            elapsed = time.time() - t0

            output_path = _build_image_filename(idx, row, output_dir)
            image.save(output_path)

            result_entry["output_path"] = output_path
            result_entry["seed"] = row_seed
            result_entry["inference_time_s"] = round(elapsed, 2)
            result_entry["success"] = True

        except Exception as exc:
            result_entry["error"] = f"{type(exc).__name__}: {exc}"
            result_entry["traceback"] = traceback.format_exc()
            tqdm.write(f"  [{idx}] FAILED: {result_entry['error']}")

        results.append(result_entry)

    return results


# ---------------------------------------------------------------------------
# Save metadata
# ---------------------------------------------------------------------------
def save_metadata(results: list[dict], output_dir: str) -> str:
    _safe_mkdir(output_dir)
    jsonl_path = os.path.join(output_dir, "results.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Also save a readable summary
    summary_path = os.path.join(output_dir, "results_summary.txt")
    total = len(results)
    ok = sum(1 for r in results if r["success"])
    failed = total - ok

    # Per-image timing
    success_times = [r["inference_time_s"] for r in results
                     if r["success"] and r.get("inference_time_s") is not None]
    avg_time = sum(success_times) / len(success_times) if success_times else 0

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Total samples:  {total}\n")
        f.write(f"Successful:     {ok}\n")
        f.write(f"Failed:         {failed}\n")
        f.write(f"Average time:   {avg_time:.2f}s/sample\n")
        f.write(f"\nPer-image inference times:\n")
        for r in results:
            t_str = f"{r['inference_time_s']:.2f}s" if r.get("inference_time_s") else "N/A"
            status = "OK" if r["success"] else "FAIL"
            f.write(f"  [{r['row_index']}] {status} | {t_str} | {r.get('csv_id', '?')}\n")
        if failed:
            f.write("\nFailures:\n")
            for r in results:
                if not r["success"]:
                    f.write(f"  [{r['row_index']}] {r['error']}\n")

    print(f"\nMetadata saved to: {jsonl_path}")
    print(f"Summary  saved to: {summary_path}")
    return jsonl_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="FLUX.2-klein-9B + DreamFace LoRA batch CSV inference"
    )
    # Paths
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--lora_path", default=DEFAULT_LORA_PATH)
    parser.add_argument("--csv_path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt_column", default=None,
                        help="CSV column name containing prompts. "
                             "Auto-detected if not specified.")
    parser.add_argument("--reference_image_column", default="home_img",
                        help="CSV column name containing reference image URLs "
                             "or local paths (default: home_img)")
    parser.add_argument("--ref_image_dir", default=DEFAULT_REF_IMAGE_DIR,
                        help="Local directory with pre-downloaded reference "
                             "images named {id}.png. If set, images are "
                             "loaded from here instead of downloading URLs.")

    # Generation params
    parser.add_argument("--num_inference_steps", type=int, default=4,
                        help="Number of denoising steps (default 4; "
                             "FLUX.2-klein is step-distilled to 4 steps)")
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                        help="Guidance scale (default 1.0; "
                             "FLUX.2-klein is flow-matching, no CFG needed)")
    parser.add_argument("--height", type=int, default=1440)
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed (per-sample seed = base + row_index)")

    # Execution
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of CSV rows to process (for testing)")
    parser.add_argument("--device", default="cuda",
                        help="Device string: cuda / cpu")
    parser.add_argument("--dtype", default="bfloat16",
                        help="Torch dtype: bfloat16 / float16 / float32")
    parser.add_argument("--skip_metadata", action="store_true",
                        help="Do not save results metadata")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Resolve dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map.get(args.dtype, torch.bfloat16)

    # ------------------------------------------------------------------
    # Print key information
    # ------------------------------------------------------------------
    print("=" * 72)
    print("  FLUX.2-klein-9B + DreamFace LoRA — Batch CSV Inference")
    print("=" * 72)
    print(f"  Base model:      {args.base_model}")
    print(f"  LoRA path:       {args.lora_path}")
    print(f"  CSV path:        {args.csv_path}")
    print(f"  Output dir:      {args.output_dir}")
    print(f"  Prompt column:    {'auto-detect' if args.prompt_column is None else args.prompt_column}")
    print(f"  Reference column: {args.reference_image_column}")
    print(f"  Ref image dir:    {args.ref_image_dir}")
    print(f"  Image size:      {args.height}×{args.width}")
    print(f"  Inference steps: {args.num_inference_steps}")
    print(f"  Guidance scale:  {args.guidance_scale}")
    print(f"  Seed (base):     {args.seed}")
    print(f"  Max samples:     {args.max_samples if args.max_samples else 'all'}")
    print(f"  Device:          {args.device}")
    print(f"  Dtype:           {args.dtype}")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Read CSV
    # ------------------------------------------------------------------
    print("\nReading CSV ...")
    rows, columns = read_csv_rows(args.csv_path, args.max_samples)
    print(f"  Columns: {columns}")
    print(f"  Rows loaded: {len(rows)}")

    if not rows:
        print("ERROR: No rows to process. Exiting.")
        sys.exit(1)

    # Resolve prompt column
    if args.prompt_column is not None:
        prompt_column = args.prompt_column
    else:
        prompt_column = _detect_prompt_column(columns)
    if prompt_column is None or prompt_column not in columns:
        print(f"ERROR: Cannot find a prompt column in {columns}")
        print("       Use --prompt_column to specify the column name explicitly.")
        sys.exit(1)
    print(f"  Using prompt column: '{prompt_column}'")

    # Preview first prompt
    first = rows[0].get(prompt_column, "")
    print(f"  First prompt preview: {first[:120]}...")

    # ------------------------------------------------------------------
    # Load pipeline
    # ------------------------------------------------------------------
    print("\nLoading pipeline + LoRA ...")
    pipe = load_pipeline(
        base_model=args.base_model,
        lora_path=args.lora_path,
        device=args.device,
        dtype=dtype,
    )
    print("Pipeline ready.\n")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    t_start = time.time()
    results = run_inference(
        pipe=pipe,
        rows=rows,
        prompt_column=prompt_column,
        reference_image_column=args.reference_image_column,
        ref_image_dir=args.ref_image_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
    )
    elapsed = time.time() - t_start

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    ok = sum(1 for r in results if r["success"])
    failed = len(results) - ok
    print(f"\n{'=' * 72}")
    print(f"  Done in {elapsed:.1f}s ({elapsed/len(results):.1f}s/sample)")
    print(f"  Success: {ok} / {len(results)}")
    if failed:
        print(f"  Failed:  {failed} / {len(results)}")
    print(f"{'=' * 72}")

    # Metadata
    if not args.skip_metadata:
        save_metadata(results, args.output_dir)

    # Exit code
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
