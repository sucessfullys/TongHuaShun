#!/usr/bin/env python3
"""
FLUX.2-Klein-9B-True-V2 + DreamFace LoRA batch i2i inference script.

1. Loads base FLUX.2-klein-9B pipeline (VAE / tokenizer / scheduler config).
2. Converts True-V2 single-file safetensors → diffusers format via official
   convert_flux2_transformer_checkpoint_to_diffusers() utility.
3. Overlays converted weights into the transformer.
4. Loads & fuses DreamFace LoRA in memory.
5. Enables CPU offload (saves VRAM, official recommended).

No files under base model / True-V2 checkpoint / LoRA are modified.

Task: home_img (reference face) + replaced_prompt (editing instruction)
      → output image (True-V2 + DreamFace stylized edit)

Official recommended params (True-V2, non-distilled):
  - cfg (guidance_scale): 1.0
  - inference steps:      20-30
  - edit / inpainting:    10-25

Usage (defaults):
    python infer_flux2_True_V2_lora.py

Usage (custom):
    python infer_flux2_True_V2_lora.py \\
        --base_model /path/to/FLUX.2-klein-9B \\
        --ckpt_path /path/to/Flux2-Klein-9B-True-v2-bf16.safetensors \\
        --lora_path /path/to/diffusers_lora.safetensors \\
        --csv_path /path/to/csv-fluxout.csv \\
        --output_dir /path/to/output \\
        --reference_image_column home_img \\
        --prompt_column replaced_prompt \\
        --num_inference_steps 20 \\
        --guidance_scale 1.0 \\
        --height 1024 --width 1024 \\
        --max_samples 5 \\
        --seed 0
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

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
# Original FLUX.2-klein-9B diffusers dir (provides config / tokenizer / VAE / scheduler)
DEFAULT_BASE_MODEL = "/mnt/data/image-edit/datasets/shensheng/models/black-forest-labs/FLUX.2-klein-9B"

# True-V2 single-file fine-tuned checkpoint
DEFAULT_CKPT_PATH = "/mnt/data/image-edit/models/wikeeyang/Flux2-Klein-9B-True-V2/Flux2-Klein-9B-True-v2-bf16.safetensors"

# DreamFace LoRA
DEFAULT_LORA_PATH = "/mnt/data/image-edit/datasets/shensheng/models/hithink-image-labs/DreamFace_lora/v2.1/diffusers_lora.safetensors"

DEFAULT_CSV_PATH = "/mnt/data/image-edit/datasets/shensheng/code/stable/DreamFaceOmini/exp_out/csv-fluxout-1k/csv-fluxout.csv"
DEFAULT_OUTPUT_DIR = "/mnt/image-edit/datasets/duanyufa/task_shengsheng/output/lora_flux_9B_True_V2_1440x1080_28steps"
DEFAULT_REF_IMAGE_DIR = "/mnt/image-edit/datasets/duanyufa/task_shengsheng/Ref_image"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _detect_prompt_column(columns: list[str]) -> str | None:
    candidates = [
        "replaced_prompt",
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
    for col in columns:
        low = col.lower()
        if any(kw in low for kw in ["prompt", "text", "caption", "desc", "input"]):
            return col
    return None


def _safe_mkdir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _load_image_from_url(url: str) -> Image.Image | None:
    if _HAS_REQUESTS:
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception:
            pass
    try:
        from urllib.request import urlopen
        with urlopen(url, timeout=30) as resp:  # type: ignore[no-any]
            return Image.open(BytesIO(resp.read())).convert("RGB")
    except Exception:
        return None


def _resolve_reference_image(
    row: dict,
    ref_image_dir: str | None,
    reference_image_column: str,
) -> Image.Image | None:
    row_id = row.get("id", "").strip()

    # 1) Local directory
    if ref_image_dir and row_id:
        ref_dir = Path(ref_image_dir)
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            candidate = ref_dir / f"{row_id}{ext}"
            if candidate.is_file():
                try:
                    return Image.open(candidate).convert("RGB")
                except Exception:
                    continue

    # 2) Fallback: URL / path from CSV column
    src = row.get(reference_image_column, "").strip()
    if not src:
        return None
    parsed = urlparse(src)
    if not parsed.scheme:
        path = Path(src)
        if path.is_file():
            try:
                return Image.open(path).convert("RGB")
            except Exception:
                return None
        return None
    return _load_image_from_url(src)


def _build_image_filename(row_index: int, row: dict, output_dir: str) -> str:
    row_id = row.get("id", None)
    file_name = row.get("file_name", None)
    if row_id:
        base = str(row_id)
    elif file_name:
        base = str(file_name)
    else:
        base = f"{row_index:06d}"
    if not base.lower().endswith(".png"):
        base = base + ".png"
    return os.path.join(output_dir, base)


# ---------------------------------------------------------------------------
# Pipeline loading  (base model → True-V2 weights → DreamFace LoRA)
# ---------------------------------------------------------------------------
def load_pipeline(
    base_model: str,
    ckpt_path: str,
    lora_path: str,
    lora_scale: float,
    device: str,
    dtype: torch.dtype,
):
    """
    Load pipeline:
      1. Load base FLUX.2-klein-9B for pipeline structure (config/VAE/tokenizer)
      2. Convert True-V2 single-file ckpt → diffusers format via official utility
      3. Overlay converted weights into transformer
      4. Load + fuse DreamFace LoRA
    """
    from diffusers import Flux2KleinPipeline

    # ---- Stage 1: Load base pipeline (provides VAE / tokenizer / scheduler config) ----
    print(f"[1/5] Loading base pipeline from {base_model} ...")
    pipe = Flux2KleinPipeline.from_pretrained(base_model, torch_dtype=dtype)
    print("       Base pipeline loaded.")

    # ---- Stage 2: Convert True-V2 ckpt → diffusers format ----
    print(f"[2/5] Loading & converting True-V2 checkpoint from {ckpt_path} ...")
    from safetensors.torch import load_file
    from diffusers.loaders.single_file_utils import (
        convert_flux2_transformer_checkpoint_to_diffusers,
    )
    ckpt_sd = load_file(ckpt_path)
    converted_sd = convert_flux2_transformer_checkpoint_to_diffusers(ckpt_sd)
    print(f"       Converted {len(converted_sd)} keys to diffusers format.")

    # ---- Stage 3: Load converted weights into transformer ----
    print(f"[3/6] Loading converted True-V2 weights into transformer ...")
    transformer_sd = pipe.transformer.state_dict()
    model_total = len(transformer_sd)
    matched, missing = _load_matched_weights(converted_sd, transformer_sd)
    pipe.transformer.load_state_dict(transformer_sd, strict=False)
    print(f"       matched={matched}/{len(converted_sd)} converted, "
          f"missing={missing}/{model_total} model keys")
    if missing > 0:
        missing_keys = sorted(set(transformer_sd.keys()) - set(converted_sd.keys()))
        print(f"       First 5 missing model keys: {missing_keys[:5]}")
    if matched == 0:
        raise RuntimeError(
            "Zero keys matched after conversion. "
            "First 5 converted keys: " + str(list(converted_sd.keys())[:5]) + "\n"
            "First 5 model keys: " + str(list(transformer_sd.keys())[:5])
        )

    # ---- Stage 4: Load + fuse DreamFace LoRA (BEFORE CPU offload) ----
    if lora_path and lora_path.lower() != "none":
        print(f"[4/6] Loading & fusing DreamFace LoRA "
              f"(scale={lora_scale}) from {lora_path} ...")
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora(lora_scale=lora_scale)
        print("       LoRA loaded & fused in memory.")
    else:
        print(f"[4/6] LoRA disabled (--lora_path=none), skipping.")

    # ---- Stage 5: Verify pipeline I2I API ----
    print(f"[5/6] Verifying pipeline image-to-image API ...")
    _verify_pipeline_i2i_api(pipe)
    print("       OK.")

    pipe = pipe.to(device)
    # ---- Stage 6: Enable CPU offload ----
    # print(f"[6/6] Enabling model CPU offload ...")
    # pipe.enable_model_cpu_offload()
    # print("       Done.")

    return pipe


def _load_matched_weights(
    src_sd: dict[str, torch.Tensor],
    dst_sd: dict[str, torch.Tensor],
) -> tuple[int, int]:
    """Copy weights from src_sd into dst_sd where keys & shapes match."""
    dst_keys = set(dst_sd.keys())
    matched = 0
    for k, v in src_sd.items():
        if k in dst_keys and dst_sd[k].shape == v.shape:
            dst_sd[k] = v.clone()
            matched += 1
    missing = len(dst_keys - set(src_sd.keys()))
    return matched, missing


def _verify_pipeline_i2i_api(pipe) -> None:
    """
    Check that the pipeline's __call__ actually accepts an 'image' parameter.
    Prints a warning if it doesn't (which would mean the ref image is silently
    ignored and the model runs as text-to-image).
    """
    import inspect
    sig = inspect.signature(pipe.__call__)
    params = list(sig.parameters.keys())
    if "image" not in params and "images" not in params:
        print(f"       ⚠ WARNING: pipeline __call__ does NOT accept 'image' or "
              f"'images' parameter!")
        print(f"       Accepted params: {params}")
        print(f"       The reference image may be silently ignored → pure T2I.")
    else:
        # Check which one is used
        for p in ("image", "images"):
            if p in params:
                print(f"       Pipeline accepts '{p}' parameter for I2I.")


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------
def read_csv_rows(csv_path: str, max_samples: int | None) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    columns: list[str] = []
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

        if not prompt or not prompt.strip():
            result_entry["error"] = "Empty or missing prompt"
            results.append(result_entry)
            tqdm.write(f"  [{idx}] SKIP: empty prompt")
            continue

        ref_image = _resolve_reference_image(row, ref_image_dir, reference_image_column)
        if ref_image is None:
            result_entry["error"] = (
                f"Failed to load reference image "
                f"(id={row.get('id', '?')}, src={ref_src[:80] if ref_src else 'N/A'})"
            )
            results.append(result_entry)
            tqdm.write(f"  [{idx}] FAILED: cannot load reference image")
            continue

        try:
            row_seed = seed + idx if seed is not None else None
            row_generator = (
                torch.Generator(device=pipe.device).manual_seed(row_seed)
                if row_seed is not None
                else None
            )

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
        description="FLUX.2-Klein-9B-True-V2 + DreamFace LoRA batch i2i inference"
    )
    # Paths
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL,
                        help="Original FLUX.2-klein-9B diffusers dir (config/Vocab/VAE)")
    parser.add_argument("--ckpt_path", default=DEFAULT_CKPT_PATH,
                        help="True-V2 single-file safetensors checkpoint")
    parser.add_argument("--lora_path", default=DEFAULT_LORA_PATH,
                        help="DreamFace LoRA safetensors path. "
                             "Use 'none' to disable LoRA loading.")
    parser.add_argument("--lora_scale", type=float, default=1.0,
                        help="LoRA weight scale (default 1.0). "
                             "Try 0.3-0.6 if LoRA causes quality issues "
                             "on fine-tuned base models.")
    parser.add_argument("--csv_path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt_column", default=None,
                        help="CSV column for prompts (auto-detected)")
    parser.add_argument("--reference_image_column", default="home_img",
                        help="CSV column for reference images")
    parser.add_argument("--ref_image_dir", default=DEFAULT_REF_IMAGE_DIR,
                        help="Local dir with pre-downloaded ref images ({id}.png)")

    # Generation params
    parser.add_argument("--num_inference_steps", type=int, default=28,
                        help="Denoising steps (default 30; "
                             "official: 20-30 inference, 10-25 editing)")
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                        help="Guidance scale (default 1.0 for flow-matching)")
    parser.add_argument("--height", type=int, default=1440)
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed (per-sample = base + row_index)")

    # Execution
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit rows (for testing)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16",
                        help="bfloat16 / float16 / float32")
    parser.add_argument("--skip_metadata", action="store_true")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

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
    print("=" * 72)
    print("  FLUX.2-Klein-9B-True-V2 + DreamFace LoRA — i2i Inference")
    print("=" * 72)
    print(f"  Base model:       {args.base_model}")
    print(f"  True-V2 ckpt:     {args.ckpt_path}")
    print(f"  LoRA path:        {args.lora_path}")
    print(f"  LoRA scale:        {args.lora_scale}")
    print(f"  CSV path:         {args.csv_path}")
    print(f"  Output dir:       {args.output_dir}")
    print(f"  Prompt column:    {'auto-detect' if args.prompt_column is None else args.prompt_column}")
    print(f"  Reference column: {args.reference_image_column}")
    print(f"  Ref image dir:    {args.ref_image_dir}")
    print(f"  Image size:       {args.height}×{args.width}")
    print(f"  Inference steps:  {args.num_inference_steps}")
    print(f"  Guidance scale:   {args.guidance_scale}")
    print(f"  Seed (base):      {args.seed}")
    print(f"  Max samples:      {args.max_samples if args.max_samples else 'all'}")
    print(f"  Device:           {args.device}")
    print(f"  Dtype:            {args.dtype}")
    print("=" * 72)

    # ------------------------------------------------------------------
    print("\nReading CSV ...")
    rows, columns = read_csv_rows(args.csv_path, args.max_samples)
    print(f"  Columns: {columns}")
    print(f"  Rows:    {len(rows)}")
    if not rows:
        print("ERROR: No rows to process.")
        sys.exit(1)

    prompt_column = args.prompt_column or _detect_prompt_column(columns)
    if prompt_column is None or prompt_column not in columns:
        print(f"ERROR: Cannot find prompt column in {columns}")
        print("       Use --prompt_column to specify explicitly.")
        sys.exit(1)
    print(f"  Prompt column: '{prompt_column}'")
    first_prompt = rows[0].get(prompt_column, "")
    print(f"  First prompt:  {first_prompt[:120]}...")

    # ------------------------------------------------------------------
    print("\nLoading pipeline: base model → True-V2 weights → DreamFace LoRA ...")
    pipe = load_pipeline(
        base_model=args.base_model,
        ckpt_path=args.ckpt_path,
        lora_path=args.lora_path,
        lora_scale=args.lora_scale,
        device=args.device,
        dtype=dtype,
    )
    print("Pipeline ready.\n")

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
    ok = sum(1 for r in results if r["success"])
    failed = len(results) - ok
    print(f"\n{'=' * 72}")
    print(f"  Done in {elapsed:.1f}s ({elapsed/max(len(results), 1):.1f}s/sample)")
    print(f"  Success: {ok} / {len(results)}")
    if failed:
        print(f"  Failed:  {failed} / {len(results)}")
    print(f"{'=' * 72}")

    if not args.skip_metadata:
        save_metadata(results, args.output_dir)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
