"""Batch image editing from CSV: home_img + replaced_prompt -> flux result CSV."""

import argparse
import csv
import hashlib
import multiprocessing as mp
import os
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import torch
from PIL import Image

from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig

DEFAULT_CSV = (
    "/mnt/data/image-edit/datasets/shensheng/datasets/benchmark/filter/"
    "filter-intersection-06-04.csv"
)
DEFAULT_LORA = (
    "/mnt/data/image-edit/datasets/shensheng/models/hithink-image-labs/"
    "DreamFace_lora/v2.1/diffsynth_lora.safetensors"
)
DEFAULT_OUTPUT_ROOT = (
    "/mnt/data/image-edit/datasets/shensheng/code/stable/DreamFaceOmini/exp_out/csv-fluxout"
)
DEFAULT_TRANSFORMER_V2 = (
    "/mnt/data/image-edit/datasets/shensheng/models/wikeeyang/"
    "Flux2-Klein-9B-True-V2/Flux2-Klein-9B-True-v2-bf16.safetensors"
)

parser = argparse.ArgumentParser(description="Batch FLUX.2-klein-9B inference from CSV")
parser.add_argument("--csv", default=DEFAULT_CSV, help="Input CSV path")
parser.add_argument("--output", default=DEFAULT_OUTPUT_ROOT, help="Output root directory")
parser.add_argument("--lora", default=DEFAULT_LORA, help="DiffSynth LoRA .safetensors path")
parser.add_argument(
    "--transformer",
    default=DEFAULT_TRANSFORMER_V2,
    help="Flux2 DiT single-file .safetensors or HF model id",
)
parser.add_argument("--lora_alpha", type=float, default=1.0)
parser.add_argument("--steps", type=int, default=4)
parser.add_argument("--cfg", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--height", type=int, default=1152)
parser.add_argument("--width", type=int, default=896)
parser.add_argument("--gpus", default="0", help='GPU ids, e.g. "0,1,2,3"')
parser.add_argument("--offload", action="store_true")
parser.add_argument("--num", type=int, default=None, help="Max rows to process (default: all)")
parser.add_argument("--skip", type=int, default=0, help="Skip first N rows")
parser.add_argument("--force", action="store_true", help="Re-run even if result exists")
args = parser.parse_args()

OUTPUT_CSV_NAME = "csv-fluxout.csv"
INPUT_FIELDS = ["id", "replaced_prompt", "home_img", "test_result_imgs"]
OUTPUT_FIELDS = INPUT_FIELDS + ["flux_result", "status", "error"]


def load_csv_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return [{k: (row.get(k) or "").strip() for k in INPUT_FIELDS} for row in reader]


def is_url(value):
    parsed = urlparse(value)
    return parsed.scheme in ("http", "https")


def cache_path_for(value, cache_dir, uid):
    if is_url(value):
        ext = Path(urlparse(value).path).suffix or ".png"
        return cache_dir / f"{uid}{ext}"
    return Path(value)


def load_image(value, cache_dir, uid):
    cache_dir.mkdir(parents=True, exist_ok=True)
    if is_url(value):
        local_path = cache_path_for(value, cache_dir, uid)
        if not local_path.exists():
            req = Request(value, headers={"User-Agent": "DreamFaceOmini/1.0"})
            with urlopen(req, timeout=60) as resp:
                data = resp.read()
            local_path.write_bytes(data)
        return Image.open(local_path).convert("RGB"), str(local_path)
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {value}")
    return Image.open(path).convert("RGB"), str(path)


def parse_gpu_ids(gpu_arg):
    return [int(g.strip()) for g in gpu_arg.split(",") if g.strip()]


def _offload_config(device):
    return dict(
        offload_dtype=torch.bfloat16,
        offload_device="cpu",
        onload_dtype=torch.bfloat16,
        onload_device=device,
        preparing_dtype=torch.bfloat16,
        preparing_device=device,
        computation_dtype=torch.bfloat16,
        computation_device=device,
    )


def transformer_model_config(extra):
    transformer = args.transformer.strip()
    if os.path.isfile(transformer):
        return ModelConfig(path=transformer, **extra)
    return ModelConfig(
        model_id=transformer,
        origin_file_pattern="transformer/*.safetensors",
        **extra,
    )


def build_pipeline(device):
    extra = _offload_config(device) if args.offload else {}
    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(
                model_id="black-forest-labs/FLUX.2-klein-base-9B",
                origin_file_pattern="text_encoder/*.safetensors",
                **extra,
            ),
            transformer_model_config(extra),
            ModelConfig(
                model_id="black-forest-labs/FLUX.2-klein-base-9B",
                origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
            ),
        ],
        tokenizer_config=ModelConfig(
            model_id="black-forest-labs/FLUX.2-klein-base-9B",
            origin_file_pattern="tokenizer/",
        ),
    )
    if args.lora:
        print(f"Loading LoRA: {args.lora} (alpha={args.lora_alpha})")
        pipe.load_lora(pipe.dit, args.lora, alpha=args.lora_alpha)
    return pipe


def run_one(row, pipe, device, output_root):
    uid = row["id"] or hashlib.md5(row["home_img"].encode()).hexdigest()[:12]
    result_path = output_root / "outputs" / uid / "result.webp"
    input_cache_dir = output_root / "inputs"

    if result_path.exists() and not args.force:
        return {
            **row,
            "flux_result": str(result_path),
            "status": "skipped",
            "error": "",
        }

    try:
        image, input_local = load_image(row["home_img"], input_cache_dir, uid)
        result = pipe(
            row["replaced_prompt"],
            negative_prompt="",
            edit_image=[image],
            edit_image_scale=1,
            s2_scale=0.0,
            s2_drop_ratio=0.3,
            s2_start=0.1,
            s2_end=0.9,
            seed=args.seed,
            rand_device=device,
            num_inference_steps=args.steps,
            cfg_scale=args.cfg,
            height=args.height,
            width=args.width,
        )

        out_dir = output_root / "outputs" / uid
        out_dir.mkdir(parents=True, exist_ok=True)
        image.save(out_dir / "input.webp", format="WEBP", quality=90)
        result.save(result_path, format="WEBP", quality=90)

        return {
            **row,
            "flux_result": str(result_path),
            "status": "ok",
            "error": "",
        }
    except Exception as exc:
        return {
            **row,
            "flux_result": "",
            "status": "error",
            "error": str(exc),
        }


def worker_main(gpu_id, worker_idx, worker_total, rows, output_root, result_queue):
    device = f"cuda:{gpu_id}"
    print(f"[worker {worker_idx + 1}/{worker_total}] loading pipeline on {device}, rows={len(rows)}")
    pipe = build_pipeline(device)
    for idx, row in enumerate(rows, start=1):
        uid = row.get("id", "")
        print(f"[worker {worker_idx + 1}/{worker_total}] ({idx}/{len(rows)}) id={uid}")
        result = run_one(row, pipe, device, Path(output_root))
        result_queue.put(result)
        print(f"  -> {result['status']}: {result.get('flux_result') or result.get('error')}")


def chunk_rows(rows, num_chunks):
    chunks = [[] for _ in range(num_chunks)]
    for idx, row in enumerate(rows):
        chunks[idx % num_chunks].append(row)
    return chunks


def write_output_csv(rows, output_root):
    out_csv = Path(output_root) / OUTPUT_CSV_NAME
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Output CSV written: {out_csv}")
    return out_csv


def main():
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows = load_csv_rows(args.csv)
    if args.skip:
        all_rows = all_rows[args.skip :]
    if args.num is not None:
        all_rows = all_rows[: args.num]

    print(f"Input CSV: {args.csv}")
    print(f"Rows to process: {len(all_rows)}")
    print(f"Output root: {output_root}")

    gpu_ids = parse_gpu_ids(args.gpus)
    if not gpu_ids:
        raise RuntimeError("No GPU configured. Pass --gpus.")

    if len(gpu_ids) == 1:
        device = f"cuda:{gpu_ids[0]}"
        pipe = build_pipeline(device)
        results = []
        for idx, row in enumerate(all_rows, start=1):
            uid = row.get("id", "")
            print(f"[{idx}/{len(all_rows)}] id={uid}")
            result = run_one(row, pipe, device, output_root)
            results.append(result)
            print(f"  -> {result['status']}: {result.get('flux_result') or result.get('error')}")
    else:
        row_chunks = [chunk for chunk in chunk_rows(all_rows, len(gpu_ids)) if chunk]
        active_gpu_ids = gpu_ids[: len(row_chunks)]
        mp_ctx = mp.get_context("spawn")
        result_queue = mp_ctx.Queue()
        processes = []
        for worker_idx, (gpu_id, worker_rows) in enumerate(zip(active_gpu_ids, row_chunks)):
            process = mp_ctx.Process(
                target=worker_main,
                args=(gpu_id, worker_idx, len(active_gpu_ids), worker_rows, str(output_root), result_queue),
            )
            process.start()
            processes.append(process)

        results = []
        expected = len(all_rows)
        while len(results) < expected:
            results.append(result_queue.get())

        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(f"Worker failed with exit code {process.exitcode}")

        id_order = {row["id"]: i for i, row in enumerate(all_rows)}
        results.sort(key=lambda r: id_order.get(r["id"], 10**9))

    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = sum(1 for r in results if r["status"] == "error")
    print(f"\nDone: ok={ok}, skipped={skipped}, error={failed}")
    write_output_csv(results, output_root)


if __name__ == "__main__":
    main()
