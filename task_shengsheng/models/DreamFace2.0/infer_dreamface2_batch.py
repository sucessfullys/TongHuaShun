#!/usr/bin/env python3
import argparse
import csv
import json
import time
from pathlib import Path

import torch
from PIL import Image
from diffusers import Flux2KleinPipeline


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_rows(path: Path):
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid jsonl: {exc}") from exc
    return rows


def normalize_ref(value):
    if value is None:
        return None
    if isinstance(value, list):
        value = value[0] if value else None
    if not value:
        return None
    p = Path(str(value))
    if not p.exists():
        raise FileNotFoundError(f"reference image not found: {p}")
    return p


def get_first(row, keys, default=None):
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def get_seed(row, default_seed):
    value = get_first(row, ["seed", "random_seed", "generation_seed"], default_seed)
    if value in (None, ""):
        return None
    seed = int(value)
    return seed if seed >= 0 else None


def safe_name(text, fallback):
    keep = []
    for ch in str(text):
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    name = "".join(keep).strip("_")
    return name[:120] or fallback


def build_rows_from_dir(input_dir: Path, prompt: str):
    rows = []
    for idx, path in enumerate(sorted(input_dir.iterdir()), 1):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            rows.append(
                {
                    "id": f"image_{idx:04d}_{path.stem}",
                    "prompt": prompt,
                    "reference_image": str(path),
                }
            )
    return rows


def run_pipe(pipe, prompt, reference, height, width, steps, cfg, seed):
    generator = None
    if seed is not None and seed >= 0:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    kwargs = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "guidance_scale": cfg,
        "num_inference_steps": steps,
        "generator": generator,
    }
    if reference is not None:
        kwargs["image"] = Image.open(reference).convert("RGB")

    return pipe(**kwargs).images[0]


def main():
    parser = argparse.ArgumentParser(
        description="Batch inference for merged DreamFace2.0 Diffusers model."
    )
    parser.add_argument(
        "--model-path",
        default="/mnt/image-edit/datasets/duanyufa/task_shengsheng/models/DreamFace2.0",
        help="DreamFace2.0 model directory. It should already contain merged weights.",
    )
    parser.add_argument("--input-jsonl", default=None, help="JSONL input with prompt/reference image fields.")
    parser.add_argument("--input-csv", default=None, help="CSV input with prompt/reference image fields.")
    parser.add_argument("--input-dir", default=None, help="Optional folder of reference images.")
    parser.add_argument("--prompt", default=None, help="Prompt used for --input-dir or single-image inference.")
    parser.add_argument("--reference-image", default=None, help="Optional single reference image.")
    parser.add_argument(
        "--output-dir",
        default="/mnt/image-edit/datasets/duanyufa/task_shengsheng/models/DreamFace2.0/output",
    )
    parser.add_argument("--height", type=int, default=1152)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--cpu-offload", action="store_true", help="Use CPU offload instead of pipe.to('cuda').")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_jsonl:
        rows = load_rows(Path(args.input_jsonl))
    elif args.input_csv:
        rows = load_rows(Path(args.input_csv))
    elif args.input_dir:
        if not args.prompt:
            raise SystemExit("--input-dir requires --prompt")
        rows = build_rows_from_dir(Path(args.input_dir), args.prompt)
    else:
        if not args.prompt:
            raise SystemExit("Provide --input-jsonl, --input-csv, --input-dir + --prompt, or --prompt.")
        rows = [{"id": "single", "prompt": args.prompt, "reference_image": args.reference_image}]

    if args.limit is not None:
        rows = rows[: args.limit]

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    print(f"[load] model_path={model_path}")
    pipe = Flux2KleinPipeline.from_pretrained(model_path, torch_dtype=dtype_map[args.torch_dtype])
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    results_path = output_dir / "results.jsonl"
    print(f"[data] rows={len(rows)}")
    print(f"[output] {output_dir}")

    with results_path.open("a", encoding="utf-8") as fout:
        for idx, row in enumerate(rows, 1):
            prompt = get_first(row, ["prompt", "caption", "text"])
            if not prompt:
                raise ValueError(f"row {idx}: missing prompt/caption/text")

            ref_value = get_first(row, ["reference_image", "ref_image", "image", "edit_image", "input_image"])
            reference = normalize_ref(ref_value)
            row_id = get_first(row, ["id", "uid", "name"], f"sample_{idx:04d}")
            row_seed = get_seed(row, args.seed)
            item_dir = output_dir / safe_name(row_id, f"sample_{idx:04d}")
            item_dir.mkdir(parents=True, exist_ok=True)
            out_image = item_dir / "result.png"

            print(f"[infer] {idx}/{len(rows)} id={row_id} seed={row_seed} ref={reference if reference else '<none>'}")
            start = time.time()
            if args.skip_existing and out_image.exists():
                status = "skipped"
            else:
                image = run_pipe(
                    pipe=pipe,
                    prompt=prompt,
                    reference=reference,
                    height=args.height,
                    width=args.width,
                    steps=args.steps,
                    cfg=args.cfg,
                    seed=row_seed,
                )
                image.save(out_image)
                status = "ok"

            record = {
                "id": row_id,
                "prompt": prompt,
                "reference_image": str(reference) if reference else None,
                "output_image": str(out_image),
                "status": status,
                "elapsed_sec": round(time.time() - start, 3),
                "height": args.height,
                "width": args.width,
                "steps": args.steps,
                "cfg": args.cfg,
                "seed": row_seed,
                "model_path": str(model_path),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"[done] results={results_path}")


if __name__ == "__main__":
    main()
