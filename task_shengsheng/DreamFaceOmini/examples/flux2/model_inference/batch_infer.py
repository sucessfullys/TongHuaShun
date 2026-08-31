import json
from json import JSONDecodeError
import random
import os
import argparse
import multiprocessing as mp
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
import torch
from PIL import Image

# ── 参数配置 ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--jsonl",    default="/mnt/data/image-edit/datasets/shensheng/datasets/merged_all.jsonl")
parser.add_argument("--output",   default="./batch_results_1")
parser.add_argument("--num",      type=int, default=20,    help="随机抽取条数")
parser.add_argument("--seed",     type=int, default=42,    help="随机种子（固定抽样）")
parser.add_argument("--steps",    type=int, default=28)
parser.add_argument("--cfg",      type=float, default=1.0)
parser.add_argument("--epoch",    default="epoch-9", help="模型 epoch 名，如 epoch-0 / epoch-9")
parser.add_argument("--lora",     default=None, help="可选 LoRA 权重路径（.safetensors）")
parser.add_argument("--lora_alpha", type=float, default=1.0, help="LoRA alpha 缩放系数")
parser.add_argument("--gpus",     default="0,1", help='使用的 GPU 编号，例 "0,1,2"；默认自动使用全部可见 GPU')
parser.add_argument("--offload",  action="store_true", help="启用 CPU offload 以降低单卡显存占用")
parser.add_argument("--height",   type=int, default=None, help="图片高度")
parser.add_argument("--width",    type=int, default=None, help="图片宽度")
args = parser.parse_args()

TRAIN_MODEL_DIR = "/mnt/data/image-edit/datasets/shensheng/code/dev/DiffSynth-Studio/models/train/FLUX.2-klein-base-9B_full"

os.makedirs(args.output, exist_ok=True)

def load_entries_compat(path):
    """兼容 JSON 数组（[...]）和 JSONL（每行一个对象）两种格式。"""
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return [data]
            raise ValueError(f"Unsupported JSON root type: {type(data)}")
        except JSONDecodeError:
            pass

    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {line_no}: {e}") from e
    return entries


def resize_with_padding(img, target_w, target_h, bg_color=(255, 255, 255)):
    """等比缩放后居中贴到 target_w x target_h 的白底上"""
    img.thumbnail((target_w, target_h), Image.LANCZOS)
    padded = Image.new("RGB", (target_w, target_h), bg_color)
    offset_x = (target_w - img.width) // 2
    offset_y = (target_h - img.height) // 2
    padded.paste(img, (offset_x, offset_y))
    return padded


def parse_gpu_ids(gpu_arg):
    if gpu_arg is not None:
        return [int(gpu_id.strip()) for gpu_id in gpu_arg.split(",") if gpu_id.strip()]
    return list(range(torch.cuda.device_count()))


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


def build_pipeline(device):
    extra = _offload_config(device) if args.offload else {}
    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id="black-forest-labs/FLUX.2-klein-base-9B", origin_file_pattern="text_encoder/*.safetensors", **extra),
            # ModelConfig(path=os.path.join(TRAIN_MODEL_DIR, f"{args.epoch}.safetensors")),
            # ModelConfig(model_id="black-forest-labs/FLUX.2-klein-base-9B", origin_file_pattern="transformer/*.safetensors", **extra),
            ModelConfig(model_id="black-forest-labs/FLUX.2-klein-9B", origin_file_pattern="transformer/*.safetensors", **extra),
            ModelConfig(model_id="black-forest-labs/FLUX.2-klein-base-9B", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        tokenizer_config=ModelConfig(model_id="black-forest-labs/FLUX.2-klein-base-9B", origin_file_pattern="tokenizer/"),
    )
    if args.lora:
        print(f"-------------------------------Loading LoRA: {args.lora} (alpha={args.lora_alpha})--------------------------------")
        pipe.load_lora(pipe.dit, args.lora, alpha=args.lora_alpha)
    return pipe


def run_inference(entry, pipe, device, worker_idx, worker_total):
    uid = entry.get("uid", "")
    prompt = entry["prompt"]
    input_paths = entry["edit_image"]
    gt_path = entry.get("image","")
    height = args.height
    width = args.width

    print(f"\n[worker {worker_idx + 1}/{worker_total}] uid={uid}")
    print(f"  prompt: {prompt[:80]}...")

    out_dir = os.path.join(args.output, uid)
    result_path = os.path.join(out_dir, "result.png")
    if os.path.exists(result_path):
        print(f"[worker {worker_idx + 1}/{worker_total}] uid={uid} already exists, skipped.")
        return

    try:
        images = [Image.open(p).convert("RGB") for p in input_paths]
        if height is None:
            height = images[0].height
        if width is None:
            width = images[0].width
        result = pipe(
            prompt,
            # prompt+'authentic, high quality, high resolution, clean composition, rich details, low ISO, pristine quality, front-facing, subject in the center of the frame',
            negative_prompt='',
            edit_image=images,
            edit_image_scale=1,
            s2_scale=0.0,       # 开启 S²-Guidance，强度 0.25
            s2_drop_ratio=0.3,   # 丢弃 ~10% 的 single stream blocks（约 5 个）
            s2_start=0.1,        # 在去噪 10%~90% 区间内激活
            s2_end=0.9,
            seed=args.seed,
            rand_device=device,
            num_inference_steps=args.steps,
            cfg_scale=args.cfg,
            height=height,
            width=width,
        )

        out_dir = os.path.join(args.output, uid)
        os.makedirs(out_dir, exist_ok=True)

        for idx, img in enumerate(images):
            img.save(os.path.join(out_dir, f"input_{idx}.webp"))
        result.save(os.path.join(out_dir, "result.png"))

        compare_images = images + [result]
        if gt_path and os.path.exists(gt_path):
            gt_img = Image.open(gt_path).convert("RGB")
            gt_img.save(os.path.join(out_dir, "gt.webp"))
            compare_images.append(gt_img)
        else:
            print(f"  [WARN] uid={uid}: gt image not found, skipped: {gt_path}")

        # 横拼对比图：展示所有输入图，并保持长宽比，空白区域补白边
        cell_w = max(img.width for img in compare_images)
        cell_h = max(img.height for img in compare_images)
        compare = Image.new("RGB", (cell_w * len(compare_images), cell_h), (255, 255, 255))
        for idx, img in enumerate(compare_images):
            compare.paste(
                resize_with_padding(img.copy(), cell_w, cell_h),
                (cell_w * idx, 0)
            )
        compare.save(os.path.join(out_dir, "compare.webp"))
        print(f"  saved -> {out_dir}")
    except Exception as e:
        print(f"  [ERROR] uid={uid}: {e}")


def worker_main(gpu_id, worker_idx, worker_total, worker_samples):
    device = f"cuda:{gpu_id}"
    print(f"[worker {worker_idx + 1}/{worker_total}] loading pipeline on {device}, samples={len(worker_samples)}")
    pipe = build_pipeline(device)
    for entry in worker_samples:
        run_inference(entry, pipe, device, worker_idx, worker_total)


def chunk_samples(samples, num_chunks):
    chunks = [[] for _ in range(num_chunks)]
    for idx, entry in enumerate(samples):
        chunks[idx % num_chunks].append(entry)
    return chunks


def main():
    # ── 随机抽样 ──────────────────────────────────────────────────────────────
    entries = load_entries_compat(args.jsonl)

    random.seed(args.seed)
    samples = random.sample(entries, min(args.num, len(entries)))
    print(f"Total entries: {len(entries)}, sampled: {len(samples)}")
    gpu_ids = parse_gpu_ids(args.gpus)
    if not gpu_ids:
        raise RuntimeError("No GPU available. Please check CUDA visibility or pass --gpus.")

    sample_chunks = [chunk for chunk in chunk_samples(samples, len(gpu_ids)) if chunk]
    active_gpu_ids = gpu_ids[:len(sample_chunks)]

    print(f"Using GPUs: {active_gpu_ids}")

    if len(active_gpu_ids) == 1:
        worker_main(active_gpu_ids[0], 0, 1, sample_chunks[0])
    else:
        mp_ctx = mp.get_context("spawn")
        processes = []
        for worker_idx, (gpu_id, worker_samples) in enumerate(zip(active_gpu_ids, sample_chunks)):
            process = mp_ctx.Process(
                target=worker_main,
                args=(gpu_id, worker_idx, len(active_gpu_ids), worker_samples),
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(f"Worker process failed with exit code {process.exitcode}")

    print(f"\nDone. Results saved to {args.output}")


if __name__ == "__main__":
    main()