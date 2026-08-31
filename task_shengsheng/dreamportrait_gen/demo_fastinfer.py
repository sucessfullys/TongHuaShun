#!/usr/bin/env python3
"""DreamFace2.0 纯推理脚本 —— 无服务包裹，仅 diffusers 推理。

用法:
    # text-to-image
    python demo_fastinfer.py --prompt "a portrait photo"

    # image-to-image（单张或多张参考图）
    python demo_fastinfer.py --prompt "high quality portrait" --images test_imgs/model.png

    # 指定输出路径、分辨率、步数等
    python demo_fastinfer.py \
        --prompt "cinematic portrait" \
        --images ref1.png ref2.png \
        --height 1024 --width 768 \
        --steps 8 --seed 123 \
        --output result.png

    # 双卡均衡分布
    python demo_fastinfer.py --prompt "portrait" --device-map balanced \
        --max-memory "0:23GiB,1:23GiB,cpu:60GiB"
"""

import argparse
import math
import os
import time

import torch
from diffusers import Flux2KleinPipeline
from PIL import Image

REFERENCE_IMAGE_AREA = 1024 * 1024


def parse_max_memory(raw: str) -> dict[int | str, str] | None:
    if not raw:
        return None
    parsed = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        key, value = item.split(":", 1)
        key = key.strip()
        parsed[int(key) if key.isdigit() else key] = value.strip()
    return parsed or None


def prepare_reference_image(image: Image.Image) -> Image.Image:
    """将参考图裁切/缩放到 ~1M 像素并对齐 32 的倍数。"""
    image = image.convert("RGB")
    ratio = image.width / image.height
    tw = math.sqrt(REFERENCE_IMAGE_AREA * ratio)
    th = tw / ratio
    tw, th = round(tw / 32) * 32, round(th / 32) * 32

    scale = max(tw / image.width, th / image.height)
    image = image.resize(
        (round(image.width * scale), round(image.height * scale)),
        resample=Image.Resampling.BILINEAR,
    )
    left = max((image.width - tw) // 2, 0)
    top = max((image.height - th) // 2, 0)
    return image.crop((left, top, left + tw, top + th))


def load_pipeline(args) -> Flux2KleinPipeline:
    model_path = args.model
    if not os.path.exists(model_path):
        base = os.environ.get(
            "DIFFUSERS_MODEL_BASE_PATH",
            os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", ""),
        )
        if base:
            candidate = os.path.join(base, model_path)
            if os.path.exists(candidate):
                model_path = candidate

    print(f"[DreamFace] 加载模型: {model_path}")
    load_kwargs: dict = {"torch_dtype": torch.bfloat16}

    if args.device_map:
        load_kwargs["device_map"] = args.device_map
        mm = parse_max_memory(args.max_memory)
        if mm:
            load_kwargs["max_memory"] = mm

    pipe = Flux2KleinPipeline.from_pretrained(model_path, **load_kwargs)

    if args.device_map:
        print(f"[DreamFace] device_map={args.device_map}")
    elif args.cpu_offload:
        pipe.enable_model_cpu_offload()
        print("[DreamFace] CPU offload 已启用")
    else:
        pipe.to(args.device)
        print(f"[DreamFace] Pipeline -> {args.device}")

    if args.lora:
        lora_path = args.lora
        print(f"[DreamFace] 加载 LoRA: {lora_path} (alpha={args.lora_alpha})")
        try:
            pipe.load_lora_weights(lora_path, adapter_name="default")
        except TypeError:
            pipe.load_lora_weights(lora_path)
        if hasattr(pipe, "set_adapters"):
            pipe.set_adapters(["default"], adapter_weights=[args.lora_alpha])

    return pipe


@torch.inference_mode()
def run_inference(pipe: Flux2KleinPipeline, args) -> Image.Image:
    ref_images = None
    if args.images:
        ref_images = [prepare_reference_image(Image.open(p)) for p in args.images]
        print(f"[DreamFace] 参考图 x{len(ref_images)}")

    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    result = pipe(
        prompt=args.prompt,
        image=ref_images,
        height=args.height,
        width=args.width,
        guidance_scale=args.cfg,
        num_inference_steps=args.steps,
        generator=generator,
    )

    return result.images[0] if hasattr(result, "images") else result[0][0]


def main():
    parser = argparse.ArgumentParser(description="DreamFace2.0 快速推理")
    parser.add_argument("--prompt", type=str, default="Maintain absolute consistency of the subject's facial features. Medium shot portrait perspective, with the female subject occupying two-thirds of the frame. She is elegantly sitting sideways on a pink velvet stool, holding a strawberry near her lips, with a serene and sweet expression of joy. The scene is a pink-themed birthday party photoshoot, featuring a soft pink velvet backdrop with cascading folds, surrounded by matte white and sequin rose-gold birthday balloons. The entire scene is bathed in a warm, soft ambient glow. She is wearing a light pink strapless feathered short dress, accessorized with sparkling silver long earrings and a delicate silver bracelet. Photography style: shot on a Canon R5 with an 85mm f/1.4 prime lens, soft diffused studio lighting, highly detailed and realistic skin texture, high-fashion magazine cover quality.", help="生成提示词")
    parser.add_argument("--images", nargs="+", default=["/mnt/image-edit/datasets/duanyufa/task_shengsheng/DreamFaceOmini/exp_out/csv-fluxout/inputs/6a211a9a21f59e0d4bc62580.png"], help="参考图路径（可多张）")
    parser.add_argument("--output", type=str, default="output/result.png", help="输出图片路径")
    parser.add_argument("--model", type=str, default="/mnt/image-edit/models/hithink-image-labs/DreamFace2.0", help="模型 ID 或本地路径")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--cfg", type=float, default=1.0, help="guidance_scale")
    parser.add_argument("--height", type=int, default=1152)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--cpu-offload", action="store_true", help="启用 model CPU offload（24G 显存推荐）")
    parser.add_argument("--device-map", type=str, default="", help="Accelerate device_map（如 balanced / auto）")
    parser.add_argument("--max-memory", type=str, default="", help="max_memory 配置，如 0:22GiB,1:22GiB,cpu:60GiB")
    parser.add_argument("--lora", type=str, default="", help="LoRA 权重路径")
    parser.add_argument("--lora-alpha", type=float, default=1.0)
    args = parser.parse_args()

    pipe = load_pipeline(args)

    print(f"[DreamFace] prompt: {args.prompt}")
    print(f"[DreamFace] 分辨率: {args.width}x{args.height}, steps={args.steps}, cfg={args.cfg}, seed={args.seed}")

    t0 = time.time()
    image = run_inference(pipe, args)
    elapsed = time.time() - t0

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    image.save(args.output)
    print(f"[DreamFace] 完成! 耗时 {elapsed:.2f}s -> {args.output}")


if __name__ == "__main__":
    main()
