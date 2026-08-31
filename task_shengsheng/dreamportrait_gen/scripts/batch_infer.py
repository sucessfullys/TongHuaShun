#!/usr/bin/env python3
"""批量推理 —— 8 卡分片，跳过已生成，不重复。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from diffusers import Flux2KleinPipeline
from PIL import Image
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--prompts-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--meta-file", type=Path, required=True)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_prompts(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    prompts = [item["prompt"] for item in data]
    return prompts


def main():
    args = parse_args()

    accelerator = Accelerator()
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    device = accelerator.device

    # 加载模型（只加载一次）
    if accelerator.is_main_process:
        print(f"Loading model: {args.model_path}")
    pipe = Flux2KleinPipeline.from_pretrained(
        str(args.model_path), torch_dtype=torch.bfloat16
    )
    pipe.to(device)
    pipe.transformer.eval()

    # 加载 prompts
    all_prompts = load_prompts(args.prompts_file)
    total = len(all_prompts)
    if accelerator.is_main_process:
        print(f"Total prompts: {total}")

    # 分片：rank N 处理 N, N+8, N+16, ...
    shard = list(range(rank, total, world_size))

    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta_tmp = Path(str(args.meta_file) + f".rank{rank}.tmp")

    # 打开 meta 临时文件（追加模式）
    meta_fp = meta_tmp.open("a", encoding="utf-8")

    generator_base = torch.Generator(device="cpu").manual_seed(args.seed)

    generated = 0
    skipped = 0

    iterator = tqdm(shard, desc=f"rank{rank}", disable=not accelerator.is_main_process)
    for idx in iterator:
        img_name = f"img_{idx:08d}.png"
        img_path = args.output_dir / img_name

        # 跳过已生成
        if img_path.exists():
            skipped += 1
            continue

        prompt = all_prompts[idx]
        # 每张图独立 seed（基于 base_seed + index）
        g = torch.Generator(device="cpu").manual_seed(args.seed + idx)

        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                height=args.height,
                width=args.width,
                guidance_scale=args.cfg,
                num_inference_steps=args.steps,
                generator=g,
            )
        image: Image.Image = result.images[0]

        # 原子写入（先写 tmp 再 rename）
        tmp_path = img_path.with_suffix(".tmp")
        image.save(tmp_path, format="PNG")
        tmp_path.replace(img_path)

        # 写 meta
        meta_fp.write(json.dumps({
            "image": img_name,
            "prompt": prompt,
            "height": args.height,
            "width": args.width,
            "index": idx,
        }, ensure_ascii=False) + "\n")
        meta_fp.flush()
        generated += 1

    meta_fp.close()

    accelerator.wait_for_everyone()

    # 合并各 rank 的临时 meta 文件（仅 rank 0 执行）
    if accelerator.is_main_process:
        import shutil
        # 收集所有 rank 的 tmp 文件
        all_tmp = sorted(Path(args.meta_file).parent.glob(
            Path(args.meta_file).name + ".rank*.tmp"
        ))
        with open(args.meta_file, "w", encoding="utf-8") as out:
            for tmp in all_tmp:
                with open(tmp, "r", encoding="utf-8") as inf:
                    shutil.copyfileobj(inf, out)
                tmp.unlink()  # 删除临时文件
        # 排序 meta 保证 index 顺序
        lines = Path(args.meta_file).read_text(encoding="utf-8").strip().split("\n")
        lines.sort(key=lambda s: json.loads(s)["index"])
        Path(args.meta_file).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Meta merged: {len(lines)} records -> {args.meta_file}")

    print(f"rank={rank} generated={generated} skipped={skipped}")


if __name__ == "__main__":
    main()
