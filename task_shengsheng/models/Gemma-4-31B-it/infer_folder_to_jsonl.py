#!/usr/bin/env python3
"""Run Gemma-4-31B-it on every image in a folder and save raw outputs to JSONL."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "/mnt/image-edit/datasets/duanyufa/models/gemma-4-31B-it"
DEFAULT_OUTPUT_DIR = (
    "/mnt/image-edit/datasets/duanyufa/task_shengsheng/models/Gemma-4-31B-it/output"
)
DEFAULT_SYSTEM_PROMPT = (
    "你是人体结构异常检测助手。请判断图中是否存在多手异常，并给出简洁、可见的判断理由。一定要中文回答"
    "最终只能用 <conclusion>normal</conclusion>、<conclusion>abnormal</conclusion> "
    "或 <conclusion>non_human</conclusion> 输出结论。"
)
DEFAULT_USER_PROMPT = "请根据图片判断是否有人体多手异常，并输出理由(一定要中文回答)和结论<conclusion>normal</conclusion>、<conclusion>abnormal</conclusion>或 <conclusion>non_human</conclusion>。"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemma-4 model directory.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder containing images.")
    parser.add_argument("--output-jsonl", type=Path, default=None, help="JSONL output path.")
    parser.add_argument("--recursive", action="store_true", help="Read images recursively.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32", "auto"])
    parser.add_argument("--attn-impl", default=None)
    parser.add_argument("--device", default=None, help="CUDA_VISIBLE_DEVICES value, e.g. 0 or 2,3.")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--user-prompt", default=DEFAULT_USER_PROMPT)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Gemma4 thinking mode. Default: false.",
    )
    parser.add_argument(
        "--strip-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save only the final answer when Gemma4 emits thinking/channel text. Default: true.",
    )
    return parser.parse_args()


def iter_images(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    images = [
        p
        for p in input_dir.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return sorted(images)


def write_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def strip_gemma_thinking(text: str) -> str:
    """Remove Gemma4 channel/thinking text when a final answer is available."""
    text = (text or "").strip()
    if not text:
        return text

    final_markers = [
        "<|channel>final\n",
        "<|channel>final",
        "<|channel|>final<|message|>",
    ]
    for marker in final_markers:
        if marker in text:
            text = text.split(marker, 1)[-1].strip()
            break

    # Non-thinking Gemma4 often returns: "<|channel>thought\n<channel|>answer".
    if text.startswith("<|channel>thought") and "<channel|>" in text:
        text = text.split("<channel|>")[-1].strip()

    if text.startswith("<|channel>thought"):
        return ""

    for marker in ["<|channel>thought\n", "<|channel>thought", "<channel|>", "<turn|>"]:
        text = text.replace(marker, "")
    return text.strip()


def main() -> None:
    args = parse_args()
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input dir not found: {input_dir}")
    model_path = Path(args.model).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"model dir not found: {model_path}")

    images = iter_images(input_dir, args.recursive)
    if args.max_samples is not None:
        images = images[: args.max_samples]
    if not images:
        raise ValueError(f"No images found under {input_dir}")

    output_jsonl = args.output_jsonl
    if output_jsonl is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_jsonl = Path(DEFAULT_OUTPUT_DIR) / f"gemma4_infer_{input_dir.name}_{timestamp}.jsonl"
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if output_jsonl.exists():
        output_jsonl.unlink()
    output_jsonl.touch()
    print(f"[output] {output_jsonl}", flush=True)
    print(f"[data] found {len(images)} images under {input_dir}", flush=True)

    import torch
    from swift import InferRequest, RequestConfig, TransformersEngine

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "auto": None,
    }
    engine = TransformersEngine(
        str(model_path),
        model_type="gemma4",
        max_batch_size=args.batch_size,
        torch_dtype=dtype_map[args.torch_dtype],
        attn_impl=args.attn_impl,
    )
    request_config_kwargs = {"max_tokens": args.max_new_tokens}
    if args.temperature is not None:
        request_config_kwargs["temperature"] = args.temperature
    if args.top_p is not None:
        request_config_kwargs["top_p"] = args.top_p
    if args.top_k is not None:
        request_config_kwargs["top_k"] = args.top_k
    request_config = RequestConfig(**request_config_kwargs)

    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": args.user_prompt},
    ]
    total = len(images)
    for start in range(0, total, args.batch_size):
        batch = images[start : start + args.batch_size]
        print(f"[infer] {start + 1}-{start + len(batch)}/{total}", flush=True)
        t0 = time.time()
        requests = [
            InferRequest(
                messages=messages,
                images=[str(image_path)],
                chat_template_kwargs={"enable_thinking": args.enable_thinking},
            )
            for image_path in batch
        ]
        responses = engine.infer(requests, request_config)
        elapsed = time.time() - t0
        for image_path, response in zip(batch, responses):
            caption = response.choices[0].message.content
            if args.strip_thinking:
                caption = strip_gemma_thinking(caption)
            write_jsonl_row(
                output_jsonl,
                {
                    "id": image_path.stem,
                    "image": str(image_path),
                    "system_prompt": args.system_prompt,
                    "user_prompt": args.user_prompt,
                    "caption": caption,
                },
            )
        print(f"[done] {start + 1}-{start + len(batch)}/{total} elapsed={elapsed:.1f}s", flush=True)

    print(f"Saved {total} rows to {output_jsonl}", flush=True)


if __name__ == "__main__":
    main()
