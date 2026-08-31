#!/usr/bin/env python3
"""Single-image inference for the body deformity reward model."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


DEFAULT_PROJECT_DIR = Path("/mnt/image-edit/datasets/duanyufa/task_shengsheng/models/ms-swift")
DEFAULT_MODEL_PATH = Path("/mnt/image-edit/datasets/duanyufa/models/Qwen3-VL-8B-Instruct")
DEFAULT_ADAPTER_PATH = Path(
    "/mnt/image-edit/datasets/duanyufa/交接/多模态畸形检测reward model/checkpoint-3450"
)

SYSTEM_PROMPT = (
    "你是人体结构异常检测助手。请判断图中是否存在多手、多腿、肢体分叉或异常连接等人体肢体异常，"
    "并给出简洁、可见的判断理由。最终只能用 <conclusion>normal</conclusion>、"
    "<conclusion>abnormal</conclusion> 或 <conclusion>non_human</conclusion> 输出结论。"
)

USER_PROMPT = "<image>请判断画面中是否有多手、多腿、肢体分叉或异常连接现象，并给出理由和结论。"

CONCLUSION_RE = re.compile(
    r"<conclusion>\s*(normal|abnormal|non_human)\s*</conclusion>", re.I
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-image Qwen3-VL LoRA inference for body deformity detection."
    )
    parser.add_argument("--image", type=Path, required=True, help="Input image path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. If omitted, only prints to stdout.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--device", default="0", help="CUDA device id, for example 0.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--image-max-token-num", default="2048")
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32", "auto"],
    )
    parser.add_argument("--attn-impl", default=None)
    return parser.parse_args()


def require_path(path: Path, kind: str) -> None:
    if kind == "file" and not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    if kind == "dir" and not path.is_dir():
        raise FileNotFoundError(f"Missing directory: {path}")


def parse_conclusion(text: str) -> str | None:
    match = CONCLUSION_RE.search(text)
    return match.group(1).lower() if match else None


def main() -> None:
    args = parse_args()
    require_path(args.image, "file")
    require_path(args.model, "dir")
    require_path(args.adapter, "dir")
    require_path(args.project_dir, "dir")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.device))
    os.environ.setdefault("IMAGE_MAX_TOKEN_NUM", str(args.image_max_token_num))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    sys.path.insert(0, str(args.project_dir))

    import torch
    from swift import InferRequest, RequestConfig, TransformersEngine

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "auto": None,
    }

    engine = TransformersEngine(
        str(args.model),
        model_type="qwen3_vl",
        max_batch_size=1,
        torch_dtype=dtype_map[args.torch_dtype],
        attn_impl=args.attn_impl,
        adapters=[str(args.adapter)],
    )

    request = InferRequest(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        images=[str(args.image.resolve())],
    )
    config = RequestConfig(max_tokens=args.max_new_tokens, temperature=args.temperature)
    response = engine.infer([request], config)[0].choices[0].message.content

    result = {
        "image": str(args.image.resolve()),
        "model": str(args.model),
        "adapter": str(args.adapter),
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": USER_PROMPT,
        "response": response,
        "conclusion": parse_conclusion(response),
    }

    print(response)
    print()
    print(json.dumps({"conclusion": result["conclusion"]}, ensure_ascii=False))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
