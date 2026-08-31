#!/usr/bin/env python3
"""批量 LRE（LR Encoding）策略推理——基于微调后的 FLUX.2 Deblur Template checkpoint。

与原版 ``infer_flux2_klein_base_4b_deblur_dataset.py`` 的区别：

  原版：去噪起点 = 纯随机噪声，从 t=T 开始完整去噪
  LRE： 去噪起点 = (1-σ) × LR_latent + σ × noise，从 t=T 开始完整去噪

关键：LRE 不是在去噪中途加入（那会截断步数），而是在 t=T 时刻把 LR 结构
混入初始噪声，然后跑完整 50 步去噪。管线通过 ``initial_noise`` 参数实现，
绕过 img2img 的 timestep 截断逻辑。

LRE 的好处：保留输入图像的结构信息（人脸 identity、布局、颜色），
模型只需专注于增强细节、去模糊、去噪，而不是重建整张图。

--lre-strength 控制噪声比例 σ：
  1.0 = 纯噪声（等价于原版）
  0.8 = 80%% 噪声 + 20%% LR 信号（推荐默认值，参考 DiT4SR）
  0.5 = 50%% 噪声 + 50%% LR（更强的结构保留，适合人脸）
  0.0 = 纯 LR latent（无增强效果）

参考：DiT4SR (ICCV 2025) LRE 策略
"""

from __future__ import annotations

import os

# 在导入任何模型库之前设置离线标志
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MODELSCOPE_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from accelerate import Accelerator
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from diffsynth.core import ModelConfig, load_state_dict
from diffsynth.diffusion.template import TemplatePipeline
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline


# 默认模型路径
DEFAULT_BASE_MODEL = Path("/mnt/image-edit/datasets/duanyufa/FLUX.2-klein-base-4B")
DEFAULT_TEMPLATE_MODEL = REPO_ROOT / "Template-KleinBase4B-Upscaler"
DEFAULT_METADATA = Path("/mnt/image-edit/datasets/duanyufa/Face/test/metadata.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "使用 LRE 策略批量增强图像。从带噪 LR latent 开始去噪，"
            "保留结构信息的同时增强细节。"
        )
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="微调后的 Template checkpoint，例如 epoch-0.safetensors。",
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="输出目录。默认为 CHECKPOINT_DIR/test_results/CHECKPOINT_STEM/lre-STRENGTH。",
    )
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--template-model", type=Path, default=DEFAULT_TEMPLATE_MODEL)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--embedded-guidance", type=float, default=4.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--lre-strength", type=float, default=0.8,
        help=(
            "LRE 噪声比例 σ。初始 latent = (1-σ) × LR_latent + σ × noise。"
            "1.0=纯噪声（等价原版 template 模式），"
            "0.8=推荐值（80%% 噪声 + 20%% LR），"
            "0.5=更强结构保留。"
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"缺少文件 {description}: {path}")


def validate_models(base: Path, template: Path) -> list[str]:
    """校验本地模型文件是否完整。"""
    text_encoder = sorted((base / "text_encoder").glob("*.safetensors"))
    if not text_encoder:
        raise FileNotFoundError(
            f"在 {base / 'text_encoder'} 中未找到 text encoder 权重"
        )
    require_file(
        base / "transformer" / "diffusion_pytorch_model.safetensors",
        "FLUX.2 transformer",
    )
    require_file(base / "vae" / "diffusion_pytorch_model.safetensors", "FLUX.2 VAE")
    if not (base / "tokenizer").is_dir():
        raise FileNotFoundError(f"缺少 tokenizer 目录: {base / 'tokenizer'}")
    require_file(template / "model.py", "Template model.py")
    require_file(template / "model.safetensors", "Template 基础权重")
    return [str(path) for path in text_encoder]


def load_records(metadata_path: Path, limit: int | None) -> list[dict]:
    """从 metadata.jsonl 加载推理记录。"""
    require_file(metadata_path, "测试 metadata")
    records = []
    with metadata_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                relative_path = Path(record["image"])
                template_inputs = record["template_inputs"]
                Path(template_inputs["image"])
                record["prompt"]
                template_inputs["prompt"]
            except (KeyError, TypeError) as error:
                raise ValueError(
                    f"metadata 第 {line_number} 行格式错误: {error}"
                ) from error
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(
                    f"metadata image 必须是相对路径，第 {line_number} 行: {relative_path}"
                )
            record["_relative_path"] = relative_path.as_posix()
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"{metadata_path} 中没有找到记录")
    return records


def deterministic_seed(base_seed: int, relative_path: str) -> int:
    """为每张图生成确定性种子，保证可复现。"""
    digest = hashlib.sha256(
        f"{base_seed}:{relative_path}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big")


def reflect_pad_to_multiple(image: Image.Image, multiple: int = 16) -> tuple[Image.Image, tuple[int, int]]:
    width, height = image.size
    pad_width = (-width) % multiple
    pad_height = (-height) % multiple
    if pad_width == 0 and pad_height == 0:
        return image, (width, height)
    array = np.asarray(image)
    mode = "reflect" if width > 1 and height > 1 else "edge"
    padded = np.pad(array, ((0, pad_height), (0, pad_width), (0, 0)), mode=mode)
    return Image.fromarray(padded), (width, height)


def load_models(
    accelerator: Accelerator,
    base: Path,
    template_dir: Path,
    checkpoint: Path,
    text_encoder_files: list[str],
):
    """加载 FLUX.2 pipeline 和微调后的 Template 权重。"""
    device = accelerator.device
    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device=device,
        model_configs=[
            ModelConfig(path=text_encoder_files),
            ModelConfig(
                path=str(base / "transformer" / "diffusion_pytorch_model.safetensors")
            ),
            ModelConfig(
                path=str(base / "vae" / "diffusion_pytorch_model.safetensors")
            ),
        ],
        tokenizer_config=ModelConfig(path=str(base / "tokenizer")),
    )
    template = TemplatePipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device=device,
        model_configs=[ModelConfig(path=str(template_dir))],
    )
    state_dict = load_state_dict(str(checkpoint), torch_dtype=torch.bfloat16)
    load_result = template.models[0].load_state_dict(state_dict, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Template checkpoint 不匹配: "
            f"missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}"
        )
    template.eval()
    pipe.dit.eval()
    return pipe, template


def resolve_output_dir(args: argparse.Namespace) -> Path:
    """确定输出目录。"""
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()
    return (
        args.checkpoint.expanduser().resolve().parent
        / "test_results"
        / args.checkpoint.stem
        / f"lre-{args.lre_strength:g}"
    )


@torch.inference_mode()
def make_lre_initial_noise(
    pipe: Flux2ImagePipeline,
    lr_image: Image.Image,
    lre_strength: float,
    seed: int,
) -> torch.Tensor:
    """构造 LRE 初始 latent：(1-σ) × VAE(LR) + σ × randn()。

    返回形状 (1, 128, H//16, W//16)，可直接作为 ``initial_noise`` 传入管线。
    管线内部的 ``Flux2Unit_NoiseInitializer`` 会将其重排为序列格式。
    """
    width, height = lr_image.size
    latent_h, latent_w = height // 16, width // 16

    # 预处理并 VAE 编码 LR 图像 → latent
    lr_tensor = pipe.preprocess_image(
        lr_image, torch_dtype=pipe.torch_dtype, device=pipe.device,
    )
    lr_latent = pipe.vae.encode(lr_tensor)  # (1, 128, H//16, W//16)

    # Generate deterministic noise on CPU, then move it to the pipeline device.
    # Some PyTorch versions reject using a CPU generator for CUDA randn directly.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    noise = torch.randn(
        (1, 128, latent_h, latent_w),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).to(device=pipe.device, dtype=pipe.torch_dtype)

    # LRE 混合
    lre_latent = (1.0 - lre_strength) * lr_latent + lre_strength * noise
    return lre_latent


@torch.inference_mode()
def infer_record(
    pipe: Flux2ImagePipeline,
    template: TemplatePipeline,
    record: dict,
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    """对单条 metadata 记录执行 LRE 推理。

    两条独立路径同时工作：
    1. Template 路径：LR 图像 → Template VAE → Template DiT → KV cache
       → 注入冻结的 Base DiT，指导去噪方向
    2. LRE 路径：手动构造 (1-σ) × VAE(LR) + σ × noise
       → 作为 initial_noise 传入，跑完整去噪轨迹（不截断步数）
    """
    lr_path = Path(record["template_inputs"]["image"]).expanduser().resolve()
    require_file(lr_path, "LR 测试图像")
    with Image.open(lr_path) as opened:
        source = opened.convert("RGB")
    lr_image, original_size = reflect_pad_to_multiple(source)
    width, height = lr_image.size

    prompt = record["prompt"]
    template_prompt = record["template_inputs"]["prompt"]
    file_seed = deterministic_seed(args.seed, record["_relative_path"])

    # 构造 LRE 初始 latent（在 t=T 时刻混入 LR 结构，不截断去噪步数）
    lre_noise = make_lre_initial_noise(pipe, lr_image, args.lre_strength, file_seed)

    image = template(
        pipe,
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        height=height, width=width,
        seed=file_seed,
        rand_device="cpu",
        cfg_scale=args.cfg_scale,
        embedded_guidance=args.embedded_guidance,
        num_inference_steps=args.num_inference_steps,
        # === LRE：预构造的带噪 LR latent，跑完整去噪 ===
        initial_noise=lre_noise,
        # === Template KV cache 条件注入 ===
        template_inputs=[{"image": lr_image, "prompt": template_prompt}],
        negative_template_inputs=[{"image": lr_image, "prompt": ""}],
        progress_bar_cmd=lambda values: values,
    )
    image = image.crop((0, 0, original_size[0], original_size[1]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    image.save(temporary_path, format="PNG")
    temporary_path.replace(output_path)


def main() -> None:
    args = parse_args()
    if not 0 < args.lre_strength <= 1:
        raise ValueError("--lre-strength 必须在 (0, 1] 范围内。")
    if args.num_inference_steps < 1:
        raise ValueError("--num-inference-steps 必须为正整数。")

    accelerator = Accelerator()
    if accelerator.device.type != "cuda":
        raise RuntimeError("FLUX.2 Deblur 推理需要 CUDA。")

    checkpoint = args.checkpoint.expanduser().resolve()
    metadata = args.metadata.expanduser().resolve()
    base = args.base_model.expanduser().resolve()
    template_dir = args.template_model.expanduser().resolve()
    require_file(checkpoint, "微调后的 Template checkpoint")
    text_encoder_files = validate_models(base, template_dir)
    records = load_records(metadata, args.limit)
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    if accelerator.is_main_process:
        print(f"Checkpoint: {checkpoint}")
        print(f"Metadata: {metadata}")
        print(f"记录数: {len(records)}")
        print(f"进程数: {accelerator.num_processes}")
        print(f"LRE strength: {args.lre_strength:.2f}")
        print(f"输出目录: {output_dir}")
    accelerator.wait_for_everyone()

    pipe, template = load_models(
        accelerator, base, template_dir, checkpoint, text_encoder_files,
    )
    accelerator.wait_for_everyone()

    shard = records[accelerator.process_index :: accelerator.num_processes]
    iterator = tqdm(
        shard,
        desc=f"rank {accelerator.process_index}",
        disable=not accelerator.is_main_process,
    )
    generated = 0
    skipped = 0
    for record in iterator:
        output_path = output_dir / record["_relative_path"]
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        infer_record(pipe, template, record, output_path, args)
        generated += 1

    print(
        f"rank={accelerator.process_index} 已生成={generated} 跳过={skipped}",
        flush=True,
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output_count = sum(
            1 for path in output_dir.rglob("*.png") if path.is_file()
        )
        print(f"推理完成。输出目录下 PNG 文件数: {output_count}")


if __name__ == "__main__":
    main()
