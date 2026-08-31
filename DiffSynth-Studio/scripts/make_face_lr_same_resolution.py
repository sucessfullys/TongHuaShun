#!/usr/bin/env python3
"""Generate spatially aligned LR images with Real-ESRGAN two-stage degradation.

This is an independent data-preparation script. It does not modify SeeSR's
original ``utils_data/make_paired_data.py``.

Content alignment guarantees:
* no crop, flip, rotation, affine transform, warp, or generative model;
* each LR image keeps the HR relative filename and exact pixel dimensions;
* only spatially centered filtering, resampling, noise, and JPEG degradation;
* deterministic per-file randomness, independent of processing order.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import types

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
SEESR_ROOT = REPO_ROOT.parent / "SeeSR"
DEFAULT_HR_DIR = REPO_ROOT.parent / "Face" / "HR"
DEFAULT_LR_DIR = REPO_ROOT.parent / "Face" / "LR"
SUPPORTED_SUFFIXES = {".png"}
RESIZE_MODES = ("area", "bilinear", "bicubic")
KERNEL_TYPES = (
    "iso",
    "aniso",
    "generalized_iso",
    "generalized_aniso",
    "plateau_iso",
    "plateau_aniso",
)
KERNEL_PROBS = (0.45, 0.25, 0.12, 0.03, 0.12, 0.03)
KERNEL_SIZES = tuple(range(7, 22, 2))


def load_python_module(name: str, path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Required SeeSR module is missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_seesr_degradation_modules():
    # SeeSR targets an older torchvision import path. Provide the compatible
    # symbol without changing any SeeSR source file.
    try:
        from torchvision.transforms.functional_tensor import rgb_to_grayscale  # type: ignore
    except ImportError:
        from torchvision.transforms.functional import rgb_to_grayscale

        compatibility = types.ModuleType("torchvision.transforms.functional_tensor")
        compatibility.rgb_to_grayscale = rgb_to_grayscale
        sys.modules[compatibility.__name__] = compatibility

    degradations = load_python_module(
        "seesr_realesrgan_degradations",
        SEESR_ROOT / "basicsr" / "data" / "degradations.py",
    )
    diffjpeg = load_python_module(
        "seesr_realesrgan_diffjpeg",
        SEESR_ROOT / "basicsr" / "utils" / "diffjpeg.py",
    )
    return degradations, diffjpeg.DiffJPEG


DEGRADATIONS, DiffJPEG = load_seesr_degradation_modules()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate same-resolution LR/HR pairs using SeeSR's Real-ESRGAN "
            "two-stage degradation while preserving spatial content alignment."
        )
    )
    parser.add_argument("--hr-dir", type=Path, default=DEFAULT_HR_DIR)
    parser.add_argument("--lr-dir", type=Path, default=DEFAULT_LR_DIR)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        help="Internal Real-ESRGAN degradation scale; output is resized back to HR size.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for degradation, e.g. cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite matching LR files. Unrelated files are never deleted.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Defaults to LR_DIR's parent/degradation_manifest.jsonl.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N files; intended for validation runs.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.hr_dir.is_dir():
        raise FileNotFoundError(f"HR directory does not exist: {args.hr_dir}")
    if args.hr_dir.resolve() == args.lr_dir.resolve():
        raise ValueError("HR and LR directories must be different.")
    if args.scale < 1:
        raise ValueError("--scale must be at least 1.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")


def list_images(root: Path, limit: int | None) -> list[Path]:
    images = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"No PNG images found under {root}")
    return images if limit is None else images[:limit]


def seed_for_file(base_seed: int, relative_path: Path) -> int:
    digest = hashlib.sha256(
        f"{base_seed}:{relative_path.as_posix()}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big")


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def interpolate(image: torch.Tensor, *, size=None, scale_factor=None, mode: str):
    kwargs = {"mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    return F.interpolate(image, size=size, scale_factor=scale_factor, **kwargs)


def filter2d(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    kernel_size = kernel.shape[-1]
    batch, channels, height, width = image.shape
    padded = F.pad(
        image,
        (kernel_size // 2,) * 4,
        mode="reflect",
    )
    weights = kernel.view(batch, 1, kernel_size, kernel_size)
    weights = weights.repeat(1, channels, 1, 1).view(
        batch * channels, 1, kernel_size, kernel_size
    )
    padded = padded.view(1, batch * channels, padded.shape[-2], padded.shape[-1])
    output = F.conv2d(padded, weights, groups=batch * channels)
    return output.view(batch, channels, height, width)


def choose_weighted(rng: random.Random, values, weights):
    return rng.choices(values, weights=weights, k=1)[0]


def make_blur_kernel(
    rng: random.Random,
    *,
    sigma_range: tuple[float, float],
    sinc_probability: float,
    device: torch.device,
):
    kernel_size = rng.choice(KERNEL_SIZES)
    if rng.random() < sinc_probability:
        omega_min = math.pi / 3 if kernel_size < 13 else math.pi / 5
        omega = rng.uniform(omega_min, math.pi)
        kernel = DEGRADATIONS.circular_lowpass_kernel(
            omega, kernel_size, pad_to=False
        )
        kernel_kind = "sinc"
    else:
        kernel_kind = choose_weighted(rng, KERNEL_TYPES, KERNEL_PROBS)
        kernel = DEGRADATIONS.random_mixed_kernels(
            [kernel_kind],
            [1.0],
            kernel_size,
            sigma_range,
            sigma_range,
            [-math.pi, math.pi],
            [0.5, 4.0],
            [1.0, 2.0],
            noise_range=None,
        )
    pad = (21 - kernel_size) // 2
    kernel = np.pad(kernel, ((pad, pad), (pad, pad)))
    tensor = torch.tensor(kernel, dtype=torch.float32, device=device).unsqueeze(0)
    return tensor, kernel_kind, kernel_size


def make_final_sinc_kernel(rng: random.Random, device: torch.device):
    if rng.random() < 0.8:
        kernel_size = rng.choice(KERNEL_SIZES)
        omega = rng.uniform(math.pi / 3, math.pi)
        kernel = DEGRADATIONS.circular_lowpass_kernel(omega, kernel_size, pad_to=21)
        kind = "sinc"
    else:
        kernel = np.zeros((21, 21), dtype=np.float32)
        kernel[10, 10] = 1.0
        kind = "pulse"
        kernel_size = 1
    tensor = torch.tensor(kernel, dtype=torch.float32, device=device).unsqueeze(0)
    return tensor, kind, kernel_size


def add_noise(
    image: torch.Tensor,
    rng: random.Random,
    *,
    gaussian_probability: float,
    gaussian_range: tuple[float, float],
    poisson_range: tuple[float, float],
):
    gray = rng.random() < 0.4
    if rng.random() < gaussian_probability:
        sigma = rng.uniform(*gaussian_range)
        output = DEGRADATIONS.random_add_gaussian_noise_pt(
            image,
            sigma_range=(sigma, sigma),
            gray_prob=float(gray),
            clip=True,
            rounds=False,
        )
        return output, "gaussian_gray" if gray else "gaussian_color", sigma
    scale = rng.uniform(*poisson_range)
    output = DEGRADATIONS.random_add_poisson_noise_pt(
        image,
        scale_range=(scale, scale),
        gray_prob=float(gray),
        clip=True,
        rounds=False,
    )
    return output, "poisson_gray" if gray else "poisson_color", scale


def apply_jpeg(
    image: torch.Tensor,
    rng: random.Random,
    jpeg: torch.nn.Module,
    quality_range: tuple[float, float],
):
    quality = rng.uniform(*quality_range)
    quality_tensor = torch.tensor([quality], dtype=image.dtype, device=image.device)
    return jpeg(image.clamp(0, 1), quality=quality_tensor), quality


def save_png(image: torch.Tensor, output_path: Path) -> None:
    array = image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    array = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    output = Image.fromarray(array)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    output.save(temporary_path, format="PNG", compress_level=4)
    temporary_path.replace(output_path)


@torch.inference_mode()
def degrade_one(
    hr_path: Path,
    hr_root: Path,
    lr_root: Path,
    base_seed: int,
    scale: int,
    device: torch.device,
    jpeg: torch.nn.Module,
    overwrite: bool,
) -> dict:
    relative_path = hr_path.relative_to(hr_root)
    output_path = lr_root / relative_path
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"LR output already exists: {output_path}. Use --overwrite to replace it."
        )

    file_seed = seed_for_file(base_seed, relative_path)
    set_all_seeds(file_seed)
    rng = random.Random(file_seed)
    with Image.open(hr_path) as opened:
        hr_image = opened.convert("RGB")
    width, height = hr_image.size
    hr_array = np.asarray(hr_image, dtype=np.float32) / 255.0
    hr = torch.from_numpy(hr_array).permute(2, 0, 1).unsqueeze(0).to(device)

    kernel1, kernel1_kind, kernel1_size = make_blur_kernel(
        rng, sigma_range=(0.2, 3.0), sinc_probability=0.1, device=device
    )
    kernel2, kernel2_kind, kernel2_size = make_blur_kernel(
        rng, sigma_range=(0.2, 1.5), sinc_probability=0.1, device=device
    )
    sinc_kernel, sinc_kind, sinc_size = make_final_sinc_kernel(rng, device)

    # First Real-ESRGAN degradation stage.
    output = filter2d(hr, kernel1)
    first_resize_type = choose_weighted(rng, ("up", "down", "keep"), (0.2, 0.7, 0.1))
    if first_resize_type == "up":
        first_resize_scale = rng.uniform(1.0, 1.5)
    elif first_resize_type == "down":
        first_resize_scale = rng.uniform(0.15, 1.0)
    else:
        first_resize_scale = 1.0
    first_resize_mode = rng.choice(RESIZE_MODES)
    output = interpolate(
        output, scale_factor=first_resize_scale, mode=first_resize_mode
    )
    output, first_noise_type, first_noise_level = add_noise(
        output,
        rng,
        gaussian_probability=0.5,
        gaussian_range=(1.0, 30.0),
        poisson_range=(0.05, 3.0),
    )
    output, first_jpeg_quality = apply_jpeg(output, rng, jpeg, (30.0, 95.0))

    # Second Real-ESRGAN degradation stage.
    second_blur = rng.random() < 0.8
    if second_blur:
        output = filter2d(output, kernel2)
    second_resize_type = choose_weighted(rng, ("up", "down", "keep"), (0.3, 0.4, 0.3))
    if second_resize_type == "up":
        second_resize_scale = rng.uniform(1.0, 1.2)
    elif second_resize_type == "down":
        second_resize_scale = rng.uniform(0.3, 1.0)
    else:
        second_resize_scale = 1.0
    second_resize_mode = rng.choice(RESIZE_MODES)
    internal_height = max(1, int(height / scale * second_resize_scale))
    internal_width = max(1, int(width / scale * second_resize_scale))
    output = interpolate(
        output,
        size=(internal_height, internal_width),
        mode=second_resize_mode,
    )
    output, second_noise_type, second_noise_level = add_noise(
        output,
        rng,
        gaussian_probability=0.5,
        gaussian_range=(1.0, 25.0),
        poisson_range=(0.05, 2.5),
    )

    # Preserve the official randomized final order, then resize the degraded
    # low-resolution result back to the exact HR dimensions for paired training.
    final_height = max(1, height // scale)
    final_width = max(1, width // scale)
    final_resize_mode = rng.choice(RESIZE_MODES)
    if rng.random() < 0.5:
        final_order = "resize_sinc_then_jpeg"
        output = interpolate(
            output, size=(final_height, final_width), mode=final_resize_mode
        )
        output = filter2d(output, sinc_kernel)
        output, second_jpeg_quality = apply_jpeg(output, rng, jpeg, (30.0, 95.0))
    else:
        final_order = "jpeg_then_resize_sinc"
        output, second_jpeg_quality = apply_jpeg(output, rng, jpeg, (30.0, 95.0))
        output = interpolate(
            output, size=(final_height, final_width), mode=final_resize_mode
        )
        output = filter2d(output, sinc_kernel)

    output = interpolate(output.clamp(0, 1), size=(height, width), mode="bicubic")
    output = output.clamp(0, 1)
    if output.shape[-2:] != hr.shape[-2:]:
        raise RuntimeError(
            f"Spatial size mismatch for {relative_path}: "
            f"{tuple(output.shape[-2:])} vs {tuple(hr.shape[-2:])}"
        )
    mean_absolute_difference = float((output - hr).abs().mean().item() * 255.0)
    if mean_absolute_difference <= 0:
        raise RuntimeError(f"Degradation produced an unchanged image: {relative_path}")

    save_png(output, output_path)
    with Image.open(output_path) as saved:
        if saved.size != hr_image.size:
            raise RuntimeError(
                f"Saved LR size mismatch for {relative_path}: "
                f"{saved.size} vs {hr_image.size}"
            )

    return {
        "relative_path": relative_path.as_posix(),
        "seed": file_seed,
        "width": width,
        "height": height,
        "scale": scale,
        "kernel1": {"type": kernel1_kind, "size": kernel1_size},
        "first_resize": {
            "type": first_resize_type,
            "scale": round(first_resize_scale, 6),
            "mode": first_resize_mode,
        },
        "first_noise": {
            "type": first_noise_type,
            "level": round(first_noise_level, 6),
        },
        "first_jpeg_quality": round(first_jpeg_quality, 6),
        "kernel2": {
            "applied": second_blur,
            "type": kernel2_kind,
            "size": kernel2_size,
        },
        "second_resize": {
            "type": second_resize_type,
            "scale": round(second_resize_scale, 6),
            "mode": second_resize_mode,
        },
        "second_noise": {
            "type": second_noise_type,
            "level": round(second_noise_level, 6),
        },
        "final_sinc": {"type": sinc_kind, "size": sinc_size},
        "final_order": final_order,
        "final_resize_mode": final_resize_mode,
        "second_jpeg_quality": round(second_jpeg_quality, 6),
        "restored_to_original_size": True,
        "mean_absolute_difference": round(mean_absolute_difference, 6),
    }


def write_manifest(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    validate_args(args)
    hr_root = args.hr_dir.expanduser().resolve()
    lr_root = args.lr_dir.expanduser().resolve()
    manifest = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else lr_root.parent / "degradation_manifest.jsonl"
    )
    images = list_images(hr_root, args.limit)
    lr_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    jpeg = DiffJPEG(differentiable=False).to(device)

    print(f"HR directory: {hr_root}")
    print(f"LR directory: {lr_root}")
    print(f"Images: {len(images)}")
    print(f"Device: {device}")
    print(
        "Alignment policy: no crop/flip/rotation/warp; Real-ESRGAN "
        f"two-stage degradation at internal scale {args.scale}, followed by "
        "bicubic restoration to each HR image's exact dimensions."
    )

    records = []
    for index, image_path in enumerate(images, start=1):
        record = degrade_one(
            image_path,
            hr_root,
            lr_root,
            args.seed,
            args.scale,
            device,
            jpeg,
            args.overwrite,
        )
        records.append(record)
        if index % 25 == 0 or index == len(images):
            print(f"Processed {index}/{len(images)}")

    write_manifest(manifest, records)
    print(f"Successfully generated and verified {len(records)} aligned LR images.")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
