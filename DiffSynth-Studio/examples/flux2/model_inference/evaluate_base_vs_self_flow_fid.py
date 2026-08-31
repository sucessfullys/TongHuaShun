import argparse
import io
import json
import os
import tarfile
from collections import OrderedDict
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image, ImageOps
from torch.distributed.elastic.multiprocessing.errors import record

from diffsynth.core import ModelConfig, load_state_dict
from diffsynth.core.data.operators import ImageCropAndResize
from diffsynth.metrics import FIDMetric
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline


DEFAULT_BASE_MODEL = (
    "/mnt/image-edit/datasets/duanyufa/FLUX.2-klein-base-4B"
)
DEFAULT_COMPONENTS_MODEL = (
    "/mnt/image-edit/datasets/duanyufa/FLUX.2-klein-base-4B"
)
DEFAULT_TEST_METADATA = (
    "/mnt/image-edit/datasets/duanyufa/"
    "DiffSynth-Studio/data/test_clean.jsonl"
)
DEFAULT_TRAIN_OUTPUT = (
    "/mnt/image-edit/datasets/duanyufa/outputs/"
    "flux2_klein_base_4b_self_flow"
)
DEFAULT_EVAL_OUTPUT = (
    "/mnt/image-edit/datasets/duanyufa/outputs/"
    "flux2_klein_base_4b_self_flow/fid_evaluation"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate matched FLUX.2 base/Self-Flow samples and compute FID."
    )
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--components_model",
        default=DEFAULT_COMPONENTS_MODEL,
        help="Official Klein-4B text encoder, VAE, and tokenizer source.",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--train_output_dir", default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--metadata_path", default=DEFAULT_TEST_METADATA)
    parser.add_argument("--output_dir", default=DEFAULT_EVAL_OUTPUT)
    parser.add_argument("--image_column", default="image")
    parser.add_argument("--caption_column", default="caption")
    parser.add_argument("--tar_column", default="tar_file")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Zero evaluates every metadata record.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Fixed height. Leave height and width unset for GT-matched resolution.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Fixed width. Leave height and width unset for GT-matched resolution.",
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=1024 * 1024,
        help="Pixel budget for GT-matched resolution, identical to training.",
    )
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--embedded_guidance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--rand_device",
        choices=["cpu", "cuda"],
        default="cuda",
    )
    parser.add_argument("--fid_batch_size", type=int, default=32)
    parser.add_argument("--fid_num_workers", type=int, default=4)
    parser.add_argument(
        "--fid_model",
        default=None,
        help="Optional local FID model.safetensors path.",
    )
    parser.add_argument(
        "--generation_only",
        action="store_true",
        help="Generate/cache images without computing FID.",
    )
    return parser.parse_args()


def distributed_context():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FLUX.2 inference.")
    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(hours=6),
        )
    rank = dist.get_rank() if dist.is_initialized() else 0
    return rank, local_rank, world_size


def barrier():
    if dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])


def read_metadata(path, max_samples):
    records = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
                if max_samples > 0 and len(records) >= max_samples:
                    break
    if not records:
        raise ValueError(f"No records found in {path}.")
    return records


def checkpoint_label(checkpoint):
    parent = checkpoint.parent.name
    return parent if parent.startswith("checkpoint-") else checkpoint.stem


def inference_label(args):
    cfg = str(args.cfg_scale).replace(".", "p")
    guidance = str(args.embedded_guidance).replace(".", "p")
    resolution = (
        f"{args.width}x{args.height}"
        if args.height is not None
        else f"gt_ratio_maxpix{args.max_pixels}"
    )
    return (
        f"{resolution}_steps{args.num_inference_steps}"
        f"_cfg{cfg}_guidance{guidance}_seed{args.seed}"
    )


def latest_student_checkpoint(train_output_dir):
    checkpoints = []
    for path in Path(train_output_dir).glob("checkpoint-*"):
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        student = path / "student.safetensors"
        if student.is_file():
            checkpoints.append((step, student))
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoint-*/student.safetensors found under {train_output_dir}"
        )
    return max(checkpoints, key=lambda item: item[0])[1]


def validate_args(args):
    base = Path(args.base_model)
    components = Path(args.components_model)
    required = [
        base / "transformer",
        components / "text_encoder",
        components / "vae" / "diffusion_pytorch_model.safetensors",
        components / "tokenizer",
        Path(args.metadata_path),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))
    if (args.height is None) != (args.width is None):
        raise ValueError("height and width must either both be set or both be unset.")
    if args.height is not None and (args.height % 16 or args.width % 16):
        raise ValueError("height and width must be divisible by 16.")
    if args.max_pixels < 16 * 16:
        raise ValueError("max_pixels must be at least 256.")


def model_weight_files(directory):
    files = sorted(str(path) for path in Path(directory).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors weights found under {directory}")
    return files[0] if len(files) == 1 else files


def load_base_pipeline(base_model, components_model, device):
    base = Path(base_model)
    components = Path(components_model)
    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(path=model_weight_files(components / "text_encoder")),
            ModelConfig(path=model_weight_files(base / "transformer")),
            ModelConfig(
                path=str(
                    components / "vae" / "diffusion_pytorch_model.safetensors"
                )
            ),
        ],
        tokenizer_config=ModelConfig(path=str(components / "tokenizer")),
    )
    pipe.dit.eval()
    return pipe


class TarReaderCache:
    def __init__(self, max_open=8):
        self.max_open = max_open
        self.archives = OrderedDict()

    def get(self, path):
        path = str(path)
        archive = self.archives.pop(path, None)
        if archive is None:
            archive = tarfile.open(path, mode="r:*")
        self.archives[path] = archive
        while len(self.archives) > self.max_open:
            _, oldest = self.archives.popitem(last=False)
            oldest.close()
        return archive

    def close(self):
        for archive in self.archives.values():
            archive.close()
        self.archives.clear()


def cache_reference_images_and_sizes(
    records,
    output_dir,
    rank,
    world_size,
    args,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = TarReaderCache()
    image_processor = ImageCropAndResize(
        height=args.height,
        width=args.width,
        max_pixels=args.max_pixels,
        height_division_factor=16,
        width_division_factor=16,
    )
    target_sizes = {}
    try:
        for index in range(rank, len(records), world_size):
            record = records[index]
            suffix = Path(record[args.image_column]).suffix.lower() or ".jpg"
            output_path = output_dir / f"{index:08d}{suffix}"
            if output_path.is_file() and output_path.stat().st_size > 0:
                image_bytes = output_path.read_bytes()
            else:
                archive = cache.get(record[args.tar_column])
                member = archive.extractfile(record[args.image_column])
                if member is None:
                    raise FileNotFoundError(
                        f"{record[args.image_column]} not found in "
                        f"{record[args.tar_column]}"
                    )
                image_bytes = member.read()
                temporary_path = output_path.with_suffix(
                    output_path.suffix + ".tmp"
                )
                temporary_path.write_bytes(image_bytes)
                temporary_path.replace(output_path)
            with Image.open(io.BytesIO(image_bytes)) as image:
                image = ImageOps.exif_transpose(image)
                target_sizes[index] = image_processor.get_height_width(image)
    finally:
        cache.close()
    return target_sizes


@torch.inference_mode()
def generate_images(
    pipe,
    records,
    target_sizes,
    output_dir,
    rank,
    world_size,
    args,
    label,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(rank, len(records), world_size):
        height, width = target_sizes[index]
        output_path = output_dir / f"{index:08d}.png"
        if output_path.is_file() and output_path.stat().st_size > 0:
            try:
                with Image.open(output_path) as cached_image:
                    if cached_image.size == (width, height):
                        continue
            except OSError:
                pass
        try:
            image = pipe(
                prompt=str(records[index][args.caption_column]),
                negative_prompt="",
                cfg_scale=args.cfg_scale,
                embedded_guidance=args.embedded_guidance,
                height=height,
                width=width,
                seed=args.seed + index,
                rand_device=args.rand_device,
                num_inference_steps=args.num_inference_steps,
            )
            temporary_path = output_path.with_suffix(".tmp.png")
            image.save(temporary_path)
            temporary_path.replace(output_path)
        except Exception as error:
            raise RuntimeError(
                f"{label} generation failed on rank={rank}, index={index}, "
                f"size={width}x{height}, seed={args.seed + index}"
            ) from error
        print(
            f"[rank {rank}] {label}: {index + 1}/{len(records)} "
            f"({width}x{height}) -> {output_path}",
            flush=True,
        )


def validate_generated_images(path, num_samples):
    missing = [
        index
        for index in range(num_samples)
        if not (Path(path) / f"{index:08d}.png").is_file()
    ]
    if missing:
        preview = ", ".join(str(index) for index in missing[:10])
        raise RuntimeError(
            f"{len(missing)} generated images are missing under {path}; "
            f"first missing indices: {preview}"
        )


def validate_reference_images(path, num_samples):
    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
    }
    missing = []
    duplicate = []
    for index in range(num_samples):
        matches = [
            item
            for item in Path(path).glob(f"{index:08d}.*")
            if item.suffix.lower() in image_extensions
        ]
        if not matches:
            missing.append(index)
        elif len(matches) > 1:
            duplicate.append(index)
    if missing or duplicate:
        raise RuntimeError(
            f"Invalid reference cache under {path}: "
            f"missing={missing[:10]}, duplicate={duplicate[:10]}"
        )


def build_fid_metric(args):
    if args.fid_model:
        model_config = ModelConfig(path=args.fid_model)
    else:
        model_config = ModelConfig(
            model_id="DiffSynth-Studio/ImageMetrics",
            origin_file_pattern="FID/model.safetensors",
        )
    return FIDMetric.from_pretrained(
        model_config=model_config,
        device="cuda",
        batch_size=args.fid_batch_size,
        num_workers=args.fid_num_workers,
    )


@record
def main():
    args = parse_args()
    validate_args(args)
    rank, local_rank, world_size = distributed_context()
    device = torch.device("cuda", local_rank)

    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else latest_student_checkpoint(args.train_output_dir)
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Student checkpoint not found: {checkpoint}")

    records = read_metadata(args.metadata_path, args.max_samples)
    output_root = Path(args.output_dir)
    reference_dir = output_root / "reference"
    run_label = inference_label(args)
    base_dir = output_root / f"base_{run_label}"
    self_flow_dir = (
        output_root / f"self_flow_{checkpoint_label(checkpoint)}_{run_label}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        resolution_description = (
            f"{args.width}x{args.height}"
            if args.height is not None
            else f"GT aspect ratio, max_pixels={args.max_pixels}"
        )
        print(
            f"Evaluating {len(records)} samples with {world_size} process(es).\n"
            f"Student checkpoint: {checkpoint}\n"
            f"Inference resolution: {resolution_description}, "
            f"steps={args.num_inference_steps}, cfg={args.cfg_scale}, "
            f"seed={args.seed}+index"
        )

    target_sizes = cache_reference_images_and_sizes(
        records,
        reference_dir,
        rank,
        world_size,
        args,
    )
    barrier()

    pipe = load_base_pipeline(args.base_model, args.components_model, device)
    generate_images(
        pipe,
        records,
        target_sizes,
        base_dir,
        rank,
        world_size,
        args,
        "base",
    )
    barrier()

    state_dict = load_state_dict(
        str(checkpoint),
        torch_dtype=torch.bfloat16,
        device="cpu",
    )
    pipe.dit.load_state_dict(state_dict, strict=True)
    del state_dict
    pipe.dit.eval()
    generate_images(
        pipe,
        records,
        target_sizes,
        self_flow_dir,
        rank,
        world_size,
        args,
        "self_flow",
    )
    barrier()

    del pipe
    torch.cuda.empty_cache()

    if dist.is_initialized():
        dist.destroy_process_group()

    if rank == 0:
        expected = len(records)
        validate_reference_images(reference_dir, expected)
        validate_generated_images(base_dir, expected)
        validate_generated_images(self_flow_dir, expected)

        results = {
            "num_samples": expected,
            "checkpoint": str(checkpoint),
            "height": args.height,
            "width": args.width,
            "resolution_mode": (
                "fixed" if args.height is not None else "gt_aspect_ratio"
            ),
            "max_pixels": args.max_pixels,
            "num_inference_steps": args.num_inference_steps,
            "cfg_scale": args.cfg_scale,
            "embedded_guidance": args.embedded_guidance,
            "seed": args.seed,
        }
        if not args.generation_only:
            metric = build_fid_metric(args)
            results["fid_base"] = metric.compute(
                reference_dir,
                base_dir,
            )
            results["fid_self_flow"] = metric.compute(
                reference_dir,
                self_flow_dir,
            )
            results["fid_delta_self_flow_minus_base"] = (
                results["fid_self_flow"] - results["fid_base"]
            )
        results_path = output_root / (
            f"fid_results_{checkpoint_label(checkpoint)}_{run_label}.json"
        )
        results_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
