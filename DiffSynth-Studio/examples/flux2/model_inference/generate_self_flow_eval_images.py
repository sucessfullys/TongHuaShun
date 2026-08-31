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
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline


DEFAULT_BASE_MODEL = (
    "/mnt/image-edit/datasets/duanyufa/FLUX.2-klein-base-4B"
)
DEFAULT_COMPONENTS_MODEL = (
    "/mnt/image-edit/datasets/duanyufa/FLUX.2-klein-base-4B"
)
DEFAULT_METADATA = (
    "/mnt/image-edit/datasets/duanyufa/"
    "DiffSynth-Studio/data/test_clean.jsonl"
)
DEFAULT_TRAIN_OUTPUT = (
    "/mnt/image-edit/datasets/duanyufa/outputs/"
    "flux2_klein_base_4b_self_flow"
)
DEFAULT_REFERENCE_DIR = (
    "/mnt/image-edit/datasets/duanyufa/outputs/"
    "flux2_klein_base_4b_self_flow/fid_evaluation/reference"
)
DEFAULT_OUTPUT_DIR = (
    "/mnt/image-edit/datasets/duanyufa/outputs/"
    "flux2_klein_base_4b_self_flow/fid_evaluation/self_flow_custom"
)
IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a test-set image directory from a Self-Flow student."
    )
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--components_model",
        default=DEFAULT_COMPONENTS_MODEL,
        help="Official Klein-4B text encoder, VAE, and tokenizer source.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="student.safetensors path. Defaults to the latest checkpoint.",
    )
    parser.add_argument("--train_output_dir", default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--metadata_path", default=DEFAULT_METADATA)
    parser.add_argument(
        "--reference_dir",
        default=DEFAULT_REFERENCE_DIR,
        help=(
            "Existing indexed GT directory used only to read image dimensions. "
            "Set to an empty string to read dimensions from metadata tar files."
        ),
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image_column", default="image")
    parser.add_argument("--caption_column", default="caption")
    parser.add_argument("--tar_column", default="tar_file")
    parser.add_argument("--tar_cache_size", type=int, default=8)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Zero generates every metadata record.",
    )
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=1024 * 1024)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--embedded_guidance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--rand_device",
        choices=["cpu", "cuda"],
        default="cuda",
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


def latest_student_checkpoint(train_output_dir):
    checkpoints = []
    for checkpoint_dir in Path(train_output_dir).glob("checkpoint-*"):
        try:
            step = int(checkpoint_dir.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        student = checkpoint_dir / "student.safetensors"
        if student.is_file():
            checkpoints.append((step, student))
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoint-*/student.safetensors found under {train_output_dir}"
        )
    return max(checkpoints, key=lambda item: item[0])[1]


def model_weight_files(directory):
    files = sorted(str(path) for path in Path(directory).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors weights found under {directory}")
    return files[0] if len(files) == 1 else files


def validate_args(args, checkpoint):
    base = Path(args.base_model)
    components = Path(args.components_model)
    required = [
        base / "transformer",
        components / "text_encoder",
        components / "vae" / "diffusion_pytorch_model.safetensors",
        components / "tokenizer",
        Path(args.metadata_path),
        checkpoint,
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
    if args.reference_dir and not Path(args.reference_dir).is_dir():
        raise FileNotFoundError(
            f"Reference directory does not exist: {args.reference_dir}"
        )


def load_pipeline(args, checkpoint, device):
    base = Path(args.base_model)
    components = Path(args.components_model)
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
    state_dict = load_state_dict(
        str(checkpoint),
        torch_dtype=torch.bfloat16,
        device="cpu",
    )
    pipe.dit.load_state_dict(state_dict, strict=True)
    del state_dict
    pipe.dit.eval()
    return pipe


class TarReaderCache:
    def __init__(self, max_open):
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


class TargetSizeReader:
    def __init__(self, args):
        self.args = args
        self.reference_dir = (
            Path(args.reference_dir) if args.reference_dir else None
        )
        self.tar_cache = TarReaderCache(args.tar_cache_size)
        self.image_processor = ImageCropAndResize(
            height=args.height,
            width=args.width,
            max_pixels=args.max_pixels,
            height_division_factor=16,
            width_division_factor=16,
        )

    def _reference_path(self, index):
        matches = [
            path
            for path in self.reference_dir.glob(f"{index:08d}.*")
            if path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one reference image for index {index}, got {matches}"
            )
        return matches[0]

    def get(self, index, record):
        if self.args.height is not None:
            return self.args.height, self.args.width
        if self.reference_dir is not None:
            with Image.open(self._reference_path(index)) as image:
                image = ImageOps.exif_transpose(image)
                return self.image_processor.get_height_width(image)
        archive = self.tar_cache.get(record[self.args.tar_column])
        member = archive.extractfile(record[self.args.image_column])
        if member is None:
            raise FileNotFoundError(
                f"{record[self.args.image_column]} not found in "
                f"{record[self.args.tar_column]}"
            )
        with Image.open(io.BytesIO(member.read())) as image:
            image = ImageOps.exif_transpose(image)
            return self.image_processor.get_height_width(image)

    def close(self):
        self.tar_cache.close()


def valid_cached_image(path, width, height):
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            return image.size == (width, height)
    except OSError:
        return False


@torch.inference_mode()
def generate_images(pipe, records, output_dir, rank, world_size, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    size_reader = TargetSizeReader(args)
    try:
        for index in range(rank, len(records), world_size):
            record = records[index]
            height, width = size_reader.get(index, record)
            output_path = output_dir / f"{index:08d}.png"
            if valid_cached_image(output_path, width, height):
                continue
            try:
                image = pipe(
                    prompt=str(record[args.caption_column]),
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
                    f"Generation failed on rank={rank}, index={index}, "
                    f"size={width}x{height}, seed={args.seed + index}"
                ) from error
            print(
                f"[rank {rank}] {index + 1}/{len(records)} "
                f"({width}x{height}) -> {output_path}",
                flush=True,
            )
    finally:
        size_reader.close()


def validate_outputs(output_dir, num_samples):
    missing = []
    for index in range(num_samples):
        path = output_dir / f"{index:08d}.png"
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(index)
    if missing:
        raise RuntimeError(
            f"{len(missing)} outputs are missing; first indices: {missing[:10]}"
        )


@record
def main():
    args = parse_args()
    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else latest_student_checkpoint(args.train_output_dir)
    )
    validate_args(args, checkpoint)
    rank, local_rank, world_size = distributed_context()
    records = read_metadata(args.metadata_path, args.max_samples)
    output_dir = Path(args.output_dir)

    if rank == 0:
        resolution = (
            f"{args.width}x{args.height}"
            if args.height is not None
            else f"GT aspect ratio with max_pixels={args.max_pixels}"
        )
        print(
            f"Generating {len(records)} samples with {world_size} process(es).\n"
            f"Checkpoint: {checkpoint}\n"
            f"Output: {output_dir}\n"
            f"Inference: {resolution}, steps={args.num_inference_steps}, "
            f"cfg={args.cfg_scale}, seed={args.seed}+index"
        )

    pipe = load_pipeline(
        args,
        checkpoint,
        torch.device("cuda", local_rank),
    )
    generate_images(
        pipe,
        records,
        output_dir,
        rank,
        world_size,
        args,
    )
    barrier()

    if rank == 0:
        validate_outputs(output_dir, len(records))
        manifest = {
            "num_samples": len(records),
            "metadata_path": str(Path(args.metadata_path).resolve()),
            "checkpoint": str(checkpoint.resolve()),
            "base_model": str(Path(args.base_model).resolve()),
            "components_model": str(Path(args.components_model).resolve()),
            "reference_dir": (
                str(Path(args.reference_dir).resolve())
                if args.reference_dir
                else None
            ),
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
        manifest_path = output_dir / "generation_config.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Generation complete. Manifest saved to {manifest_path}")

    del pipe
    torch.cuda.empty_cache()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
