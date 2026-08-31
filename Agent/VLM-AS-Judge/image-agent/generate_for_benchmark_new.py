import argparse
import json
import os
import sys
import traceback
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable

from datasets import load_dataset, load_from_disk
from PIL import Image


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from models import get_editor, ALL_MODELS, get_resized_dimensions


DEFAULT_BENCH_PATH = (
    CURRENT_DIR.parent / "benchmark" / "GEditBench_v2" / "datasets" / "GEditBench-v2"
)
DEFAULT_IMAGE_SAVE_DIR = (
    CURRENT_DIR.parent
    / "benchmark"
    / "GEditBench_v2"
    / "datasets"
    / "GEditBench-v2-CandidatesGallery"
)
DEFAULT_MERGE_TO_METADATA = (
    CURRENT_DIR.parent
    / "benchmark"
    / "GEditBench_v2"
    / "datasets"
    / "GEditBench-v2-CandidatesGallery"
    / "metadata.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run image editing models on GEditBench v2 and export results in candidates/gallery format."
    )
    parser.add_argument(
        "--bench-path",
        type=str,
        default=str(DEFAULT_BENCH_PATH),
        help="Path to the local GEditBench-v2 dataset.",
    )
    parser.add_argument(
        "--image-save-dir",
        type=str,
        default=str(DEFAULT_IMAGE_SAVE_DIR),
        help="Candidates gallery root. Outputs are saved to <image-save-dir>/<model-name>/<key>.png.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="FireRed_Image_Edit",
        help="Folder/model name used in candidates/gallery and merged metadata. Must match a deployed service.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Override the local model path. Only valid with --no-use-api.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override the local pipeline device, such as cuda:0. Only valid with --no-use-api.",
    )
    parser.add_argument(
        "--merge-to-metadata",
        type=str,
        default=str(DEFAULT_MERGE_TO_METADATA),
        help="Existing metadata.jsonl to merge generated candidate paths into.",
    )
    parser.add_argument(
        "--results-jsonl",
        type=str,
        default=None,
        help="Output jsonl path. Defaults to <image-save-dir>/<model-name>_generation_results.jsonl.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Only run the first N samples, useful for smoke tests.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip samples before this index.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=2560,
        help="Resize overly large inputs so the longest side is at most this value.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate images even if cache or existing files are found.",
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        default=True,
        help="Use API-based editor (default).",
    )
    parser.add_argument(
        "--no-use-api",
        action="store_false",
        dest="use_api",
        help="Use local pipeline instead of API.",
    )
    return parser.parse_args()


def resolve_output_paths(args: argparse.Namespace) -> Dict[str, Path]:
    bench_path = Path(args.bench_path).expanduser().resolve()
    image_save_dir = Path(args.image_save_dir).expanduser().resolve()
    model_dir = image_save_dir / args.model_name
    results_jsonl = (
        Path(args.results_jsonl).expanduser().resolve()
        if args.results_jsonl
        else image_save_dir / f"{args.model_name}_generation_results.jsonl"
    )
    merge_to_metadata = (
        Path(args.merge_to_metadata).expanduser().resolve()
        if args.merge_to_metadata
        else None
    )
    return {
        "bench_path": bench_path,
        "image_save_dir": image_save_dir,
        "model_dir": model_dir,
        "results_jsonl": results_jsonl,
        "merge_to_metadata": merge_to_metadata,
    }


def load_benchmark_dataset(bench_path: Path):
    parquet_root = bench_path / "data"
    parquet_files = sorted(parquet_root.glob("*.parquet")) if parquet_root.exists() else []
    if parquet_files:
        return load_dataset(
            "parquet",
            data_files={"train": [str(path) for path in parquet_files]},
            split="train",
        )

    try:
        dataset_obj = load_from_disk(str(bench_path))
    except Exception as exc:
        raise FileNotFoundError(
            f"Cannot load benchmark from {bench_path}. "
            "Expected either a HuggingFace dataset directory or parquet files under data/."
        ) from exc

    if hasattr(dataset_obj, "keys") and "train" in dataset_obj:
        return dataset_obj["train"]
    return dataset_obj


def ensure_rgb_image(image_like) -> Image.Image:
    if isinstance(image_like, Image.Image):
        return image_like.convert("RGB")
    if isinstance(image_like, str):
        return Image.open(image_like).convert("RGB")
    if hasattr(image_like, "convert"):
        return image_like.convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image_like)!r}")


def normalize_record(item: dict, idx: int) -> dict:
    key = item.get("key")
    if not key:
        task_name = str(item.get("task", "sample"))
        key = f"{task_name}_{idx:06d}"

    instruction = item.get("instruction") or item.get("prompt")
    if not instruction:
        raise ValueError(f"Sample {key} is missing instruction/prompt.")

    image_field = None
    for candidate in ("source_image", "input_image_raw", "input_image", "image"):
        if candidate in item and item[candidate] is not None:
            image_field = item[candidate]
            break
    if image_field is None:
        raise ValueError(f"Sample {key} is missing source image.")

    return {
        "key": key,
        "instruction": instruction,
        "task": item.get("task"),
        "source_image": ensure_rgb_image(image_field),
    }


def build_result_record(key: str, instruction: str, image_path: Path) -> dict:
    return {
        "key": key,
        "image_path": str(image_path.resolve()),
        "instruction": instruction,
    }


def collect_results(dataset_items: Iterable[dict], model_dir: Path) -> Dict[str, dict]:
    results: Dict[str, dict] = {}
    for item in dataset_items:
        image_path = model_dir / f"{item['key']}.png"
        if not image_path.exists():
            continue
        results[item["key"]] = build_result_record(item["key"], item["instruction"], image_path)
    return results


def write_results_jsonl(results_path: Path, dataset_items: Iterable[dict], results: Dict[str, dict]) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as handle:
        for item in dataset_items:
            result = results.get(item["key"])
            if result is None:
                continue
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")


def merge_to_metadata(metadata_path: Path, model_name: str, dataset_items: Iterable[dict], results: Dict[str, dict]) -> Path:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = [json.loads(line) for line in handle if line.strip()]

    key_to_image = {
        item["key"]: os.path.relpath(
            results[item["key"]]["image_path"],
            start=str(metadata_path.parent),
        )
        for item in dataset_items
        if item["key"] in results
    }

    merged_metadata = []
    for item in metadata:
        rel_image_path = key_to_image.get(item.get("key"))
        if rel_image_path is None:
            continue
        candidates = item.setdefault("candidates", [])
        candidates = [cand for cand in candidates if cand.get("model") != model_name]
        candidates.append({"model": model_name, "image": rel_image_path})
        item["candidates"] = candidates
        merged_metadata.append(item)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_output = metadata_path.parent / f"metadata_{timestamp}.jsonl"

    # 只保存有新增数据的条目
    with merged_output.open("w", encoding="utf-8") as handle:
        for item in merged_metadata:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return merged_output


def main() -> int:
    args = parse_args()

    if args.model_name not in ALL_MODELS:
        print(f"Error: unknown model '{args.model_name}'. Available: {sorted(ALL_MODELS)}")
        return 1

    if args.use_api and (args.model_path is not None or args.device is not None):
        print("Error: --model-path and --device are only valid with --no-use-api.")
        return 1

    editor_kwargs = {}
    if args.model_path is not None:
        editor_kwargs["model_path"] = args.model_path
    if args.device is not None:
        editor_kwargs["device"] = args.device
    editor = get_editor(args.model_name, use_api=args.use_api, **editor_kwargs)

    paths = resolve_output_paths(args)
    dataset = load_benchmark_dataset(paths["bench_path"])

    print("start read dataset")
    t1 = time.time()

    # 数据集共1200张图片，取200张进行测试，节约时间
    # indices = [idx for idx in range(args.start_index, len(dataset)) if idx % 6 == 0]
    indices = [idx for idx in range(args.start_index, len(dataset))]
    if args.max_items is not None:
        indices = indices[:args.max_items]

    raw_items = []
    for idx in indices:
        item = dataset[idx]
        raw_items.append(normalize_record(item, idx))

    if not raw_items:
        print("No samples to process.")
        return 0

    print(f"finish read dataset, time cost: {time.time() - t1:.2f}s")
    mode_str = "API" if args.use_api else "Pipeline"
    print(f"Using {mode_str} mode for model {args.model_name}")

    paths["model_dir"].mkdir(parents=True, exist_ok=True)
    if("HiDream_O1_Image" in paths["model_dir"].name):
        paths["model_dir_ori"] = paths["model_dir"].parent / "HiDream_O1_Image_orisize"
        paths["model_dir_ori"].mkdir(parents=True, exist_ok=True)

    total = len(raw_items)
    success_count = 0
    skip_count = 0
    fail_count = 0

    for idx, item in enumerate(raw_items, start=1):
        image_path = paths["model_dir"] / f"{item['key']}.png"

        if not args.overwrite:
            if image_path.exists():
                skip_count += 1
                print(f"[{idx}/{total}] skip existing file: {item['key']}")
                continue

        try:
            image = item["source_image"]
            result_image = editor.edit(image, item["instruction"])

            image_path.parent.mkdir(parents=True, exist_ok=True)
            result_image.save(image_path, format="PNG")

            if("HiDream_O1_Image" in paths["model_dir"].name):
                target_w, target_h = get_resized_dimensions(image)
                image_path.parent.mkdir(parents=True, exist_ok=True)
                result_image = result_image.resize((target_w, target_h), resample=Image.LANCZOS)
                result_image.save(paths["model_dir_ori"] / f"{item['key']}.png", format="PNG")

            success_count += 1
            print(f"[{idx}/{total}] done: {item['key']}")
        except Exception as exc:
            fail_count += 1
            print(f"[{idx}/{total}] failed: {item['key']} -> {exc}")
            traceback.print_exc()

    results = collect_results(raw_items, paths["model_dir"])
    write_results_jsonl(paths["results_jsonl"], raw_items, results)
    print(f"Generation results saved to: {paths['results_jsonl']}")

    if paths["merge_to_metadata"] is not None:
        merged_output = merge_to_metadata(
            metadata_path=paths["merge_to_metadata"],
            model_name=args.model_name,
            dataset_items=raw_items,
            results=results,
        )
        print(f"Merged metadata saved to: {merged_output}")

    print(
        "Finished. "
        f"success={success_count}, skipped={skip_count}, failed={fail_count}, total={total}"
    )
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
