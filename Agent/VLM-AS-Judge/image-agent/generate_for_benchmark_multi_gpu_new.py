#!/usr/bin/env python3
"""Run a registered local image-editing pipeline on GEditBench-v2 with multiple GPUs."""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq
from PIL import Image


DEFAULT_BENCH_PATH = (
    Path("/mnt/image-edit/datasets/dingjianbiao/agent/benchmark")
    / "GEditBench_v2"
    / "datasets"
    / "GEditBench-v2"
)
DEFAULT_IMAGE_SAVE_DIR = (
    Path("/mnt/image-edit/datasets/dingjianbiao/agent/benchmark")
    / "GEditBench_v2"
    / "datasets"
    / "GEditBench-v2-CandidatesGallery"
)
DEFAULT_MERGE_TO_METADATA = DEFAULT_IMAGE_SAVE_DIR / "metadata.jsonl"


def parse_gpu_ids(value: str) -> list[str]:
    gpu_ids: list[str] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise argparse.ArgumentTypeError(f"Invalid GPU range: {part}")
            gpu_ids.extend(str(index) for index in range(start, end + 1))
        else:
            gpu_ids.append(str(int(part)))
    if not gpu_ids:
        raise argparse.ArgumentTypeError("At least one GPU ID is required")
    if len(gpu_ids) != len(set(gpu_ids)):
        raise argparse.ArgumentTypeError("GPU IDs must not contain duplicates")
    return gpu_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-path", type=Path, default=DEFAULT_BENCH_PATH)
    parser.add_argument("--image-save-dir", type=Path, default=DEFAULT_IMAGE_SAVE_DIR)
    parser.add_argument("--model-name", default="FLUX2_klein_9b")
    parser.add_argument(
        "--output-model-name",
        default=None,
        help=(
            "Model name used for output folder, generation_results.jsonl, and "
            "merged metadata. Defaults to --model-name."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Local model weights directory passed to the registered Pipeline Editor.",
    )
    parser.add_argument(
        "--gpus",
        type=parse_gpu_ids,
        default=parse_gpu_ids("0-7"),
        help="Physical GPU IDs, such as '0-7' or '0,1,4,5'. Default: 0-7.",
    )
    parser.add_argument(
        "--merge-to-metadata",
        type=Path,
        default=DEFAULT_MERGE_TO_METADATA,
        help="Existing gallery metadata.jsonl used as the merge base.",
    )
    parser.add_argument(
        "--results-jsonl",
        type=Path,
        default=None,
        help="Defaults to <image-save-dir>/<model-name>_generation_results.jsonl.",
    )
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_items is not None and args.max_items < 0:
        parser.error("--max-items must be >= 0")
    if args.start_index < 0:
        parser.error("--start-index must be >= 0")
    return args


def get_parquet_files(bench_path: Path) -> list[Path]:
    parquet_files = sorted((bench_path / "data").glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {bench_path / 'data'}")
    return parquet_files


def iter_samples(parquet_files: list[Path]) -> Iterator[dict]:
    global_index = 0
    columns = ["key", "instruction", "source_image", "task"]
    for parquet_path in parquet_files:
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(batch_size=1, columns=columns):
            sample = batch.to_pylist()[0]
            sample["global_index"] = global_index
            global_index += 1
            yield sample


def iter_selected_samples(
    parquet_files: list[Path], start_index: int, max_items: int | None
) -> Iterator[dict]:
    selected = 0
    for sample in iter_samples(parquet_files):
        if sample["global_index"] < start_index:
            continue
        if max_items is not None and selected >= max_items:
            break
        selected += 1
        yield sample


def decode_source_image(source_image) -> Image.Image:
    if isinstance(source_image, Image.Image):
        return source_image.convert("RGB")
    if isinstance(source_image, str):
        return Image.open(source_image).convert("RGB")
    if isinstance(source_image, dict):
        image_bytes = source_image.get("bytes")
        image_path = source_image.get("path")
        if image_bytes:
            with Image.open(io.BytesIO(image_bytes)) as image:
                return image.convert("RGB")
        if image_path:
            return Image.open(image_path).convert("RGB")
    raise TypeError(f"Unsupported source image: {type(source_image)!r}")


def output_path_for_sample(model_dir: Path, sample: dict) -> Path:
    return model_dir / f"{sample['key']}.png"


def producer_loop(samples, task_queue, result_queue, worker_count: int) -> None:
    selected = 0
    try:
        for sample in samples:
            task_queue.put(sample)
            selected += 1
    except Exception as exc:
        result_queue.put({"kind": "producer_error", "error": str(exc)})
    finally:
        for _ in range(worker_count):
            task_queue.put(None)
        result_queue.put({"kind": "producer_done", "selected": selected})


def worker_loop(
    worker_id: int,
    gpu_id: str,
    model_name: str,
    model_path: str,
    model_dir: str,
    overwrite: bool,
    task_queue,
    result_queue,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is not available for physical GPU {gpu_id}")
        torch.cuda.set_device(0)

        from models import get_editor

        print(f"[worker {worker_id} | GPU {gpu_id}] loading model ...", flush=True)
        editor = get_editor(
            model_name,
            use_api=False,
            model_path=model_path,
            device="cuda:0",
        )
        load_pipeline = getattr(editor, "_load_pipeline", None)
        if load_pipeline is not None:
            load_pipeline()
        result_queue.put({"kind": "ready", "worker_id": worker_id, "gpu_id": gpu_id})
        print(f"[worker {worker_id} | GPU {gpu_id}] ready", flush=True)
    except Exception as exc:
        result_queue.put(
            {
                "kind": "load_error",
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "error": str(exc),
            }
        )
        return

    while True:
        sample = task_queue.get()
        if sample is None:
            break
        output_path = output_path_for_sample(Path(model_dir), sample)
        try:
            if output_path.exists() and not overwrite:
                result_queue.put({"kind": "skipped", "key": sample["key"], "gpu_id": gpu_id})
                continue
            result_image = editor.edit(
                decode_source_image(sample["source_image"]),
                sample["instruction"],
            )
            result_image.save(output_path, format="PNG")
            result_queue.put({"kind": "ok", "key": sample["key"], "gpu_id": gpu_id})
        except Exception as exc:
            result_queue.put(
                {
                    "kind": "error",
                    "key": sample["key"],
                    "gpu_id": gpu_id,
                    "error": str(exc),
                }
            )
    result_queue.put({"kind": "done", "worker_id": worker_id, "gpu_id": gpu_id})


def wait_until_workers_ready(workers: list, result_queue) -> None:
    ready_workers = 0
    while ready_workers < len(workers):
        result = result_queue.get()
        if result["kind"] == "ready":
            ready_workers += 1
            continue
        if result["kind"] == "load_error":
            raise RuntimeError(
                f"Worker {result['worker_id']} on GPU {result['gpu_id']} failed to load model: "
                f"{result['error']}"
            )
        raise RuntimeError(f"Unexpected worker message during startup: {result}")


def terminate_workers(workers: list) -> None:
    for worker in workers:
        if worker.is_alive():
            worker.terminate()
    for worker in workers:
        worker.join(timeout=5)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def collect_results(samples: list[dict], model_dir: Path) -> list[dict]:
    records = []
    for sample in samples:
        image_path = output_path_for_sample(model_dir, sample)
        if image_path.is_file():
            records.append(
                {
                    "key": sample["key"],
                    "image_path": str(image_path.resolve()),
                    "instruction": sample["instruction"],
                }
            )
    return records


def merge_to_metadata(metadata_path: Path, model_name: str, results: list[dict]) -> Path:
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    result_by_key = {record["key"]: record for record in results}
    with metadata_path.open(encoding="utf-8") as input_file:
        metadata = [json.loads(line) for line in input_file if line.strip()]

    merged_metadata = []
    for item in metadata:
        result = result_by_key.get(item.get("key"))
        if result is None:
            continue
        candidates = [
            candidate
            for candidate in item.get("candidates", [])
            if candidate.get("model") != model_name
        ]
        candidates.append(
            {
                "model": model_name,
                "image": os.path.relpath(result["image_path"], start=metadata_path.parent),
            }
        )
        merged_item = dict(item)
        merged_item["candidates"] = candidates
        merged_metadata.append(merged_item)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_output = metadata_path.parent / f"metadata_{timestamp}.jsonl"
    write_jsonl(merged_output, merged_metadata)
    return merged_output


def main() -> int:
    args = parse_args()
    output_model_name = args.output_model_name or args.model_name
    model_path = args.model_path.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    parquet_files = get_parquet_files(args.bench_path.expanduser().resolve())
    samples = list(iter_selected_samples(parquet_files, args.start_index, args.max_items))
    if not samples:
        print("No samples to process.")
        return 0

    image_save_dir = args.image_save_dir.expanduser().resolve()
    model_dir = image_save_dir / output_model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl = (
        args.results_jsonl.expanduser().resolve()
        if args.results_jsonl
        else image_save_dir / f"{output_model_name}_generation_results.jsonl"
    )

    context = mp.get_context("spawn")
    task_queue = context.Queue(maxsize=max(2 * len(args.gpus), 1))
    result_queue = context.Queue()
    workers = []
    for worker_id, gpu_id in enumerate(args.gpus):
        worker = context.Process(
            target=worker_loop,
            args=(
                worker_id,
                gpu_id,
                args.model_name,
                str(model_path),
                str(model_dir),
                args.overwrite,
                task_queue,
                result_queue,
            ),
        )
        worker.start()
        workers.append(worker)

    started_at = time.time()
    try:
        wait_until_workers_ready(workers, result_queue)
        producer = threading.Thread(
            target=producer_loop,
            args=(iter(samples), task_queue, result_queue, len(workers)),
            daemon=True,
        )
        producer.start()

        completed = 0
        skipped = 0
        failed = 0
        finished_workers = 0
        dispatched = None
        while finished_workers < len(workers):
            try:
                result = result_queue.get(timeout=5)
            except queue.Empty:
                dead_workers = [
                    index
                    for index, worker in enumerate(workers)
                    if not worker.is_alive() and worker.exitcode not in (None, 0)
                ]
                if dead_workers:
                    raise RuntimeError(f"Workers exited unexpectedly: {dead_workers}")
                continue

            kind = result["kind"]
            if kind == "producer_done":
                dispatched = result["selected"]
                continue
            if kind == "producer_error":
                raise RuntimeError(f"Dataset producer failed: {result['error']}")
            if kind == "done":
                finished_workers += 1
                continue
            if kind == "ok":
                completed += 1
            elif kind == "skipped":
                skipped += 1
            elif kind == "error":
                failed += 1
                print(
                    f"ERROR GPU {result['gpu_id']} key={result['key']}: {result['error']}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                raise RuntimeError(f"Unexpected worker message: {result}")

            processed = completed + skipped + failed
            print(
                f"Progress: {processed}/{dispatched or len(samples)} | done={completed} "
                f"skipped={skipped} failed={failed} | elapsed={time.time() - started_at:.1f}s",
                flush=True,
            )

        producer.join()
    finally:
        terminate_workers(workers)

    results = collect_results(samples, model_dir)
    write_jsonl(results_jsonl, results)
    merged_metadata = merge_to_metadata(
        args.merge_to_metadata.expanduser().resolve(),
        output_model_name,
        results,
    )
    print(
        f"Finished: selected={len(samples)} done={completed} skipped={skipped} "
        f"failed={failed} elapsed={time.time() - started_at:.1f}s",
        flush=True,
    )
    print(f"Generation results: {results_jsonl}")
    print(f"Merged metadata for eval.sh: {merged_metadata}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
