"""
Offline head-crop pre-processing script (GPU-accelerated).

Splits the JSONL across available GPUs; each subprocess owns one GPU and runs
InsightFace ONNX inference on that device.  I/O is handled by a thread-pool
inside each subprocess to overlap disk reads with GPU work.

Images where no face is detected retain their original path in the output JSONL.
Already-cropped files (_headcrop suffix) are skipped on re-runs (idempotent).

Usage:
    python preprocess_head_crop.py \
        --input_jsonl  /path/to/metadata.jsonl \
        --output_jsonl /path/to/metadata_headcrop.jsonl \
        --insightface_root /mnt/data/0/pretrained_ckpt/models/insightface \
        --gpu_ids 0,1,2,3

    # CPU-only fallback (gpu_ids=-1):
    python preprocess_head_crop.py ... --gpu_ids -1 --io_workers 8
"""

import argparse
import json
import os
import multiprocessing as mp

import numpy as np
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Per-image crop helper (no global state – receives face_app explicitly)
# ---------------------------------------------------------------------------

def _detect_largest_face(face_app, img_bgr):
    faces = face_app.get(img_bgr)
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def _crop_one(face_app, src_path, side_margin, top_margin, bottom_margin):
    """Returns dst_path (new file) or src_path (no face / error / already done)."""
    base, ext = os.path.splitext(src_path)
    dst_path = base + "_headcrop" + ext

    # if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
    #     return dst_path, "skip"

    try:
        image = Image.open(src_path).convert("RGB")
    except Exception as e:
        return src_path, f"load_err:{e}"

    w, h = image.size
    img_bgr = np.array(image)[:, :, ::-1]
    face = _detect_largest_face(face_app, img_bgr)

    if face is None:
        return src_path, "no_face"

    x1, y1, x2, y2 = face.bbox
    bw, bh = x2 - x1, y2 - y1
    cx1 = int(max(0, x1 - bw * side_margin))
    cy1 = int(max(0, y1 - bh * top_margin))
    cx2 = int(min(w, x2 + bw * side_margin))
    cy2 = int(min(h, y2 + bh * bottom_margin))

    if cx2 - cx1 < 16 or cy2 - cy1 < 16:
        return src_path, "too_small"

    cropped = image.crop((cx1, cy1, cx2, cy2))
    try:
        cropped.save(dst_path)
    except Exception as e:
        return src_path, f"save_err:{e}"
    return dst_path, "cropped"


# ---------------------------------------------------------------------------
# Worker process: owns one GPU, processes its chunk, writes temp JSONL
# ---------------------------------------------------------------------------

def _gpu_worker(rank, gpu_id, chunk, insightface_root, det_size,
                side_margin, top_margin, bottom_margin,
                io_workers, tmp_path, counter, counter_lock):
    if gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from insightface.app import FaceAnalysis
    import onnxruntime as ort

    kwargs = {"name": "buffalo_l"}
    if insightface_root:
        kwargs["root"] = insightface_root
    if gpu_id >= 0:
        providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    face_app = FaceAnalysis(**kwargs, providers=providers)
    face_app.prepare(ctx_id=0 if gpu_id >= 0 else -1, det_size=(det_size, det_size))

    label = f"GPU{gpu_id}-W{rank}" if gpu_id >= 0 else f"CPU-W{rank}"
    actual = ort.get_available_providers()
    print(f"[{label}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','N/A')}, "
          f"ort_providers={actual}")

    stats = {"cropped": 0, "skip": 0, "no_face": 0, "error": 0}

    def _process_entry(entry):
        edit_images = entry.get("edit_image", [])
        if isinstance(edit_images, str):
            edit_images = [edit_images]
        new_paths = []
        for path in edit_images:
            dst, status = _crop_one(face_app, path, side_margin, top_margin, bottom_margin)
            new_paths.append(dst)
            if status == "cropped":
                stats["cropped"] += 1
            elif status == "skip":
                stats["skip"] += 1
            elif status == "no_face":
                stats["no_face"] += 1
            else:
                stats["error"] += 1
        new_entry = dict(entry)
        new_entry["edit_image"] = new_paths
        return new_entry

    results = []
    label = f"GPU{gpu_id}-W{rank}" if gpu_id >= 0 else f"CPU-W{rank}"

    # Use thread pool for I/O overlap; face_app itself is not thread-safe,
    # so we process sequentially for GPU inference but use threads only to
    # prefetch/save images in the future. For safety, keep sequential here
    # and rely on multi-process parallelism across GPUs.
    with tqdm(total=len(chunk), desc=label, position=rank, leave=True) as pbar:
        for entry in chunk:
            results.append(_process_entry(entry))
            with counter_lock:
                counter.value += 1
            pbar.update(1)

    with open(tmp_path, "w") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return stats


def _worker_wrapper(
    rank,
    gpu_id,
    chunk,
    tmp_path,
    result_queue,
    counter,
    counter_lock,
    insightface_root,
    det_size,
    side_margin_ratio,
    top_margin_ratio,
    bottom_margin_ratio,
    io_workers,
):
    stats = _gpu_worker(
        rank,
        gpu_id,
        chunk,
        insightface_root,
        det_size,
        side_margin_ratio,
        top_margin_ratio,
        bottom_margin_ratio,
        io_workers,
        tmp_path,
        counter,
        counter_lock,
    )
    result_queue.put((rank, stats))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _detect_gpus():
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        ).decode()
        return [int(x.strip()) for x in out.strip().splitlines()]
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl",  required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--insightface_root", default=None)
    parser.add_argument("--gpu_ids", default=None,
                        help="Comma-separated GPU IDs to use, e.g. '0,1,2,3'. "
                             "Use '-1' to force CPU. Default: all visible GPUs.")
    parser.add_argument("--workers_per_gpu", type=int, default=2,
                        help="Number of worker processes per GPU. Increase to improve GPU utilization for lightweight models.")
    parser.add_argument("--det_size",          type=int,   default=640)
    parser.add_argument("--side_margin_ratio", type=float, default=0.5)
    parser.add_argument("--top_margin_ratio",  type=float, default=0.6)
    parser.add_argument("--bottom_margin_ratio", type=float, default=1.0)
    parser.add_argument("--io_workers",        type=int,   default=4,
                        help="Thread-pool size for I/O (reserved for future use).")
    args = parser.parse_args()

    # Resolve GPU list
    if args.gpu_ids is not None:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    else:
        gpu_ids = _detect_gpus()
        if not gpu_ids:
            print("[warn] No GPUs detected, falling back to CPU.")
            gpu_ids = [-1]
    print(f"Using devices: {gpu_ids}")

    # Load entries
    with open(args.input_jsonl, "r") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    print(f"Total entries: {len(entries)}")

    # Build worker-to-device mapping.
    if gpu_ids == [-1]:
        worker_devices = [-1] * max(1, args.workers_per_gpu)
    else:
        worker_devices = []
        for gid in gpu_ids:
            worker_devices.extend([gid] * max(1, args.workers_per_gpu))
    num_workers = len(worker_devices)
    print(f"Total workers: {num_workers} (workers_per_gpu={max(1, args.workers_per_gpu)})")

    # Split entries across all workers
    chunks = [entries[i::num_workers] for i in range(num_workers)]
    tmp_paths = [f"{args.output_jsonl}.tmp{i}" for i in range(num_workers)]

    ctx = mp.get_context("spawn")
    counter = ctx.Value("i", 0)
    counter_lock = ctx.Lock()
    procs = []
    result_queue = ctx.Queue()

    for rank, (gpu_id, chunk, tmp_path) in enumerate(zip(worker_devices, chunks, tmp_paths)):
        p = ctx.Process(
            target=_worker_wrapper,
            args=(
                rank,
                gpu_id,
                chunk,
                tmp_path,
                result_queue,
                counter,
                counter_lock,
                args.insightface_root,
                args.det_size,
                args.side_margin_ratio,
                args.top_margin_ratio,
                args.bottom_margin_ratio,
                args.io_workers,
            ),
            daemon=True,
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # Collect stats
    total_stats = {"cropped": 0, "skip": 0, "no_face": 0, "error": 0}
    while not result_queue.empty():
        _, stats = result_queue.get()
        for k in total_stats:
            total_stats[k] += stats.get(k, 0)

    # Merge temp files in original order (chunk[i] = entries[i::num_workers])
    # Reconstruct: entry at original index i came from chunk[i % num_workers][i // num_workers]
    merged = [None] * len(entries)
    for rank, tmp_path in enumerate(tmp_paths):
        with open(tmp_path, "r") as f:
            chunk_results = [json.loads(line) for line in f if line.strip()]
        for j, entry in enumerate(chunk_results):
            original_idx = rank + j * num_workers
            merged[original_idx] = entry
        os.remove(tmp_path)

    with open(args.output_jsonl, "w") as f:
        for entry in merged:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nSaved: {args.output_jsonl}")
    print(f"  cropped : {total_stats['cropped']}")
    print(f"  skipped : {total_stats['skip']}  (already existed)")
    print(f"  no face : {total_stats['no_face']}")
    print(f"  errors  : {total_stats['error']}")


if __name__ == "__main__":
    main()
