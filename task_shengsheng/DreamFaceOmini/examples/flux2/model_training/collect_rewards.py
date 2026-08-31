"""
collect_rewards.py — Data collection phase for DiffusionNFT RL.

For each training sample: generate N images with the current model,
score each with ArcFace + Aesthetic rewards, output reward JSONL.

Usage:
    python collect_rewards.py \
        --input_jsonl /path/to/train.jsonl \
        --output_jsonl /path/to/rewards.jsonl \
        --lora /path/to/lora.safetensors \
        --num_images_per_prompt 4 \
        --gpus 1,2,3,4,5,6,7 \
        --arcface_ckpt /path/to/arcface.pth \
        --aesthetic_ckpt /path/to/aesthetic.pth
"""

import json, os, random, argparse, time
import multiprocessing as mp
import torch
import numpy as np
from PIL import Image
from json import JSONDecodeError


def load_entries(path):
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except JSONDecodeError:
                    pass
    return entries


def parse_gpu_ids(gpu_arg):
    if gpu_arg:
        return [int(g.strip()) for g in gpu_arg.split(",") if g.strip()]
    return list(range(torch.cuda.device_count()))


def build_pipeline(device, lora_path, lora_alpha):
    from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id="black-forest-labs/FLUX.2-klein-base-9B",
                        origin_file_pattern="text_encoder/*.safetensors"),
            ModelConfig(model_id="black-forest-labs/FLUX.2-klein-base-9B",
                        origin_file_pattern="transformer/*.safetensors"),
            ModelConfig(model_id="black-forest-labs/FLUX.2-klein-base-9B",
                        origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        tokenizer_config=ModelConfig(model_id="black-forest-labs/FLUX.2-klein-base-9B",
                                     origin_file_pattern="tokenizer/"),
    )
    if lora_path:
        print(f"[collect] Loading LoRA: {lora_path} (alpha={lora_alpha})")
        pipe.load_lora(pipe.dit, lora_path, alpha=lora_alpha)
    return pipe


def build_rewards(device, arcface_ckpt, insightface_root, aesthetic_ckpt,
                  arcface_weight, aesthetic_weight):
    from rewards import ArcFaceReward, AestheticReward, CompositeReward
    reward_fns = {}
    if arcface_ckpt and arcface_weight > 0:
        reward_fns["arcface"] = (
            ArcFaceReward(arcface_ckpt, insightface_root, device=device),
            arcface_weight,
        )
    if aesthetic_ckpt and aesthetic_weight > 0:
        reward_fns["aesthetic"] = (
            AestheticReward(aesthetic_ckpt=aesthetic_ckpt, device=device),
            aesthetic_weight,
        )
    return CompositeReward(reward_fns)


def worker_main(gpu_id, worker_idx, num_workers, entries, args, output_path):
    device = f"cuda:{gpu_id}"
    print(f"[worker {worker_idx}/{num_workers}] device={device}, samples={len(entries)}")

    pipe = build_pipeline(device, args.lora, args.lora_alpha)
    reward_fn = build_rewards(
        device, args.arcface_ckpt, args.insightface_root,
        args.aesthetic_ckpt, args.arcface_weight, args.aesthetic_weight,
    )

    results = []
    for ei, entry in enumerate(entries):
        prompt = entry["prompt"]
        edit_image_paths = entry.get("edit_image", [])
        if isinstance(edit_image_paths, str):
            edit_image_paths = [edit_image_paths]
        gt_image_path = entry.get("image", "")
        uid = entry.get("uid", f"w{worker_idx}_{ei}")

        try:
            edit_images = [Image.open(p).convert("RGB") for p in edit_image_paths]
        except Exception as e:
            print(f"  [SKIP] uid={uid}: cannot load edit_image: {e}")
            continue

        ref_image = edit_images[0] if edit_images else None

        for gen_idx in range(args.num_images_per_prompt):
            seed = args.seed_base + ei * args.num_images_per_prompt + gen_idx
            try:
                gen_img = pipe(
                    prompt,
                    negative_prompt="",
                    edit_image=edit_images,
                    edit_image_scale=1.0,
                    seed=seed,
                    rand_device=device,
                    num_inference_steps=args.steps,
                    cfg_scale=args.cfg,
                    height=args.height,
                    width=args.width,
                )
            except Exception as e:
                print(f"  [ERROR] uid={uid} gen={gen_idx}: {e}")
                continue

            gen_path = os.path.join(output_path, "images", f"{uid}_gen{gen_idx}.png")
            os.makedirs(os.path.dirname(gen_path), exist_ok=True)
            gen_img.save(gen_path)

            scores = reward_fn.score([gen_img], [ref_image], [prompt])

            record = {
                "uid": uid,
                "prompt": prompt,
                "edit_image": edit_image_paths,
                "gt_image": gt_image_path,
                "generated_image": gen_path,
                "seed": seed,
                "gen_idx": gen_idx,
            }
            for k, v in scores.items():
                record[f"reward_{k}"] = v[0] if isinstance(v, list) else v
            results.append(record)

        if (ei + 1) % 10 == 0:
            print(f"  [worker {worker_idx}] {ei+1}/{len(entries)} done")

    worker_jsonl = os.path.join(output_path, f"rewards_worker{worker_idx}.jsonl")
    with open(worker_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[worker {worker_idx}] saved {len(results)} records -> {worker_jsonl}")


def compute_advantages(input_path: str, output_path: str, adv_clip_max: float = 5.0):
    """Per-prompt z-score normalization of composite reward → advantage."""
    records = load_entries(input_path)
    if not records:
        return

    from collections import defaultdict
    by_prompt = defaultdict(list)
    for r in records:
        by_prompt[r["uid"]].append(r)

    all_rewards = [r["reward_composite"] for r in records]
    global_std = max(np.std(all_rewards), 1e-6)

    for uid, group in by_prompt.items():
        rewards = [r["reward_composite"] for r in group]
        mean = np.mean(rewards)
        std = max(np.std(rewards), 1e-6)
        for r in group:
            adv = (r["reward_composite"] - mean) / std
            r["advantage"] = float(np.clip(adv, -adv_clip_max, adv_clip_max))

    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[advantage] {len(records)} records with advantages -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="DiffusionNFT data collection: generate + score")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lora", default=None)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--num_images_per_prompt", type=int, default=4)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--height", type=int, default=1152)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--seed_base", type=int, default=1000)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of input samples (for testing)")
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--arcface_ckpt", default=None)
    parser.add_argument("--insightface_root", default=None)
    parser.add_argument("--aesthetic_ckpt", default=None)
    parser.add_argument("--arcface_weight", type=float, default=0.7)
    parser.add_argument("--aesthetic_weight", type=float, default=0.3)
    parser.add_argument("--adv_clip_max", type=float, default=5.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    entries = load_entries(args.input_jsonl)
    if args.max_samples:
        random.seed(42)
        entries = random.sample(entries, min(args.max_samples, len(entries)))
    print(f"[collect] {len(entries)} samples, {args.num_images_per_prompt} images each")

    gpu_ids = parse_gpu_ids(args.gpus)
    chunks = [[] for _ in range(len(gpu_ids))]
    for i, e in enumerate(entries):
        chunks[i % len(gpu_ids)].append(e)
    chunks = [c for c in chunks if c]
    active_gpus = gpu_ids[:len(chunks)]

    if len(active_gpus) == 1:
        worker_main(active_gpus[0], 0, 1, chunks[0], args, args.output_dir)
    else:
        ctx = mp.get_context("spawn")
        procs = []
        for wi, (gid, chunk) in enumerate(zip(active_gpus, chunks)):
            p = ctx.Process(target=worker_main,
                            args=(gid, wi, len(active_gpus), chunk, args, args.output_dir))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Worker failed with exit code {p.exitcode}")

    merged_path = os.path.join(args.output_dir, "rewards_merged.jsonl")
    with open(merged_path, "w", encoding="utf-8") as out:
        for wi in range(len(active_gpus)):
            wp = os.path.join(args.output_dir, f"rewards_worker{wi}.jsonl")
            if os.path.exists(wp):
                with open(wp) as f:
                    out.write(f.read())
    print(f"[collect] merged -> {merged_path}")

    final_path = os.path.join(args.output_dir, "rewards_with_advantage.jsonl")
    compute_advantages(merged_path, final_path, args.adv_clip_max)


if __name__ == "__main__":
    main()
