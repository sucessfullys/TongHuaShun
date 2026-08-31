import argparse
import csv
import json
import math
import os
import random
from json import JSONDecodeError
from typing import Any

import numpy as np
import torch
from PIL import Image

from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig


parser = argparse.ArgumentParser()
parser.add_argument("--jsonl", required=True)
parser.add_argument("--output", default="./exp_out_attention/debug")
parser.add_argument("--num", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--steps", type=int, default=4)
parser.add_argument("--cfg", type=float, default=1.0)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--lora", default=None)
parser.add_argument("--lora_alpha", type=float, default=1.0)
parser.add_argument("--offload", action="store_true")
parser.add_argument("--height", type=int, default=None)
parser.add_argument("--width", type=int, default=None)
parser.add_argument("--probe_steps", default="0,-1")
parser.add_argument("--double_blocks", default="0,4,7")
parser.add_argument("--single_blocks", default="0,12,24,47")
parser.add_argument("--query_chunk", type=int, default=32)
parser.add_argument("--no_heatmaps", action="store_true")
args = parser.parse_args()


def parse_int_list(spec: str) -> set[int]:
    if spec is None or spec.strip() == "":
        return set()
    return {int(part.strip()) for part in spec.split(",") if part.strip()}


def load_entries_compat(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return [data]
            raise ValueError(f"Unsupported JSON root type: {type(data)}")
        except JSONDecodeError:
            pass

    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
    return entries


def resize_with_padding(img, target_w, target_h, bg_color=(255, 255, 255)):
    img.thumbnail((target_w, target_h), Image.LANCZOS)
    padded = Image.new("RGB", (target_w, target_h), bg_color)
    offset_x = (target_w - img.width) // 2
    offset_y = (target_h - img.height) // 2
    padded.paste(img, (offset_x, offset_y))
    return padded


def offload_config(device):
    return dict(
        offload_dtype=torch.bfloat16,
        offload_device="cpu",
        onload_dtype=torch.bfloat16,
        onload_device=device,
        preparing_dtype=torch.bfloat16,
        preparing_device=device,
        computation_dtype=torch.bfloat16,
        computation_device=device,
    )


def build_pipeline(device):
    extra = offload_config(device) if args.offload else {}
    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(
                model_id="black-forest-labs/FLUX.2-klein-base-9B",
                origin_file_pattern="text_encoder/*.safetensors",
                **extra,
            ),
            ModelConfig(
                model_id="black-forest-labs/FLUX.2-klein-9B",
                origin_file_pattern="transformer/*.safetensors",
                **extra,
            ),
            ModelConfig(
                model_id="black-forest-labs/FLUX.2-klein-base-9B",
                origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
            ),
        ],
        tokenizer_config=ModelConfig(
            model_id="black-forest-labs/FLUX.2-klein-base-9B",
            origin_file_pattern="tokenizer/",
        ),
    )
    if args.lora:
        print(f"Loading LoRA: {args.lora} (alpha={args.lora_alpha})")
        pipe.load_lora(pipe.dit, args.lora, alpha=args.lora_alpha)
    return pipe


class Flux2AttentionProbe:
    def __init__(
        self,
        probe_steps: set[int],
        double_blocks: set[int],
        single_blocks: set[int],
        query_chunk: int = 32,
        save_heatmaps: bool = True,
    ):
        self.probe_steps = probe_steps
        self.double_blocks = double_blocks
        self.single_blocks = single_blocks
        self.query_chunk = query_chunk
        self.save_heatmaps = save_heatmaps
        self.metadata = {}
        self.token_region_rows = []
        self.summaries = []
        self.heatmaps = []

    def set_sample_metadata(self, metadata: dict[str, Any]) -> None:
        self.metadata = metadata

    def resolved_steps(self) -> set[int]:
        total_steps = int(self.metadata.get("num_inference_steps", 0) or 0)
        resolved = set()
        for step in self.probe_steps:
            resolved.add(total_steps + step if step < 0 else step)
        return resolved

    def should_capture(self, context: dict[str, Any]) -> bool:
        step = context.get("step")
        block_type = context.get("block_type")
        block_index = context.get("block_index")
        if step is None or step not in self.resolved_steps():
            return False
        if block_type == "double":
            return block_index in self.double_blocks
        if block_type == "single":
            return block_index in self.single_blocks
        return False

    def capture_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attention_mask: torch.Tensor | None,
        context: dict[str, Any],
    ) -> None:
        with torch.no_grad():
            q = query.detach()[0]
            k = key.detach()[0]
            if q.ndim != 3 or k.ndim != 3:
                return
            regions = self._regions(context, seq_len=q.shape[0])
            if not regions:
                return
            token_records = self._token_records(context)
            active_text_indices = [
                rec["index"]
                for rec in token_records
                if rec.get("active", False) and rec["index"] < context["text_len"]
            ]

            capture_id = {
                "step": context.get("step"),
                "block_type": context.get("block_type"),
                "block_index": context.get("block_index"),
            }
            self._capture_prompt_rows(q, k, attention_mask, active_text_indices, token_records, regions, capture_id)
            target_summary = self._capture_target_summary(q, k, attention_mask, regions, context, capture_id)
            self.summaries.append({
                **capture_id,
                "regions": [{"name": r["name"], "start": r["start"], "end": r["end"]} for r in regions],
                "target_to_regions": target_summary,
            })

    def _token_records(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        records = context.get("prompt_token_records") or self.metadata.get("prompt_token_records") or []
        if records and isinstance(records[0], list):
            return records[0]
        return records

    def _regions(self, context: dict[str, Any], seq_len: int) -> list[dict[str, Any]]:
        text_len = int(context["text_len"])
        target_len = int(context["target_len"])
        regions = [
            {"name": "text", "start": 0, "end": text_len},
            {"name": "target", "start": text_len, "end": text_len + target_len},
        ]
        edit_base = text_len + target_len
        for ref in context.get("edit_ref_ranges", []) or []:
            start = edit_base + int(ref["start"])
            end = edit_base + int(ref["end"])
            if start < seq_len:
                regions.append({
                    **ref,
                    "name": f"ref{ref['ref_index']}",
                    "start": start,
                    "end": min(end, seq_len),
                })
        return [region for region in regions if region["start"] < region["end"] <= seq_len]

    def _attention_chunk(self, q, k, query_indices, attention_mask):
        idx = torch.as_tensor(query_indices, device=q.device, dtype=torch.long)
        q_sel = q.index_select(0, idx).float()
        k = k.float()
        scores = torch.einsum("qhd,khd->hqk", q_sel, k) / math.sqrt(q_sel.shape[-1])
        if attention_mask is not None:
            mask = attention_mask.detach()
            if mask.ndim == 4:
                mask = mask[0, 0, idx, :].float()
                scores = scores + mask.unsqueeze(0)
        return torch.softmax(scores, dim=-1)

    def _capture_prompt_rows(self, q, k, attention_mask, query_indices, token_records, regions, capture_id):
        token_by_index = {rec["index"]: rec for rec in token_records}
        for start in range(0, len(query_indices), self.query_chunk):
            chunk = query_indices[start:start + self.query_chunk]
            attn = self._attention_chunk(q, k, chunk, attention_mask)
            for region in regions:
                mass = attn[:, :, region["start"]:region["end"]].sum(dim=-1).mean(dim=0)
                for local_idx, query_index in enumerate(chunk):
                    rec = token_by_index.get(query_index, {})
                    self.token_region_rows.append({
                        **capture_id,
                        "query_type": "prompt_token",
                        "token_index": int(query_index),
                        "token_id": rec.get("id"),
                        "token_text": rec.get("text", ""),
                        "region": region["name"],
                        "score": float(mass[local_idx].detach().cpu()),
                    })

    def _capture_target_summary(self, q, k, attention_mask, regions, context, capture_id):
        text_len = int(context["text_len"])
        target_len = int(context["target_len"])
        query_indices = list(range(text_len, text_len + target_len))
        region_sums = {region["name"]: 0.0 for region in regions}
        region_count = 0
        heatmap_sums = {
            region["name"]: torch.zeros(region["end"] - region["start"], dtype=torch.float64)
            for region in regions
            if region["name"].startswith("ref")
        }
        heatmap_count = 0

        for start in range(0, len(query_indices), self.query_chunk):
            chunk = query_indices[start:start + self.query_chunk]
            attn = self._attention_chunk(q, k, chunk, attention_mask)
            region_count += attn.shape[0] * attn.shape[1]
            heatmap_count += attn.shape[0] * attn.shape[1]
            for region in regions:
                region_attn = attn[:, :, region["start"]:region["end"]]
                region_sums[region["name"]] += float(region_attn.sum().detach().cpu())
                if region["name"] in heatmap_sums:
                    heatmap_sums[region["name"]] += region_attn.sum(dim=(0, 1)).double().cpu()

        target_summary = {
            name: value / max(region_count, 1)
            for name, value in region_sums.items()
        }
        if self.save_heatmaps:
            for region in regions:
                name = region["name"]
                if name not in heatmap_sums:
                    continue
                grid_h = int(region.get("grid_height", 0) or 0)
                grid_w = int(region.get("grid_width", 0) or 0)
                if grid_h * grid_w != heatmap_sums[name].numel():
                    continue
                heatmap = (heatmap_sums[name] / max(heatmap_count, 1)).reshape(grid_h, grid_w).numpy()
                self.heatmaps.append({
                    **capture_id,
                    "ref_index": int(region["ref_index"]),
                    "name": name,
                    "grid_height": grid_h,
                    "grid_width": grid_w,
                    "resized_width": int(region.get("resized_width", grid_w)),
                    "resized_height": int(region.get("resized_height", grid_h)),
                    "values": heatmap,
                })
        return target_summary

    def save(self, out_dir: str, input_images: list[Image.Image]) -> None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "prompt_tokens.json"), "w", encoding="utf-8") as f:
            json.dump(self.metadata.get("prompt_token_records", []), f, ensure_ascii=False, indent=2)
        summary = {
            "metadata": self.metadata,
            "captures": self.summaries,
        }
        with open(os.path.join(out_dir, "attention_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "token_ref_scores.csv"), "w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "step", "block_type", "block_index", "query_type",
                "token_index", "token_id", "token_text", "region", "score",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.token_region_rows)
        self._save_heatmaps(os.path.join(out_dir, "heatmaps"), input_images)

    def _save_heatmaps(self, heatmap_dir: str, input_images: list[Image.Image]) -> None:
        if not self.heatmaps:
            return
        os.makedirs(heatmap_dir, exist_ok=True)
        for item in self.heatmaps:
            values = item["values"]
            arr = values.astype(np.float32)
            arr = arr - float(arr.min())
            if float(arr.max()) > 0:
                arr = arr / float(arr.max())
            gray = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
            resized = gray.resize((item["resized_width"], item["resized_height"]), Image.BICUBIC)
            stem = f"step{item['step']}_{item['block_type']}_block{item['block_index']}_{item['name']}"
            resized.save(os.path.join(heatmap_dir, f"{stem}.png"))

            ref_idx = item["ref_index"]
            if 0 <= ref_idx < len(input_images):
                base = input_images[ref_idx].convert("RGB").resize(resized.size, Image.LANCZOS).convert("RGBA")
                heat = Image.new("RGBA", resized.size, (255, 0, 0, 0))
                heat.putalpha(resized.point(lambda x: int(x * 0.65)))
                Image.alpha_composite(base, heat).save(os.path.join(heatmap_dir, f"{stem}_overlay.png"))


def run_sample(entry, pipe, device, sample_index):
    uid = entry.get("uid") or f"sample_{sample_index:04d}"
    prompt = entry["prompt"]
    input_paths = entry["edit_image"]
    if isinstance(input_paths, str):
        input_paths = [input_paths]
    gt_path = entry.get("image", "")
    out_dir = os.path.join(args.output, uid)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[attention_probe] uid={uid}")
    print(f"  prompt: {prompt[:120]}...")
    images = [Image.open(path).convert("RGB") for path in input_paths]
    height = args.height if args.height is not None else images[0].height
    width = args.width if args.width is not None else images[0].width

    probe = Flux2AttentionProbe(
        probe_steps=parse_int_list(args.probe_steps),
        double_blocks=parse_int_list(args.double_blocks),
        single_blocks=parse_int_list(args.single_blocks),
        query_chunk=args.query_chunk,
        save_heatmaps=not args.no_heatmaps,
    )
    result = pipe(
        prompt,
        negative_prompt="",
        edit_image=images,
        edit_image_scale=1,
        s2_scale=0.0,
        seed=args.seed,
        rand_device=device,
        num_inference_steps=args.steps,
        cfg_scale=args.cfg,
        height=height,
        width=width,
        attention_probe=probe,
    )

    for idx, img in enumerate(images):
        img.save(os.path.join(out_dir, f"input_{idx}.webp"))
    result.save(os.path.join(out_dir, "result.png"))
    compare_images = images + [result]
    if gt_path and os.path.exists(gt_path):
        gt_img = Image.open(gt_path).convert("RGB")
        gt_img.save(os.path.join(out_dir, "gt.webp"))
        compare_images.append(gt_img)

    cell_w = max(img.width for img in compare_images)
    cell_h = max(img.height for img in compare_images)
    compare = Image.new("RGB", (cell_w * len(compare_images), cell_h), (255, 255, 255))
    for idx, img in enumerate(compare_images):
        compare.paste(resize_with_padding(img.copy(), cell_w, cell_h), (cell_w * idx, 0))
    compare.save(os.path.join(out_dir, "compare.webp"))
    probe.save(out_dir, images)
    print(f"  saved -> {out_dir}")


def main():
    os.makedirs(args.output, exist_ok=True)
    entries = load_entries_compat(args.jsonl)
    random.seed(args.seed)
    samples = random.sample(entries, min(args.num, len(entries)))
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Loading pipeline on {device}, sampled={len(samples)}")
    pipe = build_pipeline(device)
    for idx, entry in enumerate(samples):
        run_sample(entry, pipe, device, idx)
    print(f"\nDone. Results saved to {args.output}")


if __name__ == "__main__":
    main()
