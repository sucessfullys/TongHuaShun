#!/usr/bin/env python3
"""Ivy-Fake: AI图像真伪检测 —— 底模 + Ivy-Fake checkpoint 合并推理。

底模提供完整架构（lm_head + visual），checkpoint 覆盖微调过的 transformer 层。
"""

import argparse
import json
import torch
import safetensors.torch as st
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from huggingface_hub import hf_hub_download

# Ivy-Fake checkpoint 的 hidden_size=2048、36 layers，对应 Qwen2.5-VL-3B 规格。
# 不要用 7B：7B hidden size 不匹配，会导致 checkpoint 无法加载。
BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
CKPT_REPO = "AI-Safeguard/Ivy-Fake"
MODEL = None
PROCESSOR = None

SYSTEM_PROMPT = (
    "You are an AI-generated content detector. Classify the media as real or fake. "
    "Provide reasoning inside <think>...</think> tags. "
    "End with exactly one word—real or fake—wrapped in <conclusion>...</conclusion>."
)


def remap_ivyfake_key(key):
    """Map Ivy-Fake old-style Qwen2.5-VL keys to current transformers keys."""
    if key.startswith("visual."):
        return "model." + key
    if key.startswith("model."):
        # Old checkpoints store language model keys as model.layers.* / model.embed_tokens.*.
        # Current transformers stores them as model.language_model.layers.*.
        if key.startswith("model.language_model.") or key.startswith("model.visual."):
            return key
        return "model.language_model." + key[len("model."):]
    return key


def load_ivyfake_checkpoint(model, ckpt_repo):
    idx_path = hf_hub_download(ckpt_repo, "model.safetensors.index.json")
    with open(idx_path, encoding="utf-8") as f:
        idx = json.load(f)

    target_state = model.state_dict()
    remapped_sd = {}
    skipped_shape = []
    skipped_name = []

    shard_files = sorted(set(idx["weight_map"].values()))
    for shard_file in shard_files:
        shard_path = hf_hub_download(ckpt_repo, shard_file)
        shard_sd = st.load_file(shard_path)
        for old_key, tensor in shard_sd.items():
            new_key = remap_ivyfake_key(old_key)
            if new_key not in target_state:
                skipped_name.append((old_key, new_key, tuple(tensor.shape)))
                continue
            if tuple(target_state[new_key].shape) != tuple(tensor.shape):
                skipped_shape.append(
                    (old_key, new_key, tuple(tensor.shape), tuple(target_state[new_key].shape))
                )
                continue
            remapped_sd[new_key] = tensor

    missing, unexpected = model.load_state_dict(remapped_sd, strict=False)
    print(f"  Checkpoint tensors loaded: {len(remapped_sd)}")
    print(f"  Missing after partial load (OK, from base): {len(missing)}")
    print(f"  Unexpected after remap: {len(unexpected)}")
    if skipped_name:
        print(f"  Skipped by name: {len(skipped_name)}")
        for item in skipped_name[:5]:
            print(f"    name: {item[0]} -> {item[1]} shape={item[2]}")
    if skipped_shape:
        print(f"  Skipped by shape: {len(skipped_shape)}")
        for item in skipped_shape[:5]:
            print(f"    shape: {item[0]} -> {item[1]} ckpt={item[2]} base={item[3]}")
    if len(remapped_sd) == 0:
        raise RuntimeError(
            "No Ivy-Fake checkpoint tensors were loaded. "
            "Check base model size and key mapping."
        )


def get_model(base_model=BASE_MODEL, ckpt_repo=CKPT_REPO):
    global MODEL, PROCESSOR
    if MODEL is None:
        attn = "sdpa"
        print(f"Loading base model: {base_model} ...")
        MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model, torch_dtype=torch.bfloat16,
            attn_implementation=attn, device_map="auto",
        )

        print(f"Loading checkpoint: {ckpt_repo} ...")
        load_ivyfake_checkpoint(MODEL, ckpt_repo)

        PROCESSOR = AutoProcessor.from_pretrained(base_model)
        MODEL.eval()
        print("Model ready.")
    return MODEL, PROCESSOR


def detect(image_path, base_model=BASE_MODEL, ckpt_repo=CKPT_REPO, max_new_tokens=2048):
    model, processor = get_model(base_model, ckpt_repo)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": "Is this image real or fake?"},
        ]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = inputs.to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--ckpt-repo", default=CKPT_REPO)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    args = p.parse_args()

    result = detect(args.input, args.base_model, args.ckpt_repo, args.max_new_tokens)
    print("\n" + "=" * 50)
    print(result)
    print("=" * 50)


if __name__ == "__main__":
    main()
