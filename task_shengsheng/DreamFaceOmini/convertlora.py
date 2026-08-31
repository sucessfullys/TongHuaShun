# from safetensors.torch import load_file, save_file
# import os

# src = '/mnt/data/image-edit/datasets/shensheng/code/stable/DreamFaceOmini/models/nft/dreamface_nft_vlm_gt_identity_gemma4/checkpoints/checkpoint-80/adapter_model.safetensors'
# dst = '/mnt/data/image-edit/datasets/shensheng/code/stable/DreamFaceOmini/models/nft/dreamface_nft_vlm_gt_identity_gemma4/checkpoints/checkpoint-80/adapter_diffsynth_model.safetensors'

# d = load_file(src)
# new_d = {}
# stripped = 0
# for k, v in d.items():
#     if k.startswith('base_model.model.'):
#         new_k = k[len('base_model.model.'):]
#         stripped += 1
#     else:
#         new_k = k
#     new_d[new_k] = v

# save_file(new_d, dst)
# print(f'Done. {stripped}/{len(d)} keys stripped.')
# print('Sample keys:')
# for k in list(new_d.keys())[:5]:
#     print(' ', k)

#!/usr/bin/env python3
import argparse
import json
import os
import re
from collections import OrderedDict

import torch
from safetensors.torch import load_file, save_file


DEFAULT_INPUT_DIR = "/mnt/data/image-edit/datasets/bxh/Flow-Factory/saves/flux2-klein_lora_grpo_20260624_154256/checkpoints/checkpoint-20"
PEFT_PREFIXES = ("base_model.model.", "model.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a PEFT Flux2 LoRA checkpoint for diffusers and DiffSynth-Studio."
    )
    parser.add_argument("--input_dir", default=DEFAULT_INPUT_DIR, help="Directory containing adapter_model.safetensors.")
    parser.add_argument("--output_dir", default=None, help="Directory to save converted LoRA files. Defaults to input_dir.")
    parser.add_argument("--adapter_name", default="adapter_model.safetensors", help="PEFT adapter safetensors file name.")
    parser.add_argument("--diffusers_name", default="diffusers_lora.safetensors", help="Output file name for diffusers.")
    parser.add_argument("--diffsynth_name", default="diffsynth_lora.safetensors", help="Output file name for DiffSynth-Studio.")
    parser.add_argument("--comfyui_name", default="comfyui_lora.safetensors", help="Output file name for ComfyUI 0.9.2.")
    parser.add_argument("--strict", action="store_true", help="Fail on keys that are not recognized PEFT LoRA keys.")
    return parser.parse_args()


def strip_peft_prefix(key):
    for prefix in PEFT_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):]
    return None


def is_lora_weight_key(key):
    return key.endswith(".lora_A.weight") or key.endswith(".lora_B.weight")


def read_lora_alpha(input_dir):
    config_path = os.path.join(input_dir, "adapter_config.json")
    if not os.path.isfile(config_path):
        return None

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    lora_alpha = config.get("lora_alpha")
    if lora_alpha is None:
        return None
    return float(lora_alpha)


def split_lora_suffix(key):
    for suffix in (".lora_A.weight", ".lora_B.weight"):
        if key.endswith(suffix):
            return key[: -len(suffix)], suffix
    return key, ""


def to_comfyui_092_key(diffusers_key):
    stem, suffix = split_lora_suffix(diffusers_key)
    normalized_stem = stem
    if normalized_stem.startswith("transformer."):
        normalized_stem = normalized_stem[len("transformer."):]

    single_qkv_mlp = re.fullmatch(
        r"single_transformer_blocks\.(\d+)\.attn\.to_qkv_mlp_proj",
        normalized_stem,
    )
    if single_qkv_mlp:
        block = single_qkv_mlp.group(1)
        return f"diffusion_model.single_blocks.{block}.linear1{suffix}"

    single_out = re.fullmatch(
        r"single_transformer_blocks\.(\d+)\.attn\.to_out(?:\.0)?",
        normalized_stem,
    )
    if single_out:
        block = single_out.group(1)
        return f"diffusion_model.single_blocks.{block}.linear2{suffix}"

    double_mappings = (
        (r"transformer_blocks\.(\d+)\.ff\.linear_in", "diffusion_model.double_blocks.{block}.img_mlp.0"),
        (r"transformer_blocks\.(\d+)\.ff\.linear_out", "diffusion_model.double_blocks.{block}.img_mlp.2"),
        (r"transformer_blocks\.(\d+)\.ff_context\.linear_in", "diffusion_model.double_blocks.{block}.txt_mlp.0"),
        (r"transformer_blocks\.(\d+)\.ff_context\.linear_out", "diffusion_model.double_blocks.{block}.txt_mlp.2"),
        (r"transformer_blocks\.(\d+)\.attn\.to_out(?:\.0)?", "diffusion_model.double_blocks.{block}.img_attn.proj"),
        (r"transformer_blocks\.(\d+)\.attn\.to_add_out", "diffusion_model.double_blocks.{block}.txt_attn.proj"),
    )
    for pattern, replacement in double_mappings:
        match = re.fullmatch(pattern, normalized_stem)
        if match:
            return f"{replacement.format(block=match.group(1))}{suffix}"

    qkv_pattern = (
        r"transformer_blocks\.\d+\.attn\."
        r"(?:to_q|to_k|to_v|add_q_proj|add_k_proj|add_v_proj)"
    )
    if re.fullmatch(qkv_pattern, normalized_stem):
        return diffusers_key

    return diffusers_key


def add_alpha_keys(state_dict, lora_alpha):
    if lora_alpha is None:
        return

    alpha_tensor = torch.tensor(float(lora_alpha), dtype=torch.float32)
    for key in list(state_dict.keys()):
        stem, suffix = split_lora_suffix(key)
        if suffix:
            state_dict[f"{stem}.alpha"] = alpha_tensor.clone()


def convert_state_dict(state_dict, lora_alpha=None, strict=False):
    diffusers_state_dict = OrderedDict()
    diffsynth_state_dict = OrderedDict()
    comfyui_state_dict = OrderedDict()
    skipped_keys = []

    for key in sorted(state_dict):
        if not is_lora_weight_key(key):
            skipped_keys.append(key)
            continue

        normalized_key = strip_peft_prefix(key)
        if normalized_key is None:
            skipped_keys.append(key)
            continue

        diffusers_key = normalized_key
        if not diffusers_key.startswith("transformer."):
            diffusers_key = "transformer." + diffusers_key

        diffsynth_key = diffusers_key
        if diffsynth_key.startswith("transformer."):
            diffsynth_key = diffsynth_key[len("transformer."):]

        diffusers_state_dict[diffusers_key] = state_dict[key]
        diffsynth_state_dict[diffsynth_key] = state_dict[key]
        comfyui_state_dict[to_comfyui_092_key(diffusers_key)] = state_dict[key]

    if strict and skipped_keys:
        preview = "\n".join(skipped_keys[:10])
        raise ValueError(f"Found {len(skipped_keys)} unsupported keys:\n{preview}")
    if not diffusers_state_dict or not diffsynth_state_dict or not comfyui_state_dict:
        raise ValueError("No LoRA weights were converted. Please check the input checkpoint format.")

    add_alpha_keys(comfyui_state_dict, lora_alpha)
    return diffusers_state_dict, diffsynth_state_dict, comfyui_state_dict, skipped_keys


def save_converted_files(
    diffusers_state_dict,
    diffsynth_state_dict,
    comfyui_state_dict,
    output_dir,
    diffusers_name,
    diffsynth_name,
    comfyui_name,
):
    os.makedirs(output_dir, exist_ok=True)
    diffusers_path = os.path.join(output_dir, diffusers_name)
    diffsynth_path = os.path.join(output_dir, diffsynth_name)
    comfyui_path = os.path.join(output_dir, comfyui_name)

    save_file(diffusers_state_dict, diffusers_path)
    save_file(diffsynth_state_dict, diffsynth_path)
    save_file(comfyui_state_dict, comfyui_path)
    return diffusers_path, diffsynth_path, comfyui_path


def print_summary(name, path, state_dict):
    print(f"{name}: {path}")
    print(f"  keys: {len(state_dict)}")
    for key in list(state_dict.keys())[:5]:
        tensor = state_dict[key]
        print(f"  {key}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")



def main():
    args = parse_args()
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir or args.input_dir)
    adapter_path = os.path.join(input_dir, args.adapter_name)

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not os.path.isfile(adapter_path):
        raise FileNotFoundError(f"Adapter file does not exist: {adapter_path}")

    lora_alpha = read_lora_alpha(input_dir)
    state_dict = load_file(adapter_path, device="cpu")
    diffusers_state_dict, diffsynth_state_dict, comfyui_state_dict, skipped_keys = convert_state_dict(
        state_dict,
        lora_alpha=lora_alpha,
        strict=args.strict,
    )
    diffusers_path, diffsynth_path, comfyui_path = save_converted_files(
        diffusers_state_dict,
        diffsynth_state_dict,
        comfyui_state_dict,
        output_dir,
        args.diffusers_name,
        args.diffsynth_name,
        args.comfyui_name,
    )

    print(f"Input: {adapter_path}")
    print(f"Input keys: {len(state_dict)}")
    print(f"LoRA alpha: {lora_alpha}")
    print(f"Skipped keys: {len(skipped_keys)}")
    print_summary("Diffusers LoRA", diffusers_path, diffusers_state_dict)
    print_summary("DiffSynth-Studio LoRA", diffsynth_path, diffsynth_state_dict)
    print_summary("ComfyUI 0.9.2 LoRA", comfyui_path, comfyui_state_dict)


if __name__ == "__main__":
    main()