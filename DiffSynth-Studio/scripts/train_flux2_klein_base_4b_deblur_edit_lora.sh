#!/usr/bin/env bash
set -euo pipefail

# Paired image-restoration LoRA for FLUX.2 Klein Base 4B.
#
# HR is the flow-matching target (`image`); the aligned LR image is supplied to
# the frozen Base model through its native `edit_image` conditioning path. Only
# LoRA adapters attached to the DiT are trainable. No Template model is loaded.
#
# Usage (all arguments are optional):
#   bash scripts/train_flux2_klein_base_4b_deblur_edit_lora.sh \
#     [HR_root] [metadata.jsonl] [output_dir]

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_BASE_PATH="${1:-/mnt/image-edit/datasets/duanyufa/Face/HR}"
DATASET_METADATA_PATH="${2:-/mnt/image-edit/datasets/duanyufa/Face/metadata_flux2_edit_lora.jsonl}"
OUTPUT_PATH="${3:-${REPO_ROOT}/outputs/FLUX.2_Base_deblur_LoRA}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
LORA_RANK="${LORA_RANK:-32}"
NUM_EPOCHS="${NUM_EPOCHS:-4}"
DATASET_REPEAT="${DATASET_REPEAT:-1}"
MAX_PIXELS="${MAX_PIXELS:-2097152}"
LOG_PATH="${LOG_PATH:-${OUTPUT_PATH}/train_edit_lora_8gpu.log}"
FLUX2_BASE_DIR="${FLUX2_BASE_DIR:-/mnt/image-edit/datasets/duanyufa/FLUX.2-klein-base-4B}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required offline file: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Missing required offline directory: $1" >&2
    exit 1
  fi
}

require_dir "${DATASET_BASE_PATH}"
require_file "${DATASET_METADATA_PATH}"
require_dir "${FLUX2_BASE_DIR}/text_encoder"
require_dir "${FLUX2_BASE_DIR}/tokenizer"
require_file "${FLUX2_BASE_DIR}/vae/diffusion_pytorch_model.safetensors"
require_file "${FLUX2_BASE_DIR}/transformer/diffusion_pytorch_model.safetensors"

# Build local-only model_paths JSON and verify the text encoder shards before
# Accelerate creates worker processes.
MODEL_PATHS="$({ python - "${FLUX2_BASE_DIR}" <<'PY'
import glob
import json
import os
import sys

base = sys.argv[1]
text_encoder = sorted(glob.glob(os.path.join(base, "text_encoder", "*.safetensors")))
if not text_encoder:
    raise SystemExit(f"No local text-encoder weights found in {base}/text_encoder")
print(json.dumps([
    text_encoder,
    os.path.join(base, "transformer", "diffusion_pytorch_model.safetensors"),
    os.path.join(base, "vae", "diffusion_pytorch_model.safetensors"),
]))
PY
} )"

# These are the same FLUX.2 Klein attention/FF projections used by the existing
# repository LoRA trainer. They cover all 5 double-stream and 20 single-stream
# transformer blocks while leaving VAE/text encoder/base weights frozen.
LORA_TARGET_MODULES="to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out,linear_in,linear_out,to_qkv_mlp_proj,single_transformer_blocks.0.attn.to_out,single_transformer_blocks.1.attn.to_out,single_transformer_blocks.2.attn.to_out,single_transformer_blocks.3.attn.to_out,single_transformer_blocks.4.attn.to_out,single_transformer_blocks.5.attn.to_out,single_transformer_blocks.6.attn.to_out,single_transformer_blocks.7.attn.to_out,single_transformer_blocks.8.attn.to_out,single_transformer_blocks.9.attn.to_out,single_transformer_blocks.10.attn.to_out,single_transformer_blocks.11.attn.to_out,single_transformer_blocks.12.attn.to_out,single_transformer_blocks.13.attn.to_out,single_transformer_blocks.14.attn.to_out,single_transformer_blocks.15.attn.to_out,single_transformer_blocks.16.attn.to_out,single_transformer_blocks.17.attn.to_out,single_transformer_blocks.18.attn.to_out,single_transformer_blocks.19.attn.to_out"

export DIFFSYNTH_SKIP_DOWNLOAD=true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MODELSCOPE_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "${OUTPUT_PATH}"
exec > >(tee -a "${LOG_PATH}") 2>&1
echo
echo "===== FLUX.2 edit-image LoRA launch $(date '+%Y-%m-%d %H:%M:%S') ====="
echo "Experiment: paired LR edit_image -> HR target"
echo "Template: disabled"
echo "Base weights: frozen"
echo "LoRA: DiT rank ${LORA_RANK}"
echo "Processes: ${NUM_PROCESSES}"
echo "Gradient accumulation: ${GRADIENT_ACCUMULATION_STEPS}"
echo "Global batch: $((NUM_PROCESSES * GRADIENT_ACCUMULATION_STEPS))"
echo "Dataset: ${DATASET_METADATA_PATH} repeat=${DATASET_REPEAT} epochs=${NUM_EPOCHS}"
echo "Output: ${OUTPUT_PATH}"
echo "Log: ${LOG_PATH}"

cd "${REPO_ROOT}"
accelerate launch \
  --multi_gpu \
  --num_machines 1 \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision "${MIXED_PRECISION}" \
  --dynamo_backend no \
  examples/flux2/model_training/train.py \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${DATASET_METADATA_PATH}" \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image,edit_image_auto_resize" \
  --max_pixels "${MAX_PIXELS}" \
  --dataset_repeat "${DATASET_REPEAT}" \
  --model_paths "${MODEL_PATHS}" \
  --tokenizer_path "${FLUX2_BASE_DIR}/tokenizer" \
  --learning_rate "${LEARNING_RATE}" \
  --num_epochs "${NUM_EPOCHS}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --lora_base_model "dit" \
  --lora_target_modules "${LORA_TARGET_MODULES}" \
  --lora_rank "${LORA_RANK}" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_PATH}" \
  --log_every 1 \
  --use_gradient_checkpointing
