#!/usr/bin/env bash
set -euo pipefail

# Cached native edit_image LoRA checkpoint comparison.
#
# Training/inference consistency:
#   - Uses FLUX.2 native edit_image condition.
#   - Keeps edit_image_auto_resize disabled.
#   - Caches VAE(edit_image) -> edit_latents/edit_image_ids once per process.
#   - Loops checkpoints by hotloading LoRA weights.

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/outputs/FLUX2_KleinBase4B_Deblur_all_edit_lora_lr1e-5_r32_rep1_ep2_4GPU_px2097152_noauto_20260817_213623}"
METADATA="${METADATA:-/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/test_all/测试集合/metadata.jsonl}"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/test_all}"
NUM_GPUS="${NUM_GPUS:-4}"
LIMIT="${LIMIT:-}"
SEED="${SEED:-42}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
CFG_SCALE="${CFG_SCALE:-4.0}"
EMBEDDED_GUIDANCE="${EMBEDDED_GUIDANCE:-4.0}"
LORA_SCALE="${LORA_SCALE:-1.0}"
CACHE_DEVICE="${CACHE_DEVICE:-cpu}"
CHECKPOINT_PATTERN="${CHECKPOINT_PATTERN:-*.safetensors}"
CHECKPOINT_NAMES="${CHECKPOINT_NAMES:-}"
OVERWRITE="${OVERWRITE:-true}"

SCRIPT="/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/test_all/compare_checkpoints_edit_lora_cached.py"

cd /mnt/image-edit/datasets/duanyufa/DiffSynth-Studio

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "Missing checkpoint directory: ${CHECKPOINT_DIR}" >&2
  exit 1
fi
if [[ ! -f "${METADATA}" ]]; then
  echo "Missing metadata file: ${METADATA}" >&2
  exit 1
fi
if [[ ! -f "${SCRIPT}" ]]; then
  echo "Missing script: ${SCRIPT}" >&2
  exit 1
fi

ARGS=(
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --metadata "${METADATA}"
  --output-base "${OUTPUT_BASE}"
  --checkpoint-pattern "${CHECKPOINT_PATTERN}"
  --num-inference-steps "${NUM_INFERENCE_STEPS}"
  --cfg-scale "${CFG_SCALE}"
  --embedded-guidance "${EMBEDDED_GUIDANCE}"
  --seed "${SEED}"
  --lora-scale "${LORA_SCALE}"
  --cache-device "${CACHE_DEVICE}"
)

if [[ -n "${LIMIT}" ]]; then
  ARGS+=(--limit "${LIMIT}")
fi
if [[ -n "${CHECKPOINT_NAMES}" ]]; then
  ARGS+=(--checkpoint-names "${CHECKPOINT_NAMES}")
fi
if [[ "${OVERWRITE}" == "true" ]]; then
  ARGS+=(--overwrite)
fi

echo "=========================================="
echo "  Cached edit_image LoRA checkpoint comparison launch"
echo "  CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "  METADATA=${METADATA}"
echo "  OUTPUT_BASE=${OUTPUT_BASE}"
echo "  NUM_GPUS=${NUM_GPUS}"
echo "  LIMIT=${LIMIT:-all}"
echo "  CHECKPOINT_PATTERN=${CHECKPOINT_PATTERN}"
echo "  CHECKPOINT_NAMES=${CHECKPOINT_NAMES:-all}"
echo "  LORA_SCALE=${LORA_SCALE}"
echo "  CACHE_DEVICE=${CACHE_DEVICE}"
echo "=========================================="

accelerate launch --num_processes="${NUM_GPUS}" "${SCRIPT}" "${ARGS[@]}"
