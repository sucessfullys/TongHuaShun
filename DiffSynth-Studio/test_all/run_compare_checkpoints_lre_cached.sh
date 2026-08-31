#!/usr/bin/env bash
set -euo pipefail

# Cached LRE checkpoint comparison.
#
# Optimizes:
#   1. Load FLUX.2 base/text/VAE/template once per process.
#   2. Encode LR images into LRE initial noise once per process.
#   3. Loop over all checkpoints by swapping only Template checkpoint weights.

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/outputs/Template-KleinBase4B-Deblur_all_LRE_lre0.8_lr5e-6_rep2_ep2_20260818_111338}"
METADATA="${METADATA:-/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/test_all/测试集合/metadata.jsonl}"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/test_all}"
NUM_GPUS="${NUM_GPUS:-4}"
LIMIT="${LIMIT:-}"
SEED="${SEED:-42}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
CFG_SCALE="${CFG_SCALE:-4.0}"
EMBEDDED_GUIDANCE="${EMBEDDED_GUIDANCE:-4.0}"
LRE_STRENGTH="${LRE_STRENGTH:-0.8}"
CACHE_DEVICE="${CACHE_DEVICE:-cpu}"
CHECKPOINT_PATTERN="${CHECKPOINT_PATTERN:-*.safetensors}"
CHECKPOINT_NAMES="${CHECKPOINT_NAMES:-}"
OVERWRITE="${OVERWRITE:-true}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
DIFFSYNTH_ENV="${DIFFSYNTH_ENV:-/mnt/image-edit/datasets/duanyufa/conda_envs/DiffSynth}"
PYTHON_BIN="${PYTHON_BIN:-${DIFFSYNTH_ENV}/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${DIFFSYNTH_ENV}/bin/accelerate}"

SCRIPT="/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/test_all/compare_checkpoints_lre_cached.py"

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
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing executable PYTHON_BIN: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -x "${ACCELERATE_BIN}" ]]; then
  echo "Missing executable ACCELERATE_BIN: ${ACCELERATE_BIN}" >&2
  exit 1
fi

LOG_ROOT="${OUTPUT_BASE}/logs_lre_cached/$(basename "${CHECKPOINT_DIR}")"
mkdir -p "${LOG_ROOT}"
LOG_PATH="${LOG_PATH:-${LOG_ROOT}/test_${RUN_ID}.log}"
exec > >(tee -a "${LOG_PATH}") 2>&1

export PATH="${DIFFSYNTH_ENV}/bin:${PATH}"

ARGS=(
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --metadata "${METADATA}"
  --output-base "${OUTPUT_BASE}"
  --checkpoint-pattern "${CHECKPOINT_PATTERN}"
  --num-inference-steps "${NUM_INFERENCE_STEPS}"
  --cfg-scale "${CFG_SCALE}"
  --embedded-guidance "${EMBEDDED_GUIDANCE}"
  --seed "${SEED}"
  --lre-strength "${LRE_STRENGTH}"
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
echo "  Cached LRE checkpoint comparison launch"
echo "  CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "  METADATA=${METADATA}"
echo "  OUTPUT_BASE=${OUTPUT_BASE}"
echo "  NUM_GPUS=${NUM_GPUS}"
echo "  LIMIT=${LIMIT:-all}"
echo "  CHECKPOINT_PATTERN=${CHECKPOINT_PATTERN}"
echo "  CHECKPOINT_NAMES=${CHECKPOINT_NAMES:-all}"
echo "  LRE_STRENGTH=${LRE_STRENGTH}"
echo "  CACHE_DEVICE=${CACHE_DEVICE}"
echo "  PYTHON_BIN=${PYTHON_BIN}"
echo "  ACCELERATE_BIN=${ACCELERATE_BIN}"
echo "  LOG_PATH=${LOG_PATH}"
echo "=========================================="

"${ACCELERATE_BIN}" launch --num_processes="${NUM_GPUS}" "${SCRIPT}" "${ARGS[@]}"
