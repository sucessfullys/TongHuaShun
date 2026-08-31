#!/bin/bash
set -euo pipefail

# 4K CSV batch inference on 48GB L20 GPUs (FLUX.2 Klein + PiD 2kto4k).
#
# Phase-swapped VRAM strategy: text encoder / FLUX transformer / PiD decoder
# are moved on/off the GPU per phase, so each phase fits in 48GB at full 4K
# (3584x4608) without reducing resolution. Per-sample swap cost is a few
# seconds, negligible vs minute-scale 4K compute on L20.
#
# Uses the same env as ../run_csv_pid_batch.sh (first-time: bash ../setup_pid_env.sh).
#
# Run:
#   bash run_csv_pid_batch_l20.sh
#
# Optional Ada/L20 acceleration (validate quality before enabling in production):
#   TRANSFORMER_QUANT=fp8dq SAGE_ATTENTION=1 COMPILE_PID=1 bash run_csv_pid_batch_l20.sh
#
# 2K instead of 4K (single swap is usually enough; faster):
#   RESOLUTION=1792,2304 PID_CKPT_TYPE=2k SWAP_TRANSFORMER=0 bash run_csv_pid_batch_l20.sh

PROJ_DIR="$(cd "$(dirname "$0")/../../../.." && pwd)"
# shellcheck disable=SC1091
source "${SHENSHENG_ROOT:-/mnt/data/image-edit/datasets/shensheng}/config/paths.sh"
PID_ROOT="${PID_ROOT:-${SHENSHENG_PID_ROOT}}"
PID_VENV="${PID_VENV:-${SHENSHENG_PID_VENV:-/root/.venv-dreamface-pid}}"
MODELS_ROOT="${DIFFSYNTH_MODEL_BASE_PATH:-${SHENSHENG_MODELS}}"
BACKBONE_MODEL_ID="${BACKBONE_MODEL_ID:-${MODELS_ROOT}/black-forest-labs/FLUX.2-klein-9B}"
LORA="${LORA:-${MODELS_ROOT}/hithink-image-labs/DreamFace_lora/v2.1/diffusers_lora.safetensors}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"

ENV_FILE="${PROJ_DIR}/examples/flux2/model_inference/.pid_env"
if [ -f "${ENV_FILE}" ]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi
if [ -f "${PID_VENV}/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${PID_VENV}/bin/activate"
fi

CSV="${CSV:-/mnt/data/image-edit/datasets/shensheng/datasets/benchmark/filter/filter-intersection-06-04.csv}"
OUTPUT="${OUTPUT:-${PROJ_DIR}/exp_out/csv-pidout-l20-4k}"
GPUS="${GPUS:-0}"
BACKBONE="${BACKBONE:-flux2-klein-9b}"
RESOLUTION="${RESOLUTION:-3584,4608}"   # final PiD output W,H (must be divisible by 16)
SCALE="${SCALE:-4}"                     # LDM runs at resolution/scale
PID_CKPT_TYPE="${PID_CKPT_TYPE:-2kto4k}"

# Phase swap (1=on). All three are required to fit 4K in 48GB.
SWAP_TEXT_ENCODER="${SWAP_TEXT_ENCODER:-1}"
SWAP_TRANSFORMER="${SWAP_TRANSFORMER:-1}"
SWAP_PID="${SWAP_PID:-1}"

# Acceleration (off by default).
TE_QUANT="${TE_QUANT:-none}"                    # none|4bit|8bit (bitsandbytes; TE stays resident)
TRANSFORMER_QUANT="${TRANSFORMER_QUANT:-none}"  # none|fp8dq|fp8wo|int8wo (torchao, fp8 needs SM89+)
SAGE_ATTENTION="${SAGE_ATTENTION:-0}"           # SageAttention INT8 SDPA patch
COMPILE_PID="${COMPILE_PID:-0}"
COMPILE_LDM="${COMPILE_LDM:-0}"

export PID_ROOT
export PYTHONPATH="${PID_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/mnt/data/image-edit/datasets/shensheng/models/huggingface}"
export DIFFSYNTH_MODEL_BASE_PATH="${MODELS_ROOT}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
# The PiD decode phase runs close to the 48GB limit: avoid allocator fragmentation.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [ "${LOCAL_FILES_ONLY}" = "1" ]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

cd "${PID_ROOT}"

echo "PiD root: ${PID_ROOT}"
echo "Models root: ${MODELS_ROOT}"
echo "Backbone: ${BACKBONE_MODEL_ID}"
echo "LoRA: ${LORA}"
echo "CSV: ${CSV}"
echo "Output: ${OUTPUT}"
echo "Resolution (W,H): ${RESOLUTION}, scale=${SCALE}, pid_ckpt=${PID_CKPT_TYPE}"
echo "Swap: te=${SWAP_TEXT_ENCODER} transformer=${SWAP_TRANSFORMER} pid=${SWAP_PID}"
echo "Accel: te_quant=${TE_QUANT} transformer_quant=${TRANSFORMER_QUANT} sage=${SAGE_ATTENTION} compile_pid=${COMPILE_PID} compile_ldm=${COMPILE_LDM}"
echo "Alloc conf: ${PYTORCH_CUDA_ALLOC_CONF}"

EXTRA_ARGS=()
if [ "${LOCAL_FILES_ONLY}" = "1" ]; then
  EXTRA_ARGS+=(--local-files-only)
fi
if [ "${SWAP_TEXT_ENCODER}" = "0" ]; then
  EXTRA_ARGS+=(--no-swap-text-encoder)
fi
if [ "${SWAP_TRANSFORMER}" = "0" ]; then
  EXTRA_ARGS+=(--no-swap-transformer)
fi
if [ "${SWAP_PID}" = "0" ]; then
  EXTRA_ARGS+=(--no-swap-pid)
fi
if [ "${TE_QUANT}" != "none" ]; then
  EXTRA_ARGS+=(--te-quant "${TE_QUANT}")
fi
if [ "${TRANSFORMER_QUANT}" != "none" ]; then
  EXTRA_ARGS+=(--quant-transformer "${TRANSFORMER_QUANT}")
fi
if [ "${SAGE_ATTENTION}" = "1" ]; then
  EXTRA_ARGS+=(--sage-attention)
fi
if [ "${COMPILE_PID}" = "1" ]; then
  EXTRA_ARGS+=(--compile-pid)
fi
if [ "${COMPILE_LDM}" = "1" ]; then
  EXTRA_ARGS+=(--compile-ldm)
fi

python3 "${PROJ_DIR}/examples/flux2/model_inference/l20/csv_pid_batch_infer_l20.py" \
  --csv "${CSV}" \
  --output "${OUTPUT}" \
  --pid-root "${PID_ROOT}" \
  --backbone "${BACKBONE}" \
  --backbone-model-id "${BACKBONE_MODEL_ID}" \
  --lora "${LORA}" \
  --pid-ckpt-type "${PID_CKPT_TYPE}" \
  --resolution "${RESOLUTION}" \
  --scale "${SCALE}" \
  --gpus "${GPUS}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
