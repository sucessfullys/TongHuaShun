#!/bin/bash
set -euo pipefail

# CSV batch inference with official diffusers + PiD (no diffsynth).
#
# First-time setup:
#   bash setup_pid_env.sh
#
# Then run:
#   bash run_csv_pid_batch.sh

PROJ_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
# shellcheck disable=SC1091
source "${SHENSHENG_ROOT:-/mnt/data/image-edit/datasets/shensheng}/config/paths.sh"
PID_ROOT="${PID_ROOT:-${SHENSHENG_PID_ROOT}}"
PID_VENV="${PID_VENV:-${SHENSHENG_PID_VENV:-/root/.venv-dreamface-pid}}"
MODELS_ROOT="${DIFFSYNTH_MODEL_BASE_PATH:-${SHENSHENG_MODELS}}"
BACKBONE_MODEL_ID="${BACKBONE_MODEL_ID:-${MODELS_ROOT}/black-forest-labs/FLUX.2-klein-9B}"
LORA="${LORA:-${MODELS_ROOT}/hithink-image-labs/DreamFace_lora/v2.1/diffusers_lora.safetensors}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
LOW_VRAM="${LOW_VRAM:-0}"
MAX_VRAM_GB="${MAX_VRAM_GB:-80}"
ENV_FILE="${PROJ_DIR}/examples/flux2/model_inference/.pid_env"
if [ -f "${ENV_FILE}" ]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi
if [ -f "${PID_VENV}/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${PID_VENV}/bin/activate"
fi
CSV="${CSV:-${SHENSHENG_DATA}/benchmark/filter/filter-intersection-06-04.csv}"
OUTPUT="${OUTPUT:-${PROJ_DIR}/exp_out/csv-pidout20482730}"
GPUS="${GPUS:-0}"
BACKBONE="${BACKBONE:-flux2-klein-9b}"
RESOLUTION="${RESOLUTION:-1792,2304}"   # final PiD output W,H
SCALE="${SCALE:-4}"                    # LDM runs at resolution/scale

export PID_ROOT
export PYTHONPATH="${PID_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${SHENSHENG_MODELS}/huggingface}"
export DIFFSYNTH_MODEL_BASE_PATH="${MODELS_ROOT}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
if [ "${LOCAL_FILES_ONLY}" = "1" ]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

cd "${PID_ROOT}"

echo "PiD root: ${PID_ROOT}"
echo "Models root: ${MODELS_ROOT}"
echo "Backbone: ${BACKBONE_MODEL_ID}"
echo "LoRA: ${LORA}"
echo "Local files only: ${LOCAL_FILES_ONLY}"
echo "CSV: ${CSV}"
echo "Output: ${OUTPUT}"
echo "Backbone tag: ${BACKBONE}"
echo "Resolution (W,H): ${RESOLUTION}, scale=${SCALE}"
echo "Low VRAM mode: ${LOW_VRAM}"
echo "Max VRAM budget: ${MAX_VRAM_GB}GB"

EXTRA_ARGS=()
if [ "${LOCAL_FILES_ONLY}" = "1" ]; then
  EXTRA_ARGS+=(--local-files-only)
fi
if [ "${LOW_VRAM}" = "1" ]; then
  EXTRA_ARGS+=(--low-vram)
fi
if [ -n "${MAX_VRAM_GB}" ]; then
  EXTRA_ARGS+=(--max-vram-gb "${MAX_VRAM_GB}")
fi

python3 "${PROJ_DIR}/examples/flux2/model_inference/csv_pid_batch_infer.py" \
  --csv "${CSV}" \
  --output "${OUTPUT}" \
  --pid-root "${PID_ROOT}" \
  --backbone "${BACKBONE}" \
  --backbone-model-id "${BACKBONE_MODEL_ID}" \
  --lora "${LORA}" \
  --resolution "${RESOLUTION}" \
  --scale "${SCALE}" \
  --gpus "${GPUS}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
