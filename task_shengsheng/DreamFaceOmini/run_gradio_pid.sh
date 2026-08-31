#!/bin/bash
set -euo pipefail
source /root/.venv-dreamface-pid/bin/activate 
# ╔══════════════════════════════════════════════════════════════════╗
# ║  PiD Gradio — 每次启动只改这里                                    ║
# ╚══════════════════════════════════════════════════════════════════╝

EXP_NAME="pid-gradio"
BACKBONE="flux2-klein-9b"
BACKBONE_MODEL_ID="/mnt/data/image-edit/datasets/shensheng/models/black-forest-labs/FLUX.2-klein-9B"
PID_CKPT_TYPE="2kto4k"                 # 2k | 2kto4k
LORA="/mnt/data/image-edit/datasets/shensheng/models/hithink-image-labs/DreamFace_lora/v2.1/diffusers_lora.safetensors"
LORA_SCALE=1.0

SCALE=4                            # 固定 4 倍，不可改（其他倍率效果差）
WIDTH=3584
HEIGHT=4608
SEED=-1
LDM_STEPS=4
GUIDANCE=1.0
PID_STEPS=4
PID_CFG=1.0

PHYSICAL_GPU=0
SERVER_NAME="0.0.0.0"
SERVER_PORT=7863
LOW_VRAM=0
LOCAL_FILES_ONLY=1
LOG_DIR="${SHENSHENG_ROOT:-/mnt/data/image-edit/datasets/shensheng}/lab/code/stable/DreamFaceOmini/exp_out/gradio_pid_logs"
DISABLE_LOG=0

# ╔══════════════════════════════════════════════════════════════════╗
# ║  以下不用改                                                       ║
# ╚══════════════════════════════════════════════════════════════════╝

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SHENSHENG_ROOT:-/mnt/data/image-edit/datasets/shensheng}/config/paths.sh"

PID_ROOT="${PID_ROOT:-${SHENSHENG_PID_ROOT}}"
PID_VENV="${PID_VENV:-${SHENSHENG_PID_VENV:-/root/.venv-dreamface-pid}}"
MODELS_ROOT="${DIFFSYNTH_MODEL_BASE_PATH:-${SHENSHENG_MODELS}}"

ENV_FILE="${PROJ_DIR}/examples/flux2/model_inference/.pid_env"
if [ -f "${ENV_FILE}" ]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi
if [ -f "${PID_VENV}/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${PID_VENV}/bin/activate"
fi

if [ -n "${LORA}" ] && [ ! -f "${LORA}" ]; then
  echo "[WARN] LoRA 文件不存在，可在 Gradio 里切换: ${LORA}"
fi

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export PID_ROOT
export PYTHONPATH="${PID_ROOT}:${PROJ_DIR}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${MODELS_ROOT}/huggingface}"
export DIFFSYNTH_MODEL_BASE_PATH="${MODELS_ROOT}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
if [ "${LOCAL_FILES_ONLY}" = "1" ]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

EXTRA_ARGS=()
if [ "${LOW_VRAM}" = "1" ]; then
  EXTRA_ARGS+=(--low-vram)
fi
if [ "${LOCAL_FILES_ONLY}" = "1" ]; then
  EXTRA_ARGS+=(--local-files-only)
fi
if [ "${DISABLE_LOG}" = "1" ]; then
  EXTRA_ARGS+=(--no-log)
fi

export GRADIO_PID_LOG_DIR="${LOG_DIR}"

echo ""
echo "============================================"
echo "  PiD Gradio: ${EXP_NAME}"
echo "  GPU: physical ${PHYSICAL_GPU} -> cuda:0"
echo "  PiD root: ${PID_ROOT}"
echo "  Backbone: ${BACKBONE}"
echo "  Model: ${BACKBONE_MODEL_ID}"
echo "  PiD ckpt: ${PID_CKPT_TYPE}"
echo "  LoRA: ${LORA:-none} (scale=${LORA_SCALE})"
echo "  Output: ${WIDTH}x${HEIGHT}, scale=${SCALE}"
echo "  Log dir: ${LOG_DIR}"
echo "  URL: http://${SERVER_NAME}:${SERVER_PORT}"
echo "============================================"
echo ""

cd "${PID_ROOT}"

python3 "${PROJ_DIR}/examples/flux2/model_inference/gradio_pid_infer.py" \
  --pid-root "${PID_ROOT}" \
  --backbone "${BACKBONE}" \
  --backbone-model-id "${BACKBONE_MODEL_ID}" \
  --pid-ckpt-type "${PID_CKPT_TYPE}" \
  --lora "${LORA}" \
  --lora-scale "${LORA_SCALE}" \
  --scale "${SCALE}" \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --seed "${SEED}" \
  --ldm-steps "${LDM_STEPS}" \
  --guidance-scale "${GUIDANCE}" \
  --pid-steps "${PID_STEPS}" \
  --pid-cfg "${PID_CFG}" \
  --gpu 0 \
  --server-name "${SERVER_NAME}" \
  --server-port "${SERVER_PORT}" \
  --log-dir "${LOG_DIR}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
