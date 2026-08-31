#!/usr/bin/env bash
set -euo pipefail

HANDOFF_DIR="/mnt/image-edit/datasets/duanyufa/交接/多模态畸形检测reward model"
INFER_SCRIPT="${HANDOFF_DIR}/推理/infer_single_body_deformity.py"

PYTHON_BIN="${PYTHON_BIN:-/mnt/image-edit/datasets/duanyufa/conda_envs/miniconda3/envs/ms-swift/bin/python}"
PROJECT_DIR="${PROJECT_DIR:-/mnt/image-edit/datasets/duanyufa/task_shengsheng/models/ms-swift}"
MODEL_PATH="${MODEL_PATH:-/mnt/image-edit/datasets/duanyufa/models/Qwen3-VL-8B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-${HANDOFF_DIR}/checkpoint-3450}"

INPUT_IMAGE="${1:-${INPUT_IMAGE:-}}"
OUTPUT_JSON="${2:-${OUTPUT_JSON:-${HANDOFF_DIR}/推理/output_single.json}}"

if [[ -z "${INPUT_IMAGE}" ]]; then
  echo "Usage: bash ${INFER_SCRIPT%/*}/run_infer_single.sh /path/to/image.png /path/to/output.json" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Error: PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-2048}"

"${PYTHON_BIN}" "${INFER_SCRIPT}" \
  --image "${INPUT_IMAGE}" \
  --output "${OUTPUT_JSON}" \
  --model "${MODEL_PATH}" \
  --adapter "${ADAPTER_PATH}" \
  --project-dir "${PROJECT_DIR}" \
  --device "${DEVICE:-0}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-256}" \
  --temperature "${TEMPERATURE:-0}" \
  --image-max-token-num "${IMAGE_MAX_TOKEN_NUM}"
