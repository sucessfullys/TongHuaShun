#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/mnt/image-edit/datasets/duanyufa/task_shengsheng/models/ms-swift"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/image-edit/datasets/duanyufa/conda_envs/miniconda3/envs/ms-swift/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/image-edit/datasets/duanyufa/models/Qwen3-VL-8B-Instruct}"
DEFAULT_ADAPTER_PATH="/mnt/image-edit/datasets/duanyufa/task_shengsheng/models/ms-swift/output/body_deformity_qwen3_vl_multihand_reason_v3_lora_sft/frombase_lora_r16_a32_ep2_lr5e-5_gb8_img2048_len4096_fixedval_20260722_173506/v0-20260722-173540/checkpoint-1250"
if [[ -z "${ADAPTER_PATH+x}" ]]; then
    ADAPTER_PATH="${DEFAULT_ADAPTER_PATH}"
fi
TEST_ROOT="${TEST_ROOT:-/mnt/image-edit/datasets/duanyufa/task_shengsheng/Open_dataset/Ours_bad/hand_foder/test}"

RUN_TAG="${RUN_TAG:-best1250_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${TEST_ROOT}/multihand_reason_v3_eval/${RUN_TAG}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Error: PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Error: MODEL_PATH does not exist: ${MODEL_PATH}" >&2
    exit 1
fi
if [[ -n "${ADAPTER_PATH}" && ! -d "${ADAPTER_PATH}" ]]; then
    echo "Error: ADAPTER_PATH does not exist: ${ADAPTER_PATH}" >&2
    exit 1
fi
if [[ ! -d "${TEST_ROOT}" ]]; then
    echo "Error: TEST_ROOT does not exist: ${TEST_ROOT}" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-2048}"
DEVICE_ARG="${CUDA_VISIBLE_DEVICES%%,*}"
if [[ -z "${DEVICE_ARG}" ]]; then
    DEVICE_ARG="0"
fi

mkdir -p "${OUTPUT_DIR}/logs"
LOG_PATH="${OUTPUT_DIR}/logs/eval_$(date +%Y%m%d_%H%M%S).log"

echo "Project:    ${PROJECT_DIR}"
echo "Python:     ${PYTHON_BIN}"
echo "Model:      ${MODEL_PATH}"
echo "Adapter:    ${ADAPTER_PATH:-<none>}"
echo "Test root:  ${TEST_ROOT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "GPU:        ${CUDA_VISIBLE_DEVICES}"
echo "Device arg: ${DEVICE_ARG}"
echo "Log:        ${LOG_PATH}"

ARGS=(
    "${PROJECT_DIR}/examples/train/body_deformity_qwen3_vl/eval_multihand_reason_v3.py"
    --model "${MODEL_PATH}"
    --test-root "${TEST_ROOT}"
    --output-dir "${OUTPUT_DIR}"
    --batch-size "${BATCH_SIZE:-1}"
    --max-new-tokens "${MAX_NEW_TOKENS:-256}"
    --temperature "${TEMPERATURE:-0}"
    --device "${DEVICE_ARG}"
    --image-max-token-num "${IMAGE_MAX_TOKEN_NUM}"
)

if [[ -n "${ADAPTER_PATH}" ]]; then
    ARGS+=(--adapter "${ADAPTER_PATH}")
fi

if [[ -n "${MAX_SAMPLES:-}" ]]; then
    ARGS+=(--max-samples "${MAX_SAMPLES}")
fi
if [[ -n "${SHARD_COUNT:-}" ]]; then
    ARGS+=(--shard-count "${SHARD_COUNT}")
fi
if [[ -n "${SHARD_INDEX:-}" ]]; then
    ARGS+=(--shard-index "${SHARD_INDEX}")
fi

"${PYTHON_BIN}" "${ARGS[@]}" 2>&1 | tee "${LOG_PATH}"
