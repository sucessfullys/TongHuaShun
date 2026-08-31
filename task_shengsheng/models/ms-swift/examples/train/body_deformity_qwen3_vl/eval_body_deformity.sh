#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/image-edit/datasets/duanyufa/conda_envs/miniconda3/envs/ms-swift/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/image-edit/datasets/duanyufa/task_shengsheng/models/ms-swift/output/body_deformity_qwen3_vl_grounding_v2_schemeA_full_sft/frombase_ep1_lr1e-5_gb16_img1536_len4096_dszero2_val0.02_20260716_174225/v0-20260716-174353/checkpoint-4199}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
EVAL_JSONL="${EVAL_JSONL:-/mnt/image-edit/datasets/duanyufa/task_shengsheng/models/ms-swift/examples/train/body_deformity_qwen3_vl/body_deformity_grounding_base_val.jsonl}"
OUTPUT="${OUTPUT:-/mnt/image-edit/datasets/duanyufa/task_shengsheng/models/ms-swift/output/body_deformity_qwen3_vl_grounding_v2_schemeA_eval/predictions.jsonl}"

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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1536}"
export QWENVL_BBOX_FORMAT="${QWENVL_BBOX_FORMAT:-new}"

ARGS=(
    "${PROJECT_DIR}/examples/train/body_deformity_qwen3_vl/eval_body_deformity.py"
    --model "${MODEL_PATH}"
    --output "${OUTPUT}"
    --batch-size "${BATCH_SIZE:-1}"
    --max-new-tokens "${MAX_NEW_TOKENS:-1024}"
    --temperature "${TEMPERATURE:-0}"
    --device "${CUDA_VISIBLE_DEVICES%%,*}"
    --image-max-token-num "${IMAGE_MAX_TOKEN_NUM}"
)

if [[ -n "${ADAPTER_PATH}" ]]; then
    ARGS+=(--adapter "${ADAPTER_PATH}")
fi

if [[ -n "${IMAGE:-}" ]]; then
    ARGS+=(--image "${IMAGE}")
elif [[ -n "${IMAGE_DIR:-}" ]]; then
    ARGS+=(--image-dir "${IMAGE_DIR}")
    if [[ -n "${LABEL_DIR:-}" ]]; then
        ARGS+=(--label-dir "${LABEL_DIR}")
    fi
else
    ARGS+=(--eval-jsonl "${EVAL_JSONL}")
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

if [[ -n "${DRAW_DIR:-}" ]]; then
    ARGS+=(--draw-dir "${DRAW_DIR}")
fi

"${PYTHON_BIN}" "${ARGS[@]}"
