#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/mnt/image-edit/datasets/duanyufa/task_shengsheng/models/ms-swift"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/examples/train/body_deformity_qwen3_vl:${PROJECT_DIR}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/image-edit/datasets/duanyufa/conda_envs/miniconda3/envs/ms-swift/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/image-edit/datasets/duanyufa/models/Qwen3-VL-8B-Instruct}"
DEFAULT_ADAPTER_PATH="/mnt/image-edit/datasets/duanyufa/task_shengsheng/models/ms-swift/output/body_deformity_qwen3_vl_multihand_reason_v3_lora_sft/frombase_lora_r16_a32_ep2_lr5e-5_gb8_img2048_len4096_fixedval_20260722_173506/v0-20260722-173540/checkpoint-1250"
if [[ -z "${ADAPTER_PATH+x}" ]]; then
    ADAPTER_PATH="${DEFAULT_ADAPTER_PATH}"
fi
TEST_ROOT="${TEST_ROOT:-/mnt/image-edit/datasets/duanyufa/task_shengsheng/Open_dataset/Ours_bad/hand_foder/test}"
EVAL_JSONL="${EVAL_JSONL:-}"
GPUS="${GPUS:-2,3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
SHARD_TIMEOUT="${SHARD_TIMEOUT:-3600}"
GPU_TAG="${GPUS//,/_}"
if [[ -n "${ADAPTER_PATH}" ]]; then
    DEFAULT_RUN_PREFIX="best1250"
else
    DEFAULT_RUN_PREFIX="base"
fi
RUN_TAG="${RUN_TAG:-${DEFAULT_RUN_PREFIX}_test_v3_gpus${GPU_TAG}_tok${MAX_NEW_TOKENS}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${TEST_ROOT}/multilimb_reason_v3_eval/${RUN_TAG}}"
CLEANUP_SHARDS="${CLEANUP_SHARDS:-false}"

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
if [[ -n "${EVAL_JSONL}" && ! -f "${EVAL_JSONL}" ]]; then
    echo "Error: EVAL_JSONL does not exist: ${EVAL_JSONL}" >&2
    exit 1
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
SHARD_COUNT="${#GPU_ARRAY[@]}"
if [[ "${SHARD_COUNT}" -lt 1 ]]; then
    echo "Error: GPUS is empty" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/shards"

echo "Project:        ${PROJECT_DIR}"
echo "Python:         ${PYTHON_BIN}"
echo "Model:          ${MODEL_PATH}"
echo "Adapter:        ${ADAPTER_PATH:-<none>}"
echo "Test root:      ${TEST_ROOT}"
echo "Eval jsonl:     ${EVAL_JSONL:-<none>}"
echo "Output dir:     ${OUTPUT_DIR}"
echo "GPUs:           ${GPUS}"
echo "Shard count:    ${SHARD_COUNT}"
echo "Max new tokens: ${MAX_NEW_TOKENS}"
echo "Shard timeout:  ${SHARD_TIMEOUT}"
echo "Cleanup shards: ${CLEANUP_SHARDS}"

pids=()
for shard_index in "${!GPU_ARRAY[@]}"; do
    gpu="${GPU_ARRAY[${shard_index}]}"
    shard_dir="${OUTPUT_DIR}/shards/shard_${shard_index}"
    shard_log="${OUTPUT_DIR}/logs/shard_${shard_index}_gpu${gpu}.log"
    echo "Launching shard ${shard_index}/${SHARD_COUNT} on GPU ${gpu}; log=${shard_log}"
    (
        export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
        export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-2048}"
        ARGS=(
            "${PROJECT_DIR}/examples/train/body_deformity_qwen3_vl/eval_multilimb_reason_v3.py"
            --model "${MODEL_PATH}"
            --test-root "${TEST_ROOT}"
            --output-dir "${shard_dir}"
            --batch-size "${BATCH_SIZE:-1}"
            --max-new-tokens "${MAX_NEW_TOKENS}"
            --temperature "${TEMPERATURE:-0}"
            --device "${gpu}"
            --image-max-token-num "${IMAGE_MAX_TOKEN_NUM}"
            --shard-count "${SHARD_COUNT}"
            --shard-index "${shard_index}"
        )
        if [[ -n "${ADAPTER_PATH}" ]]; then
            ARGS+=(--adapter "${ADAPTER_PATH}")
        fi
        if [[ -n "${EVAL_JSONL}" ]]; then
            ARGS+=(--eval-jsonl "${EVAL_JSONL}")
        fi
        echo "Shard eval jsonl: ${EVAL_JSONL:-<none>}"
        printf 'Shard command: %q ' "${PYTHON_BIN}" "${ARGS[@]}"
        printf '\n'
        if [[ -n "${MAX_SAMPLES:-}" ]]; then
            ARGS+=(--max-samples "${MAX_SAMPLES}")
        fi
        if [[ "${SHARD_TIMEOUT}" == "0" ]]; then
            "${PYTHON_BIN}" "${ARGS[@]}"
        else
            timeout --foreground "${SHARD_TIMEOUT}" "${PYTHON_BIN}" "${ARGS[@]}"
        fi
    ) > "${shard_log}" 2>&1 &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        failed=1
    fi
done

if [[ "${failed}" -ne 0 ]]; then
    echo "Error: at least one eval shard failed. Check logs in ${OUTPUT_DIR}/logs" >&2
    exit 1
fi

"${PYTHON_BIN}" - <<PY
from pathlib import Path
import json
import sys

project_dir = Path("${PROJECT_DIR}")
sys.path.insert(0, str(project_dir / "examples/train/body_deformity_qwen3_vl"))
from eval_multilimb_reason_v3 import build_metrics, write_outputs

output_dir = Path("${OUTPUT_DIR}")
results = []
for shard_path in sorted((output_dir / "shards").glob("shard_*/predictions.jsonl")):
    with shard_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
results.sort(key=lambda r: (r.get("gt_conclusion", ""), r.get("group", ""), r.get("id", "")))
metrics = build_metrics(results)
write_outputs(results, metrics, output_dir)
print(json.dumps(metrics, ensure_ascii=False, indent=2))
PY

if [[ "${CLEANUP_SHARDS}" == "true" ]]; then
    rm -rf "${OUTPUT_DIR}/shards"
fi

echo "Merged eval complete: ${OUTPUT_DIR}"
