#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/image-edit/datasets/duanyufa/conda_envs/miniconda3/envs/ms-swift/bin/python}"
EVAL_SCRIPT="${PROJECT_DIR}/examples/train/body_deformity_qwen3_vl/eval_body_deformity.sh"
EVAL_JSONL="${EVAL_JSONL:-/mnt/image-edit/datasets/duanyufa/task_shengsheng/models/ms-swift/examples/train/body_deformity_qwen3_vl/body_deformity_grounding_base_val.jsonl}"
OUT_DIR="${OUT_DIR:-/mnt/image-edit/datasets/duanyufa/task_shengsheng/models/ms-swift/output/body_deformity_qwen3_vl_grounding_eval_val1000_8gpu}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
SHARD_COUNT="${SHARD_COUNT:-8}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DRAW="${DRAW:-true}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Error: PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -f "${EVAL_JSONL}" ]]; then
    echo "Error: EVAL_JSONL does not exist: ${EVAL_JSONL}" >&2
    exit 1
fi
if [[ ! -x "${EVAL_SCRIPT}" ]]; then
    echo "Error: EVAL_SCRIPT is not executable: ${EVAL_SCRIPT}" >&2
    exit 1
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [[ "${#GPU_ARRAY[@]}" -ne "${SHARD_COUNT}" ]]; then
    echo "Error: GPU_IDS count (${#GPU_ARRAY[@]}) must equal SHARD_COUNT (${SHARD_COUNT})" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}/predictions" "${OUT_DIR}/logs"
if [[ "${DRAW}" == "true" ]]; then
    mkdir -p "${OUT_DIR}/visualizations"
fi

echo "PROJECT_DIR=${PROJECT_DIR}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "EVAL_JSONL=${EVAL_JSONL}"
echo "OUT_DIR=${OUT_DIR}"
echo "MAX_SAMPLES=${MAX_SAMPLES}"
echo "SHARD_COUNT=${SHARD_COUNT}"
echo "GPU_IDS=${GPU_IDS}"
echo "DRAW=${DRAW}"

pids=()
for shard_index in $(seq 0 $((SHARD_COUNT - 1))); do
    gpu_id="${GPU_ARRAY[$shard_index]}"
    pred_path="${OUT_DIR}/predictions/shard_${shard_index}.jsonl"
    log_path="${OUT_DIR}/logs/shard_${shard_index}.log"
    draw_dir="${OUT_DIR}/visualizations/shard_${shard_index}"

    echo "Launching shard ${shard_index}/${SHARD_COUNT} on GPU ${gpu_id}; log=${log_path}"
    if [[ "${DRAW}" == "true" ]]; then
        CUDA_VISIBLE_DEVICES="${gpu_id}" \
        PYTHON_BIN="${PYTHON_BIN}" \
        EVAL_JSONL="${EVAL_JSONL}" \
        OUTPUT="${pred_path}" \
        DRAW_DIR="${draw_dir}" \
        MAX_SAMPLES="${MAX_SAMPLES}" \
        SHARD_COUNT="${SHARD_COUNT}" \
        SHARD_INDEX="${shard_index}" \
        BATCH_SIZE="${BATCH_SIZE}" \
        bash "${EVAL_SCRIPT}" > "${log_path}" 2>&1 &
    else
        CUDA_VISIBLE_DEVICES="${gpu_id}" \
        PYTHON_BIN="${PYTHON_BIN}" \
        EVAL_JSONL="${EVAL_JSONL}" \
        OUTPUT="${pred_path}" \
        MAX_SAMPLES="${MAX_SAMPLES}" \
        SHARD_COUNT="${SHARD_COUNT}" \
        SHARD_INDEX="${shard_index}" \
        BATCH_SIZE="${BATCH_SIZE}" \
        bash "${EVAL_SCRIPT}" > "${log_path}" 2>&1 &
    fi
    pids+=("$!")
done

failed=0
for idx in "${!pids[@]}"; do
    pid="${pids[$idx]}"
    if wait "${pid}"; then
        echo "Shard ${idx} finished."
    else
        echo "Shard ${idx} failed. See ${OUT_DIR}/logs/shard_${idx}.log" >&2
        failed=1
    fi
done

if [[ "${failed}" -ne 0 ]]; then
    echo "At least one shard failed; skip merge." >&2
    exit 1
fi

merged_path="${OUT_DIR}/val${MAX_SAMPLES}_predictions.jsonl"
cat "${OUT_DIR}"/predictions/shard_*.jsonl > "${merged_path}"
line_count="$(wc -l < "${merged_path}")"
echo "Merged predictions: ${merged_path}"
echo "Merged line count: ${line_count}"

EXPECTED_LINES="${MAX_SAMPLES}" MERGED_PATH="${merged_path}" OUT_DIR="${OUT_DIR}" PYTHON_BIN="${PYTHON_BIN}" "${PYTHON_BIN}" - <<'PY'
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/image-edit/datasets/duanyufa/task_shengsheng/models/ms-swift/examples/train/body_deformity_qwen3_vl")
import eval_body_deformity as e

pred_path = Path(os.environ["MERGED_PATH"])
out_dir = Path(os.environ["OUT_DIR"])
expected = int(os.environ["EXPECTED_LINES"])
results = [json.loads(line) for line in pred_path.open(encoding="utf-8") if line.strip()]
if len(results) != expected:
    raise SystemExit(f"Expected {expected} predictions, got {len(results)}")

metrics = e.build_metrics(results, [0.3, 0.5])
metric_path = out_dir / f"val{expected}_predictions.metrics.json"
metric_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(metrics, ensure_ascii=False, indent=2))
print(f"metrics: {metric_path}")
PY

echo "Done."
