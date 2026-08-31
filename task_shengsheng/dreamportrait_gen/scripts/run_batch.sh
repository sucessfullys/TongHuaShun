#!/bin/bash
# ============================================================
# DreamPortrait 批量推理配置 —— 改参数就改这里
# ============================================================
set -euo pipefail

# ---- 模型 ----
MODEL="9B"                        # 9B | 4B
case "$MODEL" in
    9B) MODEL_PATH="/mnt/image-edit/models/black-forest-labs/FLUX.2-klein-base-9B" ;;
    4B) MODEL_PATH="/mnt/image-edit/datasets/duanyufa/FLUX.2-klein-base-4B" ;;
    *) echo "Unknown MODEL: $MODEL"; exit 1 ;;
esac

# ---- 推理参数 ----
STEPS=28
CFG=4.0
HEIGHT=1024
WIDTH=1024
SEED=42
NUM_GPUS=3
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,4,5}"   # 程序1用0,1,2；程序2改3,4,5

# ---- 数据 ----
PROMPTS_FILE="/mnt/image-edit/datasets/duanyufa/task_shengsheng/Gen/prompts_10000_normal.json"

# ---- 输出 ----
OUTPUT_BASE="/mnt/image-edit/datasets/duanyufa/task_shengsheng/dreamportrait_gen/Outputs"
DATASET_TAG="normal"              # 数据集标识，加到命名末尾
FOLDER_NAME="steps${STEPS}_cfg${CFG}_h${HEIGHT}_w${WIDTH}_model${MODEL}_seed${SEED}_${DATASET_TAG}"
OUTPUT_DIR="${OUTPUT_BASE}/${FOLDER_NAME}"
META_FILE="${OUTPUT_BASE}/${FOLDER_NAME}_meta.jsonl"

# ---- 日志 ----
RUN_TS=$(date '+%Y%m%d_%H%M%S')
LOG_DIR="${OUTPUT_BASE}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${FOLDER_NAME}_${RUN_TS}.log"

# ---- Python ----
PYTHON="/root/.venv/dreamface-omni/bin/python"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo "  Model:       ${MODEL} (${MODEL_PATH})"
echo "  Steps:       ${STEPS}  CFG: ${CFG}"
echo "  Resolution:  ${WIDTH}x${HEIGHT}"
echo "  GPUs:        ${NUM_GPUS}"
echo "  Output:      ${OUTPUT_DIR}"
echo "  Meta:        ${META_FILE}"
echo "  Log:         ${LOG_FILE}"
echo "============================================"

mkdir -p "${OUTPUT_DIR}"

"${PYTHON}" -m accelerate.commands.accelerate_cli launch \
    --multi_gpu \
    --num_machines 1 \
    --num_processes "${NUM_GPUS}" \
    "${SCRIPT_DIR}/batch_infer.py" \
    --model-path "${MODEL_PATH}" \
    --prompts-file "${PROMPTS_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --meta-file "${META_FILE}" \
    --steps "${STEPS}" \
    --cfg "${CFG}" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --seed "${SEED}" \
    2>&1 | tee "${LOG_FILE}"

echo "Done. Log: ${LOG_FILE}"
