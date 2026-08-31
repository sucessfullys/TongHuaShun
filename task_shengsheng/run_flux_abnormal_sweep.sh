#!/usr/bin/env bash
# Run multiple FLUX abnormal-hand generation jobs sequentially.
#
# Each job writes to:
#   /mnt/image-edit/datasets/duanyufa/task_shengsheng/Open_dataset/Ours_bad/hand_foder/abnormal/${RUN_NAME}
#
# Usage:
#   bash /mnt/image-edit/datasets/duanyufa/task_shengsheng/run_flux_abnormal_sweep.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run_flux_base_9b_folder.sh"

INPUT_DIR="${INPUT_DIR:-/mnt/image-edit/datasets/duanyufa/task_shengsheng/Open_dataset/Ours_bad/hand_foder/test/dreamface2_filter_multihand_vlm_20260728/benchmark_norm2_seg_hand}"
BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-/mnt/image-edit/datasets/duanyufa/task_shengsheng/Open_dataset/Ours_bad/hand_foder/test/dreamface2_filter_multihand_vlm_20260728/benchmark_norm2_seg_hand_Gen_abnormal}"
PROMPT_FILE="${PROMPT_FILE:-/mnt/image-edit/datasets/duanyufa/task_shengsheng/Open_dataset/Ours_bad/hand_foder/abnormal_hand.txt}"

NUM_GPUS="${NUM_GPUS:-5}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
PRESERVE_SIZE="${PRESERVE_SIZE:-true}"
MAX_RESOLUTION="${MAX_RESOLUTION:-1024}"
OVERWRITE="${OVERWRITE:-false}"
RECURSIVE="${RECURSIVE:-false}"

if [[ ! -f "${RUN_SCRIPT}" ]]; then
  echo "Error: run script not found: ${RUN_SCRIPT}" >&2
  exit 1
fi
if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "Error: input dir not found: ${INPUT_DIR}" >&2
  exit 1
fi
if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "Error: prompt file not found: ${PROMPT_FILE}" >&2
  exit 1
fi

JOBS=(
  "base4b 4.0 50 0 base4b_cfg4.0_step50_seed0"
  "base9b 4.0 28 0 base9b_cfg4.0_step28_seed0"
  "base9b 4.0 28 42 base9b_cfg4.0_step28_seed42"
)

SWEEP_LOG_DIR="${BASE_OUTPUT_ROOT}/sweep_logs"
mkdir -p "${SWEEP_LOG_DIR}"
SWEEP_LOG="${SWEEP_LOG_DIR}/sweep_$(date +%Y%m%d_%H%M%S).log"

echo "Sweep log: ${SWEEP_LOG}" | tee -a "${SWEEP_LOG}"
echo "Input dir: ${INPUT_DIR}" | tee -a "${SWEEP_LOG}"
echo "Output root: ${BASE_OUTPUT_ROOT}" | tee -a "${SWEEP_LOG}"
echo "Prompt file: ${PROMPT_FILE}" | tee -a "${SWEEP_LOG}"
echo "GPUs: NUM_GPUS=${NUM_GPUS}, GPU_IDS=${GPU_IDS}" | tee -a "${SWEEP_LOG}"
echo "Size: PRESERVE_SIZE=${PRESERVE_SIZE}, MAX_RESOLUTION=${MAX_RESOLUTION}" | tee -a "${SWEEP_LOG}"
echo "Jobs: ${#JOBS[@]}" | tee -a "${SWEEP_LOG}"

job_index=0
for job in "${JOBS[@]}"; do
  job_index=$((job_index + 1))
  read -r model_variant cfg steps seed run_name <<<"${job}"
  output_dir="${BASE_OUTPUT_ROOT}/${run_name}"
  job_log="${SWEEP_LOG_DIR}/${run_name}_$(date +%Y%m%d_%H%M%S).log"

  echo "" | tee -a "${SWEEP_LOG}"
  echo "============================================================" | tee -a "${SWEEP_LOG}"
  echo "[${job_index}/${#JOBS[@]}] ${run_name}" | tee -a "${SWEEP_LOG}"
  echo "model=${model_variant} cfg=${cfg} steps=${steps} seed=${seed}" | tee -a "${SWEEP_LOG}"
  echo "output=${output_dir}" | tee -a "${SWEEP_LOG}"
  echo "job_log=${job_log}" | tee -a "${SWEEP_LOG}"
  echo "start=$(date '+%F %T')" | tee -a "${SWEEP_LOG}"

  MODEL_VARIANT="${model_variant}" \
  CFG="${cfg}" \
  STEPS="${steps}" \
  SEED="${seed}" \
  RUN_NAME="${run_name}" \
  INPUT_DIR="${INPUT_DIR}" \
  BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT}" \
  OUTPUT_DIR="${output_dir}" \
  PROMPT_FILE="${PROMPT_FILE}" \
  NUM_GPUS="${NUM_GPUS}" \
  GPU_IDS="${GPU_IDS}" \
  MAX_SAMPLES="${MAX_SAMPLES}" \
  PRESERVE_SIZE="${PRESERVE_SIZE}" \
  MAX_RESOLUTION="${MAX_RESOLUTION}" \
  OVERWRITE="${OVERWRITE}" \
  RECURSIVE="${RECURSIVE}" \
  bash "${RUN_SCRIPT}" 2>&1 | tee "${job_log}"

  echo "finish=$(date '+%F %T')" | tee -a "${SWEEP_LOG}"
  echo "done: ${run_name}" | tee -a "${SWEEP_LOG}"
done

echo "" | tee -a "${SWEEP_LOG}"
echo "All jobs finished: $(date '+%F %T')" | tee -a "${SWEEP_LOG}"
