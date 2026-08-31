#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/mnt/image-edit/datasets/duanyufa/conda_envs/sam3/bin/python}"
SAM3_DIR="/mnt/image-edit/datasets/duanyufa/task_shengsheng/models/sam3"
SCRIPT="${SAM3_DIR}/annotate_images_sam3_boxes.py"

INPUT_DIR="${INPUT_DIR:-/mnt/image-edit/datasets/duanyufa/task_shengsheng/Open_dataset/Ours/Version1/images_1}"
LABEL_DIR="${LABEL_DIR:-/mnt/image-edit/datasets/duanyufa/task_shengsheng/Open_dataset/Ours/Version1/labels_1}"
LOG_DIR="${LOG_DIR:-/mnt/image-edit/datasets/duanyufa/task_shengsheng/Open_dataset/Ours/Version1/sam3_person_logs_images_1}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/image-edit/datasets/duanyufa/task_shengsheng/models/sam3/hf_sam3.1/sam3.1_multiplex.pt}"
PROMPT="${PROMPT:-person}"
THRESHOLD="${THRESHOLD:-0.5}"
MAX_OBJECTS="${MAX_OBJECTS:-0}"
SHARD_COUNT="${SHARD_COUNT:-8}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
RESUME="${RESUME:-true}"
LIMIT="${LIMIT:-0}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Error: PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${SCRIPT}" ]]; then
  echo "Error: script not found: ${SCRIPT}" >&2
  exit 1
fi
if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "Error: input dir not found: ${INPUT_DIR}" >&2
  exit 1
fi

mkdir -p "${LABEL_DIR}" "${LOG_DIR}"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
if [[ "${#GPUS[@]}" -ne "${SHARD_COUNT}" ]]; then
  echo "Error: SHARD_COUNT=${SHARD_COUNT}, but GPU_IDS has ${#GPUS[@]} entries: ${GPU_IDS}" >&2
  exit 1
fi

echo "Using Python: $(${PYTHON_BIN} -V 2>&1)"
echo "Input images: ${INPUT_DIR}"
echo "Output labels: ${LABEL_DIR}"
echo "Logs: ${LOG_DIR}"
echo "Prompt: ${PROMPT}"

pids=()
for shard_index in $(seq 0 $((SHARD_COUNT - 1))); do
  gpu="${GPUS[$shard_index]}"
  log_file="${LOG_DIR}/shard_${shard_index}.log"
  json_log="${LOG_DIR}/shard_${shard_index}.jsonl"
  resume_flag=()
  if [[ "${RESUME}" == "true" ]]; then
    resume_flag=(--resume)
  fi
  echo "Launching shard ${shard_index}/${SHARD_COUNT} on GPU ${gpu}; log=${log_file}"
  (
    cd "${SAM3_DIR}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${SCRIPT}" \
      --input-dir "${INPUT_DIR}" \
      --label-dir "${LABEL_DIR}" \
      --checkpoint-path "${CHECKPOINT_PATH}" \
      --prompt "${PROMPT}" \
      --threshold "${THRESHOLD}" \
      --max-objects "${MAX_OBJECTS}" \
      --device "cuda:0" \
      --shard-index "${shard_index}" \
      --shard-count "${SHARD_COUNT}" \
      --limit "${LIMIT}" \
      --log-jsonl "${json_log}" \
      "${resume_flag[@]}"
  ) >"${log_file}" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

echo "Shard jobs finished. failed=${failed}"
echo "images=$(find "${INPUT_DIR}" -maxdepth 1 -type f | wc -l)"
echo "labels=$(find "${LABEL_DIR}" -maxdepth 1 -type f -name '*.txt' | wc -l)"

exit "${failed}"
