#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/mnt/image-edit/datasets/duanyufa/conda_envs/sam3/bin/python}"
SAM3_DIR="/mnt/image-edit/datasets/duanyufa/task_shengsheng/models/sam3"
SCRIPT="${SAM3_DIR}/batch_segment_best_mask_sam3.py"

INPUT_IMAGE_DIR="${INPUT_IMAGE_DIR:-/mnt/image-edit/datasets/duanyufa/task_shengsheng/Open_dataset/Ours_bad/hand_foder/test/dreamface2_filter_multihand_vlm_20260728/benchmark/normal2}"
OUTPUT_IMAGE_DIR="${OUTPUT_IMAGE_DIR:-/mnt/image-edit/datasets/duanyufa/task_shengsheng/Open_dataset/Ours_bad/hand_foder/test/dreamface2_filter_multihand_vlm_20260728/benchmark_norm2_seg_hand}"
LOG_DIR="${LOG_DIR:-${OUTPUT_IMAGE_DIR}/sam3_hand_seg_logs}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/image-edit/datasets/duanyufa/task_shengsheng/models/sam3/hf_sam3.1/sam3.1_multiplex.pt}"
PROMPT="${PROMPT:-arms}"
THRESHOLD="${THRESHOLD:-0.35}"
SHARD_COUNT="${SHARD_COUNT:-5}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4}"
RESUME="${RESUME:-true}"
LIMIT="${LIMIT:-0}"
RECURSIVE="${RECURSIVE:-false}"
ALPHA="${ALPHA:-0.5}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Error: PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${SCRIPT}" ]]; then
  echo "Error: script not found: ${SCRIPT}" >&2
  exit 1
fi
if [[ ! -d "${INPUT_IMAGE_DIR}" ]]; then
  echo "Error: input image dir not found: ${INPUT_IMAGE_DIR}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_IMAGE_DIR}" "${LOG_DIR}"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
if [[ "${#GPUS[@]}" -ne "${SHARD_COUNT}" ]]; then
  echo "Error: SHARD_COUNT=${SHARD_COUNT}, but GPU_IDS has ${#GPUS[@]} entries: ${GPU_IDS}" >&2
  exit 1
fi

echo "Using Python: $(${PYTHON_BIN} -V 2>&1)"
echo "Input images: ${INPUT_IMAGE_DIR}"
echo "Output hand segmentation images: ${OUTPUT_IMAGE_DIR}"
echo "Prompt: ${PROMPT}"
echo "Threshold: ${THRESHOLD}"
echo "Render: highest-confidence mask only"
echo "Logs: ${LOG_DIR}"

pids=()
for shard_index in $(seq 0 $((SHARD_COUNT - 1))); do
  gpu="${GPUS[$shard_index]}"
  log_file="${LOG_DIR}/shard_${shard_index}.log"
  json_log="${LOG_DIR}/shard_${shard_index}.jsonl"
  resume_flag=()
  recursive_flag=()
  if [[ "${RESUME}" == "true" ]]; then
    resume_flag=(--resume)
  fi
  if [[ "${RECURSIVE}" == "true" ]]; then
    recursive_flag=(--recursive)
  fi
  echo "Launching shard ${shard_index}/${SHARD_COUNT} on GPU ${gpu}; log=${log_file}"
  (
    cd "${SAM3_DIR}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${SCRIPT}" \
      --input-dir "${INPUT_IMAGE_DIR}" \
      --output-dir "${OUTPUT_IMAGE_DIR}" \
      --checkpoint-path "${CHECKPOINT_PATH}" \
      --prompt "${PROMPT}" \
      --threshold "${THRESHOLD}" \
      --alpha "${ALPHA}" \
      --device "cuda:0" \
      --shard-index "${shard_index}" \
      --shard-count "${SHARD_COUNT}" \
      --limit "${LIMIT}" \
      --log-jsonl "${json_log}" \
      "${resume_flag[@]}" \
      "${recursive_flag[@]}"
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
echo "segmented_images=$(find "${OUTPUT_IMAGE_DIR}" -type f -name '*.png' ! -path '*/sam3_hand_seg_logs/*' | wc -l)"
echo "logs=${LOG_DIR}"

exit "${failed}"
