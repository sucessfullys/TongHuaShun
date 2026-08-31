#!/usr/bin/env bash
# Batch image-to-image generation with FLUX base models, no LoRA.
#
# Usage:
#   bash /mnt/image-edit/datasets/duanyufa/task_shengsheng/run_flux_base_9b_folder.sh
#
# Key options:
#   MODEL_VARIANT=base9b|base4b|dev   Select model.
#   NUM_GPUS=1/2/4/8              Number of parallel shard processes.
#   GPU_IDS=0,1,2,3,4,5,6,7      Physical GPU ids used by those shards.
#   PROMPT='...'                  Inline prompt. Ignored when PROMPT_FILE is set.
#   PROMPT_FILE=/abs/path.txt     Prompt text file.
#   PRESERVE_SIZE=true            Keep each output image the same size as input.
#   MAX_RESOLUTION=1024           Clamp preserved size to max 1024x1024, keeping ratio.
#   CFG=4.0                       Guidance scale.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_VARIANT="${MODEL_VARIANT:-base9b}"
BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-/mnt/image-edit/datasets/duanyufa/task_shengsheng/Open_dataset/Ours_bad/hand_foder/abnormal_leg}"

case "${MODEL_VARIANT}" in
  base9b)
    MODEL_NAME="base9b"
    BACKEND="${BACKEND:-diffusers}"
    DEFAULT_MODEL_PATH="/mnt/image-edit/models/black-forest-labs/FLUX.2-klein-base-9B"
    ;;
  base4b)
    MODEL_NAME="base4b"
    BACKEND="${BACKEND:-diffusers}"
    DEFAULT_MODEL_PATH="/mnt/image-edit/models/black-forest-labs/FLUX.2-klein-base-4B"
    ;;
  dev)
    MODEL_NAME="dev"
    BACKEND="${BACKEND:-diffsynth}"
    DEFAULT_MODEL_PATH="/mnt/data/image-edit/datasets/shensheng/models/black-forest-labs/FLUX.2-dev"
    ;;
  *)
    echo "Error: MODEL_VARIANT must be one of: base9b, base4b, dev. Got: ${MODEL_VARIANT}" >&2
    exit 1
    ;;
esac

MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
INPUT_DIR="${INPUT_DIR:-/mnt/image-edit/datasets/duanyufa/task_shengsheng/Open_dataset/Ours_bad/hand_foder/normal_leg_seg}"
PROMPT="${PROMPT:-}"
PROMPT_FILE="${PROMPT_FILE:-/mnt/image-edit/datasets/duanyufa/task_shengsheng/Open_dataset/Ours_bad/hand_foder/abnormal_leg.txt}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
STEPS="${STEPS:-28}"
CFG="${CFG:-4.0}"
HEIGHT="${HEIGHT:-1024}"
WIDTH="${WIDTH:-1024}"
PRESERVE_SIZE="${PRESERVE_SIZE:-true}"
MAX_RESOLUTION="${MAX_RESOLUTION:-1024}"
SEED="${SEED:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
RECURSIVE="${RECURSIVE:-false}"
OVERWRITE="${OVERWRITE:-false}"
KEEP_EXT="${KEEP_EXT:-false}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4}"
IFS=',' read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
NUM_GPUS="${NUM_GPUS:-5}"

RUN_NAME="${RUN_NAME:-${MODEL_NAME}_cfg${CFG}_step${STEPS}_seed${SEED}}_5"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_OUTPUT_ROOT}/${RUN_NAME}}"
RESULTS_FILE="${RESULTS_FILE:-${OUTPUT_DIR}/results.jsonl}"

if [[ "${NUM_GPUS}" -lt 1 ]]; then
  NUM_GPUS=1
fi
if [[ "${#GPU_ID_ARRAY[@]}" -lt "${NUM_GPUS}" ]]; then
  GPU_ID_ARRAY=()
  for ((i = 0; i < NUM_GPUS; i++)); do
    GPU_ID_ARRAY+=("${i}")
  done
fi

if [[ -z "${INPUT_DIR}" ]]; then
  echo "Error: INPUT_DIR is required." >&2
  exit 1
fi

if [[ -z "${PROMPT}" && -z "${PROMPT_FILE}" ]]; then
  echo "Error: PROMPT or PROMPT_FILE is required." >&2
  exit 1
fi

ARGS=(
  --backend "${BACKEND}"
  --model-path "${MODEL_PATH}"
  --input-dir "${INPUT_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --results-file "${RESULTS_FILE}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --steps "${STEPS}"
  --cfg "${CFG}"
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --max-resolution "${MAX_RESOLUTION}"
  --seed "${SEED}"
  --max-samples "${MAX_SAMPLES}"
)

if [[ -n "${PROMPT_FILE}" ]]; then
  ARGS+=(--prompt-file "${PROMPT_FILE}")
else
  ARGS+=(--prompt "${PROMPT}")
fi

if [[ "${RECURSIVE}" == "true" ]]; then
  ARGS+=(--recursive)
fi
if [[ "${OVERWRITE}" == "true" ]]; then
  ARGS+=(--overwrite)
fi
if [[ "${KEEP_EXT}" == "true" ]]; then
  ARGS+=(--keep-ext)
fi
if [[ "${PRESERVE_SIZE}" == "true" ]]; then
  ARGS+=(--preserve-size)
fi

echo "=============================================="
echo " FLUX folder image editing"
echo "=============================================="
echo "Python:     ${PYTHON_BIN}"
echo "Variant:    ${MODEL_VARIANT}"
echo "Backend:    ${BACKEND}"
echo "Model:      ${MODEL_PATH}"
echo "Input dir:  ${INPUT_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
if [[ -n "${PROMPT_FILE}" ]]; then
  echo "Prompt:     ${PROMPT_FILE}"
else
  echo "Prompt:     inline PROMPT"
fi
echo "Steps/cfg:  ${STEPS} / ${CFG}"
echo "GPUs:       ${NUM_GPUS} (${GPU_ID_ARRAY[*]})"
if [[ "${PRESERVE_SIZE}" == "true" ]]; then
  echo "Size:       preserve ratio, max ${MAX_RESOLUTION}x${MAX_RESOLUTION}"
else
  echo "Size:       ${WIDTH}x${HEIGHT}"
fi
echo

if [[ "${NUM_GPUS}" -eq 1 ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID_ARRAY[0]}" \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/infer_flux_base_9b_folder.py" "${ARGS[@]}"
else
  RESULTS_DIR="$(dirname "${RESULTS_FILE}")"
  RESULTS_NAME="$(basename "${RESULTS_FILE}")"
  SHARD_DIR="${RESULTS_DIR}/.flux_base_9b_shards_${RESULTS_NAME}"
  LOG_DIR="${RESULTS_DIR}/logs"
  mkdir -p "${SHARD_DIR}" "${LOG_DIR}"
  rm -f "${SHARD_DIR}"/shard_*.jsonl

  PIDS=()
  for ((rank = 0; rank < NUM_GPUS; rank++)); do
    gpu="${GPU_ID_ARRAY[$rank]}"
    shard_results="${SHARD_DIR}/shard_${rank}.jsonl"
    shard_log="${LOG_DIR}/shard_${rank}.log"
    echo "Launching shard ${rank}/${NUM_GPUS} on GPU ${gpu}; log=${shard_log}"
    (
      CUDA_VISIBLE_DEVICES="${gpu}" \
        "${PYTHON_BIN}" "${SCRIPT_DIR}/infer_flux_base_9b_folder.py" \
        "${ARGS[@]}" \
        --device cuda \
        --shard-index "${rank}" \
        --shard-count "${NUM_GPUS}" \
        --results-file "${shard_results}"
    ) >"${shard_log}" 2>&1 &
    PIDS+=("$!")
  done

  FAILED=0
  for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
      FAILED=1
    fi
  done

  if [[ "${FAILED}" -ne 0 ]]; then
    echo "Error: at least one shard failed. Check logs in ${LOG_DIR}" >&2
    exit 1
  fi

  "${PYTHON_BIN}" - "${SHARD_DIR}" "${RESULTS_FILE}" <<'PY'
import json
import sys
from pathlib import Path

shard_dir = Path(sys.argv[1])
out_path = Path(sys.argv[2])
records = []
for path in sorted(shard_dir.glob("shard_*.jsonl")):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
records.sort(key=lambda x: x.get("index", -1))
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
print(f"Merged {len(records)} records -> {out_path}")
PY
fi
