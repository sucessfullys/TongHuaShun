#!/usr/bin/env bash
set -euo pipefail

# Pure FLUX.2 Klein Base 4B baseline.
# No Template checkpoint, no LoRA checkpoint.
# The LR image is passed through the native FLUX.2 edit_image path.

METADATA="${METADATA:-/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/test_all/测试集合/metadata.jsonl}"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/test_all}"
OUTPUT_NAME="${OUTPUT_NAME:-FLUX2_KleinBase4B_Base_native_edit_image}"
BASE_MODEL="${BASE_MODEL:-/mnt/image-edit/datasets/duanyufa/FLUX.2-klein-base-4B}"
NUM_GPUS="${NUM_GPUS:-4}"
LIMIT="${LIMIT:-}"
SEED="${SEED:-42}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
CFG_SCALE="${CFG_SCALE:-4.0}"
EMBEDDED_GUIDANCE="${EMBEDDED_GUIDANCE:-4.0}"
OVERWRITE="${OVERWRITE:-true}"

SCRIPT="/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/inference/infer_flux2_base_edit_deblur_dataset.py"
OUTPUT_DIR="${OUTPUT_BASE}/${OUTPUT_NAME}"

cd /mnt/image-edit/datasets/duanyufa/DiffSynth-Studio

if [[ ! -f "${METADATA}" ]]; then
    echo "Missing metadata file: ${METADATA}" >&2
    exit 1
fi
if [[ ! -d "${BASE_MODEL}" ]]; then
    echo "Missing base model directory: ${BASE_MODEL}" >&2
    exit 1
fi
if [[ ! -f "${SCRIPT}" ]]; then
    echo "Missing inference script: ${SCRIPT}" >&2
    exit 1
fi

ARGS=(
    --metadata "${METADATA}"
    --output-dir "${OUTPUT_DIR}"
    --base-model "${BASE_MODEL}"
    --num-inference-steps "${NUM_INFERENCE_STEPS}"
    --cfg-scale "${CFG_SCALE}"
    --embedded-guidance "${EMBEDDED_GUIDANCE}"
    --seed "${SEED}"
)

if [[ -n "${LIMIT}" ]]; then
    ARGS+=(--limit "${LIMIT}")
fi
if [[ "${OVERWRITE}" == "true" ]]; then
    ARGS+=(--overwrite)
fi

echo "=========================================="
echo "  FLUX.2 Klein Base 4B baseline test"
echo "  Base model: ${BASE_MODEL}"
echo "  Metadata: ${METADATA}"
echo "  Output: ${OUTPUT_DIR}"
echo "  GPUs: ${NUM_GPUS}"
echo "  Limit: ${LIMIT:-all}"
echo "  Steps: ${NUM_INFERENCE_STEPS}"
echo "  cfg_scale: ${CFG_SCALE}"
echo "  embedded_guidance: ${EMBEDDED_GUIDANCE}"
echo "=========================================="

accelerate launch --num_processes="${NUM_GPUS}" "${SCRIPT}" "${ARGS[@]}"

count="$(find "${OUTPUT_DIR}" -type f -name "*.png" | wc -l)"
echo "=========================================="
echo "  Base test done: ${count} PNG files"
echo "  Results: ${OUTPUT_DIR}"
echo "=========================================="
