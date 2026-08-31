#!/bin/bash
# Compare all .safetensors checkpoints in one checkpoint directory.
# Usage:
#   bash /mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/test_all/run_compare_epochs.sh
#
# Override examples:
#   CHECKPOINT_DIR=/abs/output_dir NUM_GPUS=4 LIMIT=10 bash test_all/run_compare_epochs.sh
#   CHECKPOINT_DIR=/abs/output_dir CHECKPOINT_NAMES=step-34000 NUM_GPUS=4 bash test_all/run_compare_epochs.sh
#   INFER_MODE=lre LRE_STRENGTH=0.8 CHECKPOINT_DIR=/abs/lre_output NUM_GPUS=4 bash test_all/run_compare_epochs.sh

set -euo pipefail

CHECKPOINT_DIR="${CHECKPOINT_DIR:-outputs/Template-KleinBase4B-Deblur_full_1e5}"
METADATA="${METADATA:-/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/test_all/测试集合/metadata.jsonl}"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/test_all}"
INFER_MODE="${INFER_MODE:-normal}"
INFER_SCRIPT="${INFER_SCRIPT:-}"
NUM_GPUS="${NUM_GPUS:-8}"
LIMIT="${LIMIT:-}"
SEED="${SEED:-42}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
CFG_SCALE="${CFG_SCALE:-4.0}"
EMBEDDED_GUIDANCE="${EMBEDDED_GUIDANCE:-4.0}"
LRE_STRENGTH="${LRE_STRENGTH:-0.8}"
CHECKPOINT_NAMES="${CHECKPOINT_NAMES:-}"

cd /mnt/image-edit/datasets/duanyufa/DiffSynth-Studio

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
    echo "Missing checkpoint directory: ${CHECKPOINT_DIR}" >&2
    exit 1
fi
if [[ ! -f "${METADATA}" ]]; then
    echo "Missing metadata file: ${METADATA}" >&2
    exit 1
fi

if [[ -z "${INFER_SCRIPT}" ]]; then
    case "${INFER_MODE}" in
        normal)
            INFER_SCRIPT="scripts/infer_flux2_klein_base_4b_deblur_real_dataset.py"
            ;;
        lre)
            INFER_SCRIPT="scripts/infer_flux2_klein_base_4b_deblur_dataset_lre.py"
            ;;
        *)
            echo "Invalid INFER_MODE=${INFER_MODE}. Expected: normal or lre" >&2
            exit 1
            ;;
    esac
fi
if [[ ! -f "${INFER_SCRIPT}" ]]; then
    echo "Missing inference script: ${INFER_SCRIPT}" >&2
    exit 1
fi

CHECKPOINT_NAME="$(basename "${CHECKPOINT_DIR%/}")"
OUTPUT_ROOT="${OUTPUT_BASE}/${CHECKPOINT_NAME}"

if [[ -n "${CHECKPOINT_NAMES}" ]]; then
    CHECKPOINTS=()
    IFS=',' read -ra REQUESTED_CHECKPOINTS <<< "${CHECKPOINT_NAMES}"
    for name in "${REQUESTED_CHECKPOINTS[@]}"; do
        name="$(echo "${name}" | xargs)"
        [[ -z "${name}" ]] && continue
        if [[ "${name}" != *.safetensors ]]; then
            name="${name}.safetensors"
        fi
        checkpoint_path="${CHECKPOINT_DIR}/${name}"
        if [[ ! -f "${checkpoint_path}" ]]; then
            echo "Missing requested checkpoint: ${checkpoint_path}" >&2
            exit 1
        fi
        CHECKPOINTS+=("${checkpoint_path}")
    done
else
    mapfile -t CHECKPOINTS < <(find "${CHECKPOINT_DIR}" -maxdepth 1 -type f -name "*.safetensors" | sort -V)
fi
if [[ "${#CHECKPOINTS[@]}" -eq 0 ]]; then
    echo "No .safetensors checkpoints found in: ${CHECKPOINT_DIR}" >&2
    exit 1
fi

echo "=========================================="
echo "  Checkpoint Comparison Test"
echo "  Metadata: ${METADATA}"
echo "  Limit: ${LIMIT:-all}"
echo "  GPUs: ${NUM_GPUS}"
echo "  Inference mode: ${INFER_MODE}"
echo "  Inference script: ${INFER_SCRIPT}"
echo "  Steps: ${NUM_INFERENCE_STEPS}"
echo "  cfg_scale: ${CFG_SCALE}"
echo "  embedded_guidance: ${EMBEDDED_GUIDANCE}"
if [[ "${INFER_MODE}" == "lre" || "${INFER_SCRIPT}" == *"_lre.py" ]]; then
    echo "  lre_strength: ${LRE_STRENGTH}"
fi
echo "  Checkpoint dir: ${CHECKPOINT_DIR}"
echo "  Checkpoint names: ${CHECKPOINT_NAMES:-all}"
echo "  Output root: ${OUTPUT_ROOT}"
echo "  Checkpoints: ${#CHECKPOINTS[@]}"
echo "=========================================="

mkdir -p "${OUTPUT_ROOT}"

for checkpoint in "${CHECKPOINTS[@]}"; do
    checkpoint_file="$(basename "${checkpoint}")"
    checkpoint_name="${checkpoint_file%.safetensors}"
    output_dir="${OUTPUT_ROOT}/${checkpoint_name}"

    echo ""
    echo ">>> [$(date '+%H:%M:%S')] Testing ${checkpoint_file} <<<"

    INFER_ARGS=()
    if [[ -n "${LIMIT}" ]]; then
        INFER_ARGS+=(--limit "${LIMIT}")
    fi
    if [[ "${INFER_MODE}" == "lre" || "${INFER_SCRIPT}" == *"_lre.py" ]]; then
        INFER_ARGS+=(--lre-strength "${LRE_STRENGTH}")
    fi

    accelerate launch --num_processes=${NUM_GPUS} \
        "${INFER_SCRIPT}" \
        --checkpoint "${checkpoint}" \
        --metadata "${METADATA}" \
        --output-dir "${output_dir}" \
        --num-inference-steps "${NUM_INFERENCE_STEPS}" \
        --cfg-scale "${CFG_SCALE}" \
        --embedded-guidance "${EMBEDDED_GUIDANCE}" \
        --seed ${SEED} \
        --overwrite \
        "${INFER_ARGS[@]}"

    count="$(find "${output_dir}" -type f -name "*.png" | wc -l)"
    echo ">>> [$(date '+%H:%M:%S')] ${checkpoint_name} done: ${count} images"
done

echo ""
echo "=========================================="
echo "  All done! Results: ${OUTPUT_ROOT}/"
echo "=========================================="
find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort -V
