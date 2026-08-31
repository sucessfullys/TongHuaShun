#!/usr/bin/env bash
set -euo pipefail

cd /mnt/image-edit/datasets/duanyufa/DiffSynth-Studio

CONFIG="configs/train/flux2_klein_base_4b_self_flow.yaml"
ACCELERATE_CONFIG="configs/train/accelerate_flux2_klein_base_4b_self_flow_zero3.yaml"
TRAIN_SCRIPT="examples/flux2/model_training/train_self_flow.py"
BASE_MODEL="/mnt/image-edit/datasets/duanyufa/FLUX.2-klein-base-4B"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/image-edit/datasets/duanyufa/outputs/flux2_klein_base_4b_self_flow_gamma1}"

if ! python -c "import deepspeed" >/dev/null 2>&1; then
  echo "DeepSpeed is required. Install it in the active environment first."
  exit 1
fi

MODE="train"
if [[ $# -gt 0 && ("$1" == "train" || "$1" == "smoke" || "$1" == "--smoke") ]]; then
  MODE="$1"
  shift
fi

if [[ "${MODE}" == "smoke" || "${MODE}" == "--smoke" ]]; then
  echo "Running the 8-GPU FLUX.2-klein-base-4B Self-Flow smoke test."
  accelerate launch \
    --config_file "${ACCELERATE_CONFIG}" \
    "${TRAIN_SCRIPT}" \
    --config "${CONFIG}" \
    --base_model "${BASE_MODEL}" \
    --output_dir "${OUTPUT_DIR}/smoke" \
    --dataset_type dummy \
    --height 128 \
    --width 128 \
    --max_steps 2 \
    --checkpointing_steps 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --use_gradient_checkpointing \
    "$@"
else
  echo "Running 8-GPU FLUX.2-klein-base-4B full-parameter Self-Flow training."
  accelerate launch \
    --config_file "${ACCELERATE_CONFIG}" \
    "${TRAIN_SCRIPT}" \
    --config "${CONFIG}" \
    --base_model "${BASE_MODEL}" \
    --output_dir "${OUTPUT_DIR}" \
    --use_gradient_checkpointing \
    "$@"
fi
