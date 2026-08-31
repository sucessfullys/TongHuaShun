#!/usr/bin/env bash
set -euo pipefail

# Offline fine-tuning for the official Template-KleinBase4B-Upscaler.
#
# This script uses the unmodified official FLUX.2 training entrypoint. It keeps
# the official Template full-finetuning setup while adjusting repeat/epochs for
# the 3,000-pair deblur dataset. The dataset must use the official metadata format:
# {"prompt":"Remove blur...","image":"hq/0001.png","template_inputs":{"image":"/absolute/path/to/lq/0001.png","prompt":"Remove blur..."}}
#
# Usage:
#   bash /mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/scripts/train_flux2_klein_base_4b_deblur_template.sh
# Optional overrides:
#   bash /mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/scripts/train_flux2_klein_base_4b_deblur_template.sh \
#     [/absolute/dataset_root] [/absolute/metadata.jsonl] \
#     [/absolute/Template-KleinBase4B-Upscaler] [/absolute/output_dir]

DIFFSYNTH_ENV="${DIFFSYNTH_ENV:-/mnt/image-edit/datasets/duanyufa/conda_envs/DiffSynth}"
PYTHON_BIN="${DIFFSYNTH_ENV}/bin/python"
TRAIN_SCRIPT="/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/examples/flux2/model_training/train.py"
DATASET_BASE_PATH="${1:-/mnt/image-edit/datasets/duanyufa/Face/HR}"
DATASET_METADATA_PATH="${2:-/mnt/image-edit/datasets/duanyufa/Face/metadata.jsonl}"
TEMPLATE_MODEL_DIR="${3:-/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/Template-KleinBase4B-Upscaler}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_EPOCHS="${NUM_EPOCHS:-8}"
DATASET_REPEAT="${DATASET_REPEAT:-2}"
OUTPUT_PATH="${4:-/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/outputs/Template-KleinBase4B-Deblur_lr${LEARNING_RATE}_rep${DATASET_REPEAT}_ep${NUM_EPOCHS}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MAX_PIXELS="${MAX_PIXELS:-2097152}"
export TEMPLATE_MAX_PIXELS="${MAX_PIXELS}"
RUN_TIMESTAMP="$(date '+%Y-%m-%d_%H-%M-%S')"
RUN_STARTED_AT="$(date '+%Y-%m-%d %H:%M:%S')"
LOG_DIR="${OUTPUT_PATH}/logs/${RUN_TIMESTAMP}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/train.log}"

# Give every launch its own timestamped log directory. Prepare the directory
# and log redirection before printing the launch marker, so it is the first
# entry in the new log file.
mkdir -p "${OUTPUT_PATH}" "$(dirname "${LOG_PATH}")"
exec > >(tee -a "${LOG_PATH}") 2>&1
echo "===== Training launch ${RUN_STARTED_AT} ====="

# The Base checkpoint supplies the trainable architecture's DiT, while the
# non-Base Klein checkpoint supplies the official text encoder, VAE and tokenizer.
FLUX2_BASE_DIR="${FLUX2_BASE_DIR:-/mnt/image-edit/datasets/duanyufa/FLUX.2-klein-base-4B}"
FLUX2_COMPONENT_DIR="${FLUX2_COMPONENT_DIR:-/mnt/image-edit/datasets/duanyufa/FLUX.2-klein-base-4B}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required offline file: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Missing required offline directory: $1" >&2
    exit 1
  fi
}

require_absolute_path() {
  if [[ "$1" != /* ]]; then
    echo "Path must be absolute: $1" >&2
    exit 1
  fi
}

require_absolute_path "${DIFFSYNTH_ENV}"
require_absolute_path "${TRAIN_SCRIPT}"
require_absolute_path "${DATASET_BASE_PATH}"
require_absolute_path "${DATASET_METADATA_PATH}"
require_absolute_path "${TEMPLATE_MODEL_DIR}"
require_absolute_path "${OUTPUT_PATH}"
require_absolute_path "${FLUX2_BASE_DIR}"
require_absolute_path "${FLUX2_COMPONENT_DIR}"
require_absolute_path "${LOG_PATH}"
require_dir "${DATASET_BASE_PATH}"
require_file "${DATASET_METADATA_PATH}"
require_file "${PYTHON_BIN}"
require_file "${TRAIN_SCRIPT}"
require_dir "${TEMPLATE_MODEL_DIR}"
require_file "${TEMPLATE_MODEL_DIR}/model.py"
require_file "${TEMPLATE_MODEL_DIR}/model.safetensors"
require_dir "${FLUX2_COMPONENT_DIR}/text_encoder"
require_dir "${FLUX2_COMPONENT_DIR}/tokenizer"
require_file "${FLUX2_COMPONENT_DIR}/vae/diffusion_pytorch_model.safetensors"
require_file "${FLUX2_BASE_DIR}/transformer/diffusion_pytorch_model.safetensors"

# Build --model_paths as valid JSON and verify that all text-encoder shards are
# local before Accelerate starts any worker processes.
MODEL_PATHS="$({ "${PYTHON_BIN}" - "${FLUX2_COMPONENT_DIR}" "${FLUX2_BASE_DIR}" <<'PY'
import glob
import json
import os
import sys

components, base = sys.argv[1:]
text_encoder = sorted(glob.glob(os.path.join(components, "text_encoder", "*.safetensors")))
if not text_encoder:
    raise SystemExit(f"No local text-encoder weights found in {components}/text_encoder")
paths = [
    text_encoder,
    os.path.join(base, "transformer", "diffusion_pytorch_model.safetensors"),
    os.path.join(components, "vae", "diffusion_pytorch_model.safetensors"),
]
print(json.dumps(paths))
PY
} )"

# Defense in depth: even an accidental model_id cannot trigger network access.
export DIFFSYNTH_SKIP_DOWNLOAD=true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MODELSCOPE_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PATH="${DIFFSYNTH_ENV}/bin:${PATH}"

cd "/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio"

echo "Launching FLUX.2 Deblur Template training"
echo "  python=${PYTHON_BIN}"
echo "  visible_gpus=${CUDA_VISIBLE_DEVICES}"
echo "  processes=${NUM_PROCESSES} mixed_precision=${MIXED_PRECISION}"
echo "  per_gpu_batch=1 gradient_accumulation=${GRADIENT_ACCUMULATION_STEPS} global_batch=$((NUM_PROCESSES * GRADIENT_ACCUMULATION_STEPS))"
echo "  dataset=${DATASET_METADATA_PATH} repeat=${DATASET_REPEAT} epochs=${NUM_EPOCHS}"
echo "  max_pixels=${MAX_PIXELS} lr=${LEARNING_RATE}"
echo "  output=${OUTPUT_PATH}"
echo "  log=${LOG_PATH}"

# Core parameters follow the official Upscaler setup. Dataset repeat and epoch
# count are adjusted from 50x2 to 1x5 for the 3,000-pair dataset.
"${PYTHON_BIN}" -m accelerate.commands.accelerate_cli launch \
  --multi_gpu \
  --num_machines 1 \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision "${MIXED_PRECISION}" \
  "${TRAIN_SCRIPT}" \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${DATASET_METADATA_PATH}" \
  --extra_inputs "template_inputs" \
  --max_pixels "${MAX_PIXELS}" \
  --dataset_repeat "${DATASET_REPEAT}" \
  --model_paths "${MODEL_PATHS}" \
  --template_model_id_or_path "${TEMPLATE_MODEL_DIR}" \
  --tokenizer_path "${FLUX2_COMPONENT_DIR}/tokenizer" \
  --learning_rate "${LEARNING_RATE}" \
  --num_epochs "${NUM_EPOCHS}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --log_every 1 \
  --remove_prefix_in_ckpt "pipe.template_model." \
  --output_path "${OUTPUT_PATH}" \
  --trainable_models "template_model" \
  --use_gradient_checkpointing \
  --find_unused_parameters
