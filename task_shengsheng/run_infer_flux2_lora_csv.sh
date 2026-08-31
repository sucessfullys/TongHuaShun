#!/bin/bash
# ============================================================================
# Launch script: FLUX.2-klein-9B + DreamFace LoRA batch i2i CSV inference
#
# Task: home_img (reference face) + replaced_prompt → output image
#
# Usage:
#   bash run_infer_flux2_lora_csv.sh
#
# This runs infer_flux2_lora.py with all default paths.
# Edit the variables below to customise the run.
# ============================================================================
set -euo pipefail

# ---- Script directory ------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Python environment ----------------------------------------------
# Activate the conda environment that has diffusers / torch / transformers.
# Adjust this line to match your setup.
if command -v conda &>/dev/null; then
    # shellcheck disable=SC1090
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate /mnt/data/image-edit/datasets/shensheng/env/flux-http 2>/dev/null || true
fi

# ---- Optional overrides ----------------------------------------------
# Uncomment and change any of these to override defaults
# BASE_MODEL="/mnt/data/image-edit/datasets/shensheng/models/black-forest-labs/FLUX.2-klein-9B"
# LORA_PATH="/mnt/data/image-edit/datasets/shensheng/models/hithink-image-labs/DreamFace_lora/v2.1/diffusers_lora.safetensors"
# CSV_PATH="/mnt/data/image-edit/datasets/shensheng/code/stable/DreamFaceOmini/exp_out/csv-fluxout-1k/csv-fluxout.csv"
# OUTPUT_DIR="/mnt/image-edit/datasets/duanyufa/task_shengsheng/output/lora_flux_9B"
# PROMPT_COLUMN="replaced_prompt"
# REFERENCE_IMAGE_COLUMN="home_img"
# MAX_SAMPLES=""          # set to e.g. 3 for a quick test
# STEPS=4                 # FLUX.2-klein is step-distilled to 4 steps
# GUIDANCE=1.0            # flow-matching, no CFG needed
# HEIGHT=1024
# WIDTH=1024
# SEED=42
# DEVICE="cuda"
# DTYPE="bfloat16"

# ---- Run inference ---------------------------------------------------
echo "=============================================="
echo " FLUX.2-klein-9B + DreamFace LoRA (i2i mode)"
echo "=============================================="

# Build argument list
ARGS=()
ARGS+=(--base_model "${BASE_MODEL:-/mnt/data/image-edit/datasets/shensheng/models/black-forest-labs/FLUX.2-klein-9B}")
ARGS+=(--lora_path  "${LORA_PATH:-/mnt/data/image-edit/datasets/shensheng/models/hithink-image-labs/DreamFace_lora/v2.1/diffusers_lora.safetensors}")
ARGS+=(--csv_path   "${CSV_PATH:-/mnt/data/image-edit/datasets/shensheng/code/stable/DreamFaceOmini/exp_out/csv-fluxout-1k/csv-fluxout.csv}")
ARGS+=(--output_dir "${OUTPUT_DIR:-/mnt/image-edit/datasets/duanyufa/task_shengsheng/output/lora_flux_9B}")
ARGS+=(--num_inference_steps "${STEPS:-4}")
ARGS+=(--guidance_scale       "${GUIDANCE:-1.0}")
ARGS+=(--height "${HEIGHT:-1024}")
ARGS+=(--width  "${WIDTH:-1024}")
ARGS+=(--seed   "${SEED:-42}")
ARGS+=(--device "${DEVICE:-cuda}")
ARGS+=(--dtype  "${DTYPE:-bfloat16}")
ARGS+=(--reference_image_column "${REFERENCE_IMAGE_COLUMN:-home_img}")

if [ -n "${MAX_SAMPLES:-}" ]; then
    ARGS+=(--max_samples "$MAX_SAMPLES")
fi
if [ -n "${PROMPT_COLUMN:-}" ]; then
    ARGS+=(--prompt_column "$PROMPT_COLUMN")
fi

echo "Command: python infer_flux2_lora.py ${ARGS[*]}"
echo ""

python infer_flux2_lora.py "${ARGS[@]}"

echo ""
echo "Done."
