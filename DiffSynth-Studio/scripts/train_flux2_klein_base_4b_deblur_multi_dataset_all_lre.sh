#!/usr/bin/env bash
set -euo pipefail

# All-dataset LRE training wrapper.
#
# This keeps train_flux2_klein_base_4b_deblur_multi_dataset_lre.sh unchanged,
# and extends its default Face/Instargram/xhs datasets with the remaining
# datasets from train_flux2_klein_base_4b_deblur_multi_dataset_all.sh.

BASE_LRE_SCRIPT="/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/scripts/train_flux2_klein_base_4b_deblur_multi_dataset_lre.sh"

LEARNING_RATE="${LEARNING_RATE:-5e-6}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
DATASET_REPEAT="${DATASET_REPEAT:-1}"
LRE_STRENGTH="${LRE_STRENGTH:-0.7}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

OUTPUT_PATH="${OUTPUT_PATH:-/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/outputs/Template-KleinBase4B-Deblur_all_LRE_lre${LRE_STRENGTH}_lr${LEARNING_RATE}_rep${DATASET_REPEAT}_ep${NUM_EPOCHS}_${RUN_ID}}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

require_file "${BASE_LRE_SCRIPT}"

if [[ -e "${OUTPUT_PATH}" ]]; then
  echo "Refusing to reuse existing OUTPUT_PATH: ${OUTPUT_PATH}" >&2
  echo "Please set a new RUN_ID or OUTPUT_PATH." >&2
  exit 1
fi

export LEARNING_RATE
export NUM_EPOCHS
export DATASET_REPEAT
export LRE_STRENGTH
export OUTPUT_PATH
export RUN_ID

bash "${BASE_LRE_SCRIPT}" \
  --dataset "/mnt/image-edit/datasets/duanyufa/Face/Other_data/Instargram_new1/HR:/mnt/image-edit/datasets/duanyufa/Face/Other_data/Instargram_new1/LR:/mnt/image-edit/datasets/duanyufa/Face/Other_data/Instargram_new1/metadata.jsonl" \
  --dataset "/mnt/image-edit/datasets/duanyufa/SR_Dataset/4KLSDB/images/HR:/mnt/image-edit/datasets/duanyufa/SR_Dataset/4KLSDB/images/LR:/mnt/image-edit/datasets/duanyufa/SR_Dataset/4KLSDB/images/metadata.jsonl" \
  --dataset "/mnt/image-edit/datasets/duanyufa/SR_Dataset/DESCAN-18K/DESCAN-18K/HR:/mnt/image-edit/datasets/duanyufa/SR_Dataset/DESCAN-18K/DESCAN-18K/LR:/mnt/image-edit/datasets/duanyufa/SR_Dataset/DESCAN-18K/DESCAN-18K/metadata.jsonl" \
  --dataset "/mnt/image-edit/datasets/duanyufa/SR_Dataset/SHHQ-1.0/SHHQ-1.0/HR:/mnt/image-edit/datasets/duanyufa/SR_Dataset/SHHQ-1.0/SHHQ-1.0/LR:/mnt/image-edit/datasets/duanyufa/SR_Dataset/SHHQ-1.0/SHHQ-1.0/metadata.jsonl" \
  --dataset "/mnt/image-edit/datasets/duanyufa/SR_Dataset/文档图片/HR:/mnt/image-edit/datasets/duanyufa/SR_Dataset/文档图片/LR:/mnt/image-edit/datasets/duanyufa/SR_Dataset/文档图片/metadata.jsonl" \
  --dataset "/mnt/image-edit/datasets/duanyufa/SR_Dataset/VITON-HD/HR:/mnt/image-edit/datasets/duanyufa/SR_Dataset/VITON-HD/LR:/mnt/image-edit/datasets/duanyufa/SR_Dataset/VITON-HD/metadata.jsonl" \
  --dataset "/mnt/image-edit/datasets/duanyufa/task_shengsheng/Open_dataset/Ours/Version1/HR:/mnt/image-edit/datasets/duanyufa/task_shengsheng/Open_dataset/Ours/Version1/LR:/mnt/image-edit/datasets/duanyufa/task_shengsheng/Open_dataset/Ours/Version1/metadata.jsonl" \
  --dataset "/mnt/image-edit/datasets/duanyufa/SR_Dataset/FFHQ/ffhq-dataset/HR:/mnt/image-edit/datasets/duanyufa/SR_Dataset/FFHQ/ffhq-dataset/LR:/mnt/image-edit/datasets/duanyufa/SR_Dataset/FFHQ/ffhq-dataset/metadata.jsonl" \
  --dataset "/mnt/image-edit/datasets/duanyufa/Face/Other_data/Old_Photo_2/HR:/mnt/image-edit/datasets/duanyufa/Face/Other_data/Old_Photo_2/LR:/mnt/image-edit/datasets/duanyufa/Face/Other_data/Old_Photo_2/metadata.jsonl" \
  "$@"
