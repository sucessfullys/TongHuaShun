#!/usr/bin/env bash
set -euo pipefail

# Train the FLUX.2 KleinBase4B deblur template with all prepared HR/LR metadata
# datasets, without moving images or modifying the original single-dataset
# training script.
#
# Dataset spec format, repeatable:
#   --dataset /abs/HR_DIR:/abs/LR_DIR:/abs/metadata.jsonl
#
# The wrapper builds one merged metadata file where:
#   - image is converted to the absolute HR image path
#   - template_inputs.image is converted to the absolute LR image path
# Then it launches the original train.py with dataset_base_path=/ so absolute
# paths are accepted by UnifiedDataset.

DIFFSYNTH_ENV="${DIFFSYNTH_ENV:-/mnt/image-edit/datasets/duanyufa/conda_envs/DiffSynth}"
PYTHON_BIN="${DIFFSYNTH_ENV}/bin/python"
TRAIN_SCRIPT="/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/examples/flux2/model_training/train.py"
TEMPLATE_MODEL_DIR="/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/Template-KleinBase4B-Upscaler"
OUTPUT_PATH="/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/outputs/Template-KleinBase4B-Deblur_all_lr1e-5_rep2_ep5_5GPU"

DATASETS=(
  "/mnt/image-edit/datasets/duanyufa/Face/HR:/mnt/image-edit/datasets/duanyufa/Face/LR:/mnt/image-edit/datasets/duanyufa/Face/metadata.jsonl"
  "/mnt/image-edit/datasets/duanyufa/Face/Other_data/Instargram/HR:/mnt/image-edit/datasets/duanyufa/Face/Other_data/Instargram/LR:/mnt/image-edit/datasets/duanyufa/Face/Other_data/Instargram/metadata.jsonl"
  "/mnt/image-edit/datasets/duanyufa/Face/Other_data/Instargram_new1/HR:/mnt/image-edit/datasets/duanyufa/Face/Other_data/Instargram_new1/LR:/mnt/image-edit/datasets/duanyufa/Face/Other_data/Instargram_new1/metadata.jsonl"
  "/mnt/image-edit/datasets/duanyufa/Face/Other_data/xhs/HR:/mnt/image-edit/datasets/duanyufa/Face/Other_data/xhs/LR:/mnt/image-edit/datasets/duanyufa/Face/Other_data/xhs/metadata.jsonl"
  "/mnt/image-edit/datasets/duanyufa/SR_Dataset/4KLSDB/images/HR:/mnt/image-edit/datasets/duanyufa/SR_Dataset/4KLSDB/images/LR:/mnt/image-edit/datasets/duanyufa/SR_Dataset/4KLSDB/images/metadata.jsonl"
  "/mnt/image-edit/datasets/duanyufa/SR_Dataset/DESCAN-18K/DESCAN-18K/HR:/mnt/image-edit/datasets/duanyufa/SR_Dataset/DESCAN-18K/DESCAN-18K/LR:/mnt/image-edit/datasets/duanyufa/SR_Dataset/DESCAN-18K/DESCAN-18K/metadata.jsonl"
  "/mnt/image-edit/datasets/duanyufa/SR_Dataset/SHHQ-1.0/SHHQ-1.0/HR:/mnt/image-edit/datasets/duanyufa/SR_Dataset/SHHQ-1.0/SHHQ-1.0/LR:/mnt/image-edit/datasets/duanyufa/SR_Dataset/SHHQ-1.0/SHHQ-1.0/metadata.jsonl"
  "/mnt/image-edit/datasets/duanyufa/SR_Dataset/文档图片/HR:/mnt/image-edit/datasets/duanyufa/SR_Dataset/文档图片/LR:/mnt/image-edit/datasets/duanyufa/SR_Dataset/文档图片/metadata.jsonl"
  "/mnt/image-edit/datasets/duanyufa/SR_Dataset/VITON-HD/HR:/mnt/image-edit/datasets/duanyufa/SR_Dataset/VITON-HD/LR:/mnt/image-edit/datasets/duanyufa/SR_Dataset/VITON-HD/metadata.jsonl"
  "/mnt/image-edit/datasets/duanyufa/task_shengsheng/Open_dataset/Ours/Version1/HR:/mnt/image-edit/datasets/duanyufa/task_shengsheng/Open_dataset/Ours/Version1/LR:/mnt/image-edit/datasets/duanyufa/task_shengsheng/Open_dataset/Ours/Version1/metadata.jsonl"
  "/mnt/image-edit/datasets/duanyufa/SR_Dataset/FFHQ/ffhq-dataset/HR:/mnt/image-edit/datasets/duanyufa/SR_Dataset/FFHQ/ffhq-dataset/LR:/mnt/image-edit/datasets/duanyufa/SR_Dataset/FFHQ/ffhq-dataset/metadata.jsonl"
  "/mnt/image-edit/datasets/duanyufa/Face/Other_data/Old_Photo_2/HR:/mnt/image-edit/datasets/duanyufa/Face/Other_data/Old_Photo_2/LR:/mnt/image-edit/datasets/duanyufa/Face/Other_data/Old_Photo_2/metadata.jsonl"
)
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
DATASET_REPEAT="${DATASET_REPEAT:-2}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}"
NUM_PROCESSES="${NUM_PROCESSES:-5}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MAX_PIXELS="${MAX_PIXELS:-1572864}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
FLUX2_BASE_DIR="${FLUX2_BASE_DIR:-/mnt/image-edit/datasets/duanyufa/FLUX.2-klein-base-4B}"
FLUX2_COMPONENT_DIR="${FLUX2_COMPONENT_DIR:-/mnt/image-edit/datasets/duanyufa/FLUX.2-klein-base-4B}"
LOG_PATH=""

usage() {
  cat <<'EOF'
Usage:
  bash train_flux2_klein_base_4b_deblur_multi_dataset_all.sh \
    --dataset /abs/hr1:/abs/lr1:/abs/metadata1.jsonl \
    --dataset /abs/hr2:/abs/lr2:/abs/metadata2.jsonl \
    --output /abs/output_dir

Optional:
  --template-model /abs/Template-KleinBase4B-Upscaler
  --lr 1e-5
  --epochs 8
  --repeat 2
  --gpus 0,1,2,3,4,5,6,7
  --num-processes 8
  --max-pixels 1572864
  --mixed-precision bf16
  --grad-accum 1
  --save-steps 5000
  --save-total-limit 5
  --resume-from-checkpoint /abs/previous.safetensors
  --log-path /abs/train.log

Notes:
  Direct run defaults to all prepared datasets:
    Face, Instargram, Instargram_new1, xhs, 4KLSDB, DESCAN-18K,
    SHHQ-1.0, 文档图片, VITON-HD, Version1.
    output=Template-KleinBase4B-Deblur_all_lr1e-5_rep2_ep5
    lr=1e-5 epochs=5 repeat=2 gpus=0,1,2,3,4 num-processes=5 max-pixels=1572864

  Dataset metadata keeps the standard format:
    {"prompt": "...", "image": "xxx.png", "template_inputs": {"image": "/abs/LR/xxx.png", "prompt": "..."}}
  This wrapper rewrites paths in a generated merged metadata file only.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASETS+=("$2"); shift 2 ;;
    --output) OUTPUT_PATH="$2"; shift 2 ;;
    --template-model) TEMPLATE_MODEL_DIR="$2"; shift 2 ;;
    --lr) LEARNING_RATE="$2"; shift 2 ;;
    --epochs) NUM_EPOCHS="$2"; shift 2 ;;
    --repeat) DATASET_REPEAT="$2"; shift 2 ;;
    --gpus) CUDA_VISIBLE_DEVICES_VALUE="$2"; shift 2 ;;
    --num-processes) NUM_PROCESSES="$2"; shift 2 ;;
    --max-pixels) MAX_PIXELS="$2"; shift 2 ;;
    --mixed-precision) MIXED_PRECISION="$2"; shift 2 ;;
    --grad-accum) GRADIENT_ACCUMULATION_STEPS="$2"; shift 2 ;;
    --save-steps) SAVE_STEPS="$2"; shift 2 ;;
    --save-total-limit) SAVE_TOTAL_LIMIT="$2"; shift 2 ;;
    --resume-from-checkpoint) RESUME_FROM_CHECKPOINT="$2"; shift 2 ;;
    --log-path) LOG_PATH="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ "${#DATASETS[@]}" -eq 0 ]]; then
  echo "Error: at least one --dataset is required." >&2
  usage >&2
  exit 1
fi

if [[ -z "${OUTPUT_PATH}" ]]; then
  OUTPUT_PATH="/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/outputs/Template-KleinBase4B-Deblur_multi_lr${LEARNING_RATE}_rep${DATASET_REPEAT}_ep${NUM_EPOCHS}_$(date +%Y%m%d_%H%M%S)"
fi

require_absolute_path() {
  if [[ "$1" != /* ]]; then
    echo "Path must be absolute: $1" >&2
    exit 1
  fi
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Missing required directory: $1" >&2
    exit 1
  fi
}

require_absolute_path "${DIFFSYNTH_ENV}"
require_absolute_path "${TRAIN_SCRIPT}"
require_absolute_path "${TEMPLATE_MODEL_DIR}"
require_absolute_path "${OUTPUT_PATH}"
require_absolute_path "${FLUX2_BASE_DIR}"
require_absolute_path "${FLUX2_COMPONENT_DIR}"
require_file "${PYTHON_BIN}"
require_file "${TRAIN_SCRIPT}"
require_dir "${TEMPLATE_MODEL_DIR}"
require_file "${TEMPLATE_MODEL_DIR}/model.py"
require_file "${TEMPLATE_MODEL_DIR}/model.safetensors"
require_dir "${FLUX2_COMPONENT_DIR}/text_encoder"
require_dir "${FLUX2_COMPONENT_DIR}/tokenizer"
require_file "${FLUX2_COMPONENT_DIR}/vae/diffusion_pytorch_model.safetensors"
require_file "${FLUX2_BASE_DIR}/transformer/diffusion_pytorch_model.safetensors"
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  require_absolute_path "${RESUME_FROM_CHECKPOINT}"
  require_file "${RESUME_FROM_CHECKPOINT}"
fi

for spec in "${DATASETS[@]}"; do
  IFS=':' read -r hr_dir lr_dir metadata_path extra <<< "${spec}"
  if [[ -n "${extra:-}" || -z "${hr_dir:-}" || -z "${lr_dir:-}" || -z "${metadata_path:-}" ]]; then
    echo "Invalid --dataset spec: ${spec}" >&2
    echo "Expected: /abs/HR:/abs/LR:/abs/metadata.jsonl" >&2
    exit 1
  fi
  require_absolute_path "${hr_dir}"
  require_absolute_path "${lr_dir}"
  require_absolute_path "${metadata_path}"
  require_dir "${hr_dir}"
  require_dir "${lr_dir}"
  require_file "${metadata_path}"
done

RUN_TIMESTAMP="$(date '+%Y-%m-%d_%H-%M-%S')"
RUN_STARTED_AT="$(date '+%Y-%m-%d %H:%M:%S')"
MERGE_DIR="${OUTPUT_PATH}/merged_metadata/${RUN_TIMESTAMP}"
MERGED_METADATA="${MERGE_DIR}/metadata.jsonl"
MERGED_SUMMARY="${MERGE_DIR}/summary.json"
LOG_DIR="${OUTPUT_PATH}/logs/${RUN_TIMESTAMP}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/train.log}"

mkdir -p "${MERGE_DIR}" "${OUTPUT_PATH}" "$(dirname "${LOG_PATH}")"
exec > >(tee -a "${LOG_PATH}") 2>&1
echo "===== Multi-dataset training launch ${RUN_STARTED_AT} ====="

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES_VALUE}"
if [[ "${#GPU_IDS[@]}" -ne "${NUM_PROCESSES}" ]]; then
  echo "Error: --gpus has ${#GPU_IDS[@]} IDs but --num-processes=${NUM_PROCESSES}" >&2
  echo "Use matching values, for example: --gpus 0,1,4 --num-processes 3" >&2
  exit 1
fi

# Build the merged metadata using newline-separated specs to avoid shell JSON
# quoting pitfalls.
printf '%s\n' "${DATASETS[@]}" > "${MERGE_DIR}/dataset_specs.txt"

"${PYTHON_BIN}" - "${MERGE_DIR}/dataset_specs.txt" "${MERGED_METADATA}" "${MERGED_SUMMARY}" <<'PY'
from pathlib import Path
import json
import sys
from collections import Counter

spec_file = Path(sys.argv[1])
out_meta = Path(sys.argv[2])
out_summary = Path(sys.argv[3])

records = []
summary = []
seen_hr = set()
missing_hr = []
missing_lr = []
duplicate_hr = []

def resolve_under(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path

for dataset_index, line in enumerate(spec_file.read_text(encoding="utf-8").splitlines()):
    if not line.strip():
        continue
    hr_s, lr_s, meta_s = line.split(":", 2)
    hr_root = Path(hr_s)
    lr_root = Path(lr_s)
    meta_path = Path(meta_s)
    manifest_path = meta_path.parent / "degradation_params.jsonl"
    source_by_filename = {}
    if manifest_path.is_file():
        for raw_manifest in manifest_path.open(encoding="utf-8"):
            if not raw_manifest.strip():
                continue
            manifest_item = json.loads(raw_manifest)
            if "filename" in manifest_item and "source" in manifest_item:
                source_by_filename[manifest_item["filename"]] = manifest_item["source"]
    count = 0
    for line_no, raw in enumerate(meta_path.open(encoding="utf-8"), start=1):
        if not raw.strip():
            continue
        item = json.loads(raw)
        prompt = item.get("prompt", item.get("template_inputs", {}).get("prompt", ""))
        hr_path = resolve_under(hr_root, item["image"])
        if not hr_path.is_file() and item["image"] in source_by_filename:
            hr_path = Path(source_by_filename[item["image"]])
        lr_value = item.get("template_inputs", {}).get("image")
        if not lr_value:
            raise ValueError(f"{meta_path}:{line_no}: missing template_inputs.image")
        lr_path = resolve_under(lr_root, lr_value)
        if not hr_path.is_file():
            missing_hr.append(str(hr_path))
        if not lr_path.is_file():
            missing_lr.append(str(lr_path))
        hr_key = str(hr_path.resolve())
        if hr_key in seen_hr:
            duplicate_hr.append(hr_key)
        seen_hr.add(hr_key)
        new_item = dict(item)
        new_item["image"] = hr_key
        new_item["prompt"] = prompt
        template_inputs = dict(item.get("template_inputs", {}))
        template_inputs["image"] = str(lr_path.resolve())
        template_inputs["prompt"] = prompt
        new_item["template_inputs"] = template_inputs
        records.append(new_item)
        count += 1
    summary.append({
        "dataset_index": dataset_index,
        "hr_dir": str(hr_root),
        "lr_dir": str(lr_root),
        "metadata": str(meta_path),
        "degradation_manifest": str(manifest_path) if manifest_path.is_file() else None,
        "records": count,
    })

if missing_hr or missing_lr:
    out_summary.write_text(json.dumps({
        "total_records_seen": len(records),
        "datasets": summary,
        "missing_hr": missing_hr,
        "missing_lr": missing_lr,
        "missing_hr_count": len(missing_hr),
        "missing_lr_count": len(missing_lr),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    raise SystemExit(
        "Missing files detected before training:\n"
        f"missing_hr={len(missing_hr)} examples={missing_hr[:5]}\n"
        f"missing_lr={len(missing_lr)} examples={missing_lr[:5]}\n"
        f"Full missing-file list was written to: {out_summary}"
    )

out_meta.parent.mkdir(parents=True, exist_ok=True)
with out_meta.open("w", encoding="utf-8") as f:
    for item in records:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

stats = {
    "total_records": len(records),
    "datasets": summary,
    "duplicate_hr_images": len(duplicate_hr),
    "duplicate_hr_examples": duplicate_hr[:10],
    "prompt_counts": Counter(r["prompt"] for r in records).most_common(20),
}
out_summary.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(stats, ensure_ascii=False, indent=2))
PY

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

export DIFFSYNTH_SKIP_DOWNLOAD=true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MODELSCOPE_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PATH="${DIFFSYNTH_ENV}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
export TEMPLATE_MAX_PIXELS="${MAX_PIXELS}"

cd "/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio"

echo "Launching FLUX.2 Deblur Template multi-dataset training"
echo "  python=${PYTHON_BIN}"
echo "  visible_gpus=${CUDA_VISIBLE_DEVICES}"
echo "  processes=${NUM_PROCESSES} mixed_precision=${MIXED_PRECISION}"
echo "  per_gpu_batch=1 gradient_accumulation=${GRADIENT_ACCUMULATION_STEPS} global_batch=$((NUM_PROCESSES * GRADIENT_ACCUMULATION_STEPS))"
echo "  merged_metadata=${MERGED_METADATA}"
echo "  dataset_base_path=/"
echo "  repeat=${DATASET_REPEAT} epochs=${NUM_EPOCHS}"
echo "  max_pixels=${MAX_PIXELS} lr=${LEARNING_RATE}"
echo "  save_steps=${SAVE_STEPS} save_total_limit=${SAVE_TOTAL_LIMIT}"
echo "  resume_from_checkpoint=${RESUME_FROM_CHECKPOINT:-<none>}"
echo "  output=${OUTPUT_PATH}"
echo "  log=${LOG_PATH}"
echo "  merge_summary=${MERGED_SUMMARY}"

TRAIN_EXTRA_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  TRAIN_EXTRA_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
if [[ -n "${SAVE_STEPS}" ]]; then
  TRAIN_EXTRA_ARGS+=(--save_steps "${SAVE_STEPS}")
fi
if [[ -n "${SAVE_TOTAL_LIMIT}" ]]; then
  TRAIN_EXTRA_ARGS+=(--save_total_limit "${SAVE_TOTAL_LIMIT}")
fi

"${PYTHON_BIN}" -m accelerate.commands.accelerate_cli launch \
  --multi_gpu \
  --num_machines 1 \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision "${MIXED_PRECISION}" \
  "${TRAIN_SCRIPT}" \
  --dataset_base_path "/" \
  --dataset_metadata_path "${MERGED_METADATA}" \
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
  "${TRAIN_EXTRA_ARGS[@]}" \
  --remove_prefix_in_ckpt "pipe.template_model." \
  --output_path "${OUTPUT_PATH}" \
  --trainable_models "template_model" \
  --use_gradient_checkpointing \
  --find_unused_parameters
