#!/usr/bin/env bash
set -euo pipefail

# Multi-dataset paired restoration LoRA for FLUX.2 Klein Base 4B.
#
# This is the edit-image conditioning counterpart of:
#   train_flux2_klein_base_4b_deblur_multi_dataset_all.sh
#
# Difference:
#   - target HR image is saved as metadata["image"]
#   - aligned LR image is saved as metadata["edit_image"]
#   - FLUX.2 encodes edit_image with VAE, concatenates edit_latents with noisy
#     target latents along the token sequence dimension, and feeds both into DiT
#   - edit_image_auto_resize is explicitly disabled so HR/LR paired restoration
#     keeps the same dataset-level MAX_PIXELS crop/resize policy
#   - only DiT LoRA parameters are trainable; base DiT/VAE/text encoder stay frozen
#
# Dataset spec format, repeatable:
#   --dataset /abs/HR_DIR:/abs/LR_DIR:/abs/metadata.jsonl

REPO_ROOT="/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio"
DIFFSYNTH_ENV="${DIFFSYNTH_ENV:-/mnt/image-edit/datasets/duanyufa/conda_envs/DiffSynth}"
PYTHON_BIN="${DIFFSYNTH_ENV}/bin/python"
TRAIN_SCRIPT="${REPO_ROOT}/examples/flux2/model_training/train.py"
OUTPUT_PATH="/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/outputs/FLUX2_KleinBase4B_Deblur_all_edit_lora_lr2e-5_r32_rep1_ep4_px2097152_noauto"

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

LEARNING_RATE="${LEARNING_RATE:-2e-5}"
LORA_RANK="${LORA_RANK:-32}"
NUM_EPOCHS="${NUM_EPOCHS:-4}"
DATASET_REPEAT="${DATASET_REPEAT:-1}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MAX_PIXELS="${MAX_PIXELS:-2097152}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-100}"
LORA_CHECKPOINT="${LORA_CHECKPOINT:-}"
FLUX2_BASE_DIR="${FLUX2_BASE_DIR:-/mnt/image-edit/datasets/duanyufa/FLUX.2-klein-base-4B}"
LOG_PATH=""

usage() {
  cat <<'EOF'
Usage:
  bash train_flux2_klein_base_4b_deblur_multi_dataset_all_edit_lora.sh \
    --output /abs/output_dir \
    --lr 2e-5 \
    --lora-rank 32 \
    --epochs 4 \
    --repeat 1 \
    --gpus 0,1,2,3 \
    --num-processes 4

Optional:
  --dataset /abs/HR:/abs/LR:/abs/metadata.jsonl
  --max-pixels 2097152
  --grad-accum 1
  --mixed-precision bf16
  --save-steps 2000
  --save-total-limit 100
  --lora-checkpoint /abs/previous_lora.safetensors
  --log-path /abs/train.log

Compared with the template multi-dataset script, this trains native FLUX.2
edit_image conditioning LoRA: VAE(edit_image) tokens are concatenated with
noisy target image tokens before DiT self-attention. edit_image_auto_resize is
disabled in metadata for paired restoration alignment.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASETS+=("$2"); shift 2 ;;
    --output) OUTPUT_PATH="$2"; shift 2 ;;
    --lr) LEARNING_RATE="$2"; shift 2 ;;
    --lora-rank) LORA_RANK="$2"; shift 2 ;;
    --epochs) NUM_EPOCHS="$2"; shift 2 ;;
    --repeat) DATASET_REPEAT="$2"; shift 2 ;;
    --gpus) CUDA_VISIBLE_DEVICES_VALUE="$2"; shift 2 ;;
    --num-processes) NUM_PROCESSES="$2"; shift 2 ;;
    --max-pixels) MAX_PIXELS="$2"; shift 2 ;;
    --mixed-precision) MIXED_PRECISION="$2"; shift 2 ;;
    --grad-accum) GRADIENT_ACCUMULATION_STEPS="$2"; shift 2 ;;
    --save-steps) SAVE_STEPS="$2"; shift 2 ;;
    --save-total-limit) SAVE_TOTAL_LIMIT="$2"; shift 2 ;;
    --lora-checkpoint) LORA_CHECKPOINT="$2"; shift 2 ;;
    --log-path) LOG_PATH="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

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
require_absolute_path "${OUTPUT_PATH}"
require_absolute_path "${FLUX2_BASE_DIR}"
require_file "${PYTHON_BIN}"
require_file "${TRAIN_SCRIPT}"
require_dir "${FLUX2_BASE_DIR}/text_encoder"
require_dir "${FLUX2_BASE_DIR}/tokenizer"
require_file "${FLUX2_BASE_DIR}/vae/diffusion_pytorch_model.safetensors"
require_file "${FLUX2_BASE_DIR}/transformer/diffusion_pytorch_model.safetensors"
if [[ -n "${LORA_CHECKPOINT}" ]]; then
  require_absolute_path "${LORA_CHECKPOINT}"
  require_file "${LORA_CHECKPOINT}"
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

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES_VALUE}"
if [[ "${#GPU_IDS[@]}" -ne "${NUM_PROCESSES}" ]]; then
  echo "Error: --gpus has ${#GPU_IDS[@]} IDs but --num-processes=${NUM_PROCESSES}" >&2
  echo "Use matching values, for example: --gpus 0,1,4 --num-processes 3" >&2
  exit 1
fi

RUN_TIMESTAMP="$(date '+%Y-%m-%d_%H-%M-%S')"
RUN_STARTED_AT="$(date '+%Y-%m-%d %H:%M:%S')"
MERGE_DIR="${OUTPUT_PATH}/merged_metadata/${RUN_TIMESTAMP}"
MERGED_METADATA="${MERGE_DIR}/metadata.jsonl"
MERGED_SUMMARY="${MERGE_DIR}/summary.json"
LOG_DIR="${OUTPUT_PATH}/logs/${RUN_TIMESTAMP}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/train.log}"

mkdir -p "${MERGE_DIR}" "${OUTPUT_PATH}" "$(dirname "${LOG_PATH}")"
exec > >(tee -a "${LOG_PATH}") 2>&1
echo "===== Multi-dataset edit_image LoRA launch ${RUN_STARTED_AT} ====="

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
        lr_value = item.get("edit_image") or item.get("template_inputs", {}).get("image")
        if not lr_value:
            raise ValueError(f"{meta_path}:{line_no}: missing edit_image or template_inputs.image")
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
        new_item["edit_image"] = str(lr_path.resolve())
        new_item["edit_image_auto_resize"] = False
        new_item["prompt"] = prompt
        new_item.pop("template_inputs", None)
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

MODEL_PATHS="$({ "${PYTHON_BIN}" - "${FLUX2_BASE_DIR}" <<'PY'
import glob
import json
import os
import sys

base = sys.argv[1]
text_encoder = sorted(glob.glob(os.path.join(base, "text_encoder", "*.safetensors")))
if not text_encoder:
    raise SystemExit(f"No local text-encoder weights found in {base}/text_encoder")
paths = [
    text_encoder,
    os.path.join(base, "transformer", "diffusion_pytorch_model.safetensors"),
    os.path.join(base, "vae", "diffusion_pytorch_model.safetensors"),
]
print(json.dumps(paths))
PY
} )"

LORA_TARGET_MODULES="to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out,linear_in,linear_out,to_qkv_mlp_proj,single_transformer_blocks.0.attn.to_out,single_transformer_blocks.1.attn.to_out,single_transformer_blocks.2.attn.to_out,single_transformer_blocks.3.attn.to_out,single_transformer_blocks.4.attn.to_out,single_transformer_blocks.5.attn.to_out,single_transformer_blocks.6.attn.to_out,single_transformer_blocks.7.attn.to_out,single_transformer_blocks.8.attn.to_out,single_transformer_blocks.9.attn.to_out,single_transformer_blocks.10.attn.to_out,single_transformer_blocks.11.attn.to_out,single_transformer_blocks.12.attn.to_out,single_transformer_blocks.13.attn.to_out,single_transformer_blocks.14.attn.to_out,single_transformer_blocks.15.attn.to_out,single_transformer_blocks.16.attn.to_out,single_transformer_blocks.17.attn.to_out,single_transformer_blocks.18.attn.to_out,single_transformer_blocks.19.attn.to_out"

export DIFFSYNTH_SKIP_DOWNLOAD=true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MODELSCOPE_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PATH="${DIFFSYNTH_ENV}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"

cd "${REPO_ROOT}"

echo "Launching FLUX.2 native edit_image LoRA multi-dataset training"
echo "  python=${PYTHON_BIN}"
echo "  visible_gpus=${CUDA_VISIBLE_DEVICES}"
echo "  processes=${NUM_PROCESSES} mixed_precision=${MIXED_PRECISION}"
echo "  per_gpu_batch=1 gradient_accumulation=${GRADIENT_ACCUMULATION_STEPS} global_batch=$((NUM_PROCESSES * GRADIENT_ACCUMULATION_STEPS))"
echo "  merged_metadata=${MERGED_METADATA}"
echo "  dataset_base_path=/"
echo "  repeat=${DATASET_REPEAT} epochs=${NUM_EPOCHS}"
echo "  max_pixels=${MAX_PIXELS} lr=${LEARNING_RATE}"
echo "  lora_rank=${LORA_RANK}"
echo "  save_steps=${SAVE_STEPS} save_total_limit=${SAVE_TOTAL_LIMIT}"
echo "  lora_checkpoint=${LORA_CHECKPOINT:-<none>}"
echo "  output=${OUTPUT_PATH}"
echo "  log=${LOG_PATH}"
echo "  merge_summary=${MERGED_SUMMARY}"

TRAIN_EXTRA_ARGS=()
if [[ -n "${SAVE_STEPS}" ]]; then
  TRAIN_EXTRA_ARGS+=(--save_steps "${SAVE_STEPS}")
fi
if [[ -n "${SAVE_TOTAL_LIMIT}" ]]; then
  TRAIN_EXTRA_ARGS+=(--save_total_limit "${SAVE_TOTAL_LIMIT}")
fi
if [[ -n "${LORA_CHECKPOINT}" ]]; then
  TRAIN_EXTRA_ARGS+=(--lora_checkpoint "${LORA_CHECKPOINT}")
fi

"${PYTHON_BIN}" -m accelerate.commands.accelerate_cli launch \
  --multi_gpu \
  --num_machines 1 \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision "${MIXED_PRECISION}" \
  --dynamo_backend no \
  "${TRAIN_SCRIPT}" \
  --dataset_base_path "/" \
  --dataset_metadata_path "${MERGED_METADATA}" \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image,edit_image_auto_resize" \
  --max_pixels "${MAX_PIXELS}" \
  --dataset_repeat "${DATASET_REPEAT}" \
  --model_paths "${MODEL_PATHS}" \
  --tokenizer_path "${FLUX2_BASE_DIR}/tokenizer" \
  --learning_rate "${LEARNING_RATE}" \
  --num_epochs "${NUM_EPOCHS}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --lora_base_model "dit" \
  --lora_target_modules "${LORA_TARGET_MODULES}" \
  --lora_rank "${LORA_RANK}" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_PATH}" \
  --log_every 1 \
  "${TRAIN_EXTRA_ARGS[@]}" \
  --use_gradient_checkpointing
