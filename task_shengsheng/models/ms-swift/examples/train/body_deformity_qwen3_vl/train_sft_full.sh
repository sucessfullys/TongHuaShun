#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-/mnt/image-edit/datasets/duanyufa/models/Qwen3-VL-8B-Instruct}"
DATASET_PATH="${DATASET_PATH:-/mnt/image-edit/datasets/duanyufa/task_shengsheng/models/ms-swift/examples/train/body_deformity_qwen3_vl/body_deformity_grounding_v2_schemeA_train.jsonl}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/image-edit/datasets/duanyufa/conda_envs/miniconda3/envs/ms-swift/bin/python}"

DEFAULT_NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
DEFAULT_GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
DEFAULT_GLOBAL_BATCH_SIZE=$((DEFAULT_NPROC_PER_NODE * DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE * DEFAULT_GRADIENT_ACCUMULATION_STEPS))
DEFAULT_NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
DEFAULT_LEARNING_RATE="${LEARNING_RATE:-1e-5}"
DEFAULT_WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
DEFAULT_LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
DEFAULT_MAX_LENGTH="${MAX_LENGTH:-4096}"
DEFAULT_IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1536}"

RUN_NAME="${RUN_NAME:-body_deformity_qwen3_vl_grounding_v2_schemeA_full_sft}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-frombase_ep${DEFAULT_NUM_TRAIN_EPOCHS}_lr${DEFAULT_LEARNING_RATE}_gb${GLOBAL_BATCH_SIZE:-${DEFAULT_GLOBAL_BATCH_SIZE}}_img${DEFAULT_IMAGE_MAX_TOKEN_NUM}_len${DEFAULT_MAX_LENGTH}_ds${DEEPSPEED:-zero2}_val${SPLIT_DATASET_RATIO:-0.02}_${RUN_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/image-edit/datasets/duanyufa/task_shengsheng/models/ms-swift/output/${RUN_NAME}/${RUN_TAG}}"

require_abs_path() {
    local name="$1"
    local value="$2"
    if [[ "${value}" != /* ]]; then
        echo "Error: ${name} must be an absolute path, got: ${value}" >&2
        exit 1
    fi
}

require_abs_path MODEL_PATH "${MODEL_PATH}"
require_abs_path DATASET_PATH "${DATASET_PATH}"
require_abs_path OUTPUT_DIR "${OUTPUT_DIR}"
require_abs_path PYTHON_BIN "${PYTHON_BIN}"

mkdir -p "${OUTPUT_DIR}/logs"
if [[ "${ENABLE_TEE_LOG:-true}" == "true" && -z "${TEE_LOG_ACTIVE:-}" ]]; then
    export TEE_LOG_ACTIVE=1
    LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/logs/train_$(date +%Y%m%d_%H%M%S).log}"
    echo "Writing terminal log to: ${LOG_FILE}"
    exec > >(tee -a "${LOG_FILE}") 2>&1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Error: MODEL_PATH does not exist: ${MODEL_PATH}" >&2
    exit 1
fi

if [[ ! -f "${DATASET_PATH}" ]]; then
    echo "Error: DATASET_PATH does not exist: ${DATASET_PATH}" >&2
    exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Error: PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
    echo "Set PYTHON_BIN to the python executable in your ms-swift environment." >&2
    exit 1
fi

echo "Using Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" -V
echo "Training mode: from base model path, unless RESUME_FROM_CHECKPOINT is explicitly set."
echo "MODEL_PATH=${MODEL_PATH}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "NUM_TRAIN_EPOCHS=${DEFAULT_NUM_TRAIN_EPOCHS}"
echo "LEARNING_RATE=${DEFAULT_LEARNING_RATE}"
echo "LR_SCHEDULER_TYPE=${DEFAULT_LR_SCHEDULER_TYPE}"
echo "WARMUP_RATIO=${DEFAULT_WARMUP_RATIO}"
echo "MAX_LENGTH=${DEFAULT_MAX_LENGTH}"
echo "IMAGE_MAX_TOKEN_NUM=${DEFAULT_IMAGE_MAX_TOKEN_NUM}"

"${PYTHON_BIN}" - <<'PY'
import importlib.metadata as metadata
from packaging.version import Version

try:
    version = metadata.version("qwen-vl-utils")
except metadata.PackageNotFoundError:
    raise SystemExit(
        "Missing dependency: qwen-vl-utils>=0.0.14\n"
        "Install it in the active ms-swift environment:\n"
        "  python -m pip install -U 'qwen-vl-utils>=0.0.14' decord\n"
    )

if Version(version) < Version("0.0.14"):
    raise SystemExit(
        f"qwen-vl-utils is too old: {version}. Required: >=0.0.14\n"
        "Upgrade it in the active ms-swift environment:\n"
        "  python -m pip install -U 'qwen-vl-utils>=0.0.14' decord\n"
    )
PY

if [[ -n "${DEEPSPEED:-zero2}" ]]; then
    "${PYTHON_BIN}" - <<'PY'
import importlib.metadata as metadata

try:
    version = metadata.version("deepspeed")
except metadata.PackageNotFoundError:
    raise SystemExit(
        "Missing dependency: deepspeed\n"
        "Install it in the ms-swift environment before distributed full SFT:\n"
        "  /mnt/image-edit/datasets/duanyufa/conda_envs/miniconda3/envs/ms-swift/bin/python -m pip install deepspeed\n"
        "Or run without DeepSpeed for debugging only:\n"
        "  DEEPSPEED= bash /mnt/image-edit/datasets/duanyufa/task_shengsheng/models/ms-swift/examples/train/body_deformity_qwen3_vl/train_sft_full.sh\n"
    )

print(f"DeepSpeed version: {version}")
PY
fi

MODEL_PATH_FOR_CHECK="${MODEL_PATH}" "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

model_dir = Path(os.environ["MODEL_PATH_FOR_CHECK"])
shards = sorted(model_dir.glob("*.safetensors"))
if not shards:
    raise SystemExit(f"No .safetensors files found in MODEL_PATH: {model_dir}")

bad = []
for path in shards:
    try:
        head = path.read_bytes()[:80]
    except OSError as exc:
        bad.append(f"{path}: cannot read file ({exc})")
        continue
    if head.startswith(b"version https://git-lfs.github.com/spec/v1"):
        bad.append(f"{path}: Git LFS pointer, real weights were not downloaded")
    elif path.stat().st_size < 1024 * 1024:
        bad.append(f"{path}: suspiciously small ({path.stat().st_size} bytes)")

if bad:
    msg = "\n".join(bad)
    raise SystemExit(
        "Invalid model weight files detected:\n"
        f"{msg}\n\n"
        "Fix the model directory before training, for example:\n"
        f"  cd {model_dir}\n"
        "  git lfs install\n"
        "  git lfs pull\n"
    )
PY

DATASET_PATH_FOR_CHECK="${DATASET_PATH}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

path = Path(os.environ["DATASET_PATH_FOR_CHECK"])
stats = Counter()
for line_no, line in enumerate(path.open(encoding="utf-8"), start=1):
    item = json.loads(line)
    assistant = item["messages"][-1]["content"]
    n_bbox = assistant.count("<bbox>")
    n_objects = len(item.get("objects", {}).get("bbox", []))
    if n_bbox != n_objects:
        raise SystemExit(f"{path}:{line_no}: <bbox> count {n_bbox} != objects.bbox count {n_objects}")
    conclusion = assistant.split("<conclusion>", 1)[1].split("</conclusion>", 1)[0]
    if conclusion in {"normal", "non_human"} and "objects" in item:
        raise SystemExit(f"{path}:{line_no}: {conclusion} sample should not contain objects")
    stats[conclusion] += 1
    stats["records"] += 1
    stats["bbox"] += n_objects

print(
    "Dataset check passed: "
    f"records={stats['records']}, abnormal={stats['abnormal']}, "
    f"normal={stats['normal']}, non_human={stats['non_human']}, bbox={stats['bbox']}"
)
PY

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export IMAGE_MAX_TOKEN_NUM="${DEFAULT_IMAGE_MAX_TOKEN_NUM}"
export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-64}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-8}"
export QWENVL_BBOX_FORMAT="${QWENVL_BBOX_FORMAT:-new}"

SWIFT_CMD=("${PYTHON_BIN}" -m swift.cli.main)

DS_ARGS=()
if [[ -n "${DEEPSPEED:-zero2}" ]]; then
    DS_ARGS=(--deepspeed "${DEEPSPEED:-zero2}")
fi

RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    require_abs_path RESUME_FROM_CHECKPOINT "${RESUME_FROM_CHECKPOINT}"
    RESUME_ARGS=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

"${SWIFT_CMD[@]}" sft \
    --model "${MODEL_PATH}" \
    --model_type qwen3_vl \
    --dataset "${DATASET_PATH}" \
    --load_from_cache_file false \
    --split_dataset_ratio "${SPLIT_DATASET_RATIO:-0.02}" \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs "${DEFAULT_NUM_TRAIN_EPOCHS}" \
    --max_steps "${MAX_STEPS:--1}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-1}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-2}" \
    --learning_rate "${DEFAULT_LEARNING_RATE}" \
    --lr_scheduler_type "${DEFAULT_LR_SCHEDULER_TYPE}" \
    --warmup_ratio "${DEFAULT_WARMUP_RATIO}" \
    --weight_decay "${WEIGHT_DECAY:-0.1}" \
    --adam_beta1 "${ADAM_BETA1:-0.9}" \
    --adam_beta2 "${ADAM_BETA2:-0.95}" \
    --max_grad_norm "${MAX_GRAD_NORM:-1.0}" \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS:-500}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
    --eval_strategy "${EVAL_STRATEGY:-steps}" \
    --eval_steps "${EVAL_STEPS:-500}" \
    --logging_steps "${LOGGING_STEPS:-10}" \
    --max_length "${DEFAULT_MAX_LENGTH}" \
    --output_dir "${OUTPUT_DIR}" \
    --seed "${SEED:-42}" \
    --data_seed "${DATA_SEED:-42}" \
    --dataset_shuffle "${DATASET_SHUFFLE:-true}" \
    --dataset_num_proc "${DATASET_NUM_PROC:-8}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-8}" \
    --save_only_model "${SAVE_ONLY_MODEL:-false}" \
    "${DS_ARGS[@]}" \
    "${RESUME_ARGS[@]}"
