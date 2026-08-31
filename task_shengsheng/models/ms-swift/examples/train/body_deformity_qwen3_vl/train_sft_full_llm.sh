#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-/mnt/image-edit/datasets/duanyufa/models/Qwen3-VL-8B-Instruct}"
DATASET_PATH="${DATASET_PATH:-/mnt/image-edit/datasets/duanyufa/task_shengsheng/models/ms-swift/examples/train/body_deformity_qwen3_vl/body_deformity_multihand_reason_v3_train_with_noise_version4.jsonl}"
VAL_DATASET_PATH="${VAL_DATASET_PATH:-/mnt/image-edit/datasets/duanyufa/task_shengsheng/models/ms-swift/examples/train/body_deformity_qwen3_vl/body_deformity_multihand_reason_v3_val_fixed.jsonl}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/image-edit/datasets/duanyufa/conda_envs/miniconda3/envs/ms-swift/bin/python}"

DEFAULT_NPROC_PER_NODE="${NPROC_PER_NODE:-5}"
DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
DEFAULT_GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
DEFAULT_GLOBAL_BATCH_SIZE=$((DEFAULT_NPROC_PER_NODE * DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE * DEFAULT_GRADIENT_ACCUMULATION_STEPS))
DEFAULT_NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
DEFAULT_LEARNING_RATE="${LEARNING_RATE:-5e-6}"
DEFAULT_WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
DEFAULT_LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
DEFAULT_MAX_LENGTH="${MAX_LENGTH:-4096}"
DEFAULT_IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-2048}"

RUN_NAME="${RUN_NAME:-body_deformity_qwen3_vl_multihand_reason_v3_full_llm_sft}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-frombase_full_llm_ep${DEFAULT_NUM_TRAIN_EPOCHS}_lr${DEFAULT_LEARNING_RATE}_gb${GLOBAL_BATCH_SIZE:-${DEFAULT_GLOBAL_BATCH_SIZE}}_img${DEFAULT_IMAGE_MAX_TOKEN_NUM}_len${DEFAULT_MAX_LENGTH}_withnoisev4_dreamface2_fixedval_${RUN_ID}}"
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
require_abs_path VAL_DATASET_PATH "${VAL_DATASET_PATH}"
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

if [[ ! -f "${VAL_DATASET_PATH}" ]]; then
    echo "Error: VAL_DATASET_PATH does not exist: ${VAL_DATASET_PATH}" >&2
    exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Error: PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
    echo "Set PYTHON_BIN to the python executable in your ms-swift environment." >&2
    exit 1
fi

echo "Using Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" -V
echo "Training mode: full LLM SFT from base model path; ViT and aligner stay frozen unless flags are changed."
echo "MODEL_PATH=${MODEL_PATH}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "VAL_DATASET_PATH=${VAL_DATASET_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "NUM_TRAIN_EPOCHS=${DEFAULT_NUM_TRAIN_EPOCHS}"
echo "LEARNING_RATE=${DEFAULT_LEARNING_RATE}"
echo "LR_SCHEDULER_TYPE=${DEFAULT_LR_SCHEDULER_TYPE}"
echo "WARMUP_RATIO=${DEFAULT_WARMUP_RATIO}"
echo "MAX_LENGTH=${DEFAULT_MAX_LENGTH}"
echo "IMAGE_MAX_TOKEN_NUM=${DEFAULT_IMAGE_MAX_TOKEN_NUM}"
echo "NPROC_PER_NODE=${DEFAULT_NPROC_PER_NODE}"
echo "PER_DEVICE_TRAIN_BATCH_SIZE=${DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "GRADIENT_ACCUMULATION_STEPS=${DEFAULT_GRADIENT_ACCUMULATION_STEPS}"
echo "GLOBAL_BATCH_SIZE=${DEFAULT_GLOBAL_BATCH_SIZE}"

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

DATASET_PATH_FOR_CHECK="${DATASET_PATH}" VAL_DATASET_PATH_FOR_CHECK="${VAL_DATASET_PATH}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

def check_dataset(path: Path) -> Counter:
    stats = Counter()
    for line_no, line in enumerate(path.open(encoding="utf-8"), start=1):
        item = json.loads(line)
        assistant = item["messages"][-1]["content"]
        if "<think>" in assistant or "</think>" in assistant:
            raise SystemExit(f"{path}:{line_no}: assistant content should use <evidence>, not <think>")
        if "<bbox>" in assistant or "objects" in item:
            raise SystemExit(f"{path}:{line_no}: v3 multihand reason dataset should not contain bbox/objects")
        if "<evidence>" not in assistant or "</evidence>" not in assistant:
            raise SystemExit(f"{path}:{line_no}: missing <evidence>...</evidence>")
        if "<conclusion>" not in assistant or "</conclusion>" not in assistant:
            raise SystemExit(f"{path}:{line_no}: missing <conclusion>...</conclusion>")
        conclusion = assistant.split("<conclusion>", 1)[1].split("</conclusion>", 1)[0]
        if conclusion not in {"normal", "abnormal", "non_human"}:
            raise SystemExit(f"{path}:{line_no}: invalid conclusion: {conclusion}")
        if item.get("label") and item["label"] != conclusion:
            raise SystemExit(f"{path}:{line_no}: label {item['label']} != conclusion {conclusion}")
        stats[conclusion] += 1
        stats["records"] += 1
    return stats

train_path = Path(os.environ["DATASET_PATH_FOR_CHECK"])
val_path = Path(os.environ["VAL_DATASET_PATH_FOR_CHECK"])
train_stats = check_dataset(train_path)
val_stats = check_dataset(val_path)

print(
    "Train dataset check passed: "
    f"records={train_stats['records']}, abnormal={train_stats['abnormal']}, "
    f"normal={train_stats['normal']}, non_human={train_stats['non_human']}"
)
print(
    "Val dataset check passed: "
    f"records={val_stats['records']}, abnormal={val_stats['abnormal']}, "
    f"normal={val_stats['normal']}, non_human={val_stats['non_human']}"
)
PY

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}"
if [[ "${DEFAULT_NPROC_PER_NODE}" == "1" ]]; then
    unset NPROC_PER_NODE NNODES NODE_RANK MASTER_ADDR MASTER_PORT
else
    export NPROC_PER_NODE="${NPROC_PER_NODE:-${DEFAULT_NPROC_PER_NODE}}"
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export IMAGE_MAX_TOKEN_NUM="${DEFAULT_IMAGE_MAX_TOKEN_NUM}"
export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-64}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-8}"
export QWENVL_BBOX_FORMAT="${QWENVL_BBOX_FORMAT:-new}"

SWIFT_CMD=("${PYTHON_BIN}" -m swift.cli.main)

DS_ARGS=()
if [[ -n "${DEEPSPEED:-}" ]]; then
    DS_ARGS=(--deepspeed "${DEEPSPEED}")
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
    --val_dataset "${VAL_DATASET_PATH}" \
    --load_from_cache_file false \
    --split_dataset_ratio 0 \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs "${DEFAULT_NUM_TRAIN_EPOCHS}" \
    --max_steps "${MAX_STEPS:--1}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-2}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-1}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
    --learning_rate "${DEFAULT_LEARNING_RATE}" \
    --lr_scheduler_type "${DEFAULT_LR_SCHEDULER_TYPE}" \
    --warmup_ratio "${DEFAULT_WARMUP_RATIO}" \
    --weight_decay "${WEIGHT_DECAY:-0.01}" \
    --adam_beta1 "${ADAM_BETA1:-0.9}" \
    --adam_beta2 "${ADAM_BETA2:-0.95}" \
    --max_grad_norm "${MAX_GRAD_NORM:-1.0}" \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS:-50}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT:-30}" \
    --eval_strategy "${EVAL_STRATEGY:-steps}" \
    --eval_steps "${EVAL_STEPS:-50}" \
    --load_best_model_at_end "${LOAD_BEST_MODEL_AT_END:-true}" \
    --metric_for_best_model "${METRIC_FOR_BEST_MODEL:-eval_loss}" \
    --greater_is_better "${GREATER_IS_BETTER:-false}" \
    --early_stop_interval "${EARLY_STOP_INTERVAL:-5}" \
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
