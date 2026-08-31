#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_multihand_reason_v3_multi_gpu.sh"

CHECKPOINT_PARENT=""
EVAL_JSONL=""
GPUS="2,3"
START_STEP=""
END_STEP=""
STEP_INTERVAL="50"
RUN_PREFIX="checkpoint"
RUN_SUFFIX=""
MAX_NEW_TOKENS="128"
SHARD_TIMEOUT="1800"
TEST_ROOT=""
MODEL_PATH=""
PYTHON_BIN=""
IMAGE_MAX_TOKEN_NUM=""
BATCH_SIZE=""
CLEANUP_SHARDS=""
SKIP_EXISTING="true"

usage() {
    cat <<'EOF'
Usage:
  bash eval_multihand_reason_v3_checkpoints.sh \
    --checkpoint-parent /path/to/v0-xxxx \
    --eval-jsonl /path/to/eval.jsonl \
    --gpus 2,3 \
    --start-step 1800 \
    --end-step 2200 \
    --step-interval 50 \
    --run-prefix checkpointv6_2_try \
    --run-suffix benchmark_normal2_dreamface_tok128_clean

Notes:
  --end-step can be omitted; then all checkpoints >= --start-step are evaluated.
  Existing completed output dirs are skipped by default. Use --no-skip-existing to rerun.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint-parent) CHECKPOINT_PARENT="$2"; shift 2 ;;
        --eval-jsonl) EVAL_JSONL="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --start-step) START_STEP="$2"; shift 2 ;;
        --end-step) END_STEP="$2"; shift 2 ;;
        --step-interval) STEP_INTERVAL="$2"; shift 2 ;;
        --run-prefix) RUN_PREFIX="$2"; shift 2 ;;
        --run-suffix) RUN_SUFFIX="$2"; shift 2 ;;
        --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --shard-timeout) SHARD_TIMEOUT="$2"; shift 2 ;;
        --test-root) TEST_ROOT="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --python-bin) PYTHON_BIN="$2"; shift 2 ;;
        --image-max-token-num) IMAGE_MAX_TOKEN_NUM="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --cleanup-shards) CLEANUP_SHARDS="true"; shift ;;
        --no-skip-existing) SKIP_EXISTING="false"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [[ -z "${CHECKPOINT_PARENT}" || -z "${EVAL_JSONL}" || -z "${START_STEP}" ]]; then
    echo "Error: --checkpoint-parent, --eval-jsonl and --start-step are required." >&2
    usage >&2
    exit 1
fi

if [[ "${CHECKPOINT_PARENT}" != /* || "${EVAL_JSONL}" != /* ]]; then
    echo "Error: --checkpoint-parent and --eval-jsonl must be absolute paths." >&2
    exit 1
fi

if [[ ! -d "${CHECKPOINT_PARENT}" ]]; then
    echo "Error: checkpoint parent does not exist: ${CHECKPOINT_PARENT}" >&2
    exit 1
fi

if [[ ! -f "${EVAL_JSONL}" ]]; then
    echo "Error: eval jsonl does not exist: ${EVAL_JSONL}" >&2
    exit 1
fi

if [[ ! -f "${EVAL_SCRIPT}" ]]; then
    echo "Error: eval script does not exist: ${EVAL_SCRIPT}" >&2
    exit 1
fi

mapfile -t CHECKPOINT_STEPS < <(
    find "${CHECKPOINT_PARENT}" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' \
        | sed -n 's/^checkpoint-\([0-9][0-9]*\)$/\1/p' \
        | sort -n \
        | awk -v start="${START_STEP}" -v end="${END_STEP}" -v interval="${STEP_INTERVAL}" '
            $1 >= start && (end == "" || $1 <= end) && (($1 - start) % interval == 0) {print $1}
        '
)

if [[ "${#CHECKPOINT_STEPS[@]}" -eq 0 ]]; then
    echo "Error: no matched checkpoints in ${CHECKPOINT_PARENT}" >&2
    echo "start=${START_STEP}, end=${END_STEP:-<auto>}, interval=${STEP_INTERVAL}" >&2
    exit 1
fi

echo "Checkpoint parent: ${CHECKPOINT_PARENT}"
echo "Eval jsonl:        ${EVAL_JSONL}"
echo "GPUs:              ${GPUS}"
echo "Steps:             ${CHECKPOINT_STEPS[*]}"
echo "Run prefix:        ${RUN_PREFIX}"
echo "Run suffix:        ${RUN_SUFFIX:-<none>}"
echo "Skip existing:     ${SKIP_EXISTING}"

for step in "${CHECKPOINT_STEPS[@]}"; do
    adapter_path="${CHECKPOINT_PARENT}/checkpoint-${step}"
    if [[ -n "${RUN_SUFFIX}" ]]; then
        run_tag="${RUN_PREFIX}_${step}_${RUN_SUFFIX}"
    else
        run_tag="${RUN_PREFIX}_${step}"
    fi

    output_dir="${TEST_ROOT:-/mnt/image-edit/datasets/duanyufa/task_shengsheng/Open_dataset/Ours_bad/hand_foder/test}/multihand_reason_v3_eval/${run_tag}"
    if [[ "${SKIP_EXISTING}" == "true" && -f "${output_dir}/metrics.json" && -f "${output_dir}/predictions.jsonl" ]]; then
        echo "Skip completed checkpoint-${step}: ${output_dir}"
        continue
    fi

    echo "============================================================"
    echo "Evaluating checkpoint-${step}"
    echo "Adapter: ${adapter_path}"
    echo "Run tag: ${run_tag}"
    echo "Output:  ${output_dir}"

    env_args=(
        "GPUS=${GPUS}"
        "ADAPTER_PATH=${adapter_path}"
        "EVAL_JSONL=${EVAL_JSONL}"
        "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
        "SHARD_TIMEOUT=${SHARD_TIMEOUT}"
        "RUN_TAG=${run_tag}"
    )
    [[ -n "${TEST_ROOT}" ]] && env_args+=("TEST_ROOT=${TEST_ROOT}")
    [[ -n "${MODEL_PATH}" ]] && env_args+=("MODEL_PATH=${MODEL_PATH}")
    [[ -n "${PYTHON_BIN}" ]] && env_args+=("PYTHON_BIN=${PYTHON_BIN}")
    [[ -n "${IMAGE_MAX_TOKEN_NUM}" ]] && env_args+=("IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM}")
    [[ -n "${BATCH_SIZE}" ]] && env_args+=("BATCH_SIZE=${BATCH_SIZE}")
    [[ -n "${CLEANUP_SHARDS}" ]] && env_args+=("CLEANUP_SHARDS=${CLEANUP_SHARDS}")

    env "${env_args[@]}" bash "${EVAL_SCRIPT}"
done

echo "All matched checkpoints evaluated."
