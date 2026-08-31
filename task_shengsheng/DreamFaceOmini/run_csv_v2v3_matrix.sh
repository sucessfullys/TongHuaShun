#!/bin/bash
set -euo pipefail

# V2/V3 × 1:1/4:3/16:9(竖) × step4/8/28 × cfg1/2 全矩阵 batch
# 命名沿用 run_csv_demo.sh：csv-fluxout-true{v2|v3}-cfg{N}-step{S}-{ratio}

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_OUT="${PROJ_DIR}/exp_out"
INPUT_CSV="/mnt/data/image-edit/datasets/shensheng/datasets/benchmark/filter/filter-intersection-06-04.csv"
LORA_PATH="/mnt/data/image-edit/datasets/shensheng/models/hithink-image-labs/DreamFace_lora/v2.1/diffsynth_lora.safetensors"
TRANSFORMER_V2="/mnt/data/image-edit/datasets/shensheng/models/wikeeyang/Flux2-Klein-9B-True-V2/Flux2-Klein-9B-True-v2-bf16.safetensors"
TRANSFORMER_V3="/mnt/data/image-edit/datasets/shensheng/models/wikeeyang/Flux2-Klein-9B-True-V3/Flux2-Klein-9B-True-V3-bf16.safetensors"
SEED=42
GPUS="${GPUS:-0,1,2,3}"
EXPECTED_ROWS=222
LOG_DIR="${EXP_OUT}/csv-v2v3-matrix-logs"
MATRIX_LOG="${LOG_DIR}/matrix_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR"

export DIFFSYNTH_MODEL_BASE_PATH="/mnt/data/image-edit/datasets/shensheng/models"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONPATH="${PROJ_DIR}:${PYTHONPATH:-}"

PYTHON="${DREAMFACE_OMINI_VENV:-/root/.venv/diffsynth-gradio}/bin/python"
if [ ! -x "$PYTHON" ]; then
    PYTHON="python"
fi

ratio_width() {
    case "$1" in
        "1:1") echo 1024 ;;
        "4:3") echo 1024 ;;
        "16:9") echo 720 ;;
        *) echo "[ERROR] unknown ratio: $1" >&2; exit 1 ;;
    esac
}

ratio_height() {
    case "$1" in
        "1:1") echo 1024 ;;
        "4:3") echo 1365 ;;
        "16:9") echo 1280 ;;
        *) echo "[ERROR] unknown ratio: $1" >&2; exit 1 ;;
    esac
}

transformer_for_model() {
    case "$1" in
        v2) echo "$TRANSFORMER_V2" ;;
        v3) echo "$TRANSFORMER_V3" ;;
        *) echo "[ERROR] unknown model tag: $1" >&2; exit 1 ;;
    esac
}

is_job_complete() {
    local out_dir="$1"
    local csv="${out_dir}/csv-fluxout.csv"
    [ -f "$csv" ] || return 1
    "$PYTHON" - "$csv" "$EXPECTED_ROWS" <<'PY'
import csv, sys
path = sys.argv[1]
expected = int(sys.argv[2])
with open(path, newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))
ok = sum(1 for r in rows if r.get("status") in ("ok", "skipped"))
sys.exit(0 if len(rows) >= expected and ok >= expected else 1)
PY
}

run_job() {
    local model_tag="$1"
    local ratio="$2"
    local steps="$3"
    local cfg="$4"
    local width height transformer output_dir

    width="$(ratio_width "$ratio")"
    height="$(ratio_height "$ratio")"
    transformer="$(transformer_for_model "$model_tag")"
    output_dir="${EXP_OUT}/csv-fluxout-true${model_tag}-cfg${cfg}-step${steps}-${ratio}"

    if is_job_complete "$output_dir"; then
        echo "[SKIP] already complete: ${output_dir}" | tee -a "$MATRIX_LOG"
        return 0
    fi

    echo "" | tee -a "$MATRIX_LOG"
    echo "============================================" | tee -a "$MATRIX_LOG"
    echo "[RUN] model=${model_tag} ratio=${ratio} (${width}x${height}) steps=${steps} cfg=${cfg}" | tee -a "$MATRIX_LOG"
    echo "      out=${output_dir}" | tee -a "$MATRIX_LOG"
    echo "============================================" | tee -a "$MATRIX_LOG"

    local job_log="${LOG_DIR}/true${model_tag}-cfg${cfg}-step${steps}-${ratio}.log"
    "$PYTHON" "${PROJ_DIR}/examples/flux2/model_inference/csv_batch_infer.py" \
        --csv "$INPUT_CSV" \
        --output "$output_dir" \
        --transformer "$transformer" \
        --lora "$LORA_PATH" \
        --steps "$steps" \
        --cfg "$cfg" \
        --seed "$SEED" \
        --height "$height" \
        --width "$width" \
        --gpus "$GPUS" \
        2>&1 | tee "$job_log" | tee -a "$MATRIX_LOG"

    if ! is_job_complete "$output_dir"; then
        echo "[FAIL] incomplete results: ${output_dir}" | tee -a "$MATRIX_LOG"
        return 1
    fi
    echo "[DONE] ${output_dir}" | tee -a "$MATRIX_LOG"
}

exec > >(tee -a "$MATRIX_LOG") 2>&1

echo "Matrix log: $MATRIX_LOG"
echo "Expected rows per job: $EXPECTED_ROWS"
echo "GPUs: $GPUS"

for model_tag in v2 v3; do
    for ratio in "1:1" "4:3" "16:9"; do
        for steps in 4 8 28; do
            for cfg in 1 2; do
                run_job "$model_tag" "$ratio" "$steps" "$cfg" || echo "[WARN] job failed, continue matrix" | tee -a "$MATRIX_LOG"
            done
        done
    done
done

echo ""
echo "[ALL DONE] matrix finished. Master log: $MATRIX_LOG"
