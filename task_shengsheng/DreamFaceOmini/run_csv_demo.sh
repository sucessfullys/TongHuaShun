#!/bin/bash
set -euo pipefail

# ╔══════════════════════════════════════════════════════════════════╗
# ║  CSV 批量推理 demo — 每次运行只改这里                             ║
# ╚══════════════════════════════════════════════════════════════════╝

INPUT_CSV="/mnt/data/image-edit/datasets/shensheng/datasets/benchmark/filter/filter-intersection-06-04.csv"
LORA_PATH="/mnt/data/image-edit/datasets/shensheng/models/hithink-image-labs/DreamFace_lora/v2.1/diffsynth_lora.safetensors"
# LORA_PATH="/mnt/data/image-edit/datasets/shensheng/code/stable/DreamFaceOmini/models/train/sft_facemask_w3_captions_dual/epoch-10.safetensors"
OUTPUT_DIR="/mnt/data/image-edit/datasets/shensheng/code/stable/DreamFaceOmini/exp_out/csv-fluxout-truev2-cfg2-step28-4:3"



STEPS=28
CFG=1.0
SEED=42
HEIGHT=1365
WIDTH=1024
GPUS=0,1,2,3
# NUM=10          # 取消注释可只跑前 N 条做 smoke test
# SKIP=0
# FORCE=1         # 取消注释可强制重跑已有结果

# ╔══════════════════════════════════════════════════════════════════╗
# ║  以下不用改                                                       ║
# ╚══════════════════════════════════════════════════════════════════╝

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -f "$INPUT_CSV" ]; then
    echo "[ERROR] 输入 CSV 不存在: $INPUT_CSV"
    exit 1
fi

if [ ! -f "$LORA_PATH" ]; then
    echo "[ERROR] LoRA 不存在: $LORA_PATH"
    exit 1
fi

export DIFFSYNTH_MODEL_BASE_PATH="/mnt/data/image-edit/datasets/shensheng/models"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONPATH="${PROJ_DIR}:${PYTHONPATH:-}"

CMD=(
    python "${PROJ_DIR}/examples/flux2/model_inference/csv_batch_infer.py"
    --csv "$INPUT_CSV"
    --output "$OUTPUT_DIR"
    --lora "$LORA_PATH"
    --steps "$STEPS"
    --cfg "$CFG"
    --seed "$SEED"
    --height "$HEIGHT"
    --width "$WIDTH"
    --gpus "$GPUS"
)

if [ "${NUM:-}" != "" ]; then
    CMD+=(--num "$NUM")
fi

if [ "${SKIP:-0}" != "0" ]; then
    CMD+=(--skip "$SKIP")
fi

if [ "${FORCE:-0}" = "1" ]; then
    CMD+=(--force)
fi

echo ""
echo "============================================"
echo "  CSV batch demo"
echo "  Input : $INPUT_CSV"
echo "  LoRA  : $LORA_PATH"
echo "  Model : FLUX.2-klein-9B + FLUX.2-klein-base-9B"
echo "  Output: $OUTPUT_DIR"
echo "  CSV   : $OUTPUT_DIR/csv-fluxout.csv"
echo "============================================"
echo ""

"${CMD[@]}"

echo ""
echo "[DONE] 结果目录: $OUTPUT_DIR"
echo "       输出 CSV: $OUTPUT_DIR/csv-fluxout.csv"
