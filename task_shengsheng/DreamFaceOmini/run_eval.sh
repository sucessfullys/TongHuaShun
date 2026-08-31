#!/bin/bash
set -euo pipefail

# ╔══════════════════════════════════════════════════════════════════╗
# ║  每次评估只改这个区域                                              ║
# ╚══════════════════════════════════════════════════════════════════╝

EXP_NAME="v2.22"   # 对应训练实验名
EPOCH=-1                                 # 评估哪个 epoch
EVAL_NOTE="看v2.22的效果。使用更多数据"
SUFFIX="cfg1.0_s2guidance0_scale1-robot-flux9b-e5resume"

# -- 推理参数 --
# BENCHMARK=/mnt/data/image-edit/datasets/shensheng/datasets/benchmark/test20260129/test20260129_abs.json #"/mnt/data/image-edit/datasets/shensheng/datasets/benchmark/test20260129/test20260129_abs.json"
# BENCHMARK=/mnt/data/image-edit/datasets/shensheng/datasets/benchmark/明星/demo/02-2.json
BENCHMARK=/mnt/data/image-edit/datasets/shensheng/datasets/benchmark/机器人/demo/02-2.json
NUM_SAMPLES=200
SEED=42
STEPS=4
CFG=1.0
HEIGHT=1152
WIDTH=896
# -- 硬件 --
GPUS=4,5,6,7

# ╔══════════════════════════════════════════════════════════════════╗
# ║  以下不用改                                                       ║
# ╚══════════════════════════════════════════════════════════════════╝

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
LORA_PATH="/mnt/data/image-edit/datasets/shensheng/code/stable/DreamFaceOmini/models/nft/dreamface_nft_vlm_triplet_human_gemma4_v2.22_resume/checkpoints/checkpoint-5/diffsynth_lora.safetensors" #"${PROJ_DIR}/models/train/${EXP_NAME}/epoch-${EPOCH}.safetensors"
OUTPUT_DIR="${PROJ_DIR}/exp_out/${EXP_NAME}_e${EPOCH}_${SUFFIX}"
LOG_DIR="${PROJ_DIR}/exp_logs/${EXP_NAME}"
mkdir -p "$LOG_DIR"

if [ ! -f "$LORA_PATH" ]; then
    echo "[ERROR] LoRA 文件不存在: ${LORA_PATH}"
    echo "  可用的 epoch:"
    ls "${PROJ_DIR}/models/train/${EXP_NAME}/" 2>/dev/null | grep -o 'epoch-[0-9]*' | sort -t- -k2 -n
    exit 1
fi

# -- 保存评估配置 --
EVAL_CONFIG="${LOG_DIR}/eval_e${EPOCH}.json"
cat > "$EVAL_CONFIG" << EOFCFG
{
  "exp_name": "${EXP_NAME}",
  "eval_note": "${EVAL_NOTE}",
  "date": "$(date +%Y-%m-%d_%H:%M:%S)",
  "epoch": ${EPOCH},
  "lora_path": "${LORA_PATH}",
  "benchmark": "${BENCHMARK}",
  "num_samples": ${NUM_SAMPLES},
  "seed": ${SEED},
  "steps": ${STEPS},
  "cfg": ${CFG},
  "height": ${HEIGHT},
  "width": ${WIDTH},
  "output_dir": "${OUTPUT_DIR}"
}
EOFCFG

echo ""
echo "============================================"
echo "  评估: ${EXP_NAME} epoch-${EPOCH}"
echo "  备注: ${EVAL_NOTE}"
echo "  LoRA: ${LORA_PATH}"
echo "  输出: ${OUTPUT_DIR}"
echo "============================================"
echo ""

export DIFFSYNTH_MODEL_BASE_PATH="/mnt/data/image-edit/datasets/shensheng/models"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONPATH="${PROJ_DIR}:${PYTHONPATH:-}"

python "${PROJ_DIR}/examples/flux2/model_inference/batch_infer.py" \
  --jsonl "$BENCHMARK" \
  --output "$OUTPUT_DIR" \
  --num "$NUM_SAMPLES" \
  --seed "$SEED" \
  --steps "$STEPS" \
  --gpus "$GPUS" \
  --cfg "$CFG" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --lora "$LORA_PATH"

echo ""
echo "[DONE] 评估完成: ${OUTPUT_DIR}"
echo "  配置已保存: ${EVAL_CONFIG}"
