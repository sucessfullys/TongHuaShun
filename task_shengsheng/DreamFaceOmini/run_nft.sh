#!/bin/bash
set -euo pipefail
# shellcheck disable=SC1091
source "${SHENSHENG_ROOT:-/mnt/data/image-edit/datasets/shensheng}/config/paths.sh"

# ╔══════════════════════════════════════════════════════════════════╗
# ║  DiffusionNFT RL Training — collect → train outer loop         ║
# ╚══════════════════════════════════════════════════════════════════╝

EXP_NAME="nft_v1"
EXP_NOTE="DiffusionNFT RL: ArcFace+Aesthetic reward, beta=0.5, kl=0.01"

# -- NFT 超参 --
NFT_BETA=0.5
KL_BETA=0.01
ADV_CLIP_MAX=5.0
EMA_DECAY=0.999
NUM_OUTER_ITERS=5       # collect→train 的外层迭代次数
NUM_INNER_EPOCHS=1      # 每次 train phase 的 epoch 数
LR=5e-5
LORA_RANK=32
MAX_GRAD_NORM=1.0
MAX_PIXELS=2073600

# -- 数据收集参数 --
NUM_IMAGES_PER_PROMPT=4
COLLECT_STEPS=28
COLLECT_CFG=1.0
COLLECT_HEIGHT=1152
COLLECT_WIDTH=896

# -- Reward 权重 --
ARCFACE_WEIGHT=0.7
AESTHETIC_WEIGHT=0.3

# -- 路径 --
INIT_LORA="${SHENSHENG_WEIGHTS_LEGACY}/v2.151e20.safetensors"
DATASET="${SHENSHENG_DATA}/merged_train_4349.jsonl"
ARCFACE=/mnt/data/image-edit/models/arcface/weights/arcface-r100-glint360k.pth
INSIGHTFACE="${SHENSHENG_MODELS}/insightface"
AESTHETIC_CKPT=""       # 留空则不用 aesthetic reward；设为 .pth 路径启用

# -- 硬件 --
GPUS=1,2,3,4,5,6,7
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)

# ╔══════════════════════════════════════════════════════════════════╗
# ║  以下不用改                                                       ║
# ╚══════════════════════════════════════════════════════════════════╝

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="${PROJ_DIR}/examples/flux2/model_training/train_nft.py"
COLLECT_SCRIPT="${PROJ_DIR}/examples/flux2/model_training/collect_rewards.py"
OUTPUT_BASE="${PROJ_DIR}/models/nft/${EXP_NAME}"
LOG_DIR="${PROJ_DIR}/exp_logs/${EXP_NAME}"
mkdir -p "$OUTPUT_BASE" "$LOG_DIR"

CURRENT_LORA="$INIT_LORA"

echo "═══════════════════════════════════════════════════"
echo "  DiffusionNFT RL: ${EXP_NAME}"
echo "  Outer iterations: ${NUM_OUTER_ITERS}"
echo "  Init LoRA: ${INIT_LORA}"
echo "  Dataset: ${DATASET}"
echo "═══════════════════════════════════════════════════"

for ITER in $(seq 1 $NUM_OUTER_ITERS); do
    echo ""
    echo "━━━ Iteration ${ITER}/${NUM_OUTER_ITERS} ━━━"
    ITER_DIR="${OUTPUT_BASE}/iter_${ITER}"
    COLLECT_DIR="${ITER_DIR}/collected"
    TRAIN_DIR="${ITER_DIR}/trained"
    mkdir -p "$COLLECT_DIR" "$TRAIN_DIR"

    # ── Phase 1: Collect ─────────────────────────────────
    echo "[Phase 1] Collecting rewards with LoRA: ${CURRENT_LORA}"
    CUDA_VISIBLE_DEVICES=$GPUS python "$COLLECT_SCRIPT" \
        --input_jsonl "$DATASET" \
        --output_dir "$COLLECT_DIR" \
        --lora "$CURRENT_LORA" \
        --num_images_per_prompt $NUM_IMAGES_PER_PROMPT \
        --steps $COLLECT_STEPS \
        --cfg $COLLECT_CFG \
        --height $COLLECT_HEIGHT \
        --width $COLLECT_WIDTH \
        --gpus "$GPUS" \
        --arcface_ckpt "$ARCFACE" \
        --insightface_root "$INSIGHTFACE" \
        ${AESTHETIC_CKPT:+--aesthetic_ckpt "$AESTHETIC_CKPT"} \
        --arcface_weight $ARCFACE_WEIGHT \
        --aesthetic_weight $AESTHETIC_WEIGHT \
        --adv_clip_max $ADV_CLIP_MAX \
        2>&1 | tee "${LOG_DIR}/collect_iter${ITER}.log"

    REWARD_JSONL="${COLLECT_DIR}/rewards_with_advantage.jsonl"
    if [ ! -f "$REWARD_JSONL" ]; then
        echo "[ERROR] Reward JSONL not found: $REWARD_JSONL"
        exit 1
    fi

    # ── Phase 2: Train ───────────────────────────────────
    echo "[Phase 2] NFT training iteration ${ITER}"
    CUDA_VISIBLE_DEVICES=$GPUS accelerate launch \
        --num_processes $NUM_GPUS \
        --mixed_precision bf16 \
        "$TRAIN_SCRIPT" \
        --reward_jsonl "$REWARD_JSONL" \
        --model_id_with_origin_paths "black-forest-labs/FLUX.2-klein-base-9B" \
        --trainable_models "dit" \
        --lora_base_model "dit" \
        --lora_rank $LORA_RANK \
        --lora_checkpoint "$CURRENT_LORA" \
        --output_path "$TRAIN_DIR" \
        --learning_rate $LR \
        --max_pixels $MAX_PIXELS \
        --num_epochs $NUM_INNER_EPOCHS \
        --nft_beta $NFT_BETA \
        --kl_beta $KL_BETA \
        --adv_clip_max $ADV_CLIP_MAX \
        --ema_decay $EMA_DECAY \
        --max_grad_norm $MAX_GRAD_NORM \
        --use_gradient_checkpointing \
        --extra_inputs "edit_image" \
        2>&1 | tee "${LOG_DIR}/train_iter${ITER}.log"

    # Update LoRA for next iteration
    LATEST_CKPT=$(ls -t "${TRAIN_DIR}"/epoch-*.safetensors 2>/dev/null | head -1)
    if [ -z "$LATEST_CKPT" ]; then
        LATEST_CKPT=$(ls -t "${TRAIN_DIR}"/step-*.safetensors 2>/dev/null | head -1)
    fi
    if [ -n "$LATEST_CKPT" ]; then
        CURRENT_LORA="$LATEST_CKPT"
        echo "[Iter ${ITER}] Next LoRA: ${CURRENT_LORA}"
    else
        echo "[WARN] No checkpoint found in ${TRAIN_DIR}, reusing previous LoRA"
    fi
done

echo ""
echo "═══════════════════════════════════════════════════"
echo "  DiffusionNFT RL complete: ${EXP_NAME}"
echo "  Final LoRA: ${CURRENT_LORA}"
echo "═══════════════════════════════════════════════════"
