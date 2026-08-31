#!/bin/bash
set -euo pipefail

# ╔══════════════════════════════════════════════════════════════════╗
# ║  纯 SFT + 人脸区域加权 MSE（无 ArcFace / 无 face ID loss）          ║
# ╚══════════════════════════════════════════════════════════════════╝

EXP_NAME="sft_facemask_w3_captions_dual"
EXP_NOTE="纯SFT: 人脸区域MSE加权x3(归一化), 无face_id_loss, klein-9B全量transformer"

# -- 训练超参 --
LR=1e-4
NUM_EPOCHS=20
MAX_PIXELS=4194304
LORA_RANK=32

# -- 人脸区域加权 MSE（核心新参数）--
FACE_MSE_WEIGHT=3.0       # 人脸 token 的 MSE 权重倍数；1.0=关闭
FACE_MASK_EXPAND=1.4      # bbox 外扩比例（覆盖头发/头部）

# -- 梯度裁剪 (0=禁用) --
MAX_GRAD_NORM=1.0
INITIAL_GRAD_NORM_RATIO=5.0
GRAD_CLIP_WARMUP_STEPS=500
ABNORMAL_GRAD_RATIO=5.0

# -- 增强参数（headcrop 防复制贴图，与 loss 无关，建议保留）--
GEO_AUG_PROB=0
GEO_AUG_MAX_ROTATION=0
REAL_WORLD_DEGRADATION_PROB=0
HEADCROP_PROB=0

# -- 基座 & 数据 --
LORA_CKPT=                # 留空=从头训 LoRA
DATASET=/mnt/data/image-edit/datasets/shensheng/datasets/merged_captions_dual_train-20260611.jsonl
INSIGHTFACE=/mnt/data/image-edit/datasets/shensheng/models/insightface

# transformer: klein-9B（推理 28 步）；如训 base 蒸馏版改回 klein-base-9B
TRANSFORMER_ID="black-forest-labs/FLUX.2-klein-9B"

# -- 硬件 --
GPUS=0,1,2,3,4,5,6,7

# ╔══════════════════════════════════════════════════════════════════╗
# ║  以下不用改                                                       ║
# ╚══════════════════════════════════════════════════════════════════╝

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_PATH="${PROJ_DIR}/models/train/${EXP_NAME}"
LOG_DIR="${PROJ_DIR}/exp_logs/${EXP_NAME}"
mkdir -p "$LOG_DIR"

cd "$PROJ_DIR"
GIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "no-git")

cat > "${LOG_DIR}/config.json" << EOFCFG
{
  "exp_name": "${EXP_NAME}",
  "note": "${EXP_NOTE}",
  "date": "$(date +%Y-%m-%d_%H:%M:%S)",
  "server": "$(hostname)",
  "git_commit": "${GIT_HASH}",
  "gpus": "${GPUS}",
  "learning_rate": "${LR}",
  "num_epochs": ${NUM_EPOCHS},
  "max_pixels": ${MAX_PIXELS},
  "lora_rank": ${LORA_RANK},
  "lora_checkpoint": "${LORA_CKPT}",
  "dataset": "${DATASET}",
  "transformer": "${TRANSFORMER_ID}",
  "face_mse_weight": ${FACE_MSE_WEIGHT},
  "face_mask_expand": ${FACE_MASK_EXPAND},
  "output_path": "${OUTPUT_PATH}",
  "geo_aug_prob": ${GEO_AUG_PROB},
  "geo_aug_max_rotation": ${GEO_AUG_MAX_ROTATION},
  "degradation_prob": ${REAL_WORLD_DEGRADATION_PROB},
  "headcrop_prob": ${HEADCROP_PROB},
  "max_grad_norm": ${MAX_GRAD_NORM}
}
EOFCFG

echo ""
echo "============================================"
echo "  实验: ${EXP_NAME}"
echo "  备注: ${EXP_NOTE}"
echo "  Git:  ${GIT_HASH}"
echo "  输出: ${OUTPUT_PATH}"
echo "  日志: ${LOG_DIR}/train.log"
echo "============================================"
echo ""

cp "$0" "${LOG_DIR}/run_train_sft.sh"

export DIFFSYNTH_MODEL_BASE_PATH="/mnt/data/image-edit/datasets/shensheng/models"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONPATH="${PROJ_DIR}:${PYTHONPATH:-}"

TRAIN_CMD="CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch examples/flux2/model_training/train.py \
  --dataset_base_path / \
  --dataset_metadata_path ${DATASET} \
  --data_file_keys image,edit_image \
  --extra_inputs edit_image \
  --max_pixels ${MAX_PIXELS} \
  --dataset_repeat 1 \
  --model_id_with_origin_paths black-forest-labs/FLUX.2-klein-9B:text_encoder/*.safetensors,${TRANSFORMER_ID}:transformer/*.safetensors,black-forest-labs/FLUX.2-klein-9B:vae/diffusion_pytorch_model.safetensors \
  --tokenizer_path black-forest-labs/FLUX.2-klein-9B:tokenizer/ \
  --learning_rate ${LR} \
  --num_epochs ${NUM_EPOCHS} \
  --remove_prefix_in_ckpt pipe.dit. \
  --output_path ${OUTPUT_PATH} \
  --lora_base_model dit \
  --lora_target_modules to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out,linear_in,linear_out,to_qkv_mlp_proj,single_transformer_blocks.0.attn.to_out,single_transformer_blocks.1.attn.to_out,single_transformer_blocks.2.attn.to_out,single_transformer_blocks.3.attn.to_out,single_transformer_blocks.4.attn.to_out,single_transformer_blocks.5.attn.to_out,single_transformer_blocks.6.attn.to_out,single_transformer_blocks.7.attn.to_out,single_transformer_blocks.8.attn.to_out,single_transformer_blocks.9.attn.to_out,single_transformer_blocks.10.attn.to_out,single_transformer_blocks.11.attn.to_out,single_transformer_blocks.12.attn.to_out,single_transformer_blocks.13.attn.to_out,single_transformer_blocks.14.attn.to_out,single_transformer_blocks.15.attn.to_out,single_transformer_blocks.16.attn.to_out,single_transformer_blocks.17.attn.to_out,single_transformer_blocks.18.attn.to_out,single_transformer_blocks.19.attn.to_out,single_transformer_blocks.20.attn.to_out,single_transformer_blocks.21.attn.to_out,single_transformer_blocks.22.attn.to_out,single_transformer_blocks.23.attn.to_out \
  --lora_rank ${LORA_RANK} \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --dataset_num_workers 8 \
  --face_mse_weight ${FACE_MSE_WEIGHT} \
  --face_mask_expand ${FACE_MASK_EXPAND} \
  --insightface_root ${INSIGHTFACE} \
  --geo_aug_prob ${GEO_AUG_PROB} \
  --geo_aug_max_rotation ${GEO_AUG_MAX_ROTATION} \
  --degradation_prob ${REAL_WORLD_DEGRADATION_PROB} \
  --head_crop_prob ${HEADCROP_PROB} \
  --max_grad_norm ${MAX_GRAD_NORM} \
  --initial_grad_norm_ratio ${INITIAL_GRAD_NORM_RATIO} \
  --grad_clip_warmup_steps ${GRAD_CLIP_WARMUP_STEPS} \
  --abnormal_grad_ratio ${ABNORMAL_GRAD_RATIO}"

if [ -n "${LORA_CKPT}" ] && [ -f "${LORA_CKPT}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --lora_checkpoint ${LORA_CKPT}"
fi

cd "$PROJ_DIR"
eval $TRAIN_CMD 2>&1 | tee "${LOG_DIR}/train.log"

echo ""
echo "[DONE] 实验 ${EXP_NAME} 训练完成，日志在 ${LOG_DIR}/train.log"
