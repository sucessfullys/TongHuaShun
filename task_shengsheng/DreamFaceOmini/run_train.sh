#!/bin/bash
set -euo pipefail
# shellcheck disable=SC1091
source "${SHENSHENG_ROOT:-/mnt/data/image-edit/datasets/shensheng}/config/paths.sh"

# ╔══════════════════════════════════════════════════════════════════╗
# ║  每次实验只改这个区域，下面全部自动                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

EXP_NAME="v2.17_0413_1_firered_w0.05"
EXP_NOTE="v2.17版本小样实验4000数据：降低face_id_weight到0.05,启用增强（headcrop0.3,geo_aug0.5,real_world_degradation0.1）"

# -- 训练超参 --
LR=1e-4
NUM_EPOCHS=100
MAX_PIXELS=2073600
LORA_RANK=32

# -- 梯度裁剪 (0=禁用) --
MAX_GRAD_NORM=1.0
INITIAL_GRAD_NORM_RATIO=5.0
GRAD_CLIP_WARMUP_STEPS=500
ABNORMAL_GRAD_RATIO=5.0

# -- ID Loss 参数 --
FACE_ID_MODE=firered          # firered | withanyone | 留空=不用idloss
FACE_ID_WEIGHT=0.05
FACE_SIGMA_CAP=0.9
FACE_CL_WEIGHT=0.0
FACE_CL_TEMP=0.07

# -- 增强参数 --
GEO_AUG_PROB=0.5
GEO_AUG_MAX_ROTATION=30.0
REAL_WORLD_DEGRADATION_PROB=0.1
HEADCROP_PROB=0.3

# -- 基座 & 数据 --
LORA_CKPT="${SHENSHENG_WEIGHTS_LEGACY}/v2.151e20.safetensors"
DATASET="${SHENSHENG_DATA}/merged_train_4349.jsonl"
ARCFACE=/mnt/data/image-edit/models/arcface/weights/arcface-r100-glint360k.pth
INSIGHTFACE="${SHENSHENG_MODELS}/insightface"

# -- 硬件 --
GPUS=1,2,3,4,5,6,7

# ╔══════════════════════════════════════════════════════════════════╗
# ║  以下不用改                                                       ║
# ╚══════════════════════════════════════════════════════════════════╝

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_PATH="${PROJ_DIR}/models/train/${EXP_NAME}"
LOG_DIR="${PROJ_DIR}/exp_logs/${EXP_NAME}"
mkdir -p "$LOG_DIR"

# -- 自动 git commit（如果有未提交的改动）--
cd "$PROJ_DIR"
if ! git diff --quiet HEAD 2>/dev/null || ! git diff --cached --quiet HEAD 2>/dev/null; then
    git add -A
    git commit -m "${EXP_NAME}: ${EXP_NOTE}" --allow-empty
    echo "[GIT] 已自动 commit: ${EXP_NAME}"
else
    echo "[GIT] 代码无变化，跳过 commit"
fi
GIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "no-git")

# -- 保存实验配置 --
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
  "face_id_mode": "${FACE_ID_MODE}",
  "face_id_weight": ${FACE_ID_WEIGHT},
  "face_sigma_cap": ${FACE_SIGMA_CAP},
  "face_cl_weight": ${FACE_CL_WEIGHT},
  "face_cl_temperature": ${FACE_CL_TEMP},
  "output_path": "${OUTPUT_PATH}",
  "geo_aug_prob": ${GEO_AUG_PROB},
  "geo_aug_max_rotation": ${GEO_AUG_MAX_ROTATION},
  "degradation_prob ": ${REAL_WORLD_DEGRADATION_PROB},
  "headcrop_prob": ${HEADCROP_PROB},
  "max_grad_norm": ${MAX_GRAD_NORM},
  "initial_grad_norm_ratio": ${INITIAL_GRAD_NORM_RATIO},
  "grad_clip_warmup_steps": ${GRAD_CLIP_WARMUP_STEPS},
  "abnormal_grad_ratio": ${ABNORMAL_GRAD_RATIO}
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

# -- 复制本次使用的启动脚本到日志目录 --
cp "$0" "${LOG_DIR}/run_train.sh"

export DIFFSYNTH_MODEL_BASE_PATH="/mnt/data/image-edit/datasets/shensheng/models"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONPATH="${PROJ_DIR}:${PYTHONPATH:-}"

# -- 构建训练命令 --
TRAIN_CMD="CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch examples/flux2/model_training/train.py \
  --dataset_base_path / \
  --dataset_metadata_path ${DATASET} \
  --data_file_keys image,edit_image \
  --extra_inputs edit_image \
  --max_pixels ${MAX_PIXELS} \
  --dataset_repeat 1 \
  --model_id_with_origin_paths black-forest-labs/FLUX.2-klein-9B:text_encoder/*.safetensors,black-forest-labs/FLUX.2-klein-base-9B:transformer/*.safetensors,black-forest-labs/FLUX.2-klein-9B:vae/diffusion_pytorch_model.safetensors \
  --tokenizer_path black-forest-labs/FLUX.2-klein-9B:tokenizer/ \
  --learning_rate ${LR} \
  --num_epochs ${NUM_EPOCHS} \
  --remove_prefix_in_ckpt pipe.dit. \
  --output_path ${OUTPUT_PATH} \
  --lora_base_model dit \
  --lora_target_modules to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out,linear_in,linear_out,to_qkv_mlp_proj,single_transformer_blocks.0.attn.to_out,single_transformer_blocks.1.attn.to_out,single_transformer_blocks.2.attn.to_out,single_transformer_blocks.3.attn.to_out,single_transformer_blocks.4.attn.to_out,single_transformer_blocks.5.attn.to_out,single_transformer_blocks.6.attn.to_out,single_transformer_blocks.7.attn.to_out,single_transformer_blocks.8.attn.to_out,single_transformer_blocks.9.attn.to_out,single_transformer_blocks.10.attn.to_out,single_transformer_blocks.11.attn.to_out,single_transformer_blocks.12.attn.to_out,single_transformer_blocks.13.attn.to_out,single_transformer_blocks.14.attn.to_out,single_transformer_blocks.15.attn.to_out,single_transformer_blocks.16.attn.to_out,single_transformer_blocks.17.attn.to_out,single_transformer_blocks.18.attn.to_out,single_transformer_blocks.19.attn.to_out,single_transformer_blocks.20.attn.to_out,single_transformer_blocks.21.attn.to_out,single_transformer_blocks.22.attn.to_out,single_transformer_blocks.23.attn.to_out \
  --lora_rank ${LORA_RANK} \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --geo_aug_prob ${GEO_AUG_PROB} \
  --geo_aug_max_rotation ${GEO_AUG_MAX_ROTATION} \
  --degradation_prob ${REAL_WORLD_DEGRADATION_PROB} \
  --head_crop_prob ${HEADCROP_PROB} \
  --max_grad_norm ${MAX_GRAD_NORM} \
  --initial_grad_norm_ratio ${INITIAL_GRAD_NORM_RATIO} \
  --grad_clip_warmup_steps ${GRAD_CLIP_WARMUP_STEPS} \
  --abnormal_grad_ratio ${ABNORMAL_GRAD_RATIO}"

# 添加 lora_checkpoint（如果有）
if [ -n "${LORA_CKPT}" ] && [ -f "${LORA_CKPT}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --lora_checkpoint ${LORA_CKPT}"
fi

# 添加 face id loss 参数（如果启用）
if [ -n "${FACE_ID_MODE}" ] && [ "${FACE_ID_WEIGHT}" != "0" ] && [ "${FACE_ID_WEIGHT}" != "0.0" ]; then
    TRAIN_CMD="${TRAIN_CMD} \
      --face_id_mode ${FACE_ID_MODE} \
      --face_id_weight ${FACE_ID_WEIGHT} \
      --face_sigma_cap ${FACE_SIGMA_CAP} \
      --face_cl_weight ${FACE_CL_WEIGHT} \
      --face_cl_temperature ${FACE_CL_TEMP} \
      --arcface_ckpt_path ${ARCFACE} \
      --insightface_root ${INSIGHTFACE}"
fi

# -- 运行 --
cd "$PROJ_DIR"
eval $TRAIN_CMD 2>&1 | tee "${LOG_DIR}/train.log"

echo ""
echo "[DONE] 实验 ${EXP_NAME} 训练完成，日志在 ${LOG_DIR}/train.log"
