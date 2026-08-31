#!/bin/bash
set -euo pipefail

# ╔══════════════════════════════════════════════════════════════════╗
# ║  每次启动只改这个区域                                              ║
# ╚══════════════════════════════════════════════════════════════════╝

EXP_NAME="captions_dual_klein9b_lora_w4"
TRANSFORMER_MODEL_ID="black-forest-labs/FLUX.2-klein-9B"
LORA_PATH="/mnt/image-edit/models/hithink-image-labs/DreamFace_lora/v2.1/diffsynth_lora.safetensors"
LORA_ALPHA=1.0
VAE_PATH="" #"/mnt/data/image-edit/datasets/shensheng/models/SeFi-Image/SeFi-Image-5B-turbo/vae"

# -- 推理参数（对齐 wandb _run_full_inference） --
SEED=-1
STEPS=28
CFG=1.0
EMBEDDED_GUIDANCE=1.0
HEIGHT=1024
WIDTH=1024

# -- S²-Guidance 实验开关（ω=0 即关闭，等价于之前对齐 wandb 的行为） --
S2_SCALE=0.25
S2_DROP_RATIO=0.1
S2_START=0.1
S2_END=0.9

# -- Sigma Schedule 实验开关（<=0 即关闭 = 自动 mu，等价于之前对齐 wandb 的行为；谨慎手动改） --
SIGMA_MU=0.0

# -- 每次生图都会存一份（output.png + 参考图 + meta.json），按实验名分目录 --
LOG_DIR="/mnt/image-edit/datasets/duanyufa/task_shengsheng/DreamFaceOmini/outputs/gradio_dreamface_omini/${EXP_NAME}"

# -- 单卡服务；32B(dev) 有双卡时会自动用 0,1 --
PHYSICAL_GPUS="${PHYSICAL_GPUS:-${PHYSICAL_GPU:-0}}"
SERVER_NAME="0.0.0.0"
SERVER_PORT=7862
GRADIO_OFFLOAD_ARGS=()

# ╔══════════════════════════════════════════════════════════════════╗
# ║  以下不用改                                                       ║
# ╚══════════════════════════════════════════════════════════════════╝

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
# setup.sh 默认安装到 /root/.venv/dreamface-omni（非 diffsynth-gradio）
# 如需覆盖：export DREAMFACE_OMINI_VENV=/your/path
VENV_DIR="${DREAMFACE_OMINI_VENV:-/root/.venv/dreamface-omni}"

# LoRA 模型路径
#   - adapter_model.safetensors 是 PEFT/diffusers 格式（base_model.model. 前缀），不兼容
#   - diffsynth_lora.safetensors 是 DiffSynth 格式，匹配 pipe.dit 层名
LORA_PATH="${LORA_PATH:-/mnt/image-edit/models/hithink-image-labs/DreamFace_lora/v2.1/diffsynth_lora.safetensors}"

if [ -f "${VENV_DIR}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
fi
PYTHON="${VENV_DIR}/bin/python"
# if [ ! -x "${PYTHON}" ]; then
#     PYTHON="/usr/bin/python"
# fi

if [ -n "$LORA_PATH" ] && [ ! -f "$LORA_PATH" ]; then
    echo "[ERROR] LoRA 文件不存在: ${LORA_PATH}"
    exit 1
fi

if [ -n "$VAE_PATH" ]; then
    if [ -d "$VAE_PATH" ] && [ ! -f "${VAE_PATH}/diffusion_pytorch_model.safetensors" ]; then
        echo "[ERROR] VAE 目录中找不到 diffusion_pytorch_model.safetensors: ${VAE_PATH}"
        exit 1
    elif [ ! -d "$VAE_PATH" ] && [ ! -f "$VAE_PATH" ]; then
        echo "[ERROR] VAE 路径不存在: ${VAE_PATH}"
        exit 1
    fi
fi

# 32B / FLUX.2-dev：默认双卡；仅 1 卡时由 Python 侧自动 CPU offload
if [[ "${TRANSFORMER_MODEL_ID}" == *"FLUX.2-dev"* ]] || [[ "${TRANSFORMER_MODEL_ID,,}" == *"32b"* ]]; then
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
    if [[ "${GPU_COUNT}" -ge 2 ]] && [[ "${PHYSICAL_GPUS}" != *,* ]]; then
        PHYSICAL_GPUS="0,1"
        echo "[vram] 32B 模型：自动使用双卡 CUDA_VISIBLE_DEVICES=${PHYSICAL_GPUS}"
    elif [[ "${GPU_COUNT}" -lt 2 ]]; then
        GRADIO_OFFLOAD_ARGS=(--offload)
        echo "[vram] 32B 模型：仅 ${GPU_COUNT} 卡，启用 CPU offload"
    fi
fi

echo ""
echo "============================================"
echo "  Gradio: ${EXP_NAME}"
echo "  GPU: physical ${PHYSICAL_GPUS} -> cuda:0[,1]"
echo "  Transformer model id: ${TRANSFORMER_MODEL_ID}"
echo "  LoRA: ${LORA_PATH:-none}"
echo "  LoRA alpha: ${LORA_ALPHA}"
echo "  VAE: ${VAE_PATH:-default (FLUX.2-klein-base-9B)}"
echo "  S²-Guidance: scale=${S2_SCALE} drop_ratio=${S2_DROP_RATIO} range=[${S2_START}, ${S2_END}]"
echo "  Sigma mu override: ${SIGMA_MU} (<=0 表示自动)"
echo "  生图日志: ${LOG_DIR}"
echo "  Python: ${PYTHON}"
echo "  URL: http://${SERVER_NAME}:${SERVER_PORT}"
echo "============================================"
echo ""

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPUS}"
export DIFFSYNTH_MODEL_BASE_PATH="/mnt/image-edit/models"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONPATH="${PROJ_DIR}:${PYTHONPATH:-}"

"${PYTHON}" "${PROJ_DIR}/examples/flux2/model_inference/gradio_infer.py" \
  --transformer_model_id "$TRANSFORMER_MODEL_ID" \
  --lora "$LORA_PATH" \
  --lora_alpha "$LORA_ALPHA" \
  --vae_path "${VAE_PATH:-}" \
  --seed "$SEED" \
  --steps "$STEPS" \
  --cfg "$CFG" \
  --embedded_guidance "$EMBEDDED_GUIDANCE" \
  --s2_scale "$S2_SCALE" \
  --s2_drop_ratio "$S2_DROP_RATIO" \
  --s2_start "$S2_START" \
  --s2_end "$S2_END" \
  --sigma_mu "$SIGMA_MU" \
  --log_dir "$LOG_DIR" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --gpu 0 \
  "${GRADIO_OFFLOAD_ARGS[@]}" \
  --server_name "$SERVER_NAME" \
  --server_port "$SERVER_PORT"
