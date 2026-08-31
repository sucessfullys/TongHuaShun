#!/usr/bin/env bash
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"

# 可通过环境变量覆盖，例如:
#   VENV_DIR=/root/.venv/my-env bash setup.sh
VENV_DIR="${VENV_DIR:-/root/.venv/dreamface-omni}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
UV="${UV:-uv}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu129}"
TORCH_VERSION="${TORCH_VERSION:-2.12.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.27.1}"

if ! command -v "${UV}" >/dev/null 2>&1; then
    echo "[ERROR] 未找到 uv。安装:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "=== DreamFaceOmini uv setup ==="
echo "Project: ${PROJ_DIR}"
echo "Venv:    ${VENV_DIR}"
echo "Python:  ${PYTHON_VERSION}"
echo "Torch:   ${TORCH_VERSION} (index: ${TORCH_INDEX})"
echo ""

mkdir -p "$(dirname "${VENV_DIR}")"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
    echo "[1/3] 创建 venv ..."
    "${UV}" venv "${VENV_DIR}" --python "${PYTHON_VERSION}"
else
    echo "[1/3] venv 已存在，跳过创建"
fi

PY="${VENV_DIR}/bin/python"

echo "[2/3] 安装 PyTorch (CUDA) + torchaudio ..."
"${UV}" pip install --python "${PY}" \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    torchaudio \
    --index-url "${TORCH_INDEX}"

echo "[3/3] 安装 Gradio 推理依赖 ..."
"${UV}" pip install --python "${PY}" -r "${PROJ_DIR}/requirements-gradio.txt"

echo ""
echo "验证安装 ..."
"${PY}" - <<PY
import sys
sys.path.insert(0, "${PROJ_DIR}")
import torch
import gradio
import diffusers
import transformers
import safetensors
import einops
import modelscope
import imageio
import torchaudio
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig

print(f"  torch        {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"  torchaudio   {torchaudio.__version__}")
print(f"  gradio       {gradio.__version__}")
print(f"  diffusers    {diffusers.__version__}")
print(f"  transformers {transformers.__version__}")
print(f"  modelscope   {modelscope.__version__}")
print(f"  imageio      {imageio.__version__}")
print("  diffsynth import ok")
PY

echo ""
echo "Done."
echo "  activate:  source ${VENV_DIR}/bin/activate"
echo "  gradio:    bash ${PROJ_DIR}/run_gradio.sh"
echo ""
echo "环境变量（可选）:"
echo "  export DREAMFACE_OMINI_VENV=${VENV_DIR}"
