#!/usr/bin/env bash
set -euo pipefail

HANDOFF_DIR="/mnt/image-edit/datasets/duanyufa/交接/增强模型文件"
PYTHON_BIN="${PYTHON_BIN:-/mnt/image-edit/datasets/duanyufa/conda_envs/DiffSynth/bin/python}"
INPUT_IMAGE="${1:-${INPUT_IMAGE:-}}"
ANNOTATION="${2:-${ANNOTATION:-}}"
OUTPUT_IMAGE="${3:-${OUTPUT_IMAGE:-${HANDOFF_DIR}/output.png}}"

if [[ -z "${INPUT_IMAGE}" || -z "${ANNOTATION}" ]]; then
  echo "Usage: bash ${HANDOFF_DIR}/run_infer_single.sh /path/input.png /path/annotation.txt /path/output.png" >&2
  echo "Annotation can be .txt, .json, or .jsonl." >&2
  exit 1
fi

"${PYTHON_BIN}" "${HANDOFF_DIR}/infer_single_lre.py" \
  --input "${INPUT_IMAGE}" \
  --annotation "${ANNOTATION}" \
  --output "${OUTPUT_IMAGE}" \
  --checkpoint "${HANDOFF_DIR}/step-10000.safetensors" \
  --base-model "${BASE_MODEL:-/mnt/image-edit/datasets/duanyufa/FLUX.2-klein-base-4B}" \
  --template-model "${TEMPLATE_MODEL:-${HANDOFF_DIR}/Template-KleinBase4B-Enhance}" \
  --device "${DEVICE:-cuda:0}" \
  --num-inference-steps "${NUM_INFERENCE_STEPS:-50}" \
  --cfg-scale "${CFG_SCALE:-4.0}" \
  --embedded-guidance "${EMBEDDED_GUIDANCE:-4.0}" \
  --lre-strength "${LRE_STRENGTH:-0.8}"
