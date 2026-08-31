#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_ROOT="${REPO_ROOT}/src"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${REPO_ROOT}:${SRC_ROOT}:${PYTHONPATH}"
else
  export PYTHONPATH="${REPO_ROOT}:${SRC_ROOT}"
fi

PYTHON_BIN=${PYTHON_BIN:-python3}

BOOTSTRAP=${1:-100}
# DEFAULT_EXCLUDE_MODELS="FLUX2_dev_turbo,FLUX2_klein_4b,LongCat_Image_Edit,FLUX2_dev,GLM_Image,Qwen_Image_Edit,Qwen_Image_Edit_2509,Step1X_Edit_v1p2,BAGEL,FLUX1_Kontext_dev,OmniGen2"
DEFAULT_EXCLUDE_MODELS=""
EXCLUDE_MODELS=${2:-$DEFAULT_EXCLUDE_MODELS}

RESULTS_ROOT="${REPO_ROOT}/data/e_geditv2_pair_res0821_2/openedit"

# Pick the latest run dir (name ends with YYYYMMDD_HHMMSS) and latest *.jsonl inside it.
find_latest_jsonl_under_glob() {
  local glob_rel="$1"
  local -a dirs=()
  local -a jsonls=()
  local newest_dir=""

  shopt -s nullglob
  dirs=( "${RESULTS_ROOT}"/${glob_rel} )
  shopt -u nullglob

  if [[ ${#dirs[@]} -eq 0 ]]; then
    echo "error: no result directory matching '${glob_rel}' under ${RESULTS_ROOT}" >&2
    return 1
  fi

  newest_dir="$(printf '%s\n' "${dirs[@]}" | sort | tail -n 1)"

  shopt -s nullglob
  jsonls=( "${newest_dir}"/*.jsonl )
  shopt -u nullglob

  if [[ ${#jsonls[@]} -eq 0 ]]; then
    echo "error: no .jsonl under ${newest_dir}" >&2
    return 1
  fi

  newest_jsonl="$(printf '%s\n' "${jsonls[@]}" | sort | tail -n 1)"
  printf '%s' "${newest_jsonl}"
}

IF_RESULTS_PATH="$(find_latest_jsonl_under_glob "eval_if_gemma4_metadata_metadata_*")" || exit 1
VC_RESULTS_PATH="$(find_latest_jsonl_under_glob "eval_vc_metadata_metadata_*")" || exit 1
VQ_RESULTS_PATH="$(find_latest_jsonl_under_glob "eval_vq_gemma4_metadata_metadata_*")" || exit 1

echo "IF_RESULTS_PATH=${IF_RESULTS_PATH}"
echo "VC_RESULTS_PATH=${VC_RESULTS_PATH}"
echo "VQ_RESULTS_PATH=${VQ_RESULTS_PATH}"

CMD=(
  "${PYTHON_BIN}" "${SRC_ROOT}/common_utils/elo_score.py"
  --result-files "$VC_RESULTS_PATH,$VQ_RESULTS_PATH,$IF_RESULTS_PATH"
  --bootstrap "$BOOTSTRAP"
  --alpha 1
  --dimension-weighting "balanced"
  --seed 42
  --table-output "${REPO_ROOT}/tmp_elo_table.html"
)

if [[ -n "${EXCLUDE_MODELS}" ]]; then
  CMD+=(--exclude-models "${EXCLUDE_MODELS}")
fi

"${CMD[@]}"
