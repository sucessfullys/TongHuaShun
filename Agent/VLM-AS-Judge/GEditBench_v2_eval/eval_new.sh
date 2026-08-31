# CHANGE HISTORY:
# - 2026-06-02: Copied from eval.sh.
# - 2026-06-02: Run evaluation pipelines with IE_agent_custom Python because
#   the vllm_new environment does not provide an `autopipeline` executable.
EVAL_PYTHON="/mnt/image-edit/datasets/dingbaojin/conda_envs/IE_agent_custom/bin/python"

BASE_DIR="/mnt/image-edit/datasets/dingjianbiao/agent/benchmark/GEditBench_v2"
export PYTHONPATH="$BASE_DIR/src:$BASE_DIR:${PYTHONPATH:-}"
#GEDITV2_METADATA_FILE="$BASE_DIR/datasets/GEditBench-v2-CandidatesGallery/metadata_20260602_114247.jsonl"
GEDITV2_METADATA_FILE="$BASE_DIR/datasets/GEditBench-v2-CandidatesGallery/metadata_20260604_160443.jsonl"

#SAVE_PATH="$BASE_DIR/data/e_geditv2_pair_res0602_1"
SAVE_PATH="$BASE_DIR/data/e_geditv2_pair_res_flux2_klein_4b_0604"
LOG_DIR="$BASE_DIR/logs"


cleanup_vllm_servers() {
  local status=$?

  trap - EXIT INT TERM

  for pid in "${gemma_server_pid:-}" "${pvc_server_pid:-}"; do
    if [ -n "$pid" ] && kill -0 "$pid" >/dev/null 2>&1; then
      echo "Stopping vLLM server process group: $pid"
      kill -TERM -- "-$pid" >/dev/null 2>&1 || kill -TERM "$pid" >/dev/null 2>&1 || true
    fi
  done

  sleep 5

  for pid in "${gemma_server_pid:-}" "${pvc_server_pid:-}"; do
    if [ -n "$pid" ] && kill -0 "$pid" >/dev/null 2>&1; then
      echo "Force stopping vLLM server process group: $pid"
      kill -KILL -- "-$pid" >/dev/null 2>&1 || kill -KILL "$pid" >/dev/null 2>&1 || true
    fi
  done

  exit "$status"
}

wait_for_vllm() {
  local name="$1"
  local port="$2"
  local pid="$3"
  local timeout_seconds="${4:-1800}"
  local start_time

  start_time="$(date +%s)"
  echo "Waiting for $name vLLM server on port $port..."

  while true; do
    if curl -fsS "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
      echo "$name vLLM server is ready."
      return 0
    fi

    if ! kill -0 "$pid" >/dev/null 2>&1; then
      echo "$name vLLM server exited before becoming ready." >&2
      return 1
    fi

    if [ "$(($(date +%s) - start_time))" -ge "$timeout_seconds" ]; then
      echo "Timed out waiting for $name vLLM server on port $port." >&2
      return 1
    fi

    sleep 5
  done
}

# 启动gemma4和PVC-Judge vLLM服务器
mkdir -p "$LOG_DIR"
trap cleanup_vllm_servers EXIT INT TERM

CUDA_VISIBLE_DEVICES=0,1 setsid bash vllm_deploy_scripts/gemma4.sh > "$LOG_DIR/gemma4.log" 2>&1 &
gemma_server_pid=$!

CUDA_VISIBLE_DEVICES=2 setsid bash vllm_deploy_scripts/pvc_judge.sh > "$LOG_DIR/pvc_judge.log" 2>&1 &
pvc_server_pid=$!

echo "Started gemma4 vLLM server: pid=$gemma_server_pid, log=$LOG_DIR/gemma4.log"
echo "Started PVC-Judge vLLM server: pid=$pvc_server_pid, log=$LOG_DIR/pvc_judge.log"

wait_for_vllm "gemma4" 25929 "$gemma_server_pid" || exit 1
wait_for_vllm "PVC-Judge" 25930 "$pvc_server_pid" || exit 1


# 开始两两比对，得到VQ VC IF比对结果
# python -m src.cli.autopipeline eval \
(
  # CHANGED 2026-06-02: original command was `autopipeline eval`.
  "$EVAL_PYTHON" -m src.cli.autopipeline eval \
    --bmk geditv2 \
    --pipeline-config-path ./configs/pipelines/vlm_as_a_judge/eval_vc.yaml \
    --user-config ./configs/pipelines/user_config.yaml \
    --save-path "$SAVE_PATH" \
    --max-workers 8 \
    --geditv2-metadata-file "$GEDITV2_METADATA_FILE"
) &
vc_pid=$!

(
  # CHANGED 2026-06-02: original command was `autopipeline eval`.
  "$EVAL_PYTHON" -m src.cli.autopipeline eval \
    --bmk geditv2 \
    --pipeline-config-path ./configs/pipelines/vlm_as_a_judge/eval_vq_gemma4.yaml \
    --user-config ./configs/pipelines/user_config.yaml \
    --save-path "$SAVE_PATH" \
    --max-workers 8 \
    --geditv2-metadata-file "$GEDITV2_METADATA_FILE"
) &
vq_pid=$!

(
  # CHANGED 2026-06-02: original command was `autopipeline eval`.
  "$EVAL_PYTHON" -m src.cli.autopipeline eval \
    --bmk geditv2 \
    --pipeline-config-path ./configs/pipelines/vlm_as_a_judge/eval_if_gemma4.yaml \
    --user-config ./configs/pipelines/user_config.yaml \
    --save-path "$SAVE_PATH" \
    --max-workers 8 \
    --geditv2-metadata-file "$GEDITV2_METADATA_FILE"
) &
if_pid=$!


status=0
wait "$vc_pid" || status=1
wait "$vq_pid" || status=1
wait "$if_pid" || status=1
exit "$status"
