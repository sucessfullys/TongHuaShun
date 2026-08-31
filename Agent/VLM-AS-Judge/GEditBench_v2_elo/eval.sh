BASE_DIR="/mnt/workspace_dbj/subtask_eval/benchmark/GEditBench_v2"
GEDITV2_METADATA_FILE="$BASE_DIR/datasets/GEditBench-v2-CandidatesGallery/metadata_20260520_171922.jsonl"
SAVE_PATH="$BASE_DIR/data/e_geditv2_pair_res_0522"
LOG_DIR="$BASE_DIR/logs"
export GEDITV2_GEMMA4_PORT=8431

export PYTHONPATH="$BASE_DIR/src:$BASE_DIR:$PYTHONPATH"

CONDA_VLLM_ENV="vllm"
CONDA_TEST_ENV="IE_agent_custom"


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

# gemma4 服务已在本地部署 (port 8431, GPU 2,3)，脚本不再启动
# trap cleanup_vllm_servers EXIT INT TERM

# CUDA_VISIBLE_DEVICES=0,1 setsid bash vllm_deploy_scripts/gemma4.sh > "$LOG_DIR/gemma4.log" 2>&1 &
# gemma_server_pid=$!

# echo "Started gemma4 vLLM server: pid=$gemma_server_pid, log=$LOG_DIR/gemma4.log"
# wait_for_vllm "gemma4" 25931 "$gemma_server_pid" || exit 1

# PVC-Judge 使用 GPU 0 启动
mkdir -p "$LOG_DIR"
trap cleanup_vllm_servers EXIT INT TERM

CUDA_VISIBLE_DEVICES=1 setsid conda run -n "$CONDA_VLLM_ENV" bash vllm_deploy_scripts/pvc_judge.sh > "$LOG_DIR/pvc_judge.log" 2>&1 &
pvc_server_pid=$!

echo "Started PVC-Judge vLLM server: pid=$pvc_server_pid, log=$LOG_DIR/pvc_judge.log"

wait_for_vllm "PVC-Judge" 25930 "$pvc_server_pid" || exit 1

cd "$BASE_DIR"

# 开始两两比对，得到VQ VC IF比对结果
(
  conda run -n "$CONDA_TEST_ENV" python3 -m src.cli.autopipeline eval \
    --bmk geditv2 \
    --pipeline-config-path ./configs/pipelines/vlm_as_a_judge/eval_vc.yaml \
    --user-config ./configs/pipelines/user_config.yaml \
    --save-path "$SAVE_PATH" \
    --max-workers 8 \
    --geditv2-metadata-file "$GEDITV2_METADATA_FILE"
) &
vc_pid=$!

(
  conda run -n "$CONDA_TEST_ENV" python3 -m src.cli.autopipeline eval \
    --bmk geditv2 \
    --pipeline-config-path ./configs/pipelines/vlm_as_a_judge/eval_vq_gemma4.yaml \
    --user-config ./configs/pipelines/user_config.yaml \
    --save-path "$SAVE_PATH" \
    --max-workers 8 \
    --geditv2-metadata-file "$GEDITV2_METADATA_FILE"
) &
vq_pid=$!

(
  conda run -n "$CONDA_TEST_ENV" python3 -m src.cli.autopipeline eval \
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