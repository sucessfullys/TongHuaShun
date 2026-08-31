#!/usr/bin/env bash
# run_full_eval.sh — Deploy pipeline to 3h100, install deps, run full evaluation.
# Usage: bash eval_run/run_full_eval.sh [--stage generate|evaluate|all] [--limit N]
set -euo pipefail

SSH="ssh -o StrictHostKeyChecking=no 3h100"
REMOTE_DIR="/mnt/image-edit/datasets/xywang/code/System-Prompt-Retrieval-Agent/eval_run"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_RESULTS="/Volumes/970SSD/Code/Git/System-Prompt-Retrieval-Agent/eval_results"

STAGE="${1:-all}"
LIMIT="${2:-0}"

echo "=========================================="
echo "V0.1 Full Evaluation Pipeline"
echo "Stage: $STAGE  Limit: $LIMIT"
echo "=========================================="

# 1. Create remote directory
echo "[1/6] Creating remote directory..."
$SSH "mkdir -p $REMOTE_DIR"

# 2. Deploy pipeline scripts
echo "[2/6] Deploying pipeline to remote..."
rsync -av --progress \
    -e "ssh -o StrictHostKeyChecking=no" \
    "$LOCAL_DIR/eval_pipeline.py" \
    "$LOCAL_DIR/prompts_data.py" \
    "3h100:$REMOTE_DIR/"

# 3. Install missing dependencies
echo "[3/6] Installing missing dependencies on remote..."
$SSH "pip3 install diffusers 2>&1 | tail -5"

# 4. Run pipeline
echo "[4/6] Running pipeline (stage=$STAGE, limit=$LIMIT)..."
echo "  This will take a while. Monitor with: ssh 3h100 'tail -f $REMOTE_DIR/eval_pipeline.log'"
$SSH "cd $REMOTE_DIR && python3 eval_pipeline.py --stage $STAGE --limit $LIMIT 2>&1 | tee pipeline_output.log"

# 5. Copy results back to local
echo "[5/6] Copying results to local machine..."
mkdir -p "$LOCAL_RESULTS"
rsync -av --progress \
    -e "ssh -o StrictHostKeyChecking=no" \
    "3h100:/mnt/image-edit/datasets/xywang/code/System-Prompt-Retrieval-Agent/eval_results/" \
    "$LOCAL_RESULTS/"

# 6. Print summary
echo "[6/6] Done!"
echo ""
if [ -f "$LOCAL_RESULTS/final_summary.json" ]; then
    echo "=== FINAL RESULTS ==="
    python3 -c "
import json
with open('$LOCAL_RESULTS/final_summary.json') as f:
    data = json.load(f)
for r in data.get('results', []):
    print(f\"  {r['prompt_id']}: pass_rate={r['pass_rate_pct']} ({r['yes']} yes / {r['total_evaluated']} total)\")
"
else
    echo "Results not yet available (check remote logs)"
fi
