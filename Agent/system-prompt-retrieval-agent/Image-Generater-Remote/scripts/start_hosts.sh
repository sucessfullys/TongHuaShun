#!/usr/bin/env bash
# start_hosts.sh — Start 3 GPU-pinned supervisors with respawn-on-exit loop.
# Usage: bash scripts/start_hosts.sh [config.yaml]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONFIG="${1:-config.yaml}"
VENV="${PROJECT_DIR}/.venv/bin/python"
PYTHON="${VENV}"
if [ ! -f "$PYTHON" ]; then
    PYTHON="python"
fi

LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "$LOG_DIR"

# S5/F10f: Run grep guard before starting (non-fatal warning for now)
if [ -f "$SCRIPT_DIR/grep_guard_negative_prompt.sh" ]; then
    echo "[start_hosts] Running negative-prompt grep guard..."
    bash "$SCRIPT_DIR/grep_guard_negative_prompt.sh" || echo "[start_hosts] WARNING: grep guard failed (non-fatal)"
fi

PORTS=(17610 17611 17612)
GPUS=(0 1 2)

# Kill NoGPUAlarm if present
echo "[start_hosts] Killing NoGPUAlarmNew.py if running..."
pkill -f "NoGPUAlarmNew.py" 2>/dev/null || true
sleep 1
pkill -9 -f "NoGPUAlarmNew.py" 2>/dev/null || true

# Snapshot nvidia-smi
echo "[start_hosts] GPU snapshot before start:"
nvidia-smi --query-gpu=index,memory.used,memory.total,memory.free --format=csv 2>/dev/null || echo "(nvidia-smi not available)"

# Start supervisors with respawn loop
for i in 0 1 2; do
    PORT="${PORTS[$i]}"
    GPU="${GPUS[$i]}"
    PID_FILE="${LOG_DIR}/supervisor_${PORT}.pid"
    LOG_FILE="${LOG_DIR}/supervisor_${PORT}.log"

    echo "[start_hosts] Starting supervisor on port ${PORT} (GPU ${GPU})..."

    # Respawn loop in background
    (
        while true; do
            CUDA_VISIBLE_DEVICES="${GPU}" \
            SUPERVISOR_PORT="${PORT}" \
            CONFIG_PATH="${CONFIG}" \
            "$PYTHON" -m server.app >> "$LOG_FILE" 2>&1
            EXIT_CODE=$?
            echo "[respawn] Supervisor port=${PORT} exited with code=${EXIT_CODE} at $(date)" >> "$LOG_FILE"
            if [ $EXIT_CODE -eq 0 ]; then
                echo "[respawn] Clean exit — not respawning port=${PORT}" >> "$LOG_FILE"
                break
            fi
            echo "[respawn] Non-zero exit — respawning port=${PORT} in 2s..." >> "$LOG_FILE"
            sleep 2
        done
    ) &
    LOOP_PID=$!
    echo "$LOOP_PID" > "$PID_FILE"
    echo "[start_hosts] Supervisor port=${PORT} respawn loop PID=${LOOP_PID}"
done

# Write ports manifest
cat > "${LOG_DIR}/ports_manifest.json" << EOF
{
    "ports": [17610, 17611, 17612],
    "gpus": [0, 1, 2],
    "config": "${CONFIG}",
    "started_at": "$(date -Iseconds 2>/dev/null || date)"
}
EOF

# Wait for health checks
echo "[start_hosts] Waiting for supervisors to become healthy..."
sleep 3
FAILED=0
for PORT in "${PORTS[@]}"; do
    if curl -s --max-time 5 "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
        echo "[start_hosts] Port ${PORT}: healthy"
    else
        echo "[start_hosts] Port ${PORT}: NOT responding"
        FAILED=1
    fi
done

if [ $FAILED -ne 0 ]; then
    echo "[start_hosts] WARNING: Not all supervisors are healthy"
    exit 1
fi

echo "[start_hosts] All 3 supervisors running."
