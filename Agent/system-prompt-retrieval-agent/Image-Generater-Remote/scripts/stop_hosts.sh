#!/usr/bin/env bash
# stop_hosts.sh — Stop all supervisors gracefully, SIGKILL after 10s.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${PROJECT_DIR}/logs"

PORTS=(17610 17611 17612)

for PORT in "${PORTS[@]}"; do
    PID_FILE="${LOG_DIR}/supervisor_${PORT}.pid"
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        echo "[stop_hosts] Stopping respawn loop for port ${PORT} (PID ${PID})..."
        kill "$PID" 2>/dev/null || true
        rm -f "$PID_FILE"
    fi
done

# Also kill any remaining uvicorn/server.app processes
echo "[stop_hosts] Sending SIGTERM to supervisor processes..."
pkill -f "server.app" 2>/dev/null || true

sleep 2

# Check if any are still running
REMAINING=$(pgrep -f "server.app" 2>/dev/null || true)
if [ -n "$REMAINING" ]; then
    echo "[stop_hosts] Some processes still alive — sending SIGKILL..."
    sleep 8
    pkill -9 -f "server.app" 2>/dev/null || true
fi

echo "[stop_hosts] All supervisors stopped."
