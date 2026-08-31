#!/usr/bin/env bash
# kill_nogpualarm.sh — Kill NoGPUAlarmNew.py processes on remote.
set -euo pipefail
echo "[kill_nogpualarm] Attempting normal kill..."
ssh 3h100 'pkill -f "NoGPUAlarmNew.py" 2>/dev/null || echo "No processes found"'
sleep 2
echo "[kill_nogpualarm] Checking if still running..."
REMAINING=$(ssh 3h100 'pgrep -f "NoGPUAlarmNew.py" 2>/dev/null || true')
if [ -n "$REMAINING" ]; then
    echo "[kill_nogpualarm] Still running — sending SIGKILL..."
    ssh 3h100 'pkill -9 -f "NoGPUAlarmNew.py" 2>/dev/null || true'
    echo "[kill_nogpualarm] SIGKILL sent."
else
    echo "[kill_nogpualarm] Clean — no remaining processes."
fi
