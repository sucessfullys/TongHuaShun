#!/usr/bin/env bash
# Start the full remote stack on 3h100:
#   (a) 3 GPU supervisors on 17610/17611/17612 via scripts/start_hosts.sh
#   (b) The workflow controller on 17700 via `python -m workflow.controller`
#       (this is the endpoint the local agent dials — required, not optional).
#
# Idempotent: each piece skips itself if already healthy on its port.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

LABEL="remote"

# ── (a) GPU supervisors ────────────────────────────────────────────────
log "${LABEL}" "starting GPU supervisors on ${REMOTE_HOST} (${REMOTE_APP_DIR})"
ssh "${REMOTE_HOST}" "cd ${REMOTE_APP_DIR} && bash scripts/start_hosts.sh config.yaml"

# ── (b) Workflow controller on 17700 ───────────────────────────────────
log "${LABEL}" "starting workflow controller on ${REMOTE_HOST}:${CONTROLLER_PORT} (dataset=${REMOTE_DATASET})"

# Single-shot remote one-liner: skip if /health already green; otherwise
# nohup the controller, write a remote PID file, wait up to 30s for /health.
ssh "${REMOTE_HOST}" "bash -s" <<REMOTE_SH
set -euo pipefail
cd "${REMOTE_APP_DIR}"

PID_FILE="logs/controller.pid"
LOG_FILE="logs/controller.log"
mkdir -p logs

# Already healthy?
if curl -fsS --max-time 3 "http://127.0.0.1:${CONTROLLER_PORT}/health" >/dev/null 2>&1; then
    echo "[remote-controller] already healthy on :${CONTROLLER_PORT}"
    exit 0
fi

# Stale PID file — clean up
if [ -f "\$PID_FILE" ]; then
    OLD=\$(cat "\$PID_FILE" 2>/dev/null || true)
    if [ -n "\$OLD" ] && kill -0 "\$OLD" 2>/dev/null; then
        echo "[remote-controller] killing stale pid=\$OLD (port not healthy)"
        kill "\$OLD" 2>/dev/null || true
        sleep 2
        kill -9 "\$OLD" 2>/dev/null || true
    fi
    rm -f "\$PID_FILE"
fi

PYTHON=".venv/bin/python"
if [ ! -x "\$PYTHON" ]; then
    echo "CRITICAL: remote controller venv python missing or not executable: \$PYTHON"
    exit 2
fi

echo "[remote-controller] launching: \$PYTHON -m workflow.controller --config config.yaml --dataset ${REMOTE_DATASET} --port ${CONTROLLER_PORT}"
nohup "\$PYTHON" -m workflow.controller \
    --config config.yaml \
    --dataset "${REMOTE_DATASET}" \
    --port "${CONTROLLER_PORT}" \
    >> "\$LOG_FILE" 2>&1 &
echo \$! > "\$PID_FILE"
PID=\$(cat "\$PID_FILE")
echo "[remote-controller] started pid=\$PID; waiting for /health (≤30s)"

i=0
while [ \$i -lt 30 ]; do
    if curl -fsS --max-time 3 "http://127.0.0.1:${CONTROLLER_PORT}/health" >/dev/null 2>&1; then
        echo "[remote-controller] healthy"
        exit 0
    fi
    sleep 1
    i=\$((i + 1))
done

echo "[remote-controller] FAILED to become healthy within 30s — see \$LOG_FILE"
exit 1
REMOTE_SH

log "${LABEL}" "all remote services healthy (supervisors 17610/17611/17612 + controller ${CONTROLLER_PORT})"
