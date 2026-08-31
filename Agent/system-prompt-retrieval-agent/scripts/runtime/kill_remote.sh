#!/usr/bin/env bash
# Stop the full remote stack on 3h100:
#   (a) workflow controller on 17700  (via remote logs/controller.pid)
#   (b) GPU supervisors via scripts/stop_hosts.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

LABEL="remote"

# ── (a) Controller ─────────────────────────────────────────────────────
log "${LABEL}" "stopping workflow controller on ${REMOTE_HOST}:${CONTROLLER_PORT}"
ssh "${REMOTE_HOST}" "bash -s" <<'REMOTE_SH' || log "${LABEL}" "controller stop reported non-zero (continuing)"
set -uo pipefail
cd /mnt/image-edit/datasets/xywang/code/System-Prompt-Retrieval-Agent/Image-Generater-Remote || exit 0
PID_FILE="logs/controller.pid"
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE" 2>/dev/null || true)
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        echo "[remote-controller] SIGTERM pid=$PID"
        kill "$PID" 2>/dev/null || true
        i=0
        while [ $i -lt 10 ] && kill -0 "$PID" 2>/dev/null; do
            sleep 1; i=$((i + 1))
        done
        if kill -0 "$PID" 2>/dev/null; then
            echo "[remote-controller] SIGKILL pid=$PID"
            kill -9 "$PID" 2>/dev/null || true
        fi
    else
        echo "[remote-controller] pid=$PID not alive"
    fi
    rm -f "$PID_FILE"
else
    echo "[remote-controller] no pid file"
fi
REMOTE_SH

# ── (b) GPU supervisors ────────────────────────────────────────────────
log "${LABEL}" "stopping GPU supervisors on ${REMOTE_HOST}"
ssh "${REMOTE_HOST}" "cd ${REMOTE_APP_DIR} && bash scripts/stop_hosts.sh" || {
    RC=$?
    log "${LABEL}" "remote stop_hosts.sh exited rc=${RC} (treated as best-effort)"
    exit 0
}
log "${LABEL}" "stopped"
