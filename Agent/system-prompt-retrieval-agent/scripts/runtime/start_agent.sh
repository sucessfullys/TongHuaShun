#!/usr/bin/env bash
# Start the local production agent CLI in the background.
# All args after `--` (or simply trailing args) pass through to:
#   system-prompt-retrieval-agent run --config <config.yaml> [extra args]
# Idempotent: if a live agent PID is recorded, exits 0.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

LABEL="agent"

EXISTING="$(read_pid_file "${AGENT_PID_FILE}")"
if [[ -n "${EXISTING}" ]]; then
    log "${LABEL}" "already running pid=${EXISTING}"
    exit 0
fi

CONFIG="${LOCAL_AGENT_ROOT}/config.yaml"
if [[ ! -f "${CONFIG}" ]]; then
    log "${LABEL}" "config not found: ${CONFIG}"
    exit 2
fi

VENV_BIN="${LOCAL_AGENT_ROOT}/.venv/bin"
CLI="${VENV_BIN}/system-prompt-retrieval-agent"
if [[ ! -x "${CLI}" ]]; then
    log "${LABEL}" "CLI not found: ${CLI} — run 'pip install -e .' under ${LOCAL_AGENT_ROOT}"
    exit 2
fi

EXTRA_ARGS=("$@")
log "${LABEL}" "launching CLI with config=${CONFIG} extra_args=[${EXTRA_ARGS[*]:-}]"

(
    cd "${LOCAL_AGENT_ROOT}"
    nohup "${CLI}" run --config "${CONFIG}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
        >> "${AGENT_LOG}" 2>&1 &
    echo $! > "${AGENT_PID_FILE}"
)

PID="$(cat "${AGENT_PID_FILE}")"
log "${LABEL}" "started pid=${PID}; waiting for startup marker in ${AGENT_LOG} (≤30s)"

# Readiness: process alive AND log shows V022Runner / round / stage_dispatcher
agent_started() {
    pid_alive "${PID}" || return 1
    [[ -s "${AGENT_LOG}" ]] || return 1
    grep -qE "V022Runner|round|stage_dispatcher|round_id" "${AGENT_LOG}" 2>/dev/null
}

if wait_for 30 agent_started; then
    log "${LABEL}" "ready pid=${PID}"
    exit 0
fi

if pid_alive "${PID}"; then
    log "${LABEL}" "process is alive but no startup marker yet — leaving running; check ${AGENT_LOG}"
    exit 0
fi

log "${LABEL}" "FAILED — process exited; see ${AGENT_LOG}"
rm -f "${AGENT_PID_FILE}"
exit 1
