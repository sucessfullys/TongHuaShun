#!/usr/bin/env bash
# Bring up the full stack: remote supervisors → SSH tunnel → (optional)
# local agent. Use --run-agent to also launch the agent CLI; any
# additional flags after --run-agent are forwarded to the CLI.
#
# Examples:
#   bash scripts/runtime/start_all.sh
#   bash scripts/runtime/start_all.sh --run-agent
#   bash scripts/runtime/start_all.sh --run-agent --max-rounds 6 --limit 30

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

LABEL="start_all"

RUN_AGENT=0
AGENT_ARGS=()
for arg in "$@"; do
    if [[ "${arg}" == "--run-agent" ]]; then
        RUN_AGENT=1
    else
        AGENT_ARGS+=("${arg}")
    fi
done

log "${LABEL}" "step 1/3 — remote supervisors"
bash "${SCRIPT_DIR}/start_remote.sh"

log "${LABEL}" "step 2/3 — SSH tunnel"
bash "${SCRIPT_DIR}/start_tunnel.sh"

if (( RUN_AGENT == 1 )); then
    log "${LABEL}" "step 3/3 — local agent"
    bash "${SCRIPT_DIR}/start_agent.sh" ${AGENT_ARGS[@]+"${AGENT_ARGS[@]}"}
    log "${LABEL}" "stack up — agent running (pid in ${AGENT_PID_FILE})"
else
    log "${LABEL}" "stack up — agent NOT started (pass --run-agent to launch it)"
    log "${LABEL}" "to start agent later: bash ${SCRIPT_DIR}/start_agent.sh [extra args]"
fi
