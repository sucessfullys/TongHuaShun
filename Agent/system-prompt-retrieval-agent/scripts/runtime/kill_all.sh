#!/usr/bin/env bash
# Tear down the full stack: agent → tunnel → remote supervisors.
# Best-effort: a failure in one step does NOT prevent the next.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

LABEL="kill_all"

log "${LABEL}" "step 1/3 — local agent"
bash "${SCRIPT_DIR}/kill_agent.sh" || log "${LABEL}" "kill_agent reported non-zero (continuing)"

log "${LABEL}" "step 2/3 — SSH tunnel"
bash "${SCRIPT_DIR}/kill_tunnel.sh" || log "${LABEL}" "kill_tunnel reported non-zero (continuing)"

log "${LABEL}" "step 3/3 — remote supervisors"
bash "${SCRIPT_DIR}/kill_remote.sh" || log "${LABEL}" "kill_remote reported non-zero (continuing)"

log "${LABEL}" "stack down"
exit 0
