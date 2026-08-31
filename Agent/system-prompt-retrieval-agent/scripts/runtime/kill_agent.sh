#!/usr/bin/env bash
# Stop the local agent CLI tracked in .runtime/agent.pid.
# Idempotent: no-op if PID file is missing or PID is dead.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

safe_kill "${AGENT_PID_FILE}" "agent"
