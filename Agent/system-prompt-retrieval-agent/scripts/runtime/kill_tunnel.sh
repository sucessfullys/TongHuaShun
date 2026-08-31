#!/usr/bin/env bash
# Stop the SSH tunnel tracked in .runtime/tunnel.pid.
# Idempotent: no-op if PID file is missing or PID is dead.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

safe_kill "${TUNNEL_PID_FILE}" "tunnel"
