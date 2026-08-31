#!/usr/bin/env bash
# Open an SSH tunnel from the local Mac to 3h100, forwarding the
# controller port (17700) plus all 3 GPU supervisor ports (17610-12).
# Idempotent: if a live tunnel PID already answers /health, exits 0.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

LABEL="tunnel"

# Already running and healthy? Require the controller port (17700) specifically —
# a healthy GPU supervisor on 17610 is NOT proof the agent endpoint is reachable.
EXISTING="$(read_pid_file "${TUNNEL_PID_FILE}")"
if [[ -n "${EXISTING}" ]] && port_healthy "${CONTROLLER_PORT}"; then
    log "${LABEL}" "already running pid=${EXISTING} (controller :${CONTROLLER_PORT} healthy)"
    exit 0
fi

# Build -L flags for every forwarded port
FORWARD_ARGS=()
for p in "${RUNTIME_PORTS[@]}"; do
    FORWARD_ARGS+=(-L "${p}:127.0.0.1:${p}")
done

log "${LABEL}" "opening tunnel to ${REMOTE_HOST} forwarding ports: ${RUNTIME_PORTS[*]}"

# nohup + setsid-ish detach. -N = no remote command, -T = no tty,
# -o ServerAliveInterval keeps the tunnel from being torn down by NAT idle.
nohup ssh -N -T \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    "${FORWARD_ARGS[@]}" \
    "${REMOTE_HOST}" \
    >> "${TUNNEL_LOG}" 2>&1 &
TUNNEL_PID=$!
echo "${TUNNEL_PID}" > "${TUNNEL_PID_FILE}"
log "${LABEL}" "started pid=${TUNNEL_PID}; waiting for controller :${CONTROLLER_PORT} via tunnel (≤30s)"

if wait_for 30 port_healthy "${CONTROLLER_PORT}"; then
    log "${LABEL}" "ready (controller :${CONTROLLER_PORT} reachable)"
    exit 0
fi

log "${LABEL}" "FAILED — controller :${CONTROLLER_PORT} did not respond within 30s; see ${TUNNEL_LOG}"
# Kill the dud tunnel so a retry can succeed
kill "${TUNNEL_PID}" 2>/dev/null || true
rm -f "${TUNNEL_PID_FILE}"
exit 1
