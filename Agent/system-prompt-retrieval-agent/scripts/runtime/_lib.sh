#!/usr/bin/env bash
# Shared helpers for scripts/runtime/*.sh — sourced, not executed.
# All start/kill scripts must `source` this file at the top.

set -euo pipefail

# Repo layout
RUNTIME_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_DIR="${RUNTIME_REPO_ROOT}/.runtime"
LOG_DIR="${RUNTIME_DIR}"
LOCAL_AGENT_ROOT="${RUNTIME_REPO_ROOT}/System-Prompt-Retrieval-Agent"
REMOTE_HOST="3h100"
REMOTE_APP_DIR="/mnt/image-edit/datasets/xywang/code/System-Prompt-Retrieval-Agent/Image-Generater-Remote"
REMOTE_DATASET="${REMOTE_DATASET:-/data/local_datasets/VirtualTryOn_whq_test500/test_data}"

# Authoritative controller port (the URL the local agent dials).
# The tunnel must forward this AND must not report ready until 17700 answers.
CONTROLLER_PORT=17700
# Ports the tunnel forwards (controller + 3 GPU supervisors).
RUNTIME_PORTS=(17700 17610 17611 17612)

# PID files
TUNNEL_PID_FILE="${RUNTIME_DIR}/tunnel.pid"
AGENT_PID_FILE="${RUNTIME_DIR}/agent.pid"
TUNNEL_LOG="${LOG_DIR}/tunnel.log"
AGENT_LOG="${LOG_DIR}/agent.log"

mkdir -p "${RUNTIME_DIR}"

log() { printf '[%s] [%s] %s\n' "$(date +%H:%M:%S)" "${1:-runtime}" "${2:-}"; }

# pid_alive <pid> — returns 0 if pid is a live process we can signal
pid_alive() {
    local pid="${1:-}"
    [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

# read_pid_file <path> — echoes the pid if file exists and contains a live pid; else empty
read_pid_file() {
    local path="$1"
    if [[ -f "${path}" ]]; then
        local pid
        pid="$(cat "${path}" 2>/dev/null || true)"
        if pid_alive "${pid}"; then
            echo "${pid}"
            return 0
        fi
        # Stale file — clean up
        rm -f "${path}"
    fi
    echo ""
}

# wait_for <timeout_seconds> <command...> — polls command at 1Hz, returns 0 on first success
wait_for() {
    local timeout="$1"; shift
    local i=0
    while (( i < timeout )); do
        if "$@" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
        i=$((i + 1))
    done
    return 1
}

# port_healthy <port> — check /health on localhost
port_healthy() {
    curl -fsS --max-time 3 "http://127.0.0.1:${1}/health" >/dev/null 2>&1
}

# any_port_healthy — true if any RUNTIME_PORT answers /health locally
any_port_healthy() {
    local p
    for p in "${RUNTIME_PORTS[@]}"; do
        port_healthy "${p}" && return 0
    done
    return 1
}

# safe_kill <pid_file> <label> — SIGTERM, wait 10s, SIGKILL fallback, clean PID file
safe_kill() {
    local pid_file="$1"
    local label="$2"
    local pid
    pid="$(read_pid_file "${pid_file}")"
    if [[ -z "${pid}" ]]; then
        log "${label}" "not running"
        return 0
    fi
    log "${label}" "stopping pid=${pid}"
    kill "${pid}" 2>/dev/null || true
    local i=0
    while (( i < 10 )) && pid_alive "${pid}"; do
        sleep 1
        i=$((i + 1))
    done
    if pid_alive "${pid}"; then
        log "${label}" "SIGTERM unresponsive — sending SIGKILL"
        kill -9 "${pid}" 2>/dev/null || true
        sleep 1
    fi
    rm -f "${pid_file}"
    log "${label}" "stopped"
}
