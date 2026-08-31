#!/usr/bin/env bash
# sync_to_remote.sh — Dry-run by default, real rsync with --apply.
# Usage: bash scripts/sync_to_remote.sh [--apply]
set -euo pipefail

LOCAL_ROOT="/Volumes/970SSD/Code/Git/System-Prompt-Retrieval-Agent/Image-Generater-Remote/"
REMOTE_ROOT="3h100:/mnt/image-edit/datasets/xywang/code/System-Prompt-Retrieval-Agent/Image-Generater-Remote/"

EXCLUDES=(
    --exclude='.venv'
    --exclude='config.yaml'
    --exclude='.env'
    --exclude='.env.*'
    --exclude='*.key'
    --exclude='*.pem'
    --exclude='runs/'
    --exclude='outputs/'
    --exclude='logs/'
    --exclude='__pycache__/'
    --exclude='.pytest_cache/'
)

# Create remote directory first
echo "[sync] Ensuring remote directory exists..."
ssh 3h100 'mkdir -p /mnt/image-edit/datasets/xywang/code/System-Prompt-Retrieval-Agent/Image-Generater-Remote'

if [ "${1:-}" = "--apply" ]; then
    echo "[sync] REAL rsync (--apply flag set)..."
    rsync -av "${EXCLUDES[@]}" "$LOCAL_ROOT" "$REMOTE_ROOT"
    echo "[sync] Done. Verifying remote contents..."
    ssh 3h100 'ls /mnt/image-edit/datasets/xywang/code/System-Prompt-Retrieval-Agent/Image-Generater-Remote/'
else
    echo "[sync] DRY-RUN (pass --apply for real rsync)..."
    rsync -av --dry-run "${EXCLUDES[@]}" "$LOCAL_ROOT" "$REMOTE_ROOT"
    echo ""
    echo "[sync] Dry-run complete. Review above and run with --apply to execute."
fi
