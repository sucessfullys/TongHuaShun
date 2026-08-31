#!/usr/bin/env bash
# prepare_pilot_dataset.sh — Build the 30-sample curated pilot subset.
#
# Only runs if the resolved candidate has >30 samples AND the destination
# does not already exist (or --force is passed).
#
# Selection: first 10 sample directories per category (dress, lower, upper)
# by lexicographic sort. Each sample directory is copied intact with rsync -a.
#
# Destination: /mnt/image-edit/datasets/xywang/dataset/VirtualTryOn_whq_test500_pilot30/
#
# Usage:
#   bash scripts/prepare_pilot_dataset.sh --resolved <resolved_pilot_input.json> [--force]
set -euo pipefail

DEST_ROOT="/mnt/image-edit/datasets/xywang/dataset/VirtualTryOn_whq_test500_pilot30"
CATEGORIES=(dress lower upper)
SAMPLES_PER_CAT=10
FORCE=false
RESOLVED_JSON=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resolved)
            RESOLVED_JSON="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ -z "$RESOLVED_JSON" ]; then
    echo "Usage: $0 --resolved <resolved_pilot_input.json> [--force]"
    exit 1
fi

if [ ! -f "$RESOLVED_JSON" ]; then
    echo "ERROR: Resolved input file not found: $RESOLVED_JSON"
    exit 1
fi

# Extract the picked source path from resolved JSON
SOURCE_ROOT=$(python3 -c "import json; print(json.load(open('$RESOLVED_JSON'))['picked'])")
SAMPLE_COUNT=$(python3 -c "import json; d=json.load(open('$RESOLVED_JSON')); print(d.get('sample_count', -1))")

echo "[prepare_pilot] Source: $SOURCE_ROOT"
echo "[prepare_pilot] Total samples found: $SAMPLE_COUNT"

# Check if destination already exists
if [ -d "$DEST_ROOT" ] && [ "$FORCE" = false ]; then
    echo "[prepare_pilot] Destination already exists: $DEST_ROOT"
    echo "[prepare_pilot] Checking contents..."
    EXISTING=0
    for CAT in "${CATEGORIES[@]}"; do
        if [ -d "$DEST_ROOT/$CAT" ]; then
            COUNT=$(ls -1d "$DEST_ROOT/$CAT"/*/ 2>/dev/null | wc -l)
            echo "  $CAT: $COUNT samples"
            EXISTING=$((EXISTING + COUNT))
        fi
    done
    if [ "$EXISTING" -ge 30 ]; then
        echo "[prepare_pilot] Already has $EXISTING samples — skipping. Use --force to rebuild."
        exit 0
    fi
fi

# Check if curation is needed
if [ "$SAMPLE_COUNT" -le 30 ]; then
    echo "[prepare_pilot] Source has $SAMPLE_COUNT samples (<= 30). Using as-is."
    echo "[prepare_pilot] No curation needed — source IS the pilot root."
    # Create a symlink or just report
    echo "[prepare_pilot] Pilot root: $SOURCE_ROOT"
    exit 0
fi

# Curate: select first 10 per category
echo "[prepare_pilot] Building curated 30-sample subset..."
mkdir -p "$DEST_ROOT"

MANIFEST=""
for CAT in "${CATEGORIES[@]}"; do
    CAT_SRC="$SOURCE_ROOT/$CAT"
    CAT_DST="$DEST_ROOT/$CAT"
    mkdir -p "$CAT_DST"

    if [ ! -d "$CAT_SRC" ]; then
        echo "WARNING: Category dir not found: $CAT_SRC"
        continue
    fi

    # Get first N sample directories by lexicographic sort
    SELECTED=$(ls -1d "$CAT_SRC"/*/ 2>/dev/null | head -n "$SAMPLES_PER_CAT")
    COUNT=0
    while IFS= read -r SAMPLE_DIR; do
        [ -z "$SAMPLE_DIR" ] && continue
        SAMPLE_DIR="${SAMPLE_DIR%/}"          # F8: strip trailing slash so rsync copies the dir itself
        SAMPLE_NAME=$(basename "$SAMPLE_DIR")
        rsync -a "$SAMPLE_DIR" "$CAT_DST/"   # copies DIR into CAT_DST (not its contents)
        MANIFEST="$MANIFEST\n  $CAT/$SAMPLE_NAME"
        COUNT=$((COUNT + 1))
    done <<< "$SELECTED"

    echo "[prepare_pilot] $CAT: selected $COUNT samples"

    # F8: invariant assert — must have exactly SAMPLES_PER_CAT sample dirs
    GOT=$(ls -1d "$CAT_DST"/*/ 2>/dev/null | wc -l | tr -d ' ')
    if [ "$GOT" -ne "$SAMPLES_PER_CAT" ]; then
        echo "FATAL: $CAT has $GOT sample dirs, expected $SAMPLES_PER_CAT" >&2
        exit 2
    fi
done

echo ""
echo "[prepare_pilot] Done. Manifest of selected directories:"
echo -e "$MANIFEST"
echo ""

# Verify
TOTAL=0
for CAT in "${CATEGORIES[@]}"; do
    if [ -d "$DEST_ROOT/$CAT" ]; then
        COUNT=$(ls -1d "$DEST_ROOT/$CAT"/*/ 2>/dev/null | wc -l)
        TOTAL=$((TOTAL + COUNT))
    fi
done
echo "[prepare_pilot] Total: $TOTAL samples in $DEST_ROOT"

if [ "$TOTAL" -ne 30 ]; then
    echo "WARNING: Expected 30 samples, got $TOTAL"
fi
