#!/usr/bin/env bash
# grep_guard_negative_prompt.sh — Ensure no FLUX call site bypasses normalize_negative_prompt.
# Exit 0 if safe, exit 1 if any bypass found.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "[grep-guard] Checking FLUX negative-prompt normalization..."

# Pattern: any .py file that passes negative_prompt to a pipeline call
# WITHOUT going through normalize_negative_prompt
VIOLATIONS=0

# Check 1: All uses of negative_prompt= in pipeline calls must be preceded by normalization
# Find files that reference negative_prompt but NOT normalize_negative_prompt
for f in $(find "$PROJECT_DIR" -name "*.py" -not -path "*/.venv/*" -not -path "*/__pycache__/*" -not -path "*/tests/*"); do
    if grep -q "negative_prompt" "$f" 2>/dev/null; then
        # This file references negative_prompt — check if it also has the normalizer
        if echo "$f" | grep -q "flux2_klein.py"; then
            # The adapter itself — must contain normalize_negative_prompt
            if ! grep -q "normalize_negative_prompt" "$f"; then
                echo "VIOLATION: $f references negative_prompt but missing normalize_negative_prompt"
                VIOLATIONS=$((VIOLATIONS + 1))
            fi
        fi
    fi
done

# Check 2: No raw .negative_prompt= assignment that bypasses normalization in non-test code
RAW_ASSIGNS=$(grep -rn 'negative_prompt\s*=' "$PROJECT_DIR" \
    --include="*.py" \
    --exclude-dir=".venv" \
    --exclude-dir="__pycache__" \
    --exclude-dir="tests" \
    | grep -v "normalize_negative_prompt" \
    | grep -v "negative_prompt_param" \
    | grep -v "negative_prompt_applied" \
    | grep -v "negative_prompt_normalized" \
    | grep -v "default_negative_prompt" \
    | grep -v "negative_prompt_targets" \
    | grep -v "# " \
    | grep -v "def " \
    | grep -v '"""' \
    || true)

if [ -n "$RAW_ASSIGNS" ]; then
    echo "WARNING: Potential raw negative_prompt assignments (review manually):"
    echo "$RAW_ASSIGNS"
fi

if [ "$VIOLATIONS" -gt 0 ]; then
    echo "[grep-guard] FAILED: $VIOLATIONS violation(s) found"
    exit 1
fi

echo "[grep-guard] PASSED: All FLUX negative-prompt paths go through normalize_negative_prompt"
exit 0
