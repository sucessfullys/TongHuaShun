#!/usr/bin/env bash
#
# ERA preflight — verify the repo/environment is in a runnable state before
# /era:start or /era:resume enter the autonomous loop.
#
# It deliberately does NOT import or depend on the `era` package working — it
# is the check for exactly that. A broken repo (mid-merge, conflict markers,
# missing venv) otherwise surfaces only as a raw Python traceback once the
# loop is already underway.
#
# Usage: preflight.sh [repo_root]
#   repo_root defaults to the ERA repo inferred from this script's location
#   (<repo>/plugin/scripts/preflight.sh -> <repo>).
#
# Exit 0 = PASS · Exit 1 = FAIL (with an actionable message on stderr).

set -u

if [[ -n "${1:-}" ]]; then
  REPO="$1"
else
  REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
PY="$REPO/.venv/bin/python3"

fail() {
  echo "era preflight: FAIL — $1" >&2
  echo "Resolve the ERA repo, then re-run the command." >&2
  exit 1
}

# 1. repo root + virtualenv
[[ -d "$REPO" ]] || fail "repo root not found: $REPO"
[[ -x "$PY" ]] || fail "venv python missing: $PY
   run:  python3 -m venv .venv && .venv/bin/pip install -e ."

# 2. the era package imports (catches conflict markers, syntax errors, no install)
if ! import_err="$("$PY" -c 'import era' 2>&1)"; then
  fail "the 'era' package will not import — the repo may be mid-merge or hold
   conflict markers / syntax errors:
$import_err"
fi

# 3. git is not mid-merge / has no unresolved conflicts
if [[ -d "$REPO/.git" ]]; then
  [[ -f "$REPO/.git/MERGE_HEAD" ]] && fail "the ERA repo is mid-merge \
(.git/MERGE_HEAD present) — finish or abort the merge."
  if [[ -n "$(git -C "$REPO" ls-files -u 2>/dev/null)" ]]; then
    fail "the ERA repo has unmerged paths (a merge or squash left conflicts)."
  fi
fi

# 4. loop engine — the ralph-loop plugin's Stop hook needs `jq` to re-feed the
#    prompt each iteration. Missing `jq` is NOT fatal: /era:start routes to the
#    manual-fallback loop (which drives itself, with no Stop hook). Report the
#    mode so the slash-command can pick it without a cryptic mid-run crash.
if command -v jq >/dev/null 2>&1; then
  LOOP_ENGINE="ralph-loop plugin (jq present)"
else
  LOOP_ENGINE="manual fallback (jq not found — the ralph-loop Stop hook needs it)"
fi

# 5. Stage 6 code reviewer — the opt-in era-codex-reviewer needs the `codex`
#    CLI (its MCP mode). Missing `codex` is NOT fatal: Stage 6 self-reviews
#    each runner inline unless config.yaml sets experiment.codex_reviewer.
if command -v codex >/dev/null 2>&1; then
  CODEX_ENGINE="codex CLI present (era-codex-reviewer available when enabled)"
else
  CODEX_ENGINE="codex CLI not found — Stage 6 self-reviews runners inline"
fi

echo "era preflight: PASS — repo $REPO; 'era' importable; no merge in progress."
echo "era preflight: loop engine = $LOOP_ENGINE"
echo "era preflight: code reviewer = $CODEX_ENGINE"
exit 0
