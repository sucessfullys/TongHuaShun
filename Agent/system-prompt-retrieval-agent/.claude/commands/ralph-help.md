---
description: "Project-local Ralph Loop quick reference"
---

# Ralph Loop — Quick Reference (project-local)

This project vendors the Ralph Loop plugin under `.claude/`. Source: Anthropic's
official `ralph-loop` plugin (Apache 2.0). License copy: `.claude/hooks/RALPH_LOOP_LICENSE`.

## Files

```
.claude/commands/ralph-loop.md     ← /ralph-loop slash command
.claude/commands/cancel-ralph.md   ← /cancel-ralph slash command
.claude/commands/ralph-help.md     ← this file
.claude/hooks/ralph-stop-hook.sh   ← Stop-event hook (vendored verbatim)
.claude/scripts/setup-ralph-loop.sh ← initialiser (vendored verbatim)
.claude/settings.json              ← registers the Stop hook for the project
.claude/ralph-loop.local.md        ← runtime state (gitignored)
```

## Commands

```
/ralph-loop "Refactor X" --max-iterations 20
/ralph-loop "Fix auth bug" --completion-promise "FIXED" --max-iterations 10
/ralph-loop --help          # full option reference
/cancel-ralph               # remove the state file
```

## When to use it in this project

- Iterating until `pytest tests/` is fully green for a V0.2 module.
- Wiring follow-ups that require repeated edit/run cycles
  (`LocalApiEvaluator.evaluate_many` ↔ FLUX outputs; comparison-grid `image_map`).
- Tightening a single prompt-pair until `overall_score ≥ 0.80` against the mock controller.

## When NOT to use it

- Real-remote pilot runs (`limit=30+`) — these touch shared GPUs and OpenAI billing;
  use a regular session with explicit user approval per CLAUDE.md §Iron Law.
- `git push --force`, `rm -rf`, or any destructive operation that needs human review.
- Tasks with vague success criteria — Ralph cannot judge "good enough"; provide a
  concrete `--completion-promise` tied to a check (test pass, file present, score ≥ X).

## Rules (also enforced by CLAUDE.md §Ralph Loop Policy)

- Always set `--max-iterations` and `--completion-promise` together — never run a
  truly unbounded loop in this project.
- Never emit a false `<promise>` to escape.
- The state file `.claude/ralph-loop.local.md` is gitignored and per-session.
