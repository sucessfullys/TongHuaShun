---
name: todo-status-sync
description: >
  Updates the status of a todo item in the active todo.md after verification
  gates pass. Resolves the active todo path from CLAUDE.md at runtime; never
  embeds a version string or operates on _cc/_cx temporary files. Always checks
  gates before writing — never mutates status first.
---

## Purpose

Mark todo items with the correct status label after each verified milestone.
Protect the todo file's structure — never rewrite, reorder, or delete items.
Verification gates are checked **before** any write. No mutation if a gate fails.

## Status Labels

| Label | Meaning |
| --- | --- |
| `[ ]` | Not started |
| `[~]` | In progress |
| `[!]` | Blocked — smoke hard-stopped or halt condition |
| `[x]` | Done — all gates passed |

## Steps

1. Read `CLAUDE.md` §Source of Truth Pointer. Derive the active todo path by
   substituting `{active-version}`. The file is always `todo.md` — never a
   `_cc` or `_cx` file.

2. Open the active `todo.md`. Locate the item by its identifier or description.

3. Determine which gates apply for the requested transition (see below).
   **Run all applicable gates before writing anything.** If any gate fails,
   leave the item unchanged and surface the failure.

4. Only after all gates pass: update the status label in `todo.md`.

5. Update the progress ledger entry per the transition table:

   | Transition | Ledger action |
   | --- | --- |
   | `[ ]` → `[~]` | Check the **Ground** section boxes in the slot-checklist entry; leave other sections unchecked. |
   | `[~]` → `[!]` | Record the blocking reason and failed command/output in the ledger entry. |
   | `[~]` → `[x]` | All gates must pass first; check all remaining checklist boxes; mark "Todo item marked `[x]`" — this is the only transition that records completion. |

   Never mark "Todo item marked done / `[x]`" for `[~]` or `[!]` transitions.

## Verification Gates (checked before any write)

**Before `[x]` — all of the following must be confirmed:**

- Code exists in the correct scope: local project files under
  `System-Prompt-Retrieval-Agent/` excluding protected/generated paths, or
  remote server files under `Image-Generater-Remote/`.
- Smoke check passed and output is captured in the progress ledger.
- `dynamic-wiki-sync` has updated the matching wiki function page.
- **Config change:** both `config.yaml.example` and the `config_management`
  wiki page are updated.
- **Evaluator item:** all OpenAI API calls confirmed to route through
  `rate_limiter.py`; ≤3 req/s verified.
- **Remote server item:** `mkdir -p` ran, dry-run reviewed, user confirmed,
  real rsync via `3h100` alias completed, Python version verified, and remote
  smoke check passed.
- **Parallel session:** If `parallel-programming-coordinator` was used, confirm:
  ownership table is recorded in ledger; no worker edited `todo.md`, `plan.md`,
  wiki, or progress ledger; parent (not a worker) ran the integration smoke check;
  parent (not a worker) is now calling this skill.

**Before `[~]`:** Ground section of slot-checklist confirmed (paths resolved,
plan/todo/ledger present).

**Before `[!]`:** Blocking reason documented.

## Allowed Structural Edits

- Minimally correct a factual error with an inline annotation.
- Append new follow-up items to an existing "Follow-ups" or
  "Implementation Progress Ledger" section.
- If no safe append section exists, surface the gap instead of restructuring.

## Output Contract

- Confirmed path of the updated `todo.md`.
- The item identifier and old → new status label.
- Any verification gate that was not met (if transition was blocked).

## Hard Rules

- **Always check gates before writing status. No exceptions.**
- Never operate on `_cc` or `_cx` temporary files; only on `todo.md`.
- Never embed a version string as a constant; always derive from `CLAUDE.md`.
- Never rewrite, reorder, or delete todo items.
- Never insert new items into the middle of planned stages; append to follow-ups only.
- If unsure whether a verification gate is met, leave the item unchanged and
  surface the question to the user.
- Never edit `plan.md` from this skill.
