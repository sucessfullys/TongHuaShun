---
name: implementing-todo-item
description: >
  Executes the ironclad two-phase loop for a single selected todo item: Phase 1
  implements code and runs smoke checks; Phase 2 updates wiki and marks the item
  done. Never stops at partial progress during implementation. Bounded retry
  prevents infinite loops on persistent failures.
---

## Purpose

Complete a selected todo item end-to-end without pausing for non-blocking choices.
During Phase 1, always choose the best option, state the choice and reasoning in
one line, and proceed. Only halt for: destructive actions, missing credentials or
API keys, unavailable remote hardware, missing model assets, or genuine ambiguity
that cannot be resolved by reading plan/todo/wiki/source.

See `reference/slot-checklist.md` for the canonical per-item checklist.

---

## Phase 1 — Code

1. Apply `reading-current-plan`. Confirm `plan.md`, `todo.md`, and progress
   ledger exist at the resolved active paths.

1b. **Parallelization check:** If the selected todo item has ≥2 subtasks with
    non-overlapping file scopes and no shared choke points in scope, apply
    `parallel-programming-coordinator` before step 2.
    - If coordinator returns `"single-agent"`: continue with step 2 as normal.
    - If coordinator returns integration complete: skip to step 9 (smoke check).
    Workers may not proceed to Phase 2; Phase 2 is always parent-only.

2. Add a progress ledger entry for the selected item using the canonical
   checklist from `reference/slot-checklist.md`. Mark **Ground** section done.

3. Mark todo item status `[~]` (in progress) via `todo-status-sync`.

4. **Evaluator items:** confirm that all OpenAI API calls route through
   `rate_limiter.py` and respect the ≤3 req/s cap from `CLAUDE.md` §Iron Law.

5. **Remote items:** confirm that missing model checkpoints raise a clear error
   and do not trigger any download. Remote path must be
   `/mnt/image-edit/datasets/xywang/code/System-Prompt-Retrieval-Agent/Image-Generater-Remote`.

6. **Config changes:** any new or modified runtime parameter must be written to
   `config.yaml.example` AND the `config_management` wiki page simultaneously.
   Do not defer either.

7. Implement in the correct directory per `CLAUDE.md` §Iron Law rule 2:
   - Local project code → `System-Prompt-Retrieval-Agent/` root only, excluding
     protected/generated paths.
   - Remote server code → `Image-Generater-Remote/` only.
   - Never mix local project and remote server ownership boundaries.
   Mark **Code implemented** in the progress ledger.

8. **Remote server items:** invoke `remote-image-service-deploy` after local
   code is ready.

9. Run the smoke check. Smoke-check selection order:

   1. If the todo item body contains an `Acceptance:` line, run that command.
      Format: `Acceptance: <shell command>`
   2. Else run the narrowest existing test that covers the changed code.
   3. Else run the minimum health command for the item type:
      - Local agent item: `python -c "import <touched_module>"`
      - Remote server item: `ssh 3h100 'cd <remote-path> && python -c "import <module>"'`
      - Evaluator/rate-limit item: `python -c "from rate_limiter import RateLimiter"`
      - Config-only item: `python -c "import yaml; yaml.safe_load(open('config.yaml.example'))"`
   4. Record the exact command and full output in the progress ledger.

   **Bounded retry:**
   - On failure, attempt one targeted fix and rerun smoke.
   - **Hard stop** if either condition is met:
     a. 3 consecutive failures on the same item.
     b. The same error message appears twice.
   - On hard stop: mark todo `[!]`, mark ledger `[!]`, surface full
     command + output history. Do not continue.

   Pass → proceed to Phase 2.

---

## Phase 2 — Record (only after Phase 1 passes)

1. Apply `dynamic-wiki-sync` to update the matching wiki function page.
   Mark **Wiki function page updated** in the progress ledger.

2. Apply `todo-status-sync` to mark the todo item `[x]` done.
   Mark all remaining checklist items in the progress ledger entry complete.

3. If either step fails:
   - Leave the todo item unchanged (do not mark `[x]`).
   - Surface the failure to the user with the specific error.

The skill does not terminate until Phase 2 completes or a genuine hard stop applies.

---

## Hard Rules

- Never stop at partial progress during implementation. Choose the best option
  and proceed.
- Never embed any version path as a constant; derive all active paths from
  `CLAUDE.md` §Source of Truth Pointer. Fixed project/deploy paths from
  `CLAUDE.md` §Fixed Project Paths are allowed.
- Never mark a todo item `[x]` before Phase 2 completes successfully.
- Never run this skill against `_cc` or `_cx` temporary planning files.
- Smoke retry cap: 3 attempts maximum, or 2 with the same error — whichever
  comes first.
