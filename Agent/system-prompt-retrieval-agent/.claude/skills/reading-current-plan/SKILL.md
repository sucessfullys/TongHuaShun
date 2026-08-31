---
name: reading-current-plan
description: >
  Resolves the active plan, todo, wiki, and progress ledger paths from CLAUDE.md
  §Source of Truth Pointer, then reads all context required before any task.
  Invoke before every implementation, wiki-sync, or deploy operation.
---

## Purpose

Derive active paths at runtime from `CLAUDE.md`. Never embed a fixed version
string or file path. If the user specifies a version explicitly, that override
wins over `CLAUDE.md`.

## Steps

1. Read `CLAUDE.md` §Source of Truth Pointer.
   - Extract `Default active version` and path patterns containing `{active-version}`.
   - Apply §Version Discovery Rules:
     - User-specified version wins.
     - Otherwise, use Glob pattern `knowledge/plans/[Vv]*/` to list version folders.
       Parse the numeric suffix after `V`/`v` (e.g., `0.1`, `0.10`, `2.0`).
       Sort by numeric tuple: `(0, 10) > (0, 2)`. Discard non-parseable folders.
       Pick the highest-numbered valid folder.
    - Fall back to `V0.1` if no valid folder is found.
   - Substitute `{active-version}` in every path pattern. These are the active paths.

2. Confirm `plan.md` exists at the resolved plan path.
   - If missing: surface the §Pre-Coding Migration Check step — finalize from
     `_cc` or `_cx` before proceeding. **Stop here. Do not continue.**

3. Confirm `todo.md` exists at the resolved todo path.
   - If missing: same migration step. **Stop here. Do not continue.**

4. Confirm the progress ledger exists at the resolved ledger path.
   - If missing **and this skill is called from `implementing-todo-item`**: create
     it now with the heading:
     ```markdown
     # Implementation Progress

     <!-- Entries added by implementing-todo-item. Use the canonical template at:
          .claude/skills/implementing-todo-item/reference/slot-checklist.md -->
     ```
   - If missing **and called from any other skill** (wiki-sync, deploy): do NOT
     create it. Surface the gap and continue only if the caller can proceed without it.

5. Read `project proposal.md` (root of the working directory).

6. Read the active `plan.md` — focus on sections relevant to the current task.

7. Read the active `todo.md` — locate the specific item(s) in scope.

8. Read the active wiki `README.md`.

9. Read relevant `functions/` wiki pages for every area the task touches.

10. Read `docs_update_policy.md` from the active wiki root.
    Skip and warn if missing — do not block implementation.

11. **Remote server tasks only:** also read these exact wiki pages:
    `remote_image_service.md`, `config_management.md`, `trace_logging_and_copyback.md`.

12. **Evaluator tasks only:** also read these exact wiki pages:
    `openai_vlm_evaluator.md`, `benchmark_and_batch.md`.

## Output Contract

Report:

- Active version used and how it was resolved (user-specified / numeric-highest / fallback).
- All resolved paths (plan, todo, wiki root, progress ledger).
- Matched todo item(s) with current status.
- Codebase scope (local agent, remote server, or both).
- Relevant wiki pages read.
- Any constraints (rate limit, model-path, config/wiki coupling) that apply.
- Blocking questions, if any (path missing, credentials absent, etc.).
- Warning if `docs_update_policy.md` was absent.
- Parallelization candidates: list of todo subtasks with non-overlapping file
  scopes, if ≥2 exist. "None" if single-agent is required.
- Unsafe shared files: files in scope that would block parallelization
  (`config.py`, `schemas.py`, `config.yaml.example`, trace schema, wiki, todo,
  plan, `requirements.txt`, deploy commands).

## Hard Rules

- Never embed any version string (`V0.1`, `V0.2`, etc.) or active path as a constant.
  Fixed project/deploy paths from `CLAUDE.md` §Fixed Project Paths are allowed.
- Never edit `plan.md` or `todo.md` from this skill.
- If the matching wiki version does not exist, copy the nearest lower wiki
  version as the base, reset version-scoped progress/log files, and record a
  bootstrap note in the new progress ledger before continuing.
- If no wiki version exists at all, halt and instruct the user to create the
  wiki version directory before starting implementation.
- Ledger creation is only allowed when called from `implementing-todo-item`.
