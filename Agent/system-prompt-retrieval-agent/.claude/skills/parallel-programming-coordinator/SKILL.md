---
name: parallel-programming-coordinator
description: >
  Decomposes a selected todo item into safe parallel implementation slots,
  assigns disjoint write scopes to worker subagents, collects and integrates
  their outputs, and prepares for the parent smoke check. Invoked from
  implementing-todo-item when independent subtasks are detected. The parent
  agent retains full ownership of plan.md, todo.md, wiki, progress ledger,
  remote deploy, smoke check, and final todo status.
---

## Purpose

Accelerate implementation of todo items with independent code subtasks by
running up to 3 local or remote code-writing workers in parallel, while
preserving all state-consistency guarantees of the ironclad loop.

## When to Invoke

Invoked by `implementing-todo-item` Phase 1 when ALL of the following hold:
- The todo item has ≥2 independent subtasks with non-overlapping file scopes.
- No shared choke point is in scope (see below).
- At least one worker type matches the task (local, remote, or both).

## Shared Choke Points — Never Parallelize Over These

A subtask that touches any of the following blocks parallelization for that slot:
`config.py`, `schemas.py`, `config.yaml.example`, trace schema module, any wiki
page, `todo.md`, `plan.md`, progress ledger, `requirements.txt`, remote deploy.
These files must be handled by the parent agent in single-agent flow.

## Steps

1. **Analyze** the todo item. List all source files each subtask would write.

2. **Partition** subtasks: group into safe parallel slots (disjoint write
   sets) and choke-point slots (must be sequential/parent-handled).

3. **If fewer than 2 safe parallel slots:** return `"single-agent"`. Caller
   continues with the standard single-agent flow.

4. **Create the ownership table** (record in progress ledger):

   | Worker | Agent | Assigned files | Forbidden shared files |
   |---|---|---|---|
   | W1 | local-agent-worker | `src/.../module_a.py` | `config.py`, `schemas.py` |
   | W2 | local-agent-worker | `src/.../module_b.py` | `config.py`, `schemas.py` |

   Maximum: 3 code-writing workers (local or remote). 4 read-only agents total.

5. **Spawn workers** via Agent tool in a single message (parallel).
   Each worker prompt must include: write-scope table, forbidden files,
   relevant plan/todo context, acceptance test to run locally.

6. **Collect outputs.** For each worker: verify no files outside write scope
   were modified. If any overlap found: flag as conflict, do not integrate.

7. **Integrate** worker changes into the working tree. Resolve trivial
   conflicts (different files) automatically; surface non-trivial conflicts
   to the user before proceeding.

8. **Return to `implementing-todo-item`** Phase 1 step 9 (smoke check).
   The parent runs the integrated smoke check — not the workers.

## Output Contract

- Ownership table (worker → assigned files → forbidden files).
- List of workers spawned and their outputs.
- Integration status (clean / conflicts found / partial).
- Any blocked choke-point subtasks for sequential parent handling.
- Handoff statement: "Integration complete — ready for parent smoke check."

## Hard Rules

- Parent owns: plan.md, todo.md, wiki, progress ledger, remote deploy, smoke check.
- No two workers may write to the same file. No exceptions.
- Workers do not mark todo items done. Only the parent calls todo-status-sync.
- Workers do not edit wiki pages. Parent calls dynamic-wiki-sync after integration.
- Maximum 3 code-writing workers; 4 read-only agents total.
- Remote deploy (rsync, SSH venv) is always a parent foreground action.
- Never embed active plan/todo/wiki version paths as constants.
