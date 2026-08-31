# System-Prompt-Retrieval-Agent — Repo Operating Contract

## Source of Truth Pointer

This is the one section skills read to resolve all active paths.
Change only this section when moving to a new version.

```text
Default active version:  V0.2.1

Active plan path:   knowledge/plans/{active-version}/plan.md
Active todo path:   knowledge/plans/{active-version}/todo.md
Active wiki root:   knowledge/wiki/{active-version}/

Progress ledger:    knowledge/plans/{active-version}/progress/implementation_progress.md
```

Skills read this section, substitute `{active-version}`, and derive all paths.
They never embed a fixed version string or file path.

If the user explicitly specifies a version, that override wins over this file.

---

## Version Discovery Rules

When deriving `{active-version}` programmatically:

1. User-specified version wins (e.g. "use V0.2").
2. Otherwise, scan `knowledge/plans/` and pick the highest-numbered version
   folder. Sort **numerically**, not lexicographically: `V0.10` > `V0.2`.
   Match case-insensitively: `v0.1` equals `V0.1`.
3. Fall back to `V0.1` if no version folder is found.
4. If the matching wiki version does not exist, copy the nearest lower wiki
   version as the base for the new version, then reset version-scoped progress
   and log files so the new version records only work done in that version.

---

## Pre-Coding Migration Check

Before running any implementation skill, confirm:

- `plan.md` exists at the active plan path.
- `todo.md` exists at the active todo path.
- The progress ledger exists; create it if missing (see §Progress Ledger).

If `plan.md` or `todo.md` is missing, finalize it from the temporary `_cc` or
`_cx` planning files first. Never run implementation skills against temporary files.

---

## Fixed Project Paths (never change)

```text
Source proposal:            project proposal.md
Local agent root:           /Volumes/970SSD/Code/Git/System-Prompt-Retrieval-Agent
Local remote-server source: /Volumes/970SSD/Code/Git/System-Prompt-Retrieval-Agent/Image-Generater-Remote
Remote project root:        /mnt/image-edit/datasets/xywang/code/System-Prompt-Retrieval-Agent
Remote app path:            /mnt/image-edit/datasets/xywang/code/System-Prompt-Retrieval-Agent/Image-Generater-Remote
Remote dataset root:        /mnt/image-edit/datasets/xywang/dataset
SSH alias:                  3h100  (host 10.217.219.2, user root, port 2891)
```

---

## Iron Law (Non-Negotiable)

1. **During implementation of a selected todo item, never stop at partial
   progress.** Choose the best option, state the choice and reasoning in one
   line, proceed. Only halt for: destructive actions, missing credentials/API
   keys, unavailable remote hardware, missing model assets requiring manual
   upload, or genuine ambiguity that cannot be resolved by reading
   plan/todo/wiki/source.

2. **Code boundaries.** Local project work belongs to the
   `System-Prompt-Retrieval-Agent/` root, excluding protected/generated paths:
   `.claude/`, `.git/`, `knowledge/`, `Image-Generater-Remote/`, caches, logs,
   runs, and outputs. Remote server code → `Image-Generater-Remote/` only.

3. **API rate limiting is ironclad.** Online OpenAI VLM calls ≤ 3 req/s,
   enforced in `rate_limiter.py`. No bypass for any reason.

4. **Never download model checkpoints.** Raise a clear error instructing the
   user to upload. No automatic downloads, no fallback download paths.

5. **Config/wiki coupling.** Any runtime config change must update both
   `config.yaml.example` and the `config_management` wiki page.

---

## Progress Ledger

The progress ledger lives at the active wiki progress path from §Source of
Truth Pointer. Create it when missing.

Each selected todo item gets one progress entry. Use the canonical template:

```text
.claude/skills/implementing-todo-item/reference/slot-checklist.md
```

Allowed status labels for todo items:

```text
[ ] not started
[~] in progress
[!] blocked
[x] done (all gates passed)
```

Only `[x]` after full verification, wiki sync, and todo sync. Never mark `[x]`
while any checklist item in the canonical template is incomplete.

---

## Parallel Programming Policy

Parallel work is allowed when a todo item can be decomposed into ≥2 independent
code subtasks with disjoint write scopes. The parent agent is the single
integration owner for all parallel sessions.

**Owner table:** Before spawning workers, the parent must create a write-scope
ownership table in the progress ledger: worker name, agent type, assigned files,
forbidden shared files.

**Parent-only operations (never delegated to workers):**

- Editing `plan.md`, `todo.md`, wiki pages, or progress ledger.
- Running dry-run review, real rsync, or remote venv mutations.
- Running the integration smoke check.
- Marking todo items `[~]`, `[!]`, or `[x]`.
- Calling `todo-status-sync` or `dynamic-wiki-sync`.

**Subagent limits:** Maximum 3 code-writing workers per task. Maximum 4
read-only agents (explorer/verifier) per session.

**Shared choke points:** Files that must not be included in any parallel write
scope: `config.py`, `schemas.py`, `config.yaml.example`, trace schema module,
wiki pages, `todo.md`, `plan.md`, progress ledger, `requirements.txt`,
remote deploy commands.

Apply `parallel-programming-coordinator` to decompose and coordinate.

---

## Read-Before-Coding Order

Before any task:

1. `project proposal.md`
2. Active `plan.md`
3. Active `todo.md`
4. Active wiki `README.md`
5. Relevant `functions/` pages for every area the task touches
6. `docs_update_policy.md`

For remote server tasks: also read these exact wiki pages:
`remote_image_service.md`, `config_management.md`, `trace_logging_and_copyback.md`.

For evaluator tasks: also read these exact wiki pages:
`openai_vlm_evaluator.md`, `benchmark_and_batch.md`.

---

## Anti-Patterns

- Auto-downloading model checkpoints.
- Hardcoded model paths, inference params, or credentials outside config.
- Hardcoding active plan/todo/wiki **version paths** in skills — these must always
  go through `CLAUDE.md` §Source of Truth Pointer and `{active-version}` substitution.
  (Fixed project/deploy paths listed in §Fixed Project Paths are always allowed.)
- Running implementation skills against `_cc` or `_cx` temporary planning files.
- Calling OpenAI API without routing through `rate_limiter.py`.
- Exceeding 3 online VLM req/s for any reason.
- Code written in protected/generated paths or mixed across local and remote
  server ownership boundaries.
- Using `root@10.217.219.2` directly in rsync or ssh — always use the `3h100` alias.
- Deploying remote server code outside `/mnt/image-edit/datasets/xywang/code/System-Prompt-Retrieval-Agent/Image-Generater-Remote`.
- Running rsync without a dry-run review first.
- Committing `config.yaml` or raw API keys.
- Marking a todo item `[x]` before the progress ledger checklist is complete.
- Merging wiki pages — one page per function area, always.
- Inserting new todo items into the middle of planned stages (append under a
  follow-ups section only).
- Editing canonical plan files (they are historical records; only `todo.md`
  may be updated, via `todo-status-sync`).
- Spawning write-capable subagents without a write-scope ownership table.
- Allowing subagents to edit `todo.md`, `plan.md`, wiki pages, or progress ledger.
- Allowing subagents to run dry-run review, real rsync, or remote venv mutations.
- Assigning overlapping write scopes to two or more workers.

---

## Skills Index

- `reading-current-plan/` — resolves active paths from §Source of Truth Pointer; reads before any task
- `implementing-todo-item/` — ironclad two-phase implementation loop for one todo item
- `dynamic-wiki-sync/` — updates the correct multi-file wiki page after code changes
- `todo-status-sync/` — marks items done after full verification; only path for editing `todo.md`
- `remote-image-service-deploy/` — dry-run then real rsync; remote venv setup
- `skill-authoring-project/` — creates or updates project-local skills
- `parallel-programming-coordinator/` — decomposes todo items into safe parallel slots; coordinates workers and integrates output

---

## Bounded-Retry Policy (replaces former Ralph Loop)

V0.2.1 does not use Ralph Loop. Iterative test-greening is performed by the
parent agent under a bounded-retry budget with explicit machine-checkable
completion criteria.

### Bounded-retry contract

For any failing gate the parent may retry **at most twice**:

1. **First attempt** — fix-forward in-place by the parent (parent edits the
   minimum set of files that own the failing assertion).
2. **Second attempt** — spawn one focused worker subagent (`local-agent-worker`
   or `remote-server-worker` depending on scope) with a tight write-scope
   bounded by the assertion's owning module; parent integrates and re-runs
   the gate.

If the second attempt also fails, the parent **halts** and reports full
failure details. There is no third attempt; halting is preferred over
unbounded iteration.

Every retry must declare a machine-checkable completion criterion before
starting (e.g. "pytest tests/test_schemas.py exits 0", "ranking.json
contains a row with `pair_overall ≥ 0.70`"). Vague goals such as "DONE"
are forbidden.

### What bounded-retry is allowed for

- Tightening a single V0.2.1 module until `pytest tests/` is green
  (e.g. cell-keyed barrier in `remote/barrier.py`, the v0.2.1
  `long_memory.csv` flattener, or `LocalApiEvaluator.evaluate_many`).
- Driving a single prompt-pair against the mock controller until
  `pair_overall ≥ 0.80` with the language-balanced base mean.
- Iterative scaffolding inside one writer-scope (W1/W2/W3) post-merge.

### What bounded-retry is NOT allowed for

- Real-remote pilot runs (`limit ≥ 30`, real OpenAI billing, GPU pilot,
  rsync/ssh, remote venv mutations) — these remain parent-only per §Iron
  Law and §Parallel Programming Policy and require explicit user approval
  per turn. The pilot retry budget is also two attempts but each pass is
  driven manually by the parent.
- Mutations to `plan.md`, `todo.md`, wiki pages, the progress ledger,
  `config.yaml.example`, `pyproject.toml`, `requirements.txt`, or the
  remote controller. These are parent-only and not retry-driven.
- Tasks without a concrete, machine-checkable completion criterion.

All docs and skills are written in English.

### Kill NoGPUAlarmNew.py

`NoGPUAlarmNew.py` is a remote GPU-preservation watchdog on `3h100` that
holds a small VRAM reservation when the host is idle. It self-exits when
real model loading starts. This is a parent-only fallback, not a routine
deploy step.

**Run only when all true:** real-remote operation approved this turn;
loading/startup is observed blocked by VRAM pressure; the watchdog is
still running and is the plausible holder (confirm via `pgrep` +
`nvidia-smi`).

**Do not run** if VRAM is held by a different PID (fix that PID instead),
if `pgrep` is empty (already gone), or preemptively before loading starts.

**Procedure** — stop as soon as VRAM clears:

```bash
# 1. Inspect (always run first)
ssh 3h100 'pgrep -af "NoGPUAlarmNew.py" || true; nvidia-smi'

# 2. Graceful (SIGTERM)
ssh 3h100 'pkill -f "NoGPUAlarmNew.py"; sleep 3; pgrep -af "NoGPUAlarmNew.py" || true'

# 3. Forced (SIGKILL — only if step 2 left the process running)
ssh 3h100 'pkill -9 -f "NoGPUAlarmNew.py"; sleep 1; pgrep -af "NoGPUAlarmNew.py" || true'

# 4. Verify VRAM freed
ssh 3h100 'nvidia-smi'
```

If step 3 fails, or post-kill VRAM is still held, halt and report — do
not loop. Always use the `3h100` alias; never `root@10.217.219.2`.

**Restart GPU_OCU before the final report** of an approved real-remote
session, unless already healthy:

```bash
ssh 3h100 'cd /mnt/image-edit/datasets/xywang/code/GPU_OCU && bash start.sh'
ssh 3h100 'pgrep -af "NoGPUAlarmNew.py" || true; nvidia-smi'
```

**Audit** — record in the active progress ledger entry: step reached
(2, 3, or aborted at 1), pre-kill and post-kill `nvidia-smi`, and whether
GPU_OCU was restarted.
