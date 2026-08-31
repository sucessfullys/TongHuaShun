# Per-Item Implementation Checklist

Copy this checklist into the progress ledger entry for each selected todo item.
Replace `<item id>` and `<description>` with the actual todo item identifier and
its one-line description.

---

## Item: \<item id\> — \<description\>

### Ground

- [ ] Active version resolved from `CLAUDE.md` §Source of Truth Pointer
- [ ] `plan.md` confirmed at resolved path
- [ ] `todo.md` confirmed at resolved path
- [ ] Progress ledger confirmed (created if missing)
- [ ] Todo item marked `[~]` in progress

### Parallel Ownership *(only when subagents were used)*

- [ ] Write-scope ownership table created and recorded in ledger
- [ ] Shared choke points confirmed absent from parallel slots
- [ ] Forbidden shared files listed (plan.md, todo.md, wiki, ledger, deploy)
- [ ] Workers spawned: `<list worker names and agent types>`
- [ ] Worker outputs received and reviewed (no out-of-scope edits)
- [ ] Parent integration complete — no unresolved conflicts
- [ ] Integrated smoke check passed (parent-run, not worker-run)

### Constraint Checks

- [ ] **Rate-limit routing** — OpenAI API calls route through `rate_limiter.py`;
      ≤3 req/s enforced *(evaluator items only)*
- [ ] **Model-path error** — missing checkpoint raises clear error; no download
      triggered *(remote server items only)*
- [ ] **Config/wiki coupling** — new params written to `config.yaml.example` AND
      `config_management` wiki page simultaneously *(config-touching items only)*

### Implement

- [ ] Code written in correct scope
      (`System-Prompt-Retrieval-Agent/` local files, excluding protected/generated
      paths; `Image-Generater-Remote/` for remote)
- [ ] No protected/generated path edits; no cross-boundary mixing

### Remote Deploy *(remote server items only)*

- [ ] `mkdir -p` run on remote path before rsync
- [ ] Dry-run rsync output reviewed — no unexpected or secret files
- [ ] User confirmed dry-run; real rsync completed via `3h100` alias
- [ ] Remote venv Python version verified (≥3.10); venv installed/updated
- [ ] Remote file presence confirmed with `ssh 3h100 ls <remote-path>`
- [ ] Ledger: "Remote deploy completed" ✓

### Smoke Check

- [ ] Smoke-check command selected (Acceptance line / existing test / fallback import)
- [ ] Smoke-check command captured: `<command>`
- [ ] Smoke-check output captured (pass/fail noted)
- [ ] **Exit gate:** pass → Phase 2; fail → fix + retry; hard stop after 3 attempts
      or 2 identical errors → mark `[!]`

### Phase 2 — Record

- [ ] `dynamic-wiki-sync` applied; matching wiki function page updated
- [ ] Wiki README updated if a new function page was created
- [ ] `todo-status-sync` applied; todo item marked `[x]`
- [ ] Progress ledger entry marked complete (all boxes above checked)

---

## Status Labels

| Label | Meaning |
| --- | --- |
| `[ ]` | Not started |
| `[~]` | In progress — Ground section boxes checked, code being written |
| `[!]` | Blocked — smoke hard-stopped or genuine halt condition reached |
| `[x]` | Done — all checklist items complete, verification passed |

Only promote to `[x]` after every checkbox in this template is checked.
