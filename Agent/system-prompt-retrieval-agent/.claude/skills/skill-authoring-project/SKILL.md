---
name: skill-authoring-project
description: >
  Creates or updates project-local skills under .claude/skills/ for the
  System-Prompt-Retrieval-Agent project. Enforces naming conventions, structure rules, and
  project-specific requirements including version-agnostic path resolution.
---

## Purpose

Add new skills or update existing ones in `.claude/skills/` so they integrate
cleanly with the project's CLAUDE.md, progress ledger, and ironclad loop.

## Naming

- Gerund form, lowercase-hyphenated, ≤64 characters.
  Examples: `reading-current-plan`, `implementing-todo-item`.
- One directory per skill: `.claude/skills/<skill-name>/SKILL.md`.
- Long reference material goes in `<skill-name>/reference/`; forward-slash paths only.

## SKILL.md Structure

Required YAML frontmatter:

```yaml
---
name: <gerund-form-name>
description: >
  <Third-person, what the skill does + when to invoke it. ≤1024 characters.
   No dates. No version strings.>
---
```

Required sections (body ≤120 lines; overflow to `reference/`):

1. **Purpose** — one short paragraph on why the skill exists.
2. **Steps** or **When to Invoke** — numbered steps or trigger conditions.
3. **Output Contract** — what the skill reports when complete.
4. **Hard Rules** — non-negotiable constraints.

## Steps

1. Read `CLAUDE.md` §Source of Truth Pointer to confirm the active version and
   paths. Use those paths in the skill's documentation where applicable.
   Do not embed any version string as a constant.

2. Draft the SKILL.md using the structure above.

3. Apply project-specific requirements (see below) to the draft.

4. Write the file at `.claude/skills/<skill-name>/SKILL.md`.

4b. If the skill enables subagent spawning, also create the corresponding
    `.claude/agents/<agent-name>.md` file. Agents live under `.claude/agents/`,
    not under `.claude/skills/`.

5. If reference material is needed, write it to
   `.claude/skills/<skill-name>/reference/<file>.md`.

6. Add the skill to `CLAUDE.md` §Skills Index:
   `- \`<skill-name>/\` — <one-line description of purpose>`

## Project-Specific Requirements

Every new skill for this project must satisfy:

- **Version-agnostic (two-category rule):**
  - **FORBIDDEN:** Active plan/todo/wiki version paths. These must always go
    through `CLAUDE.md` §Source of Truth Pointer and `{active-version}`
    substitution. Example of what is forbidden: `knowledge/plans/V0.1/todo.md`.
  - **ALWAYS ALLOWED:** Every path listed in `CLAUDE.md` §Fixed Project Paths
    (local code roots, remote paths, SSH alias). These are fixed by definition
    and must not be abstracted further.

- **Evaluator skills:** If the skill touches OpenAI API calls, cite the ≤3 req/s
  rate-limit rule from `CLAUDE.md` §Iron Law and confirm routing through
  `rate_limiter.py`.

- **Remote server skills:** If the skill touches the remote cluster:
  - Always use the `3h100` SSH alias; never the raw IP.
  - Include a dry-run rsync step before real rsync.
  - Cite the no-model-download rule from `CLAUDE.md` §Iron Law.
  - Deploy target: `/mnt/image-edit/datasets/xywang/code/System-Prompt-Retrieval-Agent/Image-Generater-Remote/`.

- **Todo-touching skills:** If the skill updates `todo.md`, it must also update
  the progress ledger accordingly, and it must never operate on `_cc` or `_cx` files.

- **Wiki-touching skills:** If the skill updates wiki pages, it must use the
  version + change-note header format (not bare dates) and must never merge pages.

- **Subagent-enabling skills:** If the skill spawns worker subagents, it must
  define in its body: ownership table format, forbidden shared files,
  parent-only operations (todo.md, wiki, deploy, smoke check), and integration
  gates before handoff.

- **Agent file requirements:** Agent `.md` files under `.claude/agents/` must
  embed key project rules in their body (CLAUDE.md is not inherited by subagents).
  List only needed skills in `skills:` frontmatter. Never include `Agent` in
  `tools:` for any subagent (prevents nesting).

## Output Contract

- Path of the new or updated `SKILL.md`.
- Path of any `reference/` files created.
- Confirmation that `CLAUDE.md` §Skills Index was updated.
- Any project-specific requirements that were applied.

## Hard Rules

- Never embed active plan/todo/wiki version paths in any skill (two-category rule above).
- Never write a skill body longer than 120 lines; move overflow to `reference/`.
- Description field: ≤1024 characters, third-person, no dates.
- After creating a skill, always add it to `CLAUDE.md` §Skills Index.
- Skills and agents are separate: skills go in `.claude/skills/<name>/SKILL.md`;
  agent definitions go in `.claude/agents/<name>.md`. Never mix them.
- Agent files must not include `Agent` in their `tools:` list.
