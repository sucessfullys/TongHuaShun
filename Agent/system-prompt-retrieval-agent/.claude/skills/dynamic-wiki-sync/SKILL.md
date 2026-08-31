---
name: dynamic-wiki-sync
description: >
  Updates the correct wiki function page after any implemented change to CLI,
  config, FastAPI endpoints, OpenAI API behavior, candidate/routing logic,
  refinement policy, trace schema, copy-back, or remote deploy. Resolves the
  active wiki path from CLAUDE.md at runtime; never embeds a version string.
---

## Purpose

Keep the dynamic multi-file wiki at `knowledge/wiki/{active-version}/` accurate
and up to date after each implemented change. One page per function area — never
merge pages. The wiki records only confirmed shipped behavior for the active
version.

## When to Invoke

After any implemented change to:

- CLI commands or arguments
- `config.yaml.example` or config schema
- FastAPI endpoint signatures or behavior
- OpenAI VLM API calls or Batch API usage
- Candidate generation or routing logic
- Refinement policy or recipe memory
- Trace schema or trace logging
- Image copy-back paths or behavior
- Remote deploy steps or rsync invocation

## Steps

1. Read `CLAUDE.md` §Source of Truth Pointer. Derive the active wiki root path
   by substituting `{active-version}`. Never embed the version string as a constant.

2. Check whether the active wiki version folder exists:
   - **Exists:** use it.
   - **Missing, but a lower version exists:** copy the entire nearest
     lower-numbered wiki version folder to the active wiki root as the base.
     Reset version-scoped progress and log files in the new version so they do
     not carry old-version entries forward. Record a bootstrap note in the new
     progress ledger using real values:
     ```text
     Wiki version V0.2 missing; copied V0.1 as the base and reset V0.2 progress/log entries.
     ```
   - **No wiki version exists at all:** halt immediately. Instruct the user:
     "No previous wiki version found. Create `knowledge/wiki/{active-version}/`
     before starting implementation." Do not continue.

3. Determine which wiki function page to update:
   - `functions/config_management.md`
   - `functions/remote_image_service.md`
   - `functions/openai_vlm_evaluator.md`
   - `functions/candidate_generation.md`
   - `functions/refinement_policy.md`
   - `functions/trace_logging_and_copyback.md`
   - `functions/benchmark_and_batch.md`

   If the change spans a new function area not covered by existing pages, create
   a new page. Never merge two function areas into one page.

4. Update the target page. Required header block immediately below the H1 title:

   ```text
   Status: [planned | in-progress | implemented]
   Last updated: {active-version}, after <short change note>
   Primary implementation files: <repo-relative paths>
   Related config keys: <comma-separated keys or "none">
   ```

   Use a version + change-note in `Last updated` (not a bare date). Change notes
   are self-describing and do not go stale.

5. Write only confirmed shipped behavior for the active version. Entries must
   be concise and version-scoped: document what was done in this version, not
   what changed from previous versions. Remove stale details — do not leave
   outdated descriptions alongside new ones.

6. If a new wiki function page was created in step 3, also update the wiki
   `README.md` index to include an entry for the new page.

7. Mark **Wiki function page updated** in the progress ledger entry for the
   current todo item.

## Output Contract

- Path of the updated wiki page (relative to wiki root).
- Summary of what changed (one paragraph or bullet list).
- Any bootstrap notes recorded in the progress ledger.
- Confirmation that wiki README was updated if a new page was created.

## Hard Rules

- Never merge wiki pages.
- Never embed a version string as a constant; always derive from `CLAUDE.md`.
- Never edit `plan.md` or `todo.md` from this skill.
- Use version + change-note for `Last updated`; never a bare date.
- Bootstrap notes must use real version values — never literal `{active-version}` text.
- Halt and instruct the user when no wiki version exists at all.
- Never silently update a lower wiki version when the active version is missing.
- New wiki function pages must be registered in the wiki `README.md` index.
- Wiki updates are parent-only. Subagents may return proposed wiki notes in
  their output, but must not write to wiki pages directly. The parent agent
  applies wiki-sync after integration.
