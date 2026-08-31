---
description: "Start a project-local Ralph Loop in this session"
argument-hint: "PROMPT [--max-iterations N] [--completion-promise TEXT]"
allowed-tools: ["Bash(${CLAUDE_PROJECT_DIR}/.claude/scripts/setup-ralph-loop.sh:*)"]
---

# Ralph Loop (project-local)

Initialise a Ralph Loop using the project-vendored setup script. The Stop hook
registered in `.claude/settings.json` will keep feeding this same prompt back
each time you try to exit, until either:

- you output `<promise>YOUR_PROMISE</promise>` matching the configured promise, or
- `--max-iterations N` is reached.

Run setup:

```!
"${CLAUDE_PROJECT_DIR}/.claude/scripts/setup-ralph-loop.sh" $ARGUMENTS
```

Now please work on the task described above. When you try to end your turn the
Ralph Loop will fire and the SAME PROMPT will be sent back to you for the next
iteration. Use the file system, git history, and `.claude/ralph-loop.local.md`
to see your prior work.

CRITICAL RULE: only emit the completion promise when its statement is
unequivocally TRUE. Do not emit a false promise to escape the loop.
