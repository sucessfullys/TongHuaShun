---
description: "Cancel the active project-local Ralph Loop"
allowed-tools: ["Bash(test -f .claude/ralph-loop.local.md:*)", "Bash(rm .claude/ralph-loop.local.md)", "Read(.claude/ralph-loop.local.md)"]
---

# Cancel Ralph Loop

1. Run: `test -f .claude/ralph-loop.local.md && echo EXISTS || echo NOT_FOUND`
2. If `NOT_FOUND`, report "No active Ralph loop found." and stop.
3. If `EXISTS`:
   - Read `.claude/ralph-loop.local.md` and extract the `iteration:` field from the YAML frontmatter.
   - Run: `rm .claude/ralph-loop.local.md`
   - Report: `Cancelled Ralph loop (was at iteration N)`.
