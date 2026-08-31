---
name: era-literature
description: ERA Stage 1 — survey the evaluation methods, metrics, benchmarks, and human-correlation studies for a project's AIGC task by fanning out parallel literature-scout sub-agents over arXiv + GitHub (MCP) + web search, then writing one structured survey to research/literature.md.
allowed-tools: Read, Write, Glob, Grep, Bash, Task, WebSearch, WebFetch, mcp__arxiv-mcp-server__search_papers, mcp__arxiv-mcp-server__download_paper, mcp__arxiv-mcp-server__read_paper, mcp__arxiv-mcp-server__list_papers, mcp__google-scholar__search_google_scholar_key_words, mcp__google-scholar__search_google_scholar_advanced, mcp__github__search_repositories, mcp__github__search_code, mcp__github__search_issues, mcp__github__get_file_contents
---

# ERA Literature Research (Stage 1)

Survey the **evaluation methods, metrics, benchmarks, and human-correlation
studies** relevant to this ERA project's AIGC task, and write a single
structured survey into the workspace for the later pipeline stages to consume.

The workspace path is passed as `$ARGUMENTS`.

## How to run

The **ERA repo root** is the parent of `${CLAUDE_PLUGIN_ROOT}` — it contains
`era/`, `plugin/`, `docs/`, and `workspaces/`.

1. Read the behavioral prompt `docs/prompts/stage1_research.md` (relative to the
   ERA repo root).
2. Follow its steps **exactly**, for the workspace given as `$ARGUMENTS`.

You are the **orchestrator**: the behavioral prompt has you decompose the survey
into search directions, fan them out to parallel `literature-scout` sub-agents
(via the Task tool), merge their digests, and write
`<workspace>/research/literature.md`.

**Resilience:** if a scout sub-agent fails or returns nothing, continue with the
others and note the gap — never block on one scout. If sub-agents are
unavailable entirely, fall back to searching the directions yourself,
sequentially. Retry a transient tool failure once before treating it as real.

**Never ask the operator anything** — this skill runs inside ERA's autonomous
loop. Resolve every ambiguity from `spec.md` / `config.yaml`, decide, and record
the decision in `<workspace>/logs/iterations/`.
