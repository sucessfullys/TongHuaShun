# MCP servers for ERA

ERA's Stage 1 literature research searches **arXiv** and **GitHub** (and,
optionally, **Google Scholar**) through MCP servers. `WebSearch` / `WebFetch` are
built into Claude Code and need no setup — they are the fallback when no MCP
server resolves.

## How ERA registers MCP — automatic, per launch directory

Claude Code loads MCP servers from a **`.mcp.json` in the directory it is
launched in** — and nowhere else (no parent-directory search). ERA's pipeline
runs Claude Code from inside `workspaces/<project>/`, so a server registered
elsewhere (e.g. `claude mcp add --scope local` in the repo root) never reaches
the workspace. That is exactly why an earlier Stage 1 run found *"the
arxiv-mcp-server … not registered"* and degraded to web search.

ERA fixes this by **shipping the registration itself**:

1. **The arXiv MCP server is a dependency.** `pip install -e .` installs
   `arxiv-mcp-server` into the repo `.venv` — no `uv`/`uvx`/`pipx` needed.
2. **`/era:init` scaffolds `.mcp.json` into every workspace**, pointing at the
   absolute `.venv/bin/arxiv-mcp-server` path. It is written *before* you launch
   Claude Code in the workspace, so the server loads on session start.
3. **`.claude/settings.json` carries `enableAllProjectMcpServers: true`**, so
   the autonomous loop never stops on an MCP trust prompt.

`era/orchestration/mcp.py` is the single source of truth — its `MCP_SERVERS`
registry is rendered into every `.mcp.json`.

### Setup — what you actually run

```bash
# 1. Installs ERA + the arXiv MCP server into .venv
python3 -m venv .venv && .venv/bin/pip install -e .

# 2. Register MCP for the repo root (one-time; /era:init handles each
#    workspace automatically). Re-run it if the repo moves or .venv is rebuilt.
.venv/bin/python3 -m era.cli write-mcp-config
```

That's it — every workspace `/era:init` creates is MCP-ready.

## Why server names matter

ERA's literature skill (`plugin/skills/era-literature/SKILL.md`) and its
`literature-scout` agent (`plugin/agents/literature-scout.md`) reference MCP
tools by their fully-qualified name `mcp__<server-name>__<tool-name>` — e.g.
`mcp__arxiv-mcp-server__search_papers`. **The server name in `.mcp.json` must
match the `<server-name>` in those references**, or the tools will not resolve.
ERA's registry fixes `arxiv-mcp-server` for that reason — do not rename it.

## arXiv — shipped, on by default

Server name: **`arxiv-mcp-server`**. Tools used: `search_papers`,
`download_paper`, `read_paper`, `list_papers`. Installed and registered by the
setup above — no manual step.

## GitHub — repository & code search

Server name: **`github`**. Tools used: `search_repositories`, `search_code`,
`search_issues`, `get_file_contents`. ERA uses it in Stage 1's *open-source
implementations* direction — locating candidate evaluator repos and reading
their code / license precisely, instead of guessing from web results.

ERA registers GitHub's **remote hosted** MCP endpoint
(`https://api.githubcopilot.com/mcp/`, HTTP transport) — no Docker, Go, or local
binary needed. It is in the `MCP_SERVERS` registry, so it is scaffolded into
every `.mcp.json` automatically.

**It needs a GitHub Personal Access Token.** The `.mcp.json` entry sends
`Authorization: Bearer ${GITHUB_PERSONAL_ACCESS_TOKEN}` — Claude Code expands
`${...}` from the **process environment** at launch (it does not read `.env`
itself). So:

1. Put the token in `.env`: `GITHUB_PERSONAL_ACCESS_TOKEN=ghp_…` — a classic or
   fine-grained PAT; for public-repo search no extra scopes are needed.
2. Export it into the shell that launches Claude Code, e.g.
   `set -a; source .env; set +a` before running `claude …`.

A **read-only** token is recommended — ERA only ever calls GitHub *search* tools
(the scout/skill tool allowlists permit nothing else, and the `.mcp.json` entry
also requests the read-only, search-relevant toolsets).

Without the token the `github` server fails to connect; Stage 1 scouts then fall
back to `WebSearch` for repositories — the survey is still produced.

## Google Scholar — optional, off by default

Server name: **`google-scholar`**. Tools used:
`search_google_scholar_key_words`, `search_google_scholar_advanced`. It adds
citation-ranked coverage; the literature stage works fine without it.

To add it, register the entry in `era/orchestration/mcp.py`'s `MCP_SERVERS`
registry — it then flows into every newly-scaffolded `.mcp.json`. For a one-off,
add it by hand:

```bash
claude mcp add --scope project google-scholar -- npx -y google-scholar-mcp
```

## Verify

```bash
claude mcp list          # run from the repo root, or from any workspace
```

`arxiv-mcp-server` should show **✓ Connected** — and `github` too, once
`GITHUB_PERSONAL_ACCESS_TOKEN` is exported. Inside a Claude Code session the
tools then resolve as `mcp__arxiv-mcp-server__*` and `mcp__github__*`.

## No MCP? Graceful degradation

If no MCP server resolves, ERA's `literature-scout` sub-agents fall back to
`WebSearch` / `WebFetch` only. The survey is still produced — it just leans on
the open web instead of arXiv's structured search.

## Editing `.mcp.json` by hand

`ensure_mcp_config` never overwrites an existing `.mcp.json`, so a workspace
`.mcp.json` you edit by hand is preserved. The repo-root `.mcp.json` is
git-ignored (it carries a machine-specific absolute path) — regenerate it any
time with `era.cli write-mcp-config`.
