"""``.mcp.json`` generation — register ERA's MCP servers per launch directory.

Claude Code loads MCP servers from a ``.mcp.json`` in **the directory it is
launched in**, and nowhere else (no parent-directory search). ERA's Stage 1
literature research runs inside ``workspaces/<project>/`` — a different
directory from where an operator would set up MCP — so a server registered with
``claude mcp add --scope local`` in the repo root never reaches the workspace.

ERA therefore scaffolds a ``.mcp.json`` into the repo root **and every
workspace**, so the arXiv + GitHub MCP servers are always available wherever
``/era:start`` runs.
``.mcp.json`` servers still need approval; ERA's ``.claude/settings.json``
carries ``enableAllProjectMcpServers`` (see ``ralph.era_claude_settings``) so
the autonomous loop never stops on a trust prompt.

``MCP_SERVERS`` is the single source of truth: add a future server here and it
flows into every newly-scaffolded ``.mcp.json``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..paths import repo_root

# ERA's MCP server registry — rendered verbatim into ``.mcp.json``. To add a
# server later (Google Scholar, an SSH server for remote GPUs, …), append an
# entry here; ``mcp_config()`` picks it up for every directory ERA scaffolds.
#
# A stdio server's ``command`` may contain the placeholder ``{venv_bin}`` —
# replaced with the ERA repo's absolute ``.venv/bin`` directory, so a
# venv-installed server launches correctly regardless of the working directory.
# An HTTP server (``"type": "http"``) instead carries a ``url`` (+ optional
# ``headers``); ``${VAR}`` references in it are left literal for Claude Code to
# expand from the environment.
MCP_SERVERS: dict[str, dict] = {
    # arXiv paper search — ERA Stage 1 literature research. The server name
    # MUST stay ``arxiv-mcp-server``: the literature skill and scout agent
    # reference its tools as ``mcp__arxiv-mcp-server__*``. The console script
    # is installed by the ``arxiv-mcp-server`` dependency (see pyproject.toml);
    # its shebang pins the venv interpreter.
    "arxiv-mcp-server": {
        "type": "stdio",
        "command": "{venv_bin}/arxiv-mcp-server",
        "args": [],
        "env": {},
    },
    # GitHub repository & code search — ERA Stage 1, the "open-source
    # implementations" direction. Registered as GitHub's remote hosted MCP
    # endpoint (HTTP); no local binary or Docker needed. Authenticates with a
    # GitHub PAT — ``${GITHUB_PERSONAL_ACCESS_TOKEN}`` is left literal for
    # Claude Code to expand from the environment at launch. Without the token
    # the server fails to connect and scouts fall back to web search. The
    # X-MCP-* headers ask the remote for only search-relevant toolsets,
    # read-only. The server name MUST stay ``github``: scout/skill prompts
    # reference its tools as ``mcp__github__*``.
    "github": {
        "type": "http",
        "url": "https://api.githubcopilot.com/mcp/",
        "headers": {
            "Authorization": "Bearer ${GITHUB_PERSONAL_ACCESS_TOKEN}",
            "X-MCP-Toolsets": "repos,issues",
            "X-MCP-Readonly": "true",
        },
    },
    # OpenAI Codex — the Stage 6 independent code reviewer (era-codex-reviewer).
    # Registered via the Codex CLI's MCP mode. It is **opt-in**: Stage 6 calls
    # the reviewer only when ``config.yaml``'s ``experiment.codex_reviewer`` is
    # true (default false). When the ``codex`` CLI is not installed the server
    # simply reports unavailable — harmless, since the default config never
    # dispatches the reviewer. The server name MUST stay ``codex``: the
    # era-codex-reviewer agent references its tool as ``mcp__codex__codex``.
    "codex": {
        "type": "stdio",
        "command": "codex",
        "args": ["mcp"],
        "env": {},
    },
}


def _venv_bin() -> str:
    """Absolute path to the ERA repo virtualenv's ``bin`` directory."""
    return str(repo_root() / ".venv" / "bin")


def mcp_config() -> dict:
    """Build the ``.mcp.json`` payload registering ERA's MCP servers.

    ``{venv_bin}`` placeholders — in any server field, e.g. a stdio server's
    ``command`` — are resolved to the absolute repo-venv ``bin`` path. ``${VAR}``
    references (e.g. an HTTP server's auth header) are left literal for Claude
    Code to expand from the environment at launch. Handles both stdio and http
    server specs.
    """
    raw = json.dumps({"mcpServers": MCP_SERVERS}, ensure_ascii=False)
    return json.loads(raw.replace("{venv_bin}", _venv_bin()))


def ensure_mcp_config(root: str | Path) -> Path:
    """Ensure ``<root>/.mcp.json`` registers ERA's MCP servers.

    Creates the file when absent. When it already exists, **merges in** any
    registry server missing from it — so a server added to ``MCP_SERVERS`` in a
    later ERA version (e.g. ``codex``) still reaches an already-scaffolded
    directory — while leaving every server already present untouched, so
    operator edits and customized entries survive. Written atomically; a
    malformed existing file is left alone. Returns the ``.mcp.json`` path.
    """
    path = Path(root) / ".mcp.json"
    desired = mcp_config()
    if not path.exists():
        _write_text_atomic(
            path,
            json.dumps(desired, indent=2, ensure_ascii=False) + "\n",
        )
        return path

    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return path  # an unreadable / malformed operator file is left alone
    if not isinstance(existing, dict):
        return path

    servers = existing.setdefault("mcpServers", {})
    changed = False
    for name, spec in desired["mcpServers"].items():
        if name not in servers:
            servers[name] = spec
            changed = True
    if changed:
        _write_text_atomic(
            path,
            json.dumps(existing, indent=2, ensure_ascii=False) + "\n",
        )
    return path


def _write_text_atomic(path: Path, text: str) -> None:
    """Write text via a temp file + ``os.replace`` so readers never see a
    partial file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
