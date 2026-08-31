"""Tests for ERA MCP-server registration (`.mcp.json` generation)."""

from __future__ import annotations

import io
import json
from pathlib import Path

from era.cli import main
from era.orchestration.mcp import ensure_mcp_config, mcp_config
from era.paths import repo_root


def test_mcp_config_registers_arxiv_server():
    server = mcp_config()["mcpServers"]["arxiv-mcp-server"]
    assert server["type"] == "stdio"


def test_mcp_config_command_is_absolute_venv_path():
    """The server command must be absolute (cwd-independent) and point into the
    repo venv, so it launches correctly from any workspace directory."""
    command = mcp_config()["mcpServers"]["arxiv-mcp-server"]["command"]
    assert Path(command).is_absolute()
    assert "{venv_bin}" not in command
    assert command == str(repo_root() / ".venv" / "bin" / "arxiv-mcp-server")


def test_mcp_config_registers_github_http_server():
    """The github server is HTTP-transport with GitHub's hosted endpoint."""
    server = mcp_config()["mcpServers"]["github"]
    assert server["type"] == "http"
    assert server["url"] == "https://api.githubcopilot.com/mcp/"
    assert "command" not in server  # http servers carry a url, not a command


def test_mcp_config_registers_codex_stdio_server():
    """The codex server (Stage 6 reviewer) is stdio-transport via the codex CLI."""
    server = mcp_config()["mcpServers"]["codex"]
    assert server["type"] == "stdio"
    assert server["command"] == "codex"
    assert server["args"] == ["mcp"]


def test_mcp_config_renders_mixed_stdio_and_http():
    """Regression: the renderer handles a stdio server (with `command`) and an
    http server (with `url`, no `command`) in one config, and leaves `${VAR}`
    references literal for Claude Code to expand."""
    servers = mcp_config()["mcpServers"]
    assert "command" in servers["arxiv-mcp-server"]
    assert "url" in servers["github"]
    auth = servers["github"]["headers"]["Authorization"]
    assert auth == "Bearer ${GITHUB_PERSONAL_ACCESS_TOKEN}"


def test_ensure_mcp_config_writes_file(tmp_path: Path):
    path = ensure_mcp_config(tmp_path)
    assert path == tmp_path / ".mcp.json"
    assert path.is_file()

    data = json.loads(path.read_text(encoding="utf-8"))
    assert "arxiv-mcp-server" in data["mcpServers"]


def test_ensure_mcp_config_merges_missing_servers(tmp_path: Path):
    """An existing .mcp.json keeps its servers and gains missing registry ones."""
    custom = '{"mcpServers": {"custom": {"command": "x"}}}\n'
    (tmp_path / ".mcp.json").write_text(custom, encoding="utf-8")

    ensure_mcp_config(tmp_path)
    data = json.loads((tmp_path / ".mcp.json").read_text(encoding="utf-8"))
    # the operator's own entry survives untouched
    assert data["mcpServers"]["custom"] == {"command": "x"}
    # registry servers missing from the file are merged in
    for name in ("arxiv-mcp-server", "github", "codex"):
        assert name in data["mcpServers"]


def test_ensure_mcp_config_idempotent_when_complete(tmp_path: Path):
    """A second call on an already-complete .mcp.json does not rewrite it."""
    ensure_mcp_config(tmp_path)
    first = (tmp_path / ".mcp.json").read_text(encoding="utf-8")
    ensure_mcp_config(tmp_path)
    assert (tmp_path / ".mcp.json").read_text(encoding="utf-8") == first


def test_ensure_mcp_config_leaves_malformed_file_alone(tmp_path: Path):
    """A malformed operator .mcp.json is not clobbered."""
    bad = "{not json"
    (tmp_path / ".mcp.json").write_text(bad, encoding="utf-8")
    ensure_mcp_config(tmp_path)
    assert (tmp_path / ".mcp.json").read_text(encoding="utf-8") == bad


def test_cli_write_mcp_config_round_trip(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.stdin", io.StringIO(json.dumps({"target_dir": str(tmp_path)}))
    )
    code = main(["write-mcp-config"])
    result = json.loads(capsys.readouterr().out)

    assert code == 0
    assert result["status"] == "ok"
    assert result["servers"] == ["arxiv-mcp-server", "codex", "github"]
    assert (tmp_path / ".mcp.json").is_file()
    assert (tmp_path / ".claude" / "settings.json").is_file()


def test_cli_write_mcp_config_rejects_missing_dir(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.stdin", io.StringIO(json.dumps({"target_dir": "/no/such/era/dir"}))
    )
    code = main(["write-mcp-config"])
    result = json.loads(capsys.readouterr().out)

    assert code == 1
    assert result["error"] == "no_such_dir"
