"""Tests for Stage 2-4 sub-agent tier resolution + the era-<tier> sub-agents."""

from __future__ import annotations

import io
import json
from pathlib import Path

import yaml
from conftest import valid_params

from era.cli import main
from era.config import ERAConfig
from era.orchestration.agents import resolve_agent_tier
from era.paths import repo_root
from era.workspace import Workspace


def _workspace_with_config(base: Path, per_sample_root: Path, **agent_modes):
    """Scaffold a workspace and write a config.yaml carrying agent_modes."""
    params = valid_params(base, per_sample_root)
    if agent_modes:
        params["agent_modes"] = agent_modes
    cfg = ERAConfig.from_params(params)
    ws = Workspace(base, params["project_name"])
    ws.scaffold()
    ws.create_iteration(1)
    ws.set_current(1)
    ws.write_status({
        "project_name": params["project_name"], "stage": "research",
        "stage_index": 1, "iteration": 1, "run_state": "running",
    })
    ws.write_file("config.yaml", cfg.to_commented_yaml())
    return ws


# ---- resolve_agent_tier -------------------------------------------------

def test_resolve_agent_tier_defaults(tmp_path: Path, per_sample_root: Path):
    ws = _workspace_with_config(tmp_path, per_sample_root)
    assert resolve_agent_tier(ws.root, "plan_brainstorm")["agent"] == "era-standard"
    assert resolve_agent_tier(ws.root, "multi_review")["agent"] == "era-light"
    assert resolve_agent_tier(ws.root, "plan_decision")["agent"] == "era-heavy"


def test_resolve_agent_tier_custom(tmp_path: Path, per_sample_root: Path):
    ws = _workspace_with_config(tmp_path, per_sample_root,
                                plan_brainstorm="heavy")
    result = resolve_agent_tier(ws.root, "plan_brainstorm")
    assert result["tier"] == "heavy"
    assert result["agent"] == "era-heavy"


def test_resolve_agent_tier_bad_stage(tmp_path: Path, per_sample_root: Path):
    ws = _workspace_with_config(tmp_path, per_sample_root)
    assert resolve_agent_tier(ws.root, "nonexistent_stage")["error"] == "bad_stage"


def test_resolve_agent_tier_experiment_stages(
    tmp_path: Path, per_sample_root: Path
):
    ws = _workspace_with_config(tmp_path, per_sample_root)
    assert resolve_agent_tier(ws.root, "experiment_plan")["agent"] == "era-heavy"
    assert (resolve_agent_tier(ws.root, "full_experiment")["agent"]
            == "era-standard")


def test_resolve_agent_tier_no_config(tmp_path: Path):
    result = resolve_agent_tier(tmp_path / "nope", "plan_brainstorm")
    assert result["error"] == "no_config"


def test_cli_agent_tier_round_trip(
    tmp_path: Path, per_sample_root: Path, monkeypatch, capsys
):
    ws = _workspace_with_config(tmp_path, per_sample_root)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "stage": "plan_decision"})))
    code = main(["agent-tier"])
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["agent"] == "era-heavy"


def test_cli_agent_tier_missing_params(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
    code = main(["agent-tier"])
    result = json.loads(capsys.readouterr().out)
    assert code == 1
    assert result["error"] == "missing_params"


# ---- era-<tier> sub-agent frontmatter -----------------------------------

def _frontmatter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"{path} has no YAML frontmatter"
    return yaml.safe_load(text.split("---\n", 2)[1])


def test_tier_subagent_frontmatter():
    agents_dir = repo_root() / "plugin" / "agents"
    expect_model = {
        "era-heavy": "opus", "era-standard": "opus", "era-light": "sonnet",
    }
    for name, model in expect_model.items():
        fm = _frontmatter(agents_dir / f"{name}.md")
        assert fm["name"] == name
        assert fm["model"] == model
        assert "AskUserQuestion" not in fm["tools"]


def test_standard_tier_has_research_mcp_tools():
    fm = _frontmatter(repo_root() / "plugin" / "agents" / "era-standard.md")
    assert "mcp__arxiv-mcp-server__search_papers" in fm["tools"]
    assert "mcp__github__search_repositories" in fm["tools"]


def test_light_tier_is_file_only():
    fm = _frontmatter(repo_root() / "plugin" / "agents" / "era-light.md")
    assert "Bash" not in fm["tools"]
    assert "WebSearch" not in fm["tools"]


def test_era_auto_validator_frontmatter():
    """Phase C-2 era-auto-validator: light-tier (haiku) semantic judge,
    file-only tools, never asks the operator."""
    fm = _frontmatter(
        repo_root() / "plugin" / "agents" / "era-auto-validator.md"
    )
    assert fm["name"] == "era-auto-validator"
    assert fm["model"] == "haiku"
    # Iron rule: autonomous sub-agents must not have AskUserQuestion.
    assert "AskUserQuestion" not in fm["tools"]
    # File-only: the sub-agent reads inputs and writes judgments — no
    # Bash, no MCP, no web.
    assert "Bash" not in fm["tools"]
    assert "WebSearch" not in fm["tools"]
    assert "Read" in fm["tools"]
    assert "Write" in fm["tools"]
