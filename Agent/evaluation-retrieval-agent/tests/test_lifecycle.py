"""Tests for the ERA runtime lifecycle helpers + CLI subcommands."""

from __future__ import annotations

import io
import json
from pathlib import Path

import yaml

from era.cli import main
from era.orchestration.lifecycle import (
    RUN_STATES,
    STAGES,
    status_summary,
    update_status,
)
from era.paths import repo_root
from era.workspace import Workspace


def _make_workspace(base: Path, name: str = "demo-eval") -> Workspace:
    ws = Workspace(base, name)
    ws.scaffold()
    ws.create_iteration(1)
    ws.set_current(1)
    ws.write_status({
        "project_name": name,
        "stage": "task_init",
        "stage_index": 0,
        "iteration": 1,
        "run_state": "running",
    })
    return ws


# ---- update_status ------------------------------------------------------

def test_update_status_patches_whitelisted_fields(tmp_path: Path):
    ws = _make_workspace(tmp_path)
    result = update_status(ws.root, stage="research", stage_index=1)
    assert result["status"] == "ok"
    status = ws.read_status()
    assert status["stage"] == "research"
    assert status["stage_index"] == 1
    assert "updated_at" in status


def test_update_status_run_state_values(tmp_path: Path):
    ws = _make_workspace(tmp_path)
    for state in RUN_STATES:
        result = update_status(ws.root, run_state=state)
        assert result["status"] == "ok"
        assert ws.read_status()["run_state"] == state


def test_update_status_rejects_unknown_field(tmp_path: Path):
    ws = _make_workspace(tmp_path)
    result = update_status(ws.root, bogus="x")
    assert result["error"] == "unknown_fields"
    assert "bogus" not in ws.read_status()


def test_update_status_rejects_bad_run_state(tmp_path: Path):
    ws = _make_workspace(tmp_path)
    result = update_status(ws.root, run_state="paused")
    assert result["error"] == "bad_run_state"


def test_update_status_not_a_workspace(tmp_path: Path):
    result = update_status(tmp_path / "nope")
    assert result["error"] == "not_a_workspace"


def test_update_status_resolves_current_pointer(tmp_path: Path):
    ws = _make_workspace(tmp_path)
    result = update_status(ws.root / "current", run_state="stopped")
    assert result["status"] == "ok"
    assert ws.read_status()["run_state"] == "stopped"


# ---- status_summary -----------------------------------------------------

def test_status_summary_all(tmp_path: Path):
    workspaces = tmp_path / "workspaces"
    _make_workspace(workspaces, "proj-a")
    _make_workspace(workspaces, "proj-b")
    result = status_summary(workspaces)
    assert result["count"] == 2
    assert {p["project_name"] for p in result["projects"]} == {"proj-a", "proj-b"}
    assert all("run_state" in p for p in result["projects"])
    assert all(p["has_literature"] is False for p in result["projects"])


def test_status_summary_one(tmp_path: Path):
    workspaces = tmp_path / "workspaces"
    ws = _make_workspace(workspaces, "proj-a")
    _make_workspace(workspaces, "proj-b")
    result = status_summary(workspaces, ws.root)
    assert result["count"] == 1
    assert result["projects"][0]["project_name"] == "proj-a"


def test_status_summary_empty(tmp_path: Path):
    result = status_summary(tmp_path / "nothing")
    assert result["count"] == 0
    assert result["projects"] == []


# ---- Phase C-2.5: backward-compat warnings -----------------------------

def _write_minimal_config(
    ws: Workspace, *, experiment: dict | None = None,
) -> None:
    """Write a config.yaml that triggers the warning helper's read path."""
    cfg = {"project_name": ws.root.name}
    if experiment is not None:
        cfg["experiment"] = experiment
    (ws.root / "config.yaml").write_text(
        yaml.safe_dump(cfg), encoding="utf-8"
    )


def test_status_summary_warns_when_min_passing_not_pinned(tmp_path: Path):
    """Phase C-2.5 backward-compat: a workspace whose config.yaml predates
    the M-threshold (no `auto_validate_min_passing`) gets a warning so
    operators upgrading from <v0.1.7 don't get silently auto-revised."""
    workspaces = tmp_path / "workspaces"
    ws = _make_workspace(workspaces, "proj-legacy")
    # Legacy config — no experiment block at all.
    _write_minimal_config(ws)
    result = status_summary(workspaces, ws.root)
    proj = result["projects"][0]
    assert "warnings" in proj
    msg = " ".join(proj["warnings"])
    assert "auto_validate_min_passing" in msg
    assert "v0.1.7" in msg or "C-2.5" in msg


def test_status_summary_no_warning_when_min_passing_pinned(tmp_path: Path):
    """Once the operator pins auto_validate_min_passing (any value, incl.
    legacy 1), the warning subsides."""
    workspaces = tmp_path / "workspaces"
    ws = _make_workspace(workspaces, "proj-pinned")
    _write_minimal_config(
        ws, experiment={"auto_validate_min_passing": 1},
    )
    result = status_summary(workspaces, ws.root)
    proj = result["projects"][0]
    assert "warnings" not in proj


def test_status_summary_no_warning_when_config_missing(tmp_path: Path):
    """A workspace without config.yaml (mid-init) skips the check entirely
    — surfacing the warning here would be noise."""
    workspaces = tmp_path / "workspaces"
    ws = _make_workspace(workspaces, "proj-no-config")
    result = status_summary(workspaces, ws.root)
    proj = result["projects"][0]
    assert "warnings" not in proj


def test_status_summary_no_warning_when_experiment_empty(tmp_path: Path):
    """An empty experiment block (the operator started pinning but didn't
    set min_passing) still triggers the warning — they need it
    explicitly."""
    workspaces = tmp_path / "workspaces"
    ws = _make_workspace(workspaces, "proj-empty-exp")
    _write_minimal_config(ws, experiment={})
    result = status_summary(workspaces, ws.root)
    proj = result["projects"][0]
    assert "warnings" in proj
    assert any("auto_validate_min_passing" in w for w in proj["warnings"])


def test_status_summary_handles_malformed_yaml(tmp_path: Path):
    """A corrupt config.yaml doesn't crash status_summary — the warning
    helper fails-quiet so /era:status still works."""
    workspaces = tmp_path / "workspaces"
    ws = _make_workspace(workspaces, "proj-bad-yaml")
    (ws.root / "config.yaml").write_text(
        "{ this isn't valid YAML : : :", encoding="utf-8",
    )
    result = status_summary(workspaces, ws.root)
    assert result["count"] == 1
    proj = result["projects"][0]
    # The helper failed-quiet → no warnings emitted; status still
    # populates normally.
    assert "warnings" not in proj


# ---- CLI round-trips ----------------------------------------------------

def _run_cli(argv, stdin_obj, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(stdin_obj)))
    code = main(argv)
    return code, json.loads(capsys.readouterr().out)


def test_cli_update_status_round_trip(tmp_path, monkeypatch, capsys):
    ws = _make_workspace(tmp_path)
    code, result = _run_cli(
        ["update-status"],
        {"workspace_path": str(ws.root), "run_state": "stopped"},
        monkeypatch, capsys,
    )
    assert code == 0
    assert result["status"] == "ok"
    assert ws.read_status()["run_state"] == "stopped"


def test_cli_update_status_missing_path(tmp_path, monkeypatch, capsys):
    code, result = _run_cli(["update-status"], {}, monkeypatch, capsys)
    assert code == 1
    assert result["error"] == "missing_workspace_path"


def test_cli_status_round_trip(tmp_path, monkeypatch, capsys):
    ws = _make_workspace(tmp_path)
    code, result = _run_cli(
        ["status"], {"workspace_path": str(ws.root)}, monkeypatch, capsys
    )
    assert code == 0
    assert result["count"] == 1
    assert result["projects"][0]["project_name"] == "demo-eval"


# ---- skill / agent frontmatter ------------------------------------------

def _frontmatter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"{path} has no YAML frontmatter"
    return yaml.safe_load(text.split("---\n", 2)[1])


def test_literature_skill_frontmatter():
    fm = _frontmatter(
        repo_root() / "plugin" / "skills" / "era-literature" / "SKILL.md"
    )
    assert fm["name"] == "era-literature"
    tools = fm["allowed-tools"]
    assert "mcp__arxiv-mcp-server__search_papers" in tools
    assert "Task" in tools
    assert "AskUserQuestion" not in tools


def test_literature_scout_frontmatter():
    fm = _frontmatter(repo_root() / "plugin" / "agents" / "literature-scout.md")
    assert fm["name"] == "literature-scout"
    tools = fm["tools"]
    assert "mcp__arxiv-mcp-server__search_papers" in tools
    assert "AskUserQuestion" not in tools
    assert "Bash" not in tools


def test_command_frontmatter_parses():
    cmd_dir = repo_root() / "plugin" / "commands"
    files = sorted(cmd_dir.glob("*.md"))
    # init, start, status, stop, resume, annotate
    assert len(files) == 6
    for path in files:
        fm = _frontmatter(path)
        assert fm.get("description"), f"{path.name} missing description"


def test_debate_skill_frontmatter():
    """The 3 Stage 2-4 orchestrator skills declare Task + Bash, never Ask."""
    skills_dir = repo_root() / "plugin" / "skills"
    for name in ("era-plan-brainstorm", "era-multi-review", "era-plan-decision"):
        fm = _frontmatter(skills_dir / name / "SKILL.md")
        assert fm["name"] == name
        tools = fm["allowed-tools"]
        assert "Task" in tools
        assert "Bash" in tools
        assert "AskUserQuestion" not in tools


def test_experiment_skill_frontmatter():
    """The Stage 5-6 orchestrator skills declare Task + Bash, never Ask."""
    skills_dir = repo_root() / "plugin" / "skills"
    for name in ("era-experiment-plan", "era-experiment"):
        fm = _frontmatter(skills_dir / name / "SKILL.md")
        assert fm["name"] == name
        tools = fm["allowed-tools"]
        assert "Task" in tools
        assert "Bash" in tools
        assert "AskUserQuestion" not in tools


def test_human_feedback_skill_frontmatter():
    """Stage 7 stays autonomous; Stage 8 is the lone exception that may ask."""
    skills_dir = repo_root() / "plugin" / "skills"

    pre = _frontmatter(skills_dir / "era-pre-human-comparison" / "SKILL.md")
    assert pre["name"] == "era-pre-human-comparison"
    pre_tools = pre["allowed-tools"]
    assert "Bash" in pre_tools
    assert "Write" in pre_tools
    assert "AskUserQuestion" not in pre_tools

    hf = _frontmatter(skills_dir / "era-human-feedback" / "SKILL.md")
    assert hf["name"] == "era-human-feedback"
    hf_tools = hf["allowed-tools"]
    assert "Bash" in hf_tools
    assert "Write" in hf_tools
    # Stage 8 is the one place the loop may ask the operator — the in-loop
    # wait prompt (Continue / Still working / Cancel) gates Stage 9.
    assert "AskUserQuestion" in hf_tools


def test_codex_reviewer_agent_frontmatter():
    """The Stage 6 Codex reviewer agent has the codex MCP tool, no Write/Ask."""
    fm = _frontmatter(repo_root() / "plugin" / "agents" / "era-codex-reviewer.md")
    assert fm["name"] == "era-codex-reviewer"
    tools = fm["tools"]
    assert "mcp__codex__codex" in tools
    assert "AskUserQuestion" not in tools
    assert "Write" not in tools


# ---- stage validation & derivation --------------------------------------

def test_stages_canonical_list():
    assert len(STAGES) == 11
    assert STAGES[0] == "task_init"
    assert STAGES[1] == "research"
    assert STAGES[7] == "pre_human_comparison"
    assert STAGES[8] == "human_feedback"
    assert STAGES[9] == "react"
    assert STAGES[10] == "final_report"
    assert "deployment" not in STAGES
    assert "dataset_ship" not in STAGES


def test_stages_debate_block():
    assert STAGES[2:5] == ("plan_brainstorm", "multi_review", "plan_decision")


def test_update_status_rejects_iteration(tmp_path: Path):
    ws = _make_workspace(tmp_path)
    result = update_status(ws.root, iteration=2)
    assert result["error"] == "unknown_fields"
    assert ws.read_status()["iteration"] == 1  # untouched


def test_update_status_derives_stage_from_index(tmp_path: Path):
    ws = _make_workspace(tmp_path)
    result = update_status(ws.root, stage_index=1)
    assert result["status"] == "ok"
    status = ws.read_status()
    assert status["stage"] == "research"
    assert status["stage_index"] == 1


def test_update_status_derives_index_from_stage(tmp_path: Path):
    ws = _make_workspace(tmp_path)
    result = update_status(ws.root, stage="human_feedback")
    assert result["status"] == "ok"
    assert ws.read_status()["stage_index"] == STAGES.index("human_feedback")


def test_update_status_rejects_bad_stage(tmp_path: Path):
    ws = _make_workspace(tmp_path)
    assert update_status(ws.root, stage="bogus")["error"] == "bad_stage"


def test_update_status_rejects_bad_stage_index(tmp_path: Path):
    ws = _make_workspace(tmp_path)
    assert update_status(ws.root, stage_index=99)["error"] == "bad_stage"


def test_update_status_rejects_stage_index_mismatch(tmp_path: Path):
    ws = _make_workspace(tmp_path)
    result = update_status(ws.root, stage="research", stage_index=2)
    assert result["error"] == "bad_stage"
