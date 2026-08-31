"""End-to-end mock test of ERA's iron autonomy contract.

The user's directive (`feedback-era-start-no-questions` memory file):

> *Stages 1-7 + 9-10 run unattended; Stage 8 is the ONLY operator
> hand-off. The PreToolUse hook structurally blocks AskUserQuestion
> outside Stage 8.*

This test walks the workspace through every stage's ``status.json``
shape and asserts:

1. The PreToolUse autonomy hook (``era.cli check-autonomy``) blocks
   AskUserQuestion at every stage where ``run_state`` is ``running``
   or ``blocked`` (Stages 1-7 + 9-10).
2. The hook allows AskUserQuestion only when ``run_state ==
   "awaiting_human"`` (the legitimate Stage 8 hand-off).
3. The full pre-Stage-8 failure recovery chain (Stage 7 gate fails →
   ``auto-revise`` → next iter scaffolded → run_state: running) runs
   end-to-end without any operator prompt site being triggered.
4. The skill / sub-agent surface itself can't ask: the
   `tools` / `allowed-tools` allowlists of every autonomous skill
   and sub-agent omit ``AskUserQuestion``. Only the Stage 8 skill
   (``era-human-feedback``) has it — by design, behind the
   ``awaiting_human`` marker.

If any assertion here fails, the iron rule is leaking. This is the
top-level safety test for the autonomy contract; treat it as
load-bearing.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import yaml

from era.cli import main
from era.orchestration.auto_revise import auto_revise
from era.workspace import Workspace


# ---- Fixtures ----------------------------------------------------------

def _make_workspace(
    base: Path, *,
    iteration: int = 1, max_iterations: int = 5,
    name: str = "autonomy-demo",
) -> Workspace:
    """Minimal workspace with status.json + config.yaml."""
    ws = Workspace(base, name)
    ws.scaffold()
    for n in range(1, iteration + 1):
        ws.create_iteration(n)
    ws.set_current(iteration)
    ws.write_status({
        "project_name": name, "stage": "research",
        "stage_index": 1, "iteration": iteration, "run_state": "running",
    })
    (ws.root / "config.yaml").write_text(yaml.safe_dump({
        "project_name": name, "task_family": "editing", "task_adapter": "x",
        "hardware": {"visible_gpu_ids": [0], "max_gpus_per_run": 1,
                     "per_gpu_memory_gb": 10.0},
        "data": {"data_root": "/tmp/x", "layout": "per_sample_dirs",
                 "methods": [{"method_id": "m", "path": "/tmp/m",
                              "output_file": "out.png"}],
                 "sample_glob": "*", "sample_count": 1,
                 "input_roles": {"input": "in.png"},
                 "sample_key": "relpath"},
        "react": {"max_iterations": max_iterations,
                  "endorsement_threshold": 0.80,
                  "min_alignment_samples": 20},
    }), encoding="utf-8")
    return ws


def _check_autonomy_in_workspace(monkeypatch, ws_root: Path) -> int:
    """Invoke `era.cli check-autonomy` with the workspace as cwd; return
    the exit code."""
    monkeypatch.chdir(ws_root)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    monkeypatch.setattr("sys.stderr", io.StringIO())
    return main(["check-autonomy"])


# ---- 1. Hook gates every pre-Stage-8 + post-Stage-8 stage --------------

STAGES_THAT_MUST_BLOCK = [
    # (stage_name, stage_index) — every stage where the agent must NOT
    # prompt the operator. Per the iron rule this is stages 1-7 + 9-10.
    ("research", 1),
    ("plan_brainstorm", 2),
    ("multi_review", 3),
    ("plan_decision", 4),
    ("experiment_plan", 5),
    ("full_experiment", 6),
    ("pre_human_comparison", 7),
    ("react", 9),
    ("final_report", 10),
]


def test_hook_blocks_every_pre_stage8_and_post_stage8_stage(
    tmp_path: Path, monkeypatch,
):
    """The PreToolUse hook must structurally block AskUserQuestion at
    every stage except 8. Walk through each stage by stubbing status.json
    and assert the hook exits 2."""
    ws = _make_workspace(tmp_path)
    for stage_name, stage_index in STAGES_THAT_MUST_BLOCK:
        ws.write_status({
            "project_name": "autonomy-demo",
            "stage": stage_name, "stage_index": stage_index,
            "iteration": 1, "run_state": "running",
        })
        rc = _check_autonomy_in_workspace(monkeypatch, ws.root)
        assert rc == 2, (
            f"Stage {stage_index} ({stage_name}): hook should BLOCK "
            f"AskUserQuestion when run_state=running, got exit {rc}"
        )


def test_hook_allows_stage8_awaiting_human(tmp_path: Path, monkeypatch):
    """Stage 8 is the ONLY legitimate operator hand-off. The hook allows
    AskUserQuestion when run_state=awaiting_human (the era-human-feedback
    skill sets this marker BEFORE its Continue prompt)."""
    ws = _make_workspace(tmp_path)
    ws.write_status({
        "project_name": "autonomy-demo",
        "stage": "human_feedback", "stage_index": 8,
        "iteration": 1, "run_state": "awaiting_human",
    })
    rc = _check_autonomy_in_workspace(monkeypatch, ws.root)
    assert rc == 0, (
        f"Stage 8 with run_state=awaiting_human must ALLOW the prompt, "
        f"got exit {rc}"
    )


def test_hook_blocks_stage8_if_marker_not_set(tmp_path: Path, monkeypatch):
    """A Stage 8 dispatch that has NOT yet set the awaiting_human marker
    still has run_state=running. Hook must still block — only the
    explicit marker is the green light."""
    ws = _make_workspace(tmp_path)
    ws.write_status({
        "project_name": "autonomy-demo",
        "stage": "human_feedback", "stage_index": 8,
        "iteration": 1, "run_state": "running",
    })
    rc = _check_autonomy_in_workspace(monkeypatch, ws.root)
    assert rc == 2


def test_hook_blocks_pre_stage8_blocked_state(tmp_path: Path, monkeypatch):
    """If somehow a pre-Stage-8 stage set run_state=blocked (which the
    auto-revise machinery now prevents), the hook STILL blocks — that's
    the agent's signal to auto-revise, not to ask."""
    ws = _make_workspace(tmp_path)
    ws.write_status({
        "project_name": "autonomy-demo",
        "stage": "full_experiment", "stage_index": 6,
        "iteration": 1, "run_state": "blocked",
    })
    rc = _check_autonomy_in_workspace(monkeypatch, ws.root)
    assert rc == 2


# ---- 2. Pre-Stage-8 failure auto-revises to next iter ------------------

def test_stage7_failure_routes_through_auto_revise_no_operator_prompt(
    tmp_path: Path, monkeypatch,
):
    """If Stage 7's pass/recall gate (or its pre-flight) decides the
    iter has no path forward, the prompt tells the agent to call
    auto-revise rather than block on the operator. Walk that chain:

    1. Workspace at Stage 7, run_state=running.
    2. Stage 7 detects all-fail / no-annotated-scores and calls
       auto-revise.
    3. auto-revise writes the trigger, fires react_tick(REVISE_SKIP_STAGE1),
       scaffolds iter_002, sets run_state=running on the new iter.
    4. At no point was run_state ever blocked or awaiting_human; the hook
       would have allowed nothing operator-interactive.
    """
    ws = _make_workspace(tmp_path, iteration=1, max_iterations=5)
    # Move to Stage 7 mid-loop.
    ws.write_status({
        "project_name": "autonomy-demo",
        "stage": "pre_human_comparison", "stage_index": 7,
        "iteration": 1, "run_state": "running",
    })

    # Hook MUST block any AskUserQuestion here — Stage 7.
    rc = _check_autonomy_in_workspace(monkeypatch, ws.root)
    assert rc == 2, "Stage 7 with run_state=running must block AskUserQuestion"

    # Stage 7 fails its gate → auto-revise.
    result = auto_revise(
        ws.root,
        reason="stage7_auto_validate_failed",
        source_stage=7,
        blocker_summary="all configs failed pass/recall on annotated subset",
        diagnostic={"per_config": [{"combination_id": "cfg-a",
                                     "pass_rate": 0.4, "recall_rate": 0.1}]},
    )
    assert result["status"] == "ok"
    assert result["decision"] == "REVISE_SKIP_STAGE1", (
        "Under-cap auto-revise should produce REVISE_SKIP_STAGE1, "
        f"got {result.get('decision')!r}"
    )
    assert result["forced_advance"] is False
    assert result["next_iter"] == 2

    # Trigger file exists on the prior iter for Stage 9's audit trail.
    trigger_path = ws.root / "iter_001" / "auto_revise" / "trigger.json"
    assert trigger_path.is_file()
    trigger = json.loads(trigger_path.read_text(encoding="utf-8"))
    assert trigger["reason"] == "stage7_auto_validate_failed"
    assert trigger["source_stage"] == 7

    # The new iter MUST be running (not blocked, not awaiting_human).
    status = ws.read_status()
    assert status["iteration"] == 2, "Workspace should have advanced to iter_002"
    assert status["run_state"] == "running", (
        f"After auto-revise the loop must be running, not "
        f"{status.get('run_state')!r} — never block on the operator"
    )
    # The new iter's parent_feedback carries the trigger pointer for
    # Stage 9's advisor.
    iter2_meta = json.loads(
        (ws.root / "iter_002" / "iteration.json").read_text(encoding="utf-8"))
    assert (iter2_meta["parent_feedback"]["auto_revise_trigger"]
            == "iter_001/auto_revise/trigger.json")

    # Hook on the new iter still blocks — we're back at Stage 2 / iter 2,
    # run_state=running.
    rc = _check_autonomy_in_workspace(monkeypatch, ws.root)
    assert rc == 2


def test_iter_cap_forces_advance_not_operator_prompt(
    tmp_path: Path, monkeypatch,
):
    """At react.max_iterations, auto-revise returns forced_advance=true
    instead of scaffolding another iter — the loop advances to Stage 10's
    terminal block. The operator is STILL never prompted."""
    ws = _make_workspace(tmp_path, iteration=5, max_iterations=5)
    result = auto_revise(
        ws.root,
        reason="stage6_incomplete",
        source_stage=6,
        blocker_summary="at cap",
    )
    assert result["status"] == "ok"
    assert result["forced_advance"] is True
    assert result["decision"] == "ADVANCE"
    # status.json is left for the ralph loop to advance stage_index;
    # run_state should NOT have flipped to awaiting_human or blocked.
    status = ws.read_status()
    assert status.get("run_state") != "awaiting_human"
    # Hook still blocks at iter 5 (no operator prompt at cap).
    rc = _check_autonomy_in_workspace(monkeypatch, ws.root)
    assert rc == 2


# ---- 3. Static surface checks (defence in depth) -----------------------

def test_only_stage8_skill_has_ask_user_question_in_allowed_tools():
    """The skill-level allowlist is the FIRST gate. Walk every skill in
    plugin/skills/ and assert only era-human-feedback (Stage 8) carries
    AskUserQuestion in its frontmatter."""
    repo_root = Path(__file__).resolve().parents[1]
    skill_files = sorted(
        (repo_root / "plugin" / "skills").glob("*/SKILL.md"))
    assert skill_files, "no skills found"

    for skill_path in skill_files:
        text = skill_path.read_text(encoding="utf-8")
        # Parse the YAML frontmatter.
        assert text.startswith("---\n"), f"{skill_path} has no frontmatter"
        fm = yaml.safe_load(text.split("---\n", 2)[1])
        allowed = fm.get("allowed-tools") or fm.get("tools") or ""
        if isinstance(allowed, list):
            allowed_str = ",".join(str(x) for x in allowed)
        else:
            allowed_str = str(allowed)
        has_aq = "AskUserQuestion" in allowed_str
        if skill_path.name == "SKILL.md" and skill_path.parent.name == "era-human-feedback":
            assert has_aq, (
                "era-human-feedback (Stage 8) MUST have AskUserQuestion "
                "— it's the only legitimate site"
            )
        else:
            assert not has_aq, (
                f"{skill_path.parent.name} must NOT allow AskUserQuestion "
                f"(only era-human-feedback may). Found in tools list."
            )


def test_no_sub_agent_has_ask_user_question_in_tools():
    """Every autonomous sub-agent's tools field must omit
    AskUserQuestion — they're not allowed to ask the operator either."""
    repo_root = Path(__file__).resolve().parents[1]
    agent_files = sorted((repo_root / "plugin" / "agents").glob("*.md"))
    assert agent_files, "no sub-agents found"

    for agent_path in agent_files:
        text = agent_path.read_text(encoding="utf-8")
        assert text.startswith("---\n"), f"{agent_path} has no frontmatter"
        fm = yaml.safe_load(text.split("---\n", 2)[1])
        tools = fm.get("tools") or ""
        if isinstance(tools, list):
            tools_str = ",".join(str(x) for x in tools)
        else:
            tools_str = str(tools)
        assert "AskUserQuestion" not in tools_str, (
            f"Sub-agent {agent_path.name} declares AskUserQuestion in its "
            f"tools field — violates the iron autonomy rule. Stage 8's "
            f"era-human-feedback skill is the ONLY allowed AskUserQuestion "
            f"site."
        )


def test_pre_stage8_skills_dont_set_blocked_run_state():
    """No pre-Stage-8 skill prompt should set run_state=blocked; they
    must auto-revise instead (Phase C-1 contract). Belt-and-suspenders
    grep — the ralph_loop.md already documents this, but per-stage
    prompts should not contradict."""
    repo_root = Path(__file__).resolve().parents[1]
    prompts = [
        "stage2_brainstorm.md", "stage3_review.md", "stage4_decision.md",
        "stage5_experiment_plan.md", "stage6_experiment.md",
        "stage7_pre_human_comparison.md",
    ]
    for name in prompts:
        text = (repo_root / "docs" / "prompts" / name).read_text(
            encoding="utf-8")
        # The string "run_state: blocked" must NOT appear as an
        # instruction to the agent in a pre-Stage-8 stage. Mentions
        # like "do NOT set run_state: blocked" are fine.
        for line in text.splitlines():
            if "run_state: blocked" not in line:
                continue
            # Allow lines that explicitly forbid it.
            lower = line.lower()
            if any(neg in lower
                   for neg in ("not ", "no longer", "never",
                               "instead of", "do not", "without",
                               "rather than")):
                continue
            raise AssertionError(
                f"{name}: line tells the agent to set run_state: blocked "
                f"in a pre-Stage-8 stage — violates the auto-revise "
                f"contract. Line: {line.strip()!r}"
            )


# ---- 4. Workflow integration — full stage 1→7 unattended ---------------

def test_full_stage_1_through_7_no_operator_prompts(
    tmp_path: Path, monkeypatch,
):
    """Walk through Stages 1→7 in sequence. At EVERY stage the hook
    must block AskUserQuestion. This is the integration test that
    catches a regression in the hook OR in the status.json contract."""
    ws = _make_workspace(tmp_path)
    stages = [
        ("research", 1), ("plan_brainstorm", 2), ("multi_review", 3),
        ("plan_decision", 4), ("experiment_plan", 5),
        ("full_experiment", 6), ("pre_human_comparison", 7),
    ]
    for stage_name, stage_index in stages:
        ws.write_status({
            "project_name": "autonomy-demo",
            "stage": stage_name, "stage_index": stage_index,
            "iteration": 1, "run_state": "running",
        })
        rc = _check_autonomy_in_workspace(monkeypatch, ws.root)
        assert rc == 2, (
            f"AUTONOMY VIOLATION: Stage {stage_index} ({stage_name}) "
            f"would allow AskUserQuestion. The iron rule says Stages 1-7 "
            f"run unattended; only Stage 8 may ask."
        )

    # Now flip to Stage 8 and verify the legitimate prompt fires.
    ws.write_status({
        "project_name": "autonomy-demo",
        "stage": "human_feedback", "stage_index": 8,
        "iteration": 1, "run_state": "awaiting_human",
    })
    rc = _check_autonomy_in_workspace(monkeypatch, ws.root)
    assert rc == 0, "Stage 8 awaiting_human must allow the prompt"

    # Then back to Stage 9 — block again.
    ws.write_status({
        "project_name": "autonomy-demo",
        "stage": "react", "stage_index": 9,
        "iteration": 1, "run_state": "running",
    })
    rc = _check_autonomy_in_workspace(monkeypatch, ws.root)
    assert rc == 2, "Stage 9 must block AskUserQuestion (post-Stage-8)"
