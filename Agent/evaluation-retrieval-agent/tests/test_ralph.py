"""Tests for the runtime Ralph-loop prompt compiler."""

from __future__ import annotations

import io
import json
from pathlib import Path

from era.cli import main
from era.orchestration.ralph import (
    COMPLETION_PROMISE,
    RALPH_LOOP_PLUGIN,
    compile_ralph_prompt,
    ensure_claude_settings,
    write_ralph_prompt,
)
from era.workspace import Workspace


def _make_workspace(tmp_path: Path, name: str = "demo-eval") -> Workspace:
    ws = Workspace(tmp_path, name)
    ws.scaffold()
    ws.create_iteration(1)
    ws.set_current(1)
    ws.write_status({"stage": "task_init", "iteration": 1})
    return ws


def test_compile_substitutes_placeholders(tmp_path: Path):
    ws = _make_workspace(tmp_path)
    text = compile_ralph_prompt(ws.root, "demo-eval")
    assert "demo-eval" in text
    assert str(ws.root) in text
    assert "{{" not in text and "}}" not in text
    assert COMPLETION_PROMISE in text


def test_compile_defaults_project_name_to_dir(tmp_path: Path):
    ws = _make_workspace(tmp_path, "auto-named")
    text = compile_ralph_prompt(ws.root)
    assert "auto-named" in text


def test_compile_includes_debate_stage_branches(tmp_path: Path):
    """The compiled prompt drives Stages 2-9 and stubs only Stage 10."""
    ws = _make_workspace(tmp_path)
    text = compile_ralph_prompt(ws.root)
    for skill in ("era:era-plan-brainstorm", "era:era-multi-review",
                  "era:era-plan-decision", "era:era-experiment-plan",
                  "era:era-experiment", "era:era-pre-human-comparison",
                  "era:era-human-feedback", "era:era-react"):
        assert skill in text
    # Stage 10 (final_report) is the remaining v0.1.x stub.
    assert "final_report" in text


def test_write_creates_prompt_and_state(tmp_path: Path):
    ws = _make_workspace(tmp_path)
    result = write_ralph_prompt(ws.root)
    assert result["status"] == "ok"

    prompt_path = Path(result["prompt_path"])
    state_path = Path(result["state_path"])
    assert prompt_path == ws.root / ".claude" / "ralph-prompt.txt"
    assert prompt_path.is_file()
    assert state_path.is_file()
    assert not (ws.root / ".claude" / "ralph-prompt.txt.tmp").exists()

    assert "demo-eval" in prompt_path.read_text(encoding="utf-8")

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["project_name"] == "demo-eval"
    assert state["completion_promise"] == COMPLETION_PROMISE
    assert state["workspace_path"] == str(ws.root)


def test_write_rejects_non_workspace(tmp_path: Path):
    result = write_ralph_prompt(tmp_path / "nope")
    assert result["error"] == "not_a_workspace"


def test_write_resolves_current_pointer(tmp_path: Path):
    """Passing the `current` iteration pointer resolves back to the root."""
    ws = _make_workspace(tmp_path)
    result = write_ralph_prompt(ws.root / "current")
    assert result["status"] == "ok"
    assert result["workspace_path"] == str(ws.root.resolve())


def test_cli_write_ralph_prompt_round_trip(tmp_path: Path, monkeypatch, capsys):
    ws = _make_workspace(tmp_path)
    monkeypatch.setattr(
        "sys.stdin", io.StringIO(json.dumps({"workspace_path": str(ws.root)}))
    )
    code = main(["write-ralph-prompt"])
    result = json.loads(capsys.readouterr().out)

    assert code == 0
    assert result["status"] == "ok"
    assert (ws.root / ".claude" / "ralph-prompt.txt").is_file()


def test_cli_write_ralph_prompt_missing_path(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
    code = main(["write-ralph-prompt"])
    result = json.loads(capsys.readouterr().out)

    assert code == 1
    assert result["error"] == "missing_workspace_path"


def test_stage6_prompt_advertises_phase_c2_gate():
    """Phase C-2 regression guard: the Stage 6 prompt must reference the
    auto-validate prepare/finalize CLI subcommands, the era-auto-validator
    sub-agent, and the all-fail auto-revise reason. Reverting any of these
    would silently restore the v0.1.6 "no pass/recall gate" behaviour."""
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "docs" / "prompts" / "stage6_experiment.md").read_text(
        encoding="utf-8")
    assert "auto-validate-prepare" in text
    assert "auto-validate-finalize" in text
    assert "era-auto-validator" in text
    assert "stage7_auto_validate_failed" in text
    # Phase C-2 second authorized skip path must also be documented.
    assert "auto_validate_skips" in text


def test_stage6_prompt_advertises_phase_d1_parallel_dispatch():
    """Phase D-1 regression guard: the Stage 6 behavioral prompt must
    reference `wait-for-any-done` (the event-driven completion primitive)
    and the background-runner launch pattern. If a future edit drops
    these, Stage 6 would silently revert to the serial 1-GPU dispatch this
    phase fixed."""
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "docs" / "prompts" / "stage6_experiment.md").read_text(
        encoding="utf-8")
    assert "wait-for-any-done" in text, (
        "Stage 6 prompt no longer mentions wait-for-any-done — parallel "
        "dispatch will revert to fixed-interval polling."
    )
    # The parallel-launch pattern: `nohup ... &` (background bash jobs).
    # Allow either spelling in case the prompt is reworded later.
    assert ("nohup" in text) or ("&\n" in text and "CUDA_VISIBLE_DEVICES" in text), (
        "Stage 6 prompt no longer documents background-runner dispatch."
    )


def test_ralph_loop_has_no_prestage8_block_sites(tmp_path: Path):
    """Phase C-1 regression guard: pre-Stage-8 failures must auto-revise, not
    block. Allow ``run_state: blocked`` only in semantically legitimate sites
    (bootstrap detection, Stage 8 handoff, Stage 9 self-failure fallback, the
    completion paragraph). Any new occurrence of the literal in a pre-Stage-8
    block paragraph would re-introduce the v0.1.6 silent-scope-reduction
    failure mode this phase closed."""
    ws = _make_workspace(tmp_path)
    text = compile_ralph_prompt(ws.root)

    # The literal must appear; we just bound *how many* times.
    occurrences = text.count("run_state: blocked")
    # Allowed sites (counted approximately, ceiling rather than exact):
    #   - Autonomy bullet explaining the policy
    #   - Bootstrap step 2 (terminal-state detection)
    #   - Stage 8 awaiting_human / stopped / blocked semantics (3 mentions)
    #   - Stage 9 react self-failure fallback (1)
    #   - Completion paragraph explaining the new behaviour
    assert occurrences <= 8, (
        f"ralph_loop.md has too many 'run_state: blocked' sites "
        f"({occurrences}); did a pre-Stage-8 block site get re-added?"
    )

    # And the auto-revise CLI must be wired into the prompt — it replaced
    # the old pre-Stage-8 blocks.
    assert "auto-revise" in text


def _flatten_ws(text: str) -> str:
    """Collapse runs of whitespace to single spaces so prompt assertions
    don't break on line-wrap differences."""
    import re as _re
    return _re.sub(r"\s+", " ", text)


def test_ralph_loop_forbids_completion_promise_at_revise_skip(tmp_path: Path):
    """Autonomy-rule regression guard: the Stage 9 REVISE_* branch must
    explicitly forbid the completion promise, otherwise the agent emits
    ``ERA_PIPELINE_COMPLETE`` on every iter transition and forces the
    operator to manually ``/era:resume``. This is the exact failure
    signature operators observed after iter_001 closed cleanly with
    REVISE_SKIP_STAGE1."""
    ws = _make_workspace(tmp_path)
    text = _flatten_ws(compile_ralph_prompt(ws.root))
    # The exact forbidden-promise sentence must appear in the prompt.
    # It is unique to the Stage 9 REVISE_* sub-bullet (no other site
    # tells the agent to NOT output the promise at REVISE_*).
    assert "Do NOT output `<promise>ERA_PIPELINE_COMPLETE</promise>` here." in text, (
        "Stage 9 REVISE_* branch must contain the literal "
        "'Do NOT output `<promise>ERA_PIPELINE_COMPLETE</promise>` here.' "
        "instruction so the agent never emits the promise at iter "
        "transitions"
    )
    # And the justification text must be present so future model
    # versions read this as structural, not stylistic.
    assert "violating the iron autonomy rule" in text


def test_ralph_loop_forbids_completion_promise_between_stages(tmp_path: Path):
    """The 'after any successful stage' instruction must NOT leave 'end
    the iteration' ambiguous — the agent reads that as 'emit the
    completion promise.' The tightened wording must spell out the
    no-emit rule plus the disambiguation between 'end your turn' and
    'pipeline complete.'"""
    ws = _make_workspace(tmp_path)
    text = _flatten_ws(compile_ralph_prompt(ws.root))
    # Per-stage conclusion must say "end your current turn silently"
    # (whitespace-collapsed so line wraps don't break the assertion).
    assert "end your current turn silently" in text, (
        "Per-stage conclusion must say 'end your current turn silently'"
        " to disambiguate from 'emit the completion promise'"
    )
    # And must explain that 'end your turn' means stop emitting tokens.
    assert "stop emitting tokens" in text
    # And must explicitly say "Do NOT output" with the promise name.
    assert "do NOT output `<promise>ERA_PIPELINE_COMPLETE</promise>`" in text, (
        "Per-stage conclusion must explicitly forbid the promise — "
        "agent has shown a tendency to emit it after routine stage "
        "transitions otherwise"
    )


def test_ralph_loop_has_bug_shapes_warning_section():
    """The Completion section must enumerate the false-emit signature
    operators have observed, so future Claude model versions read it
    as a structural anti-pattern rather than a soft preference."""
    repo = Path(__file__).resolve().parents[1]
    text = _flatten_ws(
        (repo / "docs" / "prompts" / "ralph_loop.md").read_text(
            encoding="utf-8"
        )
    )
    assert "Bug shapes to avoid" in text, (
        "ralph_loop.md must contain a 'Bug shapes to avoid' subsection"
    )
    assert "After Stage 9 REVISE" in text, (
        "'Bug shapes to avoid' must name the Stage 9 REVISE_* false-emit"
    )
    assert "After a successful Stage 1-7" in text, (
        "'Bug shapes to avoid' must name the between-stages false-emit"
    )
    # The exact false-emit framing the agent kept producing.
    assert "Resume with `/era:resume`" in text


def test_ralph_loop_manual_fallback_drives_loop_in_context(tmp_path: Path):
    """Manual-fallback paragraph must say 'drive the loop in-context'
    + 'never emit between stages or after Stage 9 REVISE_*'. Without
    these, the manual-fallback agent also emits the promise after
    every successful stage."""
    ws = _make_workspace(tmp_path)
    text = _flatten_ws(compile_ralph_prompt(ws.root))
    # Must explicitly tell the agent to drive the loop in-context, not
    # rely on a Stop hook.
    assert "Drive the loop in-context" in text, (
        "Manual-fallback section must say 'Drive the loop in-context'"
    )
    # Must explicitly forbid emission between stages and after REVISE_*.
    assert (
        "Never emit the promise between stages or after Stage 9 REVISE"
    ) in text, (
        "Manual-fallback section must explicitly forbid emission "
        "between stages and after Stage 9 REVISE_*"
    )


def test_ensure_claude_settings_enables_ralph_loop_and_mcp(tmp_path: Path):
    settings_path = ensure_claude_settings(tmp_path)
    assert settings_path == tmp_path / ".claude" / "settings.json"
    assert settings_path.is_file()

    data = json.loads(settings_path.read_text(encoding="utf-8"))
    assert data["enabledPlugins"][RALPH_LOOP_PLUGIN] is True
    assert "claude-plugins-official" in data["extraKnownMarketplaces"]
    assert data["enableAllProjectMcpServers"] is True


def test_ensure_claude_settings_preserves_operator_enabled_plugins(tmp_path: Path):
    """Phase D-2 merge semantics: an existing settings.json with the operator's
    own enabledPlugins value is preserved (don't fight a deliberate config).
    But the missing keys (permissions, etc.) are added on the merge."""
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    (claude_dir / "settings.json").write_text(
        '{"enabledPlugins": {"other-plugin@mkt": true}}\n',
        encoding="utf-8",
    )

    ensure_claude_settings(tmp_path)
    data = json.loads(
        (claude_dir / "settings.json").read_text(encoding="utf-8")
    )
    # Operator's enabledPlugins survives untouched.
    assert data["enabledPlugins"] == {"other-plugin@mkt": True}
    # ERA-managed keys that were absent got added.
    assert "permissions" in data
    assert "enableAllProjectMcpServers" in data


def test_write_ralph_prompt_wires_plugin_and_mcp(tmp_path: Path):
    """write_ralph_prompt leaves the workspace plugin-enabled and MCP-registered."""
    ws = _make_workspace(tmp_path)
    result = write_ralph_prompt(ws.root)

    settings = json.loads(
        (ws.root / ".claude" / "settings.json").read_text(encoding="utf-8")
    )
    assert settings["enabledPlugins"][RALPH_LOOP_PLUGIN] is True
    assert (ws.root / ".mcp.json").is_file()
    assert result["mcp_path"] == str(ws.root / ".mcp.json")


# ---- Phase C-2.2 prompt regression guards -------------------------------

def test_stage2_prompt_consumes_annotation_evidence():
    """Stage 2 brainstorm fetches list-annotations and asks personas to
    target the operator's flagged failure modes."""
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "docs" / "prompts" / "stage2_brainstorm.md").read_text(
        encoding="utf-8")
    assert "list-annotations" in text
    assert "use_annotation_evidence" in text
    assert "failure mode" in text.lower()
    assert "hypothesis_id" in text


def test_stage3_prompt_adds_annotation_coverage_dimension():
    """Stage 3 rigor-critic scores annotation coverage."""
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "docs" / "prompts" / "stage3_review.md").read_text(
        encoding="utf-8")
    assert "list-annotations" in text
    assert "Annotation coverage" in text
    assert "use_annotation_evidence" in text


def test_stage4_prompt_biases_chosen_configs_by_coverage():
    """Stage 4 synthesizer biases chosen_configs by failure-mode coverage."""
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "docs" / "prompts" / "stage4_decision.md").read_text(
        encoding="utf-8")
    assert "list-annotations" in text
    assert ("Annotation-coverage bias" in text
            or "annotation coverage" in text.lower())


def test_stage5_prompt_pins_judge_canonical_contract():
    """Stage 5 prompt explicitly forbids judge-name suffix decorations like
    '-pointwise' on the eval side."""
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "docs" / "prompts" / "stage5_experiment_plan.md").read_text(
        encoding="utf-8")
    assert "byte-equal" in text or "string-equal" in text
    assert "-pointwise" in text


def test_stage5_prompt_advertises_phase_c23_sample_window():
    """Phase C-2.3: Stage 5 prompt fetches era.cli sample-window and
    stamps samples_subset on every full-mode eval task."""
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "docs" / "prompts" / "stage5_experiment_plan.md").read_text(
        encoding="utf-8")
    assert "sample-window" in text
    assert "samples_subset" in text
    # The full-mode bullet must explicitly say samples_subset is stamped.
    assert "random N" in text or "shuffled subset" in text


def test_experiment_protocol_samples_subset_primary_rule():
    """Phase C-2.3: runner protocol §5 makes samples_subset the primary
    selection rule for all modes (fallback is first-N only when absent)."""
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "docs" / "prompts" / "_experiment_protocol.md").read_text(
        encoding="utf-8")
    assert "samples_subset" in text
    # The new wording should mark samples_subset as the primary/authoritative rule.
    assert ("authoritative when set" in text
            or "primary selection rule" in text
            or "Primary rule" in text)


# ---- Phase D-2: permissions allowlist + denylist ------------------------

def test_era_claude_settings_includes_permissions_allow():
    """The settings payload carries an `permissions.allow` block covering
    the patterns ERA's ralph-loop actually uses."""
    from era.orchestration.ralph import era_claude_settings
    payload = era_claude_settings()
    assert "permissions" in payload
    allow = payload["permissions"].get("allow") or []
    # Key patterns that MUST be allowed for an unattended run.
    assert any("era.cli" in p for p in allow)
    assert any("nvidia-smi" in p for p in allow)
    assert any("nohup" in p for p in allow)
    assert "Task(era:*)" in allow
    assert any("Write(iter_" in p for p in allow)


def test_era_claude_settings_allows_gpu_reset():
    """Phase D-4: the allowlist must cover `sudo nvidia-smi --gpu-reset`
    (long form) and `sudo nvidia-smi -r` (short form) so the autonomous
    loop can auto-clear zombie GPUs without an operator prompt. Anything
    else under `sudo nvidia-smi` is intentionally NOT auto-allowed."""
    from era.orchestration.ralph import era_claude_settings
    allow = era_claude_settings()["permissions"]["allow"]
    assert "Bash(sudo nvidia-smi --gpu-reset*)" in allow
    assert "Bash(sudo nvidia-smi -r*)" in allow
    # Negative check: persistence-mode / ECC / MIG paths must NOT have
    # been auto-allowed by accident.
    assert "Bash(sudo nvidia-smi --persistence-mode*)" not in allow
    assert "Bash(sudo nvidia-smi*)" not in allow  # too broad


def test_era_claude_settings_allows_watchdog_pkill():
    """Phase D-4 extension: the GPU watchdog cleanup commands documented
    in CLAUDE.md's ``## GPU environment`` section must be in the
    allowlist. Without these, the ralph-loop agent gets a permission
    prompt and the iron autonomy rule breaks during Stage 6 when a card
    fails to auto-release from NoGPUAlarmNew.py.

    The patterns are scoped to the watchdog process name (the literal
    invocation CLAUDE.md documents). Blanket ``Bash(pkill *)`` /
    ``Bash(sudo pkill *)`` are intentionally NOT auto-allowed — they
    would let the agent kill arbitrary processes.
    """
    from era.orchestration.ralph import era_claude_settings
    allow = era_claude_settings()["permissions"]["allow"]
    # Both literal forms documented in CLAUDE.md must be pre-approved.
    assert 'Bash(pkill -9 -f "python3 -u NoGPUAlarmNew.py")' in allow
    assert 'Bash(sudo pkill -9 -f "python3 -u NoGPUAlarmNew.py")' in allow
    # Negative checks: blanket pkill / sudo pkill are too broad and must
    # NOT have been auto-allowed by accident.
    assert "Bash(pkill *)" not in allow
    assert "Bash(sudo pkill *)" not in allow
    assert "Bash(sudo pkill -9 *)" not in allow


def test_era_claude_settings_includes_permissions_deny():
    """Catastrophic Bash patterns are explicitly denied even if a local
    override tries to allow them."""
    from era.orchestration.ralph import era_claude_settings
    payload = era_claude_settings()
    deny = payload["permissions"].get("deny") or []
    assert any("git push --force" in p for p in deny)
    assert any("git push -f" in p for p in deny)
    assert any("git config --global" in p for p in deny)
    assert any("rm -rf /" in p for p in deny)


def test_ensure_settings_merges_permissions_into_partial_file(tmp_path: Path):
    """An existing settings.json with NO permissions block gets the ERA
    defaults merged in on next ensure_claude_settings call (Phase D-2
    upgrade path for workspaces scaffolded pre-D-2)."""
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    (claude_dir / "settings.json").write_text(
        json.dumps({"enabledPlugins": {RALPH_LOOP_PLUGIN: True}}),
        encoding="utf-8",
    )
    ensure_claude_settings(tmp_path)
    data = json.loads(
        (claude_dir / "settings.json").read_text(encoding="utf-8")
    )
    assert "permissions" in data
    assert data["permissions"]["allow"]
    assert data["permissions"]["deny"]


def test_ensure_settings_unions_existing_allowlist(tmp_path: Path):
    """If the operator already pinned their own permissions.allow patterns,
    the merge takes the UNION — operator entries preserved, ERA defaults
    added."""
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    (claude_dir / "settings.json").write_text(
        json.dumps({
            "permissions": {
                "allow": ["Bash(my-custom-tool*)"],
                "deny": [],
            },
        }),
        encoding="utf-8",
    )
    ensure_claude_settings(tmp_path)
    data = json.loads(
        (claude_dir / "settings.json").read_text(encoding="utf-8")
    )
    allow = data["permissions"]["allow"]
    # Operator's pattern survived.
    assert "Bash(my-custom-tool*)" in allow
    # ERA's defaults are present too.
    assert "Task(era:*)" in allow
    assert any("nvidia-smi" in p for p in allow)


def test_ensure_settings_preserves_operator_unmanaged_keys(tmp_path: Path):
    """Operator-added top-level keys (model, etc.) survive the merge
    untouched. Operator-added hooks for OTHER matchers (e.g. Bash) also
    survive; the only hook ERA now manages is the AskUserQuestion gate
    (Phase D-3), which gets unioned in alongside the operator's
    entries."""
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    operator_bash_hook = {"matcher": "Bash", "hooks": []}
    (claude_dir / "settings.json").write_text(
        json.dumps({
            "hooks": {"PreToolUse": [operator_bash_hook]},
            "model": "claude-sonnet-4-6",
        }),
        encoding="utf-8",
    )
    ensure_claude_settings(tmp_path)
    data = json.loads(
        (claude_dir / "settings.json").read_text(encoding="utf-8")
    )
    # Operator's Bash hook survived.
    assert operator_bash_hook in data["hooks"]["PreToolUse"]
    # Operator's top-level model key survived verbatim.
    assert data["model"] == "claude-sonnet-4-6"
    # ERA-managed keys got added.
    assert "permissions" in data
    assert "enabledPlugins" in data
    # ERA's AskUserQuestion gate was unioned in.
    matchers = {e.get("matcher") for e in data["hooks"]["PreToolUse"]}
    assert "AskUserQuestion" in matchers


def test_ensure_settings_recovers_from_malformed_existing(tmp_path: Path):
    """A malformed settings.json (unparseable JSON) is treated as if absent
    — the file gets overwritten with fresh ERA defaults instead of crashing
    the autonomous loop."""
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    (claude_dir / "settings.json").write_text(
        "{ not valid json", encoding="utf-8",
    )
    ensure_claude_settings(tmp_path)
    data = json.loads(
        (claude_dir / "settings.json").read_text(encoding="utf-8")
    )
    assert data["enabledPlugins"][RALPH_LOOP_PLUGIN] is True
    assert "permissions" in data


# ---- Phase D-2 extension: repo-root settings inheritance --------------

def _scaffold_repo_root_with_pin(
    base: Path, *, allow_pin: str = "Bash(echo operator-pin)",
) -> Path:
    """Build a fake ERA repo root carrying a custom ``permissions.allow``
    pin in its ``.claude/settings.json``. Returns the fake repo path."""
    fake_repo = base / "fake-era-repo"
    fake_repo_claude = fake_repo / ".claude"
    fake_repo_claude.mkdir(parents=True)
    (fake_repo_claude / "settings.json").write_text(
        json.dumps({
            "permissions": {"allow": [allow_pin]},
        }),
        encoding="utf-8",
    )
    return fake_repo


def test_ensure_settings_inherits_repo_root_allow_pins(tmp_path: Path):
    """Phase D-2 extension: an operator pin in the repo-root
    ``.claude/settings.json`` flows into every scaffolded workspace's
    ``.claude/settings.json`` automatically — operators no longer have
    to edit ``ERA_ALLOW_PATTERNS`` to add a repo-wide custom pattern."""
    fake_repo = _scaffold_repo_root_with_pin(tmp_path)
    ws_root = tmp_path / "ws"
    ws_root.mkdir()
    ensure_claude_settings(ws_root, repo_root_path=fake_repo)
    data = json.loads(
        (ws_root / ".claude" / "settings.json").read_text(encoding="utf-8")
    )
    allow = data["permissions"]["allow"]
    # ERA defaults still present.
    assert any("era.cli" in p for p in allow)
    # Operator's repo-root pin landed in the workspace.
    assert "Bash(echo operator-pin)" in allow


def test_ensure_settings_does_not_recurse_on_repo_root_itself(tmp_path: Path):
    """When ``root`` IS the repo root (operator runs Claude from the repo
    root, not a workspace), we don't recursively self-merge — the file
    on disk gets ERA defaults applied as if no repo-root layer existed."""
    fake_repo = _scaffold_repo_root_with_pin(tmp_path)
    # The pin IS in fake_repo/.claude/settings.json. Calling
    # ensure_claude_settings(fake_repo, ...) should leave the pin
    # intact (it's already there) but should NOT cause an infinite
    # loop or double-application.
    before_size = (fake_repo / ".claude" / "settings.json").stat().st_size
    ensure_claude_settings(fake_repo, repo_root_path=fake_repo)
    data = json.loads(
        (fake_repo / ".claude" / "settings.json").read_text(encoding="utf-8")
    )
    # ERA defaults are now layered onto the operator's pre-existing
    # pin: both should be in the allow list, deduped.
    allow = data["permissions"]["allow"]
    assert allow.count("Bash(echo operator-pin)") == 1
    assert any("era.cli" in p for p in allow)
    # No infinite loop. The file is bounded (no doubled content).
    after_size = (fake_repo / ".claude" / "settings.json").stat().st_size
    assert after_size < before_size + 10000  # sane bound


def test_ensure_settings_handles_missing_repo_root_settings(tmp_path: Path):
    """When the repo-root ``.claude/settings.json`` is absent, the
    workspace still gets the ERA defaults — no crash, no spurious
    behavior."""
    empty_fake_repo = tmp_path / "empty-fake-repo"
    empty_fake_repo.mkdir()
    # No .claude/ directory at all.
    ws_root = tmp_path / "ws"
    ws_root.mkdir()
    ensure_claude_settings(ws_root, repo_root_path=empty_fake_repo)
    data = json.loads(
        (ws_root / ".claude" / "settings.json").read_text(encoding="utf-8")
    )
    assert data["enabledPlugins"][RALPH_LOOP_PLUGIN] is True
    assert "permissions" in data
    assert any("era.cli" in p for p in data["permissions"]["allow"])


def test_ensure_settings_unions_repo_root_pin_with_existing_workspace(
    tmp_path: Path,
):
    """Three-way union: ERA defaults + existing workspace operator pins
    + repo-root operator pins — all survive in the workspace allow list."""
    fake_repo = _scaffold_repo_root_with_pin(
        tmp_path, allow_pin="Bash(echo repo-root-pin)",
    )
    ws_root = tmp_path / "ws"
    claude_dir = ws_root / ".claude"
    claude_dir.mkdir(parents=True)
    # Operator's existing per-workspace pin.
    (claude_dir / "settings.json").write_text(
        json.dumps({
            "permissions": {"allow": ["Bash(echo workspace-pin)"]},
        }),
        encoding="utf-8",
    )
    ensure_claude_settings(ws_root, repo_root_path=fake_repo)
    data = json.loads(
        (claude_dir / "settings.json").read_text(encoding="utf-8")
    )
    allow = data["permissions"]["allow"]
    assert "Bash(echo workspace-pin)" in allow   # existing ws pin
    assert "Bash(echo repo-root-pin)" in allow   # repo-root pin
    assert any("era.cli" in p for p in allow)     # ERA defaults


# ---- Phase C-2.4: Stage 6 fail-loud regression --------------------------

def test_stage6_prompt_has_annotated_preflight_and_failloud():
    """Phase C-2.4: Stage 6 prompt MUST do a pre-flight check that
    annotated tasks exist before running the gate, AND it MUST branch on
    the new auto_validate error 'no_annotated_scores'. Reverting either
    re-opens the v0.1.6 false-negative the user hit live."""
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "docs" / "prompts" / "stage6_experiment.md").read_text(
        encoding="utf-8")
    # Pre-flight + new auto-revise reason.
    assert "stage5_missing_annotated_tasks" in text
    # Fail-loud error name from auto_validate.
    assert "no_annotated_scores" in text
    # Both branch outcomes documented.
    assert "no_annotated_tasks_in_plan" in text
    assert "annotated_round_didnt_run" in text


def test_stage9_prompt_handles_new_stage5_missing_annotated_reason():
    """Phase C-2.4: Stage 9 advisor must know how to handle the new
    stage5_missing_annotated_tasks reason — it's a planner drift, NOT a
    candidate-evaluator failure, so the advisor should add it to
    general_failure_modes and NOT exclude the configs."""
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "docs" / "prompts" / "stage9_react.md").read_text(
        encoding="utf-8")
    assert "stage5_missing_annotated_tasks" in text
    assert "stage5_planner_omitted_annotated_tasks" in text


# ---- Phase D-3: PreToolUse autonomy hook --------------------------------

def test_era_claude_settings_includes_autonomy_hook():
    """Phase D-3: era_claude_settings ships a PreToolUse hook that blocks
    AskUserQuestion calls (era.cli check-autonomy gate)."""
    from era.orchestration.ralph import era_claude_settings
    payload = era_claude_settings()
    hooks = payload.get("hooks", {})
    pre = hooks.get("PreToolUse", [])
    assert pre, "PreToolUse hook missing"
    matchers = {e.get("matcher") for e in pre if isinstance(e, dict)}
    assert "AskUserQuestion" in matchers
    # The matcher's command list should invoke era.cli check-autonomy.
    aq_entry = next(e for e in pre if e.get("matcher") == "AskUserQuestion")
    cmds = [h.get("command", "") for h in aq_entry.get("hooks", [])]
    assert any("era.cli check-autonomy" in c for c in cmds)


def test_ensure_settings_merges_hook_into_existing_file(tmp_path: Path):
    """Phase D-3: an existing settings.json without hooks gets the
    AskUserQuestion gate added on next ensure_claude_settings call.
    This is the auto-upgrade path for workspaces scaffolded pre-D-3."""
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    (claude_dir / "settings.json").write_text(
        json.dumps({"enabledPlugins": {RALPH_LOOP_PLUGIN: True}}),
        encoding="utf-8",
    )
    ensure_claude_settings(tmp_path)
    data = json.loads(
        (claude_dir / "settings.json").read_text(encoding="utf-8")
    )
    assert "hooks" in data
    matchers = {
        e.get("matcher") for e in data["hooks"].get("PreToolUse", [])
    }
    assert "AskUserQuestion" in matchers


def test_ensure_settings_preserves_operator_hooks(tmp_path: Path):
    """An operator-added PreToolUse hook for a DIFFERENT matcher must
    survive the merge — ERA only manages the AskUserQuestion entry."""
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    operator_hook = {
        "matcher": "Bash",
        "hooks": [{"type": "command", "command": "my-custom-audit"}],
    }
    (claude_dir / "settings.json").write_text(
        json.dumps({
            "hooks": {"PreToolUse": [operator_hook]},
        }),
        encoding="utf-8",
    )
    ensure_claude_settings(tmp_path)
    data = json.loads(
        (claude_dir / "settings.json").read_text(encoding="utf-8")
    )
    pre = data["hooks"]["PreToolUse"]
    matchers = {e.get("matcher") for e in pre}
    # Operator's Bash matcher survived.
    assert "Bash" in matchers
    # ERA's AskUserQuestion matcher was added.
    assert "AskUserQuestion" in matchers


def test_ensure_settings_does_not_duplicate_existing_ask_user_matcher(
    tmp_path: Path,
):
    """If the operator already pinned their own AskUserQuestion hook
    (or an old ERA install did), don't append a duplicate."""
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    operator_aq = {
        "matcher": "AskUserQuestion",
        "hooks": [{"type": "command", "command": "my-own-gate"}],
    }
    (claude_dir / "settings.json").write_text(
        json.dumps({
            "hooks": {"PreToolUse": [operator_aq]},
        }),
        encoding="utf-8",
    )
    ensure_claude_settings(tmp_path)
    data = json.loads(
        (claude_dir / "settings.json").read_text(encoding="utf-8")
    )
    pre = data["hooks"]["PreToolUse"]
    # Exactly one AskUserQuestion entry — no duplicate appended.
    aq_entries = [e for e in pre if e.get("matcher") == "AskUserQuestion"]
    assert len(aq_entries) == 1
    # The operator's hook command was preserved.
    assert any(
        h.get("command") == "my-own-gate"
        for h in aq_entries[0].get("hooks", [])
    )


def test_ralph_loop_prompt_warns_against_stage6_permission_prompts(
    tmp_path: Path,
):
    """Phase D-3 prompt regression guard: the ralph-loop autonomy block
    must name Stage 6, mention 'several hours', and call out the
    'answer is always continue' anti-pattern. Reverting these would
    re-open the v0.1.7.x bug where the agent prompted 'Stage 6 will
    run for several hours, how should we proceed?'."""
    ws = _make_workspace(tmp_path)
    text = compile_ralph_prompt(ws.root)
    assert "Stage 6" in text
    assert "several hours" in text
    assert "the answer is always continue" in text.lower()
    # Also: the new hook is referenced.
    assert "PreToolUse" in text or "check-autonomy" in text


# ---- Phase D-5: safe vLLM teardown ---------------------------------------

def test_stage6_prompt_calls_shutdown_judge_not_raw_kill():
    """Phase D-5: Stage 6 teardown must use era.cli shutdown-judge
    (graceful sequence), not raw `kill` (which leaks tp-workers and
    wedges GPUs)."""
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "docs" / "prompts" / "stage6_experiment.md").read_text(
        encoding="utf-8")
    assert "shutdown-judge" in text
    # The teardown branch must enumerate the four status values.
    assert "escalated_kill" in text
    assert "escalated_reset" in text
    assert "still_stuck" in text
    # Old "kill the judge PID" raw pattern must NOT appear as an
    # instruction (allow it only in negation contexts).
    for line in text.splitlines():
        if "kill the judge PID" not in line:
            continue
        if any(neg in line.lower() for neg in
               ("not", "forbidden", "instead of", "no longer")):
            continue
        raise AssertionError(
            f"Stage 6 prompt still tells the agent to 'kill the judge PID' "
            f"directly — must use shutdown-judge instead. Line: {line!r}"
        )


def test_experiment_protocol_serve_uses_start_new_session():
    """Phase D-5: the serve runner template must launch vLLM with
    start_new_session=True (or setsid) so the parent + tp-workers
    share one pgid — required for shutdown-judge's pgid SIGTERM."""
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "docs" / "prompts" / "_experiment_protocol.md").read_text(
        encoding="utf-8")
    assert "start_new_session" in text or "setsid" in text
    # Must mention process group / pgid since that's the whole point.
    assert "process group" in text.lower() or "pgid" in text


# ---- Phase C-2.5 — min_passing + EA reflection prompt regressions -----

def test_stage6_prompt_surfaces_passing_configs_diagnostic():
    """Phase C-2.5: the Stage 6 prompt's 4.5e branch must emit the
    passing_configs / failing_configs in the auto-revise diagnostic
    so Stage 9 can differentiate partial-pass from all-fail."""
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "docs" / "prompts" / "stage6_experiment.md").read_text(
        encoding="utf-8")
    assert "passing_count" in text
    assert "min_passing" in text
    assert "passing_configs" in text
    # The log note for any_passed:true must show the X/Y count + min.
    assert "passed C-2 gate" in text


def test_stage9_prompt_handles_partial_pass_vs_all_fail():
    """Phase C-2.5: the Stage 9 advisor brief must distinguish
    partial-pass (carry-forward must_include_configs) from all-fail
    (exclude everything)."""
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "docs" / "prompts" / "stage9_react.md").read_text(
        encoding="utf-8")
    assert "Partial-pass" in text or "partial-pass" in text
    assert "All-fail" in text or "all-fail" in text
    assert "must_include_configs" in text
    # The natural-language digest path must be named so Stage 2 can
    # find it.
    assert "lessons.md" in text
    # And the EA reflection deliverables must be named explicitly.
    assert "lessons_learned" in text
    assert "hall_of_fame" in text


def test_stage2_prompt_reads_must_include_and_ea_memory():
    """Phase C-2.5: Stage 2's brainstorm must consume
    must_include_configs (elitism), lessons_learned (priors +
    anti-patterns), and hall_of_fame (population memory)."""
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "docs" / "prompts" / "stage2_brainstorm.md").read_text(
        encoding="utf-8")
    assert "must_include_configs" in text
    assert "lessons_learned" in text
    assert "hall_of_fame" in text


def test_stage0_prompt_pins_min_passing_with_pass_recall_presets():
    """Phase C-2.5: the auto-validate preset question must name the
    new M-threshold value in each preset."""
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "docs" / "prompts" / "stage0_init.md").read_text(
        encoding="utf-8")
    assert "MIN-PASSING" in text or "min_passing" in text
    assert "auto_validate_min_passing" in text


def test_react_advisor_describes_ea_primitives():
    """Phase C-2.5: era-react-advisor.md must describe
    must_include_configs (elitism), hall_of_fame (population memory),
    and lessons_learned (reflection) — the three EA primitives."""
    repo_root = Path(__file__).resolve().parents[1]
    text = (
        repo_root / "plugin" / "agents" / "era-react-advisor.md"
    ).read_text(encoding="utf-8")
    assert "must_include_configs" in text
    assert "hall_of_fame" in text
    assert "lessons_learned" in text
    # Fitness composite formula keeps the advisor consistent across
    # iters.
    assert "fitness_composite" in text


def test_stage4_decision_brief_passes_evolution_state_path():
    """Phase C-2.5: Stage 4's check-experiment-brief invocation must
    pass evolution_state_path so the must_include_configs gate
    enforces elitism on the new brief."""
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "docs" / "prompts" / "stage4_decision.md").read_text(
        encoding="utf-8")
    assert "evolution_state_path" in text


# ---- Phase C-2.5 — VLM ≥30B floor prompt regressions ------------------

def test_vlm_min_size_floor_in_brainstorm_and_decision():
    """The ≥30B VLM floor (drop 7B framing) lives in Stage 2's
    judge-advocate persona brief AND Stage 4's brief gate."""
    repo_root = Path(__file__).resolve().parents[1]
    brainstorm = (
        repo_root / "docs" / "prompts" / "stage2_brainstorm.md"
    ).read_text(encoding="utf-8")
    decision = (
        repo_root / "docs" / "prompts" / "stage4_decision.md"
    ).read_text(encoding="utf-8")
    for blob in (brainstorm, decision):
        assert "30B" in blob or "≥30B" in blob
        assert "Qwen3.6-35B-A3B" in blob or "Gemma-4-31B-it" in blob


def test_react_advisor_model_upsize_drops_7b_rung():
    """era-react-advisor.md's model_upsize ladder must NOT start at
    7B — Phase C-2.5 forbids sub-30B unless the operator explicitly
    asked for it in human feedback."""
    repo_root = Path(__file__).resolve().parents[1]
    text = (
        repo_root / "plugin" / "agents" / "era-react-advisor.md"
    ).read_text(encoding="utf-8")
    # The ladder line must mention the 30B floor.
    assert "≥30B" in text or "30B class" in text
    # And the forbidden-without-operator-request clause must be present.
    assert "Sub-30B" in text and "forbidden" in text
