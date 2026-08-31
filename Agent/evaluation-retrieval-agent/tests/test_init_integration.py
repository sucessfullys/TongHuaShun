"""End-to-end tests for cli_init_project (the init-workspace entrypoint)."""

from __future__ import annotations

import json
from pathlib import Path

from conftest import valid_params

from era.config import ERAConfig
from era.orchestration.project_cli import cli_init_project
from era.paths import repo_root


def test_init_project_success(tmp_path: Path, per_sample_root: Path):
    params = valid_params(tmp_path, per_sample_root)
    result = cli_init_project(
        params, repo_root=repo_root(), workspaces_dir=tmp_path / "workspaces"
    )

    assert "error" not in result
    assert result["project_name"] == "tryon-eval"
    ws_root = Path(result["workspace_path"])

    # global Stage 0 files
    for name in ("config.yaml", "spec.md", "status.json", "CLAUDE.md",
                 ".gitignore"):
        assert (ws_root / name).is_file(), name

    # probe artifacts
    for name in ("gpu_inventory.json", "data_layout.json",
                 "checkpoints.json", "credentials.json"):
        assert (ws_root / "probe" / name).is_file(), name

    # iteration-aware layout
    assert (ws_root / "iter_001" / "iteration.json").is_file()
    assert (ws_root / "iter_001" / "design" / "candidates").is_dir()
    current = ws_root / "current"
    assert current.is_symlink() or (ws_root / "current.txt").exists()

    # status
    import json
    status = json.loads((ws_root / "status.json").read_text(encoding="utf-8"))
    assert status["stage"] == "task_init"
    assert status["iteration"] == 1
    assert status["run_state"] == "idle"

    # the ralph-loop plugin is enabled for the scaffolded workspace
    settings = json.loads(
        (ws_root / ".claude" / "settings.json").read_text(encoding="utf-8")
    )
    assert settings["enabledPlugins"]["ralph-loop@claude-plugins-official"] is True
    assert settings["enableAllProjectMcpServers"] is True

    # the arXiv MCP server is registered for the scaffolded workspace
    mcp = json.loads((ws_root / ".mcp.json").read_text(encoding="utf-8"))
    assert "arxiv-mcp-server" in mcp["mcpServers"]

    # config.yaml re-parses to an equal config
    reloaded = ERAConfig.from_yaml(ws_root / "config.yaml")
    assert reloaded.project_name == "tryon-eval"
    assert reloaded.validate() == []

    # guide is non-empty and names the next command
    assert "/era:start" in result["guide"]


def test_init_project_collision(tmp_path: Path, per_sample_root: Path):
    params = valid_params(tmp_path, per_sample_root)
    workspaces = tmp_path / "workspaces"
    first = cli_init_project(
        params, repo_root=repo_root(), workspaces_dir=workspaces
    )
    assert "error" not in first

    second = cli_init_project(
        params, repo_root=repo_root(), workspaces_dir=workspaces
    )
    assert second["error"] == "workspace_exists"


def test_init_project_invalid_config(tmp_path: Path, per_sample_root: Path):
    params = valid_params(tmp_path, per_sample_root)
    params["data"]["sample_count"] = 0
    result = cli_init_project(
        params, repo_root=repo_root(), workspaces_dir=tmp_path / "workspaces"
    )
    assert result["error"] == "invalid_config"
    assert any("sample_count" in p for p in result["problems"])


def test_init_derives_name_when_absent(tmp_path: Path, per_sample_root: Path):
    params = valid_params(tmp_path, per_sample_root)
    del params["project_name"]
    result = cli_init_project(
        params, repo_root=repo_root(), workspaces_dir=tmp_path / "workspaces"
    )
    assert "error" not in result
    # derived from task_adapter "virtual_tryon"
    assert result["project_name"] == "virtual-tryon-eval"


# The full ## GPU environment section every scaffolded workspace CLAUDE.md
# must carry, pinned inline so a future trim of any paragraph or code block
# trips the test (not only an accidental heading rename). Bytes-for-bytes the
# same body that lives in the repo-root CLAUDE.md.
_GPU_ENVIRONMENT_SECTION = """\
## GPU environment

This machine runs a GPU watchdog, `NoGPUAlarmNew.py` (in
`/mnt/image-edit/datasets/xywang/code/GPU_OCU/`), that holds otherwise-idle
cards. A GPU whose `nvidia-smi` signature looks like:

```
NVIDIA H100 80GB HBM3 | **°C, 100 % | 35061 / 81559 MB
```

— roughly 35 GB used at 100 % utilization — is running **only the watchdog, not
a real job**. The watchdog releases the card automatically as soon as a genuine
GPU task starts, so **treat such a GPU as free** when scheduling experiments.

If the watchdog does not release a card on its own, stop it manually:

```bash
sudo pkill -9 -f "python3 -u NoGPUAlarmNew.py"
```

Both forms (with and without `sudo`) are pre-approved in
`.claude/settings.json`'s `permissions.allow` (Phase D-4 extension), so
the autonomous loop can stop a misbehaving watchdog without an operator
prompt. The patterns are scoped to the watchdog name — blanket
`Bash(pkill *)` / `Bash(sudo pkill *)` are intentionally NOT auto-allowed.

After all experiments in an iteration finish, if `NoGPUAlarmNew.py` was killed
and has not restarted itself, restart it so idle cards stay protected:

```bash
cd /mnt/image-edit/datasets/xywang/code/GPU_OCU/ && bash start.sh
```"""


def test_workspace_claude_md_includes_gpu_environment(
    tmp_path: Path, per_sample_root: Path,
):
    """Every newly scaffolded workspace must inherit the full GPU watchdog
    section so operators know to treat watchdog-held GPUs as free."""
    params = valid_params(tmp_path, per_sample_root)
    result = cli_init_project(
        params, repo_root=repo_root(), workspaces_dir=tmp_path / "workspaces"
    )
    assert "error" not in result
    claude_md = (Path(result["workspace_path"]) / "CLAUDE.md").read_text(
        encoding="utf-8"
    )
    assert _GPU_ENVIRONMENT_SECTION in claude_md


def test_init_writes_iter_sample_count_to_config_yaml(
    tmp_path: Path, per_sample_root: Path,
):
    """The scaffolded config.yaml must carry data.iter_sample_count so the
    operator can see and edit it post-init."""
    params = valid_params(tmp_path, per_sample_root)
    # The valid_params fixture sets iter_sample_count to 3 (matches the
    # 3-sample fixture); confirm it round-trips through the YAML.
    result = cli_init_project(
        params, repo_root=repo_root(), workspaces_dir=tmp_path / "workspaces"
    )
    assert "error" not in result
    config_yaml = (Path(result["workspace_path"]) / "config.yaml").read_text(
        encoding="utf-8"
    )
    assert "iter_sample_count" in config_yaml
    reloaded = ERAConfig.from_yaml(Path(result["config_path"]))
    assert reloaded.data.iter_sample_count == 3
    assert reloaded.effective_iter_sample_count() == 3


def test_init_defaults_iter_sample_count_to_50(
    tmp_path: Path, per_sample_root: Path,
):
    """When the operator omits iter_sample_count, the dataclass default
    (50) is what lands in config.yaml. The 'cap to sample_count' rule then
    yields effective = min(50, sample_count)."""
    params = valid_params(tmp_path, per_sample_root)
    params["data"].pop("iter_sample_count", None)
    result = cli_init_project(
        params, repo_root=repo_root(), workspaces_dir=tmp_path / "workspaces"
    )
    assert "error" not in result
    reloaded = ERAConfig.from_yaml(Path(result["config_path"]))
    assert reloaded.data.iter_sample_count == 50
    # fixture has sample_count=3, so effective cap is 3
    assert reloaded.effective_iter_sample_count() == 3


def test_init_writes_auto_validate_thresholds(
    tmp_path: Path, per_sample_root: Path,
):
    """Operator-pinned auto-validate thresholds land in config.yaml."""
    params = valid_params(tmp_path, per_sample_root)
    params["experiment"] = {
        "auto_validate_pass_threshold": 0.85,
        "auto_validate_recall_threshold": 0.55,
        "auto_validate_min_samples": 25,
    }
    result = cli_init_project(
        params, repo_root=repo_root(), workspaces_dir=tmp_path / "workspaces"
    )
    assert "error" not in result
    reloaded = ERAConfig.from_yaml(Path(result["config_path"]))
    assert reloaded.experiment.auto_validate_pass_threshold == 0.85
    assert reloaded.experiment.auto_validate_recall_threshold == 0.55
    assert reloaded.experiment.auto_validate_min_samples == 25


def test_init_writes_annotation_probe_to_workspace(
    tmp_path: Path, per_sample_root: Path,
):
    """cli_init_project runs probe_annotations inline (when the agent
    didn't pre-probe) and writes the result to probe/annotations.json."""
    params = valid_params(tmp_path, per_sample_root)
    # Drop any pre-probed annotations so the inline probe fires.
    if "probe" in params:
        params["probe"].pop("annotations", None)
    result = cli_init_project(
        params, repo_root=repo_root(), workspaces_dir=tmp_path / "workspaces"
    )
    assert "error" not in result
    probe_file = Path(result["workspace_path"]) / "probe" / "annotations.json"
    assert probe_file.is_file()
    body = json.loads(probe_file.read_text(encoding="utf-8"))
    # Fresh fixture has no annotations dir → zero counts.
    assert body["central_count"] == 0
    assert body["per_method_count_total"] == 0


def test_init_guide_shows_auto_validate_line(
    tmp_path: Path, per_sample_root: Path,
):
    """The post-init guide must report the chosen auto-validate
    thresholds so the operator sees the gate the loop will apply."""
    params = valid_params(tmp_path, per_sample_root)
    params["experiment"] = {
        "auto_validate_pass_threshold": 0.80,
        "auto_validate_recall_threshold": 0.65,
    }
    result = cli_init_project(
        params, repo_root=repo_root(), workspaces_dir=tmp_path / "workspaces"
    )
    assert "error" not in result
    assert "Auto-validate" in result["guide"]
    assert "0.80" in result["guide"] and "0.65" in result["guide"]


def test_init_guide_warns_on_sync_drift(
    tmp_path: Path, per_sample_root: Path,
):
    """T1 — when the operator deleted a per-method mirror copy out-of-band
    (so central_count > sum(per_method_count)), the post-init guide must
    surface the 'out of sync' note pointing at era.cli annotate-mirror.
    Covers the previously-untested branch in guide._summary_block."""
    # Lay down a central annotation referencing both methods, but only
    # write one per-method mirror copy → central=1, per-method=1, but
    # sum(coverage)=2 → annotators_in_sync becomes False.
    sample_key = "sample_000"
    central = per_sample_root / "annotations" / f"{sample_key}.json"
    central.parent.mkdir(parents=True, exist_ok=True)
    central.write_text(json.dumps({
        "schema_version": "2.0",
        "sample_key": sample_key,
        "per_method": {"flux2klein": "good", "ghost": "bad"},
        "created_at": "2026-05-27T01:00:00+00:00",
        "updated_at": "2026-05-27T01:00:00+00:00",
    }), encoding="utf-8")
    # Only write the mirror for flux2klein under per_sample_root (the
    # method dir; ghost has no method dir so its slot is "in sync" as
    # missing-by-design). To force REAL drift, write a mirror for
    # flux2klein at the WRONG path — let me instead delete the mirror
    # entirely so per_method_count_total < sum(method_coverage).
    # per_sample_root IS the only method path; only flux2klein method.
    # Write the mirror copy.
    mirror = per_sample_root / sample_key / "annotation.json"
    mirror.parent.mkdir(parents=True, exist_ok=True)
    mirror.write_text(json.dumps({
        "schema_version": "2.0",
        "sample_key": sample_key,
        "method_id": "flux2klein",
        "annotation": "good",
        "created_at": "2026-05-27T01:00:00+00:00",
        "updated_at": "2026-05-27T01:00:00+00:00",
    }), encoding="utf-8")
    # Sample_000 doesn't exist as a per-sample dir under per_sample_root,
    # but the central file references both flux2klein AND a "ghost" method
    # not in cfg.data.methods. method_coverage will count both methods (it
    # counts non-empty slots in the central per_method dict), while
    # per_method_count only walks the configured methods → drift surfaces.

    params = valid_params(tmp_path, per_sample_root)
    result = cli_init_project(
        params, repo_root=repo_root(), workspaces_dir=tmp_path / "workspaces"
    )
    assert "error" not in result
    # The guide should include the out-of-sync warning text
    assert "out of sync" in result["guide"]
    assert "annotate-mirror" in result["guide"]


def test_init_inherits_repo_root_settings_pins(
    tmp_path: Path, per_sample_root: Path,
):
    """Phase D-2 extension end-to-end: a custom permissions.allow pattern
    pinned in the (fake) repo-root ``.claude/settings.json`` flows into
    the newly scaffolded workspace's ``.claude/settings.json`` via the
    cli_init_project path → ensure_claude_settings repo-root union.

    Catches a regression in any of:
    - cli_init_project plumbing repo_root through to ensure_claude_settings
    - ensure_claude_settings reading the repo-root layer
    - the merge semantics that union the repo-root allow into the
      workspace allow.
    """
    # Build a fake repo root carrying an operator pin.
    fake_repo = tmp_path / "fake-era-repo"
    fake_repo_claude = fake_repo / ".claude"
    fake_repo_claude.mkdir(parents=True)
    (fake_repo_claude / "settings.json").write_text(
        json.dumps({
            "permissions": {"allow": ["Bash(echo init-integration-pin)"]},
        }),
        encoding="utf-8",
    )

    params = valid_params(tmp_path, per_sample_root)
    result = cli_init_project(
        params, repo_root=fake_repo, workspaces_dir=tmp_path / "workspaces"
    )
    assert "error" not in result

    ws_settings = Path(result["workspace_path"]) / ".claude" / "settings.json"
    data = json.loads(ws_settings.read_text(encoding="utf-8"))
    allow = data["permissions"]["allow"]
    # ERA defaults still present (the era.cli pattern is canonical).
    assert any("era.cli" in p for p in allow)
    # The repo-root operator pin landed in the workspace allowlist.
    assert "Bash(echo init-integration-pin)" in allow
