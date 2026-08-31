"""``python -m era.cli`` — the ERA command-line interface.

Every subcommand reads a single JSON object from **stdin** and prints a single
JSON object to **stdout** (so a slash-command can pipe params via a heredoc and
avoid shell-quoting issues):

    probe                 run environment probes
    init-workspace        scaffold a project workspace
    write-ralph-prompt    compile the runtime Ralph-loop prompt into a workspace
    write-mcp-config      register ERA's MCP servers for a directory (.mcp.json)
    status                summarize ERA workspace(s)
    update-status         patch stage / run_state into a workspace's status.json
    check-autonomy        PreToolUse hook gate (blocks AskUserQuestion mid-loop)
    agent-tier            resolve a tiered pipeline stage's sub-agent tier
    debate-state          read the Stage 2-4 debate-loop state
    debate-tick           record a debate verdict, return the next loop action
    check-experiment-brief  validate a Stage 4 experiment_brief.json
    check-task-plan       validate a Stage 5 task_plan.json against the brief
    init-experiment       seed a Stage 6 experiment_state.json for one pass
    gpu-scan              report free / allowed / leased GPUs
    claim-batch           claim the next runnable Stage 6 task batch
    release-gpus          drop the GPU leases held by tasks
    ensure-watchdog       make sure the GPU watchdog is running (start if not)
    shutdown-judge        safely tear down a Stage 6 vLLM serve task
    experiment-status     snapshot Stage 6 task statuses + a detection script
    wait-for-any-done     block until any Stage 6 task writes its done.json marker
    record-task           record a Stage 6 task outcome (success / failure /
                          skipped / runtime_failed)
    check-experiment-completion  Stage 6 advance gate (every chosen scored or skipped-with-proof)
    recover-experiment    apply a marker-file scan to the experiment state
    heal-tick             classify a task failure and decide the next action
    build-review-model    normalize stage 2-6 artifacts for the Stage 8 web app
    serve-feedback        start the Stage 8 human-feedback web app (background)
    feedback-status       report the feedback server + finalize state
    finalize-feedback     finalize feedback, derive human_labels.json
    stop-feedback         stop the Stage 8 feedback web server
    serve-annotate        start the /era:annotate image-annotation web app (background)
    annotate-probe        probe a dataset's layout without launching the annotation server
    annotate-mirror       copy each saved annotation into its output-image dir
    annotate-status       report the annotation server's running state
    stop-annotate         stop the annotation web server
    serving-memory        read/write shared cross-project serving recipes
    react-aggregate       build the Stage 9 cumulative_feedback block
    react-tick            record a Stage 9 verdict; force ADVANCE at the cap
    create-next-iteration atomically advance to iter_{N+1} (Stage 9 REVISE_*)
    auto-revise           replace a pre-Stage-8 block with a Stage 9 REVISE_SKIP_STAGE1
    list-annotations      list every operator-annotated sample_key for a workspace
    sample-window         pick the iter's random N-sample window (full-mode round)
    auto-validate-prepare build per-(cid, method) sub-agent batches for Phase C-2 gate
    auto-validate-finalize aggregate sub-agent judgments into pass/recall + result.json
    check-evolution-state validate a Stage 9 evolution_state.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ERAConfig
from .orchestration.agents import resolve_agent_tier
from .orchestration.debate import debate_state, debate_tick
from .orchestration.error_heal import heal_tick
from .orchestration.experiment_brief import (
    extract_hypothesis_ids,
    load_pivot_actions,
    validate_experiment_brief,
)
from .orchestration.experiment_results import (
    aggregate_config,
    check_experiment_completion,
    write_summary,
)
from .orchestration.experiment_state import (
    apply_detection,
    complete_task,
    detection_script,
    fail_task,
    init_state,
    parse_detection_output,
    runtime_fail_task,
    skip_task,
    status_snapshot,
    wait_for_any_done,
)
from .orchestration.gpu_scheduler import (
    allowed_gpu_pool,
    claim_batch,
    ensure_watchdog_alive,
    parse_free_gpus,
    parse_gpu_snapshot,
    read_leases,
    release_gpus,
)
from .orchestration.annotate import (
    annotate_status,
    serve_annotate,
    stop_annotate,
)
from .orchestration.human_feedback import (
    feedback_status,
    finalize_feedback,
    serve_feedback,
    stop_feedback,
)
from .orchestration.lifecycle import status_summary, update_status
from .orchestration.mcp import ensure_mcp_config, mcp_config
from .orchestration.project_cli import cli_init_project
from .orchestration.ralph import ensure_claude_settings, write_ralph_prompt
from .orchestration.react import (
    aggregate_cumulative_feedback,
    check_evolution_state,
    create_next_iteration,
    react_tick,
)
from .orchestration.review_adapter import build_and_write_review_model
from .orchestration.task_plan import validate_task_plan
from .paths import repo_root, workspaces_dir
from .probe import (
    probe_checkpoints,
    probe_credentials,
    probe_data_roots,
    probe_gpus,
)
from .workspace import Workspace, resolve_workspace_root


def _read_stdin_json() -> dict:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    return json.loads(raw)


def _emit(obj: dict) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def _cmd_probe(_args: argparse.Namespace) -> int:
    """Probe GPUs, data roots, checkpoints, and credentials.

    stdin JSON: {gpu_ids:[int], data_roots:[str], model_root:str, env_file:str}
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    result = {
        "gpu": probe_gpus(params.get("gpu_ids") or []),
        "data": probe_data_roots(params.get("data_roots") or []),
        "checkpoints": probe_checkpoints(params.get("model_root") or ""),
        "credentials": probe_credentials(params.get("env_file") or ""),
    }
    _emit(result)
    return 0


def _cmd_init_workspace(_args: argparse.Namespace) -> int:
    """Scaffold a workspace from confirmed init parameters (stdin JSON)."""
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    try:
        result = cli_init_project(
            params,
            repo_root=repo_root(),
            workspaces_dir=workspaces_dir(),
        )
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        result = {"error": "exception", "message": f"{type(exc).__name__}: {exc}"}

    _emit(result)
    return 0 if "error" not in result else 1


def _cmd_write_ralph_prompt(_args: argparse.Namespace) -> int:
    """Compile the runtime Ralph-loop prompt into a workspace.

    stdin JSON: {workspace_path: str, project_name?: str}
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    workspace_path = params.get("workspace_path") or ""
    if not workspace_path:
        _emit({"error": "missing_workspace_path",
               "message": "stdin JSON must carry a 'workspace_path'"})
        return 1

    try:
        result = write_ralph_prompt(
            workspace_path,
            project_name=params.get("project_name") or None,
        )
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        result = {"error": "exception", "message": f"{type(exc).__name__}: {exc}"}

    _emit(result)
    return 0 if "error" not in result else 1


def _cmd_write_mcp_config(_args: argparse.Namespace) -> int:
    """Register ERA's MCP servers for a directory (writes .mcp.json).

    stdin JSON: {target_dir?: str}  (omit → the ERA repo root)

    Also ensures the sibling .claude/settings.json so the .mcp.json servers
    auto-trust. Re-run this after the repo moves or the venv is rebuilt — it
    regenerates the absolute server paths.
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    target = Path(params.get("target_dir") or repo_root()).expanduser()
    try:
        if not target.is_dir():
            result = {"error": "no_such_dir",
                      "message": f"{target} is not a directory"}
        else:
            mcp_path = ensure_mcp_config(target)
            settings_path = ensure_claude_settings(target)
            result = {
                "status": "ok",
                "target_dir": str(target),
                "mcp_path": str(mcp_path),
                "settings_path": str(settings_path),
                "servers": sorted(mcp_config()["mcpServers"]),
            }
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        result = {"error": "exception", "message": f"{type(exc).__name__}: {exc}"}

    _emit(result)
    return 0 if "error" not in result else 1


def _cmd_status(_args: argparse.Namespace) -> int:
    """Summarize ERA workspace(s).

    stdin JSON: {workspace_path?: str}  (omit to summarize every workspace)
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    try:
        result = status_summary(
            workspaces_dir(), params.get("workspace_path") or None
        )
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        result = {"error": "exception", "message": f"{type(exc).__name__}: {exc}"}

    _emit(result)
    return 0 if "error" not in result else 1


def _cmd_update_status(_args: argparse.Namespace) -> int:
    """Patch a workspace's status.json (stage / run_state lifecycle fields).

    stdin JSON: {workspace_path: str, stage?, stage_index?, iteration?,
    run_state?}
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    workspace_path = params.pop("workspace_path", "") or ""
    if not workspace_path:
        _emit({"error": "missing_workspace_path",
               "message": "stdin JSON must carry a 'workspace_path'"})
        return 1

    try:
        result = update_status(workspace_path, **params)
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        result = {"error": "exception", "message": f"{type(exc).__name__}: {exc}"}

    _emit(result)
    return 0 if "error" not in result else 1


def _cmd_check_autonomy(_args: argparse.Namespace) -> int:
    """PreToolUse hook gate that enforces ERA's iron autonomy rule.

    Wired into ``.claude/settings.json`` as a ``hooks.PreToolUse`` entry
    with matcher ``AskUserQuestion``. Phase D-3 fix for the bug where
    the outer ralph-loop agent emitted a permission prompt at Stage 6
    startup ("Stage 6 will run for several hours, how should we
    proceed?") despite the prompt rule forbidding it.

    Reads ``<cwd>/status.json``. The hook's working directory is the
    workspace root (Claude Code sets ``cwd`` to the launch directory).

    Exit code semantics (Claude Code PreToolUse hook contract):
    - **Exit 0** — allow the AskUserQuestion call. Used when:
      - ``run_state == "awaiting_human"``: the legitimate Stage 8
        prompt (era-human-feedback skill set this marker BEFORE
        calling AskUserQuestion).
      - ``run_state in {"idle", "stopped", "done"}`` or status.json
        absent / malformed: not inside an active /era:start run.
        Operator may be using Claude Code interactively.
    - **Exit 2** — block the call and surface the stderr message
      to the agent. Used when ``run_state`` is ``"running"`` or
      ``"blocked"`` — the ralph loop is mid-stage and the iron
      autonomy rule forbids asking.
    """
    cwd = Path.cwd()
    status_path = cwd / "status.json"
    if not status_path.is_file():
        # Fail-open: not in a workspace, or workspace not scaffolded.
        # The prompt-level rule still applies; this hook only enforces
        # during active /era:start runs.
        return 0
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # Fail-open on bad status; better to allow an edge-case prompt
        # than to silently brick AskUserQuestion forever.
        return 0
    run_state = status.get("run_state") if isinstance(status, dict) else None
    if run_state in ("awaiting_human", "idle", "stopped", "done", None):
        return 0
    # run_state in ("running", "blocked") or any other unknown active
    # state — block the AskUserQuestion call.
    sys.stderr.write(
        "ERA iron autonomy rule: AskUserQuestion is forbidden during "
        "/era:start (Stages 1-7 + 9-10). Resolve from config.yaml / "
        "spec.md / workspace files and proceed; log the decision in "
        "<workspace>/logs/iterations/. The only legitimate operator "
        "hand-off is Stage 8's Continue prompt (era:era-human-feedback "
        "skill, after it sets run_state: awaiting_human).\n"
    )
    return 2


def _cmd_agent_tier(_args: argparse.Namespace) -> int:
    """Resolve which sub-agent tier runs a tiered pipeline stage.

    stdin JSON: {workspace_path: str, stage: str}
    stage is one of: plan_brainstorm, multi_review, plan_decision,
    experiment_plan, full_experiment.
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    workspace_path = params.get("workspace_path") or ""
    stage = params.get("stage") or ""
    if not workspace_path or not stage:
        _emit({"error": "missing_params",
               "message": "stdin JSON must carry 'workspace_path' and 'stage'"})
        return 1

    try:
        result = resolve_agent_tier(workspace_path, stage)
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        result = {"error": "exception", "message": f"{type(exc).__name__}: {exc}"}

    _emit(result)
    return 0 if "error" not in result else 1


def _cmd_debate_state(_args: argparse.Namespace) -> int:
    """Read the Stage 2-4 debate-loop state (initializing it on first use).

    stdin JSON: {workspace_path: str}
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    workspace_path = params.get("workspace_path") or ""
    if not workspace_path:
        _emit({"error": "missing_workspace_path",
               "message": "stdin JSON must carry a 'workspace_path'"})
        return 1

    try:
        result = debate_state(workspace_path)
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        result = {"error": "exception", "message": f"{type(exc).__name__}: {exc}"}

    _emit(result)
    return 0 if "error" not in result else 1


def _cmd_debate_tick(_args: argparse.Namespace) -> int:
    """Record a Stage 4 synthesizer verdict and return the next loop action.

    stdin JSON: {workspace_path: str, verdict: "ADVANCE" | "REVISE"}
    optional:   {reason: str}  -- the revision brief; recorded on the history
                entry so debate_state.json says why each round happened.
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    workspace_path = params.get("workspace_path") or ""
    verdict = params.get("verdict") or ""
    reason = params.get("reason") or None
    if not workspace_path or not verdict:
        _emit({"error": "missing_params",
               "message": "stdin JSON must carry 'workspace_path' and 'verdict'"})
        return 1

    try:
        result = debate_tick(workspace_path, verdict, reason)
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        result = {"error": "exception", "message": f"{type(exc).__name__}: {exc}"}

    _emit(result)
    return 0 if "error" not in result else 1


def _cmd_check_experiment_brief(_args: argparse.Namespace) -> int:
    """Validate a Stage 4 experiment_brief.json against the v0.1.3 handoff contract.

    stdin JSON: {brief_path: str}  -- a path to the experiment_brief.json
            or: {brief: {...}}     -- the brief inline
    optional:   {hypotheses_path: str}  -- design/hypotheses.md; when given,
                every config hypothesis_id must resolve to a heading in it.
    optional:   {config_path: str}      -- workspace's config.yaml; when given,
                ``data.iter_sample_count`` (capped at ``data.sample_count``) is
                enforced against ``validation.sample_size`` and
                ``pilot.sample_count`` in the brief.
    optional:   {evolution_state_path: str}  -- prior iter's
                ``react/evolution_state.json``; when given,
                ``must_include_configs`` from it is enforced against the
                brief's ``candidate_configs`` (Phase C-2.5 elitism).
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    brief: object
    brief_path = params.get("brief_path")
    if brief_path:
        path = Path(brief_path).expanduser()
        if not path.is_file():
            _emit({"error": "no_such_file", "message": f"{path} not found"})
            return 1
        try:
            brief = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            _emit({"error": "bad_brief_json", "message": str(exc)})
            return 1
    elif "brief" in params:
        brief = params["brief"]
    else:
        _emit({"error": "missing_params",
               "message": "stdin JSON must carry 'brief_path' or 'brief'"})
        return 1

    known_hypotheses = None
    hypotheses_path = params.get("hypotheses_path")
    if hypotheses_path:
        hpath = Path(hypotheses_path).expanduser()
        if not hpath.is_file():
            _emit({"error": "no_such_file", "message": f"{hpath} not found"})
            return 1
        known_hypotheses = extract_hypothesis_ids(
            hpath.read_text(encoding="utf-8")
        )

    iter_sample_count: int | None = None
    config_path = params.get("config_path")
    if config_path:
        cpath = Path(config_path).expanduser()
        if not cpath.is_file():
            _emit({"error": "no_such_file", "message": f"{cpath} not found"})
            return 1
        try:
            iter_sample_count = (
                ERAConfig.from_yaml(cpath).effective_iter_sample_count()
            )
        except Exception as exc:  # noqa: BLE001 - surface bad config as JSON
            _emit({"error": "bad_config",
                   "message": f"{type(exc).__name__}: {exc}"})
            return 1

    must_include_configs: list[str] | None = None
    evolution_state_path = params.get("evolution_state_path")
    if evolution_state_path:
        must_include_configs, err = _load_evolution_state_must_include(
            evolution_state_path
        )
        if err is not None:
            _emit(err)
            return 1

    try:
        problems = validate_experiment_brief(
            brief, known_hypotheses, iter_sample_count=iter_sample_count,
            must_include_configs=must_include_configs,
        )
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        _emit({"error": "exception", "message": f"{type(exc).__name__}: {exc}"})
        return 1

    # Phase C-2.5 — structured ``missing_must_include`` so Stage 4's skill
    # can fold dropped elite cids back into the revision brief without
    # substring-parsing the prose problem string.
    missing_must_include: list[str] = []
    if must_include_configs:
        brief_dict = brief if isinstance(brief, dict) else {}
        present_ids = {
            c.get("combination_id") for c in brief_dict.get(
                "candidate_configs", []) or []
            if isinstance(c, dict) and c.get("combination_id")
        }
        missing_must_include = sorted(
            c for c in must_include_configs if c not in present_ids
        )

    _emit({
        "status": "ok",
        "valid": not problems,
        "problems": problems,
        "hypotheses_checked": known_hypotheses is not None,
        "iter_sample_count_checked": iter_sample_count is not None,
        "must_include_configs_checked": must_include_configs is not None,
        "must_include_configs": must_include_configs or [],
        "missing_must_include": missing_must_include,
    })
    return 0


def _load_json_param(
    params: dict, path_key: str, inline_key: str, bad_json_error: str,
) -> tuple[object, dict | None]:
    """Resolve a ``{path_key}`` file or an inline ``{inline_key}`` JSON param.

    Returns ``(value, None)`` on success or ``(None, error_dict)`` on failure.
    """
    path_value = params.get(path_key)
    if path_value:
        path = Path(path_value).expanduser()
        if not path.is_file():
            return None, {"error": "no_such_file", "message": f"{path} not found"}
        try:
            return json.loads(path.read_text(encoding="utf-8")), None
        except json.JSONDecodeError as exc:
            return None, {"error": bad_json_error, "message": str(exc)}
    if inline_key in params:
        return params[inline_key], None
    return None, {
        "error": "missing_params",
        "message": f"stdin JSON must carry {path_key!r} or {inline_key!r}",
    }


def _load_evolution_state_must_include(
    path: str,
) -> tuple[list[str] | None, dict | None]:
    """Read ``must_include_configs`` from a Stage 9 evolution_state.json.

    Phase C-2.5 helper used by ``check-experiment-brief`` to enforce
    elitism (``candidate_configs ⊇ must_include_configs``). Returns
    ``(must_include, None)`` on success — ``must_include`` is a
    possibly-empty list of cleaned strings — or ``(None, error_dict)``
    on a missing or malformed file.
    """
    p = Path(path).expanduser()
    if not p.is_file():
        return None, {"error": "no_such_file", "message": f"{p} not found"}
    try:
        es = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, {"error": "bad_evolution_state_json",
                      "message": str(exc)}
    raw = es.get("must_include_configs") if isinstance(es, dict) else None
    if isinstance(raw, list):
        return [c for c in raw if isinstance(c, str) and c], None
    return [], None


def _cmd_check_task_plan(_args: argparse.Namespace) -> int:
    """Validate a Stage 5 task_plan.json against its Stage 4 experiment brief.

    stdin JSON: {plan_path: str}   -- a path to the task_plan.json
            or: {plan: {...}}      -- the plan inline
          plus: {brief_path: str}  -- a path to the experiment_brief.json
            or: {brief: {...}}     -- the brief inline
          plus: {workspace_path: str}  -- optional; when present the serve-packing
                                          rule is read from the workspace's
                                          experiment.family_a_execution config
                                          (defaults to serial_full_pool otherwise)
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    plan, err = _load_json_param(params, "plan_path", "plan", "bad_plan_json")
    if err is not None:
        _emit(err)
        return 1
    brief, err = _load_json_param(params, "brief_path", "brief", "bad_brief_json")
    if err is not None:
        _emit(err)
        return 1

    family_a_execution = "serial_full_pool"
    workspace_path = params.get("workspace_path")
    if workspace_path:
        try:
            family_a_execution = _load_cfg(workspace_path).experiment.family_a_execution
        except Exception as exc:  # noqa: BLE001 - report any failure as JSON
            _emit({"error": "bad_workspace", "message": f"{type(exc).__name__}: {exc}"})
            return 1

    try:
        problems = validate_task_plan(plan, brief, family_a_execution)
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        _emit({"error": "exception", "message": f"{type(exc).__name__}: {exc}"})
        return 1

    _emit({"status": "ok", "valid": not problems, "problems": problems})
    return 0


# ---- Stage 6 helpers ----------------------------------------------------

def _resolve_iter_dir(workspace_path: str) -> Path:
    """Resolve a workspace path to its active iteration directory."""
    root = resolve_workspace_root(workspace_path)
    return Workspace(root.parent, root.name).iter_path()


def _load_cfg(workspace_path: str) -> ERAConfig:
    """Load a workspace's config.yaml into an ``ERAConfig``."""
    root = resolve_workspace_root(workspace_path)
    return ERAConfig.from_yaml(root / "config.yaml")


def _load_task_plan(iter_dir: Path) -> dict | None:
    """Load ``experiments/plans/task_plan.json`` for an iteration, or None."""
    path = iter_dir / "experiments" / "plans" / "task_plan.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _brief_combination_ids(iter_dir: Path) -> list[str]:
    """The candidate-config ids from the iteration's experiment_brief.json.

    Used as the expected-config set for ``write_summary`` so a config whose eval
    failed before writing any scores is still reported as incomplete.
    """
    path = iter_dir / "design" / "experiment_brief.json"
    if not path.is_file():
        return []
    try:
        brief = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return [
        c.get("combination_id")
        for c in brief.get("candidate_configs", [])
        if isinstance(c, dict) and c.get("combination_id")
    ]


def _brief_pivot_actions(iter_dir: Path) -> set[str]:
    """Thin re-export of :func:`era.orchestration.experiment_brief.load_pivot_actions`.

    Kept here so existing CLI callsites keep working; the canonical
    implementation lives in the orchestration layer so ``init_state`` and
    ``check_experiment_completion`` can share the same gate without importing
    from the CLI module.
    """
    return load_pivot_actions(iter_dir)


def _cmd_init_experiment(_args: argparse.Namespace) -> int:
    """Seed a Stage 6 experiment_state.json from the task plan.

    stdin JSON: ``{workspace_path, mode?: "pilot"|"annotated"|"full",
    skip?: [{task_id, pivot_proof, skip_reason?}],
    auto_validate_skips?: [combination_id, ...]}``. For ``eval`` tasks
    every skip entry's ``pivot_proof`` must match a Stage 4
    ``experiment_brief.pivot_matrix[*].action`` — otherwise the call returns
    ``error: "unauthorized_skip"`` and the state is left untouched.

    ``auto_validate_skips`` (Phase C-2) is a list of combination_ids whose
    tasks in this mode should be seeded ``skipped`` with
    ``skip_reason: auto_validate_failed``. Each cid must appear in
    ``<iter>/auto_validate/result.json.failing_configs``.
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    workspace_path = params.get("workspace_path") or ""
    if not workspace_path:
        _emit({"error": "missing_workspace_path",
               "message": "stdin JSON must carry a 'workspace_path'"})
        return 1

    av_skips_raw = params.get("auto_validate_skips") or []
    if not isinstance(av_skips_raw, list) or not all(
        isinstance(c, str) for c in av_skips_raw
    ):
        _emit({"error": "bad_auto_validate_skips",
               "message": "auto_validate_skips must be a list of strings"})
        return 1

    try:
        iter_dir = _resolve_iter_dir(workspace_path)
        plan = _load_task_plan(iter_dir)
        if plan is None:
            result = {"error": "no_task_plan",
                      "message": "experiments/plans/task_plan.json not found "
                                 "— run Stage 5 first"}
        else:
            result = init_state(
                iter_dir, plan, params.get("mode") or "pilot",
                tuple(params.get("skip") or ()),
                valid_pivot_actions=load_pivot_actions(iter_dir),
                auto_validate_skips=tuple(av_skips_raw),
            )
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        result = {"error": "exception", "message": f"{type(exc).__name__}: {exc}"}

    _emit(result)
    return 0 if "error" not in result else 1


def _cmd_gpu_scan(_args: argparse.Namespace) -> int:
    """Report free / allowed / leased-elsewhere GPUs from an nvidia-smi snapshot.

    stdin JSON: {workspace_path: str, nvidia_smi_output: str}
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    workspace_path = params.get("workspace_path") or ""
    if not workspace_path:
        _emit({"error": "missing_workspace_path",
               "message": "stdin JSON must carry a 'workspace_path'"})
        return 1

    try:
        cfg = _load_cfg(workspace_path)
        output = params.get("nvidia_smi_output") or ""
        allowed = allowed_gpu_pool(cfg)
        free = parse_free_gpus(
            output, threshold_mb=cfg.experiment.free_gpu_threshold_mb,
            only_gpu_ids=allowed,
        )
        root = resolve_workspace_root(workspace_path)
        leased_elsewhere = sorted(
            int(g) for g, lease in read_leases().items()
            if lease.get("workspace_root") != str(root)
        )
        result = {
            "status": "ok", "free_gpus": free, "allowed_pool": allowed,
            "snapshot": parse_gpu_snapshot(output),
            "leased_elsewhere": leased_elsewhere,
        }
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        result = {"error": "exception", "message": f"{type(exc).__name__}: {exc}"}

    _emit(result)
    return 0 if "error" not in result else 1


def _cmd_claim_batch(_args: argparse.Namespace) -> int:
    """Claim the next runnable batch of Stage 6 tasks.

    stdin JSON: {workspace_path: str, nvidia_smi_output: str, mode: str}
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    workspace_path = params.get("workspace_path") or ""
    mode = params.get("mode") or ""
    if not workspace_path or not mode:
        _emit({"error": "missing_params",
               "message": "stdin JSON must carry 'workspace_path' and 'mode'"})
        return 1

    try:
        cfg = _load_cfg(workspace_path)
        free = parse_free_gpus(
            params.get("nvidia_smi_output") or "",
            threshold_mb=cfg.experiment.free_gpu_threshold_mb,
            only_gpu_ids=allowed_gpu_pool(cfg),
        )
        result = claim_batch(workspace_path, cfg, free, mode)
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        result = {"error": "exception", "message": f"{type(exc).__name__}: {exc}"}

    _emit(result)
    return 0 if "error" not in result else 1


def _cmd_release_gpus(_args: argparse.Namespace) -> int:
    """Drop the GPU leases held by a set of tasks (e.g. on judge teardown).

    stdin JSON: {workspace_path: str, task_ids: [str]}
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    workspace_path = params.get("workspace_path") or ""
    if not workspace_path:
        _emit({"error": "missing_workspace_path",
               "message": "stdin JSON must carry a 'workspace_path'"})
        return 1

    try:
        result = release_gpus(workspace_path, params.get("task_ids") or [])
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        result = {"error": "exception", "message": f"{type(exc).__name__}: {exc}"}

    _emit(result)
    return 0 if "error" not in result else 1


def _cmd_ensure_watchdog(_args: argparse.Namespace) -> int:
    """Make sure the GPU watchdog (NoGPUAlarmNew.py) is running; start if not.

    Stage 8 calls this before the long human-feedback wait so idle experiment
    GPUs stay protected. It checks actual process liveness (pgrep) and only
    runs ``start.sh`` when no watchdog is up — never spawning a duplicate.

    stdin JSON: {workspace_path: str}
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    workspace_path = params.get("workspace_path") or ""
    if not workspace_path:
        _emit({"error": "missing_workspace_path",
               "message": "stdin JSON must carry a 'workspace_path'"})
        return 1

    try:
        iter_dir = _resolve_iter_dir(workspace_path)
        result = ensure_watchdog_alive(iter_dir)
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        result = {"error": "exception", "message": f"{type(exc).__name__}: {exc}"}

    _emit(result)
    return 0 if "error" not in result else 1


def _cmd_shutdown_judge(_args: argparse.Namespace) -> int:
    """Safely tear down a Stage 6 vLLM serve task (Phase D-5).

    Runs the graceful sequence: SIGTERM the pgid → poll-exit →
    SIGKILL on timeout → orphan sweep (pkill -TERM -f served_model) →
    nvidia-smi memory verify → escalate to ``sudo nvidia-smi
    --gpu-reset`` if wedged → port verify → drop the GPU lease.

    stdin JSON: ``{workspace_path: str, task_id: str,
    graceful_timeout_s?: float, gpu_settle_timeout_s?: float}``.

    stdout JSON: the result dict from
    :func:`era.orchestration.serve_shutdown.shutdown_judge` — see its
    docstring for the four ``status`` values
    (``ok`` / ``escalated_kill`` / ``escalated_reset`` / ``still_stuck``).
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    workspace_path = params.get("workspace_path") or ""
    if not workspace_path:
        _emit({"error": "missing_workspace_path",
               "message": "stdin JSON must carry a 'workspace_path'"})
        return 1
    task_id = params.get("task_id") or ""
    if not task_id:
        _emit({"error": "missing_task_id",
               "message": "stdin JSON must carry a 'task_id'"})
        return 1

    kwargs: dict = {}
    if "graceful_timeout_s" in params:
        try:
            kwargs["graceful_timeout_s"] = float(params["graceful_timeout_s"])
        except (TypeError, ValueError):
            _emit({"error": "bad_graceful_timeout_s",
                   "message": "graceful_timeout_s must be a number"})
            return 1
    if "gpu_settle_timeout_s" in params:
        try:
            kwargs["gpu_settle_timeout_s"] = float(
                params["gpu_settle_timeout_s"])
        except (TypeError, ValueError):
            _emit({"error": "bad_gpu_settle_timeout_s",
                   "message": "gpu_settle_timeout_s must be a number"})
            return 1

    try:
        from .orchestration.serve_shutdown import shutdown_judge
        result = shutdown_judge(workspace_path, task_id=task_id, **kwargs)
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        result = {"error": "exception",
                  "message": f"{type(exc).__name__}: {exc}"}

    _emit(result)
    return 0 if "error" not in result else 1


def _cmd_experiment_status(_args: argparse.Namespace) -> int:
    """Snapshot Stage 6 task statuses + the marker-file detection script.

    stdin JSON: {workspace_path: str}
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    workspace_path = params.get("workspace_path") or ""
    if not workspace_path:
        _emit({"error": "missing_workspace_path",
               "message": "stdin JSON must carry a 'workspace_path'"})
        return 1

    try:
        iter_dir = _resolve_iter_dir(workspace_path)
        result = status_snapshot(iter_dir)
        try:
            heartbeat_timeout = _load_cfg(workspace_path).experiment.heartbeat_timeout_s
        except Exception:  # noqa: BLE001 - fall back to the detection_script default
            heartbeat_timeout = 1800
        result["detection_script"] = detection_script(
            iter_dir, result.get("running_task_ids") or [], heartbeat_timeout,
        )
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        result = {"error": "exception", "message": f"{type(exc).__name__}: {exc}"}

    _emit(result)
    return 0 if "error" not in result else 1


def _cmd_wait_for_any_done(_args: argparse.Namespace) -> int:
    """Block until ANY task in ``task_ids`` writes a ``done.json`` marker.

    Event-driven replacement for the Stage 6 poll-interval sleep — returns
    immediately on the first task completion so freed GPUs can be re-claimed
    within hundreds of milliseconds instead of within ``poll_interval_s``.

    stdin JSON: {workspace_path: str, task_ids: [str], timeout_s?: float}

    stdout JSON: {done: [task_id, ...], still_running: [task_id, ...],
                  elapsed_s: float, timed_out: bool}
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    workspace_path = params.get("workspace_path") or ""
    if not workspace_path:
        _emit({"error": "missing_workspace_path",
               "message": "stdin JSON must carry a 'workspace_path'"})
        return 1

    task_ids = params.get("task_ids")
    if not isinstance(task_ids, list) or not all(isinstance(t, str) for t in task_ids):
        _emit({"error": "bad_task_ids",
               "message": "stdin JSON must carry a 'task_ids' list of strings"})
        return 1

    raw_timeout = params.get("timeout_s")
    if raw_timeout is None:
        # Default matches the Stage 6 poll cadence — caller can override.
        try:
            timeout_s = float(_load_cfg(workspace_path).experiment.poll_interval_s)
        except Exception:  # noqa: BLE001 - fall back to a sensible default
            timeout_s = 45.0
    else:
        try:
            timeout_s = float(raw_timeout)
        except (TypeError, ValueError):
            _emit({"error": "bad_timeout_s",
                   "message": f"timeout_s must be a number, got {raw_timeout!r}"})
            return 1

    try:
        iter_dir = _resolve_iter_dir(workspace_path)
        result = wait_for_any_done(iter_dir, task_ids=task_ids, timeout_s=timeout_s)
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        result = {"error": "exception", "message": f"{type(exc).__name__}: {exc}"}

    _emit(result)
    return 0 if "error" not in result else 1


def _cmd_record_task(_args: argparse.Namespace) -> int:
    """Record a Stage 6 task outcome; aggregate the config when it is an eval.

    stdin JSON: {workspace_path: str, task_id: str,
    outcome: "success"|"failure"|"skipped"|"runtime_failed",
    endpoint?, result_dir?, error_summary?, reason?, meta?, pivot_proof?,
    failure_category?, heal_history?}

    ``skipped`` is for a config dropped by a Stage 4 pivot-matrix decision.
    For an ``eval`` task the request MUST carry ``pivot_proof`` (a string
    matching an ``experiment_brief.pivot_matrix[*].action``); otherwise the
    request is rejected as ``unauthorized_skip`` and the state is left
    unchanged. Non-eval task types keep unrestricted skip semantics
    (``setup``/``serve``/``aggregate``/``compare`` skips are legitimate
    orchestration decisions). A skipped eval does not aggregate — no hollow
    ``config_result.json`` is written.

    ``runtime_failed`` is the deterministic forward path for an ``eval`` task
    that hit the heal-tick circuit breaker on an infra-class category
    (``oom`` / ``serving`` / ``hung``). The request MUST carry
    ``failure_category`` and the verbatim heal-tick give_up envelope as
    ``heal_history`` — the framework re-validates the circuit breaker on the
    server side, so the agent cannot fabricate a runtime_failed outcome to
    silently drop a config. The Stage 6 completion gate counts a fully
    runtime_failed config as *resolved* (not missing), with a 30 % cap before
    the gate forces a block. Like ``skipped``, no aggregate is written.
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    workspace_path = params.get("workspace_path") or ""
    task_id = params.get("task_id") or ""
    outcome = params.get("outcome") or ""
    if not workspace_path or not task_id:
        _emit({"error": "missing_params",
               "message": "stdin JSON must carry 'workspace_path' and 'task_id'"})
        return 1
    if outcome not in ("success", "failure", "skipped", "runtime_failed"):
        _emit({"error": "bad_outcome",
               "message": ("outcome must be 'success', 'failure', 'skipped', "
                           "or 'runtime_failed'")})
        return 1

    try:
        iter_dir = _resolve_iter_dir(workspace_path)
        plan = _load_task_plan(iter_dir)
        task = next(
            (t for t in (plan or {}).get("tasks", []) if t.get("id") == task_id),
            None,
        )
        if outcome == "skipped":
            # Fail-closed: an unknown task id (plan missing or typo) must not
            # short-circuit the eval-gate. We cannot prove the task is *not* an
            # eval, so refuse rather than allow the skip through.
            if task is None:
                _emit({
                    "error": "no_task_plan_or_unknown_task",
                    "message": (
                        f"cannot record skipped task {task_id!r} — no "
                        f"task_plan.json or no matching task id. A skip needs "
                        f"the plan to prove the task type; without it the "
                        f"Stage 6 completion gate cannot enforce the "
                        f"eval-only pivot_proof requirement"
                    ),
                    "task_id": task_id,
                })
                return 1
            if task.get("type") == "eval":
                pivot_proof = (params.get("pivot_proof") or "").strip()
                # Phase C-2: a Phase C-2 auto-validate-driven skip is the
                # second authorized scope-reduction path. The caller sets
                # ``auto_validate_skip: true`` and the framework re-validates
                # the task's combination_id against the on-disk
                # auto_validate/result.json.failing_configs.
                av_skip = bool(params.get("auto_validate_skip"))
                if av_skip:
                    cid = task.get("combination_id")
                    av_path = iter_dir / "auto_validate" / "result.json"
                    failing: set[str] = set()
                    if av_path.is_file():
                        try:
                            av_data = json.loads(
                                av_path.read_text(encoding="utf-8"))
                            failing = set(av_data.get("failing_configs") or [])
                        except json.JSONDecodeError:
                            failing = set()
                    if not cid or cid not in failing:
                        _emit({
                            "error": "unauthorized_skip",
                            "message": (
                                "auto_validate_skip requires the task's "
                                "combination_id to appear in "
                                "auto_validate/result.json.failing_configs"
                            ),
                            "task_id": task_id, "combination_id": cid,
                            "failing_configs": sorted(failing),
                        })
                        return 1
                    # Stamp a stable proof string so the completion gate +
                    # the state row can both recognise this skip path.
                    pivot_proof = "auto_validate_failed"
                elif not pivot_proof:
                    _emit({
                        "error": "unauthorized_skip",
                        "message": (
                            "an eval task may only be skipped with a "
                            "pivot_proof matching a Stage 4 "
                            "experiment_brief.pivot_matrix[*].action OR "
                            "with auto_validate_skip:true backed by "
                            "<iter>/auto_validate/result.json; "
                            "runtime scope-reduction (missing deps, budget "
                            "pressure) must be recorded as outcome=failure "
                            "so the Stage 6 completion gate blocks the loop"
                        ),
                        "task_id": task_id, "task_type": "eval",
                    })
                    return 1
                else:
                    valid_actions = _brief_pivot_actions(iter_dir)
                    if valid_actions and pivot_proof not in valid_actions:
                        _emit({
                            "error": "unauthorized_skip",
                            "message": (
                                f"pivot_proof {pivot_proof!r} does not match any "
                                f"experiment_brief.pivot_matrix[*].action — "
                                f"known actions are: {sorted(valid_actions)}"
                            ),
                            "task_id": task_id, "task_type": "eval",
                        })
                        return 1
        elif outcome == "runtime_failed":
            # Fail-closed on unknown task — same rationale as the skip path.
            if task is None:
                _emit({
                    "error": "no_task_plan_or_unknown_task",
                    "message": (
                        f"cannot record runtime_failed task {task_id!r} — no "
                        f"task_plan.json or no matching task id. runtime_failed "
                        f"is eval-only and the framework needs the plan to "
                        f"prove the task type"
                    ),
                    "task_id": task_id,
                })
                return 1
            if task.get("type") != "eval":
                _emit({
                    "error": "runtime_failed_wrong_task_type",
                    "message": (
                        "runtime_failed is only valid for eval tasks — a "
                        "non-eval task (serve / setup / aggregate / compare) "
                        "that hits the circuit breaker must be recorded as "
                        "outcome=failure so the operator can triage"
                    ),
                    "task_id": task_id, "task_type": task.get("type"),
                })
                return 1
        if outcome == "failure":
            result = fail_task(iter_dir, task_id,
                               error_summary=params.get("error_summary") or "")
        elif outcome == "skipped":
            # Phase C-2: an auto_validate_skip carries the proof string
            # "auto_validate_failed" (synthesized above when the request
            # passed the per-config validation). For pivot-matrix skips
            # the operator-supplied pivot_proof is recorded as-is.
            if params.get("auto_validate_skip"):
                proof = "auto_validate_failed"
            else:
                proof = (params.get("pivot_proof") or "").strip() or None
            result = skip_task(
                iter_dir, task_id,
                reason=params.get("reason") or "",
                pivot_proof=proof,
            )
        elif outcome == "runtime_failed":
            result = runtime_fail_task(
                iter_dir, task_id,
                failure_category=(params.get("failure_category") or "").strip(),
                heal_history=params.get("heal_history") or {},
            )
        else:
            result = complete_task(
                iter_dir, task_id, endpoint=params.get("endpoint"),
                result_dir=params.get("result_dir"),
            )
        # For an eval task, aggregate its scores and refresh the mode summary on
        # success or failure — a failed eval must still surface in summary.json.
        # A skipped (policy-dropped) or runtime_failed eval is left out: no
        # hollow config_result for an outcome where no scores survived.
        if (outcome not in ("skipped", "runtime_failed")
                and task and task.get("type") == "eval"
                and "error" not in result):
            meta = params.get("meta") or {}
            spec = task.get("eval") or {}
            task_mode = task.get("mode") or "full"
            result["aggregate"] = aggregate_config(
                iter_dir, task.get("combination_id"), mode=task_mode,
                family=task.get("family"),
                judge=spec.get("judge"), scope=spec.get("scope"),
                cost_usd=meta.get("cost_usd", 0.0),
                wallclock_minutes=meta.get("wallclock_minutes", 0.0),
            )
            result["summary"] = write_summary(
                iter_dir, task_mode,
                expected_configs=_brief_combination_ids(iter_dir),
            )
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        result = {"error": "exception", "message": f"{type(exc).__name__}: {exc}"}

    _emit(result)
    return 0 if "error" not in result else 1


def _cmd_check_experiment_completion(_args: argparse.Namespace) -> int:
    """The Stage 6 advance gate: every chosen_config scored or skipped-with-proof.

    stdin JSON: {workspace_path: str, mode?: "pilot"|"full"}

    Returns the orchestration-layer completion answer:
    ``{status, complete, expected_configs, scored_configs,
       skipped_with_proof, missing_configs, failed_tasks, in_progress_tasks}``.
    ``complete: true`` means it is safe to advance past Stage 6 — every chosen
    config either produced real scores OR was skipped with a Stage 4
    pivot-matrix ``skip_proof``, AND no eval task is still pending/running.
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    workspace_path = params.get("workspace_path") or ""
    if not workspace_path:
        _emit({"error": "missing_workspace_path",
               "message": "stdin JSON must carry a 'workspace_path'"})
        return 1
    mode = params.get("mode") or "full"

    try:
        iter_dir = _resolve_iter_dir(workspace_path)
        expected = _brief_combination_ids(iter_dir)
        result = check_experiment_completion(
            iter_dir, mode=mode, expected_configs=expected,
        )
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        result = {"error": "exception", "message": f"{type(exc).__name__}: {exc}"}

    _emit(result)
    return 0 if "error" not in result else 1


def _cmd_recover_experiment(_args: argparse.Namespace) -> int:
    """Apply a marker-file detection scan to the experiment state.

    stdin JSON: {workspace_path: str, detection_output: str}
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    workspace_path = params.get("workspace_path") or ""
    if not workspace_path:
        _emit({"error": "missing_workspace_path",
               "message": "stdin JSON must carry a 'workspace_path'"})
        return 1

    try:
        cfg = _load_cfg(workspace_path)
        iter_dir = _resolve_iter_dir(workspace_path)
        detection = parse_detection_output(params.get("detection_output") or "")
        recovery, state = apply_detection(
            iter_dir, detection,
            max_retries=cfg.experiment.max_task_retries,
        )
        result = {"status": "ok", **recovery.to_dict(),
                  "recovery_log": state.recovery_log[-10:]}
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        result = {"error": "exception", "message": f"{type(exc).__name__}: {exc}"}

    _emit(result)
    return 0 if "error" not in result else 1


def _cmd_heal_tick(_args: argparse.Namespace) -> int:
    """Classify a Stage 6 task failure and decide the next action.

    stdin JSON: {workspace_path: str, task_id: str, error_text: str}
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    workspace_path = params.get("workspace_path") or ""
    task_id = params.get("task_id") or ""
    if not workspace_path or not task_id:
        _emit({"error": "missing_params",
               "message": "stdin JSON must carry 'workspace_path' and 'task_id'"})
        return 1

    try:
        cfg = _load_cfg(workspace_path)
        iter_dir = _resolve_iter_dir(workspace_path)
        plan = _load_task_plan(iter_dir) or {}
        task = next((t for t in plan.get("tasks", [])
                     if t.get("id") == task_id), None) or {"id": task_id}
        result = heal_tick(iter_dir, cfg, task, params.get("error_text") or "")
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        result = {"error": "exception", "message": f"{type(exc).__name__}: {exc}"}

    _emit(result)
    return 0 if "error" not in result else 1


# ---- Stage 8 human feedback ---------------------------------------------

def _cmd_build_review_model(_args: argparse.Namespace) -> int:
    """Normalize stage 2-6 artifacts into iter_NNN/human/review_model.json.

    The deterministic first pass of the Stage 8 review adapter — run before
    serve-feedback so the web app renders any iteration, not just the demo.

    stdin JSON: {workspace_path: str, iteration?: int|str}
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    workspace_path = params.get("workspace_path") or ""
    if not workspace_path:
        _emit({"error": "missing_workspace_path",
               "message": "stdin JSON must carry a 'workspace_path'"})
        return 1

    try:
        result = build_and_write_review_model(
            workspace_path, iteration=params.get("iteration"))
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        result = {"error": "exception", "message": f"{type(exc).__name__}: {exc}"}

    _emit(result)
    return 0 if "error" not in result else 1


def _cmd_serve_feedback(_args: argparse.Namespace) -> int:
    """Start the Stage 8 human-feedback web app for an iteration (background).

    stdin JSON: {workspace_path: str, iteration?: int|str, host?: str, port?: int}
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    workspace_path = params.get("workspace_path") or ""
    if not workspace_path:
        _emit({"error": "missing_workspace_path",
               "message": "stdin JSON must carry a 'workspace_path'"})
        return 1

    try:
        result = serve_feedback(
            workspace_path, iteration=params.get("iteration"),
            host=params.get("host") or "127.0.0.1", port=params.get("port"),
        )
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        result = {"error": "exception", "message": f"{type(exc).__name__}: {exc}"}

    _emit(result)
    return 0 if "error" not in result else 1


def _cmd_feedback_status(_args: argparse.Namespace) -> int:
    """Report the Stage 8 feedback server + whether feedback is finalized.

    stdin JSON: {workspace_path: str, iteration?: int|str}
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    workspace_path = params.get("workspace_path") or ""
    if not workspace_path:
        _emit({"error": "missing_workspace_path",
               "message": "stdin JSON must carry a 'workspace_path'"})
        return 1

    try:
        result = feedback_status(workspace_path,
                                 iteration=params.get("iteration"))
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        result = {"error": "exception", "message": f"{type(exc).__name__}: {exc}"}

    _emit(result)
    return 0 if "error" not in result else 1


def _cmd_finalize_feedback(_args: argparse.Namespace) -> int:
    """Finalize Stage 8 feedback and derive human_labels.json.

    stdin JSON: {workspace_path: str, iteration?: int|str}
    optional:   {feedback: {...}}  or  {feedback_path: str}  -- when omitted,
                the feedback.json already on disk is finalized.
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    workspace_path = params.get("workspace_path") or ""
    if not workspace_path:
        _emit({"error": "missing_workspace_path",
               "message": "stdin JSON must carry a 'workspace_path'"})
        return 1

    feedback = None
    if params.get("feedback_path") or "feedback" in params:
        feedback, err = _load_json_param(
            params, "feedback_path", "feedback", "bad_feedback_json")
        if err is not None:
            _emit(err)
            return 1

    try:
        result = finalize_feedback(
            workspace_path, feedback=feedback,
            iteration=params.get("iteration"),
        )
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        result = {"error": "exception", "message": f"{type(exc).__name__}: {exc}"}

    _emit(result)
    return 0 if "error" not in result else 1


def _cmd_stop_feedback(_args: argparse.Namespace) -> int:
    """Stop the Stage 8 feedback web server for an iteration.

    stdin JSON: {workspace_path: str, iteration?: int|str}
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    workspace_path = params.get("workspace_path") or ""
    if not workspace_path:
        _emit({"error": "missing_workspace_path",
               "message": "stdin JSON must carry a 'workspace_path'"})
        return 1

    try:
        result = stop_feedback(workspace_path, iteration=params.get("iteration"))
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        result = {"error": "exception", "message": f"{type(exc).__name__}: {exc}"}

    _emit(result)
    return 0 if "error" not in result else 1


def _cmd_serve_annotate(_args: argparse.Namespace) -> int:
    """Start the /era:annotate image-annotation web app for a dataset.

    stdin JSON: ``{dataset_root: str, host?: str, port?: int,
    output_overrides?: {method_id: filename},
    input_role_overrides?: {role_id: filename}}``.

    Standalone — no workspace, no iteration. The two ``*_overrides`` are
    operator-confirmed disambiguations from the slash command when
    annotate-probe returned ``needs_confirmation``. Annotations land at
    ``<dataset_root>/annotations/<sample_key>.json`` for later pickup
    by Stage 7 / Stage 9.
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    dataset_root = params.get("dataset_root") or ""
    if not dataset_root:
        _emit({"error": "missing_dataset_root",
               "message": "stdin JSON must carry a 'dataset_root'"})
        return 1

    def _coerce_overrides(value, name):
        if value is None:
            return None
        if not isinstance(value, dict):
            return {"_err": f"{name} must be an object"}
        return {str(k): str(v) for k, v in value.items()}

    output_overrides = _coerce_overrides(
        params.get("output_overrides"), "output_overrides")
    input_role_overrides = _coerce_overrides(
        params.get("input_role_overrides"), "input_role_overrides")
    for label, val in (("output_overrides", output_overrides),
                       ("input_role_overrides", input_role_overrides)):
        if isinstance(val, dict) and "_err" in val:
            _emit({"error": "bad_overrides", "message": val["_err"]})
            return 1

    try:
        result = serve_annotate(
            dataset_root,
            host=params.get("host") or "127.0.0.1",
            port=params.get("port"),
            output_overrides=output_overrides,
            input_role_overrides=input_role_overrides,
        )
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        result = {"error": "exception",
                  "message": f"{type(exc).__name__}: {exc}"}

    _emit(result)
    return 0 if "error" not in result else 1


def _cmd_annotate_probe(_args: argparse.Namespace) -> int:
    """Probe a dataset's layout without starting a server (read-only).

    stdin JSON: ``{dataset_root: str}``.

    Mirrors what ``serve-annotate`` would discover at launch but returns
    the findings as JSON so the slash command can preview them and
    refuse to start a useless server (e.g. zero samples, no images
    resolve). Also emits ``first_sample_resolves`` — for one real
    sample, whether every (role × method) image file exists on disk.
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    dataset_root = params.get("dataset_root") or ""
    if not dataset_root:
        _emit({"error": "missing_dataset_root",
               "message": "stdin JSON must carry a 'dataset_root'"})
        return 1

    try:
        from .annotate.data import walk_dataset
        dataset = walk_dataset(Path(dataset_root))
    except FileNotFoundError as exc:
        _emit({"error": "no_dataset", "message": str(exc)})
        return 1
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        _emit({"error": "exception",
               "message": f"{type(exc).__name__}: {exc}"})
        return 1

    # Pull per-method output_candidates from the probe summary so the
    # slash command can offer them as AskUserQuestion choices when
    # confidence is needs_confirmation.
    output_candidates: dict[str, list[str]] = {}
    for entry in (dataset.probe_summary.get("methods") or []):
        mid = entry.get("method_id")
        cands = entry.get("output_candidates") or []
        if mid and cands:
            output_candidates[mid] = list(cands)

    first_sample_key: str | None = None
    first_sample_resolves: dict[str, dict[str, bool]] = {}
    if dataset.sample_keys:
        first_sample_key = dataset.sample_keys[0]
        for method_id in dataset.methods:
            roles_map: dict[str, bool] = {}
            for role in dataset.all_roles:
                p = dataset.image_path_for(method_id, first_sample_key, role)
                roles_map[role] = p is not None
            first_sample_resolves[method_id] = roles_map

    result = {
        "status": "ok",
        "dataset_root": str(dataset.root),
        "methods": list(dataset.methods),
        "sample_count": len(dataset.sample_keys),
        "sample_glob": dataset.probe_summary.get("sample_glob") or "",
        "layout": dataset.probe_summary.get("layout") or "unknown",
        "input_roles": dict(dataset.input_roles),
        "output_role": dataset.output_role,
        "method_outputs": dict(dataset.method_outputs),
        "output_candidates": output_candidates,
        "confidence": dataset.probe_summary.get("confidence") or "unknown",
        "warnings": list(dataset.warnings),
        "first_sample_key": first_sample_key,
        "first_sample_resolves": first_sample_resolves,
    }
    _emit(result)
    return 0


def _cmd_annotate_mirror(_args: argparse.Namespace) -> int:
    """Backfill per-method annotation copies into each output image dir.

    stdin JSON: ``{dataset_root: str}``.

    Walks every ``<dataset>/annotations/<sample_key>.json`` and writes a
    per-method copy at ``<method>/<sample_key>/annotation.json`` (only
    that method's text + metadata). Idempotent; can be re-run any time
    to re-sync. Also runs automatically at server startup, so calling
    this command is only needed for an explicit one-shot backfill
    without launching the web app.
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    dataset_root = params.get("dataset_root") or ""
    if not dataset_root:
        _emit({"error": "missing_dataset_root",
               "message": "stdin JSON must carry a 'dataset_root'"})
        return 1

    try:
        from .annotate.data import walk_dataset
        from .annotate.store import mirror_central_to_per_method
        dataset = walk_dataset(Path(dataset_root))
        summary = mirror_central_to_per_method(
            dataset.root, dataset.method_paths,
        )
    except FileNotFoundError as exc:
        _emit({"error": "no_dataset", "message": str(exc)})
        return 1
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        _emit({"error": "exception",
               "message": f"{type(exc).__name__}: {exc}"})
        return 1

    _emit({
        "status": "ok",
        "dataset_root": str(dataset.root),
        "methods": list(dataset.methods),
        **summary,
    })
    return 0


def _cmd_annotate_status(_args: argparse.Namespace) -> int:
    """Report the annotation server's running state for a dataset.

    stdin JSON: ``{dataset_root: str}``.
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    dataset_root = params.get("dataset_root") or ""
    if not dataset_root:
        _emit({"error": "missing_dataset_root",
               "message": "stdin JSON must carry a 'dataset_root'"})
        return 1

    try:
        result = annotate_status(dataset_root)
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        result = {"error": "exception",
                  "message": f"{type(exc).__name__}: {exc}"}

    _emit(result)
    return 0 if "error" not in result else 1


def _cmd_serving_memory(_args: argparse.Namespace) -> int:
    """Read/write the shared cross-project serving-memory store.

    stdin JSON: ``{verb: "list"|"read"|"write"|"forget", ...}``.

    - ``verb: "list"`` → emits every stored recipe's summary row.
    - ``verb: "read", model_id, backend`` → emits the recipe or
      ``{"error": "not_found"}``.
    - ``verb: "write", recipe, overwrite?: bool`` → persists recipe.
    - ``verb: "forget", model_id, backend`` → removes the file.

    Storage lives at ``~/.era/memory/serving_recipes/<slug>.json`` so
    every project's Stage 5-6 can read recipes captured by any prior
    project's successful judge serve (cross-project, per-user).
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    if not isinstance(params, dict):
        # JSON parsed cleanly but isn't an object (e.g. an array or a
        # bare string). Calling ``params.get(...)`` would raise; reject
        # at the boundary so the CLI surfaces a structured error instead.
        _emit({"error": "bad_stdin_json",
               "message": "stdin JSON must be an object, "
                          f"got {type(params).__name__}"})
        return 1

    verb = (params.get("verb") or "").strip()
    if verb not in ("list", "read", "write", "forget"):
        _emit({"error": "bad_verb",
               "message": "verb must be one of: list, read, write, forget"})
        return 1

    try:
        from .orchestration import serving_memory as sm
        if verb == "list":
            _emit({"status": "ok", "recipes": sm.list_recipes()})
            return 0
        if verb == "read":
            mid = params.get("model_id") or ""
            backend = params.get("backend") or ""
            if not mid or not backend:
                _emit({"error": "missing_params",
                       "message": "read needs 'model_id' and 'backend'"})
                return 1
            rec = sm.read_recipe(mid, backend)
            if rec is None:
                _emit({"error": "not_found",
                       "model_id": mid, "backend": backend})
                return 1
            _emit({"status": "ok", **rec})
            return 0
        if verb == "write":
            recipe = params.get("recipe")
            if not isinstance(recipe, dict):
                _emit({"error": "missing_params",
                       "message": "write needs 'recipe' (object)"})
                return 1
            try:
                record = sm.write_recipe(
                    recipe,
                    overwrite=bool(params.get("overwrite", True)),
                    captured_by_workspace=params.get("captured_by_workspace"),
                    captured_by_iteration=params.get("captured_by_iteration"),
                )
            except ValueError as exc:
                _emit({"error": "bad_recipe", "message": str(exc)})
                return 1
            _emit({"status": "ok", **record})
            return 0
        # verb == "forget"
        mid = params.get("model_id") or ""
        backend = params.get("backend") or ""
        if not mid or not backend:
            _emit({"error": "missing_params",
                   "message": "forget needs 'model_id' and 'backend'"})
            return 1
        _emit(sm.forget_recipe(mid, backend))
        return 0
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        _emit({"error": "exception",
               "message": f"{type(exc).__name__}: {exc}"})
        return 1


def _cmd_stop_annotate(_args: argparse.Namespace) -> int:
    """Stop the annotation web server for a dataset.

    stdin JSON: ``{dataset_root: str}``.
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    dataset_root = params.get("dataset_root") or ""
    if not dataset_root:
        _emit({"error": "missing_dataset_root",
               "message": "stdin JSON must carry a 'dataset_root'"})
        return 1

    try:
        result = stop_annotate(dataset_root)
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        result = {"error": "exception",
                  "message": f"{type(exc).__name__}: {exc}"}

    _emit(result)
    return 0 if "error" not in result else 1


def _cmd_react_aggregate(_args: argparse.Namespace) -> int:
    """Aggregate every iteration's finalized feedback into a cumulative block.

    stdin JSON: {workspace_path: str, iteration?: int|str}

    Returns the ``cumulative_feedback`` block of ``evolution_state.json`` —
    per-config endorsement trajectories, wrong-themes pulled from item_marks /
    comparison_marks comments, and the verbatim ``general_feedback`` from
    every iter (one bullet per iter).
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    workspace_path = params.get("workspace_path") or ""
    if not workspace_path:
        _emit({"error": "missing_workspace_path",
               "message": "stdin JSON must carry a 'workspace_path'"})
        return 1

    iteration = params.get("iteration")
    if iteration is None:
        ws = Workspace(
            resolve_workspace_root(workspace_path).parent,
            resolve_workspace_root(workspace_path).name,
        )
        iteration = ws.read_status().get("iteration", 1)

    try:
        cumulative = aggregate_cumulative_feedback(workspace_path, iteration)
        result = {"status": "ok", "iteration": iteration,
                  "cumulative_feedback": cumulative}
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        result = {"error": "exception", "message": f"{type(exc).__name__}: {exc}"}

    _emit(result)
    return 0 if "error" not in result else 1


def _cmd_react_tick(_args: argparse.Namespace) -> int:
    """Record a Stage 9 verdict; force ADVANCE at react.max_iterations.

    stdin JSON: {workspace_path: str,
                 verdict: "ADVANCE" | "REVISE_SKIP_STAGE1" | "REVISE_RERUN_STAGE1"}
    optional:   {rationale: str}
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    workspace_path = params.get("workspace_path") or ""
    verdict = params.get("verdict") or ""
    rationale = params.get("rationale") or None
    if not workspace_path or not verdict:
        _emit({"error": "missing_params",
               "message": "stdin JSON must carry 'workspace_path' and 'verdict'"})
        return 1

    try:
        result = react_tick(workspace_path, verdict, rationale)
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        result = {"error": "exception", "message": f"{type(exc).__name__}: {exc}"}

    _emit(result)
    return 0 if "error" not in result else 1


def _cmd_create_next_iteration(_args: argparse.Namespace) -> int:
    """Atomically create iter_{N+1}, swap current, reset status.

    stdin JSON: {workspace_path: str, rerun_stage1: bool}

    Sets ``stage_index`` to the *last completed* stage so the next ralph pass
    dispatches the right stage: ``0`` (``task_init``) when ``rerun_stage1`` is
    true — next pass runs Stage 1 (``research``); ``1`` (``research``)
    otherwise — next pass runs Stage 2 (``plan_brainstorm``). The
    ``iter_{N+1}/iteration.json.parent_feedback`` block is populated with
    workspace-relative paths to the prior iter's human_labels.json +
    evolution_state.json + (optional) literature_update_brief.md.
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    workspace_path = params.get("workspace_path") or ""
    if not workspace_path:
        _emit({"error": "missing_workspace_path",
               "message": "stdin JSON must carry a 'workspace_path'"})
        return 1
    if "rerun_stage1" not in params:
        _emit({"error": "missing_params",
               "message": "stdin JSON must carry 'rerun_stage1' (boolean)"})
        return 1

    rerun = bool(params["rerun_stage1"])
    try:
        result = create_next_iteration(workspace_path, rerun_stage1=rerun)
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        result = {"error": "exception", "message": f"{type(exc).__name__}: {exc}"}

    _emit(result)
    return 0 if "error" not in result else 1


def _cmd_auto_revise(_args: argparse.Namespace) -> int:
    """Replace a pre-Stage-8 block with a Stage 9 REVISE_SKIP_STAGE1.

    stdin JSON: ``{workspace_path: str, reason: str, source_stage: int,
    blocker_summary: str, diagnostic?: object}``.

    ``source_stage`` must be in ``[1, 7]`` — Stage 8 keeps its existing
    human-feedback block semantics; Stage 0 is pre-loop. The helper
    records ``<iter>/auto_revise/trigger.json`` then fires
    ``react_tick(REVISE_SKIP_STAGE1)``. Below ``react.max_iterations``,
    a fresh next iter is scaffolded; at the cap, ``react_tick`` forces
    ``ADVANCE`` and the response carries ``forced_advance: true`` (the
    ralph-loop then advances ``stage_index`` and lets Stage 10
    terminate the loop).

    Idempotent: re-calling on an iter that already auto-revised
    returns the prior trigger record without re-firing REVISE.
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    workspace_path = params.get("workspace_path") or ""
    if not workspace_path:
        _emit({"error": "missing_workspace_path",
               "message": "stdin JSON must carry a 'workspace_path'"})
        return 1
    reason = (params.get("reason") or "").strip()
    if not reason:
        _emit({"error": "missing_reason",
               "message": "stdin JSON must carry a non-empty 'reason'"})
        return 1
    source_stage = params.get("source_stage")
    if not isinstance(source_stage, int):
        _emit({"error": "missing_source_stage",
               "message": "stdin JSON must carry 'source_stage' (int 1-7)"})
        return 1
    blocker_summary = (params.get("blocker_summary") or "").strip()

    try:
        from .orchestration.auto_revise import auto_revise
        result = auto_revise(
            workspace_path,
            reason=reason,
            source_stage=source_stage,
            blocker_summary=blocker_summary,
            diagnostic=params.get("diagnostic"),
        )
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        result = {"error": "exception",
                  "message": f"{type(exc).__name__}: {exc}"}

    _emit(result)
    return 0 if "error" not in result else 1


def _cmd_auto_validate_prepare(_args: argparse.Namespace) -> int:
    """Build the per-(combination_id, method_id) sub-agent batches for
    the Phase C-2 pass/recall auto-validation gate.

    stdin JSON: {workspace_path: str, mode?: "annotated"}.

    stdout JSON: {status, batches: [...], thresholds, annotated_sample_count,
                  skipped_for_min_samples: bool, mode}.

    When the annotated sample count is below
    ``experiment.auto_validate_min_samples``, returns
    ``skipped_for_min_samples: true`` and an empty batches list — the
    Stage 6 skill then skips the sub-agent dispatch and treats every
    config as passing (identical to today's behaviour on un-annotated
    datasets).
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    workspace_path = params.get("workspace_path") or ""
    if not workspace_path:
        _emit({"error": "missing_workspace_path",
               "message": "stdin JSON must carry a 'workspace_path'"})
        return 1
    mode = params.get("mode") or "annotated"

    try:
        from .orchestration.auto_validate import build_batches
        result = build_batches(workspace_path, mode=mode)
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        result = {"error": "exception",
                  "message": f"{type(exc).__name__}: {exc}"}

    _emit(result)
    return 0 if "error" not in result else 1


def _cmd_list_annotations(_args: argparse.Namespace) -> int:
    """List every operator-annotated sample_key for a workspace.

    Stage 5's planner calls this to stamp ``samples_subset`` on each
    annotated-mode eval task — the operator's annotated subset is the
    deterministic, shared sample list for the Phase C-2 pass/recall
    gate. The probe artifact (``probe/annotations.json``) truncates to
    100 entries for context-window reasons; this CLI returns the
    untruncated list.

    stdin JSON: {workspace_path: str}.

    stdout JSON: {
        status: "ok",
        sample_keys: [sorted sample_keys with any non-empty per_method note],
        count: int,
        method_coverage: {method_id: int},
        data_root: str
    }

    Empty / missing annotations directory → ``count: 0``,
    ``sample_keys: []`` (not an error — datasets without annotations are
    a normal C-2 fall-through case).
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    workspace_path = params.get("workspace_path") or ""
    if not workspace_path:
        _emit({"error": "missing_workspace_path",
               "message": "stdin JSON must carry a 'workspace_path'"})
        return 1

    try:
        cfg = _load_cfg(workspace_path)
        from .orchestration.auto_validate import load_full_annotations
        data_root = cfg.data.data_root
        records = load_full_annotations(data_root)
        # A sample counts as "annotated" only when it carries at least
        # one non-empty per_method note — matches the probe's contract.
        sample_keys: list[str] = []
        coverage: dict[str, int] = {}
        for entry in records:
            non_empty = {
                m: text for m, text in entry.get("per_method", {}).items()
                if isinstance(text, str) and text.strip()
            }
            if not non_empty:
                continue
            sample_keys.append(entry["sample_key"])
            for method_id in non_empty:
                coverage[method_id] = coverage.get(method_id, 0) + 1
        sample_keys.sort()
        result = {
            "status": "ok",
            "sample_keys": sample_keys,
            "count": len(sample_keys),
            "method_coverage": coverage,
            "data_root": data_root,
        }
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        result = {"error": "exception",
                  "message": f"{type(exc).__name__}: {exc}"}

    _emit(result)
    return 0 if "error" not in result else 1


def _cmd_sample_window(_args: argparse.Namespace) -> int:
    """Pick the iter's random N-sample window from the full dataset.

    Phase C-2.3: replaces the legacy ``sorted(glob)[:N]`` first-N rule
    with a deterministic-random N selection. The Stage 5 planner calls
    this once per iter and stamps the returned ``sample_keys`` as
    ``samples_subset`` on every full-mode eval task so all methods
    score the same shuffled subset.

    stdin JSON: ``{workspace_path: str, iteration?: int, n?: int,
                   seed?: int}``.

    Defaults:
    - ``iteration``: active iter from ``status.json``.
    - ``n``: ``cfg.effective_iter_sample_count()`` (operator's per-iter
      cap clamped to ``data.sample_count``).
    - ``seed``: ``sha256(project_name:iteration)[:4]`` (deterministic
      per (workspace, iteration); re-running the same iter produces the
      same set).

    stdout JSON: ``{status, sample_keys, seed, iteration, count,
                    total, source, project_name}``.
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    workspace_path = params.get("workspace_path") or ""
    if not workspace_path:
        _emit({"error": "missing_workspace_path",
               "message": "stdin JSON must carry a 'workspace_path'"})
        return 1

    iteration = params.get("iteration")
    if iteration is not None and not isinstance(iteration, int):
        _emit({"error": "bad_iteration",
               "message": "iteration must be an integer when supplied"})
        return 1
    n = params.get("n")
    if n is not None and not isinstance(n, int):
        _emit({"error": "bad_n",
               "message": "n must be an integer when supplied"})
        return 1
    seed = params.get("seed")
    if seed is not None and not isinstance(seed, int):
        _emit({"error": "bad_seed",
               "message": "seed must be an integer when supplied"})
        return 1

    try:
        from .orchestration.sample_window import pick_random_window
        result = pick_random_window(
            workspace_path,
            n=n, iteration=iteration, seed=seed,
        )
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        result = {"error": "exception",
                  "message": f"{type(exc).__name__}: {exc}"}

    _emit(result)
    return 0 if "error" not in result else 1


def _cmd_auto_validate_finalize(_args: argparse.Namespace) -> int:
    """Aggregate the sub-agent judgments into per-config pass/recall and
    write ``<iter>/auto_validate/result.json``.

    stdin JSON: {workspace_path: str, mode?: "annotated"}.

    stdout JSON: the result.json shape — see
    :func:`era.orchestration.auto_validate.aggregate_judgments`. On a
    missing or malformed judgments file, returns
    ``{error: "missing_judgments", missing: [{combination_id,
    method_id, output_path}, ...]}`` so the Stage 6 skill can retry
    the dispatch (bounded once) before falling through to
    ``auto-revise``.
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    workspace_path = params.get("workspace_path") or ""
    if not workspace_path:
        _emit({"error": "missing_workspace_path",
               "message": "stdin JSON must carry a 'workspace_path'"})
        return 1
    mode = params.get("mode") or "annotated"

    try:
        from .orchestration.auto_validate import aggregate_judgments
        result = aggregate_judgments(workspace_path, mode=mode)
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        result = {"error": "exception",
                  "message": f"{type(exc).__name__}: {exc}"}

    _emit(result)
    return 0 if "error" not in result else 1


def _cmd_check_evolution_state(_args: argparse.Namespace) -> int:
    """Validate a Stage 9 evolution_state.json payload against the schema.

    stdin JSON: {state_path: str}        -- a path to evolution_state.json
            or: {state: {...}}            -- the payload inline
    """
    try:
        params = _read_stdin_json()
    except json.JSONDecodeError as exc:
        _emit({"error": "bad_stdin_json", "message": str(exc)})
        return 1

    state: object
    state_path = params.get("state_path")
    if state_path:
        path = Path(state_path).expanduser()
        if not path.is_file():
            _emit({"error": "no_such_file", "message": f"{path} not found"})
            return 1
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            _emit({"error": "bad_state_json", "message": str(exc)})
            return 1
    elif "state" in params:
        state = params["state"]
    else:
        _emit({"error": "missing_params",
               "message": "stdin JSON must carry 'state_path' or 'state'"})
        return 1

    try:
        problems = check_evolution_state(state)
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        _emit({"error": "exception", "message": f"{type(exc).__name__}: {exc}"})
        return 1

    _emit({"status": "ok", "valid": not problems, "problems": problems})
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="era",
        description="ERA command-line interface",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name, func, help_text in (
        ("probe", _cmd_probe,
         "probe the environment (reads JSON from stdin)"),
        ("init-workspace", _cmd_init_workspace,
         "scaffold a project workspace (reads JSON from stdin)"),
        ("write-ralph-prompt", _cmd_write_ralph_prompt,
         "compile the runtime Ralph-loop prompt (reads JSON from stdin)"),
        ("write-mcp-config", _cmd_write_mcp_config,
         "register ERA's MCP servers for a directory (reads JSON from stdin)"),
        ("status", _cmd_status,
         "summarize ERA workspace(s) (reads JSON from stdin)"),
        ("update-status", _cmd_update_status,
         "patch a workspace's status.json (reads JSON from stdin)"),
        ("check-autonomy", _cmd_check_autonomy,
         "PreToolUse hook gate that blocks AskUserQuestion during /era:start "
         "(no stdin; reads <cwd>/status.json)"),
        ("agent-tier", _cmd_agent_tier,
         "resolve a pipeline stage's sub-agent tier (reads JSON from stdin)"),
        ("debate-state", _cmd_debate_state,
         "read the Stage 2-4 debate-loop state (reads JSON from stdin)"),
        ("debate-tick", _cmd_debate_tick,
         "record a debate verdict, return the next action (reads JSON from stdin)"),
        ("check-experiment-brief", _cmd_check_experiment_brief,
         "validate a Stage 4 experiment_brief.json (reads JSON from stdin)"),
        ("check-task-plan", _cmd_check_task_plan,
         "validate a Stage 5 task_plan.json (reads JSON from stdin)"),
        ("init-experiment", _cmd_init_experiment,
         "seed a Stage 6 experiment_state.json (reads JSON from stdin)"),
        ("gpu-scan", _cmd_gpu_scan,
         "report free / allowed / leased GPUs (reads JSON from stdin)"),
        ("claim-batch", _cmd_claim_batch,
         "claim the next runnable Stage 6 batch (reads JSON from stdin)"),
        ("release-gpus", _cmd_release_gpus,
         "drop GPU leases held by tasks (reads JSON from stdin)"),
        ("ensure-watchdog", _cmd_ensure_watchdog,
         "make sure the GPU watchdog is running, start it if not "
         "(reads JSON from stdin)"),
        ("shutdown-judge", _cmd_shutdown_judge,
         "safely tear down a Stage 6 vLLM serve task — SIGTERM/SIGKILL/"
         "orphan-sweep/GPU-reset/lease-release (reads JSON from stdin)"),
        ("experiment-status", _cmd_experiment_status,
         "snapshot Stage 6 task statuses (reads JSON from stdin)"),
        ("wait-for-any-done", _cmd_wait_for_any_done,
         "block until any running Stage 6 task writes its done.json marker "
         "(reads JSON from stdin)"),
        ("record-task", _cmd_record_task,
         "record a Stage 6 task outcome (reads JSON from stdin)"),
        ("check-experiment-completion", _cmd_check_experiment_completion,
         "Stage 6 advance gate: every chosen_config scored or skipped-with-proof "
         "(reads JSON from stdin)"),
        ("recover-experiment", _cmd_recover_experiment,
         "apply a marker-file scan to the experiment state (JSON from stdin)"),
        ("heal-tick", _cmd_heal_tick,
         "classify a task failure, decide the next action (JSON from stdin)"),
        ("build-review-model", _cmd_build_review_model,
         "normalize stage 2-6 artifacts into review_model.json (JSON from stdin)"),
        ("serve-feedback", _cmd_serve_feedback,
         "start the Stage 8 human-feedback web app (reads JSON from stdin)"),
        ("feedback-status", _cmd_feedback_status,
         "report the feedback server + finalize state (reads JSON from stdin)"),
        ("finalize-feedback", _cmd_finalize_feedback,
         "finalize feedback, derive human_labels.json (reads JSON from stdin)"),
        ("stop-feedback", _cmd_stop_feedback,
         "stop the Stage 8 feedback web server (reads JSON from stdin)"),
        ("serve-annotate", _cmd_serve_annotate,
         "start the /era:annotate image-annotation web app (reads JSON from stdin)"),
        ("annotate-probe", _cmd_annotate_probe,
         "probe a dataset's layout without launching the annotation server (reads JSON from stdin)"),
        ("annotate-mirror", _cmd_annotate_mirror,
         "copy each saved annotation into its output-image dir (reads JSON from stdin)"),
        ("annotate-status", _cmd_annotate_status,
         "report the annotation server's running state (reads JSON from stdin)"),
        ("stop-annotate", _cmd_stop_annotate,
         "stop the annotation web server (reads JSON from stdin)"),
        ("serving-memory", _cmd_serving_memory,
         "read/write the shared cross-project serving-recipe memory (reads JSON from stdin)"),
        ("react-aggregate", _cmd_react_aggregate,
         "build the Stage 9 cumulative_feedback block (reads JSON from stdin)"),
        ("react-tick", _cmd_react_tick,
         "record a Stage 9 verdict, force ADVANCE at cap (reads JSON from stdin)"),
        ("auto-revise", _cmd_auto_revise,
         "replace a pre-Stage-8 block with a Stage 9 REVISE_SKIP_STAGE1 (reads JSON from stdin)"),
        ("list-annotations", _cmd_list_annotations,
         "list every operator-annotated sample_key for a workspace "
         "(reads JSON from stdin)"),
        ("sample-window", _cmd_sample_window,
         "pick the iter's random N-sample window for full-mode evals "
         "(reads JSON from stdin)"),
        ("auto-validate-prepare", _cmd_auto_validate_prepare,
         "build per-(cid, method_id) sub-agent batches for the Phase C-2 pass/recall gate "
         "(reads JSON from stdin)"),
        ("auto-validate-finalize", _cmd_auto_validate_finalize,
         "aggregate sub-agent judgments into pass/recall, write auto_validate/result.json "
         "(reads JSON from stdin)"),
        ("create-next-iteration", _cmd_create_next_iteration,
         "atomically advance to iter_{N+1} (reads JSON from stdin)"),
        ("check-evolution-state", _cmd_check_evolution_state,
         "validate a Stage 9 evolution_state.json (reads JSON from stdin)"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(func=func)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
