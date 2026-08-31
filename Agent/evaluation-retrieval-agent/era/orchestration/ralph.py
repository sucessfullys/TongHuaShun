"""``write_ralph_prompt`` — compile ERA's runtime Ralph-loop prompt.

Mirrors AutoResearch-SibylSystem's ``cli_write_ralph_prompt``: render the
``ralph_loop.md`` template for a project workspace and place the compiled
prompt where the ``ralph-loop`` plugin can pick it up. ``/era:start`` calls
this, then hands the compiled file to the ``ralph-loop`` plugin.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .. import __version__
from ..paths import repo_root
from ..prompt_loader import render_prompt
from ..workspace import resolve_workspace_root
from .mcp import ensure_mcp_config

RALPH_TEMPLATE = "ralph_loop"
COMPLETION_PROMISE = "ERA_PIPELINE_COMPLETE"

# The official ``ralph-loop`` plugin — ERA's autonomous-loop engine. ``/era:start``
# hands the compiled prompt to its ``/ralph-loop`` command, so the plugin must be
# enabled wherever Claude Code is launched to run the pipeline.
RALPH_LOOP_PLUGIN = "ralph-loop@claude-plugins-official"
RALPH_LOOP_MARKETPLACE = "claude-plugins-official"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Phase D-2: the permission patterns ERA's ralph-loop actually uses during
# a normal /era:start run. Pre-approving them keeps the autonomous loop
# truly unattended — without these, every Bash, Edit, Write, and Task call
# during a long Stage 6 walk would prompt the operator. Operator-side
# overrides (broader or narrower) belong in ``.claude/settings.local.json``
# (already gitignored, per-machine).
ERA_ALLOW_PATTERNS: tuple[str, ...] = (
    # The entire era.cli — every subcommand the ralph-loop drives.
    "Bash(.venv/bin/python3 -m era.cli *)",
    "Bash(./.venv/bin/python3 -m era.cli *)",
    "Bash(*/.venv/bin/python3 -m era.cli *)",
    # GPU + runner orchestration (Stage 6 saturation).
    "Bash(nvidia-smi*)",
    # Phase D-4: zombie GPU cleanup (requires root). Two glob variants
    # cover both the long form (--gpu-reset, optionally with --id=N) and
    # the short form (-r). Anything else under "sudo nvidia-smi" (e.g.
    # persistence-mode, ECC config, MIG) is intentionally NOT auto-allowed
    # — operators broaden via .claude/settings.local.json if needed.
    "Bash(sudo nvidia-smi --gpu-reset*)",
    "Bash(sudo nvidia-smi -r*)",
    # Phase D-4 extension: GPU watchdog cleanup. CLAUDE.md's
    # ``## GPU environment`` section documents these two literal
    # commands for the failure mode where NoGPUAlarmNew.py doesn't
    # auto-release a card. Pre-approving them lets the ralph-loop run
    # the documented escape hatch without an operator prompt. The
    # patterns are scoped to the watchdog name — blanket
    # ``Bash(pkill *)`` / ``Bash(sudo pkill *)`` would let the agent
    # kill arbitrary processes, which is intentionally NOT auto-allowed.
    "Bash(pkill -9 -f \"python3 -u NoGPUAlarmNew.py\")",
    "Bash(sudo pkill -9 -f \"python3 -u NoGPUAlarmNew.py\")",
    "Bash(nohup *)",
    "Bash(kill *)",
    # Read-only file inspection — anything the agent might shell out for.
    "Bash(ls *)",
    "Bash(cat *)",
    "Bash(stat *)",
    "Bash(grep *)",
    "Bash(find *)",
    "Bash(head *)",
    "Bash(tail *)",
    "Bash(wc *)",
    "Bash(file *)",
    # Directory scaffolding.
    "Bash(mkdir *)",
    "Bash(touch *)",
    # Local judge endpoint health probes (Stage 6 serve dry-probe).
    "Bash(curl http://localhost:*)",
    "Bash(curl https://localhost:*)",
    # Tiny shell utilities the ralph loop uses.
    "Bash(echo *)",
    "Bash(test *)",
    # Read tool is conceptually always allowed; pin it explicitly for
    # symmetry / no-prompt guarantee.
    "Read",
    # Workspace-internal writes — every file ERA writes lives under one
    # of these subtrees, relative to the workspace root.
    "Write(iter_*/**)",
    "Edit(iter_*/**)",
    "Write(logs/**)",
    "Edit(logs/**)",
    "Write(.claude/**)",
    "Edit(.claude/**)",
    "Write(shared/**)",
    "Edit(shared/**)",
    # Every ERA sub-agent dispatch (heavy/standard/light tiers, the
    # codex reviewer, the React advisor, the auto-validator).
    "Task(era:*)",
)

ERA_DENY_PATTERNS: tuple[str, ...] = (
    # Hard-blocked per the project safety protocol — even if an operator's
    # local override tries to allow them.
    "Bash(git push --force*)",
    "Bash(git push -f*)",
    "Bash(git config --global *)",
    "Bash(rm -rf /)",
    "Bash(rm -rf /*)",
)


def era_claude_settings() -> dict:
    """The ``.claude/settings.json`` payload ERA writes per launch directory.

    It configures three things Claude Code needs to run ERA's autonomous
    loop:

    - ``enabledPlugins`` + ``extraKnownMarketplaces`` — enable the ``ralph-loop``
      plugin (and make its marketplace resolvable without a prior ``/plugin
      marketplace add``), so ``/ralph-loop`` loads wherever Claude Code starts.
    - ``enableAllProjectMcpServers`` — auto-trust the servers declared in the
      sibling ``.mcp.json`` (see ``era/orchestration/mcp.py``), so the loop
      never stops on an MCP approval prompt.
    - ``permissions`` (Phase D-2) — pre-approve the Bash / Write / Edit /
      Task patterns ERA's runtime actually uses, plus a small deny-list of
      catastrophic operations the project safety protocol forbids. Without
      this block, the autonomous loop hits a permission prompt every few
      seconds during Stage 6.
    - ``hooks.PreToolUse`` (Phase D-3) — a structural autonomy gate that
      blocks ``AskUserQuestion`` calls during an active ``/era:start`` run
      (any stage except Stage 8). Calls ``era.cli check-autonomy`` which
      reads ``status.json.run_state`` — if running/blocked, the hook exits
      2 and the agent cannot prompt the operator. The legitimate Stage 8
      site (``era-human-feedback`` skill setting ``run_state:
      awaiting_human`` before its prompt) is allowed through.
    """
    return {
        "enabledPlugins": {RALPH_LOOP_PLUGIN: True},
        "extraKnownMarketplaces": {
            RALPH_LOOP_MARKETPLACE: {
                "source": {
                    "source": "github",
                    "repo": "anthropics/claude-plugins-official",
                },
            },
        },
        "enableAllProjectMcpServers": True,
        "permissions": {
            "allow": list(ERA_ALLOW_PATTERNS),
            "deny": list(ERA_DENY_PATTERNS),
        },
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "AskUserQuestion",
                    "hooks": [
                        {
                            "type": "command",
                            "command": (
                                "$CLAUDE_PROJECT_DIR/.venv/bin/python3 -m "
                                "era.cli check-autonomy"
                            ),
                        },
                    ],
                },
            ],
        },
    }


def _merge_into_settings(existing: dict, era_defaults: dict) -> tuple[dict, bool]:
    """Merge ERA-managed keys into an existing ``settings.json`` payload.

    Returns ``(merged, changed)`` — ``changed`` is True iff any value was
    added or extended (signal to the caller to re-write the file).

    Merge rules:
    - For top-level keys ERA does NOT manage: leave the operator's value
      untouched.
    - For ``enabledPlugins`` / ``extraKnownMarketplaces``: if the key is
      absent, add the ERA default. If present, leave the operator's value
      as-is (don't fight a deliberate disable).
    - For ``enableAllProjectMcpServers``: same — only set when absent.
    - For ``permissions.allow`` and ``permissions.deny``: take the UNION
      with the existing list, preserving operator additions and ensuring
      every ERA default pattern is present.
    """
    merged = dict(existing)  # shallow copy; we mutate sub-keys below
    changed = False

    # Plain "ensure-present" keys.
    for key in ("enabledPlugins", "extraKnownMarketplaces",
                "enableAllProjectMcpServers"):
        if key not in merged and key in era_defaults:
            merged[key] = era_defaults[key]
            changed = True

    # permissions.allow / permissions.deny — list union.
    era_perms = era_defaults.get("permissions") or {}
    if era_perms:
        existing_perms = merged.get("permissions") or {}
        new_perms = dict(existing_perms)
        for bucket in ("allow", "deny"):
            era_bucket = era_perms.get(bucket) or []
            existing_bucket = list(existing_perms.get(bucket) or [])
            missing = [p for p in era_bucket if p not in existing_bucket]
            if missing:
                new_perms[bucket] = existing_bucket + missing
                changed = True
            elif bucket not in existing_perms:
                # Key was absent; create with empty list for consistency.
                new_perms[bucket] = list(era_bucket)
                changed = changed or bool(era_bucket)
        if new_perms != existing_perms:
            merged["permissions"] = new_perms
            changed = True

    # hooks.PreToolUse — ensure ERA's autonomy-gate matcher is present.
    # Operator-added hooks for OTHER matchers are preserved verbatim.
    # The legitimate Stage 8 site is handled by the era.cli check-autonomy
    # gate at exit code level (run_state == "awaiting_human" → exit 0).
    era_hooks = era_defaults.get("hooks") or {}
    if era_hooks:
        existing_hooks = merged.get("hooks") or {}
        new_hooks = dict(existing_hooks)
        for event, era_entries in era_hooks.items():
            existing_entries = list(existing_hooks.get(event) or [])
            era_matchers = {
                entry.get("matcher") for entry in (era_entries or [])
                if isinstance(entry, dict) and entry.get("matcher")
            }
            existing_matchers = {
                entry.get("matcher") for entry in existing_entries
                if isinstance(entry, dict) and entry.get("matcher")
            }
            # For each ERA matcher missing from the operator's existing
            # entries, append the ERA default. Don't touch existing
            # matchers (the operator may have customised them).
            additions = [
                entry for entry in era_entries
                if isinstance(entry, dict)
                and entry.get("matcher") in (era_matchers - existing_matchers)
            ]
            if additions:
                new_hooks[event] = existing_entries + additions
                changed = True
            elif event not in existing_hooks:
                new_hooks[event] = list(era_entries)
                changed = changed or bool(era_entries)
        if new_hooks != existing_hooks:
            merged["hooks"] = new_hooks
            changed = True

    return merged, changed


def _read_settings_json(path: Path) -> dict | None:
    """Parse a ``.claude/settings.json`` file. Returns the parsed dict on
    success, ``None`` on absent / unreadable / non-object content. Used
    by :func:`ensure_claude_settings` to layer ERA defaults + operator
    pins on top of whatever is on disk.
    """
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def ensure_claude_settings(
    root: str | Path, *,
    repo_root_path: str | Path | None = None,
) -> Path:
    """Ensure ``<root>/.claude/settings.json`` carries ERA's managed keys.

    Merge semantics (Phase D-2 — was previously "leave existing file
    untouched"):
    - If the file is absent, write the full ``era_claude_settings()``
      payload.
    - If the file exists, parse + top-level-merge: for every ERA-managed
      key (``enabledPlugins`` / ``extraKnownMarketplaces`` /
      ``enableAllProjectMcpServers`` / ``permissions``), ensure the ERA
      defaults are present. Operator-added top-level keys (e.g.
      ``hooks``, ``model``) are preserved verbatim.
    - For ``permissions.allow`` and ``permissions.deny``, the UNION with
      existing entries is taken — operator-added patterns survive, ERA's
      defaults are guaranteed.

    **Phase D-2 extension — repo-root inheritance.** When ``root`` is NOT
    the ERA repo root itself, the operator's
    ``<era_repo>/.claude/settings.json`` is treated as an additional
    defaults layer: its ``permissions.allow`` / ``.deny`` entries union
    into the workspace, ``hooks.*`` matchers stack additively. This lets
    operators pin repo-wide custom patterns once and have every new
    workspace inherit them, without source edits to
    ``ERA_ALLOW_PATTERNS``. ``repo_root_path`` is an injection point
    for tests; defaults to :func:`era.paths.repo_root`.

    Existing workspaces upgrade automatically on the next ``/era:resume``
    (each call to ``write_ralph_prompt`` invokes ``ensure_claude_settings``).
    A malformed existing ``settings.json`` (unparseable JSON) is treated
    as if absent — the file is overwritten with the fresh defaults. Returns
    the settings path.
    """
    claude_dir = Path(root) / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    settings_path = claude_dir / "settings.json"
    era_defaults = era_claude_settings()

    existing = _read_settings_json(settings_path)
    if existing is None:
        # File missing or malformed — start from ERA defaults.
        merged = era_defaults
        changed = True
    else:
        merged, changed = _merge_into_settings(existing, era_defaults)

    # Phase D-2 extension: union in repo-root operator pins as a second
    # defaults layer. Skip when ``root`` IS the repo root (would be a
    # self-merge with the file we just read).
    rrp = Path(repo_root_path) if repo_root_path else repo_root()
    if Path(root).resolve() != rrp.resolve():
        repo_extras = _read_settings_json(rrp / ".claude" / "settings.json")
        if repo_extras:
            merged, repo_changed = _merge_into_settings(merged, repo_extras)
            changed = changed or repo_changed

    if changed:
        _write_text_atomic(
            settings_path,
            json.dumps(merged, indent=2, ensure_ascii=False) + "\n",
        )
    return settings_path


def compile_ralph_prompt(
    workspace_path: str | Path,
    project_name: str | None = None,
) -> str:
    """Render the Ralph-loop runtime prompt for ``workspace_path``."""
    root = resolve_workspace_root(workspace_path)
    name = project_name or root.name
    return render_prompt(
        RALPH_TEMPLATE,
        PROJECT_NAME=name,
        WORKSPACE_PATH=str(root),
    )


def write_ralph_prompt(
    workspace_path: str | Path,
    project_name: str | None = None,
) -> dict:
    """Compile the Ralph-loop prompt and write it into the workspace.

    Writes ``<workspace>/.claude/ralph-prompt.txt`` (the prompt the
    ``ralph-loop`` plugin runs) and ``<workspace>/.claude/ralph-state.json``
    (compile metadata). Returns a JSON-serializable result dict — an ``error``
    key on failure, otherwise ``status: ok`` plus paths.
    """
    root = resolve_workspace_root(workspace_path)
    if not (root / "status.json").exists():
        return {
            "error": "not_a_workspace",
            "message": f"{root} has no status.json — run /era:init first",
            "workspace_path": str(root),
        }

    name = project_name or root.name
    content = compile_ralph_prompt(root, name)

    claude_dir = root / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    settings_path = ensure_claude_settings(root)
    mcp_path = ensure_mcp_config(root)
    prompt_path = claude_dir / "ralph-prompt.txt"
    state_path = claude_dir / "ralph-state.json"

    _write_text_atomic(prompt_path, content)
    state = {
        "project_name": name,
        "workspace_path": str(root),
        "prompt_path": str(prompt_path),
        "era_version": __version__,
        "completion_promise": COMPLETION_PROMISE,
        "compiled_at": _now_iso(),
    }
    _write_text_atomic(
        state_path, json.dumps(state, indent=2, ensure_ascii=False) + "\n"
    )

    return {
        "status": "ok",
        "project_name": name,
        "workspace_path": str(root),
        "prompt_path": str(prompt_path),
        "state_path": str(state_path),
        "settings_path": str(settings_path),
        "mcp_path": str(mcp_path),
        "completion_promise": COMPLETION_PROMISE,
        "chars": len(content),
    }


def _write_text_atomic(path: Path, text: str) -> None:
    """Write text via a temp file + ``os.replace`` so readers never see a
    partial file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
