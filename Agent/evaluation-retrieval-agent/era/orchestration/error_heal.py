"""Stage 6 error healing — categorize a runner failure, fix it, cap retries.

When a Stage 6 evaluator runner exits non-zero, the loop captures its stderr and
calls ``heal_tick``. This module is the minimal, deterministic subset of
SibylSystem's ``error_collector`` + ``auto_fix`` + ``self_heal``:

- **Categorize** — ``categorize_runtime_text`` keyword-classifies captured
  stderr into ``oom`` / ``import`` / ``serving`` / ``missing_dir`` / ``config``
  / ``runtime``.
- **Auto-fix** — ``attempt_auto_fix`` returns a deterministic fix that **patches
  the next runner invocation** (a ``pip_install`` package, a rebound serving
  port, a halved batch size) — it never edits the runner's source, because
  Stage 6 runners are regenerated rather than maintained.
- **Circuit-break** — ``heal_tick`` caps healing at ``CIRCUIT_BREAKER_MAX``
  attempts per distinct ``error_id``; past the cap it gives up so a real bug
  cannot spin the loop forever.

``heal_tick`` returns one of three actions: ``retry_with_patch`` (apply the
patch, re-run), ``escalate`` (no deterministic fix — the experiment sub-agent
should patch the runner itself), or ``give_up`` (circuit breaker open).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

CIRCUIT_BREAKER_MAX = 3
HEAL_STATE_REL = "experiments/heal_state.json"

# Categories that the Stage 6 completion gate may treat as *runtime-failed*
# rather than as a planning miss, once the circuit breaker has opened. The set
# is intentionally narrow: ``oom`` / ``serving`` / ``hung`` are infrastructure
# failures the agent has already tried to auto-fix (halve batch, rebind port,
# retry) and where the loop genuinely has no forward path. ``import`` /
# ``missing_dir`` / ``config`` / ``runtime`` stay ineligible — those are
# planning misses or agent bugs that *must* surface to the operator. The agent
# never *chooses* runtime_failed; this set is the framework's audit boundary.
RUNTIME_FAILURE_CATEGORIES: frozenset[str] = frozenset({
    "oom", "serving", "hung",
})

# Categories, in priority order — the first whose keywords match wins.
_CATEGORY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("oom", ("cuda out of memory", "out of memory", "outofmemoryerror",
             "cuda oom")),
    ("import", ("no module named", "modulenotfounderror", "importerror")),
    ("serving", ("connection refused", "address already in use",
                 "port is already in use", "502 bad gateway",
                 "503 service", "failed to connect", "connection reset",
                 "max retries exceeded")),
    ("missing_dir", ("no such file or directory", "filenotfounderror")),
    ("config", ("jsondecodeerror", "yamlerror", "expecting value",
                "could not parse", "yaml.scanner")),
    ("hung", ("runner hung", "no heartbeat", "heartbeat timeout")),
)

# Import-error module -> safe pip package. ``attempt_auto_fix`` installs only
# values from this table, so the package name can never carry injection.
_SAFE_PACKAGES = {
    "PIL": "pillow", "pil": "pillow",
    "cv2": "opencv-python-headless",
    "clip": "open-clip-torch", "open_clip": "open-clip-torch",
    "lpips": "lpips", "skimage": "scikit-image", "sklearn": "scikit-learn",
    "transformers": "transformers", "torchvision": "torchvision",
    "scipy": "scipy", "pandas": "pandas", "tqdm": "tqdm", "timm": "timm",
    "einops": "einops", "openai": "openai", "requests": "requests",
    "yaml": "pyyaml", "numpy": "numpy",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def categorize_runtime_text(text: str) -> str:
    """Keyword-classify a runner's captured stderr into an error category."""
    low = (text or "").lower()
    for category, needles in _CATEGORY_KEYWORDS:
        if any(n in low for n in needles):
            return category
    return "runtime"


@dataclass
class StructuredError:
    """A categorized runner failure with a stable id for the circuit breaker."""
    category: str
    message: str
    error_id: str = ""

    def __post_init__(self) -> None:
        if not self.error_id:
            self.error_id = _compute_error_id(self.category, self.message)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _normalize(message: str) -> str:
    """Collapse run-to-run noise (digits, hex, paths) so the id is stable."""
    text = (message or "").strip().lower()[:600]
    text = re.sub(r"0x[0-9a-f]+", "0x", text)
    text = re.sub(r"\d+", "0", text)
    text = re.sub(r"/[^\s'\"]+", "/p", text)
    return text


def _compute_error_id(category: str, message: str) -> str:
    """A 12-hex-char id stable across retries of the *same* error."""
    digest = hashlib.sha256(
        f"{category}:{_normalize(message)}".encode("utf-8")
    ).hexdigest()
    return digest[:12]


def structured_error(error_text: str) -> StructuredError:
    """Build a ``StructuredError`` from a runner's captured stderr."""
    return StructuredError(
        category=categorize_runtime_text(error_text),
        message=(error_text or "").strip(),
    )


# ---- deterministic auto-fixes -------------------------------------------

def _read_override(iter_dir: Path, task_id: str) -> dict:
    path = iter_dir / "experiments" / "configs" / f"{task_id}.override.json"
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _fix_import(error: StructuredError) -> dict | None:
    match = re.search(r"no module named ['\"]?([\w.]+)", error.message.lower())
    if not match:
        return None
    top = match.group(1).split(".")[0]
    pkg = _SAFE_PACKAGES.get(top)
    if not pkg:
        return None  # unknown package — escalate rather than install blindly
    return {"fixed": True, "action": "pip_install",
            "detail": f"install missing dependency {pkg!r}",
            "patch": {"pip_install": pkg}}


def _fix_missing_dir(error: StructuredError, iter_dir: Path) -> dict | None:
    match = re.search(r"['\"]([^'\"]+)['\"]", error.message)
    if not match:
        return None
    target = Path(match.group(1))
    if not target.is_absolute():
        target = (iter_dir / target).resolve()
    experiments = (iter_dir / "experiments").resolve()
    # Only ever create directories inside this iteration's experiments/ tree.
    try:
        target.relative_to(experiments)
    except ValueError:
        return None
    create = target if not target.suffix else target.parent
    create.mkdir(parents=True, exist_ok=True)
    return {"fixed": True, "action": "mkdir",
            "detail": f"created missing directory {create}",
            "patch": {}}


def _fix_serving(error: StructuredError, iter_dir: Path, task_id: str,
                 cfg) -> dict | None:
    port_range = list(getattr(cfg.serving, "port_range", [8000, 8099]))
    low, high = (port_range + [8000, 8099])[:2]
    current = _read_override(iter_dir, task_id).get("port", low)
    new_port = current + 1
    if new_port > high:
        return None  # exhausted the range — escalate
    return {"fixed": True, "action": "rebind_port",
            "detail": f"rebind serving endpoint to port {new_port}",
            "patch": {"port": new_port}}


def _fix_oom(error: StructuredError, iter_dir: Path, task_id: str) -> dict:
    override = _read_override(iter_dir, task_id)
    batch_size = int(override.get("batch_size", 8))
    new_batch = max(1, batch_size // 2)
    util = float(override.get("gpu_memory_utilization", 0.90))
    new_util = round(max(0.80, util - 0.05), 2)
    return {"fixed": True, "action": "reduce_load",
            "detail": (f"halve batch_size {batch_size}->{new_batch}, "
                       f"gpu_memory_utilization {util}->{new_util}"),
            "patch": {"batch_size": new_batch,
                      "gpu_memory_utilization": new_util}}


def attempt_auto_fix(
    error: StructuredError, *, iter_dir: Path, cfg, task: dict,
) -> dict | None:
    """Return a deterministic fix for a known error category, or None.

    A returned ``patch`` is merged into the task's ``<task_id>.override.json``
    by the Stage 6 loop and consumed by the next runner invocation — the fix
    never edits runner source. None means "escalate".
    """
    task_id = task.get("id", "")
    if error.category == "import":
        return _fix_import(error)
    if error.category == "missing_dir":
        return _fix_missing_dir(error, iter_dir)
    if error.category == "serving":
        return _fix_serving(error, iter_dir, task_id, cfg)
    if error.category == "oom":
        return _fix_oom(error, iter_dir, task_id)
    return None  # config / runtime — no safe deterministic fix


# ---- circuit breaker ----------------------------------------------------

def _heal_state_path(iter_dir: Path) -> Path:
    return iter_dir / HEAL_STATE_REL


def load_heal_state(iter_dir: Path) -> dict:
    """Read the per-iteration heal state (attempt counts + history)."""
    path = _heal_state_path(iter_dir)
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}
    data.setdefault("attempts", {})
    data.setdefault("history", [])
    return data


def save_heal_state(iter_dir: Path, state: dict) -> None:
    path = _heal_state_path(iter_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(state, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _breaker_key(task_id: str, error_id: str) -> str:
    """The circuit-breaker key — scoped per (task, error) so the same error
    signature recurring on a *different* task gets its own retry budget."""
    return f"{task_id}:{error_id}"


def check_circuit(iter_dir: Path, task_id: str, error_id: str) -> bool:
    """True when a (task, error) pair has hit the retry cap — healing gives up."""
    state = load_heal_state(iter_dir)
    key = _breaker_key(task_id, error_id)
    return state["attempts"].get(key, 0) >= CIRCUIT_BREAKER_MAX


def heal_tick(iter_dir: Path, cfg, task: dict, error_text: str) -> dict:
    """Classify one runner failure and decide the next action.

    Returns ``action`` — ``retry_with_patch`` (apply ``patch``, re-run),
    ``escalate`` (no deterministic fix — sub-agent should patch the runner), or
    ``give_up`` (the circuit breaker is open for this task+error). The breaker
    counts attempts per ``(task_id, error_id)``.
    """
    error = structured_error(error_text)
    task_id = task.get("id") or ""
    key = _breaker_key(task_id, error.error_id)
    state = load_heal_state(iter_dir)
    attempts = state["attempts"].get(key, 0)

    if attempts >= CIRCUIT_BREAKER_MAX:
        return {"status": "ok", "action": "give_up",
                "error_id": error.error_id, "category": error.category,
                "attempts": attempts, "circuit_broken": True,
                "runtime_failure_eligible":
                    error.category in RUNTIME_FAILURE_CATEGORIES,
                "message": f"circuit breaker open after {attempts} attempts"}

    state["attempts"][key] = attempts + 1
    state["history"].append({
        "error_id": error.error_id, "category": error.category,
        "attempt": attempts + 1, "at": _now_iso(),
        "task_id": task.get("id"),
    })
    save_heal_state(iter_dir, state)

    fix = attempt_auto_fix(error, iter_dir=iter_dir, cfg=cfg, task=task)
    if fix is None:
        return {"status": "ok", "action": "escalate",
                "error_id": error.error_id, "category": error.category,
                "attempts": attempts + 1, "circuit_broken": False,
                "message": "no deterministic fix — patch the runner in context"}

    return {"status": "ok", "action": "retry_with_patch",
            "error_id": error.error_id, "category": error.category,
            "attempts": attempts + 1, "circuit_broken": False,
            "fix_action": fix["action"], "detail": fix["detail"],
            "patch": fix.get("patch", {})}
