"""Validate the Stage 5 experiment task plan (``experiments/plans/task_plan.json``).

``task_plan.json`` is the dependency-ordered DAG Stage 6 executes: Stage 5
expands each Stage 4 ``experiment_brief.json`` candidate configuration into
runnable tasks — serve a VLM judge, run an evaluator, aggregate scores, compare.
The DAG shape and ERA's model/metric-testing rules are deterministic
constraints, so they are checked here rather than trusted to the planning agent:

- **Structural** — every task carries the executable fields Stage 6 needs; task
  ids are unique; every ``depends_on`` resolves; the graph is acyclic.
- **Serve packing** — under ``serial_full_pool`` (Rule 6) the ``serve`` tasks
  form a single linear chain, each owning the whole GPU pool; under
  ``parallel_packed`` (default) each judge is right-sized to its
  ``tensor_parallel`` GPUs and judges stay independent so they co-reside.
- **Coverage** — every brief ``candidate_configs`` entry is run by both a pilot
  and a full ``eval`` task; every Family-A ``eval`` has a matching ``serve``.
- **Pilot gate** — exactly one gating ``compare`` separates the pilot pass from
  the full pass; every full ``eval`` is gated behind it.

``validate_task_plan`` returns a list of human-readable problems (empty ==
valid). Stage 5 runs it after writing the plan; any problem is a REVISE trigger.
"""

from __future__ import annotations

import re
from numbers import Number

from .experiment_results import scores_relpath

TASK_TYPES = ("setup", "serve", "eval", "aggregate", "compare")
# Stage 6 walks the DAG one mode at a time. ``pilot`` is the smoke pass that
# verifies runner code works on a few samples; ``annotated`` (Phase C-2) is
# the operator-annotated subset that feeds the pass/recall auto-validation
# gate; ``full`` is the N=50 round that produces the scores Stage 8 humans
# review.
EVAL_MODES = ("pilot", "annotated", "full")
# Modes Stage 5 must cover for EVERY brief config. ``annotated`` is
# additive — emitted only when the dataset has >= auto_validate_min_samples
# annotations — so it is NOT in this required set.
REQUIRED_EVAL_MODES = ("pilot", "full")
FAMILIES = ("A", "B", "hybrid")

REQUIRED_PLAN_TOP = (
    "schema_version", "iteration", "evaluation_goal", "gpu_pool", "tasks",
)
# Fields every task carries, whatever its type.
REQUIRED_TASK = (
    "id", "type", "depends_on", "gpu_count", "estimated_minutes",
    "expected_output",
)
# Task ids are interpolated into the Stage 6 bash detection script — keep them
# to a shell-safe charset so a stray quote can never break or inject it.
_VALID_TASK_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def _is_number(value: object) -> bool:
    """True for an int/float (but not bool, which is a subclass of int)."""
    return isinstance(value, Number) and not isinstance(value, bool)


# Phase C-2.2: judge names paired across ``serve.judge`` and ``eval.judge``
# must refer to the same underlying model — but the Stage 5 planner has been
# observed to decorate the eval side with descriptive modality suffixes
# (``-pointwise``, ``-pairwise``, ``-flag``, ``-judge``, ``-rubric``, or a
# trailing ``-v<digits>`` revision tag). Modality belongs in ``eval.prompt``
# / ``eval.scope``, not in the judge name. ``_normalize_judge`` strips ONLY
# those exact decorations so a planner producing ``"vlm-7b"`` + ``"vlm-7b-
# pointwise"`` validates. A real model mismatch (e.g. ``"vlm-7b"`` vs
# ``"vlm-72b"``) is still caught because the size suffix is not in the
# strip set.
_JUDGE_SUFFIX_RE = re.compile(
    r"(?:-(?:pointwise|pairwise|flag|judge|rubric|scorer))+$"
    r"|-v\d+$",
    re.IGNORECASE,
)


def _normalize_judge(name: object) -> str | None:
    """Strip known descriptive suffixes from a judge name for comparison.

    Returns ``None`` when ``name`` is not a non-empty string so the caller's
    truthiness check still works.
    """
    if not isinstance(name, str) or not name:
        return None
    out = name
    while True:
        stripped = _JUDGE_SUFFIX_RE.sub("", out)
        if stripped == out:
            return out
        out = stripped


def _is_int(value: object) -> bool:
    """True for a real int (but not bool)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _check_task_common(
    i: int, task: dict, gpu_pool: object, problems: list[str],
) -> None:
    """Validate the type-agnostic fields of one task."""
    for key in REQUIRED_TASK:
        if key not in task:
            problems.append(f"tasks[{i}] missing {key!r}")

    tid = task.get("id")
    if tid is not None and not _VALID_TASK_ID.match(str(tid)):
        problems.append(
            f"tasks[{i}] id {tid!r} must match [A-Za-z0-9._-]+ — task ids are "
            f"interpolated into the Stage 6 shell detection script"
        )

    ttype = task.get("type")
    if ttype is not None and ttype not in TASK_TYPES:
        problems.append(f"tasks[{i}].type={ttype!r} is not one of {TASK_TYPES}")

    if not isinstance(task.get("depends_on"), list):
        problems.append(f"tasks[{i}] 'depends_on' must be a list")

    gpu_count = task.get("gpu_count")
    if not _is_int(gpu_count) or gpu_count < 0:
        problems.append(f"tasks[{i}] 'gpu_count' must be an integer >= 0")
    elif _is_int(gpu_pool) and gpu_count > gpu_pool:
        problems.append(
            f"tasks[{i}] 'gpu_count' ({gpu_count}) exceeds gpu_pool ({gpu_pool})"
        )

    est = task.get("estimated_minutes")
    if not _is_number(est) or est <= 0:
        problems.append(f"tasks[{i}] 'estimated_minutes' must be a number > 0")

    if not task.get("expected_output"):
        problems.append(f"tasks[{i}] 'expected_output' must be non-empty")


def _check_pilot_block(label: str, pilot: object, problems: list[str]) -> None:
    """Validate an ``eval`` task's ``pilot`` spec."""
    if not isinstance(pilot, dict):
        problems.append(f"{label} 'pilot' must be an object")
        return
    samples = pilot.get("samples")
    if not _is_int(samples) or samples <= 0:
        problems.append(f"{label} pilot.samples must be an integer > 0")
    if not _is_int(pilot.get("seed")):
        problems.append(f"{label} pilot.seed must be an integer")
    timeout = pilot.get("timeout")
    if not _is_number(timeout) or timeout <= 0:
        problems.append(f"{label} pilot.timeout must be a number > 0")
    if not pilot.get("pass_criteria"):
        problems.append(f"{label} pilot.pass_criteria must be non-empty")


def _check_eval_task(i: int, task: dict, problems: list[str]) -> None:
    """Validate the ``eval``-specific fields of one task."""
    label = f"tasks[{i}]"
    family = task.get("family")
    if family not in FAMILIES:
        problems.append(f"{label} eval 'family' must be one of {FAMILIES}")
    if task.get("mode") not in EVAL_MODES:
        problems.append(f"{label} eval 'mode' must be one of {EVAL_MODES}")
    for key in ("combination_id", "hypothesis_id"):
        if not task.get(key):
            problems.append(f"{label} eval missing {key!r}")

    # The eval runner's scores file must land where ``aggregate_config`` reads
    # it — the canonical ``results/<mode>/<combination_id>/scores.jsonl``. A
    # divergent ``expected_output`` silently yields a hollow config_result.json
    # and a hollow summary, so it is a hard Stage-5 REVISE trigger.
    cid, mode, exp = (
        task.get("combination_id"), task.get("mode"), task.get("expected_output")
    )
    if cid and mode in EVAL_MODES and exp:
        want = scores_relpath(cid, mode)
        if not str(exp).replace("\\", "/").endswith(want):
            problems.append(
                f"{label} eval 'expected_output' {exp!r} must end with "
                f"{want!r} — the canonical path aggregate_config reads"
            )

    inputs = task.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        problems.append(f"{label} eval 'inputs' must be a non-empty list")

    spec = task.get("eval")
    if not isinstance(spec, dict):
        problems.append(f"{label} eval 'eval' must be an object")
        spec = {}
    if not spec.get("scope"):
        problems.append(f"{label} eval.eval missing 'scope'")

    # Family-independent: any eval whose spec names a judge needs a serve task
    # to call — a hybrid config with a judge is just as endpoint-dependent as a
    # Family-A one. (The serve task's existence + judge match is cross-checked
    # in validate_task_plan.)
    if spec.get("judge") and not task.get("judge_task_id"):
        problems.append(
            f"{label} eval names eval.judge {spec.get('judge')!r} but has no "
            f"'judge_task_id' pointing at a serve task"
        )

    if family == "A":
        if not spec.get("judge"):
            problems.append(f"{label} family 'A' eval needs eval.judge")
        if not spec.get("prompt"):
            problems.append(f"{label} family 'A' eval needs eval.prompt")
        if task.get("gpu_count") not in (0, None):
            problems.append(
                f"{label} family 'A' eval must have gpu_count 0 — it consumes "
                f"a served endpoint, not GPUs"
            )
    elif family == "B":
        if not spec.get("metric_subfamily"):
            problems.append(f"{label} family 'B' eval needs eval.metric_subfamily")
        gpu_count = task.get("gpu_count")
        if _is_int(gpu_count) and gpu_count < 1:
            problems.append(f"{label} family 'B' eval needs gpu_count >= 1")
    elif family == "hybrid":
        if not (spec.get("judge") or spec.get("metric_subfamily")):
            problems.append(
                f"{label} family 'hybrid' eval needs eval.judge or "
                f"eval.metric_subfamily"
            )

    # Phase C-2: annotated-mode eval tasks MUST carry a concrete sample list
    # (the operator's annotated subset). pilot/full eval tasks MUST NOT —
    # they fall back to the runner's ``sorted(glob)[:samples_per_method]``
    # selection so all methods score the same N samples by construction.
    samples_subset = task.get("samples_subset")
    mode = task.get("mode")
    if mode == "annotated":
        # Annotated-mode tasks MUST carry samples_subset (the operator's
        # annotated sample list — without it the Phase C-2 gate has
        # nothing to score against).
        if not isinstance(samples_subset, list) or not samples_subset:
            problems.append(
                f"{label} annotated eval needs a non-empty 'samples_subset' "
                f"(the operator-annotated sample keys)"
            )
        elif not all(isinstance(s, str) and s for s in samples_subset):
            problems.append(
                f"{label} annotated eval 'samples_subset' must be a list of "
                f"non-empty strings (sample_keys)"
            )
    elif samples_subset is not None:
        # Phase C-2.3: full + pilot eval tasks MAY carry samples_subset.
        # Stage 5 stamps a deterministically-random N-sample window
        # (via era.cli sample-window) on every full-mode eval task so
        # all methods score the same shuffled subset. Validate
        # well-formedness; don't reject by mode.
        if not isinstance(samples_subset, list) or not samples_subset:
            problems.append(
                f"{label} eval 'samples_subset' must be a non-empty list "
                f"when present"
            )
        elif not all(isinstance(s, str) and s for s in samples_subset):
            problems.append(
                f"{label} eval 'samples_subset' must be a list of non-empty "
                f"strings (sample_keys)"
            )

    # Phase C-2.2: the ``pilot`` block (samples/seed/timeout/pass_criteria)
    # is semantically the smoke-pass spec — only required on pilot-mode eval
    # tasks. Annotated-mode tasks use ``samples_subset``; full-mode tasks
    # use the operator's ``samples_per_method`` cap. If the planner carries
    # a descriptive pilot block on a non-pilot task (for symmetry or
    # documentation), still validate it for well-formedness.
    if task.get("mode") == "pilot" or task.get("pilot") is not None:
        _check_pilot_block(f"{label} eval", task.get("pilot"), problems)


def _check_serve_task(
    i: int, task: dict, gpu_pool: object, problems: list[str],
    parallel: bool,
) -> None:
    """Validate the ``serve``-specific fields of one task.

    ``parallel`` selects the GPU-sizing rule: under ``parallel_packed`` each
    judge claims only its ``tensor_parallel`` GPUs (so judges co-reside on the
    pool); under ``serial_full_pool`` each judge owns the whole pool (Rule 6).
    """
    label = f"tasks[{i}]"
    if task.get("family") != "A":
        problems.append(f"{label} serve task must have family 'A'")
    if task.get("mode") not in EVAL_MODES:
        problems.append(f"{label} serve 'mode' must be one of {EVAL_MODES}")
    if not task.get("judge"):
        problems.append(f"{label} serve task needs a 'judge'")

    spec = task.get("serve")
    if not isinstance(spec, dict):
        problems.append(f"{label} serve 'serve' must be an object")
        spec = {}
    for key in ("model_path", "served_model_name"):
        if not spec.get(key):
            problems.append(f"{label} serve.serve missing {key!r}")
    tp = spec.get("tensor_parallel")
    if not _is_int(tp) or tp <= 0:
        problems.append(f"{label} serve.serve.tensor_parallel must be an int > 0")

    # GPU sizing depends on the serving policy.
    gpu_count = task.get("gpu_count")
    if parallel:
        # parallel_packed — right-size to the model's tensor-parallel degree.
        if (_is_int(gpu_count) and _is_int(gpu_pool)
                and not 1 <= gpu_count <= gpu_pool):
            problems.append(
                f"{label} serve task gpu_count ({gpu_count}) must be in "
                f"[1, gpu_pool={gpu_pool}] — parallel_packed"
            )
        if _is_int(gpu_count) and _is_int(tp) and gpu_count != tp:
            problems.append(
                f"{label} serve task gpu_count ({gpu_count}) must equal "
                f"serve.tensor_parallel ({tp}) — parallel_packed right-sizes "
                f"each judge to its TP degree so judges co-reside on the pool"
            )
    elif _is_int(gpu_count) and _is_int(gpu_pool) and gpu_count != gpu_pool:
        # serial_full_pool — Rule 6, each judge owns the whole allowed pool.
        problems.append(
            f"{label} serve task must request the whole pool "
            f"(gpu_count {gpu_count} != gpu_pool {gpu_pool}) — Rule 6"
        )

    util = spec.get("gpu_memory_utilization")
    if not _is_number(util) or not 0 < util <= 1:
        problems.append(
            f"{label} serve.serve.gpu_memory_utilization must be in (0, 1]"
        )

    serves = task.get("serves")
    if not isinstance(serves, list) or not serves:
        problems.append(f"{label} serve task 'serves' must be a non-empty list")
    if not isinstance(task.get("teardown_after"), list):
        problems.append(f"{label} serve task 'teardown_after' must be a list")


def _topo_is_acyclic(tasks_by_id: dict[str, dict]) -> bool:
    """Kahn topological sort — True when the dependency graph is acyclic."""
    indegree = {tid: 0 for tid in tasks_by_id}
    for tid, task in tasks_by_id.items():
        for dep in task.get("depends_on") or []:
            if dep in tasks_by_id:
                indegree[tid] += 1
    queue = [tid for tid, d in indegree.items() if d == 0]
    seen = 0
    while queue:
        tid = queue.pop()
        seen += 1
        for other, task in tasks_by_id.items():
            if tid in (task.get("depends_on") or []):
                indegree[other] -= 1
                if indegree[other] == 0:
                    queue.append(other)
    return seen == len(tasks_by_id)


def _ancestors(task_id: str, tasks_by_id: dict[str, dict]) -> set[str]:
    """Transitive ``depends_on`` closure of a task (graph must be acyclic)."""
    out: set[str] = set()
    stack = list(tasks_by_id.get(task_id, {}).get("depends_on") or [])
    while stack:
        dep = stack.pop()
        if dep in out or dep not in tasks_by_id:
            continue
        out.add(dep)
        stack.extend(tasks_by_id[dep].get("depends_on") or [])
    return out


def _check_dependencies(
    tasks_by_id: dict[str, dict], problems: list[str],
) -> bool:
    """Validate dependency resolution + acyclicity; return True when acyclic."""
    for tid, task in tasks_by_id.items():
        for dep in task.get("depends_on") or []:
            if dep == tid:
                problems.append(f"task {tid!r} depends on itself")
            elif dep not in tasks_by_id:
                problems.append(
                    f"task {tid!r} depends on unknown task {dep!r}"
                )
    acyclic = _topo_is_acyclic(tasks_by_id)
    if not acyclic:
        problems.append("task plan has a dependency cycle")
    return acyclic


def _check_serve_chain(
    serve_tasks: list[dict], problems: list[str], parallel: bool,
) -> None:
    """Validate how a mode's ``serve`` tasks relate to each other.

    ``serial_full_pool`` (Rule 6): within each mode the ``serve`` tasks form one
    linear chain — judges load strictly one at a time (one root, every other
    judge depending on exactly one prior judge, no fork).

    ``parallel_packed``: the opposite — judges co-reside, so they must be
    **independent**; a ``serve`` that lists another same-mode ``serve`` in its
    ``depends_on`` would re-serialize them and is flagged.
    """
    by_mode: dict[object, list[dict]] = {}
    for t in serve_tasks:
        by_mode.setdefault(t.get("mode"), []).append(t)
    for mode, group in by_mode.items():
        if len(group) <= 1:
            continue
        serve_ids = {t["id"] for t in group if t.get("id")}
        if parallel:
            for t in group:
                serve_preds = [
                    d for d in (t.get("depends_on") or []) if d in serve_ids
                ]
                if serve_preds:
                    problems.append(
                        f"parallel_packed: serve task {t.get('id')!r} depends on "
                        f"other {mode} serve task(s) {serve_preds} — judges "
                        f"co-reside, so they must be independent (no serve chain)"
                    )
            continue
        roots: list[object] = []
        preds_used: list[str] = []
        for t in group:
            serve_preds = [
                d for d in (t.get("depends_on") or []) if d in serve_ids
            ]
            if len(serve_preds) == 0:
                roots.append(t.get("id"))
            elif len(serve_preds) > 1:
                problems.append(
                    f"Rule 6: serve task {t.get('id')!r} depends on multiple "
                    f"{mode} serve tasks {serve_preds} — judges run serially"
                )
            else:
                preds_used.append(serve_preds[0])
        if len(roots) != 1:
            problems.append(
                f"Rule 6: {mode} serve tasks must form one chain — found "
                f"{len(roots)} chain roots {roots} (expected exactly 1)"
            )
        if len(preds_used) != len(set(preds_used)):
            problems.append(
                f"Rule 6: a {mode} serve task is the predecessor of two judges "
                f"— the serve chain forks; judges must run strictly serially"
            )


def validate_task_plan(
    plan: object, brief: object,
    family_a_execution: str = "serial_full_pool",
) -> list[str]:
    """Return a list of problems with a Stage 5 task plan; empty == valid.

    ``brief`` is the Stage 4 ``experiment_brief.json`` the plan expands — used to
    check that every candidate configuration is covered. A non-dict ``brief``
    disables only the coverage checks. ``family_a_execution`` selects the serve
    GPU-sizing + chaining rule (``parallel_packed`` vs ``serial_full_pool``);
    Stage 6's scheduler enforces the matching runtime policy.
    """
    if not isinstance(plan, dict):
        return ["task plan must be a JSON object"]
    parallel = family_a_execution == "parallel_packed"

    problems: list[str] = []
    for key in REQUIRED_PLAN_TOP:
        if key not in plan:
            problems.append(f"missing required field: {key!r}")
    if not plan.get("evaluation_goal"):
        problems.append("evaluation_goal is empty")
    gpu_pool = plan.get("gpu_pool")
    if not _is_int(gpu_pool) or gpu_pool < 1:
        problems.append("gpu_pool must be an integer >= 1")
    if not _is_int(plan.get("iteration")):
        problems.append("iteration must be an integer")

    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        problems.append("tasks must be a non-empty list")
        return problems
    if any(not isinstance(t, dict) for t in tasks):
        problems.append("every task must be an object")
        return problems

    # Per-task structural + per-type checks.
    seen_ids: set[str] = set()
    for i, task in enumerate(tasks):
        _check_task_common(i, task, gpu_pool, problems)
        tid = task.get("id")
        if tid:
            if tid in seen_ids:
                problems.append(f"duplicate task id: {tid!r}")
            seen_ids.add(tid)
        ttype = task.get("type")
        if ttype == "eval":
            _check_eval_task(i, task, problems)
        elif ttype == "serve":
            _check_serve_task(i, task, gpu_pool, problems, parallel)
        elif ttype == "aggregate":
            if task.get("mode") not in EVAL_MODES:
                problems.append(
                    f"tasks[{i}] aggregate 'mode' must be one of {EVAL_MODES}"
                )
            if not task.get("combination_id"):
                problems.append(f"tasks[{i}] aggregate missing 'combination_id'")
        elif ttype == "compare":
            if task.get("mode") not in EVAL_MODES:
                problems.append(
                    f"tasks[{i}] compare 'mode' must be one of {EVAL_MODES}"
                )
            if not isinstance(task.get("gate"), bool):
                problems.append(f"tasks[{i}] compare 'gate' must be a boolean")

    # Tasks lacking a usable id cannot be cross-referenced — stop here.
    tasks_by_id = {t["id"]: t for t in tasks if t.get("id")}
    if len(tasks_by_id) != len(tasks):
        return problems

    acyclic = _check_dependencies(tasks_by_id, problems)

    serve_tasks = [t for t in tasks if t.get("type") == "serve"]
    eval_tasks = [t for t in tasks if t.get("type") == "eval"]
    _check_serve_chain(serve_tasks, problems, parallel)

    # Judge-bearing serve coverage — any eval whose spec names a judge (Family A
    # or a hybrid) must point at a matching serve task that brings the endpoint
    # up; otherwise Stage 6 has no VLM endpoint to call.
    for t in eval_tasks:
        if not (t.get("eval") or {}).get("judge"):
            continue
        serve_id = t.get("judge_task_id")
        if not serve_id:
            continue       # the missing judge_task_id is flagged in _check_eval_task
        serve = tasks_by_id.get(serve_id)
        if serve is None or serve.get("type") != "serve":
            problems.append(
                f"eval {t['id']!r} judge_task_id {serve_id!r} is not a serve task"
            )
            continue
        if serve_id not in (t.get("depends_on") or []):
            problems.append(
                f"eval {t['id']!r} must depend on its serve task {serve_id!r}"
            )
        want = (t.get("eval") or {}).get("judge")
        serve_judge = serve.get("judge")
        if (want and serve_judge
                and _normalize_judge(serve_judge) != _normalize_judge(want)):
            problems.append(
                f"eval {t['id']!r} judge {want!r} != serve {serve_id!r} judge "
                f"{serve_judge!r} (canonical names differ after stripping "
                f"modality suffixes — modality belongs in eval.prompt / "
                f"eval.scope, not in the judge name)"
            )

    # aggregate -> its same-config same-mode eval.
    for t in tasks:
        if t.get("type") != "aggregate":
            continue
        cid, mode = t.get("combination_id"), t.get("mode")
        deps = t.get("depends_on") or []
        if not any(
            tasks_by_id.get(d, {}).get("type") == "eval"
            and tasks_by_id[d].get("combination_id") == cid
            and tasks_by_id[d].get("mode") == mode
            for d in deps
        ):
            problems.append(
                f"aggregate {t['id']!r} must depend on the {mode} eval task for "
                f"config {cid!r}"
            )

    # Brief coverage — every candidate config has a pilot + full eval task.
    brief_configs = (
        brief.get("candidate_configs") if isinstance(brief, dict) else None
    )
    if isinstance(brief_configs, list):
        brief_ids = {
            c.get("combination_id")
            for c in brief_configs
            if isinstance(c, dict) and c.get("combination_id")
        }
        covered: dict[str, set[str]] = {cid: set() for cid in brief_ids}
        for t in eval_tasks:
            cid = t.get("combination_id")
            if cid and cid not in brief_ids:
                problems.append(
                    f"eval {t['id']!r} combination_id {cid!r} is not in the brief"
                )
            elif cid and t.get("mode") in EVAL_MODES:
                covered[cid].add(t["mode"])
        for cid in sorted(brief_ids):
            missing = sorted(set(REQUIRED_EVAL_MODES) - covered.get(cid, set()))
            if missing:
                problems.append(
                    f"brief config {cid!r} is missing a {' and '.join(missing)} "
                    f"eval task"
                )

    # The pilot gate — exactly one gating compare, every full eval behind it.
    gates = [
        t for t in tasks
        if t.get("type") == "compare" and t.get("gate") is True
    ]
    finals = [
        t for t in tasks
        if t.get("type") == "compare" and t.get("gate") is False
    ]
    if len(gates) != 1:
        problems.append(
            f"task plan needs exactly one gating compare (gate: true), "
            f"found {len(gates)}"
        )
    if not finals:
        problems.append("task plan needs a final compare task (gate: false)")

    if acyclic and len(gates) == 1:
        gate = gates[0]
        gate_anc = _ancestors(gate["id"], tasks_by_id)
        pilot_aggs = [
            t["id"] for t in tasks
            if t.get("type") == "aggregate" and t.get("mode") == "pilot"
        ]
        for agg_id in pilot_aggs:
            if agg_id not in gate_anc:
                problems.append(
                    f"gating compare {gate['id']!r} must depend on pilot "
                    f"aggregate {agg_id!r}"
                )
        gpilot = gate.get("pilot")
        if not isinstance(gpilot, dict) or not gpilot.get("pass_criteria"):
            problems.append(
                f"gating compare {gate['id']!r} needs pilot.pass_criteria"
            )
        for t in eval_tasks:
            if t.get("mode") == "full" and gate["id"] not in _ancestors(
                t["id"], tasks_by_id
            ):
                problems.append(
                    f"full eval {t['id']!r} must be gated behind the pilot "
                    f"compare {gate['id']!r}"
                )

    return problems
