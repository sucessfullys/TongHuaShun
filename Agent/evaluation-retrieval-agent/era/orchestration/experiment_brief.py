"""Validate the Stage 4 experiment-ready handoff (``design/experiment_brief.json``).

``experiment_brief.json`` is the machine-readable plan the future Stage 5
(Experiment Planning) executes **verbatim** — without re-debating or inferring
missing fields. ERA's model/metric-testing rules and the v0.1.3 handoff contract
are deterministic constraints on it, so they are checked here rather than
trusted to the agent:

- **Rule 5** — at most 10 evaluator configurations per round, slot-allocated
  (1-2 metric-only baselines, <= 3 VLM scale-ablation, <= 3 hybrids, <= 3
  task-specific, <= 1 reviewer wildcard).
- **Rule 4** — the scale-comparison invariant: >= 2 *distinct* VLM judge scales
  probed, or >= 1 VLM scale + >= 1 metric-only baseline.
- **Handoff completeness** — every config carries the executable fields Stage 5
  needs (the family-appropriate combination tuple, inputs, estimates,
  feasibility); the ``validation`` / ``pilot`` / ``resource_estimate`` /
  ``pivot_matrix`` blocks are fully populated.

``validate_experiment_brief`` returns a list of human-readable problems (empty
== valid). Stage 4 runs it after writing the brief; any problem is itself a
REVISE trigger. When ``known_hypotheses`` is supplied — the set of ``H<k>`` IDs
declared as headings in ``design/hypotheses.md`` — every config
``hypothesis_id`` must resolve to one; without it, only the field's presence is
checked. ``extract_hypothesis_ids`` parses that set out of a hypotheses.md file.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from numbers import Number
from pathlib import Path

SLOTS = ("metric_baseline", "vlm_scale", "hybrid", "task_specific", "wildcard")
FAMILIES = ("A", "B", "hybrid")
MAX_CONFIGS = 10

# Per-slot caps (Rule 5). metric_baseline also has a *minimum* of 1 (below).
SLOT_CAPS = {
    "metric_baseline": 2, "vlm_scale": 3, "hybrid": 3, "task_specific": 3,
    "wildcard": 1,
}

# The family a slot is locked to; task_specific / wildcard accept any family.
SLOT_FAMILY = {"metric_baseline": "B", "vlm_scale": "A", "hybrid": "hybrid"}

REQUIRED_TOP = (
    "evaluation_goal", "candidate_configs", "validation", "pilot",
    "resource_estimate", "scale_comparison", "pivot_matrix",
)
# Per-config fields every configuration must carry, whatever its family.
REQUIRED_CONFIG = (
    "combination_id", "slot", "family", "hypothesis_id", "scope",
    "gpu_estimate",
)
REQUIRED_VALIDATION = ("human_rating_set", "alignment_metric", "debiasing")
REQUIRED_PILOT = ("sample_count", "go_no_go")
REQUIRED_RESOURCE = ("gpu_hours", "wallclock_hours", "api_cost_usd")

# A hypotheses.md hypothesis heading: ``## H<k>: <title>``.
_HYPOTHESIS_HEADING = re.compile(r"^##\s+(H\d+)\b", re.MULTILINE)


def extract_hypothesis_ids(text: str) -> set[str]:
    """Extract the ``H<k>`` IDs declared as headings in a hypotheses.md file."""
    return set(_HYPOTHESIS_HEADING.findall(text))


def load_pivot_actions(iter_dir: Path) -> set[str]:
    """The set of ``pivot_matrix[*].action`` strings from a Stage 4 brief.

    Both ``init-experiment`` (when seeding skipped tasks) and
    ``check-experiment-completion`` (when excusing skipped tasks) gate on this
    set: a Stage 6 ``eval``-task skip is only honored when its ``skip_proof``
    is one of these pre-validated Stage 4 decisions, so a runner cannot use the
    skip path as a back door for runtime scope-reduction.

    An empty set means the brief is missing, unreadable, or has no
    ``pivot_matrix`` entries; callers fall back to "any non-empty proof string
    is accepted" in that case, since the gate's purpose is to forbid silent
    drops, not to require a specific catalog when none exists.
    """
    path = iter_dir / "design" / "experiment_brief.json"
    if not path.is_file():
        return set()
    try:
        brief = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    actions: set[str] = set()
    for entry in brief.get("pivot_matrix", []) or []:
        if isinstance(entry, dict) and entry.get("action"):
            actions.add(str(entry["action"]))
    return actions


def _is_number(value: object) -> bool:
    """True for an int/float (but not bool, which is a subclass of int)."""
    return isinstance(value, Number) and not isinstance(value, bool)


def _check_config(
    i: int,
    cfg: object,
    problems: list[str],
    seen_ids: set,
    slot_counts: dict[str, int],
    known_hypotheses: set | None,
) -> None:
    """Validate one ``candidate_configs`` entry, appending any problems."""
    if not isinstance(cfg, dict):
        problems.append(f"candidate_configs[{i}] must be an object")
        return

    for key in REQUIRED_CONFIG:
        if not cfg.get(key):
            problems.append(f"candidate_configs[{i}] missing {key!r}")

    # combination_id uniqueness.
    cid = cfg.get("combination_id")
    if cid:
        if cid in seen_ids:
            problems.append(f"duplicate combination_id: {cid!r}")
        seen_ids.add(cid)

    # inputs_needed must be a non-empty list of input roles.
    inputs = cfg.get("inputs_needed")
    if not isinstance(inputs, list) or not inputs:
        problems.append(
            f"candidate_configs[{i}] 'inputs_needed' must be a non-empty list"
        )

    # feasible must be present and true — Stage 5 cannot run an infeasible config.
    if "feasible" not in cfg:
        problems.append(f"candidate_configs[{i}] missing 'feasible'")
    elif cfg["feasible"] is not True:
        problems.append(
            f"candidate_configs[{i}] has feasible={cfg['feasible']!r} — only "
            f"feasible configs may ship in the brief"
        )

    # cost_estimate_usd must be present and a non-negative number.
    cost = cfg.get("cost_estimate_usd")
    if not _is_number(cost) or cost < 0:
        problems.append(
            f"candidate_configs[{i}] 'cost_estimate_usd' must be a number >= 0"
        )

    # hypothesis_id must resolve to a hypotheses.md heading, when the set is known.
    hid = cfg.get("hypothesis_id")
    if hid and known_hypotheses is not None and hid not in known_hypotheses:
        problems.append(
            f"candidate_configs[{i}] hypothesis_id {hid!r} is not defined in "
            f"hypotheses.md"
        )

    slot = cfg.get("slot")
    if slot in SLOTS:
        slot_counts[slot] += 1
    elif slot is not None:
        problems.append(
            f"candidate_configs[{i}].slot={slot!r} is not one of {SLOTS}"
        )

    family = cfg.get("family")
    if family is not None and family not in FAMILIES:
        problems.append(
            f"candidate_configs[{i}].family={family!r} is not one of {FAMILIES}"
        )

    # slot <-> family consistency.
    if slot in SLOT_FAMILY and family in FAMILIES:
        want = SLOT_FAMILY[slot]
        if family != want:
            problems.append(
                f"candidate_configs[{i}] slot {slot!r} requires family "
                f"{want!r}, got {family!r}"
            )

    # Family-appropriate combination-tuple fields — the executable judge/metric.
    if family == "A":
        if not cfg.get("judge"):
            problems.append(f"candidate_configs[{i}] family 'A' needs a 'judge'")
        if not cfg.get("prompt"):
            problems.append(f"candidate_configs[{i}] family 'A' needs a 'prompt'")
    elif family == "B":
        if not cfg.get("metric_subfamily"):
            problems.append(
                f"candidate_configs[{i}] family 'B' needs a 'metric_subfamily'"
            )
    elif family == "hybrid":
        if not (cfg.get("judge") or cfg.get("metric_subfamily")):
            problems.append(
                f"candidate_configs[{i}] family 'hybrid' needs a 'judge' or a "
                f"'metric_subfamily'"
            )

    # Phase C-2.5 (Step 2b) — optional lineage field naming the prior-
    # iter ancestor combination_ids this config derives from (e.g. via
    # an evolution_proposal mutation). Pure-novel configs leave it
    # empty.
    parents = cfg.get("parent_hypothesis_ids")
    if parents is not None:
        if not isinstance(parents, list) or not all(
            isinstance(p, str) and p for p in parents
        ):
            problems.append(
                f"candidate_configs[{i}].parent_hypothesis_ids must be a "
                f"list of non-empty strings"
            )


def _check_block(
    brief: dict, key: str, required: tuple[str, ...], problems: list[str],
) -> None:
    """Validate a required nested object block and its mandatory sub-fields."""
    block = brief.get(key)
    if not isinstance(block, dict):
        if key in brief:
            problems.append(f"{key} must be an object")
        return
    for sub in required:
        if not block.get(sub):
            problems.append(f"{key} missing {sub!r}")


def validate_experiment_brief(
    brief: object, known_hypotheses: Iterable[str] | None = None,
    iter_sample_count: int | None = None,
    must_include_configs: Iterable[str] | None = None,
) -> list[str]:
    """Return a list of problems with an experiment brief; empty == valid.

    When ``iter_sample_count`` is supplied (the operator-set per-iter cap
    from ``config.yaml`` — already capped to ``data.sample_count`` by the
    caller), the brief gate also enforces:

    - ``validation.sample_size == iter_sample_count`` — the full pass uses
      the cap directly; Stage 4 echoes it, never overrides it.
    - ``pilot.sample_count <= iter_sample_count`` — the pilot is a subset
      of the iter window.

    When ``must_include_configs`` is supplied (the prior iter's
    ``evolution_state.must_include_configs`` — Phase C-2.5 elitism), the
    brief gate also enforces:

    - ``candidate_configs ⊇ must_include_configs`` (by ``combination_id``)
      — every config the prior iter's Stage 9 advisor marked must-include
      must appear in this iter's brief, so Stage 6 re-evaluates the
      proven configs on the new sample window alongside any new
      complementary candidates.
    """
    if not isinstance(brief, dict):
        return ["experiment brief must be a JSON object"]

    known = set(known_hypotheses) if known_hypotheses is not None else None
    problems: list[str] = []

    for key in REQUIRED_TOP:
        if key not in brief:
            problems.append(f"missing required field: {key!r}")
    if not brief.get("evaluation_goal"):
        problems.append("evaluation_goal is empty")
    if not brief.get("scale_comparison"):
        problems.append("scale_comparison is empty")

    configs = brief.get("candidate_configs")
    if not isinstance(configs, list) or not configs:
        problems.append("candidate_configs must be a non-empty list")
        configs = []
    if len(configs) > MAX_CONFIGS:
        problems.append(
            f"candidate_configs has {len(configs)} entries — Rule 5 caps a "
            f"round at {MAX_CONFIGS}"
        )

    seen_ids: set = set()
    slot_counts = {s: 0 for s in SLOTS}
    for i, cfg in enumerate(configs):
        _check_config(i, cfg, problems, seen_ids, slot_counts, known)

    if configs:
        if slot_counts["metric_baseline"] < 1:
            problems.append("Rule 5: need >= 1 metric_baseline configuration")
        for slot, cap in SLOT_CAPS.items():
            if slot_counts[slot] > cap:
                problems.append(
                    f"Rule 5: slot {slot!r} has {slot_counts[slot]} configs "
                    f"— the cap is {cap}"
                )
        # Rule 4 — the scale-comparison invariant: >= 2 *distinct* VLM judge
        # scales, or >= 1 VLM scale + >= 1 metric-only baseline.
        distinct_vlm = {
            cfg.get("judge")
            for cfg in configs
            if isinstance(cfg, dict) and cfg.get("slot") == "vlm_scale"
            and cfg.get("judge")
        }
        if not (len(distinct_vlm) >= 2
                or (slot_counts["vlm_scale"] >= 1
                    and slot_counts["metric_baseline"] >= 1)):
            problems.append(
                "Rule 4: scale-comparison invariant unmet — need >= 2 distinct "
                "vlm_scale judges, or >= 1 vlm_scale + >= 1 metric_baseline"
            )

    # Phase C-2.5 (Step 2a) — elitism: every must_include_config from the
    # prior iter's Stage 9 advisor must appear in this iter's brief by
    # ``combination_id``.
    if must_include_configs is not None:
        required = [c for c in must_include_configs if isinstance(c, str) and c]
        if required:
            present_ids = {
                cfg.get("combination_id") for cfg in configs
                if isinstance(cfg, dict) and cfg.get("combination_id")
            }
            missing = [c for c in required if c not in present_ids]
            if missing:
                problems.append(
                    "Phase C-2.5 elitism: candidate_configs is missing "
                    "must_include_configs (carry-forward from prior iter's "
                    f"Stage 9 advisor): {sorted(missing)}. These configs "
                    "cleared the prior iter's C-2 gate (pass/recall) and "
                    "must be re-evaluated on this iter's sample window "
                    "alongside any new complementary candidates."
                )

    # validation / pilot / resource_estimate blocks.
    _check_block(brief, "validation", REQUIRED_VALIDATION, problems)
    validation = brief.get("validation")
    if isinstance(validation, dict):
        size = validation.get("sample_size")
        if not (_is_number(size) and size > 0):
            problems.append("validation.sample_size must be a number > 0")
        elif (iter_sample_count is not None
              and size != iter_sample_count):
            problems.append(
                f"validation.sample_size ({size}) must equal the operator's "
                f"per-iter cap ({iter_sample_count} = "
                f"min(data.iter_sample_count, data.sample_count)) — the full "
                f"pass uses the cap directly; the brief author may not "
                f"override it"
            )

    _check_block(brief, "pilot", REQUIRED_PILOT, problems)
    pilot = brief.get("pilot")
    if isinstance(pilot, dict):
        size = pilot.get("sample_count")
        if not (_is_number(size) and size > 0):
            problems.append("pilot.sample_count must be a number > 0")
        elif (iter_sample_count is not None
              and size > iter_sample_count):
            problems.append(
                f"pilot.sample_count ({size}) must be <= the per-iter cap "
                f"({iter_sample_count}) — the pilot is a subset of the iter "
                f"window, not a larger set"
            )

    resource = brief.get("resource_estimate")
    if isinstance(resource, dict):
        for sub in REQUIRED_RESOURCE:
            value = resource.get(sub)
            if not _is_number(value) or value < 0:
                problems.append(
                    f"resource_estimate.{sub} must be a number >= 0"
                )
    elif "resource_estimate" in brief:
        problems.append("resource_estimate must be an object")

    # pivot_matrix — a non-empty list of {pilot_outcome, action} rows.
    pivot = brief.get("pivot_matrix")
    if not isinstance(pivot, list) or not pivot:
        if "pivot_matrix" in brief:
            problems.append("pivot_matrix must be a non-empty list")
    else:
        for i, row in enumerate(pivot):
            if (not isinstance(row, dict) or not row.get("pilot_outcome")
                    or not row.get("action")):
                problems.append(
                    f"pivot_matrix[{i}] needs 'pilot_outcome' and 'action'"
                )

    return problems
