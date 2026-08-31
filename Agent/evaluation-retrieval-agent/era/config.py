"""``ERAConfig`` — the project config written to ``workspaces/{project}/config.yaml``.

Field names under ``hardware`` match the overall-plan rules verbatim so the
downstream ``model_selector`` (Rule 3) can read them unchanged.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from . import __version__

TASK_FAMILIES = ("generation", "editing")
LAYOUTS = ("per_sample_dirs", "flat_files", "separate_input_dir")
SAMPLE_KEYS = ("relpath", "basename", "index", "manifest")

# Sub-agent tiers for the Stage 2-4 debate. Each tier is an `era-<tier>`
# sub-agent under plugin/agents/ carrying a fixed Claude model; `agent_modes`
# maps each debate stage to one of these.
AGENT_TIERS = ("heavy", "standard", "light")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class HardwareConfig:
    gpu_model: str = ""
    visible_gpu_ids: list[int] = field(default_factory=list)
    reserve_gpu_ids: list[int] = field(default_factory=list)
    max_gpus_per_run: int = 0
    per_gpu_memory_gb: float = 0.0
    headroom: float = 0.90
    safety_margin_gb: float = 2.0
    driver_version: str = ""
    cuda_version: str = ""
    probe_ok: bool = False


@dataclass
class CheckpointsConfig:
    local_model_root: str = ""
    detected: list[str] = field(default_factory=list)
    user_checkpoints: list[dict] = field(default_factory=list)


@dataclass
class ServingConfig:
    backend: str = "ms-swift"
    fallbacks: list[str] = field(default_factory=lambda: ["vllm", "lmdeploy"])
    openai_compatible: bool = True
    default_host: str = "127.0.0.1"
    port_range: list[int] = field(default_factory=lambda: [8000, 8099])


@dataclass
class DataConfig:
    data_root: str = ""                       # common parent of the method dirs
    layout: str = "per_sample_dirs"           # see LAYOUTS
    methods: list[dict] = field(default_factory=list)
    # ^ one entry per generation method: {method_id, path, output_file}
    sample_glob: str = ""                     # method.path -> leaf sample dir
    sample_count: int = 0                     # samples per method (probed at init)
    # Per-iter cap on samples-per-method actually evaluated and surfaced in
    # human review. Default 50; if ``sample_count`` is smaller, the effective
    # cap is ``sample_count`` (see :meth:`ERAConfig.effective_iter_sample_count`).
    # Stage 6 runners iterate the *first* ``effective_iter_sample_count``
    # entries of the sorted ``sample_glob`` per method, so the same N samples
    # are scored across iters (apples-to-apples for Stage 9's trajectory
    # comparison). The Stage 4 brief gate enforces that
    # ``pilot.sample_count <= iter_sample_count`` and
    # ``validation.sample_size == effective_iter_sample_count``.
    iter_sample_count: int = 50
    # When True (default), Stage 2's brainstorm sub-agents read the
    # operator's pre-existing annotations under ``<data_root>/annotations/``
    # as "operator-flagged failure modes" and weight their candidate
    # evaluators toward catching those problems. Set False to skip this
    # input (Stage 2 then proceeds from literature.md alone).
    use_annotation_evidence: bool = True
    input_roles: dict[str, str] = field(default_factory=dict)
    # ^ co-located input images: role name -> filename glob
    input_root: str = ""                      # only for layout=separate_input_dir
    sample_key: str = "relpath"               # see SAMPLE_KEYS
    pairing_manifest: str = ""
    image_extensions: list[str] = field(
        default_factory=lambda: [".png", ".jpg", ".jpeg", ".webp"]
    )


@dataclass
class CredentialsConfig:
    env_file: str = ""
    openai: bool = False
    anthropic: bool = False
    google: bool = False


@dataclass
class BudgetConfig:
    # 0.0 == no paid-API spend (rely on locally-served VLM judges + metrics);
    # a positive value caps total paid-API spend in USD.
    api_cost_cap_usd: float = 0.0
    wallclock_cap_hours: float = 24.0


@dataclass
class AgentModesConfig:
    """Which sub-agent tier runs each tiered pipeline stage.

    Each value is a tier in ``AGENT_TIERS``; the stage orchestrator dispatches
    its persona sub-agents to the matching ``era-<tier>`` sub-agent.
    """
    plan_brainstorm: str = "standard"   # Stage 2 — idea generators
    multi_review: str = "light"         # Stage 3 — debate critics
    plan_decision: str = "heavy"        # Stage 4 — synthesis & decision
    experiment_plan: str = "heavy"      # Stage 5 — experiment planner
    full_experiment: str = "standard"   # Stage 6 — experiment runner


@dataclass
class DebateConfig:
    """The Stage 2-4 idea-generation / debate loop."""
    # Max Stage 2->4 ADVANCE/REVISE rounds per iteration; >= 1. The loop is
    # forced to ADVANCE once this cap is reached.
    max_rounds: int = 4


@dataclass
class ReactConfig:
    """Stage 9 — the ReAct iteration gate.

    The Stage 9 advisor consults these to decide ADVANCE vs REVISE_*; the
    deterministic ``react_tick`` forces ADVANCE once ``max_iterations`` is hit.
    """
    # Hard cap on total iterations (incl. iter_001). >= 1.
    max_iterations: int = 5
    # The advisor's strong-evidence cue for ADVANCE — a config whose endorsement
    # rate is at least this *and* whose sample count is at least
    # ``min_alignment_samples`` is considered well-aligned with the human.
    endorsement_threshold: float = 0.80
    # Minimum samples needed before an endorsement rate is trusted.
    min_alignment_samples: int = 20


# Experiment-execution policy (Stage 5-6). The Rule 6 family-A serving mode
# and the family-B scheduling mode each pick from a fixed set.
FAMILY_A_EXECUTION = ("serial_full_pool", "parallel_packed")
FAMILY_B_SCHEDULE = ("before_after_family_a", "parallel_on_unallowed_gpus")


@dataclass
class ExperimentConfig:
    """Stage 5-6 experiment execution knobs.

    ``family_a_execution`` / ``family_b_schedule`` encode Rule 6's serving
    policy; the rest tune the Stage 6 GPU scheduler and recovery loop.

    ``family_a_execution`` picks how Family-A VLM judges share the pool:
    - ``"parallel_packed"`` (default) — each judge claims only its
      ``tensor_parallel`` GPUs, multiple judges co-reside on **disjoint**
      GPU subsets, and Family-B evals backfill whatever GPUs are left. This
      saturates the pool: a 35B judge on tp 2 + a 72B judge on tp 4 run
      concurrently on an 8-GPU host instead of one-at-a-time. Co-residency is
      correctness-safe — each judge is a separate vLLM server on its own
      GPUs/port; the only shared resource is host CPU/RAM/PCIe, bounded by
      ``max_parallel_runners`` / ``max_concurrent_judges``.
    - ``"serial_full_pool"`` — one judge at a time, owning the whole pool
      (strict measurement isolation; the pre-v0.1.7.1 behaviour).

    ``codex_reviewer`` is **opt-in** (default off): when true, Stage 6 reviews
    each generated evaluator runner with a separate Codex sub-agent
    (``era-codex-reviewer`` via the ``codex`` MCP) for an independent
    second-AI perspective; when false it self-reviews the runner inline. Set it
    only when the ``codex`` CLI is installed and an extra paid API is wanted.
    """
    # Default flipped in v0.1.7.1 (Phase D-6): right-sized judges co-reside on
    # disjoint GPU subsets so the pool stays saturated. ``serial_full_pool``
    # (one judge owns the pool) stays operator-pinnable for strict isolation.
    family_a_execution: str = "parallel_packed"
    # Default flipped in v0.1.7 (Phase D-1): Family-B evals may run on GPUs
    # outside a resident judge's pool, instead of waiting for the judge to
    # finish. Operators can still pin "before_after_family_a" for strict
    # measurement isolation.
    family_b_schedule: str = "parallel_on_unallowed_gpus"
    free_gpu_threshold_mb: int = 2000     # a GPU is "free" below this used-VRAM
    poll_interval_s: int = 45             # Stage 6 marker-file poll cadence
    serve_startup_timeout_s: int = 600    # judge endpoint dry-probe timeout
    max_task_retries: int = 1             # bounded re-runs of a failed task
    heartbeat_timeout_s: int = 1800       # a runner silent this long is "hung"
    pilot_first: bool = True              # run the pilot pass before the full run
    codex_reviewer: bool = False          # use the separate Codex code reviewer
    # Phase D-1 cap on concurrent in-flight runners. 0 = no cap (use however
    # many GPUs the scheduler can fit); positive N caps claim_batch to at
    # most N tasks per batch even when more GPUs are free. Useful when host
    # CPU / disk cannot sustain N concurrent runners.
    max_parallel_runners: int = 0
    # Phase D-6 cap on co-resident Family-A judges under ``parallel_packed``.
    # 0 = uncapped (pack as many judges as fit on the pool); positive N blocks
    # additional ``serve`` tasks once N judges are already resident — useful
    # when judge startup / host RAM pressure (not GPU count) is the bottleneck.
    # No effect under ``serial_full_pool`` (which is always one judge).
    max_concurrent_judges: int = 0
    # Auto-validation gate (Phase C consumer; operator pins these at /era:init):
    # after each iter's annotated-eval pass, every eval method's per-sample
    # scores are compared against the operator's pre-existing annotations.
    # PASS_RATE = (# samples where method matches operator) / (total annotated).
    # RECALL_RATE = (# bad samples method also flagged) / (total bad samples).
    # A method "passes" iff PASS_RATE >= pass_threshold AND RECALL_RATE >=
    # recall_threshold. If fewer than ``min_samples`` annotations exist, the
    # gate is skipped (the pipeline falls through to Stage 8 human eval as
    # today).
    auto_validate_pass_threshold: float = 0.70
    auto_validate_recall_threshold: float = 0.60
    auto_validate_min_samples: int = 10
    # Minimum number of configs that must clear BOTH pass + recall thresholds
    # for ``any_passed`` to be True. Set to 1 to restore "any single config
    # passing proceeds" behaviour; default 3 enforces candidate diversity
    # (operator directive: a single-config full round loses the comparison
    # point of running multiple candidates).
    auto_validate_min_passing: int = 3


def _coerce(dc_type: type, data: Any) -> Any:
    """Build a (nested) dataclass from a dict, ignoring unknown keys."""
    if isinstance(data, dc_type):
        return data
    if not isinstance(data, dict):
        return dc_type()
    names = {f.name for f in dataclasses.fields(dc_type)}
    return dc_type(**{k: v for k, v in data.items() if k in names})


@dataclass
class ERAConfig:
    project_name: str = ""
    mission: str = ""
    task_family: str = "editing"
    task_adapter: str = "generic"
    created_at: str = ""
    era_version: str = __version__
    workspaces_dir: str = ""
    iteration_dirs: bool = True
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    checkpoints: CheckpointsConfig = field(default_factory=CheckpointsConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    credentials: CredentialsConfig = field(default_factory=CredentialsConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    agent_modes: AgentModesConfig = field(default_factory=AgentModesConfig)
    debate: DebateConfig = field(default_factory=DebateConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    react: ReactConfig = field(default_factory=ReactConfig)

    # Pipeline lifecycle (stage / stage_index / run_state / iteration) lives
    # solely in status.json — config.yaml carries only static project facts.
    _SCALARS = (
        "project_name", "mission", "task_family", "task_adapter",
        "created_at", "era_version", "workspaces_dir", "iteration_dirs",
    )

    @classmethod
    def from_params(cls, params: dict) -> "ERAConfig":
        """Build a config from the confirmed-facts dict the init flow passes."""
        cfg = cls()
        for key in cls._SCALARS:
            if params.get(key) is not None:
                setattr(cfg, key, params[key])
        cfg.hardware = _coerce(HardwareConfig, params.get("hardware"))
        cfg.checkpoints = _coerce(CheckpointsConfig, params.get("checkpoints"))
        cfg.serving = _coerce(ServingConfig, params.get("serving"))
        cfg.data = _coerce(DataConfig, params.get("data"))
        cfg.credentials = _coerce(CredentialsConfig, params.get("credentials"))
        cfg.budget = _coerce(BudgetConfig, params.get("budget"))
        cfg.agent_modes = _coerce(AgentModesConfig, params.get("agent_modes"))
        cfg.debate = _coerce(DebateConfig, params.get("debate"))
        cfg.experiment = _coerce(ExperimentConfig, params.get("experiment"))
        cfg.react = _coerce(ReactConfig, params.get("react"))
        if not cfg.created_at:
            cfg.created_at = _now_iso()
        cfg.era_version = __version__
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ERAConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_params(data)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def effective_iter_sample_count(self) -> int:
        """Per-method samples actually evaluated this iter — the cap, but
        clamped to ``data.sample_count`` when the operator set a cap larger
        than the available data.

        Falls back to ``data.iter_sample_count`` when ``sample_count`` is 0
        (pre-probe / loose-fixture state), so call sites can rely on a
        non-zero return without checking the probe ran.
        """
        cap = self.data.iter_sample_count
        total = self.data.sample_count
        if total > 0:
            return min(cap, total)
        return cap

    def to_commented_yaml(self) -> str:
        """Serialize to YAML with section header comments."""
        d = self.to_dict()
        out: list[str] = [
            "# ERA project config — generated by /era:init (Stage 0).",
            f"# era_version: {self.era_version}",
            "# Edit spec.md for human-readable intent; this file is machine state.",
            "",
        ]

        def section(title: str, keys: list[str]) -> None:
            out.append(f"# ---- {title} ----")
            sub = {k: d[k] for k in keys}
            out.append(
                yaml.safe_dump(
                    sub, sort_keys=False, allow_unicode=True,
                    default_flow_style=False,
                ).rstrip()
            )
            out.append("")

        section("Identity", [
            "project_name", "mission", "task_family", "task_adapter",
            "created_at", "era_version",
        ])
        section("Workspace", ["workspaces_dir", "iteration_dirs"])
        section("Hardware (consumed by Stage 7 model_selector)", ["hardware"])
        section("Checkpoints", ["checkpoints"])
        section("Serving", ["serving"])
        section("Data", ["data"])
        section("Credentials (presence-only; secrets stay in .env)",
                ["credentials"])
        section("Budget", ["budget"])
        section("Agent modes (Stage 2-6 sub-agent tiers)",
                ["agent_modes"])
        section("Debate (Stage 2-4 idea-generation loop)", ["debate"])
        section("Experiment (Stage 5-6 execution policy)", ["experiment"])
        section("ReAct (Stage 9 iteration gate)", ["react"])
        return "\n".join(out) + "\n"

    def validate(self) -> list[str]:
        """Return a list of human-readable problems; empty list == valid."""
        problems: list[str] = []
        if not self.project_name:
            problems.append("project_name is empty")
        if self.task_family not in TASK_FAMILIES:
            problems.append(
                f"task_family must be one of {TASK_FAMILIES}, got "
                f"{self.task_family!r}"
            )

        hw = self.hardware
        if not hw.visible_gpu_ids:
            problems.append("hardware.visible_gpu_ids is empty")
        if not set(hw.reserve_gpu_ids).issubset(set(hw.visible_gpu_ids)):
            problems.append(
                "hardware.reserve_gpu_ids must be a subset of visible_gpu_ids"
            )
        if hw.max_gpus_per_run < 0:
            problems.append("hardware.max_gpus_per_run must be >= 0")

        data = self.data
        if data.sample_count <= 0:
            problems.append("data.sample_count must be > 0")
        if data.iter_sample_count < 1:
            problems.append(
                "data.iter_sample_count must be >= 1 (the per-iter "
                "samples-per-method cap; a 0-sample eval is a config bug)"
            )
        if data.layout not in LAYOUTS:
            problems.append(
                f"data.layout must be one of {LAYOUTS}, got {data.layout!r}"
            )
        if data.sample_key not in SAMPLE_KEYS:
            problems.append(
                f"data.sample_key must be one of {SAMPLE_KEYS}, got "
                f"{data.sample_key!r}"
            )
        if not data.methods:
            problems.append("data.methods must have at least one entry")
        for i, method in enumerate(data.methods):
            for key in ("method_id", "path", "output_file"):
                if not method.get(key):
                    problems.append(
                        f"data.methods[{i}] is missing {key!r}"
                    )
        if self.task_family == "editing" and not data.input_roles:
            problems.append(
                "data.input_roles must be non-empty for an editing task"
            )

        am = self.agent_modes
        for stage_name in (
            "plan_brainstorm", "multi_review", "plan_decision",
            "experiment_plan", "full_experiment",
        ):
            tier = getattr(am, stage_name)
            if tier not in AGENT_TIERS:
                problems.append(
                    f"agent_modes.{stage_name} must be one of {AGENT_TIERS}, "
                    f"got {tier!r}"
                )
        if self.debate.max_rounds < 1:
            problems.append("debate.max_rounds must be >= 1")

        exp = self.experiment
        if exp.family_a_execution not in FAMILY_A_EXECUTION:
            problems.append(
                f"experiment.family_a_execution must be one of "
                f"{FAMILY_A_EXECUTION}, got {exp.family_a_execution!r}"
            )
        if exp.family_b_schedule not in FAMILY_B_SCHEDULE:
            problems.append(
                f"experiment.family_b_schedule must be one of "
                f"{FAMILY_B_SCHEDULE}, got {exp.family_b_schedule!r}"
            )
        if exp.max_task_retries < 0:
            problems.append("experiment.max_task_retries must be >= 0")
        if exp.poll_interval_s <= 0:
            problems.append("experiment.poll_interval_s must be > 0")
        if exp.max_parallel_runners < 0:
            problems.append("experiment.max_parallel_runners must be >= 0")
        if not 0.0 <= exp.auto_validate_pass_threshold <= 1.0:
            problems.append(
                "experiment.auto_validate_pass_threshold must be in [0.0, 1.0]"
            )
        if not 0.0 <= exp.auto_validate_recall_threshold <= 1.0:
            problems.append(
                "experiment.auto_validate_recall_threshold must be in [0.0, 1.0]"
            )
        if exp.auto_validate_min_passing < 1:
            problems.append(
                "experiment.auto_validate_min_passing must be >= 1"
            )
        if exp.auto_validate_min_samples < 1:
            problems.append(
                "experiment.auto_validate_min_samples must be >= 1"
            )

        rx = self.react
        if rx.max_iterations < 1:
            problems.append("react.max_iterations must be >= 1")
        if not 0.0 <= rx.endorsement_threshold <= 1.0:
            problems.append("react.endorsement_threshold must be in [0.0, 1.0]")
        if rx.min_alignment_samples < 1:
            problems.append("react.min_alignment_samples must be >= 1")
        return problems
