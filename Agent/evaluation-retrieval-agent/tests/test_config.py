"""Tests for ERAConfig — round-trip serialization and validation."""

from __future__ import annotations

from pathlib import Path

from conftest import valid_params

from era.config import ERAConfig


def test_round_trip(tmp_path: Path, per_sample_root: Path):
    params = valid_params(tmp_path, per_sample_root)
    cfg = ERAConfig.from_params(params)
    assert cfg.validate() == []

    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(cfg.to_commented_yaml(), encoding="utf-8")
    reloaded = ERAConfig.from_yaml(yaml_path)
    assert reloaded.to_dict() == cfg.to_dict()


def test_commented_yaml_has_sections(tmp_path: Path, per_sample_root: Path):
    cfg = ERAConfig.from_params(valid_params(tmp_path, per_sample_root))
    text = cfg.to_commented_yaml()
    assert "# ---- Hardware" in text
    assert "# ---- Data ----" in text


def test_commented_yaml_omits_pipeline_stage(
    tmp_path: Path, per_sample_root: Path
):
    """Lifecycle `stage` lives in status.json — never serialized to config.yaml."""
    cfg = ERAConfig.from_params(valid_params(tmp_path, per_sample_root))
    text = cfg.to_commented_yaml()
    assert "Pipeline state" not in text
    assert "\nstage:" not in text


def test_legacy_stage_key_ignored(tmp_path: Path, per_sample_root: Path):
    """A config.yaml from an older ERA that still carries `stage:` parses fine."""
    params = valid_params(tmp_path, per_sample_root)
    params["stage"] = "research"  # stray legacy lifecycle field
    cfg = ERAConfig.from_params(params)  # must not raise
    assert cfg.validate() == []
    assert not hasattr(cfg, "stage")


def test_validate_empty_visible_gpus(tmp_path: Path, per_sample_root: Path):
    params = valid_params(tmp_path, per_sample_root)
    params["hardware"]["visible_gpu_ids"] = []
    problems = ERAConfig.from_params(params).validate()
    assert any("visible_gpu_ids" in p for p in problems)


def test_validate_reserve_not_subset(tmp_path: Path, per_sample_root: Path):
    params = valid_params(tmp_path, per_sample_root)
    params["hardware"]["reserve_gpu_ids"] = [99]
    problems = ERAConfig.from_params(params).validate()
    assert any("reserve_gpu_ids" in p for p in problems)


def test_validate_zero_samples(tmp_path: Path, per_sample_root: Path):
    params = valid_params(tmp_path, per_sample_root)
    params["data"]["sample_count"] = 0
    problems = ERAConfig.from_params(params).validate()
    assert any("sample_count" in p for p in problems)


def test_validate_no_methods(tmp_path: Path, per_sample_root: Path):
    params = valid_params(tmp_path, per_sample_root)
    params["data"]["methods"] = []
    problems = ERAConfig.from_params(params).validate()
    assert any("methods" in p for p in problems)


def test_validate_method_missing_output_file(
    tmp_path: Path, per_sample_root: Path
):
    params = valid_params(tmp_path, per_sample_root)
    params["data"]["methods"][0].pop("output_file")
    problems = ERAConfig.from_params(params).validate()
    assert any("output_file" in p for p in problems)


def test_budget_default_is_zero():
    from era.config import BudgetConfig
    assert BudgetConfig().api_cost_cap_usd == 0.0


def test_validate_bad_task_family(tmp_path: Path, per_sample_root: Path):
    params = valid_params(tmp_path, per_sample_root)
    params["task_family"] = "nonsense"
    problems = ERAConfig.from_params(params).validate()
    assert any("task_family" in p for p in problems)


def test_unknown_keys_ignored(tmp_path: Path, per_sample_root: Path):
    params = valid_params(tmp_path, per_sample_root)
    params["hardware"]["bogus_field"] = 123
    cfg = ERAConfig.from_params(params)  # must not raise
    assert cfg.validate() == []


# ---- agent_modes / debate (Stage 2-4) -----------------------------------

def test_agent_modes_defaults():
    from era.config import AgentModesConfig
    am = AgentModesConfig()
    assert am.plan_brainstorm == "standard"
    assert am.multi_review == "light"
    assert am.plan_decision == "heavy"
    assert am.experiment_plan == "heavy"
    assert am.full_experiment == "standard"


def test_validate_bad_experiment_plan_tier(
    tmp_path: Path, per_sample_root: Path
):
    params = valid_params(tmp_path, per_sample_root)
    params["agent_modes"] = {"experiment_plan": "turbo"}
    problems = ERAConfig.from_params(params).validate()
    assert any("agent_modes.experiment_plan" in p for p in problems)


def test_debate_default_max_rounds():
    from era.config import DebateConfig
    assert DebateConfig().max_rounds == 4


def test_experiment_config_defaults():
    from era.config import ExperimentConfig
    exp = ExperimentConfig()
    # v0.1.7.1 Phase D-6: default flipped to parallel_packed so right-sized
    # judges co-reside on disjoint GPU subsets. serial_full_pool (Rule 6) is
    # still operator-pinnable.
    assert exp.family_a_execution == "parallel_packed"
    # v0.1.7 Phase D-1: default flipped so Family-B evals can run on GPUs
    # outside a resident judge's pool. Strict mode is still operator-pinnable.
    assert exp.family_b_schedule == "parallel_on_unallowed_gpus"
    assert exp.max_task_retries == 1
    assert exp.pilot_first is True
    # the extra Codex code reviewer is opt-in — default off
    assert exp.codex_reviewer is False
    # max_parallel_runners default 0 = uncapped concurrency
    assert exp.max_parallel_runners == 0
    # max_concurrent_judges default 0 = uncapped co-resident judges
    assert exp.max_concurrent_judges == 0
    # Phase C-2.5: M-threshold default is 3 (≥3 configs must clear
    # pass/recall before Stage 6 runs the full N=50 round).
    assert exp.auto_validate_min_passing == 3


def test_auto_validate_min_passing_rejects_zero(
    tmp_path: Path, per_sample_root: Path,
):
    params = valid_params(tmp_path, per_sample_root)
    params["experiment"] = {"auto_validate_min_passing": 0}
    problems = ERAConfig.from_params(params).validate()
    assert any("auto_validate_min_passing" in p for p in problems)


def test_auto_validate_min_passing_rejects_negative(
    tmp_path: Path, per_sample_root: Path,
):
    params = valid_params(tmp_path, per_sample_root)
    params["experiment"] = {"auto_validate_min_passing": -3}
    problems = ERAConfig.from_params(params).validate()
    assert any("auto_validate_min_passing" in p for p in problems)


def test_auto_validate_min_passing_accepts_one(
    tmp_path: Path, per_sample_root: Path,
):
    """M=1 restores legacy "any 1 passes" semantics."""
    params = valid_params(tmp_path, per_sample_root)
    params["experiment"] = {"auto_validate_min_passing": 1}
    cfg = ERAConfig.from_params(params)
    assert cfg.validate() == []
    assert cfg.experiment.auto_validate_min_passing == 1


def test_auto_validate_min_passing_accepts_strict_value(
    tmp_path: Path, per_sample_root: Path,
):
    params = valid_params(tmp_path, per_sample_root)
    params["experiment"] = {"auto_validate_min_passing": 5}
    cfg = ERAConfig.from_params(params)
    assert cfg.validate() == []
    assert cfg.experiment.auto_validate_min_passing == 5


def test_experiment_max_parallel_runners_validates_nonnegative(
    tmp_path: Path, per_sample_root: Path,
):
    params = valid_params(tmp_path, per_sample_root)
    params["experiment"] = {"max_parallel_runners": -1}
    problems = ERAConfig.from_params(params).validate()
    assert any("max_parallel_runners" in p for p in problems)


def test_experiment_max_parallel_runners_accepts_positive(
    tmp_path: Path, per_sample_root: Path,
):
    params = valid_params(tmp_path, per_sample_root)
    params["experiment"] = {"max_parallel_runners": 4}
    cfg = ERAConfig.from_params(params)
    assert cfg.validate() == []
    assert cfg.experiment.max_parallel_runners == 4


def test_experiment_codex_reviewer_opt_in(
    tmp_path: Path, per_sample_root: Path
):
    params = valid_params(tmp_path, per_sample_root)
    params["experiment"] = {"codex_reviewer": True}
    cfg = ERAConfig.from_params(params)
    assert cfg.validate() == []
    assert cfg.experiment.codex_reviewer is True

    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(cfg.to_commented_yaml(), encoding="utf-8")
    assert ERAConfig.from_yaml(yaml_path).experiment.codex_reviewer is True


def test_validate_bad_family_a_execution(
    tmp_path: Path, per_sample_root: Path
):
    params = valid_params(tmp_path, per_sample_root)
    params["experiment"] = {"family_a_execution": "yolo"}
    problems = ERAConfig.from_params(params).validate()
    assert any("family_a_execution" in p for p in problems)


def test_config_without_experiment_block(
    tmp_path: Path, per_sample_root: Path
):
    """A pre-v0.1.4 config with no experiment block still loads + validates."""
    params = valid_params(tmp_path, per_sample_root)
    params.pop("experiment", None)
    cfg = ERAConfig.from_params(params)
    assert cfg.validate() == []
    assert cfg.experiment.family_a_execution == "parallel_packed"


def test_validate_bad_agent_tier(tmp_path: Path, per_sample_root: Path):
    params = valid_params(tmp_path, per_sample_root)
    params["agent_modes"] = {"plan_brainstorm": "turbo"}
    problems = ERAConfig.from_params(params).validate()
    assert any("agent_modes.plan_brainstorm" in p for p in problems)


def test_validate_bad_max_rounds(tmp_path: Path, per_sample_root: Path):
    params = valid_params(tmp_path, per_sample_root)
    params["debate"] = {"max_rounds": 0}
    problems = ERAConfig.from_params(params).validate()
    assert any("debate.max_rounds" in p for p in problems)


def test_round_trip_with_custom_agent_modes(
    tmp_path: Path, per_sample_root: Path
):
    params = valid_params(tmp_path, per_sample_root)
    params["agent_modes"] = {"plan_brainstorm": "heavy",
                             "multi_review": "standard",
                             "plan_decision": "heavy"}
    params["debate"] = {"max_rounds": 6}
    cfg = ERAConfig.from_params(params)
    assert cfg.validate() == []

    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(cfg.to_commented_yaml(), encoding="utf-8")
    reloaded = ERAConfig.from_yaml(yaml_path)
    assert reloaded.to_dict() == cfg.to_dict()
    assert reloaded.agent_modes.plan_brainstorm == "heavy"
    assert reloaded.debate.max_rounds == 6


def test_commented_yaml_has_debate_sections(
    tmp_path: Path, per_sample_root: Path
):
    cfg = ERAConfig.from_params(valid_params(tmp_path, per_sample_root))
    text = cfg.to_commented_yaml()
    assert "# ---- Agent modes" in text
    assert "# ---- Debate" in text


def test_config_without_agent_modes_block(
    tmp_path: Path, per_sample_root: Path
):
    """A v0.1.2 config with no agent_modes/debate block still loads + validates."""
    params = valid_params(tmp_path, per_sample_root)
    params.pop("agent_modes", None)
    params.pop("debate", None)
    cfg = ERAConfig.from_params(params)
    assert cfg.validate() == []
    assert cfg.agent_modes.plan_brainstorm == "standard"
    assert cfg.debate.max_rounds == 4


# ---- iter_sample_count + effective_iter_sample_count -------------------

def test_iter_sample_count_default_is_50():
    """A bare ERAConfig() defaults the per-iter cap to 50 — the value the
    operator gets if they say nothing at /era:init."""
    cfg = ERAConfig()
    assert cfg.data.iter_sample_count == 50


def test_iter_sample_count_loads_from_params(
    tmp_path: Path, per_sample_root: Path
):
    """Operator-supplied params propagate through from_params."""
    params = valid_params(tmp_path, per_sample_root)
    params["data"]["iter_sample_count"] = 25
    cfg = ERAConfig.from_params(params)
    assert cfg.data.iter_sample_count == 25


def test_validate_zero_iter_sample_count(
    tmp_path: Path, per_sample_root: Path
):
    """iter_sample_count must be >= 1 (a 0-sample eval is a config bug)."""
    params = valid_params(tmp_path, per_sample_root)
    params["data"]["iter_sample_count"] = 0
    problems = ERAConfig.from_params(params).validate()
    assert any("iter_sample_count" in p for p in problems)


def test_effective_iter_sample_count_caps_to_total(
    tmp_path: Path, per_sample_root: Path
):
    """Operator asked for 50 but only 3 samples exist — effective cap is 3."""
    params = valid_params(tmp_path, per_sample_root)
    params["data"]["sample_count"] = 3
    params["data"]["iter_sample_count"] = 50
    cfg = ERAConfig.from_params(params)
    assert cfg.effective_iter_sample_count() == 3


def test_effective_iter_sample_count_uses_cap_when_smaller(
    tmp_path: Path, per_sample_root: Path
):
    """Operator asked for 25 of 100 available — effective cap is 25."""
    params = valid_params(tmp_path, per_sample_root)
    params["data"]["sample_count"] = 100
    params["data"]["iter_sample_count"] = 25
    cfg = ERAConfig.from_params(params)
    assert cfg.effective_iter_sample_count() == 25


def test_effective_iter_sample_count_falls_back_when_unprobed():
    """sample_count == 0 (pre-probe state) → effective cap stays at the
    configured iter_sample_count (no usable upper bound to clamp against)."""
    cfg = ERAConfig()
    cfg.data.iter_sample_count = 50
    cfg.data.sample_count = 0
    assert cfg.effective_iter_sample_count() == 50


def test_iter_sample_count_appears_in_commented_yaml(
    tmp_path: Path, per_sample_root: Path
):
    """The serialized config.yaml must carry the field so the operator can
    see and edit it post-init."""
    cfg = ERAConfig.from_params(valid_params(tmp_path, per_sample_root))
    assert "iter_sample_count" in cfg.to_commented_yaml()


# ---- auto-validate thresholds (Phase A) --------------------------------

def test_auto_validate_thresholds_defaults():
    """Defaults: pass=0.70, recall=0.60, min_samples=10."""
    cfg = ERAConfig()
    assert cfg.experiment.auto_validate_pass_threshold == 0.70
    assert cfg.experiment.auto_validate_recall_threshold == 0.60
    assert cfg.experiment.auto_validate_min_samples == 10


def test_auto_validate_thresholds_loaded_from_params(
    tmp_path: Path, per_sample_root: Path
):
    params = valid_params(tmp_path, per_sample_root)
    params["experiment"] = {
        "auto_validate_pass_threshold": 0.85,
        "auto_validate_recall_threshold": 0.55,
        "auto_validate_min_samples": 25,
    }
    cfg = ERAConfig.from_params(params)
    assert cfg.experiment.auto_validate_pass_threshold == 0.85
    assert cfg.experiment.auto_validate_recall_threshold == 0.55
    assert cfg.experiment.auto_validate_min_samples == 25


def test_validate_rejects_out_of_range_pass_threshold(
    tmp_path: Path, per_sample_root: Path
):
    """pass_threshold must be in [0.0, 1.0]."""
    params = valid_params(tmp_path, per_sample_root)
    params["experiment"] = {"auto_validate_pass_threshold": 1.5}
    problems = ERAConfig.from_params(params).validate()
    assert any("auto_validate_pass_threshold" in p for p in problems)


def test_validate_rejects_out_of_range_recall_threshold(
    tmp_path: Path, per_sample_root: Path
):
    params = valid_params(tmp_path, per_sample_root)
    params["experiment"] = {"auto_validate_recall_threshold": -0.1}
    problems = ERAConfig.from_params(params).validate()
    assert any("auto_validate_recall_threshold" in p for p in problems)


def test_validate_rejects_zero_min_samples(
    tmp_path: Path, per_sample_root: Path
):
    """auto_validate_min_samples must be >= 1 (else the gate is meaningless)."""
    params = valid_params(tmp_path, per_sample_root)
    params["experiment"] = {"auto_validate_min_samples": 0}
    problems = ERAConfig.from_params(params).validate()
    assert any("auto_validate_min_samples" in p for p in problems)


def test_auto_validate_thresholds_round_trip_through_yaml(
    tmp_path: Path, per_sample_root: Path
):
    """The serialized config.yaml must carry all three threshold fields."""
    cfg = ERAConfig.from_params(valid_params(tmp_path, per_sample_root))
    yaml_text = cfg.to_commented_yaml()
    assert "auto_validate_pass_threshold" in yaml_text
    assert "auto_validate_recall_threshold" in yaml_text
    assert "auto_validate_min_samples" in yaml_text


# ---- use_annotation_evidence -------------------------------------------

def test_use_annotation_evidence_defaults_true():
    """Default behavior: Stage 2 reads annotation notes as evidence."""
    cfg = ERAConfig()
    assert cfg.data.use_annotation_evidence is True


def test_use_annotation_evidence_can_be_disabled(
    tmp_path: Path, per_sample_root: Path
):
    params = valid_params(tmp_path, per_sample_root)
    params["data"]["use_annotation_evidence"] = False
    cfg = ERAConfig.from_params(params)
    assert cfg.data.use_annotation_evidence is False
    assert cfg.validate() == []  # still valid
