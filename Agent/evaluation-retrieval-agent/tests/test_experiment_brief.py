"""Tests for Stage 4 experiment-brief validation (Rules 4 & 5)."""

from __future__ import annotations

import io
import json
from pathlib import Path

from era.cli import main
from era.orchestration.experiment_brief import (
    extract_hypothesis_ids,
    validate_experiment_brief,
)


def _valid_brief() -> dict:
    """A well-formed brief: 1 metric baseline + 2 VLM scales."""
    return {
        "iteration": 1,
        "debate_round": 1,
        "evaluation_goal": "retrieve the cheapest evaluator aligned with humans",
        "candidate_configs": [
            {"combination_id": "metric-baseline", "slot": "metric_baseline",
             "family": "B", "judge": None, "metric_subfamily": "clip-dino",
             "prompt": None, "scope": "whole", "inputs_needed": ["output"],
             "gpu_estimate": "1xH100", "cost_estimate_usd": 0.0,
             "hypothesis_id": "H0", "feasible": True},
            {"combination_id": "vlm-32b-pairwise", "slot": "vlm_scale",
             "family": "A", "judge": "qwen2.5-vl-32b", "metric_subfamily": None,
             "prompt": "pairwise-rubric", "scope": "whole",
             "inputs_needed": ["input_cloth", "output"],
             "gpu_estimate": "4xH100", "cost_estimate_usd": 0.0,
             "hypothesis_id": "H1", "feasible": True},
            {"combination_id": "vlm-72b-pairwise", "slot": "vlm_scale",
             "family": "A", "judge": "qwen2.5-vl-72b", "metric_subfamily": None,
             "prompt": "pairwise-rubric", "scope": "whole",
             "inputs_needed": ["input_cloth", "output"],
             "gpu_estimate": "4xH100", "cost_estimate_usd": 0.0,
             "hypothesis_id": "H2", "feasible": True},
        ],
        "validation": {"human_rating_set": "VTON-QBench",
                       "alignment_metric": "kendall_tau",
                       "sample_size": 489, "debiasing": "order-swap"},
        "pilot": {"sample_count": 30, "go_no_go": "a tau gap is visible"},
        "scale_comparison": "two VLM scales probed: 32B and 72B",
        "resource_estimate": {"gpu_hours": 12.0, "wallclock_hours": 6.0,
                              "api_cost_usd": 0.0},
        "pivot_matrix": [{"pilot_outcome": "all tau < 0.3", "action": "revise"}],
    }


def test_valid_brief_passes():
    assert validate_experiment_brief(_valid_brief()) == []


def test_rejects_non_dict():
    assert validate_experiment_brief([]) != []


def test_rejects_too_many_configs():
    brief = _valid_brief()
    base = brief["candidate_configs"][0]
    brief["candidate_configs"] = [
        {**base, "combination_id": f"c{i}"} for i in range(11)
    ]
    problems = validate_experiment_brief(brief)
    assert any("Rule 5" in p and "10" in p for p in problems)


def test_rejects_slot_cap_violation():
    brief = _valid_brief()
    vlm = brief["candidate_configs"][1]
    brief["candidate_configs"] += [
        {**vlm, "combination_id": "vlm-extra-1"},
        {**vlm, "combination_id": "vlm-extra-2"},
    ]  # now 4 vlm_scale configs — the cap is 3
    problems = validate_experiment_brief(brief)
    assert any("vlm_scale" in p for p in problems)


def test_rejects_scale_comparison_violation():
    brief = _valid_brief()
    brief["candidate_configs"] = [brief["candidate_configs"][0]]  # no VLM
    problems = validate_experiment_brief(brief)
    assert any("Rule 4" in p for p in problems)


def test_rejects_missing_metric_baseline():
    brief = _valid_brief()
    brief["candidate_configs"] = brief["candidate_configs"][1:]  # drop baseline
    problems = validate_experiment_brief(brief)
    assert any("metric_baseline" in p for p in problems)


def test_rejects_missing_validation():
    brief = _valid_brief()
    del brief["validation"]
    problems = validate_experiment_brief(brief)
    assert any("validation" in p for p in problems)


def test_rejects_missing_hypothesis_id():
    brief = _valid_brief()
    del brief["candidate_configs"][1]["hypothesis_id"]
    problems = validate_experiment_brief(brief)
    assert any("hypothesis_id" in p for p in problems)


def test_rejects_duplicate_combination_id():
    brief = _valid_brief()
    brief["candidate_configs"][2]["combination_id"] = "vlm-32b-pairwise"
    problems = validate_experiment_brief(brief)
    assert any("duplicate combination_id" in p for p in problems)


# ---- CLI ----------------------------------------------------------------

def test_cli_check_experiment_brief_inline(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"brief": _valid_brief()})))
    code = main(["check-experiment-brief"])
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["valid"] is True
    assert result["problems"] == []


def test_cli_check_experiment_brief_from_file(tmp_path: Path, monkeypatch, capsys):
    brief = _valid_brief()
    brief["candidate_configs"] = brief["candidate_configs"][:1]  # invalid
    path = tmp_path / "experiment_brief.json"
    path.write_text(json.dumps(brief), encoding="utf-8")
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"brief_path": str(path)})))
    code = main(["check-experiment-brief"])
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["valid"] is False
    assert result["problems"]


def test_cli_check_experiment_brief_missing_params(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
    code = main(["check-experiment-brief"])
    result = json.loads(capsys.readouterr().out)
    assert code == 1
    assert result["error"] == "missing_params"


def test_cli_check_experiment_brief_with_config_path_rejects_mismatch(
    tmp_path: Path, per_sample_root: Path, monkeypatch, capsys,
):
    """End-to-end: pass config_path → the gate enforces iter_sample_count."""
    from conftest import valid_params
    from era.config import ERAConfig

    params = valid_params(tmp_path, per_sample_root)
    params["data"]["iter_sample_count"] = 50
    params["data"]["sample_count"] = 100  # cap is 50 (< 100), iter_sample_count stays
    cfg = ERAConfig.from_params(params)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(cfg.to_commented_yaml(), encoding="utf-8")

    brief = _valid_brief()
    brief["validation"]["sample_size"] = 489   # ≠ 50
    brief["pilot"]["sample_count"] = 30
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"brief": brief, "config_path": str(config_path)})))
    code = main(["check-experiment-brief"])
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["valid"] is False
    assert result["iter_sample_count_checked"] is True
    assert any("validation.sample_size" in p and "50" in p
               for p in result["problems"])


# ---- hardened handoff-contract checks -----------------------------------

def test_rejects_missing_scale_comparison():
    brief = _valid_brief()
    del brief["scale_comparison"]
    assert any("scale_comparison" in p for p in validate_experiment_brief(brief))


def test_rejects_missing_pivot_matrix():
    brief = _valid_brief()
    del brief["pivot_matrix"]
    assert any("pivot_matrix" in p for p in validate_experiment_brief(brief))


def test_rejects_bad_pivot_matrix_row():
    brief = _valid_brief()
    brief["pivot_matrix"] = [{"pilot_outcome": "x"}]  # missing 'action'
    assert any("pivot_matrix[0]" in p for p in validate_experiment_brief(brief))


# ---- iter_sample_count cross-check -------------------------------------

def test_iter_sample_count_match_passes():
    """validation.sample_size == iter_sample_count and
    pilot.sample_count <= iter_sample_count → valid."""
    brief = _valid_brief()
    brief["validation"]["sample_size"] = 50
    brief["pilot"]["sample_count"] = 10
    assert validate_experiment_brief(brief, iter_sample_count=50) == []


def test_iter_sample_count_rejects_validation_mismatch():
    """validation.sample_size != iter_sample_count → rejected."""
    brief = _valid_brief()
    brief["validation"]["sample_size"] = 100  # ≠ 50
    brief["pilot"]["sample_count"] = 10
    problems = validate_experiment_brief(brief, iter_sample_count=50)
    assert any("validation.sample_size" in p and "50" in p for p in problems)


def test_iter_sample_count_rejects_oversized_pilot():
    """pilot.sample_count > iter_sample_count → rejected."""
    brief = _valid_brief()
    brief["validation"]["sample_size"] = 50
    brief["pilot"]["sample_count"] = 75  # > 50
    problems = validate_experiment_brief(brief, iter_sample_count=50)
    assert any("pilot.sample_count" in p and "50" in p for p in problems)


def test_iter_sample_count_omitted_skips_cross_check():
    """When the caller does not pass iter_sample_count, the brief gate
    behaves as before — no cap enforcement, only schema checks."""
    brief = _valid_brief()
    brief["validation"]["sample_size"] = 489
    brief["pilot"]["sample_count"] = 30
    # No iter_sample_count → original semantics
    assert validate_experiment_brief(brief) == []


def test_rejects_config_missing_scope():
    brief = _valid_brief()
    del brief["candidate_configs"][1]["scope"]
    assert any("scope" in p for p in validate_experiment_brief(brief))


def test_rejects_config_empty_inputs_needed():
    brief = _valid_brief()
    brief["candidate_configs"][1]["inputs_needed"] = []
    assert any("inputs_needed" in p for p in validate_experiment_brief(brief))


def test_rejects_config_missing_gpu_estimate():
    brief = _valid_brief()
    del brief["candidate_configs"][1]["gpu_estimate"]
    assert any("gpu_estimate" in p for p in validate_experiment_brief(brief))


def test_rejects_infeasible_config():
    brief = _valid_brief()
    brief["candidate_configs"][1]["feasible"] = False
    assert any("feasible" in p for p in validate_experiment_brief(brief))


def test_rejects_config_missing_feasible():
    brief = _valid_brief()
    del brief["candidate_configs"][1]["feasible"]
    assert any("feasible" in p for p in validate_experiment_brief(brief))


def test_rejects_bad_cost_estimate():
    brief = _valid_brief()
    brief["candidate_configs"][1]["cost_estimate_usd"] = "free"
    assert any("cost_estimate_usd" in p for p in validate_experiment_brief(brief))


def test_rejects_family_slot_mismatch():
    brief = _valid_brief()
    # config[1] is a Family-A vlm config; relabel its slot as metric_baseline.
    brief["candidate_configs"][1]["slot"] = "metric_baseline"
    assert any("requires family" in p for p in validate_experiment_brief(brief))


def test_rejects_family_a_missing_judge():
    brief = _valid_brief()
    brief["candidate_configs"][1]["judge"] = None  # the vlm config is Family A
    assert any("family 'A' needs a 'judge'" in p
               for p in validate_experiment_brief(brief))


def test_rejects_family_a_missing_prompt():
    brief = _valid_brief()
    brief["candidate_configs"][1]["prompt"] = None
    assert any("family 'A' needs a 'prompt'" in p
               for p in validate_experiment_brief(brief))


def test_rejects_family_b_missing_metric_subfamily():
    brief = _valid_brief()
    brief["candidate_configs"][0]["metric_subfamily"] = None  # Family-B baseline
    assert any("metric_subfamily" in p for p in validate_experiment_brief(brief))


def test_rejects_non_distinct_vlm_scales():
    brief = _valid_brief()
    brief["candidate_configs"] = brief["candidate_configs"][1:]  # drop baseline
    # both VLM configs now name the same judge — not a real scale comparison
    brief["candidate_configs"][1]["judge"] = brief["candidate_configs"][0]["judge"]
    assert any("Rule 4" in p for p in validate_experiment_brief(brief))


def test_rejects_metric_baseline_over_cap():
    brief = _valid_brief()
    base = brief["candidate_configs"][0]
    brief["candidate_configs"] += [
        {**base, "combination_id": "mb-2"},
        {**base, "combination_id": "mb-3"},
    ]  # 3 metric_baseline configs — the cap is 2
    problems = validate_experiment_brief(brief)
    assert any("metric_baseline" in p and "cap" in p for p in problems)


def test_rejects_incomplete_validation():
    brief = _valid_brief()
    del brief["validation"]["debiasing"]
    assert any("debiasing" in p for p in validate_experiment_brief(brief))


def test_rejects_bad_validation_sample_size():
    brief = _valid_brief()
    brief["validation"]["sample_size"] = 0
    assert any("sample_size" in p for p in validate_experiment_brief(brief))


def test_rejects_incomplete_pilot():
    brief = _valid_brief()
    del brief["pilot"]["go_no_go"]
    assert any("go_no_go" in p for p in validate_experiment_brief(brief))


def test_rejects_incomplete_resource_estimate():
    brief = _valid_brief()
    del brief["resource_estimate"]["gpu_hours"]
    assert any("gpu_hours" in p for p in validate_experiment_brief(brief))


# ---- hypothesis-id resolution -------------------------------------------

def test_extract_hypothesis_ids():
    text = "## H0: baseline\n\n## H1: vlm judge\n\n## H2: hybrid\n\n### H9: sub\n"
    assert extract_hypothesis_ids(text) == {"H0", "H1", "H2"}


def test_known_hypotheses_resolves():
    assert validate_experiment_brief(_valid_brief(), {"H0", "H1", "H2"}) == []


def test_known_hypotheses_flags_dangling():
    problems = validate_experiment_brief(_valid_brief(), {"H0", "H1"})
    assert any("H2" in p and "hypotheses.md" in p for p in problems)


# ---- CLI: hypotheses_path -----------------------------------------------

def test_cli_check_experiment_brief_with_hypotheses(
    tmp_path: Path, monkeypatch, capsys
):
    brief_path = tmp_path / "experiment_brief.json"
    brief_path.write_text(json.dumps(_valid_brief()), encoding="utf-8")
    hyp_path = tmp_path / "hypotheses.md"
    hyp_path.write_text("## H0: a\n\n## H1: b\n\n## H2: c\n", encoding="utf-8")
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"brief_path": str(brief_path), "hypotheses_path": str(hyp_path)})))
    code = main(["check-experiment-brief"])
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["valid"] is True
    assert result["hypotheses_checked"] is True


def test_cli_check_experiment_brief_dangling_hypothesis(
    tmp_path: Path, monkeypatch, capsys
):
    brief_path = tmp_path / "experiment_brief.json"
    brief_path.write_text(json.dumps(_valid_brief()), encoding="utf-8")
    hyp_path = tmp_path / "hypotheses.md"
    hyp_path.write_text("## H0: a\n\n## H1: b\n", encoding="utf-8")  # no H2
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"brief_path": str(brief_path), "hypotheses_path": str(hyp_path)})))
    code = main(["check-experiment-brief"])
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["valid"] is False
    assert any("H2" in p for p in result["problems"])


def test_cli_check_experiment_brief_missing_hypotheses_file(
    tmp_path: Path, monkeypatch, capsys
):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"brief": _valid_brief(),
         "hypotheses_path": str(tmp_path / "nope.md")})))
    code = main(["check-experiment-brief"])
    result = json.loads(capsys.readouterr().out)
    assert code == 1
    assert result["error"] == "no_such_file"


# ---- Phase C-2.5 elitism (must_include_configs) ------------------------

def test_must_include_configs_accepts_brief_with_all_required():
    brief = _valid_brief()
    # brief carries "vlm-32b-pairwise" + "vlm-72b-pairwise" + "metric-baseline"
    problems = validate_experiment_brief(
        brief,
        must_include_configs=["vlm-32b-pairwise", "vlm-72b-pairwise"],
    )
    assert problems == []


def test_must_include_configs_rejects_brief_missing_one():
    brief = _valid_brief()
    problems = validate_experiment_brief(
        brief,
        must_include_configs=["vlm-32b-pairwise", "vlm-some-other-proven"],
    )
    assert any("Phase C-2.5 elitism" in p
               and "vlm-some-other-proven" in p
               for p in problems)


def test_must_include_configs_empty_list_is_no_op():
    brief = _valid_brief()
    assert validate_experiment_brief(brief, must_include_configs=[]) == []


def test_must_include_configs_ignores_non_string_entries():
    brief = _valid_brief()
    # Non-string entries should be silently dropped (sub-agent shouldn't
    # produce them, but be defensive).
    problems = validate_experiment_brief(
        brief, must_include_configs=["vlm-32b-pairwise", None, ""],
    )
    assert problems == []


def test_parent_hypothesis_ids_optional_field_accepted():
    brief = _valid_brief()
    brief["candidate_configs"][1]["parent_hypothesis_ids"] = [
        "iter1:vlm-32b-spatial", "iter2:vlm-pair-region"
    ]
    assert validate_experiment_brief(brief) == []


def test_parent_hypothesis_ids_rejects_non_list():
    brief = _valid_brief()
    brief["candidate_configs"][1]["parent_hypothesis_ids"] = "not a list"
    problems = validate_experiment_brief(brief)
    assert any("parent_hypothesis_ids" in p for p in problems)


def test_parent_hypothesis_ids_rejects_empty_string_entry():
    brief = _valid_brief()
    brief["candidate_configs"][1]["parent_hypothesis_ids"] = ["", "ok"]
    problems = validate_experiment_brief(brief)
    assert any("parent_hypothesis_ids" in p for p in problems)


def test_cli_check_experiment_brief_with_evolution_state_path(
    tmp_path: Path, monkeypatch, capsys
):
    brief_path = tmp_path / "experiment_brief.json"
    brief_path.write_text(json.dumps(_valid_brief()), encoding="utf-8")
    es_path = tmp_path / "evolution_state.json"
    es_path.write_text(json.dumps({
        "must_include_configs": ["vlm-32b-pairwise"],
    }), encoding="utf-8")
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"brief_path": str(brief_path),
         "evolution_state_path": str(es_path)})))
    code = main(["check-experiment-brief"])
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["valid"] is True
    assert result["must_include_configs_checked"] is True
    assert result["must_include_configs"] == ["vlm-32b-pairwise"]
    # Phase C-2.5 — structured field is present, empty on the happy path.
    assert result["missing_must_include"] == []


def test_cli_check_experiment_brief_with_evolution_state_missing_must_include(
    tmp_path: Path, monkeypatch, capsys
):
    brief_path = tmp_path / "experiment_brief.json"
    brief_path.write_text(json.dumps(_valid_brief()), encoding="utf-8")
    es_path = tmp_path / "evolution_state.json"
    es_path.write_text(json.dumps({
        "must_include_configs": ["vlm-32b-pairwise", "vlm-not-in-brief"],
    }), encoding="utf-8")
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"brief_path": str(brief_path),
         "evolution_state_path": str(es_path)})))
    code = main(["check-experiment-brief"])
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["valid"] is False
    assert any("vlm-not-in-brief" in p for p in result["problems"])
    # Phase C-2.5 — structured field carries ONLY the missing IDs so the
    # Stage 4 skill can fold them into the revision brief without
    # substring-parsing the prose problem.
    assert result["missing_must_include"] == ["vlm-not-in-brief"]


def test_cli_check_experiment_brief_missing_must_include_omitted_when_no_es(
    tmp_path: Path, monkeypatch, capsys
):
    """When the operator doesn't pass evolution_state_path, the
    ``missing_must_include`` field is an empty list (the brief gate
    didn't enforce elitism, so nothing can be missing)."""
    brief_path = tmp_path / "experiment_brief.json"
    brief_path.write_text(json.dumps(_valid_brief()), encoding="utf-8")
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"brief_path": str(brief_path)})))
    code = main(["check-experiment-brief"])
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["must_include_configs_checked"] is False
    assert result["missing_must_include"] == []


def test_cli_check_experiment_brief_with_bad_evolution_state_json(
    tmp_path: Path, monkeypatch, capsys
):
    """Step 2 regression: the extracted helper still surfaces
    ``bad_evolution_state_json`` cleanly on malformed input."""
    brief_path = tmp_path / "experiment_brief.json"
    brief_path.write_text(json.dumps(_valid_brief()), encoding="utf-8")
    es_path = tmp_path / "evolution_state.json"
    es_path.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"brief_path": str(brief_path),
         "evolution_state_path": str(es_path)})))
    code = main(["check-experiment-brief"])
    result = json.loads(capsys.readouterr().out)
    assert code == 1
    assert result["error"] == "bad_evolution_state_json"


def test_cli_check_experiment_brief_with_missing_evolution_state_file(
    tmp_path: Path, monkeypatch, capsys
):
    """Step 2 regression: the extracted helper still surfaces
    ``no_such_file`` on a missing path."""
    brief_path = tmp_path / "experiment_brief.json"
    brief_path.write_text(json.dumps(_valid_brief()), encoding="utf-8")
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"brief_path": str(brief_path),
         "evolution_state_path": str(tmp_path / "nope.json")})))
    code = main(["check-experiment-brief"])
    result = json.loads(capsys.readouterr().out)
    assert code == 1
    assert result["error"] == "no_such_file"
