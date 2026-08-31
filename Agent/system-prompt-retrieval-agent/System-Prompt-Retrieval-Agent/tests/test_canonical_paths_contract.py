"""Contract tests for canonical_paths.py (S00.13).

Asserts the locked V0.2.2 contracts:

* S00.03  canonical artifact root and stage/cell directory shape
* S00.04  canonical artifact filenames per stage
* S00.05  per-attempt / pointer / merged-view manifest paths
* S00.05a attempt_id format
* S00.06  match_fields vs informational_fields disjoint sets
* S00.07  cell record shape + validator
* S00.08  user-prompt corpus hash canonicalization (sorted, UTF-8, no whitespace)
* S00.09  execution-mode labels and production default
* S00.10  lifecycle enum
* S00.10a lifecycle mode/state matrix and warm-mode hard-disable flag
* S00.13  vendored canonical_paths byte-equality with the master
* S00.14  config_hash canonical-keys list and canonicalization
* S00.14a required-key fail-fast
* S00.15  schema_version literal "v0.2.2"
* S00.16  run_id regex format
* S00.19  prompt-pair / sample corpus hashes (drift detection)
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from system_prompt_retrieval_agent.remote._vendored import canonical_paths as cp

REPO_ROOT = Path(__file__).resolve().parents[2]
MASTER_PATH = REPO_ROOT / "Image-Generater-Remote" / "server" / "canonical_paths.py"
VENDORED_PATH = (
    REPO_ROOT
    / "System-Prompt-Retrieval-Agent"
    / "src"
    / "system_prompt_retrieval_agent"
    / "remote"
    / "_vendored"
    / "canonical_paths.py"
)


# ---------------------------------------------------------------------------
# S00.15 schema_version literal
# ---------------------------------------------------------------------------


def test_schema_version_literal_v022() -> None:
    assert cp.SCHEMA_VERSION == "v0.2.2"


# ---------------------------------------------------------------------------
# S00.13 vendored byte-equality
# ---------------------------------------------------------------------------


def test_vendored_canonical_paths_byte_equal_with_master() -> None:
    assert MASTER_PATH.is_file()
    assert VENDORED_PATH.is_file()
    assert MASTER_PATH.read_bytes() == VENDORED_PATH.read_bytes()


# ---------------------------------------------------------------------------
# S00.03 / S00.04 canonical paths and filenames
# ---------------------------------------------------------------------------


def test_outputs_root_v02() -> None:
    assert cp.OUTPUTS_ROOT == "outputs/v02"


def test_artifact_filenames_per_stage() -> None:
    assert cp.ARTIFACT_FILENAMES == {
        "gemma": "intermediate_prompt.txt",
        "flux": "result.png",
        "qwen": "eval.json",
    }


def test_artifact_filename_rejects_unknown_stage() -> None:
    with pytest.raises(ValueError):
        cp.artifact_filename("unknown")


def test_cell_dir_shape() -> None:
    rid = "20260426T010000Z-deadbeef"
    expected = (
        f"outputs/v02/{rid}/gemma/round_0/PAIR-0001/UPROMPT-0002/SAMPLE-0003"
    )
    assert (
        cp.cell_dir(rid, "gemma", 0, "PAIR-0001", "UPROMPT-0002", "SAMPLE-0003")
        == expected
    )


def test_cell_artifact_path_uses_stage_filename() -> None:
    rid = "20260426T010000Z-deadbeef"
    p = cp.cell_artifact_path(rid, "flux", 1, "P", "U", "S")
    assert p.endswith("/result.png")
    assert "/flux/round_1/P/U/S/" in p


# ---------------------------------------------------------------------------
# S00.05 / S00.05a manifest paths and attempt_id format
# ---------------------------------------------------------------------------


def test_manifest_path_layout() -> None:
    rid = "20260426T010000Z-deadbeef"
    aid = "20260426T010500Z-cafe"
    attempt = cp.stage_manifest_attempt_path(rid, "qwen", 2, aid)
    pointer = cp.stage_manifest_pointer_path(rid, "qwen", 2)
    merged = cp.stage_manifest_merged_path(rid, "qwen", 2)
    assert attempt == f"outputs/v02/{rid}/manifests/qwen_round_2_attempt_{aid}.json"
    assert pointer == f"outputs/v02/{rid}/manifests/qwen_round_2_manifest.json"
    assert merged == f"outputs/v02/{rid}/manifests/qwen_round_2_merged.json"


def test_attempt_id_regex_4hex() -> None:
    aid = cp.generate_attempt_id()
    assert cp.ATTEMPT_ID_REGEX.match(aid)
    assert re.match(r"^\d{8}T\d{6}Z-[0-9a-f]{4}$", aid)


def test_invalid_attempt_id_rejected() -> None:
    with pytest.raises(ValueError):
        cp.stage_manifest_attempt_path(
            "20260426T010000Z-deadbeef", "gemma", 0, "not-an-attempt-id"
        )


# ---------------------------------------------------------------------------
# S00.06 match_fields / informational_fields disjoint
# ---------------------------------------------------------------------------


def test_match_and_informational_fields_disjoint() -> None:
    m = set(cp.MATCH_FIELDS)
    i = set(cp.INFORMATIONAL_FIELDS)
    assert m.isdisjoint(i)


def test_match_fields_includes_all_four_hashes() -> None:
    assert "config_hash" in cp.MATCH_FIELDS
    assert "user_prompt_corpus_hash" in cp.MATCH_FIELDS
    assert "prompt_pair_corpus_hash" in cp.MATCH_FIELDS
    assert "sample_corpus_hash" in cp.MATCH_FIELDS


def test_informational_fields_only_lifecycle() -> None:
    assert set(cp.INFORMATIONAL_FIELDS) == {
        "lifecycle_mode",
        "lifecycle_state_after",
    }


# ---------------------------------------------------------------------------
# S00.07 cell record validator
# ---------------------------------------------------------------------------


def _ok_cell(**overrides):
    base = {
        "prompt_pair_id": "P",
        "user_prompt_id": "U",
        "sample_id": "S",
        "status": "ok",
        "artifact_relpath": "gemma/round_0/P/U/S/intermediate_prompt.txt",
        "artifact_size_bytes": 64,
        "artifact_sha256": "a" * 64,
    }
    base.update(overrides)
    return base


def test_validate_cell_record_ok() -> None:
    cp.validate_cell_record(_ok_cell())


def test_validate_cell_record_carried_over_requires_size_and_sha() -> None:
    cp.validate_cell_record(_ok_cell(status="carried_over"))
    with pytest.raises(ValueError):
        cp.validate_cell_record(
            {k: v for k, v in _ok_cell(status="carried_over").items()
             if k != "artifact_sha256"}
        )


def test_validate_cell_record_error_status_does_not_require_sha() -> None:
    cp.validate_cell_record(
        {
            "prompt_pair_id": "P",
            "user_prompt_id": "U",
            "sample_id": "S",
            "status": "error",
            "artifact_relpath": "",
            "error_reason": "model exception",
        }
    )


def test_validate_cell_record_invalid_sha_length() -> None:
    with pytest.raises(ValueError):
        cp.validate_cell_record(_ok_cell(artifact_sha256="short"))


def test_validate_cell_record_qwen_only_fields_rejected_for_other_stages() -> None:
    with pytest.raises(ValueError):
        cp.validate_cell_record(_ok_cell(parse_status="ok"), stage="gemma")


# ---------------------------------------------------------------------------
# S00.08 / S00.19 corpus hash canonicalization
# ---------------------------------------------------------------------------


def test_user_prompt_corpus_hash_is_order_independent() -> None:
    a = [
        {"user_prompt_id": "U1", "language": "en", "text": "hi", "enabled": True},
        {"user_prompt_id": "U2", "language": "zh", "text": "你好", "enabled": True},
    ]
    b = list(reversed(a))
    assert cp.user_prompt_corpus_hash(a) == cp.user_prompt_corpus_hash(b)


def test_user_prompt_corpus_hash_detects_text_drift() -> None:
    a = [{"user_prompt_id": "U1", "language": "en", "text": "hi", "enabled": True}]
    b = [{"user_prompt_id": "U1", "language": "en", "text": "hello", "enabled": True}]
    assert cp.user_prompt_corpus_hash(a) != cp.user_prompt_corpus_hash(b)


def test_canonical_json_is_sorted_no_whitespace_utf8() -> None:
    raw = cp.canonical_json({"b": 1, "a": [3, 2, 1]})
    assert raw == b'{"a":[3,2,1],"b":1}'


def test_prompt_pair_corpus_hash_detects_text_drift() -> None:
    a = [
        {
            "prompt_pair_id": "PP1",
            "system_prompt": "sys",
            "user_prompt_template": "tpl",
            "language": "en",
            "enabled": True,
        }
    ]
    b = [dict(a[0], system_prompt="sys2")]
    assert cp.prompt_pair_corpus_hash(a) != cp.prompt_pair_corpus_hash(b)


def test_sample_corpus_hash_detects_byte_drift() -> None:
    a = [
        {
            "sample_id": "S1",
            "model_path": "/m.png",
            "cloth_path": "/c.png",
            "model_sha256": "a" * 64,
            "cloth_sha256": "b" * 64,
        }
    ]
    b = [dict(a[0], model_sha256="c" * 64)]
    assert cp.sample_corpus_hash(a) != cp.sample_corpus_hash(b)


# ---------------------------------------------------------------------------
# S00.09 execution-mode labels
# ---------------------------------------------------------------------------


def test_execution_modes() -> None:
    assert cp.EXECUTION_MODE_V022 == "v022_stage_major"
    assert cp.PRODUCTION_EXECUTION_MODE == "v022_stage_major"
    assert "legacy_stage_all_compat" in cp.ALLOWED_EXECUTION_MODES
    assert "legacy_cartesian_compat" in cp.ALLOWED_EXECUTION_MODES


# ---------------------------------------------------------------------------
# S00.10 / S00.10a lifecycle enum + matrix + warm-mode disable
# ---------------------------------------------------------------------------


def test_lifecycle_enum_exact_values() -> None:
    assert set(cp.ALLOWED_LIFECYCLE_STATES) == {
        "disk_unloaded",
        "cpu_prefetched",
        "gpu_loaded",
        "gpu_unloaded_cpu_retained",
    }


def test_cold_mode_only_allows_disk_unloaded() -> None:
    cp.validate_lifecycle_state_after("cold", "disk_unloaded")
    for bad in ("cpu_prefetched", "gpu_loaded", "gpu_unloaded_cpu_retained"):
        with pytest.raises(ValueError):
            cp.validate_lifecycle_state_after("cold", bad)


def test_warm_mode_state_matrix_excludes_gpu_loaded() -> None:
    allowed = cp.LIFECYCLE_MODE_STATE_MATRIX["warm"]
    assert "gpu_loaded" not in allowed
    assert allowed == frozenset({"cpu_prefetched", "gpu_unloaded_cpu_retained"})


def test_warm_mode_hard_disabled_in_v022() -> None:
    assert cp.WARM_MODE_HARD_DISABLED_IN_V022 is True
    assert cp.WARM_MODE_DISABLED_ERROR == "warm mode disabled in V0.2.2"


# ---------------------------------------------------------------------------
# S00.14 / S00.14a config_hash and required-key fail-fast
# ---------------------------------------------------------------------------


_REQUIRED_PRESENT_CFG = {
    "models": {
        "gemma": {"checkpoint": "/g", "gen_params": {}},
        "flux": {"checkpoint": "/f", "gen_params": {}},
        "qwen": {"checkpoint": "/q", "gen_params": {}},
    },
    "evaluation": {"categories": ["a"], "weights": {"a": 1.0}},
    "scoring": {"gate": {"min": 0.5}},
}


def test_canonical_config_subset_all_required_present_succeeds() -> None:
    out = cp.canonical_config_subset(_REQUIRED_PRESENT_CFG)
    assert out["models.gemma.checkpoint"] == "/g"
    # Optional keys absent → null
    assert out["execution.mode"] is None


def test_config_hash_is_deterministic_across_calls() -> None:
    h1 = cp.config_hash(_REQUIRED_PRESENT_CFG)
    h2 = cp.config_hash(_REQUIRED_PRESENT_CFG)
    assert h1 == h2 and len(h1) == 64


@pytest.mark.parametrize(
    "drop",
    [
        ("models", "gemma", "checkpoint"),
        ("models", "flux", "checkpoint"),
        ("models", "qwen", "checkpoint"),
        ("models", "gemma", "gen_params"),
        ("models", "flux", "gen_params"),
        ("models", "qwen", "gen_params"),
        ("evaluation", "categories"),
        ("evaluation", "weights"),
        ("scoring", "gate"),
    ],
)
def test_config_hash_required_key_absent_aborts(drop: tuple) -> None:
    cfg = json.loads(json.dumps(_REQUIRED_PRESENT_CFG))
    cursor = cfg
    for part in drop[:-1]:
        cursor = cursor[part]
    del cursor[drop[-1]]
    with pytest.raises(ValueError, match="config_hash required keys absent"):
        cp.config_hash(cfg)


# ---------------------------------------------------------------------------
# S00.16 run_id regex
# ---------------------------------------------------------------------------


def test_run_id_regex_matches_generator_output() -> None:
    rid = cp.generate_run_id()
    assert cp.RUN_ID_REGEX.match(rid)
    assert re.match(r"^\d{8}T\d{6}Z-[0-9a-f]{8}$", rid)


def test_invalid_run_id_rejected_in_path_constructors() -> None:
    with pytest.raises(ValueError):
        cp.run_dir("not-a-run-id")
