"""S00.16a tests for the production --resume-from-run-id validator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from system_prompt_retrieval_agent.cli import build_parser
from system_prompt_retrieval_agent.remote._vendored import canonical_paths as cp
from system_prompt_retrieval_agent.resume_from_run_id import (
    CurrentRunHashes,
    ResumeDriftError,
    ResumeRunDirMissing,
    ResumeRunIdInvalid,
    validate_resume_from_run_id,
)


# ---------------------------------------------------------------------------
# CLI parser surface
# ---------------------------------------------------------------------------


def test_cli_run_subcommand_has_resume_from_run_id_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["run", "--config", "x.yaml", "--resume-from-run-id", "20260426T010000Z-deadbeef"]
    )
    assert args.cmd == "run"
    assert args.resume_from_run_id == "20260426T010000Z-deadbeef"


def test_cli_run_default_resume_from_run_id_is_none() -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "--config", "x.yaml"])
    assert args.resume_from_run_id is None


# ---------------------------------------------------------------------------
# Validator helpers
# ---------------------------------------------------------------------------


def _hashes(**overrides: str) -> CurrentRunHashes:
    base = {
        "config_hash": "c" * 64,
        "user_prompt_corpus_hash": "u" * 64,
        "prompt_pair_corpus_hash": "p" * 64,
        "sample_corpus_hash": "s" * 64,
    }
    base.update(overrides)
    return CurrentRunHashes(**base)


def _seed_prior_run(
    output_root: Path, run_id: str, hashes: CurrentRunHashes
) -> Path:
    manifests_dir = output_root / cp.OUTPUTS_ROOT / run_id / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": cp.SCHEMA_VERSION,
        "run_id": run_id,
        "round_id": 0,
        "stage": "gemma",
        **hashes.as_mapping(),
        "user_prompt_corpus_id": "v0",
        "lifecycle_mode": "cold",
        "lifecycle_state_after": "disk_unloaded",
        "cells": [],
    }
    (manifests_dir / "gemma_round_0_attempt_20260426T010005Z-cafe.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )
    return manifests_dir.parent


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_happy_path_returns_prior_run_dir(tmp_path: Path) -> None:
    rid = "20260426T010000Z-deadbeef"
    h = _hashes()
    prior = _seed_prior_run(tmp_path, rid, h)
    out = validate_resume_from_run_id(rid, output_root=tmp_path, current_hashes=h)
    assert out == prior


# ---------------------------------------------------------------------------
# Regex enforcement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "not-a-run-id",
        "20260426010000Z-deadbeef",        # missing 'T'
        "20260426T010000Z-DEADBEEF",       # uppercase hex disallowed
        "20260426T010000Z-deadbee",        # 7 hex chars
        "",
    ],
)
def test_invalid_run_id_regex_aborts(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ResumeRunIdInvalid):
        validate_resume_from_run_id(bad, output_root=tmp_path, current_hashes=_hashes())


# ---------------------------------------------------------------------------
# Missing prior run dir / empty manifests dir
# ---------------------------------------------------------------------------


def test_missing_prior_run_directory_aborts(tmp_path: Path) -> None:
    with pytest.raises(ResumeRunDirMissing):
        validate_resume_from_run_id(
            "20260426T010000Z-deadbeef",
            output_root=tmp_path,
            current_hashes=_hashes(),
        )


def test_empty_prior_manifests_dir_aborts(tmp_path: Path) -> None:
    rid = "20260426T010000Z-deadbeef"
    (tmp_path / cp.OUTPUTS_ROOT / rid / "manifests").mkdir(parents=True)
    with pytest.raises(ResumeRunDirMissing):
        validate_resume_from_run_id(
            rid, output_root=tmp_path, current_hashes=_hashes()
        )


# ---------------------------------------------------------------------------
# Drift across each of the four hashes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "diverging_field",
    [
        "config_hash",
        "user_prompt_corpus_hash",
        "prompt_pair_corpus_hash",
        "sample_corpus_hash",
    ],
)
def test_drift_in_each_hash_aborts_naming_field(
    tmp_path: Path, diverging_field: str
) -> None:
    rid = "20260426T010000Z-deadbeef"
    prior = _hashes()
    _seed_prior_run(tmp_path, rid, prior)
    current = _hashes(**{diverging_field: "0" * 64})
    with pytest.raises(ResumeDriftError) as info:
        validate_resume_from_run_id(rid, output_root=tmp_path, current_hashes=current)
    assert f"resume corpus/config drift: {diverging_field} differs" in str(info.value)
