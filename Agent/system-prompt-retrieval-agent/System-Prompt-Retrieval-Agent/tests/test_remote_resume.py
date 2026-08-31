"""Tests for remote/resume.py — build_missing_manifest and build_survivor_manifest.

Plan reference: §6.4, §6.5, §17.3, S04.09.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from system_prompt_retrieval_agent.remote.resume import (
    build_missing_manifest,
    build_survivor_manifest,
    build_survivor_manifest_with_samples,
)
from system_prompt_retrieval_agent.schemas import (
    PerPairManifest,
    PerUserPromptManifest,
    StageManifest,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _make_stage_manifest(
    pairs_ok: list[str],
    pairs_failed: list[str],
    user_prompt_ids: list[str],
    stage: str = "gemma",
    round_id: int = 1,
    corpus_hash: str = "a" * 64,
) -> StageManifest:
    pairs = {}
    for pair_id in pairs_ok:
        per_up = {
            up_id: PerUserPromptManifest(ok=1, errors=0, total=1)
            for up_id in user_prompt_ids
        }
        pairs[pair_id] = PerPairManifest(
            prompt_pair_id=pair_id,
            ok=len(user_prompt_ids),
            errors=0,
            total=len(user_prompt_ids),
            per_user_prompt=per_up,
        )
    for pair_id in pairs_failed:
        per_up = {
            up_id: PerUserPromptManifest(ok=0, errors=1, total=1)
            for up_id in user_prompt_ids
        }
        pairs[pair_id] = PerPairManifest(
            prompt_pair_id=pair_id,
            ok=0,
            errors=len(user_prompt_ids),
            total=len(user_prompt_ids),
            failure_reason="worker_crash",
            per_user_prompt=per_up,
        )
    return StageManifest(
        stage=stage,  # type: ignore[arg-type]
        run_id="r1",
        round_id=round_id,
        pairs=pairs,
        surviving_pairs=list(pairs_ok),
        failed_pairs=[{"prompt_pair_id": p, "failure_reason": "worker_crash"} for p in pairs_failed],
        user_prompt_corpus_hash=corpus_hash,
    )


# ---------------------------------------------------------------------------
# build_missing_manifest — all artifacts missing
# ---------------------------------------------------------------------------


def test_build_missing_manifest_all_missing(tmp_path: Path):
    """When no artifacts exist, all (pair, UP, sample) cells are in the manifest."""
    path = build_missing_manifest(
        local_artifacts_root=tmp_path,
        prompt_pair_ids=["pair_A", "pair_B"],
        user_prompt_ids=["zh_001", "en_001"],
        sample_ids=["s1", "s2"],
        stage="gemma",
        round_id="1",
    )

    assert path.exists()
    rows = _read_jsonl(path)
    # 2 pairs × 2 user_prompts × 2 samples = 8 rows
    assert len(rows) == 8
    # All must have the 3 required keys.
    for row in rows:
        assert "prompt_pair_id" in row
        assert "user_prompt_id" in row
        assert "sample_id" in row


# ---------------------------------------------------------------------------
# build_missing_manifest — some artifacts present
# ---------------------------------------------------------------------------


def test_build_missing_manifest_some_present(tmp_path: Path):
    """Cells with existing artifacts are excluded from the resume manifest."""
    # Pre-populate one artifact.
    artifact_dir = (
        tmp_path / "gemma" / "round_1" / "pair_A" / "zh_001" / "s1"
    )
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "intermediate_prompt.txt").touch()

    path = build_missing_manifest(
        local_artifacts_root=tmp_path,
        prompt_pair_ids=["pair_A"],
        user_prompt_ids=["zh_001", "en_001"],
        sample_ids=["s1", "s2"],
        stage="gemma",
        round_id="1",
    )

    rows = _read_jsonl(path)
    # 1 pair × 2 UPs × 2 samples = 4, minus 1 present = 3
    assert len(rows) == 3
    # The present cell must not appear.
    present = {"prompt_pair_id": "pair_A", "user_prompt_id": "zh_001", "sample_id": "s1"}
    assert present not in rows


# ---------------------------------------------------------------------------
# build_missing_manifest — all artifacts present → empty JSONL
# ---------------------------------------------------------------------------


def test_build_missing_manifest_all_present(tmp_path: Path):
    """When all artifacts exist, the resume manifest is empty."""
    for pair_id in ["pair_A"]:
        for up_id in ["zh_001"]:
            for sample_id in ["s1"]:
                d = tmp_path / "gemma" / "round_2" / pair_id / up_id / sample_id
                d.mkdir(parents=True)
                (d / "intermediate_prompt.txt").touch()

    path = build_missing_manifest(
        local_artifacts_root=tmp_path,
        prompt_pair_ids=["pair_A"],
        user_prompt_ids=["zh_001"],
        sample_ids=["s1"],
        stage="gemma",
        round_id="2",
    )

    rows = _read_jsonl(path)
    assert rows == []


# ---------------------------------------------------------------------------
# build_missing_manifest — flux and qwen artifact filenames
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stage,filename", [
    ("flux", "result.png"),
    ("qwen", "eval.json"),
])
def test_build_missing_manifest_stage_artifacts(tmp_path: Path, stage: str, filename: str):
    """Each stage uses the correct artifact filename for presence detection."""
    # Pre-populate one artifact.
    d = tmp_path / stage / "round_1" / "pair_A" / "zh_001" / "s1"
    d.mkdir(parents=True)
    (d / filename).touch()

    path = build_missing_manifest(
        local_artifacts_root=tmp_path,
        prompt_pair_ids=["pair_A"],
        user_prompt_ids=["zh_001"],
        sample_ids=["s1"],
        stage=stage,
        round_id="1",
    )

    rows = _read_jsonl(path)
    assert rows == []


# ---------------------------------------------------------------------------
# build_missing_manifest — invalid stage raises ValueError
# ---------------------------------------------------------------------------


def test_build_missing_manifest_invalid_stage(tmp_path: Path):
    with pytest.raises(ValueError, match="Unknown stage"):
        build_missing_manifest(
            local_artifacts_root=tmp_path,
            prompt_pair_ids=["pair_A"],
            user_prompt_ids=["zh_001"],
            sample_ids=["s1"],
            stage="unknown_stage",
            round_id="1",
        )


# ---------------------------------------------------------------------------
# build_missing_manifest — custom output_path
# ---------------------------------------------------------------------------


def test_build_missing_manifest_custom_output_path(tmp_path: Path):
    custom = tmp_path / "custom_dir" / "my_resume.jsonl"
    path = build_missing_manifest(
        local_artifacts_root=tmp_path,
        prompt_pair_ids=["pair_A"],
        user_prompt_ids=["zh_001"],
        sample_ids=["s1"],
        stage="gemma",
        round_id="1",
        output_path=custom,
    )
    assert path == custom
    assert path.exists()


# ---------------------------------------------------------------------------
# build_survivor_manifest — legacy compat path (S02.01: function is hard-deprecated
# under V0.2.2; these tests opt into the SPRA_LEGACY_SURVIVOR_MANIFEST_OK escape
# hatch because they explicitly cover the legacy_cartesian_compat replay tooling
# behavior, not the V0.2.2 production code path).
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=False)
def _allow_legacy_survivor_manifest(monkeypatch):
    monkeypatch.setenv("SPRA_LEGACY_SURVIVOR_MANIFEST_OK", "1")


def test_build_survivor_manifest_basic(tmp_path: Path, _allow_legacy_survivor_manifest):
    """Surviving pairs × their surviving user_prompts appear in the JSONL."""
    manifest = _make_stage_manifest(
        pairs_ok=["pair_A", "pair_B"],
        pairs_failed=["pair_C"],
        user_prompt_ids=["zh_001", "en_001"],
    )

    path = build_survivor_manifest(
        prior_stage_manifest=manifest,
        local_manifests_root=tmp_path / "manifests",
    )

    assert path.exists()
    rows = _read_jsonl(path)
    # pair_A + pair_B × 2 user_prompts = 4 rows (each with sample_id "__all__")
    assert len(rows) == 4
    pair_ids_in_rows = {r["prompt_pair_id"] for r in rows}
    assert "pair_C" not in pair_ids_in_rows
    assert "pair_A" in pair_ids_in_rows
    assert "pair_B" in pair_ids_in_rows
    for row in rows:
        assert "prompt_pair_id" in row
        assert "user_prompt_id" in row
        assert "sample_id" in row


# ---------------------------------------------------------------------------
# build_survivor_manifest — all pairs failed → empty JSONL
# ---------------------------------------------------------------------------


def test_build_survivor_manifest_all_failed(tmp_path: Path, _allow_legacy_survivor_manifest):
    manifest = _make_stage_manifest(
        pairs_ok=[],
        pairs_failed=["pair_A", "pair_B"],
        user_prompt_ids=["zh_001"],
    )

    path = build_survivor_manifest(
        prior_stage_manifest=manifest,
        local_manifests_root=tmp_path / "manifests",
    )

    rows = _read_jsonl(path)
    assert rows == []


# ---------------------------------------------------------------------------
# build_survivor_manifest — no output_path or local_manifests_root raises
# ---------------------------------------------------------------------------


def test_build_survivor_manifest_no_path_raises(_allow_legacy_survivor_manifest):
    manifest = _make_stage_manifest(
        pairs_ok=["pair_A"],
        pairs_failed=[],
        user_prompt_ids=["zh_001"],
    )
    with pytest.raises(ValueError, match="Provide either"):
        build_survivor_manifest(prior_stage_manifest=manifest)


# ---------------------------------------------------------------------------
# build_survivor_manifest_with_samples — 3-key rows
# ---------------------------------------------------------------------------


def test_build_survivor_manifest_with_samples(tmp_path: Path, _allow_legacy_survivor_manifest):
    """build_survivor_manifest_with_samples emits 3-key rows per surviving cell."""
    manifest = _make_stage_manifest(
        pairs_ok=["pair_A", "pair_B"],
        pairs_failed=[],
        user_prompt_ids=["zh_001", "en_001"],
    )

    sample_ids_for_pair = {
        "pair_A": ["s1", "s2"],
        "pair_B": ["s1"],
    }

    path = build_survivor_manifest_with_samples(
        prior_stage_manifest=manifest,
        sample_ids_for_pair=sample_ids_for_pair,
        local_manifests_root=tmp_path / "manifests",
    )

    assert path.exists()
    rows = _read_jsonl(path)
    # pair_A: 2 UPs × 2 samples = 4 rows
    # pair_B: 2 UPs × 1 sample = 2 rows → total 6
    assert len(rows) == 6
    for row in rows:
        assert "prompt_pair_id" in row
        assert "user_prompt_id" in row
        assert "sample_id" in row
        assert row["sample_id"] != "__all__"


# ---------------------------------------------------------------------------
# build_survivor_manifest_with_samples — cells with errors excluded
# ---------------------------------------------------------------------------


def test_build_survivor_manifest_with_samples_errors_excluded(tmp_path: Path, _allow_legacy_survivor_manifest):
    """Cells with errors > 0 are excluded from the survivor manifest."""
    manifest = StageManifest(
        stage="gemma",
        run_id="r1",
        round_id=1,
        pairs={
            "pair_A": PerPairManifest(
                prompt_pair_id="pair_A",
                ok=1,
                errors=1,
                total=2,
                per_user_prompt={
                    "zh_001": PerUserPromptManifest(ok=1, errors=0, total=1),
                    "en_001": PerUserPromptManifest(ok=0, errors=1, total=1),
                },
            ),
        },
        surviving_pairs=["pair_A"],
        failed_pairs=[],
        user_prompt_corpus_hash="a" * 64,
    )

    path = build_survivor_manifest_with_samples(
        prior_stage_manifest=manifest,
        sample_ids_for_pair={"pair_A": ["s1"]},
        local_manifests_root=tmp_path / "manifests",
    )

    rows = _read_jsonl(path)
    # Only zh_001 cell survives (en_001 has errors).
    assert len(rows) == 1
    assert rows[0]["user_prompt_id"] == "zh_001"
