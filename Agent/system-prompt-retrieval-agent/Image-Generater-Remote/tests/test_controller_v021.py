"""S05.13-.19 unit tests for V0.2.1 controller dispatch path.

These tests exercise pure helpers (`_v021_active`, `_per_pair_manifest_init`,
`_accumulate_pair_manifest`, `_partition_pairs`, `_v021_cell_root`,
`_read_manifest_rows`) plus the schema validators on
`StageRequest`/`GemmaUserPrompt`/`PromptPairRequest`. Real GPU execution
(`_execute_*_v021`) is not exercised here — that's S05.23 / S05.26.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# Ensure controller imports resolve when running pytest from project root
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from workflow.controller import (
    StageRequest,
    GemmaUserPrompt,
    PromptPairRequest,
    state,
    _v021_active,
    _per_pair_manifest_init,
    _per_user_prompt_init,
    _accumulate_pair_manifest,
    _partition_pairs,
    _v021_cell_root,
    _read_manifest_rows,
)


# S05.13 — V0.2.1 dispatch decision

def test_v021_inactive_for_legacy_payload():
    r = StageRequest(prompt_id="p1", limit=3)
    assert _v021_active(r) is False


def test_v021_active_when_prompt_pairs_present():
    r = StageRequest(
        run_id="r1",
        prompt_pairs=[PromptPairRequest(prompt_pair_id="A", system_prompt_id="s1")],
        user_prompts=[GemmaUserPrompt(user_prompt_id="zh_001", language="zh", text="x")],
        user_prompt_corpus_hash="a" * 64,
    )
    assert _v021_active(r) is True


def test_v021_active_when_only_user_prompts_present():
    r = StageRequest(
        user_prompts=[GemmaUserPrompt(user_prompt_id="zh_001", language="zh", text="x")],
    )
    assert _v021_active(r) is True


# S05.14 — Per-pair manifest accumulation

def test_per_pair_manifest_init_zero_state():
    mf = _per_pair_manifest_init("A")
    assert mf == {
        "prompt_pair_id": "A",
        "ok": 0,
        "errors": 0,
        "total": 0,
        "failure_reason": None,
        "per_user_prompt": {},
    }


def test_accumulate_pair_manifest_aggregates_into_pair_and_per_up():
    mf = _per_pair_manifest_init("A")
    fake = {"s1": {"status": "ok"}, "s2": {"status": "ok"}, "s3": {"status": "error"}}
    _accumulate_pair_manifest(mf, "zh_001", fake)
    assert mf["ok"] == 2 and mf["errors"] == 1 and mf["total"] == 3
    assert mf["per_user_prompt"]["zh_001"]["ok"] == 2
    assert mf["per_user_prompt"]["zh_001"]["errors"] == 1
    assert mf["per_user_prompt"]["zh_001"]["total"] == 3


def test_accumulate_pair_manifest_multiple_user_prompts_summed():
    mf = _per_pair_manifest_init("A")
    _accumulate_pair_manifest(mf, "zh_001", {"s1": {"status": "ok"}})
    _accumulate_pair_manifest(mf, "en_001", {"s1": {"status": "ok"}, "s2": {"status": "error"}})
    assert mf["ok"] == 2 and mf["errors"] == 1 and mf["total"] == 3
    assert set(mf["per_user_prompt"].keys()) == {"zh_001", "en_001"}


# S05.15 — Strict-mode partition

def test_partition_strict_marks_failed_pair_with_errors():
    good = _per_pair_manifest_init("A")
    _accumulate_pair_manifest(good, "zh_001", {"s1": {"status": "ok"}})
    bad = _per_pair_manifest_init("B")
    _accumulate_pair_manifest(bad, "zh_001", {"s1": {"status": "error"}})
    surviving, failed = _partition_pairs({"A": good, "B": bad}, allow_partial=False)
    assert surviving == ["A"]
    assert {f["prompt_pair_id"]: f["failure_reason"] for f in failed} == {"B": "worker_error"}


def test_partition_strict_marks_empty_pair_as_no_cells_dispatched():
    empty = _per_pair_manifest_init("C")
    surviving, failed = _partition_pairs({"C": empty}, allow_partial=False)
    assert surviving == []
    assert failed[0]["failure_reason"] == "no_cells_dispatched"


# S05.16 — allow_partial mode

def test_partition_allow_partial_accepts_pair_with_zero_errors():
    good = _per_pair_manifest_init("A")
    _accumulate_pair_manifest(good, "zh_001", {"s1": {"status": "ok"}})
    surviving, failed = _partition_pairs({"A": good}, allow_partial=True)
    assert surviving == ["A"]
    assert failed == []


# S05.17 — Cell output root layout

def test_v021_cell_root_includes_run_round_pair_user_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "output_root", str(tmp_path))
    monkeypatch.setattr(state, "run_id", "run_xyz")
    r = StageRequest(run_id="run_xyz", round_id=2, user_prompt_corpus_hash="a" * 64)
    out = _v021_cell_root(r, "gemma", "pairA", "zh_001")
    assert out == os.path.join(str(tmp_path), "v021", "run_xyz", "round_2", "pairA", "zh_001")


# S05.18 — Manifest JSONL row reader

def test_read_manifest_rows_parses_jsonl_3key(tmp_path):
    p = tmp_path / "manifest.jsonl"
    p.write_text("\n".join([
        json.dumps({"prompt_pair_id": "A", "user_prompt_id": "zh_001", "sample_id": "s1"}),
        json.dumps({"prompt_pair_id": "B", "user_prompt_id": "en_001", "sample_id": "s2"}),
        "",  # blank line tolerated
    ]))
    rows = _read_manifest_rows(str(p))
    assert len(rows) == 2
    assert rows[0]["prompt_pair_id"] == "A" and rows[0]["user_prompt_id"] == "zh_001"
    assert rows[1]["sample_id"] == "s2"


# S05.19 — Schema validators (manifest_purpose pairing)

def test_stage_request_manifest_path_without_purpose_raises():
    with pytest.raises(Exception) as e:
        StageRequest(sample_manifest_path="/tmp/manifest.jsonl")
    assert "missing_manifest_purpose" in str(e.value)


def test_stage_request_manifest_path_with_purpose_valid():
    r = StageRequest(
        sample_manifest_path="/tmp/manifest.jsonl",
        sample_manifest_path_purpose="resume_missing_cells",
    )
    assert r.sample_manifest_path_purpose == "resume_missing_cells"


def test_gemma_user_prompt_rejects_unknown_language():
    with pytest.raises(Exception):
        GemmaUserPrompt(user_prompt_id="ja_001", language="ja", text="x")


def test_legacy_back_compat_payload_still_parses():
    r = StageRequest(prompt_id="legacy_id", limit=3, system_prompt_text="hello")
    assert r.prompt_id == "legacy_id"
    assert r.prompt_pairs is None and r.user_prompts is None
