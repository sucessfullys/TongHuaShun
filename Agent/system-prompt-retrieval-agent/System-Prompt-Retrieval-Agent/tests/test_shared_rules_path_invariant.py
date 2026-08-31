"""Regression: memory/shared_rules.csv prompt_path stays relative to memory_root.

Previously ``MemoryManager.load_rules()`` mutated each row's ``prompt_path``
in place, prepending ``memory_root`` to relative paths. When the writer in
``runner_v022._update_shared_rules`` round-tripped that data through
``save_rules``, prior-round entries got persisted with machine-specific
absolute paths while current-round entries used the canonical relative
form. The CSV ended up with a mix and was not portable across machines.

Option A fix: ``load_rules`` returns rows untouched; callers resolve to an
absolute path at use-time via :func:`memory.shared.resolve_prompt_path`.
This test pins the invariant.
"""

from __future__ import annotations

import csv
from pathlib import Path

from system_prompt_retrieval_agent.memory.manager import MemoryManager
from system_prompt_retrieval_agent.memory.shared import (
    FIELDNAMES,
    resolve_prompt_path,
)


def _seed_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDNAMES})


def test_load_rules_does_not_mutate_relative_prompt_path(tmp_path):
    """A round-trip (load → save) must NOT rewrite a relative prompt_path
    into an absolute one. Prior bug: load_rules prepended memory_root, then
    save_rules persisted the absolute form, machine-locking the CSV."""
    mem = MemoryManager(memory_root=tmp_path / "mem")
    rel = "pairs/20260427T093743Z-4118036d/system_prompt_for_tryon_v5.yaml"
    _seed_csv(mem.shared_csv, [
        {
            "schema_version": "1.0",
            "rule_id": "seed__system_prompt_for_tryon_v5",
            "text": "seed rule",
            "source_round": "0",
            "confidence": "1.0",
            "last_seen": "0",
            "tags": "seed",
            "system_prompt_id": "system_prompt_for_tryon_v5",
            "prompt_pair_id": "system_prompt_for_tryon_v5",
            "source_run_id": "20260427T093743Z-4118036d",
            "n_cells": "",
            "pair_overall": "0.85",
            "prompt_path": rel,
            "improvement_summary": "",
            "axis_means_json": "{}",
            "worst_failure_tags": "",
            "worst_user_prompt_id": "",
            "zh_mean_score": "",
            "en_mean_score": "",
            "delta_vs_prev": "",
        },
    ])

    # Round-trip: read raw -> save_rules -> read raw again
    rows = mem.load_rules()
    assert rows[0]["prompt_path"] == rel, (
        f"load_rules must return CSV value verbatim; got {rows[0]['prompt_path']!r}"
    )

    # save_rules writes the rows back as-is; this is the failure mode the
    # mutation bug created (absolute path leaking in).
    mem.save_rules(rows)

    with mem.shared_csv.open(newline="", encoding="utf-8") as fh:
        re_read = next(csv.DictReader(fh))
    assert re_read["prompt_path"] == rel, (
        f"after save_rules, CSV prompt_path must remain relative; got "
        f"{re_read['prompt_path']!r}"
    )
    # Hard assert it is NOT an absolute path under tmp_path.
    assert not Path(re_read["prompt_path"]).is_absolute()


def test_resolve_prompt_path_helper_makes_relative_absolute(tmp_path):
    """The runtime resolver must combine relative prompt_path with
    memory_root to yield an absolute, openable Path."""
    mem_root = tmp_path / "mem"
    pair_dir = mem_root / "pairs" / "run-A"
    pair_dir.mkdir(parents=True)
    yaml_file = pair_dir / "p.yaml"
    yaml_file.write_text("system_prompt: hello\n")

    rel = "pairs/run-A/p.yaml"
    resolved = resolve_prompt_path(rel, mem_root)
    assert resolved is not None
    assert resolved.is_absolute()
    assert resolved.is_file()
    assert resolved == yaml_file


def test_resolve_prompt_path_helper_passes_through_absolute(tmp_path):
    """Absolute paths in legacy CSV rows must pass through unchanged
    (so existing CSVs from before the fix still work)."""
    mem_root = tmp_path / "mem"
    abs_file = tmp_path / "elsewhere" / "p.yaml"
    abs_file.parent.mkdir(parents=True)
    abs_file.write_text("system_prompt: x\n")

    resolved = resolve_prompt_path(str(abs_file), mem_root)
    assert resolved == abs_file


def test_resolve_prompt_path_handles_empty(tmp_path):
    """Empty / None prompt_path returns None so callers can branch
    on a single check."""
    assert resolve_prompt_path("", tmp_path) is None
    assert resolve_prompt_path(None, tmp_path) is None
