"""S09.10 — Cross-round cell_id collision detection.

cell_id_index.jsonl records every (run_id, round_id, prompt_pair_id,
user_prompt_id, sample_id) cell. The detector flags any cell tuple that
appears in two or more rounds.
"""
from __future__ import annotations

import pytest

from system_prompt_retrieval_agent.agent_loop_helpers import (
    CrossRoundCellIdCollision,
    assert_no_cross_round_cell_id_collision,
    detect_cross_round_collisions,
    write_cell_id_index_row,
)


def test_clean_index_has_no_collisions(tmp_path):
    p = tmp_path / "cell_id_index.jsonl"
    write_cell_id_index_row(p, "r1", 1, ("A", "zh_001", "s1"))
    write_cell_id_index_row(p, "r1", 1, ("A", "en_001", "s1"))
    write_cell_id_index_row(p, "r1", 2, ("B", "zh_001", "s1"))
    assert detect_cross_round_collisions(p) == []
    assert_no_cross_round_cell_id_collision(p)  # no raise


def test_same_cell_in_two_rounds_is_collision(tmp_path):
    p = tmp_path / "cell_id_index.jsonl"
    write_cell_id_index_row(p, "r1", 1, ("A", "zh_001", "s1"))
    write_cell_id_index_row(p, "r1", 2, ("A", "zh_001", "s1"))  # COLLISION
    cols = detect_cross_round_collisions(p)
    assert len(cols) == 1
    assert cols[0]["round_ids"] == [1, 2]


def test_collision_raises():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cell_id_index.jsonl"
        write_cell_id_index_row(p, "r1", 1, ("A", "zh_001", "s1"))
        write_cell_id_index_row(p, "r1", 2, ("A", "zh_001", "s1"))
        with pytest.raises(CrossRoundCellIdCollision) as exc:
            assert_no_cross_round_cell_id_collision(p)
        assert "pair='A'" in str(exc.value)
        assert "user_prompt='zh_001'" in str(exc.value)


def test_same_pair_different_user_prompt_no_collision(tmp_path):
    """Same (pair, sample) under different user_prompts is NOT a collision."""
    p = tmp_path / "cell_id_index.jsonl"
    write_cell_id_index_row(p, "r1", 1, ("A", "zh_001", "s1"))
    write_cell_id_index_row(p, "r1", 1, ("A", "en_001", "s1"))
    assert detect_cross_round_collisions(p) == []


def test_missing_index_no_raise(tmp_path):
    p = tmp_path / "does_not_exist.jsonl"
    assert detect_cross_round_collisions(p) == []
    assert_no_cross_round_cell_id_collision(p)
