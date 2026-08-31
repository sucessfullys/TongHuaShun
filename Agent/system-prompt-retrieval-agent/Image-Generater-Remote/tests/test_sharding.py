"""
tests/test_sharding.py — unit tests for pipelines.sharding
"""

import pytest
from pipelines.sharding import split_indices, shard_items


# ---------------------------------------------------------------------------
# split_indices
# ---------------------------------------------------------------------------

class TestSplitIndicesBasic:
    def test_example_seven_three(self):
        result = split_indices(7, 3)
        assert result == [[0, 3, 6], [1, 4], [2, 5]]

    def test_example_six_three(self):
        result = split_indices(6, 3)
        assert result == [[0, 3], [1, 4], [2, 5]]

    def test_example_five_two(self):
        result = split_indices(5, 2)
        assert result == [[0, 2, 4], [1, 3]]

    def test_example_one_shard(self):
        result = split_indices(5, 1)
        assert result == [[0, 1, 2, 3, 4]]

    def test_zero_items(self):
        result = split_indices(0, 3)
        assert result == [[], [], []]

    def test_zero_items_one_shard(self):
        result = split_indices(0, 1)
        assert result == [[]]


class TestSplitIndicesCoverage:
    """Every index appears exactly once across all shards (union == range(n_items))."""

    def test_coverage_7_3(self):
        shards = split_indices(7, 3)
        all_indices = [i for shard in shards for i in shard]
        assert sorted(all_indices) == list(range(7))

    def test_coverage_100_7(self):
        shards = split_indices(100, 7)
        all_indices = [i for shard in shards for i in shard]
        assert sorted(all_indices) == list(range(100))

    def test_coverage_1_1(self):
        shards = split_indices(1, 1)
        all_indices = [i for shard in shards for i in shard]
        assert all_indices == [0]

    def test_coverage_zero_items(self):
        shards = split_indices(0, 4)
        all_indices = [i for shard in shards for i in shard]
        assert all_indices == []


class TestSplitIndicesDisjoint:
    """Shards must be pairwise disjoint (no index appears in more than one shard)."""

    def test_disjoint_7_3(self):
        shards = split_indices(7, 3)
        seen = set()
        for shard in shards:
            for idx in shard:
                assert idx not in seen, f"index {idx} appears in multiple shards"
                seen.add(idx)

    def test_disjoint_50_5(self):
        shards = split_indices(50, 5)
        seen = set()
        for shard in shards:
            for idx in shard:
                assert idx not in seen
                seen.add(idx)


class TestSplitIndicesShardCount:
    """Returned list always has exactly n_shards entries."""

    def test_shard_count_matches_7_3(self):
        assert len(split_indices(7, 3)) == 3

    def test_shard_count_matches_0_5(self):
        assert len(split_indices(0, 5)) == 5

    def test_shard_count_matches_1_1(self):
        assert len(split_indices(1, 1)) == 1


class TestSplitIndicesOneShard:
    """n_shards=1 must collect everything in a single list."""

    def test_one_shard_ten(self):
        result = split_indices(10, 1)
        assert len(result) == 1
        assert result[0] == list(range(10))

    def test_one_shard_zero(self):
        result = split_indices(0, 1)
        assert result == [[]]


class TestSplitIndicesNShardsGtNItems:
    """When n_shards >= n_items some shards will be empty."""

    def test_shards_equal_items(self):
        result = split_indices(4, 4)
        assert result == [[0], [1], [2], [3]]

    def test_more_shards_than_items(self):
        result = split_indices(3, 6)
        # shards 0,1,2 get one item each; shards 3,4,5 are empty
        assert result[0] == [0]
        assert result[1] == [1]
        assert result[2] == [2]
        assert result[3] == []
        assert result[4] == []
        assert result[5] == []
        # coverage still holds
        all_indices = [i for shard in result for i in shard]
        assert sorted(all_indices) == list(range(3))

    def test_one_item_five_shards(self):
        result = split_indices(1, 5)
        assert result[0] == [0]
        for k in range(1, 5):
            assert result[k] == []


class TestSplitIndicesModuloProperty:
    """Every index in shard k must satisfy index % n_shards == k."""

    def test_modulo_property_7_3(self):
        n_shards = 3
        shards = split_indices(7, n_shards)
        for k, shard in enumerate(shards):
            for idx in shard:
                assert idx % n_shards == k

    def test_modulo_property_20_7(self):
        n_shards = 7
        shards = split_indices(20, n_shards)
        for k, shard in enumerate(shards):
            for idx in shard:
                assert idx % n_shards == k


class TestSplitIndicesErrors:
    def test_zero_shards_raises(self):
        with pytest.raises(ValueError):
            split_indices(5, 0)

    def test_negative_shards_raises(self):
        with pytest.raises(ValueError):
            split_indices(5, -1)

    def test_negative_items_raises(self):
        with pytest.raises(ValueError):
            split_indices(-1, 3)


# ---------------------------------------------------------------------------
# shard_items
# ---------------------------------------------------------------------------

class TestShardItemsBasic:
    def test_example_seven_strings(self):
        items = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        result = shard_items(items, 3)
        assert result == [['a', 'd', 'g'], ['b', 'e'], ['c', 'f']]

    def test_one_shard(self):
        items = [10, 20, 30]
        result = shard_items(items, 1)
        assert result == [[10, 20, 30]]

    def test_empty_list(self):
        result = shard_items([], 3)
        assert result == [[], [], []]

    def test_more_shards_than_items(self):
        items = ['x', 'y']
        result = shard_items(items, 5)
        assert result[0] == ['x']
        assert result[1] == ['y']
        for k in range(2, 5):
            assert result[k] == []


class TestShardItemsCoverage:
    """All items must appear exactly once across all shards."""

    def test_coverage_integers(self):
        items = list(range(13))
        shards = shard_items(items, 4)
        all_items = [v for shard in shards for v in shard]
        assert sorted(all_items) == items

    def test_coverage_strings(self):
        items = list("abcdefghij")
        shards = shard_items(items, 3)
        all_items = [v for shard in shards for v in shard]
        assert sorted(all_items) == sorted(items)


class TestShardItemsConsistency:
    """shard_items must be consistent with split_indices applied to the same list."""

    def test_consistent_with_split_indices(self):
        items = list(range(11))
        n_shards = 4
        idx_shards = split_indices(len(items), n_shards)
        item_shards = shard_items(items, n_shards)
        for k in range(n_shards):
            assert item_shards[k] == [items[i] for i in idx_shards[k]]

    def test_consistent_non_integer_items(self):
        items = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
        n_shards = 2
        idx_shards = split_indices(len(items), n_shards)
        item_shards = shard_items(items, n_shards)
        for k in range(n_shards):
            assert item_shards[k] == [items[i] for i in idx_shards[k]]
