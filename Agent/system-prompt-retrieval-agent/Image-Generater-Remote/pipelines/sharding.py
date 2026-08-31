"""
pipelines/sharding.py — index and item sharding utilities.

Distributes a sequence across N shards using a round-robin (modulo) strategy:
    item i  ->  shard (i % n_shards)

This guarantees that shard sizes differ by at most 1, and that the assignment
is deterministic and reproducible given the same n_items / n_shards values.
"""


def split_indices(n_items: int, n_shards: int) -> list[list[int]]:
    """Return a list of n_shards lists, each containing the indices for that shard.

    Indices 0 .. n_items-1 are distributed by: shard_id = index % n_shards.

    Args:
        n_items:  Total number of items (non-negative integer).
        n_shards: Number of shards (positive integer, >= 1).

    Returns:
        A list of length n_shards.  shards[k] contains every index i such that
        i % n_shards == k, in ascending order.

    Example:
        >>> split_indices(7, 3)
        [[0, 3, 6], [1, 4], [2, 5]]
    """
    if n_shards < 1:
        raise ValueError(f"n_shards must be >= 1, got {n_shards}")
    if n_items < 0:
        raise ValueError(f"n_items must be >= 0, got {n_items}")

    shards: list[list[int]] = [[] for _ in range(n_shards)]
    for i in range(n_items):
        shards[i % n_shards].append(i)
    return shards


def shard_items(items: list, n_shards: int) -> list[list]:
    """Return a list of n_shards lists containing the actual items (not indices).

    The assignment rule is the same as split_indices: item at position i goes to
    shard i % n_shards.

    Args:
        items:    The source sequence.
        n_shards: Number of shards (positive integer, >= 1).

    Returns:
        A list of length n_shards.  shards[k] contains every item whose original
        index i satisfies i % n_shards == k, in original relative order.

    Example:
        >>> shard_items(['a', 'b', 'c', 'd', 'e', 'f', 'g'], 3)
        [['a', 'd', 'g'], ['b', 'e'], ['c', 'f']]
    """
    index_shards = split_indices(len(items), n_shards)
    return [[items[i] for i in shard] for shard in index_shards]
