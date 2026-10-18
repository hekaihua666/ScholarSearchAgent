from backend.retrieval.grouped_citation_frontier_v13 import (
    head_weighted_reference_ids,
    partition_seeds,
)


def test_partition_seeds_is_stable_and_bounded():
    seeds = [{"title": str(i)} for i in range(23)]
    groups = partition_seeds(seeds, 10)
    assert [len(group) for group in groups] == [10, 10, 3]
    assert [row["title"] for group in groups for row in group] == [str(i) for i in range(23)]


def test_head_weighted_reference_policy_preserves_depth():
    seeds = [
        {"referenced_works": [f"W{index}" for index in range(1, 8)]},
        {"referenced_works": ["W8", "W9"]},
    ]
    assert head_weighted_reference_ids(seeds, 5) == ["W1", "W2", "W3", "W4", "W5"]


def test_head_weighted_reference_policy_deduplicates():
    seeds = [{"referenced_works": ["W1", "W2"]}, {"referenced_works": ["W2", "W3"]}]
    assert head_weighted_reference_ids(seeds) == ["W1", "W2", "W3"]
