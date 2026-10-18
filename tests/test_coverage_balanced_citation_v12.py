from backend.retrieval.coverage_balanced_citation_v12 import round_robin_neighbor_ids


def test_round_robin_neighbor_budget_covers_seeds_before_depth():
    seeds = [
        {"referenced_works": [f"https://openalex.org/W{i}{j}" for j in range(1, 5)]}
        for i in range(1, 4)
    ]
    ids = round_robin_neighbor_ids(seeds, "referenced_works", limit=7)
    assert ids[:3] == ["W11", "W21", "W31"]
    assert ids[3:6] == ["W12", "W22", "W32"]
    assert ids[6] == "W13"


def test_round_robin_deduplicates_shared_neighbors():
    seeds = [
        {"related_works": ["W1", "W2"]},
        {"related_works": ["https://openalex.org/W1", "W3"]},
    ]
    assert round_robin_neighbor_ids(seeds, "related_works") == ["W1", "W2", "W3"]
