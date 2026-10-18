from __future__ import annotations

from backend.retrieval.query_conditioned_citation_v11 import (
    MAX_SEEDS,
    build_frontier_queries,
    select_frontier_seeds,
)


def paper(title: str, wid: int, score: float, refs: int = 0):
    return {
        "title": title,
        "openalex_id": f"https://openalex.org/W{wid}",
        "final_score": score,
        "lexical_score": score,
        "constraint_coverage_score": score,
        "referenced_works": [f"https://openalex.org/W{wid * 100 + i}" for i in range(refs)],
    }


def test_seed_selection_is_bounded_deduplicated_and_mixes_graph_view():
    relevance = [paper(f"relevance {chr(97+i//26)}{chr(97+i%26)}", i + 1, 1 - i / 60, 1) for i in range(45)]
    bridge = paper("graph bridge", 99, .01, 80)
    seeds = select_frontier_seeds(relevance[:20], [*relevance, bridge])
    assert len(seeds) == MAX_SEEDS
    assert len({row["openalex_id"] for row in seeds}) == len(seeds)
    assert any(row["title"] == "graph bridge" for row in seeds)


def test_query_views_include_original_and_contract_aliases_without_ids():
    contract = {
        "conditions": [{"id": "C1", "description": "paper proposes target networks", "aliases": ["DQN", "deep Q-learning"]}],
        "relation": "introduced",
    }
    rows = build_frontier_queries(
        "Which papers analyze target networks for Deep Q-learning?", contract
    )
    assert [row[0] for row in rows] == ["reverse_query", "reverse_contract"]
    assert "DQN" in rows[1][1]
    assert all("query_id" not in text.lower() and "gold" not in text.lower() for _, text in rows)


def test_seed_selection_requires_openalex_identity():
    seeds = select_frontier_seeds([], [{"title": "no id", "final_score": 1}, paper("with id", 1, .2)])
    assert [row["title"] for row in seeds] == ["with id"]
