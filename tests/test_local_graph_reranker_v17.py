from __future__ import annotations

from pathlib import Path

from backend.selection.cross_encoder_selector import CrossEncoderSelector
from backend.selection.local_graph_reranker_v17 import (
    composite_score,
    contract_query,
    route_final_set,
    route_frontier_top20,
    score_query_pool,
    select_multiview_frontier_pool,
)


def paper(title: str, *, score: float = 0.0, seeds: int = 0, rank: int = 99) -> dict:
    return {
        "title": title,
        "abstract": f"Abstract for {title}",
        "final_score": score,
        "citation_seed_count": seeds,
        "_openalex_rank": rank,
    }


def attach(row: dict, cross: float, graph: float = 0.0, trace: float = 0.0) -> dict:
    return {**row, "v17_cross_pct": cross, "v17_graph_pct": graph, "v17_trace_pct": trace, "v17_cross_raw": cross}


def test_multiview_pool_excludes_incumbents_and_deduplicates() -> None:
    incumbents = [paper("Existing")]
    frontier = [paper("Existing"), paper("New", score=1), paper("New", score=2), paper("Other")]
    result = select_multiview_frontier_pool(frontier, incumbents, limit=10)
    assert {row["title"] for row in result} == {"New", "Other"}


def test_contract_query_contains_conditions_and_exclusions() -> None:
    result = contract_query("Find papers", {
        "conditions": [{"description": "uses target networks", "aliases": ["target Q-network"]}],
        "relation": "analyzes",
        "exclusions": ["mere mentions"],
    })
    assert "uses target networks" in result
    assert "target Q-network" in result
    assert "mere mentions" in result


def test_score_query_pool_uses_deterministic_fallback(tmp_path: Path) -> None:
    selector = CrossEncoderSelector("missing/local-model", tmp_path / "scores.json", local_files_only=True)
    rows, runtime = score_query_pool("target network", {}, [paper("target network analysis"), paper("biology")], selector)
    assert len(rows) == 2
    assert runtime["pairs"] == 2
    assert rows[0]["v17_cross_pct"] > rows[1]["v17_cross_pct"]


def test_top20_router_protects_head_and_requires_margin() -> None:
    incumbents = [attach(paper("head"), .1), attach(paper("weak"), .2), attach(paper("tail"), .3)]
    frontier = [attach(paper("strong"), .9)]
    routed, actions = route_frontier_top20(
        incumbents, frontier, weights=(1, 0, 0), margin=.1, max_promotions=1, protect_head=1,
    )
    assert routed[0]["title"] == "head"
    assert {row["title"] for row in routed} == {"head", "tail", "strong"}
    assert len(actions) == 1


def test_final_router_replaces_only_one_residual() -> None:
    selected = [attach(paper("a"), .4), attach(paper("b"), .5)]
    pool = [*selected, attach(paper("c"), .95), attach(paper("d"), .9)]
    routed, actions = route_final_set(
        selected, pool, weights=(1, 0, 0), margin=.1, cross_floor=.9,
    )
    assert len(routed) == 2
    assert {row["title"] for row in routed} == {"b", "c"}
    assert len(actions) == 1
    assert max(composite_score(row, (1, 0, 0)) for row in routed) == .95
