from __future__ import annotations

from backend.selection.relational_evidence_graph_v18 import (
    add_consensus_to_final,
    attach_relational_features,
    condition_score,
    direction_compatibility,
    dual_view_consensus,
    phrase_coverage,
    relational_score,
    relation_family,
    route_relational_top20,
    seed_bridge_score,
)


def paper(title: str, abstract: str = "", lane: str = "reverse_group_1") -> dict:
    return {
        "title": title,
        "abstract": abstract,
        "citation_lane": lane,
        "citation_direction": "candidate_cites_seed" if "reverse" in lane else "seed_cites_candidate",
        "v17_graph_pct": 0.5,
        "v17_trace_pct": 0.5,
        "v17_cross_pct": 0.5,
    }


def test_phrase_coverage_recognizes_source_free_paraphrase() -> None:
    text = "A universal adaptation framework without access to source data under category shift."
    phrase = "source-free universal domain adaptation with no access to source data and category shifts"
    assert phrase_coverage(text, phrase) > 0.6


def test_condition_score_uses_aliases() -> None:
    candidate = paper("DQN target Q-network analysis")
    condition = {"description": "an unrelated long description", "aliases": ["target Q-network"]}
    assert condition_score(candidate, condition) == 1.0


def test_relation_family_and_direction_are_query_conditioned() -> None:
    assert relation_family("proposed", "Who proposed it?") == "proposal"
    assert relation_family("extend", "works that extend it") == "extension"
    forward = paper("x", lane="forward_reference")
    reverse = paper("x", lane="reverse_group_1")
    assert direction_compatibility(forward, "proposal") > direction_compatibility(reverse, "proposal")
    assert direction_compatibility(reverse, "extension") > direction_compatibility(forward, "extension")


def test_seed_bridge_scores_query_aligned_paths() -> None:
    candidate = paper("follow-up")
    candidate["citation_seed_titles"] = ["Gradient descent finds global minima of deep neural networks"]
    contract = {"conditions": [{"description": "global descent for deep neural networks", "aliases": []}]}
    assert seed_bridge_score(candidate, "extend global descent", contract) > 0.4


def test_relational_router_respects_floor_and_protected_head() -> None:
    contract = {
        "conditions": [{"description": "deep Q-learning target network", "aliases": ["target network"]}],
        "relation": "analysed",
    }
    incumbents = attach_relational_features([
        paper("protected target network analysis"),
        paper("weak unrelated"),
        paper("weak tail"),
    ], "studies that analysed target networks", contract)
    frontier = attach_relational_features([
        paper("A theoretical analysis of Deep Q-learning", "We analyze the target network in DQN."),
    ], "studies that analysed target networks", contract)
    weights = (0.45, 0.20, 0.10, 0.05, 0.05, 0.05, 0.10)
    routed, actions = route_relational_top20(
        incumbents, frontier, weights=weights, margin=0.0, score_floor=0.3,
        max_promotions=1, protect_head=1,
    )
    assert routed[0]["title"] == "protected target network analysis"
    assert any(row["title"] == "A theoretical analysis of Deep Q-learning" for row in routed)
    assert len(actions) == 1


def test_adaptive_weights_follow_relation_family() -> None:
    row = paper("x")
    row.update({"v18_relation_family": "taxonomy", "v18_conjunction": 0.5, "v18_relation": 0.5,
                "v18_direction": 0.5, "v18_seed_bridge": 0.5})
    weights = {"default": (1, 0, 0, 0, 0, 0, 0), "taxonomy": (0, 0, 0, 0, 1, 0, 0)}
    assert relational_score(row, weights) == row["v17_graph_pct"]


def test_dual_view_consensus_adds_only_agreed_candidate() -> None:
    first = paper("agreed")
    first.update({"v18_conjunction": 1.0, "v18_relation": 1.0, "v18_direction": 1.0, "v18_seed_bridge": 1.0})
    second = paper("other")
    second.update({"v18_conjunction": 0.1, "v18_relation": 0.1, "v18_direction": 0.1, "v18_seed_bridge": 0.1})
    strict = (0.45, 0.15, 0.1, 0.05, 0.1, 0.05, 0.1)
    graph = (0.35, 0.15, 0.1, 0.1, 0.15, 0.1, 0.05)
    candidate, audit = dual_view_consensus([first, second], strict, graph, strict_floor=.5, graph_floor=.5)
    routed, added = add_consensus_to_final([paper("incumbent")], candidate)
    assert audit["same_candidate"] is True
    assert added is True
    assert [row["title"] for row in routed] == ["incumbent", "agreed"]
