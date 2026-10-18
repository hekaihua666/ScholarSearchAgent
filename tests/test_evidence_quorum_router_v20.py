from backend.selection.evidence_quorum_router_v20 import (
    attach_quorum_features,
    evidence_quorum_score,
    route_evidence_quorum_top20,
)


WEIGHTS = (0.45, 0.15, 0.10, 0.05, 0.10, 0.05, 0.10)


def paper(title: str, score: float, *, strong: bool = False) -> dict:
    value = 1.0 if strong else score
    return {
        "title": title,
        "final_score": score,
        "lexical_score": score,
        "v18_conjunction": score,
        "v18_relation": score,
        "v18_direction": score,
        "v18_seed_bridge": score,
        "v17_graph_pct": value,
        "v17_trace_pct": value,
        "v17_cross_pct": value,
    }


def test_quorum_score_uses_third_strongest_view():
    row = {
        "v17_cross_pct": 1.0,
        "v17_trace_pct": 0.9,
        "v17_graph_pct": 0.8,
        "v20_lexical_pct": 0.2,
        "v20_final_pct": 0.1,
    }
    assert evidence_quorum_score(row) == 0.8 + 0.05 * (1.0 + 0.9 + 0.8) / 3


def test_attach_quorum_features_is_deterministic_and_bounded():
    inc, front = attach_quorum_features([paper("A", 0.2)], [paper("B", 0.8)])
    assert inc[0]["v20_final_pct"] == 0.0
    assert front[0]["v20_final_pct"] == 1.0
    assert 0.0 <= front[0]["v20_lexical_pct"] <= 1.0


def test_strong_quorum_candidate_only_replaces_protected_tail():
    incumbents = [paper(f"inc-{index}", 0.8 - index * 0.01) for index in range(20)]
    candidate = paper("consensus candidate", 0.95, strong=True)
    routed, actions = route_evidence_quorum_top20(
        incumbents,
        [candidate],
        weights=WEIGHTS,
        reg_score_floor=2.0,
        quorum_floor=0.9,
        quorum_margin=0.1,
        quorum_protect_head=18,
    )
    assert [row["title"] for row in routed[:18]] == [row["title"] for row in incumbents[:18]]
    assert any(row["title"] == "consensus candidate" for row in routed[18:])
    assert actions[-1]["route"] == "evidence_quorum"


def test_weak_candidate_does_not_trigger_quorum_route():
    incumbents = [paper(f"inc-{index}", 0.8 - index * 0.01) for index in range(20)]
    routed, actions = route_evidence_quorum_top20(
        incumbents,
        [paper("weak", 0.1)],
        weights=WEIGHTS,
        reg_score_floor=2.0,
        quorum_floor=1.1,
    )
    assert [row["title"] for row in routed] == [row["title"] for row in incumbents]
    assert not actions
