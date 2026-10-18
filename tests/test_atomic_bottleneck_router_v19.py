from __future__ import annotations

from backend.selection.atomic_bottleneck_router_v19 import (
    atomic_bottleneck_score,
    atomic_views,
    attach_atomic_cross_evidence,
    evidence_dominance,
    route_atomic_bottleneck_top20,
    select_atomic_screen_pool,
)


class FakeSelector:
    def predict(self, pairs, batch_size=64):
        del batch_size
        scores = []
        for query, document in pairs:
            q = set(query.lower().split())
            d = set(document.lower().split())
            scores.append(float(len(q & d)))
        return scores, {"runtime": "fake", "cache_hits": 0}


def paper(title, **values):
    return {"title": title, "abstract": values.pop("abstract", ""), **values}


def test_atomic_views_keep_conditions_and_relation_separate():
    views = atomic_views("find A using B", {
        "conditions": [{"id": "a", "description": "topic A"}, {"id": "b", "description": "method B"}],
        "relation": "uses",
    })
    assert [view["kind"] for view in views] == ["condition", "condition", "relation"]


def test_atomic_evidence_penalizes_missing_condition():
    rows, runtime = attach_atomic_cross_evidence(
        [paper("both", abstract="topic A method B"), paper("partial", abstract="topic A")],
        "find topic A using method B",
        {"conditions": [{"id": "a", "description": "topic A"}, {"id": "b", "description": "method B"}]},
        FakeSelector(),
    )
    assert rows[0]["v19_atomic_min"] > rows[1]["v19_atomic_min"]
    assert runtime["pair_count"] == 4


def test_score_rewards_atomic_bottleneck():
    strong = paper("strong", v19_atomic_min=.9, v19_atomic_geo=.9, v19_relation_pct=.8,
                   v17_graph_pct=.7, v17_trace_pct=.7, v17_cross_pct=.7, v18_conjunction=.8)
    weak = {**strong, "title": "weak", "v19_atomic_min": .1}
    assert atomic_bottleneck_score(strong) > atomic_bottleneck_score(weak)


def test_dominance_counts_independent_evidence_views():
    candidate = paper("candidate", v19_atomic_min=.8, v19_atomic_geo=.8, v19_relation_pct=.8,
                      v17_graph_pct=.8, v17_trace_pct=.2, v17_cross_pct=.2)
    incumbent = paper("incumbent", v19_atomic_min=.5, v19_atomic_geo=.5, v19_relation_pct=.5,
                      v17_graph_pct=.5, v17_trace_pct=.5, v17_cross_pct=.5)
    assert evidence_dominance(candidate, incumbent) == 4


def test_router_requires_weak_condition_noninferiority():
    incumbents = [paper(str(i), v19_atomic_min=.4, v19_atomic_geo=.4, v19_relation_pct=.4,
                        v17_graph_pct=.4, v17_trace_pct=.4, v17_cross_pct=.4, v18_conjunction=.4)
                  for i in range(20)]
    candidate = paper("partial", v19_atomic_min=.1, v19_atomic_geo=.9, v19_relation_pct=.9,
                      v17_graph_pct=.9, v17_trace_pct=.9, v17_cross_pct=.9, v18_conjunction=.9)
    routed, actions = route_atomic_bottleneck_top20(
        incumbents, [candidate], score_floor=.4, margin=0, minimum_delta=0,
        dominance_required=4,
    )
    assert not actions
    assert all(row["title"] != "partial" for row in routed)


def test_router_promotes_dominant_candidate_and_protects_head():
    incumbents = [paper(str(i), v19_atomic_min=.2, v19_atomic_geo=.2, v19_relation_pct=.2,
                        v17_graph_pct=.2, v17_trace_pct=.2, v17_cross_pct=.2, v18_conjunction=.2)
                  for i in range(20)]
    candidate = paper("dominant", v19_atomic_min=.9, v19_atomic_geo=.9, v19_relation_pct=.9,
                      v17_graph_pct=.9, v17_trace_pct=.9, v17_cross_pct=.9, v18_conjunction=.9)
    routed, actions = route_atomic_bottleneck_top20(
        incumbents, [candidate], score_floor=.5, margin=.1, minimum_delta=0,
        dominance_required=5,
    )
    assert len(actions) == 1
    assert [row["title"] for row in routed[:15]] == [str(i) for i in range(15)]
    assert any(row["title"] == "dominant" for row in routed)


def test_screen_pool_is_bounded_and_multiview():
    rows = [paper(str(i), v18_conjunction=i / 20, v17_graph_pct=(20-i)/20,
                  v17_trace_pct=(i % 3)/3, v17_cross_pct=(i % 5)/5,
                  v18_relation=.5, v18_direction=.5, v18_seed_bridge=.5)
            for i in range(20)]
    selected = select_atomic_screen_pool(rows, limit=8)
    assert len(selected) == 8
    assert len({row["title"] for row in selected}) == 8
