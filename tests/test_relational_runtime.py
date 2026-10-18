from __future__ import annotations

import asyncio

import backend.selection.relational_runtime as runtime
from backend.selection.relational_runtime import (
    contract_from_understanding,
    expand_and_route_relational,
    frozen_config,
    route_runtime_frontier,
)


class FakeSelector:
    def predict(self, pairs, batch_size=64):
        del batch_size
        values = [2.0 if "target" in document.lower() else -2.0 for _, document in pairs]
        return values, {"runtime": "fake", "cache_hits": 0, "pairs": len(pairs)}


def base_paper(title: str, score: float = .2, **extra):
    return {
        "title": title, "abstract": extra.pop("abstract", ""),
        "final_score": score, "constraint_coverage_score": score,
        "citation_direction": extra.pop("citation_direction", "candidate_cites_seed"),
        "citation_lane": extra.pop("citation_lane", "reverse_group_1"),
        "citation_seed_count": extra.pop("citation_seed_count", 1),
        **extra,
    }


def test_contract_conversion_preserves_relation_and_conditions():
    contract = contract_from_understanding({"constraint_graph": {"criteria": [
        {"criterion_id": "c1", "type": "topic", "description": "graph learning"},
        {"criterion_id": "c2", "type": "method", "description": "Graph-Learning"},
        {"criterion_id": "r1", "type": "relation", "description": "extends"},
    ]}})
    assert contract["conditions"][0]["description"] == "graph learning"
    assert len(contract["conditions"]) == 1
    assert contract["relation"] == "extends"

    chinese = contract_from_understanding({"constraint_graph": {"criteria": [
        {"criterion_id": "c1", "type": "topic", "description": "图神经网络"},
    ]}})
    assert chinese["conditions"][0]["description"] == "图神经网络"


def test_contract_supplements_missing_facets_and_infers_relation():
    contract = contract_from_understanding({
        "original_query": "Which papers theoretically analyze MAML for few-shot learning?",
        "method_constraints": "MAML",
        "task_constraints": "few-shot learning",
        "constraint_graph": {"criteria": [
            {"criterion_id": "c1", "type": "method", "description": "MAML"},
        ]},
    })
    assert [condition["description"] for condition in contract["conditions"]] == [
        "MAML", "few-shot learning",
    ]
    assert contract["relation"] == "theoretically analyzes"


def test_runtime_route_attaches_explainable_relational_evidence():
    incumbents = [base_paper(f"incumbent {i}", .1) for i in range(20)]
    candidate = base_paper(
        "target extension", .9, abstract="target graph learning extends prior method",
        citation_seed_count=5,
    )
    routed, audit = route_runtime_frontier(
        "papers extending graph learning", {
            "conditions": [{"id": "c1", "description": "graph learning", "aliases": []}],
            "relation": "extends",
        }, incumbents, [candidate], FakeSelector(),
    )
    assert audit["strategy"] == "REG_MEQ_v20_relational_evidence_quorum"
    assert audit["config"]["source_commit"]
    assert len(audit["actions"]) == 1
    assert audit["actions"][0]["incoming_title"] == "target extension"
    promoted = next(paper for paper in routed if paper["title"] == "target extension")
    assert promoted["relational_evidence"]["promoted_from_frontier"] is True
    assert promoted["relational_evidence"]["weakest_condition"] == 1.0
    assert [paper["title"] for paper in routed[:15]] == [f"incumbent {i}" for i in range(15)]


def test_runtime_uses_the_frozen_v18_routing_contract():
    config = frozen_config()
    assert config["score_floor"] == 0.62
    assert config["margin"] == 0.10
    assert config["max_promotions"] == 1
    assert config["protect_head"] == 15
    assert config["quorum_floor"] == 0.97
    assert config["quorum_protect_head"] == 18
    assert config["required_views"] == 3
    assert set(config["weights"]) >= {"default", "taxonomy", "survey", "usage"}


def test_async_runtime_merges_both_frontiers_and_accounts_for_calls(monkeypatch):
    target = base_paper(
        "target extension", .9,
        abstract="target graph learning extends prior method",
        citation_seed_count=5,
    )

    async def fake_reverse(*args, **kwargs):
        del args, kwargs
        return {"candidates": [target], "api_calls": 2, "errors": []}

    async def fake_qcf(*args, **kwargs):
        del args, kwargs
        return {"candidates": [target], "api_calls": 3, "errors": []}

    monkeypatch.setattr(runtime, "retrieve_seed_conditioned_reverse_frontier", fake_reverse)
    monkeypatch.setattr(runtime, "retrieve_query_conditioned_frontier", fake_qcf)
    incumbents = [base_paper(f"incumbent {i}", .1) for i in range(20)]
    routed, audit = asyncio.run(expand_and_route_relational(
        "papers extending graph learning",
        {"constraint_graph": {"criteria": [
            {"criterion_id": "c1", "type": "topic", "description": "graph learning"},
            {"criterion_id": "r1", "type": "relation", "description": "extends"},
        ]}},
        incumbents,
        incumbents,
        selector=FakeSelector(),
    ))
    assert audit["enabled"] is True
    assert audit["api_calls"] == 5
    assert audit["reverse_candidates"] == 1
    assert audit["qcf_candidates"] == 1
    assert audit["ebp_candidates"] == 0
    assert audit["contract"]["relation"] == "extends"
    assert routed[-1]["title"] == "target extension"


def test_async_runtime_exposes_ebp_as_supplement_without_changing_top20(monkeypatch):
    target = base_paper(
        "target extension", .9,
        abstract="target graph learning extends prior method",
        citation_seed_count=5,
    )
    supplement = base_paper(
        "backprojected evidence", .4,
        openalex_id="W99", citation_backprojection_rank=7,
    )

    async def fake_reverse(*args, **kwargs):
        del args, kwargs
        return {"candidates": [target], "api_calls": 2, "errors": []}

    async def fake_qcf(*args, **kwargs):
        del args, kwargs
        return {"candidates": [target], "api_calls": 3, "errors": []}

    async def fake_ebp(*args, **kwargs):
        del args, kwargs
        return {
            "targets": [{"openalex_id": "W99"}],
            "candidates": [supplement],
            "api_calls": 1,
            "errors": [],
        }

    monkeypatch.setattr(runtime, "retrieve_seed_conditioned_reverse_frontier", fake_reverse)
    monkeypatch.setattr(runtime, "retrieve_query_conditioned_frontier", fake_qcf)
    monkeypatch.setattr(runtime, "retrieve_evidence_backprojection", fake_ebp)
    incumbents = [base_paper(f"incumbent {i}", .1) for i in range(20)]
    routed, audit = asyncio.run(expand_and_route_relational(
        "papers extending graph learning",
        {"constraint_graph": {"criteria": [
            {"criterion_id": "c1", "type": "topic", "description": "graph learning"},
            {"criterion_id": "r1", "type": "relation", "description": "extends"},
        ]}},
        incumbents,
        incumbents,
        selector=FakeSelector(),
    ))
    assert audit["api_calls"] == 6
    assert audit["ebp_targets"] == 1
    assert audit["ebp_candidates"] == 1
    assert audit["supplementary_candidates"][0]["title"] == "backprojected evidence"
    assert all(paper["title"] != "backprojected evidence" for paper in routed)
