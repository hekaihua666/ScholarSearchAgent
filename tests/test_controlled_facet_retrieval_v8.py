from __future__ import annotations

import asyncio
from datetime import datetime

from backend.eval_runtime import EvaluationSession, evaluation_session, filter_for_cutoff, frozen_call
from backend.retrieval.controlled_facet_retrieval_v8 import (
    MAX_VARIANTS,
    admission_decision,
    api_budget,
    build_facet_queries,
    core_protected_merge,
    facet_inventory,
    gap_triggered,
)


def understanding():
    return {
        "research_topic":"cost-effective counterfactual augmentation",
        "method_constraints":"large language models, LLMs",
        "task_constraints":"Counterfactually Augmented Data generation",
        "domain_constraints":"natural language processing",
        "original_query":"What works use large language models (LLMs) for Counterfactually Augmented Data (CAD)?",
        "constraint_graph":{
            "criteria":[
                {"description":"counterfactual augmentation","type":"topic"},
                {"description":"large language models","type":"method"},
                {"description":"Counterfactually Augmented Data","type":"task"},
            ],
            "key_entities":["CAD"],
        },
    }


def paper(title: str, abstract: str = "", identifier: str | None = None):
    return {
        "title":title,"abstract":abstract,"openalex_id":identifier or f"https://openalex.org/W{abs(hash(title))%100000}",
        "year":2023,"publication_date":"2023-01-01","source_api":"OpenAlex","final_score":.5,
    }


def test_facet_generation_is_bounded_and_structured():
    query=understanding()["original_query"]
    facets=build_facet_queries(query,understanding(),policy=("T1","T2","T3","T4"),max_variants=9)
    assert 1<=len(facets)<=MAX_VARIANTS
    assert all(2<=len(item.phrases)<=5 for item in facets)
    assert all(item.query!=query for item in facets)


def test_named_entities_must_come_from_original_query():
    data=understanding();data["constraint_graph"]["key_entities"]=["CAD","InventedEntity"]
    inventory=facet_inventory(data["original_query"],data)
    assert "CAD" in inventory["entity"]
    assert "InventedEntity" not in inventory["entity"]


def test_facet_schema_has_no_gold_inputs_or_query_specific_patch():
    facets=build_facet_queries(understanding()["original_query"],understanding())
    serialized=" ".join(str(item.to_dict()) for item in facets).lower()
    assert "gold" not in serialized
    assert "autoscholarquery" not in serialized
    assert "doi" not in serialized


def test_api_budget_never_exceeds_four():
    facets=build_facet_queries(understanding()["original_query"],understanding(),max_variants=2)
    assert api_budget(facets,("arxiv","openalex"))<=4


def test_cutoff_filter_excludes_future_and_missing_date(tmp_path):
    session=EvaluationSession.create("v8","q",datetime(2024,1,1),"live",tmp_path/"snapshot.json")
    rows=[paper("legal"),{**paper("future"),"publication_date":"2025-01-01"},{"title":"missing"}]
    with evaluation_session(session):kept=filter_for_cutoff(rows,"dsfr_test")
    assert [row["title"] for row in kept]==["legal"]


def test_admission_separates_a0_and_dual_source_a1():
    inventory=facet_inventory(understanding()["original_query"],understanding())
    candidate=paper("LLM Counterfactual Augmentation","Large language models generate Counterfactually Augmented Data")
    candidate.update({"topic_coverage":1.0,"exact_entity_match":False,"source_count":1,"variant_count":1,"existing_trace_score":.2,"title_coverage":.5})
    assert admission_decision(candidate,inventory,mode="A0",trace_threshold=.6)[0] is True
    assert admission_decision(candidate,inventory,mode="A1",trace_threshold=.6)[0] is False
    candidate["source_count"]=2
    assert admission_decision(candidate,inventory,mode="A1",trace_threshold=.6)[0] is True


def test_identity_duplicate_is_rejected():
    inventory=facet_inventory(understanding()["original_query"],understanding())
    candidate=paper("Same paper",identifier="https://openalex.org/W123")
    candidate.update({"topic_coverage":1.0,"exact_entity_match":True,"title_coverage":1.0})
    baseline=[paper("Same paper",identifier="https://openalex.org/W123")]
    assert admission_decision(candidate,inventory,mode="A0",trace_threshold=.4,baseline=baseline)==(False,"duplicate")


def test_core_protected_merge_keeps_top2_and_caps_top20():
    baseline=[paper(f"baseline {index}",abstract="counterfactual augmentation large language models") for index in range(20)]
    for index,row in enumerate(baseline,1):row["candidate_rank"]=index
    added=[paper(f"new {index}",abstract="counterfactual augmentation large language models") for index in range(8)]
    merged=core_protected_merge(baseline,added,understanding())
    assert [row["title"] for row in merged[:2]]==["baseline 0","baseline 1"]
    assert len(merged)==20
    assert [row["candidate_rank"] for row in merged]==list(range(1,21))


def test_gap_trigger_uses_only_production_signals():
    baseline=[paper(f"candidate {index}") for index in range(4)]
    assert isinstance(gap_triggered(baseline,understanding()["original_query"],understanding()),bool)


def test_frozen_call_replay_is_identical(tmp_path):
    path=tmp_path/"snapshot.json";cutoff=datetime(2024,1,1)
    live=EvaluationSession.create("v8","q",cutoff,"live",path)
    async def run_live():
        with evaluation_session(live):return await frozen_call("dsfr",{"q":"x"},lambda:asyncio.sleep(0,result=[{"title":"p"}]))
    live_value=asyncio.run(run_live());live.save()
    replay=EvaluationSession.create("v8","q",cutoff,"replay",path)
    async def run_replay():
        with evaluation_session(replay):return await frozen_call("dsfr",{"q":"x"},lambda:asyncio.sleep(0,result=[]))
    assert asyncio.run(run_replay())==live_value
