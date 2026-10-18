from __future__ import annotations

import json

from backend.selection.evidence_contract_ranker_v9 import (
    MAX_CARDS, build_contract_cards, build_contract_prompt, contract_score,
    infer_contract_mode, parse_contract_response, route_contract_core, route_core,
    score_cards, select_newcomers,
)


def paper(title: str, *, score: float = .4, facet: float = .5, source_rank: int = 5):
    return {"title":title,"abstract":f"Abstract evidence about {title}","final_score":score,
            "facet_coverage":facet,"topic_coverage":facet,"abstract_coverage":facet,
            "constraint_coverage_score":facet,"rrf_score":.2,"openalex_rank":source_rank,
            "openalex_id":f"https://openalex.org/{title.lower().replace(' ','-')}"}


def parsed_for(cards, newcomer_score=95, baseline_score=50):
    assessments={}
    ranking=[]
    for card in cards:
        score=newcomer_score if card.origin=="newcomer" else baseline_score
        assessments[card.card_id]={"point_score":score,"list_score":score,"verdict":"relevant",
                                   "condition_support":{"C1":3,"C2":3},"relation_support":3,
                                   "contradiction":False,"evidence":"supported"}
        ranking.append(card.card_id)
    return {"contract":{"conditions":[{"id":"C1"},{"id":"C2"}]},"ranking":ranking,"assessments":assessments}


def test_shortlist_is_bounded_source_diverse_and_opaque():
    baseline=[paper(f"base paper {chr(97+i)}") for i in range(20)]
    newcomers=[paper(f"new study {chr(97+i%26)}{chr(97+i//26)}",score=i/30,facet=(20-i)/20,source_rank=i+1) for i in range(30)]
    cards=build_contract_cards("complex query",baseline,newcomers)
    assert len(cards)==MAX_CARDS
    assert sum(card.origin=="baseline" for card in cards)==4
    assert sum(card.origin=="newcomer" for card in cards)==12
    assert all(card.card_id.startswith("P") for card in cards)
    assert all("baseline" not in card.prompt_payload() and "newcomer" not in card.prompt_payload() for card in cards)


def test_newcomer_selection_deduplicates_against_baseline():
    baseline=[paper("same")]
    newcomers=[paper("same"),paper("different")]
    selected=select_newcomers(newcomers,baseline)
    assert [row["title"] for row in selected]==["different"]


def test_prompt_has_contract_and_no_benchmark_fields():
    cards=build_contract_cards("topic method relation",[paper("base 1"),paper("base 2")],[paper("new")])
    prompt=build_contract_prompt("topic method relation",cards)
    lowered=prompt.lower()
    assert "query contract" in lowered and "condition_support" in lowered
    assert "query_id" not in lowered and "expected_cardinality" not in lowered and "matched_gold" not in lowered


def test_parse_contract_response_accepts_fences_and_normalises():
    text='```json\n'+json.dumps({"contract":{},"ranking":["P02","P01"],"assessments":{
        "P01":{"score":80,"verdict":"relevant","condition_support":[3,2],"relation_support":3},
        "P02":{"point_score":40,"list_score":50,"verdict":"partial","condition_support":{"C1":1},"relation_support":1}}})+'\n```'
    parsed=parse_contract_response(text,["P01","P02"])
    assert parsed["ranking"]==["P02","P01"]
    assert parsed["assessments"]["P01"]["point_score"]==80
    assert parsed["assessments"]["P01"]["condition_support"]["C2"]==2


def test_contract_score_requires_complete_bottleneck():
    strong={"point_score":90,"list_score":90,"verdict":"relevant","condition_support":{"C1":3,"C2":2},"relation_support":3}
    incomplete={**strong,"condition_support":{"C1":3,"C2":1}}
    strong_score,strong_complete=contract_score(strong,1,10)
    weak_score,weak_complete=contract_score(incomplete,1,10)
    assert strong_complete and not weak_complete and strong_score>weak_score


def test_router_replaces_only_when_threshold_and_margin_clear():
    baseline=[paper("base 1"),paper("base 2"),*([paper("tail")]*18)]
    newcomer=paper("newcomer")
    cards=build_contract_cards("complex query",baseline,[newcomer])
    scored=score_cards(cards,parsed_for(cards,newcomer_score=95,baseline_score=50))
    selected,actions=route_core(baseline,scored,threshold=90,margin=10,max_replacements=1)
    assert len(actions)==1 and any(row["title"]=="newcomer" for row in selected)
    selected,actions=route_core(baseline,scored,threshold=99,margin=10,max_replacements=1)
    assert not actions and [row["title"] for row in selected]==["base 1","base 2"]


def test_router_respects_replacement_cap():
    baseline=[paper("base 1"),paper("base 2")]
    cards=build_contract_cards("complex query",baseline,[paper("new 1"),paper("new 2")])
    scored=score_cards(cards,parsed_for(cards))
    _,actions=route_core(baseline,scored,threshold=80,margin=5,max_replacements=1)
    assert len(actions)==1


def test_distributive_contract_routes_condition_specialists_as_a_set():
    baseline=[paper("unrelated incumbent alpha"),paper("unrelated incumbent beta")]
    newcomers=[paper("origin method alpha"),paper("origin method beta")]
    cards=build_contract_cards("What papers introduced method alpha and method beta?",baseline,newcomers)
    assessments={};ranking=[]
    for card in cards:
        ranking.append(card.card_id)
        if card.origin=="baseline":support={"C1":0,"C2":0};score=5;verdict="irrelevant"
        elif "alpha" in card.paper["title"]:support={"C1":3,"C2":0};score=100;verdict="relevant"
        else:support={"C1":0,"C2":3};score=100;verdict="relevant"
        assessments[card.card_id]={"point_score":score,"list_score":score,"verdict":verdict,"condition_support":support,"relation_support":3,"contradiction":False}
    parsed={"contract":{"relation":"introduced","conditions":[
        {"id":"C1","description":"Paper introduces method alpha"},{"id":"C2","description":"Paper introduces method beta"}]},
        "ranking":ranking,"assessments":assessments}
    assert infer_contract_mode("What papers introduced method alpha and method beta?",parsed)=="set_distributive"
    scored=score_cards(cards,parsed)
    selected,actions,mode=route_contract_core("What papers introduced method alpha and method beta?",baseline,scored,parsed,threshold=90,margin=5,max_replacements=2)
    assert mode=="set_distributive" and len(actions)==2
    assert {row["title"] for row in selected}=={"origin method alpha","origin method beta"}
