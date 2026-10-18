from backend.research_intelligence import (
    attach_evidence_and_roles, build_evidence_matrix, build_research_landscape,
    infer_search_intent,
)


def test_intent_router_handles_foundation_and_frontier():
    assert infer_search_intent("Which work first introduced this method?") == "foundational"
    assert infer_search_intent("最新的多模态检索研究有哪些") == "frontier"


def test_evidence_is_exact_abstract_span_and_landscape_is_bounded():
    papers = [{
        "title": "Graph retrieval for science",
        "abstract": "We introduce graph retrieval. It improves scientific paper search recall.",
        "year": 2025,
        "authority_score": 0.4,
        "constraint_coverage_score": 0.8,
        "constraint_coverage": {"research_topic": 1.0},
        "retrieval_hits": [{"source": "OpenAlex", "rank": 1}],
        "source_api": "OpenAlex",
        "final_score": 0.8,
    }]
    understanding = {"original_query": "graph scientific paper retrieval", "research_topic": "graph retrieval"}
    attach_evidence_and_roles(papers, understanding)
    snippet = papers[0]["evidence_snippet"]
    assert snippet in papers[0]["abstract"]
    assert papers[0]["evidence_span"]["start"] >= 0
    assert build_evidence_matrix(papers, understanding)["rows"]
    assert build_research_landscape(papers)["role_counts"]
