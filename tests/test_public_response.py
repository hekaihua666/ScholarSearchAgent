"""网页响应只包含用户所需信息，后台计算与完整评测结果不受影响。"""

import copy

from backend.public_response import PublicPaper, build_public_response


def test_allowlist_removes_all_internal_fields_and_preserves_backend_payload():
    payload = {
        "query": "RAG 论文", "search_process": {"queries": ["internal query"]},
        "metrics": {"llm_token_count": 100}, "evidence_matrix": {"internal": True},
        "ranked_candidates": [{
            "title": "Paper", "abstract": "Actual abstract.", "authors": ["Author"], "year": 2024,
            "final_score": 0.7, "llm_relevant": True, "recommendation_reason": "Internal reason",
            "relational_evidence": {"score": 0.8}, "evidence_snippet": "Internal evidence",
            "future_internal_field": "must never reach the browser",
        }],
    }
    original = copy.deepcopy(payload)
    public = build_public_response(payload).model_dump()
    assert set(public) == {"query", "papers", "notices"}
    assert set(public["papers"][0]) == set(PublicPaper.model_fields)
    assert public["papers"][0]["abstract"] == "Actual abstract."
    assert "future_internal_field" not in str(public)
    assert payload == original


def test_same_deduplication_and_score_order_as_original_browser():
    payload = {"query": "论文", "ranked_candidates": [
        {"title": "B", "year": 2024, "final_score": 0.4},
        {"title": "A", "year": 2023, "final_score": 0.9},
        {"title": "B", "year": 2024, "final_score": 1.0},
        {"title": "C", "year": 2023, "final_score": 0.9},
    ]}
    public = build_public_response(payload)
    assert [p.title for p in public.papers] == ["A", "C", "B"]
    assert [p.rank for p in public.papers] == [1, 2, 3]


def test_all_candidates_are_returned_without_top20_cap():
    payload = {"query": "论文", "ranked_candidates": [
        {"title": f"Paper {i}", "final_score": i / 50} for i in range(35)
    ]}
    public = build_public_response(payload)
    assert len(public.papers) == 35
    assert public.papers[0].title == "Paper 34"
    assert public.papers[-1].rank == 35


def test_null_optional_metadata_still_allows_opening_papers_without_abstracts():
    public = build_public_response({"query": "论文", "papers": [{
        "title": "Missing abstract", "abstract": None, "authors": None,
        "cited_by_count": None, "url": "https://example.org/paper",
    }]})
    paper = public.papers[0]
    assert paper.abstract == ""
    assert paper.authors == []
    assert paper.url == "https://example.org/paper"


def test_only_safe_user_notices_leave_backend():
    public = build_public_response({"query": "论文", "papers": [], "search_process": {
        "fallback_queries": ["internal query"],
        "relational_evidence_graph": {"error": "private debug stack"},
        "expansion_mode": "rule-based",
    }})
    assert len(public.notices) == 2
    assert "private debug stack" not in str(public.model_dump())
    assert "internal query" not in str(public.model_dump())
    assert "规则模式" not in str(public.model_dump())


def test_http_endpoint_uses_public_contract_while_internal_search_keeps_full_result(monkeypatch):
    """从实际HTTP序列化边界验证白名单，避免只是页面隐藏。"""
    from fastapi.testclient import TestClient
    from backend import main

    internal = main.SearchResponse(
        query="RAG 论文", query_understanding={"method": "RAG"}, sub_queries=["RAG research"],
        papers=[], highly_relevant=[], partially_relevant=[], summary={"internal": True},
        search_process={"internal": True}, metrics={"internal": True},
        ranked_candidates=[{"title": "Paper", "abstract": "Abstract", "final_score": 0.9,
                            "research_role": "frontier", "llm_relevant": True}],
    )

    async def fake_search(request):
        return internal

    monkeypatch.setattr(main, "search", fake_search)
    with TestClient(main.app) as client:
        response = client.post("/api/search", json={"query": "RAG 论文"})
    assert response.status_code == 200
    assert set(response.json()) == {"query", "papers", "notices"}
    assert "final_score" not in response.text
    assert "research_role" not in response.text
    assert internal.ranked_candidates[0]["final_score"] == 0.9
    assert internal.metrics == {"internal": True}
