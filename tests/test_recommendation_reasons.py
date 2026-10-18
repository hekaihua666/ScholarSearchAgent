"""推荐理由回归：必须依据各篇元数据，旧缓存修复不能改变检索结果。"""

import asyncio
import copy

from backend import llm_client
from backend.llm_mock import make_recommendation_reason, refresh_recommendation_reasons


QUERY = "RAG 科研问答论文"
SUB_QUERIES = ["RAG research question answering scientific literature"]


def test_reason_uses_actual_abstract_after_search_text_was_removed():
    """复现线上路径：没有 search_text，仍能给出该论文的具体方法。"""
    paper = {
        "title": "CG-RAG: Research Question Answering",
        "abstract": "Research question answering requires scientific literature. "
                    "In this paper, we introduce CG-RAG with sparse and dense retrieval on citation graphs.",
    }
    reason = make_recommendation_reason(QUERY, SUB_QUERIES, paper)
    assert "sparse and dense retrieval on citation graphs" in reason
    assert "综合排序" not in reason
    assert "摘要提到" in reason


def test_different_papers_get_different_verifiable_evidence():
    """即使查询相同，也引用各自的工作描述，而非随机换一种套话。"""
    first = {"title": "RAG for nuclear physics", "abstract": "In this work, we develop a local RAG system for the Electron-Ion Collider."}
    second = {"title": "RAG over citation graphs", "abstract": "We propose graph retrieval for research question answering."}
    reasons = [make_recommendation_reason(QUERY, SUB_QUERIES, p) for p in (first, second)]
    assert reasons[0] != reasons[1]
    assert "Electron-Ion Collider" in reasons[0]
    assert "graph retrieval" in reasons[1]


def test_missing_abstract_does_not_claim_satisfaction():
    reason = make_recommendation_reason(QUERY, SUB_QUERIES, {"title": "RAG for science"})
    assert "暂无摘要" in reason
    assert "尚不能确认" in reason
    assert "RAG for science" in reason


def test_substrings_do_not_fake_a_query_match():
    # RAG 不应因为出现在 fragmentation 内部而被判定为词面命中。
    paper = {"title": "Rock fragmentation", "abstract": "We study mineral fractures under pressure."}
    reason = make_recommendation_reason("RAG", [], paper)
    assert "具体关联待核对" in reason
    assert "匹配「rag」" not in reason


def test_cache_refresh_preserves_scores_order_and_specific_model_reason():
    """同篇跨分组出现不算重复；更新缓存不能悄悄改分数或删除候选。"""
    specific = "通过引文图融合稀疏与稠密检索，为科研问答提供文献证据。"
    first = {"title": "CG-RAG", "year": 2025, "final_score": 0.81, "recommendation_reason": specific}
    second = {"title": "Local RAG", "year": 2024, "final_score": 0.7,
              "abstract": "We develop local RAG for nuclear physics.",
              "recommendation_reason": "与原始问题「RAG」的研究主题相关，并在综合排序中表现较好。"}
    payload = {"query": QUERY, "sub_queries": SUB_QUERIES, "papers": [copy.deepcopy(first)],
               "ranked_candidates": [copy.deepcopy(first), second], "metrics": {"api_calls": 9}}
    before = copy.deepcopy(payload)
    refresh_recommendation_reasons(payload)
    assert payload["papers"][0]["recommendation_reason"] == specific
    assert payload["ranked_candidates"][0]["recommendation_reason"] == specific
    assert "nuclear physics" in payload["ranked_candidates"][1]["recommendation_reason"]
    payload["ranked_candidates"][1]["recommendation_reason"] = before["ranked_candidates"][1]["recommendation_reason"]
    assert payload == before


def test_duplicate_model_reason_is_replaced_with_each_papers_evidence():
    papers = [
        {"title": "RAG graph retrieval", "abstract": "We propose citation graph retrieval for RAG.", "recommendation_reason": "该论文对研究很有帮助。"},
        {"title": "Local RAG", "abstract": "We develop a local RAG system for nuclear physics.", "recommendation_reason": "该论文对研究很有帮助。"},
    ]
    payload = {"query": QUERY, "sub_queries": SUB_QUERIES, "ranked_candidates": papers}
    refresh_recommendation_reasons(payload)
    assert papers[0]["recommendation_reason"] != papers[1]["recommendation_reason"]
    first_pass = copy.deepcopy(payload)
    refresh_recommendation_reasons(payload)
    assert payload == first_pass


def test_model_receives_method_beyond_old_100_character_cutoff(monkeypatch):
    """不增加模型调用次数，只补足同一批请求的证据和约束。"""
    captured = []

    async def chat(messages, **kwargs):
        captured.append(messages[0]["content"])
        return '{"overall_summary":"测试摘要","recommendation_reasons":{"1":"引文图检索用于科研问答。"}}', 50

    monkeypatch.setattr(llm_client, "_available", lambda: True)
    monkeypatch.setattr(llm_client, "_chat", chat)
    abstract = "Background on scientific literature. " * 4 + "We introduce citation graph retrieval."
    result, _ = asyncio.run(llm_client.llm_build_summary(QUERY, [{"title": "CG-RAG", "abstract": abstract}]))
    assert len(captured) == 1
    assert "We introduce citation graph retrieval." in captured[0]
    assert "Do not invent contributions" in captured[0]
    assert result["recommendation_reasons"]["1"]


def test_cached_search_repairs_reasons_without_external_calls(monkeypatch):
    """实际接口命中旧缓存时立即修复，不靠清缓存再联网搜索。"""
    from backend import main

    paper = {"title": "Local RAG", "abstract": "We develop local RAG for nuclear physics.",
             "final_score": 0.7, "recommendation_reason": "与原始问题「RAG」的研究主题相关，并在综合排序中表现较好。"}
    payload = {"query": QUERY, "query_understanding": {}, "sub_queries": SUB_QUERIES,
               "papers": [paper], "selected_results": [paper], "ranked_candidates": [paper],
               "highly_relevant": [paper], "partially_relevant": [], "summary": {}, "search_process": {}, "metrics": {}}
    monkeypatch.delenv("SCHOLAR_FROZEN_DEMO", raising=False)
    monkeypatch.setattr(main, "current_session", lambda: None)
    monkeypatch.setattr(main.response_cache, "load", lambda key: copy.deepcopy(payload))

    async def unexpected_call(*args, **kwargs):
        raise AssertionError("缓存命中不应发起模型或检索调用")

    monkeypatch.setattr(main, "llm_understand_query", unexpected_call)
    monkeypatch.setattr(main, "llm_build_summary", unexpected_call)
    response = asyncio.run(main.search(main.SearchRequest(query=QUERY)))
    assert response.metrics["from_cache"] is True
    assert "nuclear physics" in response.papers[0]["recommendation_reason"]
    assert response.papers[0]["final_score"] == 0.7
