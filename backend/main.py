"""研问智搜 FastAPI 后端入口。"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import httpx

# Load the local reranker runtime before sklearn/OpenMP-backed selectors on
# Windows; otherwise torch's c10.dll can fail to initialise in long-lived API
# processes. The search endpoint still degrades gracefully if it is absent.
try:
    from sentence_transformers import CrossEncoder as _CrossEncoderBootstrap  # noqa: F401
except Exception:
    _CrossEncoderBootstrap = None  # type: ignore[assignment]

# In-process cache: caches real API responses for 10 minutes.
# Prevents re-fetching identical queries during demos or repeated test runs.
_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_CACHE_TTL = 600
_ARXIV_LOCK = asyncio.Lock()
_ARXIV_LAST_CALL = 0.0


def _cache_get(key: str) -> list[dict[str, Any]] | None:
    entry = _CACHE.get(key)
    if entry and time.time() - entry[0] < _CACHE_TTL:
        return entry[1]
    return None


def _cache_set(key: str, value: list[dict[str, Any]]) -> None:
    _CACHE[key] = (time.time(), value)


async def _safe_arxiv_search(query: str, per_page: int, time_range: str | None = None) -> list[dict[str, Any]]:
    """Serialize arXiv calls and space them to avoid burst throttling."""
    global _ARXIV_LAST_CALL
    async with _ARXIV_LOCK:
        wait_for = 1.2 - (time.monotonic() - _ARXIV_LAST_CALL)
        if wait_for > 0:
            await asyncio.sleep(wait_for)
        try:
            return await search_arxiv(query, per_page, time_range)
        finally:
            _ARXIV_LAST_CALL = time.monotonic()
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend import response_cache
from backend.agent_retrieval import budget_for_mode, retrieve_with_agent
from backend.agent_tools import AgentToolbox
from backend.audit_trace import record_event, record_stage
from backend.eval_runtime import (
    current_cutoff_iso, current_session, filter_for_cutoff, frozen_call,
)
from backend.adaptive_controller import assess_retrieval
from backend.helios_lite import HELIOSLiteController, admit_candidates, build_state
from backend.arxiv_client import search_arxiv
from backend.crossref_client import search_crossref
from backend.llm_client import agent_model_name, llm_build_summary, llm_expand_queries, llm_understand_query
from backend.llm_mock import build_summary, make_recommendation_reason, refresh_recommendation_reasons
from backend.openalex_client import (
    OpenAlexRateLimitError, fetch_openalex_citing, fetch_openalex_works, search_openalex,
    search_openalex_semantic,
)
from backend.query_understanding import ACADEMIC_ERROR_MESSAGE, is_academic_query, understand_query
from backend.public_response import PublicSearchResponse, build_public_response
from backend.query_portfolio import atomicize_query, build_query_portfolio
from backend.ranker import split_relevance
from backend.retrieval_frontier import choose_dual_frontier
from backend.selection.joint_selector import select_with_champion
from backend.selection.relational_runtime import expand_and_route_relational
from backend.research_intelligence import (
    INTENT_LABELS, attach_evidence_and_roles, build_evidence_matrix,
    build_research_landscape, infer_search_intent,
)
from backend.semantic_scholar_client import search_semantic_scholar
from backend.trace_rank import build_relation_graph, canonical_key, diversify, fuse_candidates, score_candidates


BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="研问智搜 API", version="1.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")

_EXPAND_STOPWORDS = {
    "and", "are", "for", "from", "how", "into", "the", "this", "that",
    "with", "what", "which", "have", "has", "been", "our", "also", "its",
    "article", "paper", "study", "result", "method", "using", "based",
    "analysis", "research", "approach", "proposed", "show", "shows",
    "between", "each", "their", "can", "two", "new", "used", "data",
    "model", "models", "these", "such", "then", "than", "when", "present",
    "first", "however", "although", "both", "within", "while", "here",
    "where", "more", "most", "well", "work", "works", "task", "tasks",
}


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="复杂学术查询")
    search_strategy: Literal["workflow", "agent"] = Field(
        default="workflow", description="网页使用 agent 原生工具循环；workflow 保留旧评测与兼容调用"
    )
    mode: Literal["quick", "efficient", "helios", "balanced", "deep"] = Field(
        default="balanced", description="检索模式：快速、低成本、HELIOS、自适应均衡或深度"
    )
    selection_profile: Literal["f1", "balanced_set", "exploratory"] = Field(
        default="f1", description="最终集合策略：证据校准核心结果、均衡集合或探索性Top20"
    )
    retrieval_profile: Literal["v3", "portfolio", "semantic_portfolio", "citation", "joint", "lean", "relational"] = Field(
        default="relational", description="旧 workflow 路径的检索配置；agent 路径由模型自主选工具"
    )


class SearchResponse(BaseModel):
    query: str
    query_understanding: dict[str, Any]
    sub_queries: list[str]
    papers: list[dict[str, Any]]
    selected_results: list[dict[str, Any]] = Field(default_factory=list)
    supplementary_candidates: list[dict[str, Any]] = Field(default_factory=list)
    ranked_candidates: list[dict[str, Any]] = Field(default_factory=list)
    selection: dict[str, Any] = Field(default_factory=dict)
    highly_relevant: list[dict[str, Any]]
    partially_relevant: list[dict[str, Any]]
    summary: dict[str, Any]
    search_process: dict[str, Any]
    metrics: dict[str, Any]
    relation_graph: dict[str, Any] = Field(default_factory=dict)
    evidence_matrix: dict[str, Any] = Field(default_factory=dict)
    research_landscape: dict[str, Any] = Field(default_factory=dict)


@app.get("/")
async def index() -> FileResponse:
    # 首页内嵌交互脚本，整改期间每次导航重新获取，避免浏览器沿用旧版界面。
    return FileResponse(FRONTEND_DIR / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# ── Per-query search with automatic fallback ──────────────────────────────────

async def _search_one(
    query: str,
    per_page: int,
    time_range: str | None,
    semaphore: asyncio.Semaphore,
) -> tuple[list[dict[str, Any]], str]:
    """Low-latency fallback: OpenAlex → Crossref → Semantic Scholar.

    arXiv is queried independently for selected high-value queries rather than
    blocking every fallback path. Returns (papers, source_api); source_api
    is whichever source answered, or 'failed' if all sources failed.
    """
    # 1) OpenAlex (best metadata; abstracts + citation sort)
    oa_key = f"oa:{query}:{per_page}:{time_range}:{current_cutoff_iso()}"
    cached = _cache_get(oa_key)
    if cached is not None:
        return cached, "OpenAlex"
    async with semaphore:
        try:
            results = await search_openalex(query, per_page, time_range)
            _cache_set(oa_key, results)
            return results, "OpenAlex"
        except OpenAlexRateLimitError:
            pass
        except httpx.HTTPError:
            pass

    # 2) Crossref (free, no key, reliable and fast fallback)
    cr_key = f"cr:{query}:{per_page}:{time_range}:{current_cutoff_iso()}"
    cached = _cache_get(cr_key)
    if cached is not None:
        return cached, "Crossref"
    try:
        results = await search_crossref(query, per_page, time_range)
        if results:
            _cache_set(cr_key, results)
            return results, "Crossref"
    except Exception:
        pass

    # 3) Semantic Scholar (last resort)
    s2_key = f"s2:{query}:{per_page}:{current_cutoff_iso()}"
    cached = _cache_get(s2_key)
    if cached is not None:
        return cached, "Semantic Scholar"
    try:
        results = await search_semantic_scholar(query, per_page)
        if results:
            _cache_set(s2_key, results)
        return results, "Semantic Scholar"
    except Exception:
        return [], "failed"


async def _search_arxiv_supplement(query: str, per_page: int = 20) -> list[dict[str, Any]]:
    """Add an independent arXiv channel instead of using it only after failure.

    OpenAlex and arXiv have different coverage/ranking behavior. Independent
    retrieval increases recall and gives TRACE-Rank genuine cross-source hits.
    """
    key = f"ax-supplement:{query}:{per_page}:{current_cutoff_iso()}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    fallback_cached = _cache_get(f"ax:{query}:{per_page}:{current_cutoff_iso()}")
    if fallback_cached is not None:
        _cache_set(key, fallback_cached)
        return fallback_cached
    try:
        results = await _safe_arxiv_search(query, per_page)
    except Exception:
        results = []
    if results:
        _cache_set(key, results)
    return results


# ── Keyword expansion ─────────────────────────────────────────────────────────

def _extract_expanded_keywords(papers: list[dict[str, Any]], max_keywords: int = 5) -> list[str]:
    word_freq: Counter = Counter()
    for paper in papers[:5]:
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9-]{3,}", text.lower())
        for token in tokens:
            if token not in _EXPAND_STOPWORDS and len(token) >= 4:
                word_freq[token] += 1

    seen: set[str] = set()
    result: list[str] = []
    for word, _ in word_freq.most_common(max_keywords * 3):
        if word not in seen:
            seen.add(word)
            result.append(word)
        if len(result) >= max_keywords:
            break
    return result


def _build_expanded_queries(research_topic: str, keywords: list[str]) -> list[str]:
    if not keywords:
        return []
    candidates: list[str] = []
    if len(keywords) >= 2:
        candidates.append(f"{research_topic} {keywords[0]} {keywords[1]}")
    if len(keywords) >= 4:
        candidates.append(f"{research_topic} {keywords[2]} {keywords[3]}")
    if len(keywords) >= 2:
        candidates.append(f"{keywords[0]} {keywords[1]} scholarly search")
    if len(keywords) >= 3:
        candidates.append(f"{research_topic} {keywords[0]} {keywords[2]} paper")
    seen: set[str] = set()
    unique: list[str] = []
    for q in candidates:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    return unique[:2]


# ── Round merging ─────────────────────────────────────────────────────────────

def _merge_rounds(
    first_papers: list[dict[str, Any]],
    second_papers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    key_to_paper: dict[str, dict[str, Any]] = {}
    for paper in first_papers:
        key = (paper.get("doi") or paper.get("openalex_id") or paper.get("title") or "").lower().strip()
        paper["source_round"] = "first_round"
        if key:
            key_to_paper[key] = paper

    merged = list(first_papers)
    seen = set(key_to_paper.keys())

    for paper in second_papers:
        key = (paper.get("doi") or paper.get("openalex_id") or paper.get("title") or "").lower().strip()
        if not key:
            continue
        if key in key_to_paper:
            key_to_paper[key]["source_round"] = "both"
        elif key not in seen:
            seen.add(key)
            paper["source_round"] = "second_round"
            merged.append(paper)

    return merged


def _make_score_explanation(paper: dict[str, Any]) -> str:
    keyword_score  = float(paper.get("lexical_score", paper.get("keyword_score", 0)))
    openalex_score = float(paper.get("rrf_score", paper.get("openalex_score", 0)))
    citation_score = float(paper.get("authority_score", paper.get("citation_score", 0)))
    coverage_score = float(paper.get("constraint_coverage_score", 0))
    freshness_score = float(paper.get("freshness_score", 0))
    final_score    = float(paper.get("final_score", 0))
    cited_by       = int(paper.get("cited_by_count", 0))
    year           = paper.get("year", "未知")
    source_api     = paper.get("source_api", "OpenAlex")

    parts: list[str] = []
    if keyword_score >= 0.5:
        parts.append(f"与查询关键词高度匹配（关键词得分 {keyword_score:.2f}）")
    elif keyword_score > 0:
        parts.append(f"与查询关键词部分匹配（关键词得分 {keyword_score:.2f}）")
    else:
        parts.append("关键词匹配度较低")

    if openalex_score >= 0.7:
        parts.append(f"多检索路径一致命中（RRF 得分 {openalex_score:.2f}）")
    elif openalex_score > 0:
        parts.append(f"检索路径融合得分 {openalex_score:.2f}")

    if coverage_score > 0:
        parts.append(f"查询约束覆盖度 {coverage_score:.2f}")

    if cited_by >= 100:
        parts.append(f"高引用论文（引用 {cited_by} 次，引用得分 {citation_score:.2f}）")
    elif cited_by > 0:
        parts.append(f"引用 {cited_by} 次（引用得分 {citation_score:.2f}）")

    if freshness_score >= 0.7:
        parts.append(f"近期发表（{year} 年，新近性得分 {freshness_score:.2f}）")
    elif freshness_score > 0:
        parts.append(f"{year} 年发表（新近性得分 {freshness_score:.2f}）")

    parts.append(f"综合评分 {final_score:.4f}")
    return "；".join(parts) + "。"


# ── Main search endpoint ──────────────────────────────────────────────────────

def _selection_view(papers: list[dict[str, Any]], profile: str) -> dict[str, Any]:
    k = {"f1": 1, "balanced_set": 8, "exploratory": 20}.get(profile, 1)
    ranked = list(papers[:20]); selected = ranked[:k]
    selection = {
        "profile": profile,
        "strategy": "evidence_calibrated_core" if profile == "f1" else "ranked_set_view",
        "selected_count": len(selected), "supplementary_count": len(ranked) - len(selected),
        "candidate_pool_preserved": True, "protocol": "macro_set_f1",
        "evidence": "evidence-calibrated core; K=8 balanced set; Top20 retained as supplementary",
    }
    return {"papers": selected, "selected_results": selected,
            "supplementary_candidates": ranked[k:], "ranked_candidates": ranked, "selection": selection}

async def _search_with_agent(request: SearchRequest, started_at: float) -> SearchResponse:
    """把自主收集得到的真实候选接入现有评分和网页返回格式。"""
    query = request.query.strip()
    understanding = understand_query(query)
    understanding["original_query"] = query
    understanding["search_intent"] = infer_search_intent(query)
    understanding["search_intent_label"] = INTENT_LABELS[understanding["search_intent"]]
    session = current_session()
    if session is not None:
        understanding["_evaluation_year"] = session.cutoff.year
        understanding["_evaluation_cutoff"] = session.cutoff.isoformat()
    semaphore = asyncio.Semaphore(3)
    fallback_queries: list[str] = []

    async def lexical_search(text: str, limit: int, time_range: str | None) -> list[dict[str, Any]]:
        # 复用已有来源切换和缓存；工具预算统计动作数，不冒充底层 HTTP 请求数。
        papers, source = await _search_one(text, limit, time_range, semaphore)
        if source == "failed":
            raise RuntimeError("学术检索来源暂不可用")
        if source != "OpenAlex":
            fallback_queries.append(text)
        return papers

    budget = budget_for_mode(request.mode)
    retrieval = await retrieve_with_agent(
        query, understanding, AgentToolbox(lexical_search, _safe_arxiv_search), budget=budget,
    )
    if not retrieval.papers and retrieval.stop_reason not in {"no_results", "model_finished", "no_gain"}:
        # 不在 Agent 失败时静默切回旧两轮流程，以免界面与实际执行机制不一致。
        raise HTTPException(status_code=503, detail="Agent 检索未能取得论文，请检查模型配置或稍后重试。")

    ranked = retrieval.papers
    understanding["sub_queries"] = retrieval.queries
    for paper in ranked:
        paper["matched_query"] = paper.get("source_query", "")
        paper["recommendation_reason"] = make_recommendation_reason(query, retrieval.queries, paper)
    k = {"f1": 1, "balanced_set": 8, "exploratory": len(ranked)}[request.selection_profile]
    selected, supplementary = ranked[:k], ranked[k:]
    # Agent 的候选分布尚未重新校准，不套用旧 ECS 模型的性能或置信度口径。
    selection = {
        "profile": request.selection_profile, "strategy": "agent_ranked_set_view",
        "selected_count": len(selected), "supplementary_count": len(supplementary),
        "candidate_pool_preserved": True,
    }
    summary = build_summary(query, understanding, ranked)
    summary["retrieval_strategy"] = "模型通过原生 Function Calling 自主选择检索动作，结合工具反馈继续或结束；真实候选经 TRACE-Rank 评分后展示。"
    summary["overall_summary"] = f"本次收集到 {len(ranked)} 篇去重候选；停止类型为 {retrieval.stop_reason}，覆盖情况需结合论文核对。"
    record_stage("agent_final_candidates", ranked, stop_reason=retrieval.stop_reason)
    return SearchResponse(
        query=query, query_understanding=understanding, sub_queries=retrieval.queries,
        papers=selected, selected_results=selected, supplementary_candidates=supplementary,
        ranked_candidates=ranked, selection=selection, highly_relevant=selected,
        partially_relevant=supplementary, summary=summary,
        search_process={
            "search_strategy": "agent", "agent": retrieval.audit(budget),
            "fallback_queries": fallback_queries, "final_selection": selection,
            "data_sources": sorted({str(paper.get("source_api") or "unknown") for paper in ranked}),
        },
        metrics={
            "search_strategy": "agent", "search_mode": request.mode,
            "ranking_algorithm": "TRACE-Rank v4 + native Function Calling collection",
            "total_candidate_count": retrieval.raw_candidate_count, "deduplicated_count": len(ranked),
            "ranked_candidate_count": len(ranked), "final_count": len(selected),
            "supplementary_count": len(supplementary), "selection_profile": request.selection_profile,
            "llm_call_count": retrieval.model_calls, "llm_token_count": retrieval.token_count,
            "tool_call_count": retrieval.tool_calls, "executed_search_count": retrieval.executed_searches,
            "from_cache": False, "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        },
    )


async def search(request: SearchRequest) -> SearchResponse:
    """按请求分发原生 Agent 循环或兼容工作流，离线演示仍独立处理。"""
    started_at = time.perf_counter()
    query = request.query.strip()
    search_mode = request.mode
    selection_profile = request.selection_profile
    retrieval_profile = request.retrieval_profile
    if os.getenv("SCHOLAR_FROZEN_DEMO", "").strip().lower() in {"1", "true", "yes"}:
        demo_path = Path(os.getenv("SCHOLAR_FROZEN_DEMO_PATH", BASE_DIR / "demo/frozen_response.json"))
        payload = json.loads(demo_path.read_text(encoding="utf-8"))
        payload["query"] = query
        if selection_profile in {"balanced_set", "exploratory"} or not payload.get("selected_results"):
            payload.update(_selection_view(payload.get("ranked_candidates") or payload.get("papers") or [], selection_profile))
        payload.setdefault("metrics", {})["from_cache"] = True
        payload["metrics"]["frozen_demo"] = True
        payload.setdefault("search_process", {})["frozen_demo"] = True
        refresh_recommendation_reasons(payload)
        return SearchResponse.model_validate(payload)
    cache_identity = f"[{search_mode}|{selection_profile}|{retrieval_profile}] {query}"
    if request.search_strategy == "agent":
        # 独立版本和模型命名空间，旧两轮缓存不会跳过新 Agent 循环。
        cache_identity = f"[agent-v2|{agent_model_name()}|{search_mode}|{selection_profile}] {query}"
    if not is_academic_query(query):
        raise HTTPException(status_code=400, detail=ACADEMIC_ERROR_MESSAGE)

    # Full-response cache: identical queries return instantly with zero API/LLM cost.
    eval_session = current_session()
    cached_payload = response_cache.load(cache_identity) if eval_session is None else None
    if cached_payload is not None:
        try:
            if request.search_strategy == "workflow" and (selection_profile in {"balanced_set", "exploratory"} or not cached_payload.get("selected_results")):
                cached_payload.update(_selection_view(cached_payload.get("ranked_candidates") or cached_payload.get("papers") or [], selection_profile))
            cached_payload.setdefault("metrics", {})
            cached_payload["metrics"]["from_cache"] = True
            cached_payload["metrics"]["elapsed_seconds"] = round(time.perf_counter() - started_at, 3)
            if request.search_strategy == "agent":
                for key in ("llm_call_count", "llm_token_count", "tool_call_count", "executed_search_count"):
                    cached_payload["metrics"][key] = 0
            # 旧缓存只更新说明文字，不清空检索结果或触发外部调用。
            refresh_recommendation_reasons(cached_payload)
            response = SearchResponse(**cached_payload)
        except Exception:
            # Stale/incompatible cache entry (e.g. schema changed) — drop it and recompute.
            response_cache.invalidate(cache_identity)
        else:
            return response

    if request.search_strategy == "agent":
        response = await _search_with_agent(request, started_at)
        # 只缓存模型正常结束的结果；超时或临时错误不固化成七天的检索答案。
        stop_reason = response.search_process["agent"]["stop_reason"]
        if eval_session is None and response.ranked_candidates and stop_reason in {"completed", "model_finished", "model_stopped_partial"}:
            response_cache.save(cache_identity, response.model_dump())
        return response

    llm_token_total = 0

    # LLM-powered query understanding (falls back to rule-based on failure)
    llm_understanding, tok = await frozen_call(
        "llm_understanding", {"query": query}, lambda: llm_understand_query(query)
    )
    llm_token_total += tok
    query_understanding = understand_query(query)
    query_understanding["original_query"] = query
    if eval_session is not None:
        query_understanding["_evaluation_year"] = eval_session.cutoff.year
        query_understanding["_evaluation_cutoff"] = eval_session.cutoff.isoformat()
    query_understanding["search_intent"] = infer_search_intent(query)
    query_understanding["search_intent_label"] = INTENT_LABELS[query_understanding["search_intent"]]
    llm_topic = str((llm_understanding or {}).get("research_topic") or "").strip()
    # Reject malformed/generic LLM parses instead of allowing them to replace a
    # useful deterministic parse. This is a retrieval-quality safety gate.
    llm_parse_valid = bool(llm_topic and llm_topic.lower() not in {"not specified", "unknown", "n/a"})
    if llm_understanding and llm_parse_valid:
        for key in ("research_topic", "method_constraints", "task_constraints",
                    "domain_constraints", "time_range"):
            value = str(llm_understanding.get(key) or "").strip()
            if value and value.lower() not in {"not specified", "unknown", "n/a"}:
                query_understanding[key] = llm_understanding[key]
        llm_sqs = llm_understanding.get("sub_queries") or []
        title_hypotheses = [
            str(x).strip() for x in (llm_understanding.get("candidate_title_queries") or [])
            if isinstance(x, str) and x.strip()
        ][:3]
        query_understanding["candidate_title_queries"] = title_hypotheses
        if llm_sqs:
            # Keep deterministic anchor queries in the pool so a weak LLM parse
            # cannot drift the complete search away from the original intent.
            anchored = [*llm_sqs[:3], *(query_understanding.get("sub_queries") or [])]
            query_understanding["sub_queries"] = list(dict.fromkeys(q.strip() for q in anchored if q.strip()))[:5]

    constraint_graph = atomicize_query(query, query_understanding)
    query_portfolio = build_query_portfolio(query, query_understanding, constraint_graph)
    query_understanding["constraint_graph"] = constraint_graph
    query_understanding["query_portfolio"] = query_portfolio
    if retrieval_profile in {"portfolio", "joint"}:
        lexical_portfolio = [
            item["query"] for item in query_portfolio
            if item.get("form") != "pseudo_document"
        ]
        if lexical_portfolio:
            query_understanding["sub_queries"] = lexical_portfolio[:5]
    sub_queries   = query_understanding["sub_queries"]
    title_hypotheses = query_understanding.get("candidate_title_queries") or []
    if search_mode == "quick":
        sub_queries = sub_queries[:3]
    title_probe_limit = 1 if search_mode == "quick" else 2
    time_range: str | None = query_understanding.get("time_range")
    research_topic: str    = query_understanding.get("research_topic", "")

    if not sub_queries:
        raise HTTPException(status_code=400, detail="查询不能为空")
    record_event("query_planned", sub_queries=sub_queries, title_hypotheses=title_hypotheses,
                 query_portfolio=query_portfolio, constraint_graph=constraint_graph,
                 retrieval_profile=retrieval_profile, time_range=time_range, llm_tokens=llm_token_total)

    # Semaphore limits concurrent OpenAlex calls to 3 to avoid burst rate limits.
    semaphore = asyncio.Semaphore(3)

    # ── 第一轮：多源并发检索 ───────────────────────────────────────────────
    first_task_results = await asyncio.gather(*[
        frozen_call(
            f"first_lexical_{i}", {"query": sq, "per_page": 25, "time_range": time_range},
            lambda sq=sq: _search_one(sq, 25, time_range, semaphore),
        )
        for i, sq in enumerate(sub_queries)
    ])
    # Query-by-description lane. Hypotheses are retrieval probes only; papers
    # are displayed only after verification by an academic index.
    title_task_results = await asyncio.gather(*[
        frozen_call(
            f"title_probe_{i}", {"title": title, "per_page": 12, "time_range": time_range},
            lambda title=title: _search_one(f'"{title}"', 12, time_range, semaphore),
        )
        for i, title in enumerate(title_hypotheses[:title_probe_limit])
    ]) if title_hypotheses and retrieval_profile != "lean" else []
    semantic_query = query
    if retrieval_profile in {"portfolio", "semantic_portfolio", "joint"}:
        pseudo = next((item["query"] for item in query_portfolio if item.get("form") == "pseudo_document"), "")
        semantic_query = pseudo or query
    try:
        semantic_candidates = await frozen_call(
            "first_semantic", {"query": semantic_query, "per_page": 50},
            lambda: search_openalex_semantic(semantic_query, per_page=50),
        )
        semantic_candidates = filter_for_cutoff(semantic_candidates, "first_semantic")
        semantic_api_calls = 1
    except Exception:
        semantic_candidates = []
        semantic_api_calls = 0
    arxiv_supplements = await asyncio.gather(*[
        frozen_call(
            f"first_arxiv_{i}", {"query": sq, "per_page": 20},
            lambda sq=sq: _search_arxiv_supplement(sq, 20),
        )
        for i, sq in enumerate(sub_queries[:1 if search_mode == "quick" else 2])
    ]) if retrieval_profile != "lean" else []
    arxiv_supplements = [
        filter_for_cutoff(papers, f"first_arxiv_{i}")
        for i, papers in enumerate(arxiv_supplements)
    ]

    first_candidates: list[dict[str, Any]] = []
    oa_calls_1 = s2_calls_1 = cr_calls_1 = ax_calls_1 = 0
    fallback_queries_1: list[str] = []
    for (papers, source), sq in zip(first_task_results, sub_queries):
        papers = filter_for_cutoff(papers, f"first_lexical:{source}")
        first_candidates.extend(papers)
        if source == "OpenAlex":
            oa_calls_1 += 1
        elif source == "arXiv":
            ax_calls_1 += 1
            fallback_queries_1.append(sq)
        elif source == "Crossref":
            cr_calls_1 += 1
            fallback_queries_1.append(sq)
        elif source == "Semantic Scholar":
            s2_calls_1 += 1
            fallback_queries_1.append(sq)

    for (papers, source), title in zip(title_task_results, title_hypotheses):
        papers = filter_for_cutoff(papers, f"title_probe:{source}")
        for paper in papers:
            paper["retrieval_lane"] = "title_hypothesis"
        first_candidates.extend(papers)
        if source == "OpenAlex":
            oa_calls_1 += 1
        elif source == "Crossref":
            cr_calls_1 += 1
            fallback_queries_1.append(f"title:{title}")
        elif source == "Semantic Scholar":
            s2_calls_1 += 1
            fallback_queries_1.append(f"title:{title}")

    first_candidates.extend(semantic_candidates)

    for papers in arxiv_supplements:
        first_candidates.extend(papers)
    ax_calls_1 += len(arxiv_supplements)

    # Fail only if every first-round query returned nothing at all
    if not first_candidates:
        raise HTTPException(status_code=502, detail="所有数据源均请求失败，请稍后重试。")

    for paper in first_candidates:
        paper["source_round"] = "first_round"
    record_stage("first_raw", first_candidates)
    first_dedup = fuse_candidates(first_candidates)
    first_ranked = score_candidates(first_dedup, query_understanding)
    record_stage("first_fused_ranked", first_ranked)

    # ── TRACE-Loop：基于证据充分度自适应选择引文扩展或查询演化 ─────────────
    first_assessment = assess_retrieval(
        query, first_ranked, api_calls=oa_calls_1 + ax_calls_1 + cr_calls_1 + s2_calls_1 + semantic_api_calls
    )
    helios_trace: list[dict[str, Any]] = []
    helios_action = ""
    if search_mode == "helios":
        helios_state = build_state(query, query_understanding, first_ranked)
        helios_estimate, helios_alternatives = HELIOSLiteController().choose(helios_state)
        helios_action = helios_estimate.action
        helios_trace.append({
            "state_before": helios_state.to_dict(),
            "action": helios_action,
            "reason": helios_estimate.reason,
            "predicted_utility": helios_estimate.predicted_utility,
            "alternatives": [item.to_dict() for item in helios_alternatives],
        })
    citation_candidates: list[dict[str, Any]] = []
    citation_edges: list[dict[str, Any]] = []
    citation_api_calls = 0
    should_expand_citations = (
        False if retrieval_profile in {"lean", "relational"}
        else True if retrieval_profile in {"citation", "joint"}
        else helios_action == "citation_expand" if search_mode == "helios"
        else search_mode not in {"quick", "efficient"} and first_assessment.next_action == "citation_expansion"
    )
    if should_expand_citations:
        if retrieval_profile in {"citation", "joint"}:
            frontiers = choose_dual_frontier(first_ranked[:40], relevant_limit=4, bridge_limit=4)
            relevant_seeds = frontiers["relevant"]
            bridge_seeds = frontiers["bridge"]
            seeds = [*relevant_seeds, *bridge_seeds]
            record_event(
                "dual_frontier_selected",
                relevant=[p.get("title") for p in relevant_seeds],
                bridge=[p.get("title") for p in bridge_seeds],
            )
        else:
            seeds = [
                p for p in first_ranked[:12]
                if p.get("referenced_works")
                and float(p.get("lexical_score") or 0) >= 0.30
                and float(p.get("constraint_coverage_score") or 0) >= 0.25
            ][:4]
            relevant_seeds, bridge_seeds = seeds, []
        reference_ids: list[str] = []
        for seed in seeds:
            reference_neighbors = (seed.get("referenced_works") or [])[:35]
            related_limit = 20 if seed in bridge_seeds else 10
            related_neighbors = (seed.get("related_works") or [])[:related_limit]
            for ref_id in [*reference_neighbors, *related_neighbors]:
                reference_ids.append(ref_id)
                citation_edges.append({
                    "source": seed.get("openalex_id"), "target": ref_id,
                    "type": "related" if ref_id in related_neighbors else "references",
                    "seed_title": seed.get("title"), "frontier_role": seed.get("frontier_role", "relevant"),
                })
        if reference_ids:
            reference_ids = list(dict.fromkeys(reference_ids))
            try:
                resolved = await frozen_call(
                    "citation_references", {"work_ids": reference_ids[:140], "limit": 140},
                    lambda: fetch_openalex_works(reference_ids, source_query="citation:dual_frontier", limit=140),
                )
                resolved = filter_for_cutoff(resolved, "citation_references")
                citation_api_calls += 1
                citation_candidates.extend(resolved)
            except Exception:
                pass
        for seed in relevant_seeds[:2]:
            try:
                seed_id = str(seed.get("openalex_id") or "")
                source_query = f"citation:citing:{seed.get('title', '')[:60]}"
                citing = await frozen_call(
                    f"citation_citing_{len(citation_candidates)}",
                    {"work_id": seed_id, "source_query": source_query, "per_page": 12},
                    lambda seed_id=seed_id, source_query=source_query: fetch_openalex_citing(
                        seed_id, source_query=source_query, per_page=12,
                    ),
                )
                citing = filter_for_cutoff(citing, "citation_citing")
                citation_api_calls += 1
                citation_candidates.extend(citing)
                for paper in citing:
                    citation_edges.append({
                        "source": paper.get("openalex_id"), "target": seed.get("openalex_id"),
                        "type": "cites", "seed_title": seed.get("title"),
                    })
            except Exception:
                pass
        for paper in citation_candidates:
            paper["source_round"] = "citation_expansion"
        if search_mode == "helios" and citation_candidates:
            scored_citation = score_candidates(fuse_candidates(citation_candidates), query_understanding)
            accepted_citation, rejected_citation = admit_candidates(scored_citation, enabled=True, limit=50)
            accepted_keys = {canonical_key(paper) for paper in accepted_citation}
            citation_candidates = [paper for paper in citation_candidates if canonical_key(paper) in accepted_keys]
            helios_trace[-1].update({
                "actual_new_candidates": len(scored_citation),
                "actual_accepted_candidates": len(accepted_citation),
                "actual_rejected_candidates": len(rejected_citation),
            })
        if citation_candidates:
            first_candidates.extend(citation_candidates)
            first_dedup = fuse_candidates(first_candidates)
            first_ranked = score_candidates(first_dedup, query_understanding)
    record_stage("post_citation", first_ranked, citation_candidate_count=len(citation_candidates),
                 citation_api_calls=citation_api_calls)

    post_expansion_assessment = assess_retrieval(
        query, first_ranked,
        api_calls=oa_calls_1 + ax_calls_1 + cr_calls_1 + s2_calls_1 + citation_api_calls + semantic_api_calls,
    )

    # ── LLM 自主盲区分析 + 第二轮检索式生成（Agent 迭代核心）──────────────────
    # LLM 分析第一轮结果覆盖盲区，自主规划补充检索式；失败时降级为规则关键词扩展。
    expanded_keywords = _extract_expanded_keywords(first_ranked[:5])
    if search_mode == "helios" and helios_action != "second_round_search":
        gap_analysis, expanded_queries = f"HELIOS-Lite 选择动作 {helios_action or 'stop'}，不执行第二轮查询演化。", []
        expansion_mode = "helios-stop"
    elif search_mode == "efficient":
        gap_analysis, expanded_queries = "低成本模式保留完整首轮多通道检索，并停止高成本引文与二轮扩展。", []
        expansion_mode = "efficient-stop"
    elif post_expansion_assessment.next_action == "stop_sufficient":
        gap_analysis, expanded_queries = "当前证据充分度已达到停止阈值，不再增加检索成本。", []
        expansion_mode = "adaptive-stop"
    else:
        gap_analysis, expanded_queries, tok = await frozen_call(
            "llm_query_expansion",
            {"query": query, "paper_titles": [p.get("title") for p in first_ranked[:5]]},
            lambda: llm_expand_queries(query, first_ranked[:5]),
        )
        llm_token_total += tok
        expansion_mode = "llm"
        if not expanded_queries:
            expanded_queries = _build_expanded_queries(research_topic, expanded_keywords)
            expansion_mode = "rule-based"
    expanded_queries = expanded_queries[:1 if search_mode == "quick" else 2]

    # ── 第二轮：迭代检索（失败不影响第一轮结果）────────────────────────────
    second_candidates: list[dict[str, Any]] = []
    oa_calls_2 = s2_calls_2 = cr_calls_2 = ax_calls_2 = 0
    fallback_queries_2: list[str] = []
    semantic_expansion_calls = 0
    if expanded_queries:
        second_task_results = await asyncio.gather(*[
            frozen_call(
                f"second_lexical_{i}", {"query": eq, "per_page": 25, "time_range": time_range},
                lambda eq=eq: _search_one(eq, 25, time_range, semaphore),
            )
            for i, eq in enumerate(expanded_queries)
        ])
        if retrieval_profile != "lean":
            second_arxiv = await frozen_call(
                "second_arxiv", {"query": expanded_queries[0], "per_page": 20},
                lambda: _search_arxiv_supplement(expanded_queries[0], 20),
            )
            second_arxiv = filter_for_cutoff(second_arxiv, "second_arxiv")
            second_candidates.extend(second_arxiv)
            ax_calls_2 += 1
        for (papers, source), eq in zip(second_task_results, expanded_queries):
            papers = filter_for_cutoff(papers, f"second_lexical:{source}")
            second_candidates.extend(papers)
            if source == "OpenAlex":
                oa_calls_2 += 1
            elif source == "arXiv":
                ax_calls_2 += 1
                fallback_queries_2.append(eq)
            elif source == "Crossref":
                cr_calls_2 += 1
                fallback_queries_2.append(eq)
            elif source == "Semantic Scholar":
                s2_calls_2 += 1
                fallback_queries_2.append(eq)
        if search_mode != "quick":
            semantic_followups = expanded_queries[:2 if search_mode == "deep" else 1]
            for semantic_query in semantic_followups:
                try:
                    semantic_more = await frozen_call(
                        f"second_semantic_{semantic_expansion_calls}",
                        {"query": semantic_query, "per_page": 50},
                        lambda semantic_query=semantic_query: search_openalex_semantic(semantic_query, per_page=50),
                    )
                    semantic_more = filter_for_cutoff(semantic_more, "second_semantic")
                    for paper in semantic_more:
                        paper["source_round"] = "semantic_followup"
                    second_candidates.extend(semantic_more)
                    semantic_expansion_calls += 1
                except Exception:
                    pass
    if search_mode == "helios" and second_candidates:
        scored_second = score_candidates(fuse_candidates(second_candidates), query_understanding)
        accepted_second, rejected_second = admit_candidates(scored_second, enabled=True, limit=50)
        accepted_keys = {canonical_key(paper) for paper in accepted_second}
        second_candidates = [paper for paper in second_candidates if canonical_key(paper) in accepted_keys]
        helios_trace[-1].update({
            "actual_new_candidates": len(scored_second),
            "actual_accepted_candidates": len(accepted_second),
            "actual_rejected_candidates": len(rejected_second),
        })
    record_stage("second_raw", second_candidates, expanded_queries=expanded_queries,
                 semantic_followup_calls=semantic_expansion_calls)

    # ── 合并两轮，去重排序 ────────────────────────────────────────────────
    for paper in second_candidates:
        paper["source_round"] = "second_round"
    merged = fuse_candidates([*first_candidates, *second_candidates])
    record_stage("merged_candidates", merged)
    # Preserve a wider recall-oriented shortlist until semantic reranking.
    ranked_all = diversify(
        score_candidates(merged, query_understanding), limit=32, lambda_relevance=0.88
    )
    record_stage("shortlist_pre_llm", ranked_all)

    all_queries = sub_queries + expanded_queries

    # Pre-fill matched_query and score_explanation (no LLM needed)
    for paper in ranked_all:
        paper["matched_query"]     = paper.get("source_query", "")
        paper["score_explanation"] = _make_score_explanation(paper)
        paper.pop("search_text", None)
        paper.pop("_openalex_rank", None)

    # LLM: summary + recommendation reasons + relevance flags in one call
    llm_data, tok = await frozen_call(
        "llm_rerank_summary",
        {"query": query, "paper_titles": [p.get("title") for p in ranked_all]},
        lambda: llm_build_summary(query, ranked_all),
    )
    llm_token_total += tok

    llm_filtered_count = 0
    if llm_data:
        summary = {
            "question_understanding": llm_data.get("question_understanding", ""),
            "retrieval_strategy":     llm_data.get("retrieval_strategy", ""),
            "overall_summary":        llm_data.get("overall_summary", ""),
        }
        reasons = llm_data.get("recommendation_reasons") or {}
        flags   = llm_data.get("relevance_flags") or {}
        relevance_scores = llm_data.get("relevance_scores") or {}
        for i, paper in enumerate(ranked_all):
            paper["recommendation_reason"] = (
                reasons.get(str(i + 1))
                or make_recommendation_reason(query, all_queries, paper)
            )
            # LLM relevance flag: default True so we never drop papers on missing keys
            raw_score = relevance_scores.get(str(i + 1))
            try:
                llm_score = max(0.0, min(float(raw_score) / 100.0, 1.0))
            except (TypeError, ValueError):
                llm_score = None
            paper["llm_relevance_score"] = llm_score
            if llm_score is not None:
                paper["final_score"] = round(0.55 * float(paper.get("final_score") or 0) + 0.45 * llm_score, 4)
                paper["relevance_score"] = paper["final_score"]
            paper["llm_relevant"] = bool(flags.get(str(i + 1), llm_score is None or llm_score >= 0.45))

        ranked_all.sort(key=lambda p: float(p.get("final_score") or 0), reverse=True)

        # Keep the full Top-20 for recall-oriented evaluation. LLM judgments are
        # exposed as scores/flags and influence ranking, but do not hard-delete
        # candidates; hard filtering was found to damage F1 on fine-grained queries.
        llm_filtered_count = sum(not p.get("llm_relevant", True) for p in ranked_all)
    else:
        summary = build_summary(query, query_understanding, ranked_all)
        for paper in ranked_all:
            paper["recommendation_reason"] = make_recommendation_reason(query, all_queries, paper)
            paper["llm_relevant"] = True

    relational_audit: dict[str, Any] = {"enabled": False, "api_calls": 0, "actions": []}
    relational_supplementary: list[dict[str, Any]] = []
    if retrieval_profile == "relational" and ranked_all:
        baseline_top20 = list(ranked_all[:20])
        try:
            relational_pool = score_candidates(merged, query_understanding)
            routed_top20, relational_audit = await expand_and_route_relational(
                query, query_understanding, baseline_top20, relational_pool,
            )
            relational_supplementary = list(
                relational_audit.pop("supplementary_candidates", []) or []
            )
            routed_keys = {canonical_key(paper) for paper in routed_top20}
            untouched_tail = [
                paper for paper in ranked_all[20:]
                if canonical_key(paper) not in routed_keys
            ]
            ranked_all = [*routed_top20, *untouched_tail]
            for paper in ranked_all:
                paper.setdefault("matched_query", paper.get("source_query", ""))
                paper.setdefault("score_explanation", _make_score_explanation(paper))
                paper.setdefault("recommendation_reason", make_recommendation_reason(query, all_queries, paper))
                paper.setdefault("llm_relevant", True)
            for paper in relational_supplementary:
                paper.setdefault("matched_query", paper.get("source_query", ""))
                paper.setdefault("score_explanation", _make_score_explanation(paper))
                paper.setdefault("recommendation_reason", make_recommendation_reason(query, all_queries, paper))
                paper.setdefault("llm_relevant", True)
            record_stage("relational_evidence_routing", routed_top20, audit=relational_audit)
        except Exception as exc:
            relational_audit = {
                "enabled": False, "api_calls": 0, "actions": [],
                "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                "strategy": "REG_v18_graceful_fallback",
            }

    record_stage("shortlist_post_llm", ranked_all)

    ranked_all = ranked_all[:20]
    record_stage("final_top20", ranked_all)
    if selection_profile == "exploratory":
        selected_results = list(ranked_all)
        selector_metadata = {"strategy": "exploratory_top20", "profile": selection_profile}
    else:
        selected_results, selector_metadata = await select_with_champion(
            query, constraint_graph, ranked_all, selection_profile
        )
    selected_keys = {canonical_key(paper) for paper in selected_results}
    supplementary_candidates = [paper for paper in ranked_all if canonical_key(paper) not in selected_keys]
    supplementary_keys = {canonical_key(paper) for paper in supplementary_candidates}
    ranked_keys = {canonical_key(paper) for paper in ranked_all}
    for paper in relational_supplementary:
        key = canonical_key(paper)
        if key and key not in selected_keys and key not in supplementary_keys:
            supplementary_candidates.append(paper)
            supplementary_keys.add(key)
    ranked_candidates = list(ranked_all)
    for paper in relational_supplementary:
        key = canonical_key(paper)
        if key and key not in ranked_keys:
            ranked_candidates.append(paper)
            ranked_keys.add(key)
    selection = {
        **selector_metadata,
        "profile": selection_profile,
        "selected_count": len(selected_results),
        "supplementary_count": len(supplementary_candidates),
        "candidate_pool_preserved": True,
        "protocol": "macro_set_f1",
    }
    record_stage("final_selected", selected_results, selection=selection)
    attach_evidence_and_roles(ranked_all, query_understanding)
    evidence_matrix = build_evidence_matrix(ranked_all, query_understanding)
    research_landscape = build_research_landscape(
        ranked_all, int(query_understanding.get("_evaluation_year") or 0) or None
    )

    highly_relevant, _ = split_relevance(selected_results)
    partially_relevant = supplementary_candidates
    relation_graph = build_relation_graph(ranked_all)
    visible_ids = {node.get("id") for node in relation_graph.get("nodes", [])}
    for edge in citation_edges:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        # Graph node IDs use canonical keys; DOI/title nodes may not map to
        # OpenAlex IDs, so only add verifiable visible OpenAlex relationships.
        source_key = source.lower()
        target_key = target.lower()
        if source_key in visible_ids and target_key in visible_ids:
            relation_graph["edges"].append({**edge, "source": source_key, "target": target_key, "weight": 1.0})

    all_fallback = fallback_queries_1 + fallback_queries_2
    data_sources = sorted({p.get("source_api", "OpenAlex") for p in merged if p.get("source_api")})

    search_process = {
        "first_round_queries":  sub_queries,
        "expanded_keywords":    expanded_keywords,
        "second_round_queries": expanded_queries,
        "expansion_mode":       expansion_mode,
        "gap_analysis":         gap_analysis,
        "fallback_queries":     all_fallback,
        "data_sources":         data_sources,
        "strategy_explanation": (
            "系统先用大模型解析查询意图并生成多条子查询，调用 OpenAlex 进行第一轮检索；"
            "若 OpenAlex 受限，自动逐级切换 arXiv（CS/AI 强、含摘要）、Crossref、Semantic Scholar 补充检索；"
            "随后大模型分析第一轮结果的覆盖盲区，自主规划第二轮补充检索式（迭代检索）；"
            "再由大模型对候选论文做相关性判定、过滤离题结果；"
            "最后使用 TRACE-Rank 完成多路径倒数排名融合、查询约束覆盖评分和多样性重排，"
            "并生成可追踪的结构化总结与论文关系图。"
            + (
                "关系证据图模式进一步使用 SCGQ 与 QCF 探索受控引文前沿，"
                "并依据条件合取、关系谓词、引文方向和种子桥接证据受控替换 Top20 尾部结果。"
                if retrieval_profile == "relational" else ""
            )
        ),
        "helios_trace": helios_trace,
        "adaptive_controller": {
            "first_assessment": first_assessment.to_dict(),
            "post_expansion_assessment": post_expansion_assessment.to_dict(),
        },
        "final_selection": selection,
        "citation_expansion": {
            "candidate_count": len(citation_candidates),
            "api_calls": citation_api_calls,
            "edges": citation_edges[:60],
            "retrieval_profile": retrieval_profile,
        },
        "relational_evidence_graph": relational_audit,
    }

    oa_total  = oa_calls_1 + oa_calls_2
    s2_total  = s2_calls_1 + s2_calls_2
    cr_total  = cr_calls_1 + cr_calls_2
    ax_total  = ax_calls_1 + ax_calls_2
    first_cnt  = len(first_candidates)
    second_cnt = len(second_candidates)
    metrics = {
        "first_round_api_calls":        len(sub_queries),
        "second_round_api_calls":       len(expanded_queries),
        "openalex_api_calls":           oa_total,
        "openalex_semantic_api_calls":  semantic_api_calls + semantic_expansion_calls,
        "arxiv_api_calls":              ax_total,
        "crossref_api_calls":           cr_total,
        "semantic_scholar_api_calls":   s2_total,
        "fallback_count":               len(all_fallback),
        "citation_api_calls":           citation_api_calls,
        "relational_api_calls":         int(relational_audit.get("api_calls") or 0),
        "relational_promotions":        len(relational_audit.get("actions") or []),
        "api_call_count":               oa_total + ax_total + cr_total + s2_total + citation_api_calls + semantic_api_calls + semantic_expansion_calls + int(relational_audit.get("api_calls") or 0),
        "first_round_candidate_count":  first_cnt,
        "second_round_candidate_count": second_cnt,
        "total_candidate_count":        first_cnt + second_cnt,
        "candidate_count":              first_cnt + second_cnt,
        "deduplicated_count":           len(merged),
        "final_count":                  len(selected_results),
        "ranked_candidate_count":       len(ranked_candidates),
        "supplementary_count":          len(supplementary_candidates),
        "selection_profile":            selection_profile,
        "retrieval_profile":            retrieval_profile,
        "ranking_algorithm":            (
            "TRACE-Rank v4 + SCGQ v15 + REG-MEQ v20 + EBP v21"
            if retrieval_profile == "relational"
            else "TRACE-Rank v4 + TRACE-Loop v2"
        ),
        "search_mode":                  search_mode,
        "search_intent":                query_understanding.get("search_intent"),
        "relation_edge_count":          len(relation_graph.get("edges", [])),
        "llm_call_count":               3,
        "llm_token_count":              llm_token_total,
        "llm_filtered_count":           llm_filtered_count,
        "from_cache":                   False,
        "elapsed_seconds":              round(time.perf_counter() - started_at, 3),
    }

    response = SearchResponse(
        query=query,
        query_understanding=query_understanding,
        sub_queries=sub_queries,
        papers=selected_results,
        selected_results=selected_results,
        supplementary_candidates=supplementary_candidates,
        ranked_candidates=ranked_candidates,
        selection=selection,
        highly_relevant=highly_relevant,
        partially_relevant=partially_relevant,
        summary=summary,
        search_process=search_process,
        metrics=metrics,
        relation_graph=relation_graph,
        evidence_matrix=evidence_matrix,
        research_landscape=research_landscape,
    )

    # 同一批模型输出若给多篇相同套话，回退为各自可核对的摘要依据。
    payload = response.model_dump()
    refresh_recommendation_reasons(payload)
    response = SearchResponse.model_validate(payload)

    # Persist the real result so an identical query is instant + free next time.
    if eval_session is None:
        response_cache.save(cache_identity, response.model_dump())
    return response


@app.post("/api/search", response_model=PublicSearchResponse)
async def search_for_web(request: SearchRequest) -> PublicSearchResponse:
    """网页只接收论文及必要提示；评测与缓存继续使用完整的后台结果。"""
    response = await search(request)
    return build_public_response(response.model_dump())
