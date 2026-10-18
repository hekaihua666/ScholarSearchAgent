"""模型不可用时的摘要与推荐依据；只引用已有元数据，不编造研究结论。"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from backend.query_understanding import understand_query
from backend.trace_rank import tokenize


def decompose_query(query: str) -> list[str]:
    """Compatibility wrapper for older code paths."""
    return understand_query(query)["sub_queries"]


def build_summary(
    original_query: str,
    query_understanding: dict[str, Any],
    papers: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a simple structured summary for the search result."""
    top_titles = [paper["title"] for paper in papers[:3]]
    method_constraints = query_understanding.get("method_constraints") or "not specified"
    task_constraints = query_understanding.get("task_constraints") or "not specified"
    domain_constraints = query_understanding.get("domain_constraints") or "not specified"

    return {
        "question_understanding": (
            f"用户问题聚焦于「{query_understanding.get('research_topic') or original_query}」。"
            f"方法约束包括：{method_constraints}；任务约束包括：{task_constraints}；"
            f"领域限定包括：{domain_constraints}。"
        ),
        "retrieval_strategy": (
            "系统先抽取研究主题、方法、任务和领域约束，再组合英文检索式调用 OpenAlex，"
            "最后对候选论文去重、分项评分、综合排序和分层输出。"
        ),
        "overall_summary": (
            "当前结果优先推荐与查询主题、方法约束、任务目标、引用影响力和发表新近性综合匹配的论文。"
        ),
        "top_papers": top_titles,
    }


def make_recommendation_reason(query: str, sub_queries: list[str], paper: dict[str, Any]) -> str:
    """从各篇标题和摘要提取依据，避免依赖已被清理的 search_text。

    有摘要时优先选取描述论文自身工作的句子，保留原文供用户核对；
    没有摘要或未匹配到检索词时明确限制，不将召回等同于满足问题。
    """
    title = " ".join(str(paper.get("title") or "").split())
    abstract = " ".join(str(paper.get("abstract") or "").split())

    def terms(text: str) -> set[str]:
        # 复用排序模块的英文停用词，并支持以空格或标点分开的中文术语。
        return tokenize(text) | set(re.findall(r"[\u4e00-\u9fff]{2,}", text))

    query_terms = terms(" ".join([query, *sub_queries]))
    title_matches = terms(title) & query_terms
    matches = sorted(title_matches | (terms(abstract) & query_terms))
    # 优先展示题名中确实出现的术语，而非每篇相同的整条子查询。
    ordered = sorted(title_matches) + [term for term in matches if term not in title_matches]
    matched_note = "匹配「" + "、".join(ordered[:4]) + "」" if ordered else ""

    if not abstract:
        context = f"标题{matched_note}；" if title_matches else ""
        return f"{context}暂无摘要，仅凭题名「{title or '未知标题'}」尚不能确认是否满足研究问题，需核对原文。"

    sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+|(?<=[。！？])", abstract) if s.strip()]
    contribution = re.compile(
        r"\b(?:we|this (?:paper|work|study))\b.{0,65}\b(?:propos\w*|introduc\w*|develop\w*|present\w*|provid\w*)\b"
        r"|本文.{0,20}(?:提出|构建|设计|研究|综述)", re.I,
    )
    # 先排除纯背景句：自身工作描述比单纯重叠更多关键词更有解释价值。
    sentence = max(sentences, key=lambda s: (bool(contribution.search(s)), len(terms(s) & query_terms)))
    if len(sentence) > 260:
        sentence = sentence[:257].rstrip() + "…"
    prefix = matched_note + "；" if matched_note else "与问题的具体关联待核对；"
    return f"{prefix}摘要提到：「{sentence}」"


def refresh_recommendation_reasons(payload: dict[str, Any]) -> None:
    """修复旧缓存里的空泛或重复理由，保持候选、分数和检索缓存原样。

    同一论文在多个返回分组中出现不算重复；只有不同论文共用的理由
    才重建。已有具体且不重复的模型理由保留，不产生额外模型请求。
    """
    groups = ("papers", "selected_results", "supplementary_candidates", "ranked_candidates",
              "highly_relevant", "partially_relevant")
    papers = [paper for group in groups for paper in payload.get(group, [])]
    unique = {(str(paper.get("title") or ""), paper.get("year")): paper for paper in papers}
    counts = Counter(str(paper.get("recommendation_reason") or "").strip() for paper in unique.values())
    query = str(payload.get("query") or "")
    process = payload.get("search_process") or {}
    sub_queries = [q for q in [*(payload.get("sub_queries") or []), *(process.get("second_round_queries") or [])]
                   if isinstance(q, str)]
    for paper in papers:
        reason = str(paper.get("recommendation_reason") or "").strip()
        legacy = "综合排序中表现较好" in reason or "存在主题或方法关键词匹配" in reason
        if not reason or legacy or counts[reason] > 1:
            paper["recommendation_reason"] = make_recommendation_reason(query, sub_queries, paper)
