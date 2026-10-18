"""Intent routing and verifiable research-landscape evidence utilities."""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from typing import Any

from backend.trace_rank import tokenize


INTENT_LABELS = {
    "specific_paper": "特定论文定位",
    "foundational": "奠基工作追溯",
    "frontier": "前沿进展发现",
    "survey": "主题综述",
    "comparison": "方法对比",
    "method_lineage": "技术演化追踪",
    "broad": "开放主题检索",
}


def infer_search_intent(query: str) -> str:
    q = query.lower()
    if re.search(r"\b(first|original|introduced|proposed|who proposed|foundation|foundational|earliest)\b", q) or any(
        x in query for x in ("最早", "首次", "谁提出", "奠基", "开创")
    ):
        return "foundational"
    if re.search(r"\b(latest|recent|newest|state[- ]of[- ]the[- ]art|sota|frontier)\b", q) or any(
        x in query for x in ("最新", "近三年", "前沿", "近期进展")
    ):
        return "frontier"
    if re.search(r"\b(compare|comparison|versus|vs\.?|difference|better than)\b", q) or any(
        x in query for x in ("比较", "对比", "区别", "优劣")
    ):
        return "comparison"
    if re.search(r"\b(survey|review|overview|systematic review|landscape)\b", q) or any(
        x in query for x in ("综述", "系统回顾", "研究现状", "全景")
    ):
        return "survey"
    if re.search(r"\b(evolution|lineage|developed from|extend|extension|follow-up)\b", q) or any(
        x in query for x in ("演化", "脉络", "沿革", "扩展", "发展路线")
    ):
        return "method_lineage"
    if re.search(r"\b(which work|which paper|what paper|who proposed|find the paper)\b", q) or any(
        x in query for x in ("哪篇论文", "具体论文", "论文名称")
    ):
        return "specific_paper"
    return "broad"


def attach_evidence_and_roles(
    papers: list[dict[str, Any]], understanding: dict[str, Any]
) -> list[dict[str, Any]]:
    query_tokens = tokenize(str(understanding.get("original_query") or ""))
    for key in ("research_topic", "method_constraints", "task_constraints", "domain_constraints"):
        query_tokens |= tokenize(str(understanding.get(key) or ""))
    current_year = int(understanding.get("_evaluation_year") or datetime.now().year)

    for paper in papers:
        abstract = str(paper.get("abstract") or "")
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", abstract) if s.strip()]
        best_sentence, best_terms = "", set()
        for sentence in sentences:
            matched = query_tokens & tokenize(sentence)
            if len(matched) > len(best_terms):
                best_sentence, best_terms = sentence, matched
        if not best_sentence and abstract:
            best_sentence = abstract[:360].strip()
            best_terms = query_tokens & tokenize(best_sentence)
        if len(best_sentence) > 420:
            best_sentence = best_sentence[:417].rsplit(" ", 1)[0] + "..."
        start = abstract.find(best_sentence.rstrip("...")) if best_sentence else -1
        paper["evidence_snippet"] = best_sentence
        paper["evidence_matched_terms"] = sorted(best_terms)
        paper["evidence_span"] = {
            "start": start,
            "end": start + len(best_sentence.rstrip("...")) if start >= 0 else -1,
        }
        paper["evidence_source"] = "abstract" if best_sentence else "metadata"
        paper["evidence_strength"] = round(
            min(0.55 * float(paper.get("constraint_coverage_score") or 0)
                + 0.45 * min(len(best_terms) / max(len(query_tokens), 1), 1.0), 1.0), 4
        )

        year = int(paper.get("year") or current_year)
        authority = float(paper.get("authority_score") or 0)
        hits = len(paper.get("retrieval_hits") or [])
        if year >= current_year - 2:
            role = "frontier"
            reason = "近三年发表，代表当前前沿进展"
        elif year <= current_year - 7 and authority >= 0.48:
            role = "foundation"
            reason = "发表时间较早且持续获得引用，可作为基础工作"
        elif hits >= 2 or str(paper.get("source_query") or "").startswith("citation:"):
            role = "bridge"
            reason = "被多条检索路径或引文路径共同发现，连接不同研究分支"
        else:
            role = "development"
            reason = "处于基础工作与最新进展之间，体现方法发展"
        paper["research_role"] = role
        paper["research_role_reason"] = reason
    return papers


def build_evidence_matrix(
    papers: list[dict[str, Any]], understanding: dict[str, Any], limit: int = 12
) -> dict[str, Any]:
    facets = []
    labels = {
        "research_topic": "研究主题",
        "method_constraints": "方法约束",
        "task_constraints": "任务约束",
        "domain_constraints": "领域约束",
    }
    for key, label in labels.items():
        value = str(understanding.get(key) or "").strip()
        if value and value.lower() not in {"not specified", "unknown", "n/a"}:
            facets.append({"key": key, "label": label, "value": value})
    rows = []
    for paper in papers[:limit]:
        coverage = paper.get("constraint_coverage") or {}
        rows.append({
            "title": paper.get("title"),
            "paper_id": paper.get("openalex_id") or paper.get("doi"),
            "role": paper.get("research_role"),
            "final_score": paper.get("final_score"),
            "evidence_strength": paper.get("evidence_strength"),
            "evidence_snippet": paper.get("evidence_snippet"),
            "coverage": {facet["key"]: round(float(coverage.get(facet["key"], 0)), 4) for facet in facets},
        })
    return {"facets": facets, "rows": rows}


def build_research_landscape(
    papers: list[dict[str, Any]], current_year: int | None = None
) -> dict[str, Any]:
    current_year = int(current_year or datetime.now().year)
    role_counts = Counter(str(p.get("research_role") or "development") for p in papers)
    source_counts = Counter(str(p.get("source_api") or "unknown") for p in papers)
    year_buckets = Counter()
    terms = Counter()
    stop = {"using", "based", "learning", "model", "models", "approach", "study", "paper"}
    for paper in papers:
        year = int(paper.get("year") or 0)
        if year >= current_year - 2:
            year_buckets["近3年"] += 1
        elif year >= current_year - 7:
            year_buckets["4-8年"] += 1
        elif year:
            year_buckets["8年以上"] += 1
        terms.update(t for t in tokenize(str(paper.get("title") or "")) if t not in stop)
    avg_coverage = sum(float(p.get("constraint_coverage_score") or 0) for p in papers) / max(len(papers), 1)
    avg_evidence = sum(float(p.get("evidence_strength") or 0) for p in papers) / max(len(papers), 1)
    return {
        "role_counts": dict(role_counts),
        "source_counts": dict(source_counts),
        "year_distribution": dict(year_buckets),
        # Counter preserves first-seen order for ties, while tokenization may
        # originate from sets. Use an explicit lexical tie-break so frozen
        # replay is byte-for-byte deterministic.
        "top_terms": [
            {"term": term, "count": count}
            for term, count in sorted(terms.items(), key=lambda item: (-item[1], item[0]))[:10]
        ],
        "average_constraint_coverage": round(avg_coverage, 4),
        "average_evidence_strength": round(avg_evidence, 4),
    }
