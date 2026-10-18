"""Paper deduplication, scoring, and ranking."""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any


def deduplicate_papers(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate papers by DOI, OpenAlex ID, or title."""
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []

    for paper in papers:
        key = (paper.get("doi") or paper.get("openalex_id") or paper.get("title") or "").lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(paper)

    return unique


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", text.lower())
    stopwords = {
        "and",
        "are",
        "for",
        "from",
        "how",
        "into",
        "the",
        "this",
        "that",
        "with",
        "what",
        "which",
    }
    return {token for token in tokens if token not in stopwords}


def _query_tokens(query_understanding: dict[str, Any]) -> set[str]:
    fields = [
        "research_topic",
        "method_constraints",
        "task_constraints",
        "domain_constraints",
    ]
    text = " ".join(str(query_understanding.get(field) or "") for field in fields)
    return _tokenize(text)


def _openalex_score(paper: dict[str, Any]) -> float:
    rank = int(paper.get("_openalex_rank") or 100)
    return round(max(0.0, 1.0 - ((rank - 1) / 20.0)), 4)


def _keyword_score(paper: dict[str, Any], query_tokens: set[str]) -> float:
    if not query_tokens:
        return 0.0
    paper_text = f"{paper.get('title') or ''} {paper.get('abstract') or ''} {paper.get('search_text') or ''}"
    paper_tokens = _tokenize(paper_text)
    if not paper_tokens:
        return 0.0
    matched = len(query_tokens & paper_tokens)
    return round(min(matched / max(len(query_tokens), 1), 1.0), 4)


def _citation_score(paper: dict[str, Any]) -> float:
    citation_count = int(paper.get("cited_by_count") or 0)
    return round(min(math.log1p(citation_count) / math.log1p(1000), 1.0), 4)


def _freshness_score(paper: dict[str, Any]) -> float:
    year = int(paper.get("year") or 0)
    if not year:
        return 0.0
    age = max(datetime.now().year - year, 0)
    return round(max(0.0, 1.0 - (age / 12.0)), 4)


def rank_papers(
    papers: list[dict[str, Any]],
    query_understanding: dict[str, Any],
) -> list[dict[str, Any]]:
    """Score and rank papers by OpenAlex rank, keyword match, citations, and freshness."""
    tokens = _query_tokens(query_understanding)

    for paper in papers:
        openalex_score = _openalex_score(paper)
        keyword_score = _keyword_score(paper, tokens)
        citation_score = _citation_score(paper)
        freshness_score = _freshness_score(paper)
        final_score = (
            0.35 * openalex_score
            + 0.25 * keyword_score
            + 0.20 * citation_score
            + 0.20 * freshness_score
        )

        paper["openalex_score"] = openalex_score
        paper["keyword_score"] = keyword_score
        paper["citation_score"] = citation_score
        paper["freshness_score"] = freshness_score
        paper["final_score"] = round(final_score, 4)
        # Keep the old field for frontend/API compatibility.
        paper["relevance_score"] = paper["final_score"]

    return sorted(papers, key=lambda item: item["final_score"], reverse=True)


def split_relevance(
    papers: list[dict[str, Any]],
    threshold: float = 0.55,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split ranked papers into highly and partially relevant groups."""
    highly_relevant = [paper for paper in papers if float(paper.get("final_score") or 0) >= threshold]
    partially_relevant = [paper for paper in papers if float(paper.get("final_score") or 0) < threshold]
    return highly_relevant, partially_relevant
