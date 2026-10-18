"""Semantic Scholar Graph API 客户端（作为 OpenAlex 的备用源）。"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

from backend.eval_runtime import current_cutoff_iso


S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_FIELDS = "title,abstract,year,publicationDate,authors,citationCount,url,externalIds"
# Optional free API key: https://www.semanticscholar.org/product/api#api-key-form
# Set S2_API_KEY env var to get 100 req/s instead of 1 req/s.
_S2_API_KEY: str | None = os.environ.get("S2_API_KEY")


def _normalize_s2_paper(paper: dict[str, Any], rank: int, source_query: str) -> dict[str, Any]:
    authors = [
        a.get("name", "") for a in (paper.get("authors") or [])[:6] if a.get("name")
    ]
    ext = paper.get("externalIds") or {}
    doi = ext.get("DOI")
    paper_id = paper.get("paperId") or ""
    url = paper.get("url") or (f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else None)
    abstract = paper.get("abstract") or ""
    title = paper.get("title") or "Untitled"

    return {
        "openalex_id": f"s2:{paper_id}",
        "title": title,
        "authors": authors,
        "year": paper.get("year"),
        "publication_date": paper.get("publicationDate"),
        "abstract": abstract,
        "cited_by_count": paper.get("citationCount") or 0,
        "doi": doi,
        "url": doi or url or paper_id,
        "source_query": source_query,
        "source_api": "Semantic Scholar",
        "search_text": f"{title} {abstract}",
        "_openalex_rank": rank,
    }


async def search_semantic_scholar(
    query: str,
    per_page: int = 15,
) -> list[dict[str, Any]]:
    """调用 Semantic Scholar Graph API 搜索论文，429 时最多重试 2 次。"""
    params: dict[str, Any] = {
        "query": query,
        "limit": per_page,
        "fields": _FIELDS,
    }
    if current_cutoff_iso():
        params["publicationDateOrYear"] = f":{current_cutoff_iso()}"
    timeout = httpx.Timeout(16.0, connect=8.0)
    headers: dict[str, str] = {"User-Agent": "YanwenZhiso-MVP/0.1"}
    if _S2_API_KEY:
        headers["x-api-key"] = _S2_API_KEY

    for attempt in range(3):
        # With API key: no delay needed. Without: space requests to stay under 1 req/s.
        if not _S2_API_KEY:
            await asyncio.sleep(1.0 * (attempt + 1))
        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            response = await client.get(S2_SEARCH_URL, params=params)

        if response.status_code == 429:
            if attempt < 2:
                await asyncio.sleep(max(int(response.headers.get("Retry-After", 5)), 5))
                continue
            response.raise_for_status()

        response.raise_for_status()
        papers = response.json().get("data") or []
        return [
            _normalize_s2_paper(p, rank=i + 1, source_query=query)
            for i, p in enumerate(papers)
        ]

    return []
