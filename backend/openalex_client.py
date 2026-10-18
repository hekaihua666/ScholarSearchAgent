"""OpenAlex 检索客户端。"""

from __future__ import annotations

import asyncio
import html
import os
import re
from datetime import datetime
from typing import Any

import httpx
from dotenv import load_dotenv

from backend.eval_runtime import current_cutoff_iso

load_dotenv()


OPENALEX_WORKS_URL = "https://api.openalex.org/works"
_SINGLE_OPENALEX_KEY = os.environ.get("OPENALEX_API_KEY", "").strip()
_OPENALEX_API_KEYS = [
    key.strip() for key in os.environ.get("OPENALEX_API_KEYS", "").split(",") if key.strip()
]
if _SINGLE_OPENALEX_KEY and _SINGLE_OPENALEX_KEY not in _OPENALEX_API_KEYS:
    _OPENALEX_API_KEYS.insert(0, _SINGLE_OPENALEX_KEY)
_KEY_CURSOR = 0
_EXHAUSTED_KEYS: set[str] = set()
_SELECT = "id,title,display_name,authorships,publication_year,publication_date,abstract_inverted_index,cited_by_count,doi,primary_location,referenced_works,related_works"
_SEMANTIC_LOCK = asyncio.Lock()
_SEMANTIC_LAST_CALL = 0.0


def _next_api_key() -> str:
    global _KEY_CURSOR
    available = [key for key in _OPENALEX_API_KEYS if key not in _EXHAUSTED_KEYS]
    if not available:
        return ""
    key = available[_KEY_CURSOR % len(available)]
    _KEY_CURSOR += 1
    return key


def _mark_key_exhausted(key: str) -> None:
    if key:
        _EXHAUSTED_KEYS.add(key)


class OpenAlexRateLimitError(Exception):
    """Raised when OpenAlex returns 429 after exhausting retries."""


def _clean_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    if not inverted_index:
        return ""
    words: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            words.append((pos, word))
    abstract = " ".join(word for _, word in sorted(words))
    abstract = html.unescape(abstract)
    abstract = re.sub(r"\s+", " ", abstract).strip()
    return abstract


def _get_authors(work: dict[str, Any]) -> list[str]:
    authorships = work.get("authorships") or []
    authors: list[str] = []
    for item in authorships[:6]:
        author = item.get("author") or {}
        name = author.get("display_name")
        if name:
            authors.append(name)
    return authors


def _normalize_work(work: dict[str, Any], rank: int, source_query: str) -> dict[str, Any]:
    title = work.get("title") or work.get("display_name") or "Untitled"
    abstract = _clean_abstract(work.get("abstract_inverted_index"))
    doi = work.get("doi")
    landing_url = (
        work.get("primary_location", {}).get("landing_page_url")
        if work.get("primary_location")
        else None
    )
    return {
        "openalex_id": work.get("id"),
        "title": title,
        "authors": _get_authors(work),
        "year": work.get("publication_year"),
        "publication_date": work.get("publication_date"),
        "abstract": abstract,
        "cited_by_count": work.get("cited_by_count") or 0,
        "doi": doi,
        "url": doi or landing_url or work.get("id"),
        "source_query": source_query,
        "source_api": "OpenAlex",
        "search_text": f"{title} {abstract}",
        "_openalex_rank": rank,
        "referenced_works": work.get("referenced_works") or [],
        "related_works": work.get("related_works") or [],
    }


def _parse_year_filter(time_range: str | None) -> str | None:
    if not time_range or time_range == "not specified":
        return None
    current_year = datetime.now().year

    if time_range == "last 3 years":
        return f"{current_year - 3}-{current_year}"
    if time_range == "last 5 years":
        return f"{current_year - 5}-{current_year}"
    if time_range == "last 10 years":
        return f"{current_year - 10}-{current_year}"

    since_match = re.match(r"since\s+(\d{4})", time_range, re.IGNORECASE)
    if since_match:
        return f"{since_match.group(1)}-{current_year}"

    range_match = re.match(r"(\d{4})-(\d{4})", time_range)
    if range_match:
        return range_match.group(0)

    return None


async def search_openalex(
    query: str,
    per_page: int = 15,
    time_range: str | None = None,
) -> list[dict[str, Any]]:
    """调用 OpenAlex Works API 搜索论文。429 时最多重试 2 次；彻底失败则抛 OpenAlexRateLimitError。"""
    params: dict[str, Any] = {
        "search": query,
        "per-page": per_page,
        # Keep OpenAlex's relevance ordering. Sorting globally by citations
        # floods fine-grained queries with famous but off-topic papers.
        "sort": "relevance_score:desc",
        "select": _SELECT,
    }
    api_key = _next_api_key()
    if api_key:
        params["api_key"] = api_key
    year_range = _parse_year_filter(time_range)
    filters: list[str] = []
    if year_range:
        filters.append(f"publication_year:{year_range}")
    if current_cutoff_iso():
        filters.append(f"to_publication_date:{current_cutoff_iso()}")
    if filters:
        params["filter"] = ",".join(filters)

    timeout = httpx.Timeout(12.0, connect=8.0)
    headers = {"User-Agent": "YanwenZhiso-MVP/0.1 (mailto:jiazixu123@gmail.com)"}

    max_attempts = max(3, len(_OPENALEX_API_KEYS))
    for attempt in range(max_attempts):
        # Small delay to stay within per-second rate limits
        await asyncio.sleep(0.15 * (attempt + 1))
        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            response = await client.get(OPENALEX_WORKS_URL, params=params)

        if response.status_code == 429:
            _mark_key_exhausted(str(params.get("api_key") or ""))
            body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
            retry_after = int(response.headers.get("Retry-After") or body.get("retryAfter") or 3)
            # With a configured key pool, fail over to the next independently
            # provisioned key before falling back to another data source.
            if len(_OPENALEX_API_KEYS) > 1 and attempt < max_attempts - 1:
                params["api_key"] = _next_api_key()
                await asyncio.sleep(0.25)
                continue
            # Large Retry-After means quota exhaustion (e.g. daily budget), not a burst limit.
            # Raise immediately so the caller can fall back to Semantic Scholar.
            if retry_after > 30:
                raise OpenAlexRateLimitError(
                    f"OpenAlex quota exhausted for query '{query[:60]}' (retry_after={retry_after}s)"
                )
            if attempt < max_attempts - 1:
                await asyncio.sleep(max(retry_after, 3))
                continue
            raise OpenAlexRateLimitError(
                f"OpenAlex rate limited for query '{query[:60]}' after 3 attempts"
            )

        response.raise_for_status()
        results = response.json().get("results") or []
        return [
            _normalize_work(work, rank=i + 1, source_query=query)
            for i, work in enumerate(results)
        ]

    raise OpenAlexRateLimitError(f"OpenAlex: exhausted retries for '{query[:60]}'")


async def search_openalex_semantic(query: str, per_page: int = 50) -> list[dict[str, Any]]:
    """Embedding retrieval lane for long query-by-description inputs."""
    global _SEMANTIC_LAST_CALL
    params: dict[str, Any] = {
        "search.semantic": query[:2000],
        "per_page": min(per_page, 50),
        "select": _SELECT,
    }
    # OpenAlex currently rejects `filter` when it is combined with
    # `search.semantic` (HTTP 400). Evaluation mode therefore applies the
    # same exact cutoff immediately after this call, before fusion/ranking.
    api_key = _next_api_key()
    if api_key:
        params["api_key"] = api_key
    headers = {"User-Agent": "YanwenZhiso/1.2 (mailto:jiazixu123@gmail.com)"}
    async with _SEMANTIC_LOCK:
        wait_for = 1.05 - (asyncio.get_running_loop().time() - _SEMANTIC_LAST_CALL)
        if wait_for > 0:
            await asyncio.sleep(wait_for)
        async with httpx.AsyncClient(timeout=httpx.Timeout(25.0, connect=8.0), headers=headers) as client:
            response = await client.get(OPENALEX_WORKS_URL, params=params)
            if response.status_code == 429 and len(_OPENALEX_API_KEYS) > 1:
                _mark_key_exhausted(str(params.get("api_key") or ""))
                next_key = _next_api_key()
                if next_key:
                    params["api_key"] = next_key
                    response = await client.get(OPENALEX_WORKS_URL, params=params)
        _SEMANTIC_LAST_CALL = asyncio.get_running_loop().time()
    if response.status_code == 429:
        _mark_key_exhausted(str(params.get("api_key") or ""))
        raise OpenAlexRateLimitError("OpenAlex semantic search rate limited")
    response.raise_for_status()
    papers = [
        _normalize_work(work, rank=i + 1, source_query=f"semantic:{query}")
        for i, work in enumerate(response.json().get("results") or [])
    ]
    for paper in papers:
        paper["source_api"] = "OpenAlex Semantic"
        paper["retrieval_lane"] = "semantic"
    return papers


async def fetch_openalex_works(work_ids: list[str], source_query: str, limit: int = 40) -> list[dict[str, Any]]:
    """Resolve a batch of OpenAlex work IDs for citation-neighborhood expansion."""
    ids: list[str] = []
    for value in work_ids:
        wid = str(value or "").rsplit("/", 1)[-1]
        if re.fullmatch(r"W\d+", wid) and wid not in ids:
            ids.append(wid)
        if len(ids) >= limit:
            break
    if not ids:
        return []
    params: dict[str, Any] = {
        "filter": "openalex_id:" + "|".join(ids),
        "per-page": min(len(ids), 100),
        "select": _SELECT,
    }
    if current_cutoff_iso():
        params["filter"] += f",to_publication_date:{current_cutoff_iso()}"
    api_key = _next_api_key()
    if api_key:
        params["api_key"] = api_key
    headers = {"User-Agent": "YanwenZhiso/1.1 (mailto:jiazixu123@gmail.com)"}
    async with httpx.AsyncClient(timeout=httpx.Timeout(18.0, connect=8.0), headers=headers) as client:
        response = await client.get(OPENALEX_WORKS_URL, params=params)
    if response.status_code == 429:
        _mark_key_exhausted(str(params.get("api_key") or ""))
        raise OpenAlexRateLimitError("OpenAlex citation expansion rate limited")
    response.raise_for_status()
    return [
        _normalize_work(work, rank=i + 1, source_query=source_query)
        for i, work in enumerate(response.json().get("results") or [])
    ]


async def fetch_openalex_citing(work_id: str, source_query: str, per_page: int = 15) -> list[dict[str, Any]]:
    """Fetch papers that cite a seed work."""
    wid = str(work_id or "").rsplit("/", 1)[-1]
    if not re.fullmatch(r"W\d+", wid):
        return []
    params: dict[str, Any] = {
        "filter": f"cites:{wid}",
        "per-page": per_page,
        "sort": "cited_by_count:desc",
        "select": _SELECT,
    }
    if current_cutoff_iso():
        params["filter"] += f",to_publication_date:{current_cutoff_iso()}"
    api_key = _next_api_key()
    if api_key:
        params["api_key"] = api_key
    headers = {"User-Agent": "YanwenZhiso/1.1 (mailto:jiazixu123@gmail.com)"}
    async with httpx.AsyncClient(timeout=httpx.Timeout(18.0, connect=8.0), headers=headers) as client:
        response = await client.get(OPENALEX_WORKS_URL, params=params)
    if response.status_code == 429:
        _mark_key_exhausted(str(params.get("api_key") or ""))
        raise OpenAlexRateLimitError("OpenAlex citing expansion rate limited")
    response.raise_for_status()
    return [
        _normalize_work(work, rank=i + 1, source_query=source_query)
        for i, work in enumerate(response.json().get("results") or [])
    ]


async def fetch_openalex_citing_any(
    work_ids: list[str], query: str, source_query: str, per_page: int = 50,
) -> list[dict[str, Any]]:
    """Fetch works that cite any seed, constrained by a query in one call.

    OpenAlex supports OR values within one filter using ``|``.  Batching the
    citation frontier avoids one API request per seed while the text query
    keeps broad, highly cited seeds from flooding the result set.
    """
    ids: list[str] = []
    for value in work_ids:
        wid = str(value or "").rsplit("/", 1)[-1]
        if re.fullmatch(r"W\d+", wid) and wid not in ids:
            ids.append(wid)
        if len(ids) >= 50:
            break
    if not ids or not str(query or "").strip():
        return []
    params: dict[str, Any] = {
        "filter": "cites:" + "|".join(ids),
        "search": str(query).strip()[:800],
        "per-page": min(max(1, per_page), 100),
        "sort": "relevance_score:desc",
        "select": _SELECT,
    }
    if current_cutoff_iso():
        params["filter"] += f",to_publication_date:{current_cutoff_iso()}"
    api_key = _next_api_key()
    if api_key:
        params["api_key"] = api_key
    headers = {"User-Agent": "YanwenZhiso/1.1 (mailto:jiazixu123@gmail.com)"}
    timeout = httpx.Timeout(18.0, connect=8.0)
    max_attempts = max(2, len(_OPENALEX_API_KEYS))
    for attempt in range(max_attempts):
        await asyncio.sleep(0.15 * (attempt + 1))
        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            response = await client.get(OPENALEX_WORKS_URL, params=params)
        if response.status_code != 429:
            response.raise_for_status()
            return [
                _normalize_work(work, rank=i + 1, source_query=source_query)
                for i, work in enumerate(response.json().get("results") or [])
            ]
        _mark_key_exhausted(str(params.get("api_key") or ""))
        if attempt < max_attempts - 1:
            next_key = _next_api_key()
            if next_key:
                params["api_key"] = next_key
                continue
        raise OpenAlexRateLimitError("OpenAlex batched citation frontier rate limited")
    return []
