"""Crossref REST API client (secondary academic source, no API key required).

Crossref indexes 150M+ scholarly works and has a very generous "polite pool"
(just include a contact email in the User-Agent, which we do). It does not have
the daily-budget exhaustion problem of the OpenAlex free tier, so it is a
reliable fallback when OpenAlex is rate-limited.
"""

from __future__ import annotations

import asyncio
import html
import re
from datetime import datetime
from typing import Any

import httpx

from backend.eval_runtime import current_cutoff_iso


CROSSREF_WORKS_URL = "https://api.crossref.org/works"
# Only request the fields we use — keeps responses small and fast.
_SELECT = "DOI,title,author,published,published-print,published-online,abstract,is-referenced-by-count,URL"


def _clean_abstract(raw: str | None) -> str:
    if not raw:
        return ""
    # Crossref abstracts are JATS XML, e.g. "<jats:p>...</jats:p>".
    text = re.sub(r"<[^>]+>", " ", raw)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _get_authors(work: dict[str, Any]) -> list[str]:
    authors: list[str] = []
    for a in (work.get("author") or [])[:6]:
        name = " ".join(p for p in [a.get("given"), a.get("family")] if p).strip()
        if not name:
            name = a.get("name", "")
        if name:
            authors.append(name)
    return authors


def _get_year(work: dict[str, Any]) -> int | None:
    for key in ("published", "published-print", "published-online", "issued"):
        block = work.get(key) or {}
        parts = block.get("date-parts") or []
        if parts and parts[0] and parts[0][0]:
            return parts[0][0]
    return None


def _get_publication_date(work: dict[str, Any]) -> str | None:
    for key in ("published", "published-online", "published-print", "issued"):
        block = work.get(key) or {}
        parts = block.get("date-parts") or []
        if not parts or not parts[0]:
            continue
        values = list(parts[0])
        try:
            year = int(values[0])
            month = int(values[1]) if len(values) > 1 else 1
            day = int(values[2]) if len(values) > 2 else 1
            return f"{year:04d}-{month:02d}-{day:02d}"
        except (TypeError, ValueError):
            continue
    return None


def _normalize_work(work: dict[str, Any], rank: int, source_query: str) -> dict[str, Any]:
    title_list = work.get("title") or []
    title = (title_list[0] if title_list else "Untitled") or "Untitled"
    abstract = _clean_abstract(work.get("abstract"))
    doi = work.get("DOI")
    url = (f"https://doi.org/{doi}" if doi else None) or work.get("URL")
    return {
        "openalex_id": f"crossref:{doi}" if doi else f"crossref:rank{rank}",
        "title": title,
        "authors": _get_authors(work),
        "year": _get_year(work),
        "publication_date": _get_publication_date(work),
        "abstract": abstract,
        "cited_by_count": work.get("is-referenced-by-count") or 0,
        "doi": doi,
        "url": url,
        "source_query": source_query,
        "source_api": "Crossref",
        "search_text": f"{title} {abstract}",
        "_openalex_rank": rank,
    }


def _parse_year_filter(time_range: str | None) -> str | None:
    """Return a Crossref 'from-pub-date:..,until-pub-date:..' filter, or None."""
    if not time_range or time_range == "not specified":
        return None
    current_year = datetime.now().year
    start = end = None

    if time_range == "last 3 years":
        start, end = current_year - 3, current_year
    elif time_range == "last 5 years":
        start, end = current_year - 5, current_year
    elif time_range == "last 10 years":
        start, end = current_year - 10, current_year
    else:
        since_match = re.match(r"since\s+(\d{4})", time_range, re.IGNORECASE)
        range_match = re.match(r"(\d{4})-(\d{4})", time_range)
        if since_match:
            start, end = int(since_match.group(1)), current_year
        elif range_match:
            start, end = int(range_match.group(1)), int(range_match.group(2))

    if start is None:
        return None
    return f"from-pub-date:{start}-01-01,until-pub-date:{end}-12-31"


async def search_crossref(
    query: str,
    per_page: int = 15,
    time_range: str | None = None,
) -> list[dict[str, Any]]:
    """Search Crossref Works. 429 retried up to 2 times; returns [] on hard failure."""
    # Default (relevance) sort: Crossref ranks by query relevance for free, which
    # gives on-topic candidates. Our own ranker then re-scores by keyword match,
    # citations and freshness — so we don't want a pure citation sort here (it
    # floods results with globally-famous but off-topic papers).
    params: dict[str, Any] = {
        "query": query,
        "rows": per_page,
        "select": _SELECT,
    }
    filters: list[str] = []
    year_filter = _parse_year_filter(time_range)
    if year_filter:
        filters.append(year_filter)
    if current_cutoff_iso():
        filters.append(f"until-pub-date:{current_cutoff_iso()}")
    if filters:
        params["filter"] = ",".join(filters)

    timeout = httpx.Timeout(14.0, connect=8.0)
    # Polite-pool: contact email in User-Agent earns higher, more reliable limits.
    headers = {"User-Agent": "YanwenZhiso-MVP/0.1 (mailto:jiazixu123@gmail.com)"}

    for attempt in range(3):
        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            response = await client.get(CROSSREF_WORKS_URL, params=params)

        if response.status_code == 429:
            if attempt < 2:
                await asyncio.sleep(max(int(response.headers.get("Retry-After", 2)), 2))
                continue
            return []

        response.raise_for_status()
        items = ((response.json() or {}).get("message") or {}).get("items") or []
        return [
            _normalize_work(work, rank=i + 1, source_query=query)
            for i, work in enumerate(items)
        ]

    return []
