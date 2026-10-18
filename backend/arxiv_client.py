"""arXiv API client (secondary academic source, no API key required).

arXiv has excellent coverage of CS/AI papers and returns **full abstracts**,
which the Crossref fallback often lacks. It has no citation counts, so the
citation dimension of the ranker is 0 for arXiv papers; relevance, keyword and
freshness still apply. Used as a fallback when OpenAlex is rate-limited.
"""

from __future__ import annotations

import asyncio
import html
import re
from typing import Any
from xml.etree import ElementTree as ET

import httpx

from backend.eval_runtime import current_cutoff


ARXIV_API_URL = "https://export.arxiv.org/api/query"
_ATOM = "{http://www.w3.org/2005/Atom}"
_QUERY_STOPWORDS = {
    "a", "an", "and", "any", "are", "about", "based", "can", "could", "for",
    "from", "have", "in", "information", "me", "of", "on", "paper", "papers",
    "provide", "resources", "some", "studies", "study", "tell", "that", "the",
    "there", "to", "use", "using", "what", "which", "with", "works", "algorithm",
    "application", "applications", "method", "methods", "technique", "techniques",
}


def _clean(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _structured_query(query: str) -> str:
    """Translate free text to valid arXiv boolean syntax.

    `all:foo bar baz` only scopes the first token in arXiv's API and caused very
    noisy results. Explicit AND clauses make all selected concepts searchable.
    """
    title_match = re.match(r'^title:\s*["“”]?(.+?)["“”]?$', query.strip(), re.IGNORECASE)
    if title_match:
        title = title_match.group(1).strip().strip('"“”')
        return f'ti:"{title}"'
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9+.#-]*", query)
    terms: list[str] = []
    for token in tokens:
        low = token.lower()
        if low in _QUERY_STOPWORDS or len(low) < 2 or low in terms:
            continue
        terms.append(low)
    terms = terms[:6]
    if not terms:
        return f'all:"{query}"'
    return " AND ".join(f'all:"{term}"' for term in terms)


def _normalize_entry(entry: ET.Element, rank: int, source_query: str) -> dict[str, Any]:
    title = _clean(entry.findtext(f"{_ATOM}title"))
    abstract = _clean(entry.findtext(f"{_ATOM}summary"))
    published = entry.findtext(f"{_ATOM}published") or ""
    year = None
    m = re.match(r"(\d{4})", published)
    if m:
        year = int(m.group(1))

    authors: list[str] = []
    for a in entry.findall(f"{_ATOM}author")[:6]:
        name = _clean(a.findtext(f"{_ATOM}name"))
        if name:
            authors.append(name)

    arxiv_id = (entry.findtext(f"{_ATOM}id") or "").strip()

    return {
        "openalex_id": arxiv_id or f"arxiv:rank{rank}",
        "title": title or "Untitled",
        "authors": authors,
        "year": year,
        "publication_date": published[:10] or None,
        "abstract": abstract,
        "cited_by_count": 0,  # arXiv does not expose citation counts
        "doi": None,
        "url": arxiv_id or None,
        "source_query": source_query,
        "source_api": "arXiv",
        "search_text": f"{title} {abstract}",
        "_openalex_rank": rank,
    }


async def search_arxiv(
    query: str,
    per_page: int = 15,
    time_range: str | None = None,  # accepted for interface parity; not used
) -> list[dict[str, Any]]:
    """Search arXiv with both precise AND and broad parser-compatible queries."""
    timeout = httpx.Timeout(6.0, connect=4.0)
    headers = {"User-Agent": "YanwenZhiso-MVP/0.1 (mailto:jiazixu123@gmail.com)"}

    async def fetch(expression: str) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "search_query": expression,
            "start": 0,
            "max_results": per_page,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        for attempt in range(1):
            async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
                response = await client.get(ARXIV_API_URL, params=params)
            if response.status_code in (429, 503):
                return []
            response.raise_for_status()
            try:
                entries = ET.fromstring(response.text).findall(f"{_ATOM}entry")
            except ET.ParseError:
                return []
            return [_normalize_entry(e, rank=i + 1, source_query=query) for i, e in enumerate(entries)]
        return []

    expression = _structured_query(query) if query.strip().lower().startswith("title:") else f"all:{query}"
    cutoff = current_cutoff()
    if cutoff:
        # arXiv's submittedDate parser expects a valid YYYYMMDDHHMM lower
        # bound; an all-zero month/day is accepted as text but yields no hits.
        expression = f"({expression}) AND submittedDate:[199001010000 TO {cutoff.strftime('%Y%m%d')}2359]"
    return await fetch(expression)
