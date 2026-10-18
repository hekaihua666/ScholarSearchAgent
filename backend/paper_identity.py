"""Stable paper identity helpers for retrieval evaluation.

Evaluation prefers persistent identifiers and uses title normalization only as
fallback.  Fuzzy titles are exposed for diagnosis but are not counted as hits.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Iterable


_ARXIV_RE = re.compile(
    r"(?<!\d)(?:arxiv(?:\.org/(?:abs|pdf|html)/|[.:\s])?)?"
    r"((?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[a-z-]+)?/\d{7}))(?:v\d+)?",
    re.IGNORECASE,
)


def normalize_arxiv_id(value: Any) -> str:
    text = str(value or "").strip()
    match = _ARXIV_RE.search(text)
    return match.group(1).lower() if match else ""


def normalize_doi(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:\s*", "", text)
    match = re.search(r"10\.\d{4,9}/[^\s?#]+", text)
    return match.group(0).rstrip(".,;)") if match else ""


def normalize_openalex_id(value: Any) -> str:
    match = re.search(r"(?:openalex\.org/)?(W\d+)", str(value or ""), re.I)
    return match.group(1).upper() if match else ""


def normalize_s2_id(value: Any) -> str:
    text = str(value or "").strip()
    if re.fullmatch(r"[0-9a-f]{40}", text, re.I):
        return text.lower()
    match = re.search(r"semanticscholar\.org/(?:paper/)?(?:[^/]+/)?([0-9a-f]{40})", text, re.I)
    return match.group(1).lower() if match else ""


def normalize_title(value: Any) -> str:
    """Unicode-aware alphanumeric title key used by the local evaluator."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().replace("\\n", " ")
    return "".join(ch for ch in text if ch.isalnum())


def official_pasa_title(value: Any) -> str:
    """Mirror PaSa's released ``keep_letters`` title matching rule."""
    return "".join(ch for ch in str(value or "") if ch.isalpha()).casefold()


@dataclass(frozen=True)
class PaperIdentity:
    arxiv_ids: frozenset[str]
    dois: frozenset[str]
    openalex_ids: frozenset[str]
    semantic_scholar_ids: frozenset[str]
    title: str
    pasa_title: str
    raw_title: str


def _values(paper: dict[str, Any], names: Iterable[str]) -> list[Any]:
    values: list[Any] = []
    for name in names:
        value = paper.get(name)
        if isinstance(value, dict):
            values.extend(value.values())
        elif isinstance(value, (list, tuple, set)):
            values.extend(value)
        elif value:
            values.append(value)
    return values


def paper_identity(paper: dict[str, Any]) -> PaperIdentity:
    candidates = _values(
        paper,
        ("arxiv_id", "answer_arxiv_id", "doi", "url", "openalex_id", "paper_id", "external_ids", "ids"),
    )
    arxiv_ids = frozenset(filter(None, (normalize_arxiv_id(v) for v in candidates)))
    dois = frozenset(filter(None, (normalize_doi(v) for v in candidates)))
    openalex_ids = frozenset(filter(None, (normalize_openalex_id(v) for v in candidates)))
    s2_ids = frozenset(filter(None, (normalize_s2_id(v) for v in candidates)))
    title = str(paper.get("title") or "")
    return PaperIdentity(
        arxiv_ids=arxiv_ids,
        dois=dois,
        openalex_ids=openalex_ids,
        semantic_scholar_ids=s2_ids,
        title=normalize_title(title),
        pasa_title=official_pasa_title(title),
        raw_title=title,
    )


def match_identity(predicted: dict[str, Any], gold: dict[str, Any]) -> str | None:
    """Return the deterministic match route, or ``None``.

    Fuzzy title similarity is intentionally not accepted here.
    """
    pred = paper_identity(predicted)
    target = paper_identity(gold)
    if pred.arxiv_ids & target.arxiv_ids:
        return "arxiv_id"
    if pred.dois & target.dois:
        return "doi"
    if pred.openalex_ids & target.openalex_ids:
        return "openalex_id"
    if pred.semantic_scholar_ids & target.semantic_scholar_ids:
        return "semantic_scholar_id"
    if pred.title and pred.title == target.title:
        return "normalized_title"
    if pred.pasa_title and pred.pasa_title == target.pasa_title:
        return "official_pasa_title"
    return None


def fuzzy_title_similarity(left: Any, right: Any) -> float:
    a, b = normalize_title(left), normalize_title(right)
    return SequenceMatcher(None, a, b).ratio() if a and b else 0.0

