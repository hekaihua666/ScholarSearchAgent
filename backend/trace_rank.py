"""TRACE-Rank: constraint-aware fusion and diversity ranking.

The module is deliberately dependency-free so the live demo still works on a
CPU-only laptop.  It combines evidence from repeated retrieval hits instead of
trusting one provider's opaque rank, then applies query-constraint coverage and
MMR-style diversity selection.
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any

from backend.paper_identity import normalize_title


_STOPWORDS = {
    "the", "and", "for", "from", "with", "into", "using", "based", "paper",
    "study", "research", "method", "methods", "model", "models", "what",
    "which", "how", "are", "not", "specified", "application", "survey",
}

_GENERIC_RESULT_WORDS = {"survey", "review", "overview", "tutorial", "introduction"}


def canonical_key(paper: dict[str, Any]) -> str:
    doi = str(paper.get("doi") or "").lower().removeprefix("https://doi.org/").strip()
    if doi:
        return f"doi:{doi}"
    title = normalize_title(str(paper.get("title") or "").replace("\\n", " "))
    return f"title:{title}" if title else str(paper.get("openalex_id") or "")


def tokenize(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", text.lower())
        if token not in _STOPWORDS
    }


def fuse_candidates(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge duplicates while retaining every query/source/rank retrieval hit."""
    fused: dict[str, dict[str, Any]] = {}
    aliases: dict[str, str] = {}
    for raw in papers:
        title_alias = "title:" + normalize_title(str(raw.get("title") or "").replace("\\n", " "))
        doi = str(raw.get("doi") or "").lower().removeprefix("https://doi.org/").strip()
        doi_alias = f"doi:{doi}" if doi else ""
        key = aliases.get(doi_alias) or aliases.get(title_alias) or canonical_key(raw)
        if not key:
            continue
        rank = int(raw.get("_openalex_rank") or raw.get("_source_rank") or 100)
        hit = {
            "query": str(raw.get("source_query") or ""),
            "source": str(raw.get("source_api") or "unknown"),
            "rank": rank,
            "round": str(raw.get("source_round") or "unknown"),
        }
        if key not in fused:
            paper = dict(raw)
            paper["retrieval_hits"] = [hit]
            fused[key] = paper
            if title_alias != "title:":
                aliases[title_alias] = key
            if doi_alias:
                aliases[doi_alias] = key
            continue
        paper = fused[key]
        known = {(h["query"], h["source"], h["rank"]) for h in paper["retrieval_hits"]}
        if (hit["query"], hit["source"], hit["rank"]) not in known:
            paper["retrieval_hits"].append(hit)
        # Prefer richer metadata rather than whichever provider happened to run first.
        for field in ("abstract", "authors", "doi", "url", "year", "cited_by_count"):
            old, new = paper.get(field), raw.get(field)
            if not old or (field == "abstract" and len(str(new or "")) > len(str(old or ""))):
                paper[field] = new
        rounds = {h["round"] for h in paper["retrieval_hits"]}
        paper["source_round"] = "both" if {"first_round", "second_round"} <= rounds else paper.get("source_round")
        if title_alias != "title:":
            aliases[title_alias] = key
        if doi_alias:
            aliases[doi_alias] = key
    return list(fused.values())


def _query_facets(understanding: dict[str, Any]) -> list[tuple[str, set[str]]]:
    facets: list[tuple[str, set[str]]] = []
    for name in ("research_topic", "method_constraints", "task_constraints", "domain_constraints"):
        tokens = tokenize(str(understanding.get(name) or ""))
        if tokens:
            facets.append((name, tokens))
    return facets


def _lexical_score(paper: dict[str, Any], query_tokens: set[str]) -> float:
    if not query_tokens:
        return 0.0
    title = tokenize(str(paper.get("title") or ""))
    abstract = tokenize(str(paper.get("abstract") or ""))
    weighted_match = 2.0 * len(query_tokens & title) + len(query_tokens & abstract)
    return min(weighted_match / max(2.0 * len(query_tokens), 1.0), 1.0)


def _idf_lexical_scores(papers: list[dict[str, Any]], query_tokens: set[str]) -> dict[int, float]:
    """Corpus-local IDF matching with extra weight for title matches."""
    if not papers or not query_tokens:
        return {id(p): 0.0 for p in papers}
    documents = [tokenize(f"{p.get('title', '')} {p.get('abstract', '')}") for p in papers]
    n = len(documents)
    df = {token: sum(token in doc for doc in documents) for token in query_tokens}
    weights = {token: math.log((n + 1) / (df[token] + 1)) + 1.0 for token in query_tokens}
    denominator = sum(weights.values()) or 1.0
    scores: dict[int, float] = {}
    for paper, document in zip(papers, documents):
        title = tokenize(str(paper.get("title") or ""))
        value = sum(weights[t] * (1.6 if t in title else 1.0)
                    for t in query_tokens if t in document)
        scores[id(paper)] = min(value / (1.6 * denominator), 1.0)
    return scores


def _hypothesis_score(title: str, hypotheses: list[str]) -> float:
    title_tokens = tokenize(title)
    best = 0.0
    for hypothesis in hypotheses:
        tokens = tokenize(hypothesis)
        union = title_tokens | tokens
        if union:
            best = max(best, len(title_tokens & tokens) / len(union))
    return best


def _rrf_score(hits: list[dict[str, Any]], k: int = 60) -> float:
    unique = {(h.get("query"), h.get("source"), int(h.get("rank") or 100)) for h in hits}
    raw = sum(1.0 / (k + rank) for _, _, rank in unique)
    # Four top-1 hits are already strong evidence; cap beyond that.
    return min(raw / (4.0 / (k + 1)), 1.0)


def _coverage_score(text_tokens: set[str], facets: list[tuple[str, set[str]]]) -> tuple[float, dict[str, float]]:
    if not facets:
        return 0.0, {}
    detail = {
        name: min(len(tokens & text_tokens) / max(len(tokens), 1), 1.0)
        for name, tokens in facets
    }
    return sum(detail.values()) / len(detail), detail


def _authority_score(citations: int, year: int | None, current_year: int | None = None) -> float:
    reference_year = int(current_year or datetime.now().year)
    age = max(reference_year - int(year or reference_year), 0)
    annualized = max(citations, 0) / max(age + 1, 1)
    return min(math.log1p(annualized) / math.log1p(100), 1.0)


def score_candidates(
    papers: list[dict[str, Any]], understanding: dict[str, Any]
) -> list[dict[str, Any]]:
    facets = _query_facets(understanding)
    query_tokens = set().union(*(tokens for _, tokens in facets)) if facets else set()
    query_tokens |= tokenize(str(understanding.get("original_query") or ""))
    hypotheses = [str(x) for x in (understanding.get("candidate_title_queries") or []) if str(x).strip()]
    idf_scores = _idf_lexical_scores(papers, query_tokens)
    asks_for_review = bool(_GENERIC_RESULT_WORDS & tokenize(str(understanding.get("original_query") or "")))
    current_year = int(understanding.get("_evaluation_year") or datetime.now().year)
    for paper in papers:
        text_tokens = tokenize(f"{paper.get('title', '')} {paper.get('abstract', '')}")
        lexical = _lexical_score(paper, query_tokens)
        idf_lexical = idf_scores.get(id(paper), 0.0)
        hypothesis = _hypothesis_score(str(paper.get("title") or ""), hypotheses)
        rrf = _rrf_score(paper.get("retrieval_hits") or [])
        coverage, coverage_detail = _coverage_score(text_tokens, facets)
        authority = _authority_score(
            int(paper.get("cited_by_count") or 0), paper.get("year"), current_year
        )
        age = max(current_year - int(paper.get("year") or 0), 0) if paper.get("year") else 20
        freshness = max(0.0, 1.0 - age / 12.0)
        metadata = (bool(paper.get("abstract")) + bool(paper.get("doi")) + bool(paper.get("authors"))) / 3.0
        generic_penalty = 0.06 if (not asks_for_review and _GENERIC_RESULT_WORDS & tokenize(str(paper.get("title") or ""))) else 0.0
        citation_provenance = 0.05 if str(paper.get("source_query") or "").startswith("citation:references") else 0.0
        semantic_ranks = [
            int(hit.get("rank") or 100) for hit in (paper.get("retrieval_hits") or [])
            if str(hit.get("source") or "") == "OpenAlex Semantic"
        ]
        semantic_provenance = (1.0 / (1.0 + min(semantic_ranks) / 10.0)) if semantic_ranks else 0.0
        intent = str(understanding.get("search_intent") or "broad")
        intent_bonus = 0.0
        if intent == "foundational":
            historical = min(age / 12.0, 1.0)
            intent_bonus = 0.06 * authority + 0.04 * historical + 0.04 * citation_provenance
        elif intent == "frontier":
            intent_bonus = 0.10 * freshness
        elif intent == "specific_paper":
            intent_bonus = 0.08 * hypothesis + 0.05 * semantic_provenance
        elif intent == "method_lineage":
            intent_bonus = 0.05 * citation_provenance + 0.04 * rrf
        elif intent == "comparison":
            intent_bonus = 0.06 * coverage
        score = (
            0.25 * lexical + 0.18 * idf_lexical + 0.20 * rrf + 0.14 * coverage
            + 0.08 * hypothesis + 0.07 * authority + 0.04 * freshness + 0.04 * metadata
            + 0.10 * semantic_provenance + citation_provenance + intent_bonus - generic_penalty
        )
        paper.update({
            "lexical_score": round(lexical, 4),
            "idf_lexical_score": round(idf_lexical, 4),
            "title_hypothesis_score": round(hypothesis, 4),
            "generic_result_penalty": round(generic_penalty, 4),
            "citation_provenance_score": round(citation_provenance, 4),
            "semantic_provenance_score": round(semantic_provenance, 4),
            "intent_bonus_score": round(intent_bonus, 4),
            "rrf_score": round(rrf, 4),
            "constraint_coverage_score": round(coverage, 4),
            "constraint_coverage": {k: round(v, 4) for k, v in coverage_detail.items()},
            "authority_score": round(authority, 4),
            "freshness_score": round(freshness, 4),
            "metadata_score": round(metadata, 4),
            # Backward-compatible aliases used by the current frontend.
            "keyword_score": round(lexical, 4),
            "openalex_score": round(rrf, 4),
            "citation_score": round(authority, 4),
            "final_score": round(score, 4),
            "relevance_score": round(score, 4),
        })
    return sorted(papers, key=lambda p: float(p.get("final_score") or 0), reverse=True)


def diversify(papers: list[dict[str, Any]], limit: int = 20, lambda_relevance: float = 0.82) -> list[dict[str, Any]]:
    """Greedy MMR-style selection to reduce near-duplicate methods and surveys."""
    remaining = list(papers)
    selected: list[dict[str, Any]] = []
    while remaining and len(selected) < limit:
        best, best_value = None, -1.0
        for candidate in remaining:
            c_tokens = tokenize(f"{candidate.get('title', '')} {candidate.get('abstract', '')}")
            redundancy = 0.0
            for chosen in selected:
                s_tokens = tokenize(f"{chosen.get('title', '')} {chosen.get('abstract', '')}")
                union = c_tokens | s_tokens
                redundancy = max(redundancy, len(c_tokens & s_tokens) / len(union) if union else 0.0)
            value = lambda_relevance * float(candidate.get("final_score") or 0) - (1 - lambda_relevance) * redundancy
            if value > best_value:
                best, best_value = candidate, value
        assert best is not None
        best["diversity_adjusted_score"] = round(best_value, 4)
        selected.append(best)
        remaining.remove(best)
    return selected


def build_relation_graph(papers: list[dict[str, Any]]) -> dict[str, Any]:
    nodes = [
        {"id": canonical_key(p), "title": p.get("title"), "year": p.get("year"), "score": p.get("final_score")}
        for p in papers
    ]
    edges: list[dict[str, Any]] = []
    for i, left in enumerate(papers):
        lt = tokenize(f"{left.get('title', '')} {left.get('abstract', '')}")
        for right in papers[i + 1:]:
            rt = tokenize(f"{right.get('title', '')} {right.get('abstract', '')}")
            union = lt | rt
            similarity = len(lt & rt) / len(union) if union else 0.0
            if similarity >= 0.12:
                edges.append({"source": canonical_key(left), "target": canonical_key(right), "type": "semantic", "weight": round(similarity, 4)})
    return {"nodes": nodes, "edges": sorted(edges, key=lambda e: e["weight"], reverse=True)[:30]}
