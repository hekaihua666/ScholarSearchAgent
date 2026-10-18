"""Seed-Conditioned Graph Query for grouped reverse citation retrieval."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Sequence

from backend.eval_runtime import filter_for_cutoff, frozen_call
from backend.openalex_client import fetch_openalex_citing_any
from backend.paper_identity import normalize_openalex_id
from backend.retrieval.grouped_citation_frontier_v13 import GROUP_SIZE, PER_GROUP, partition_seeds
from backend.retrieval.query_conditioned_citation_v11 import _annotate_reverse, select_frontier_seeds
from backend.trace_rank import fuse_candidates, score_candidates


_GENERIC = {
    "paper", "papers", "work", "works", "study", "studies", "research",
    "method", "methods", "approach", "approaches", "classification",
    "using", "used", "based", "field", "task", "tasks", "different",
    "what", "which", "who", "could", "would", "can", "you", "discuss",
}


def _tokens(text: Any) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9-]+", str(text or ""))]


def build_seed_conditioned_query(
    query: str, understanding: dict[str, Any], seeds: Sequence[dict[str, Any]], max_terms: int = 6,
) -> str:
    base = [token for token in _tokens(understanding.get("research_topic")) if token not in _GENERIC]
    if not base:
        base = [token for token in _tokens(query) if token not in _GENERIC]
    base = list(dict.fromkeys(base))[:max_terms]
    title_tokens = [_tokens(seed.get("title")) for seed in seeds[:12]]
    frequency = Counter(token for row in title_tokens for token in set(row) if token not in _GENERIC)
    base_set = set(base)

    best: tuple[float, list[str]] | None = None
    for row in title_tokens:
        overlap = len(base_set & set(row))
        if overlap < min(2, len(base_set)):
            continue
        projected = [
            token for token in row
            if token in base_set or (frequency[token] >= 2 and token not in _GENERIC)
        ]
        projected = list(dict.fromkeys(projected))[:max_terms]
        score = 3.0 * overlap + sum(min(frequency[token], 4) for token in projected) / 10.0
        if len(projected) >= 2 and (best is None or score > best[0]):
            best = (score, projected)
    if best:
        return " ".join(best[1])
    consensus = [
        token for token, count in frequency.most_common()
        if count >= 2 and token not in base_set and token not in _GENERIC
    ]
    return " ".join([*base, *consensus[:max(0, max_terms - len(base))]])


async def retrieve_seed_conditioned_reverse_frontier(
    query: str,
    understanding: dict[str, Any],
    contract: dict[str, Any],
    baseline_top20: Sequence[dict[str, Any]],
    candidate_pool: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    del contract  # v15 deliberately tests candidate-conditioned query compression.
    seeds = select_frontier_seeds(baseline_top20, candidate_pool)
    graph_query = build_seed_conditioned_query(query, understanding, seeds)
    groups = partition_seeds(seeds, GROUP_SIZE)
    lane_results: dict[str, list[dict[str, Any]]] = {}
    errors: list[dict[str, str]] = []
    for index, group in enumerate(groups, 1):
        lane = f"reverse_group_{index}"
        ids = [str(seed.get("openalex_id") or "") for seed in group]
        try:
            papers = await frozen_call(
                f"scgq15_{lane}",
                {"seed_ids": [normalize_openalex_id(value) for value in ids],
                 "query": graph_query, "per_page": PER_GROUP},
                lambda ids=ids, lane=lane: fetch_openalex_citing_any(
                    ids, graph_query, source_query=f"scgq15:{lane}", per_page=PER_GROUP,
                ),
            )
            lane_results[lane] = _annotate_reverse(
                filter_for_cutoff(papers, f"scgq15_{lane}"), group, lane
            )
        except Exception as exc:
            lane_results[lane] = []
            errors.append({"lane": lane, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
    candidates = score_candidates(
        fuse_candidates([paper for rows in lane_results.values() for paper in rows]), understanding
    )
    return {
        "seeds": [{
            "openalex_id": seed.get("openalex_id"), "title": seed.get("title"),
            "seed_group": index // GROUP_SIZE + 1,
        } for index, seed in enumerate(seeds)],
        "queries": [{"lane": "seed_conditioned_reverse", "query": graph_query}],
        "lanes": lane_results,
        "candidates": candidates,
        "errors": errors,
        "api_calls": len(groups),
    }
