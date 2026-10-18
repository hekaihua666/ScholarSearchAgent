"""Coverage-Balanced Query-Conditioned Citation Frontier (QCF v12)."""

from __future__ import annotations

from typing import Any, Sequence

from backend.eval_runtime import filter_for_cutoff, frozen_call
from backend.openalex_client import fetch_openalex_citing_any, fetch_openalex_works
from backend.paper_identity import normalize_openalex_id
from backend.retrieval.query_conditioned_citation_v11 import (
    _annotate_reference,
    _annotate_reverse,
    build_frontier_queries,
    select_frontier_seeds,
)
from backend.trace_rank import fuse_candidates, score_candidates


MAX_FRONTIER_IDS = 100
PER_REVERSE_LANE = 100


def round_robin_neighbor_ids(
    seeds: Sequence[dict[str, Any]], field: str, limit: int = MAX_FRONTIER_IDS,
) -> list[str]:
    """Allocate a graph-ID budget across seeds before increasing depth."""
    neighborhoods: list[list[str]] = []
    for seed in seeds:
        values: list[str] = []
        for raw in seed.get(field) or []:
            wid = normalize_openalex_id(raw)
            if wid and wid not in values:
                values.append(wid)
        neighborhoods.append(values)
    output: list[str] = []
    depth = 0
    while len(output) < limit and any(depth < len(values) for values in neighborhoods):
        for values in neighborhoods:
            if depth < len(values) and values[depth] not in output:
                output.append(values[depth])
                if len(output) >= limit:
                    break
        depth += 1
    return output


def _annotate_related(
    papers: Sequence[dict[str, Any]], seeds: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = _annotate_reference(papers, [
        {**seed, "referenced_works": seed.get("related_works") or []} for seed in seeds
    ])
    for paper in output:
        paper["citation_direction"] = "seed_related_candidate"
        paper["citation_lane"] = "forward_related"
    return output


async def retrieve_coverage_balanced_frontier(
    query: str,
    understanding: dict[str, Any],
    contract: dict[str, Any],
    baseline_top20: Sequence[dict[str, Any]],
    candidate_pool: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    seeds = select_frontier_seeds(baseline_top20, candidate_pool)
    seed_ids = [str(seed.get("openalex_id") or "") for seed in seeds]
    queries = build_frontier_queries(query, contract)
    lane_results: dict[str, list[dict[str, Any]]] = {}
    errors: list[dict[str, str]] = []
    for lane, lane_query in queries:
        try:
            papers = await frozen_call(
                f"qcf12_{lane}",
                {"seed_ids": [normalize_openalex_id(value) for value in seed_ids],
                 "query": lane_query, "per_page": PER_REVERSE_LANE},
                lambda lane=lane, lane_query=lane_query: fetch_openalex_citing_any(
                    seed_ids, lane_query, source_query=f"qcf12:{lane}", per_page=PER_REVERSE_LANE,
                ),
            )
            lane_results[lane] = _annotate_reverse(
                filter_for_cutoff(papers, f"qcf12_{lane}"), seeds, lane
            )
        except Exception as exc:
            lane_results[lane] = []
            errors.append({"lane": lane, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})

    for lane, field in (("forward_reference", "referenced_works"), ("forward_related", "related_works")):
        ids = round_robin_neighbor_ids(seeds, field)
        try:
            papers = await frozen_call(
                f"qcf12_{lane}", {"work_ids": ids, "limit": MAX_FRONTIER_IDS},
                lambda ids=ids, lane=lane: fetch_openalex_works(
                    ids, source_query=f"qcf12:{lane}", limit=MAX_FRONTIER_IDS,
                ),
            ) if ids else []
            papers = filter_for_cutoff(papers, f"qcf12_{lane}")
            lane_results[lane] = (
                _annotate_reference(papers, seeds) if lane == "forward_reference"
                else _annotate_related(papers, seeds)
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
            "referenced_count": len(seed.get("referenced_works") or []),
            "related_count": len(seed.get("related_works") or []),
        } for seed in seeds],
        "queries": [{"lane": lane, "query": lane_query} for lane, lane_query in queries],
        "lanes": lane_results,
        "candidates": candidates,
        "errors": errors,
        "api_calls": len(queries) + 2,
    }
