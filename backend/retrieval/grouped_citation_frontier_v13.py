"""Grouped Query-Conditioned Citation Frontier (GQCF v13)."""

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


GROUP_SIZE = 10
PER_GROUP = 100
MAX_FORWARD_IDS = 100


def partition_seeds(
    seeds: Sequence[dict[str, Any]], group_size: int = GROUP_SIZE,
) -> list[list[dict[str, Any]]]:
    return [list(seeds[index:index + group_size]) for index in range(0, len(seeds), group_size)]


def head_weighted_reference_ids(
    seeds: Sequence[dict[str, Any]], limit: int = MAX_FORWARD_IDS,
) -> list[str]:
    """Preserve the successful v11 forward allocation as a named policy."""
    output: list[str] = []
    for seed in seeds:
        for raw in seed.get("referenced_works") or []:
            wid = normalize_openalex_id(raw)
            if wid and wid not in output:
                output.append(wid)
            if len(output) >= limit:
                return output
    return output


async def retrieve_grouped_citation_frontier(
    query: str,
    understanding: dict[str, Any],
    contract: dict[str, Any],
    baseline_top20: Sequence[dict[str, Any]],
    candidate_pool: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    seeds = select_frontier_seeds(baseline_top20, candidate_pool)
    query_views = build_frontier_queries(query, contract)
    original_query = next((text for lane, text in query_views if lane == "reverse_query"), query)
    groups = partition_seeds(seeds)
    lane_results: dict[str, list[dict[str, Any]]] = {}
    errors: list[dict[str, str]] = []
    for index, group in enumerate(groups, 1):
        lane = f"reverse_group_{index}"
        ids = [str(seed.get("openalex_id") or "") for seed in group]
        try:
            papers = await frozen_call(
                f"gqcf13_{lane}",
                {"seed_ids": [normalize_openalex_id(value) for value in ids],
                 "query": original_query, "per_page": PER_GROUP},
                lambda ids=ids, lane=lane: fetch_openalex_citing_any(
                    ids, original_query, source_query=f"gqcf13:{lane}", per_page=PER_GROUP,
                ),
            )
            lane_results[lane] = _annotate_reverse(
                filter_for_cutoff(papers, f"gqcf13_{lane}"), group, lane
            )
        except Exception as exc:
            lane_results[lane] = []
            errors.append({"lane": lane, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})

    reference_ids = head_weighted_reference_ids(seeds)
    try:
        references = await frozen_call(
            "gqcf13_forward_reference",
            {"work_ids": reference_ids, "limit": MAX_FORWARD_IDS},
            lambda: fetch_openalex_works(
                reference_ids, source_query="gqcf13:forward_reference", limit=MAX_FORWARD_IDS,
            ),
        ) if reference_ids else []
        lane_results["forward_reference"] = _annotate_reference(
            filter_for_cutoff(references, "gqcf13_forward_reference"), seeds
        )
    except Exception as exc:
        lane_results["forward_reference"] = []
        errors.append({"lane": "forward_reference", "error": f"{type(exc).__name__}: {str(exc)[:200]}"})

    candidates = score_candidates(
        fuse_candidates([paper for rows in lane_results.values() for paper in rows]), understanding
    )
    return {
        "seeds": [{
            "openalex_id": seed.get("openalex_id"), "title": seed.get("title"),
            "referenced_count": len(seed.get("referenced_works") or []),
            "seed_group": index // GROUP_SIZE + 1,
        } for index, seed in enumerate(seeds)],
        "queries": [{"lane": "reverse_grouped", "query": original_query}],
        "lanes": lane_results,
        "candidates": candidates,
        "errors": errors,
        "api_calls": len(groups) + int(bool(reference_ids)),
    }
