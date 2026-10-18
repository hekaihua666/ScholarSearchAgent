"""Facet-Compressed Grouped Citation Frontier (FC-GQCF v14)."""

from __future__ import annotations

import re
from typing import Any, Sequence

from backend.eval_runtime import filter_for_cutoff, frozen_call
from backend.openalex_client import fetch_openalex_citing_any, fetch_openalex_works
from backend.paper_identity import normalize_openalex_id, normalize_title
from backend.retrieval.grouped_citation_frontier_v13 import (
    GROUP_SIZE,
    MAX_FORWARD_IDS,
    PER_GROUP,
    head_weighted_reference_ids,
    partition_seeds,
)
from backend.retrieval.query_conditioned_citation_v11 import (
    _annotate_reference,
    _annotate_reverse,
    select_frontier_seeds,
)
from backend.trace_rank import fuse_candidates, score_candidates


def build_facet_compressed_query(
    query: str, understanding: dict[str, Any], contract: dict[str, Any], limit: int = 220,
) -> str:
    """Project conversational queries to graph-search facets, not prose."""
    parts: list[str] = []
    for field in ("research_topic", "method_constraints", "task_constraints"):
        value = str(understanding.get(field) or "").strip()
        if not value or value.lower() == "not specified":
            continue
        for item in re.split(r"[,;/]", value):
            item = re.sub(r"\s+", " ", item).strip(" .?:")
            if item and normalize_title(item) not in {normalize_title(old) for old in parts}:
                parts.append(item)
    if not parts:
        for condition in (contract.get("conditions") or [])[:4]:
            if not isinstance(condition, dict):
                continue
            aliases = [str(value).strip() for value in (condition.get("aliases") or []) if str(value).strip()]
            parts.append(aliases[0] if aliases else str(condition.get("description") or ""))
    if not parts:
        value = re.sub(r"\b(what|which|papers?|works?|studies|could you|can you|are there|any)\b", " ", query, flags=re.I)
        parts = [re.sub(r"\s+", " ", value).strip(" .?:")]
    return re.sub(r"\s+", " ", " ".join(parts)).strip()[:limit]


async def retrieve_facet_compressed_citation_frontier(
    query: str,
    understanding: dict[str, Any],
    contract: dict[str, Any],
    baseline_top20: Sequence[dict[str, Any]],
    candidate_pool: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    seeds = select_frontier_seeds(baseline_top20, candidate_pool)
    graph_query = build_facet_compressed_query(query, understanding, contract)
    groups = partition_seeds(seeds, GROUP_SIZE)
    lane_results: dict[str, list[dict[str, Any]]] = {}
    errors: list[dict[str, str]] = []
    for index, group in enumerate(groups, 1):
        lane = f"reverse_group_{index}"
        ids = [str(seed.get("openalex_id") or "") for seed in group]
        try:
            papers = await frozen_call(
                f"fcgqcf14_{lane}",
                {"seed_ids": [normalize_openalex_id(value) for value in ids],
                 "query": graph_query, "per_page": PER_GROUP},
                lambda ids=ids, lane=lane: fetch_openalex_citing_any(
                    ids, graph_query, source_query=f"fcgqcf14:{lane}", per_page=PER_GROUP,
                ),
            )
            lane_results[lane] = _annotate_reverse(
                filter_for_cutoff(papers, f"fcgqcf14_{lane}"), group, lane
            )
        except Exception as exc:
            lane_results[lane] = []
            errors.append({"lane": lane, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})

    reference_ids = head_weighted_reference_ids(seeds, MAX_FORWARD_IDS)
    try:
        references = await frozen_call(
            "fcgqcf14_forward_reference",
            {"work_ids": reference_ids, "limit": MAX_FORWARD_IDS},
            lambda: fetch_openalex_works(
                reference_ids, source_query="fcgqcf14:forward_reference", limit=MAX_FORWARD_IDS,
            ),
        ) if reference_ids else []
        lane_results["forward_reference"] = _annotate_reference(
            filter_for_cutoff(references, "fcgqcf14_forward_reference"), seeds
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
        "queries": [{"lane": "facet_compressed_reverse", "query": graph_query}],
        "lanes": lane_results,
        "candidates": candidates,
        "errors": errors,
        "api_calls": len(groups) + int(bool(reference_ids)),
    }
