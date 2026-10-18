"""Query-Conditioned Batched Citation Frontier (QCF).

QCF expands the *reverse* citation direction (papers citing current seeds),
which ordinary reference expansion under-explores.  Multiple seeds are ORed
inside one OpenAlex filter and two query views constrain the broad frontier.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Iterable, Sequence

from backend.eval_runtime import filter_for_cutoff, frozen_call
from backend.openalex_client import fetch_openalex_citing_any, fetch_openalex_works
from backend.paper_identity import match_identity, normalize_openalex_id, normalize_title
from backend.trace_rank import fuse_candidates, score_candidates


MAX_SEEDS = 40
MAX_REFERENCE_IDS = 100
PER_LANE = 50


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _identity_key(paper: dict[str, Any]) -> str:
    return (
        normalize_openalex_id(paper.get("openalex_id"))
        or normalize_title(paper.get("title"))
    )


def _dedupe(papers: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for paper in papers:
        if not paper.get("title"):
            continue
        if any(match_identity(paper, old) or match_identity(old, paper) for old in output):
            continue
        output.append(paper)
    return output


def _merge_dedupe(papers: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve ranked features while backfilling graph metadata by identity."""
    output: list[dict[str, Any]] = []
    for raw in papers:
        if not raw.get("title"):
            continue
        match = next((old for old in output if match_identity(raw, old) or match_identity(old, raw)), None)
        if match is None:
            output.append(copy.deepcopy(raw))
            continue
        for key, value in raw.items():
            if value not in (None, "", [], {}) and match.get(key) in (None, "", [], {}):
                match[key] = copy.deepcopy(value)
    return output


def select_frontier_seeds(
    baseline_top20: Sequence[dict[str, Any]],
    candidate_pool: Sequence[dict[str, Any]],
    limit: int = MAX_SEEDS,
) -> list[dict[str, Any]]:
    """Select relevance and graph bridge seeds without benchmark state."""
    merged = _merge_dedupe([*baseline_top20, *candidate_pool])
    eligible = [paper for paper in merged if normalize_openalex_id(paper.get("openalex_id"))]

    def relevance(paper: dict[str, Any]) -> float:
        return (
            .45 * _float(paper.get("final_score"))
            + .30 * _float(paper.get("trace_score"))
            + .20 * _float(paper.get("lexical_score"))
            + .20 * _float(paper.get("constraint_coverage_score"))
            + .10 * _float(paper.get("facet_coverage"))
            + .05 * min(1.0, len(paper.get("referenced_works") or []) / 30.0)
        )

    relevance_view = sorted(eligible, key=lambda paper: (-relevance(paper), _identity_key(paper)))
    graph_view = sorted(
        eligible,
        key=lambda paper: (
            -len(paper.get("referenced_works") or []),
            -len(paper.get("related_works") or []),
            -relevance(paper),
            _identity_key(paper),
        ),
    )
    relevance_budget = min(len(relevance_view), max(1, int(limit * .75)))
    selected = list(relevance_view[:relevance_budget])
    for paper in graph_view:
        if paper not in selected:
            selected.append(paper)
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        selected.extend(paper for paper in relevance_view if paper not in selected)
    return selected[:limit]


def _clean_query(text: str, limit: int = 500) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    value = re.sub(
        r"^(what|which|could you|can you|are there|any|who)\b[^a-z0-9]{0,3}",
        "", value, flags=re.I,
    ).strip(" ?.:")
    return value[:limit]


def build_frontier_queries(query: str, contract: dict[str, Any]) -> list[tuple[str, str]]:
    original = _clean_query(query)
    terms: list[str] = []
    for condition in (contract.get("conditions") or [])[:5]:
        if not isinstance(condition, dict):
            continue
        aliases = [str(alias) for alias in (condition.get("aliases") or [])[:3] if str(alias).strip()]
        if aliases:
            terms.extend(aliases)
        else:
            terms.append(str(condition.get("description") or ""))
    relation = str(contract.get("relation") or "")
    # Keep this view compact. Long contract prose makes OpenAlex lexical
    # search over-constrained and often returns zero results.
    contract_query = _clean_query(" ".join([relation, *terms[:9]]), 300)
    queries: list[tuple[str, str]] = []
    if original:
        queries.append(("reverse_query", original))
    if contract_query and normalize_title(contract_query) != normalize_title(original):
        queries.append(("reverse_contract", contract_query))
    return queries[:2]


def _annotate_reverse(
    papers: Sequence[dict[str, Any]], seeds: Sequence[dict[str, Any]], lane: str,
) -> list[dict[str, Any]]:
    seed_ids = {
        normalize_openalex_id(seed.get("openalex_id")): seed for seed in seeds
        if normalize_openalex_id(seed.get("openalex_id"))
    }
    output = []
    for raw in papers:
        paper = copy.deepcopy(raw)
        refs = {normalize_openalex_id(value) for value in paper.get("referenced_works") or []}
        hits = sorted(seed_id for seed_id in seed_ids if seed_id in refs)
        paper["citation_direction"] = "candidate_cites_seed"
        paper["citation_lane"] = lane
        paper["citation_seed_ids"] = hits
        paper["citation_seed_titles"] = [seed_ids[seed_id].get("title") for seed_id in hits]
        paper["citation_seed_count"] = len(hits)
        paper["source_round"] = "query_conditioned_citation"
        output.append(paper)
    return output


def _annotate_reference(
    papers: Sequence[dict[str, Any]], seeds: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_ref: dict[str, list[dict[str, Any]]] = {}
    for seed in seeds:
        for value in seed.get("referenced_works") or []:
            wid = normalize_openalex_id(value)
            if wid:
                by_ref.setdefault(wid, []).append(seed)
    output = []
    for raw in papers:
        paper = copy.deepcopy(raw)
        wid = normalize_openalex_id(paper.get("openalex_id"))
        parents = by_ref.get(wid) or []
        paper["citation_direction"] = "seed_cites_candidate"
        paper["citation_lane"] = "forward_reference"
        paper["citation_seed_ids"] = [normalize_openalex_id(seed.get("openalex_id")) for seed in parents]
        paper["citation_seed_titles"] = [seed.get("title") for seed in parents]
        paper["citation_seed_count"] = len(parents)
        paper["source_round"] = "query_conditioned_citation"
        output.append(paper)
    return output


async def retrieve_query_conditioned_frontier(
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
        arguments = {
            "seed_ids": [normalize_openalex_id(value) for value in seed_ids],
            "query": lane_query,
            "per_page": PER_LANE,
        }
        try:
            papers = await frozen_call(
                f"qcf_{lane}", arguments,
                lambda lane=lane, lane_query=lane_query: fetch_openalex_citing_any(
                    seed_ids, lane_query, source_query=f"qcf:{lane}", per_page=PER_LANE,
                ),
            )
            papers = filter_for_cutoff(papers, f"qcf_{lane}")
            lane_results[lane] = _annotate_reverse(papers, seeds, lane)
        except Exception as exc:
            lane_results[lane] = []
            errors.append({"lane": lane, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})

    reference_ids: list[str] = []
    for seed in seeds:
        for value in seed.get("referenced_works") or []:
            if value not in reference_ids:
                reference_ids.append(value)
            if len(reference_ids) >= MAX_REFERENCE_IDS:
                break
        if len(reference_ids) >= MAX_REFERENCE_IDS:
            break
    try:
        references = await frozen_call(
            "qcf_forward_references",
            {"work_ids": reference_ids, "limit": MAX_REFERENCE_IDS},
            lambda: fetch_openalex_works(
                reference_ids, source_query="qcf:forward_reference", limit=MAX_REFERENCE_IDS,
            ),
        ) if reference_ids else []
        references = filter_for_cutoff(references, "qcf_forward_references")
        lane_results["forward_reference"] = _annotate_reference(references, seeds)
    except Exception as exc:
        lane_results["forward_reference"] = []
        errors.append({"lane": "forward_reference", "error": f"{type(exc).__name__}: {str(exc)[:200]}"})

    all_candidates = fuse_candidates([
        paper for papers in lane_results.values() for paper in papers
    ])
    scored = score_candidates(all_candidates, understanding)
    return {
        "seeds": [{
            "openalex_id": seed.get("openalex_id"),
            "title": seed.get("title"),
            "referenced_count": len(seed.get("referenced_works") or []),
        } for seed in seeds],
        "queries": [{"lane": lane, "query": lane_query} for lane, lane_query in queries],
        "lanes": lane_results,
        "candidates": scored,
        "errors": errors,
        "api_calls": len(queries) + int(bool(reference_ids)),
    }
