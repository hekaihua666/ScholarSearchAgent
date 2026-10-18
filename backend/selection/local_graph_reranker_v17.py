"""Local graph-residual reranking for the SCGQ citation frontier.

The module deliberately separates three signals that have different failure
modes: query-document relevance from a local cross encoder, graph support from
independent citation seeds, and the existing TRACE score.  All features are
normalized within a query so that routing decisions can be calibrated by
relative margins instead of model-specific raw logits.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from backend.paper_identity import match_identity, normalize_title, paper_identity
from backend.selection.cross_encoder_selector import (
    CPU_FALLBACK_MODEL,
    CrossEncoderSelector,
)
from backend.selection.evidence_contract_ranker_v9 import _float


DEFAULT_POOL_LIMIT = 120


def _source_rank(paper: dict[str, Any]) -> int:
    try:
        return max(1, int(paper.get("_openalex_rank") or 999))
    except (TypeError, ValueError):
        return 999


def _identity_keys(paper: dict[str, Any]) -> set[str]:
    identity = paper_identity(paper)
    keys = {
        *(f"arxiv:{value}" for value in identity.arxiv_ids),
        *(f"doi:{value}" for value in identity.dois),
        *(f"openalex:{value}" for value in identity.openalex_ids),
        *(f"s2:{value}" for value in identity.semantic_scholar_ids),
    }
    if identity.title:
        keys.add(f"title:{identity.title}")
    if identity.pasa_title:
        keys.add(f"pasa:{identity.pasa_title}")
    return keys


def _identity_in(paper: dict[str, Any], rows: Sequence[dict[str, Any]]) -> bool:
    keys = _identity_keys(paper)
    return bool(keys and any(keys & _identity_keys(row) for row in rows))


def _deduplicate(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for paper in rows:
        keys = _identity_keys(paper)
        if paper.get("title") and keys and not (keys & seen):
            output.append(copy.deepcopy(paper)); seen.update(keys)
    return output


def select_multiview_frontier_pool(
    frontier: Sequence[dict[str, Any]],
    incumbents: Sequence[dict[str, Any]],
    limit: int = DEFAULT_POOL_LIMIT,
) -> list[dict[str, Any]]:
    """Select a bounded, deterministic and view-diverse frontier pool."""

    incumbent_keys = set().union(*(_identity_keys(paper) for paper in incumbents)) if incumbents else set()
    legal = [paper for paper in _deduplicate(frontier) if not (_identity_keys(paper) & incumbent_keys)]
    score_view = sorted(
        legal,
        key=lambda paper: (
            -_float(paper.get("final_score")),
            _source_rank(paper),
            normalize_title(paper.get("title")),
        ),
    )
    graph_view = sorted(
        legal,
        key=lambda paper: (
            -int(paper.get("citation_seed_count") or 0),
            -_float(paper.get("constraint_coverage_score")),
            _source_rank(paper),
            normalize_title(paper.get("title")),
        ),
    )
    source_view = sorted(
        legal,
        key=lambda paper: (
            _source_rank(paper),
            -int(paper.get("citation_seed_count") or 0),
            -_float(paper.get("final_score")),
        ),
    )
    authority_view = sorted(
        legal,
        key=lambda paper: (
            -int(paper.get("cited_by_count") or 0),
            -_float(paper.get("final_score")),
            normalize_title(paper.get("title")),
        ),
    )
    lane_views: list[list[dict[str, Any]]] = []
    for lane in sorted({str(paper.get("citation_lane") or "") for paper in legal}):
        if not lane:
            continue
        lane_views.append(sorted(
            [paper for paper in legal if str(paper.get("citation_lane") or "") == lane],
            key=lambda paper: (
                -int(paper.get("citation_seed_count") or 0),
                _source_rank(paper),
                -_float(paper.get("final_score")),
            ),
        ))

    views = [score_view, graph_view, source_view, authority_view, *lane_views]
    output: list[dict[str, Any]] = []
    output_keys: set[str] = set()
    depth = 0
    while len(output) < limit and any(depth < len(view) for view in views):
        for view in views:
            if depth >= len(view):
                continue
            paper = view[depth]
            keys = _identity_keys(paper)
            if not (keys & output_keys):
                output.append(copy.deepcopy(paper)); output_keys.update(keys)
                if len(output) >= limit:
                    break
        depth += 1
    return output


def contract_query(question: str, contract: dict[str, Any]) -> str:
    parts = [str(question or "").strip()]
    for condition in (contract.get("conditions") or [])[:6]:
        description = str(condition.get("description") or "").strip()
        aliases = ", ".join(str(value) for value in (condition.get("aliases") or [])[:5])
        if description:
            parts.append(description + (f" Aliases: {aliases}." if aliases else ""))
    relation = str(contract.get("relation") or "").strip()
    if relation:
        parts.append(f"Required relation: {relation}.")
    exclusions = "; ".join(str(value) for value in (contract.get("exclusions") or [])[:6])
    if exclusions:
        parts.append(f"Exclude: {exclusions}.")
    return " Required constraints: ".join(part for part in parts if part)[:900]


def paper_document(paper: dict[str, Any]) -> str:
    title = str(paper.get("title") or "").strip()
    abstract = str(paper.get("abstract") or "").strip()
    return f"Title: {title}. Abstract: {abstract[:700]}"


def _percentiles(values: Sequence[float]) -> list[float]:
    """Average-rank empirical CDF in [0, 1], stable under ties."""

    if not values:
        return []
    ordered = sorted((value, index) for index, value in enumerate(values))
    output = [0.0] * len(values)
    left = 0
    denominator = max(1, len(values) - 1)
    while left < len(ordered):
        right = left + 1
        while right < len(ordered) and ordered[right][0] == ordered[left][0]:
            right += 1
        average_rank = (left + right - 1) / 2.0
        percentile = average_rank / denominator
        for _, index in ordered[left:right]:
            output[index] = percentile
        left = right
    return output


def graph_support(paper: dict[str, Any]) -> float:
    seeds = max(0, int(paper.get("citation_seed_count") or 0))
    rank = _source_rank(paper)
    direction = str(paper.get("citation_direction") or "")
    direction_bonus = 0.08 if "reverse" in direction else (0.04 if "forward" in direction else 0.0)
    return min(1.0, math.log1p(seeds) / math.log(11.0) + 1.0 / (20.0 + rank) + direction_bonus)


def score_query_pool(
    question: str,
    contract: dict[str, Any],
    papers: Sequence[dict[str, Any]],
    selector: CrossEncoderSelector,
    *,
    batch_size: int = 32,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach local cross, graph and TRACE percentile features."""

    query = contract_query(question, contract)
    pairs = [(query, paper_document(paper)) for paper in papers]
    raw_scores, runtime = selector.predict(pairs, batch_size=batch_size)
    cross_pct = _percentiles(raw_scores)
    trace_values = [_float(paper.get("final_score")) for paper in papers]
    trace_pct = _percentiles(trace_values)
    graph_values = [graph_support(paper) for paper in papers]
    graph_pct = _percentiles(graph_values)
    output: list[dict[str, Any]] = []
    for paper, raw, cross, trace, graph, graph_raw in zip(
        papers, raw_scores, cross_pct, trace_pct, graph_pct, graph_values,
    ):
        row = copy.deepcopy(paper)
        row["v17_cross_raw"] = float(raw)
        row["v17_cross_pct"] = float(cross)
        row["v17_trace_pct"] = float(trace)
        row["v17_graph_pct"] = float(graph)
        row["v17_graph_raw"] = float(graph_raw)
        output.append(row)
    return output, runtime


def composite_score(paper: dict[str, Any], weights: Sequence[float]) -> float:
    cross, graph, trace = (float(value) for value in weights)
    return (
        cross * _float(paper.get("v17_cross_pct"))
        + graph * _float(paper.get("v17_graph_pct"))
        + trace * _float(paper.get("v17_trace_pct"))
    )


def route_frontier_top20(
    incumbents: Sequence[dict[str, Any]],
    frontier: Sequence[dict[str, Any]],
    *,
    weights: Sequence[float],
    margin: float,
    max_promotions: int,
    protect_head: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pairwise residual routing with an explicit incumbent protection set."""

    current = [copy.deepcopy(paper) for paper in incumbents]
    incoming = sorted(
        (copy.deepcopy(paper) for paper in frontier),
        key=lambda paper: (
            -composite_score(paper, weights),
            -_float(paper.get("v17_cross_raw")),
            normalize_title(paper.get("title")),
        ),
    )
    actions: list[dict[str, Any]] = []
    for candidate in incoming:
        if len(actions) >= max_promotions or not current:
            break
        removable = list(enumerate(current[protect_head:], start=protect_head))
        if not removable:
            break
        index, incumbent = min(
            removable,
            key=lambda item: (
                composite_score(item[1], weights),
                _float(item[1].get("v17_cross_raw")),
                -item[0],
            ),
        )
        incoming_score = composite_score(candidate, weights)
        incumbent_score = composite_score(incumbent, weights)
        if incoming_score < incumbent_score + margin:
            break
        current[index] = candidate
        actions.append({
            "incoming_title": candidate.get("title"),
            "outgoing_title": incumbent.get("title"),
            "incoming_score": incoming_score,
            "outgoing_score": incumbent_score,
            "margin": incoming_score - incumbent_score,
        })
    return current, actions


def route_final_set(
    selected: Sequence[dict[str, Any]],
    ranked_pool: Sequence[dict[str, Any]],
    *,
    weights: Sequence[float],
    margin: float,
    cross_floor: float,
    graph_floor: float = 0.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replace at most one weak selected item with a high-confidence residual."""

    enriched: list[dict[str, Any]] = []
    for paper in selected:
        match = next(
            (row for row in ranked_pool if match_identity(paper, row) or match_identity(row, paper)),
            paper,
        )
        enriched.append(copy.deepcopy(match))
    residual = [
        copy.deepcopy(paper) for paper in ranked_pool
        if not _identity_in(paper, enriched)
        and _float(paper.get("v17_cross_pct")) >= cross_floor
        and _float(paper.get("v17_graph_pct")) >= graph_floor
    ]
    if not enriched or not residual:
        return enriched, []
    candidate = max(
        residual,
        key=lambda paper: (
            composite_score(paper, weights),
            _float(paper.get("v17_cross_raw")),
            normalize_title(paper.get("title")),
        ),
    )
    index, incumbent = min(
        enumerate(enriched),
        key=lambda item: (
            composite_score(item[1], weights),
            _float(item[1].get("v17_cross_raw")),
            -item[0],
        ),
    )
    incoming_score = composite_score(candidate, weights)
    incumbent_score = composite_score(incumbent, weights)
    if incoming_score < incumbent_score + margin:
        return enriched, []
    enriched[index] = candidate
    return enriched, [{
        "incoming_title": candidate.get("title"),
        "outgoing_title": incumbent.get("title"),
        "incoming_score": incoming_score,
        "outgoing_score": incumbent_score,
        "margin": incoming_score - incumbent_score,
    }]


def make_selector(cache_path: Path) -> CrossEncoderSelector:
    selector = CrossEncoderSelector(
        CPU_FALLBACK_MODEL,
        cache_path,
        max_length=128,
        local_files_only=True,
    )
    selector._load()
    if selector.runtime != "cross_encoder_cpu":
        raise RuntimeError(f"local cross encoder unavailable: {selector.runtime}")
    return selector
