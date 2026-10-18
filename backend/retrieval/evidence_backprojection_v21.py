"""Evidence-weighted citation backprojection (EBP).

The ordinary forward-citation lane resolves the first references encountered
in a small seed list.  EBP instead treats every already-scored candidate as an
independent evidence source, projects its confidence onto unresolved reference
IDs, and materialises the strongest consensus targets in one OpenAlex batch.

The module is benchmark-agnostic: selection uses only paper metadata and model
scores available at inference time.  No ground-truth labels are consulted.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from typing import Any, Sequence

from backend.eval_runtime import filter_for_cutoff, frozen_call
from backend.openalex_client import fetch_openalex_works
from backend.paper_identity import normalize_openalex_id, normalize_title
from backend.selection.local_graph_reranker_v17 import _percentiles


MAX_BACKPROJECTION_IDS = 100
EVIDENCE_FIELDS: tuple[str, ...] = (
    "v17_cross_pct",
    "v17_trace_pct",
    "v17_graph_pct",
    "ebp_lexical_pct",
    "ebp_final_pct",
)


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def attach_backprojection_views(
    incumbents: Sequence[dict[str, Any]],
    frontier: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Copy rows and attach tie-stable lexical/fused percentiles."""

    rows = [copy.deepcopy(row) for row in [*incumbents, *frontier]]
    for source, target in (
        ("lexical_score", "ebp_lexical_pct"),
        ("final_score", "ebp_final_pct"),
    ):
        values = _percentiles([_float(row.get(source)) for row in rows])
        for row, value in zip(rows, values):
            row[target] = value
    return rows


def source_evidence_score(
    paper: dict[str, Any],
    *,
    required_views: int = 3,
    agreement_bonus: float = 0.05,
) -> float:
    """Return a robust multi-view score controlled by the kth-best view."""

    if not 1 <= required_views <= len(EVIDENCE_FIELDS):
        raise ValueError("required_views must be between 1 and the signal count")
    values = sorted((_float(paper.get(field)) for field in EVIDENCE_FIELDS), reverse=True)
    strongest = values[:required_views]
    return values[required_views - 1] + agreement_bonus * sum(strongest) / required_views


def rank_backprojection_targets(
    incumbents: Sequence[dict[str, Any]],
    frontier: Sequence[dict[str, Any]],
    *,
    limit: int = MAX_BACKPROJECTION_IDS,
    reference_length_exponent: float = 0.5,
) -> list[dict[str, Any]]:
    """Rank unresolved reference IDs by evidence-normalised source votes.

    A source that cites hundreds of papers should not cast hundreds of full
    votes.  Its evidence is therefore divided by ``len(refs) ** exponent``.
    Support counts and seed titles are retained for explanation and auditing.
    """

    if limit < 1:
        return []
    if reference_length_exponent < 0:
        raise ValueError("reference_length_exponent must be non-negative")

    rows = attach_backprojection_views(incumbents, frontier)
    existing = {
        normalize_openalex_id(row.get("openalex_id"))
        for row in rows
        if normalize_openalex_id(row.get("openalex_id"))
    }
    scores: dict[str, float] = defaultdict(float)
    supporters: dict[str, dict[str, str]] = defaultdict(dict)

    for row in rows:
        references = list(dict.fromkeys(
            normalize_openalex_id(value)
            for value in (row.get("referenced_works") or [])
            if normalize_openalex_id(value)
        ))
        unresolved = [work_id for work_id in references if work_id not in existing]
        if not unresolved:
            continue
        denominator = max(1.0, len(references) ** reference_length_exponent)
        vote = source_evidence_score(row) / denominator
        seed_id = normalize_openalex_id(row.get("openalex_id")) or normalize_title(row.get("title"))
        seed_title = str(row.get("title") or "").strip()
        for work_id in unresolved:
            scores[work_id] += vote
            if seed_id:
                supporters[work_id][seed_id] = seed_title

    ordered = sorted(
        scores,
        key=lambda work_id: (
            -scores[work_id],
            -len(supporters[work_id]),
            work_id,
        ),
    )[:limit]
    return [
        {
            "openalex_id": work_id,
            "score": scores[work_id],
            "rank": rank,
            "support_count": len(supporters[work_id]),
            "support_seed_titles": [
                title for _, title in sorted(supporters[work_id].items()) if title
            ],
        }
        for rank, work_id in enumerate(ordered, 1)
    ]


def annotate_backprojection_candidates(
    papers: Sequence[dict[str, Any]],
    targets: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach the backprojection path and supporting evidence to fetched works."""

    by_id = {
        normalize_openalex_id(target.get("openalex_id")): target
        for target in targets
        if normalize_openalex_id(target.get("openalex_id"))
    }
    output: list[dict[str, Any]] = []
    for raw in papers:
        paper = copy.deepcopy(raw)
        target = by_id.get(normalize_openalex_id(paper.get("openalex_id"))) or {}
        paper.update({
            "citation_lane": "evidence_backprojection",
            "citation_direction": "seed_cites_candidate",
            "citation_backprojection_score": _float(target.get("score")),
            "citation_backprojection_rank": int(target.get("rank") or 0),
            "citation_seed_count": int(target.get("support_count") or 0),
            "citation_seed_titles": list(target.get("support_seed_titles") or []),
            "source_round": "evidence_backprojection",
        })
        output.append(paper)
    return output


async def retrieve_evidence_backprojection(
    incumbents: Sequence[dict[str, Any]],
    frontier: Sequence[dict[str, Any]],
    *,
    limit: int = MAX_BACKPROJECTION_IDS,
) -> dict[str, Any]:
    """Materialise evidence-backed unresolved references in one API call."""

    targets = rank_backprojection_targets(incumbents, frontier, limit=limit)
    ids = [target["openalex_id"] for target in targets]
    if not ids:
        return {
            "targets": [], "candidates": [], "api_calls": 0, "errors": [],
            "strategy": "EBP_v21_evidence_weighted_citation_backprojection",
        }
    try:
        papers = await frozen_call(
            "ebp_v21_forward_references",
            {"work_ids": ids, "limit": limit},
            lambda: fetch_openalex_works(
                ids, source_query="ebp_v21:evidence_backprojection", limit=limit,
            ),
        )
        papers = filter_for_cutoff(papers, "ebp_v21_forward_references")
        candidates = annotate_backprojection_candidates(papers, targets)
        errors: list[dict[str, str]] = []
    except Exception as exc:
        candidates = []
        errors = [{"lane": "evidence_backprojection", "error": f"{type(exc).__name__}: {str(exc)[:200]}"}]
    return {
        "targets": targets,
        "candidates": candidates,
        "api_calls": 1,
        "errors": errors,
        "strategy": "EBP_v21_evidence_weighted_citation_backprojection",
    }
