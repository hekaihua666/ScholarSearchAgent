"""Evidence-aware dual frontier selection for citation expansion."""

from __future__ import annotations

import math
from typing import Any

from backend.trace_rank import canonical_key


def _score(paper: dict[str, Any]) -> float:
    return float(paper.get("final_score") or paper.get("relevance_score") or 0.0)


def _neighborhood_size(paper: dict[str, Any]) -> int:
    return len(paper.get("referenced_works") or []) + len(paper.get("related_works") or [])


def choose_dual_frontier(
    ranked: list[dict[str, Any]], relevant_limit: int = 4, bridge_limit: int = 4
) -> dict[str, list[dict[str, Any]]]:
    """Separate relevance seeds from graph-coverage bridge seeds.

    Bridge membership is only permission to explore its neighborhood.  It is
    not an acceptance decision and therefore cannot directly enter a selected
    result set through this function.
    """
    eligible = [paper for paper in ranked if _neighborhood_size(paper) > 0]
    relevant = sorted(
        (
            paper for paper in eligible
            if float(paper.get("lexical_score") or 0) >= 0.20
            and float(paper.get("constraint_coverage_score") or 0) >= 0.18
        ),
        key=lambda paper: (
            _score(paper),
            float(paper.get("constraint_coverage_score") or 0),
            _neighborhood_size(paper),
        ),
        reverse=True,
    )[:relevant_limit]
    used = {canonical_key(paper) for paper in relevant}

    def utility(paper: dict[str, Any]) -> float:
        neighborhood = _neighborhood_size(paper)
        relevance_floor = max(float(paper.get("lexical_score") or 0), 0.08)
        coverage = float(paper.get("constraint_coverage_score") or 0)
        semantic = float(paper.get("semantic_provenance_score") or 0)
        return math.log1p(neighborhood) * (0.55 * relevance_floor + 0.25 * coverage + 0.20 * semantic)

    bridge = sorted(
        (paper for paper in eligible if canonical_key(paper) not in used),
        key=utility,
        reverse=True,
    )[:bridge_limit]
    for paper in relevant:
        paper["frontier_role"] = "relevant"
        paper["frontier_utility"] = round(_score(paper), 4)
    for paper in bridge:
        paper["frontier_role"] = "bridge"
        paper["frontier_utility"] = round(utility(paper), 4)
    return {"relevant": relevant, "bridge": bridge}
