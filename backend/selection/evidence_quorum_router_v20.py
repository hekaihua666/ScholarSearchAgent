"""Multi-view evidence-quorum supplement for Relational Evidence Graph.

The strict REG path is retained for candidates with explicit compositional
evidence.  A second, tightly protected route handles candidates whose extracted
condition conjunction is sparse but whose independent semantic, TRACE, graph,
lexical and fused-ranking views agree.  It reuses existing scores and therefore
adds no retrieval, LLM or token cost.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from backend.paper_identity import match_identity, normalize_title
from backend.selection.evidence_contract_ranker_v9 import _float
from backend.selection.relational_evidence_graph_v18 import (
    relational_score,
    route_relational_top20,
)


QUORUM_SIGNALS: tuple[str, ...] = (
    "v17_cross_pct",
    "v17_trace_pct",
    "v17_graph_pct",
    "v20_lexical_pct",
    "v20_final_pct",
)


def _percentiles(rows: Sequence[dict[str, Any]], field: str) -> list[float]:
    """Return deterministic within-query percentile ranks in [0, 1]."""

    order = sorted(
        range(len(rows)),
        key=lambda index: (
            _float(rows[index].get(field)),
            normalize_title(rows[index].get("title")),
        ),
    )
    values = [0.0] * len(rows)
    denominator = max(len(rows) - 1, 1)
    for rank, index in enumerate(order):
        values[index] = rank / denominator
    return values


def attach_quorum_features(
    incumbents: Sequence[dict[str, Any]],
    frontier: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [copy.deepcopy(row) for row in [*incumbents, *frontier]]
    for source, target in (
        ("lexical_score", "v20_lexical_pct"),
        ("final_score", "v20_final_pct"),
    ):
        for row, value in zip(rows, _percentiles(rows, source)):
            row[target] = value
    n = len(incumbents)
    return rows[:n], rows[n:]


def evidence_quorum_score(
    paper: dict[str, Any],
    *,
    required_views: int = 3,
    agreement_bonus: float = 0.05,
) -> float:
    if not 1 <= required_views <= len(QUORUM_SIGNALS):
        raise ValueError("required_views must be between 1 and the signal count")
    values = sorted((_float(paper.get(field)) for field in QUORUM_SIGNALS), reverse=True)
    strongest = values[:required_views]
    return values[required_views - 1] + agreement_bonus * sum(strongest) / required_views


def route_evidence_quorum_top20(
    incumbents: Sequence[dict[str, Any]],
    frontier: Sequence[dict[str, Any]],
    *,
    weights: Sequence[float] | Mapping[str, Sequence[float]],
    reg_score_floor: float = 0.62,
    reg_margin: float = 0.10,
    reg_protect_head: int = 15,
    quorum_floor: float = 0.97,
    quorum_margin: float = 0.20,
    quorum_protect_head: int = 18,
    required_views: int = 3,
    agreement_bonus: float = 0.05,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply frozen REG followed by at most one protected quorum promotion."""

    prepared_incumbents, prepared_frontier = attach_quorum_features(incumbents, frontier)
    current, strict_actions = route_relational_top20(
        prepared_incumbents,
        prepared_frontier,
        weights=weights,
        score_floor=reg_score_floor,
        margin=reg_margin,
        max_promotions=1,
        protect_head=reg_protect_head,
    )
    actions = [{"route": "strict_reg", **action} for action in strict_actions]

    candidates = sorted(
        prepared_frontier,
        key=lambda paper: (
            -evidence_quorum_score(
                paper, required_views=required_views, agreement_bonus=agreement_bonus
            ),
            normalize_title(paper.get("title")),
        ),
    )
    candidate = next(
        (
            paper for paper in candidates
            if not any(match_identity(paper, item) or match_identity(item, paper) for item in current)
        ),
        None,
    )
    removable = list(enumerate(current[quorum_protect_head:], start=quorum_protect_head))
    if candidate is None or not removable:
        return current, actions

    index, outgoing = min(
        removable,
        key=lambda item: (
            evidence_quorum_score(
                item[1], required_views=required_views, agreement_bonus=agreement_bonus
            ),
            relational_score(item[1], weights),
            -item[0],
        ),
    )
    incoming_score = evidence_quorum_score(
        candidate, required_views=required_views, agreement_bonus=agreement_bonus
    )
    outgoing_score = evidence_quorum_score(
        outgoing, required_views=required_views, agreement_bonus=agreement_bonus
    )
    if incoming_score < quorum_floor or incoming_score < outgoing_score + quorum_margin:
        return current, actions

    current[index] = copy.deepcopy(candidate)
    actions.append({
        "route": "evidence_quorum",
        "incoming_title": candidate.get("title"),
        "outgoing_title": outgoing.get("title"),
        "incoming_score": incoming_score,
        "outgoing_score": outgoing_score,
        "margin": incoming_score - outgoing_score,
        "required_views": required_views,
        "protected_head": quorum_protect_head,
        "cross": candidate.get("v17_cross_pct"),
        "trace": candidate.get("v17_trace_pct"),
        "graph": candidate.get("v17_graph_pct"),
        "lexical": candidate.get("v20_lexical_pct"),
        "fused": candidate.get("v20_final_pct"),
    })
    return current, actions

