"""Atomic Bottleneck Evidence Routing (ABER) for complex paper search.

REG v18 models the structure of a complex query, but its condition evidence is
lexical.  ABER evaluates every required condition independently with a local
cross encoder.  Routing is then constrained by the weakest condition and by
candidate-vs-incumbent evidence dominance, rather than by a single topical
relevance score.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence

from backend.paper_identity import normalize_title
from backend.selection.evidence_contract_ranker_v9 import _float
from backend.selection.local_graph_reranker_v17 import paper_document
from backend.selection.relational_evidence_graph_v18 import relational_score


DEFAULT_WEIGHTS = (0.35, 0.15, 0.10, 0.15, 0.10, 0.10, 0.05)
SCREEN_PROFILES = (
    (0.45, 0.15, 0.10, 0.05, 0.10, 0.05, 0.10),
    (0.35, 0.15, 0.10, 0.10, 0.15, 0.10, 0.05),
    (0.35, 0.25, 0.10, 0.05, 0.10, 0.10, 0.05),
)


def _percentiles(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    ordered = sorted((float(value), index) for index, value in enumerate(values))
    output = [0.0] * len(values)
    denominator = max(1, len(values) - 1)
    left = 0
    while left < len(ordered):
        right = left + 1
        while right < len(ordered) and ordered[right][0] == ordered[left][0]:
            right += 1
        percentile = ((left + right - 1) / 2.0) / denominator
        for _, index in ordered[left:right]:
            output[index] = percentile
        left = right
    return output


def atomic_views(question: str, contract: Mapping[str, Any]) -> list[dict[str, str]]:
    """Build stable condition and relation views without benchmark labels."""

    views: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, condition in enumerate((contract.get("conditions") or [])[:4], 1):
        description = str(condition.get("description") or "").strip()
        aliases = [str(value).strip() for value in (condition.get("aliases") or [])[:4] if str(value).strip()]
        key = normalize_title(description)
        if not description or key in seen:
            continue
        seen.add(key)
        query = description
        if aliases:
            query += ". Equivalent terms: " + ", ".join(aliases)
        query += ". Research question context: " + str(question or "")
        views.append({
            "kind": "condition",
            "id": str(condition.get("id") or f"c{index}"),
            "query": query[:900],
        })
    relation = str(contract.get("relation") or "").strip()
    if relation:
        views.append({
            "kind": "relation",
            "id": "relation",
            "query": f"Required relation: {relation}. Research question: {question}"[:900],
        })
    if not any(view["kind"] == "condition" for view in views):
        views.insert(0, {"kind": "condition", "id": "query", "query": str(question or "")[:900]})
    return views


def select_atomic_screen_pool(
    frontier: Sequence[dict[str, Any]],
    limit: int = 32,
) -> list[dict[str, Any]]:
    """Cheap multi-view screen before the condition-wise cross encoder.

    The screen is a union of relation, graph, TRACE and full-query views.  It
    keeps atomic scoring bounded while avoiding a single-score preselection.
    """

    views = [
        sorted(frontier, key=lambda paper, weights=weights: (
            -relational_score(paper, weights), normalize_title(paper.get("title")),
        ))
        for weights in SCREEN_PROFILES
    ]
    for key in ("v17_graph_pct", "v17_trace_pct", "v17_cross_pct", "v18_conjunction"):
        views.append(sorted(frontier, key=lambda paper, key=key: (
            -_float(paper.get(key)), normalize_title(paper.get("title")),
        )))
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    depth = 0
    while len(output) < limit and any(depth < len(view) for view in views):
        for view in views:
            if depth >= len(view):
                continue
            paper = view[depth]
            key = normalize_title(paper.get("title"))
            if key and key not in seen:
                output.append(copy.deepcopy(paper)); seen.add(key)
                if len(output) >= limit:
                    break
        depth += 1
    return output


def attach_atomic_cross_evidence(
    papers: Sequence[dict[str, Any]],
    question: str,
    contract: Mapping[str, Any],
    selector: Any,
    *,
    batch_size: int = 64,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score every paper against every atomic view in one cached model call."""

    views = atomic_views(question, contract)
    documents = [paper_document(paper) for paper in papers]
    pairs = [(view["query"], document) for view in views for document in documents]
    raw, runtime = selector.predict(pairs, batch_size=batch_size)
    if len(raw) != len(pairs):
        raise RuntimeError(f"atomic cross encoder returned {len(raw)} scores for {len(pairs)} pairs")

    by_view_raw: list[list[float]] = []
    by_view_pct: list[list[float]] = []
    width = len(papers)
    for index in range(len(views)):
        values = [float(value) for value in raw[index * width:(index + 1) * width]]
        by_view_raw.append(values)
        by_view_pct.append(_percentiles(values))

    condition_indices = [index for index, view in enumerate(views) if view["kind"] == "condition"]
    relation_indices = [index for index, view in enumerate(views) if view["kind"] == "relation"]
    output: list[dict[str, Any]] = []
    for paper_index, paper in enumerate(papers):
        row = copy.deepcopy(paper)
        condition_pct = [by_view_pct[index][paper_index] for index in condition_indices]
        condition_raw = [by_view_raw[index][paper_index] for index in condition_indices]
        minimum = min(condition_pct, default=_float(row.get("v17_cross_pct")))
        average = sum(condition_pct) / len(condition_pct) if condition_pct else minimum
        geometric = math.exp(
            sum(math.log(max(1e-4, value)) for value in condition_pct) / len(condition_pct)
        ) if condition_pct else minimum
        relation_pct = (
            sum(by_view_pct[index][paper_index] for index in relation_indices) / len(relation_indices)
            if relation_indices else _float(row.get("v17_cross_pct"))
        )
        row["v19_atomic_condition_ids"] = [views[index]["id"] for index in condition_indices]
        row["v19_atomic_raw"] = condition_raw
        row["v19_atomic_pct"] = condition_pct
        row["v19_atomic_min"] = minimum
        row["v19_atomic_mean"] = average
        row["v19_atomic_geo"] = geometric
        row["v19_atomic_gap"] = average - minimum
        row["v19_relation_pct"] = relation_pct
        output.append(row)
    return output, {**runtime, "atomic_views": views, "pair_count": len(pairs)}


def atomic_bottleneck_score(
    paper: Mapping[str, Any],
    weights: Sequence[float] = DEFAULT_WEIGHTS,
) -> float:
    atomic_min, atomic_geo, relation, graph, trace, cross, lexical = (float(value) for value in weights)
    return (
        atomic_min * _float(paper.get("v19_atomic_min"))
        + atomic_geo * _float(paper.get("v19_atomic_geo"))
        + relation * _float(paper.get("v19_relation_pct"))
        + graph * _float(paper.get("v17_graph_pct"))
        + trace * _float(paper.get("v17_trace_pct"))
        + cross * _float(paper.get("v17_cross_pct"))
        + lexical * _float(paper.get("v18_conjunction"))
    )


def evidence_dominance(candidate: Mapping[str, Any], incumbent: Mapping[str, Any], tolerance: float = 0.0) -> int:
    dimensions = (
        "v19_atomic_min", "v19_atomic_geo", "v19_relation_pct",
        "v17_graph_pct", "v17_trace_pct", "v17_cross_pct",
    )
    return sum(_float(candidate.get(key)) + tolerance >= _float(incumbent.get(key)) for key in dimensions)


def route_atomic_bottleneck_top20(
    incumbents: Sequence[dict[str, Any]],
    frontier: Sequence[dict[str, Any]],
    *,
    weights: Sequence[float] = DEFAULT_WEIGHTS,
    score_floor: float,
    margin: float,
    minimum_delta: float,
    dominance_required: int,
    protect_head: int = 15,
    max_promotions: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Route frontier papers only when their weakest condition is non-inferior."""

    current = [copy.deepcopy(paper) for paper in incumbents]
    incoming = sorted(
        (copy.deepcopy(paper) for paper in frontier),
        key=lambda paper: (
            -atomic_bottleneck_score(paper, weights),
            -_float(paper.get("v19_atomic_min")),
            normalize_title(paper.get("title")),
        ),
    )
    actions: list[dict[str, Any]] = []
    for candidate in incoming:
        if len(actions) >= max_promotions:
            break
        candidate_score = atomic_bottleneck_score(candidate, weights)
        if candidate_score < score_floor:
            break
        removable = list(enumerate(current[protect_head:], start=protect_head))
        if not removable:
            break
        index, incumbent = min(
            removable,
            key=lambda item: (
                atomic_bottleneck_score(item[1], weights),
                _float(item[1].get("v19_atomic_min")),
                -item[0],
            ),
        )
        incumbent_score = atomic_bottleneck_score(incumbent, weights)
        weak_delta = _float(candidate.get("v19_atomic_min")) - _float(incumbent.get("v19_atomic_min"))
        dominance = evidence_dominance(candidate, incumbent)
        if candidate_score < incumbent_score + margin:
            break
        if weak_delta < minimum_delta or dominance < dominance_required:
            continue
        current[index] = candidate
        actions.append({
            "incoming_title": candidate.get("title"),
            "outgoing_title": incumbent.get("title"),
            "incoming_score": candidate_score,
            "outgoing_score": incumbent_score,
            "margin": candidate_score - incumbent_score,
            "atomic_min": candidate.get("v19_atomic_min"),
            "atomic_geo": candidate.get("v19_atomic_geo"),
            "relation": candidate.get("v19_relation_pct"),
            "weak_delta": weak_delta,
            "dominance": dominance,
        })
    return current, actions
