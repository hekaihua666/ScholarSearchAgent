"""Recall-constrained core/rescue set decisions."""

from __future__ import annotations

from typing import Any, Callable

from backend.trace_rank import canonical_key


def select_core_rescue(
    papers: list[dict[str, Any]],
    probability: Callable[[dict[str, Any]], float],
    core_threshold: float,
    rescue_threshold: float,
    min_results: int = 2,
    max_results: int = 8,
) -> list[dict[str, Any]]:
    """Select high-confidence core then fill from a stricter evidence-backed rescue queue."""
    ordered = sorted(papers, key=probability, reverse=True)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def admit(paper: dict[str, Any], decision: str) -> None:
        key = canonical_key(paper)
        if key and key not in seen and len(selected) < max_results:
            item = dict(paper)
            item["set_decision"] = decision
            item["selection_probability"] = round(probability(paper), 6)
            selected.append(item)
            seen.add(key)

    for paper in ordered:
        hard_gate = str(paper.get("hard_gate") or paper.get("ace_hard_gate") or "unknown")
        if hard_gate != "fail" and probability(paper) >= core_threshold:
            admit(paper, "accept")
    for paper in ordered:
        if len(selected) >= max_results:
            break
        hard_gate = str(paper.get("hard_gate") or paper.get("ace_hard_gate") or "unknown")
        support = int(float(paper.get("llm_relevance_score") or 0) >= 0.75) + int(
            float(paper.get("cross_encoder_probability") or 0) >= 0.65
        ) + int(float(paper.get("ace_probability") or 0) >= 0.55)
        if hard_gate != "fail" and probability(paper) >= rescue_threshold and support >= 2:
            admit(paper, "rescue")
    for paper in ordered:
        if len(selected) >= min_results:
            break
        if str(paper.get("hard_gate") or paper.get("ace_hard_gate") or "unknown") != "fail":
            admit(paper, "minimum_cardinality_rescue")
    return selected[:max_results]
