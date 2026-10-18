"""Query-adaptive conversion of ranked probabilities into a result set."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


Strategy = Literal["fixed_k", "global_threshold", "expected_f1", "score_gap", "dual_threshold"]


@dataclass
class SetDecision:
    selected_indices: list[int]
    supplementary_indices: list[int]
    strategy: str
    expected_f1: float
    reason: str


def _expected_f1(ps: list[float], k: int) -> float:
    expected_tp = sum(ps[:k])
    expected_gold = sum(ps)
    return 2 * expected_tp / (k + expected_gold) if k + expected_gold else 0.0


def select_final_set(
    probabilities: list[float], *, strategy: Strategy = "expected_f1", fixed_k: int = 20,
    threshold: float = 0.5, low_threshold: float = 0.3, max_results: int = 20,
    min_results: int = 0, rescue_mask: list[bool] | None = None, score_gap: float = 0.12,
) -> SetDecision:
    """Select from probabilities already ordered by final relevance rank."""
    ps = [min(1.0, max(0.0, float(p))) for p in probabilities]
    cap = min(max_results, len(ps))
    if strategy == "fixed_k":
        chosen = list(range(min(fixed_k, cap)))
    elif strategy == "global_threshold":
        chosen = [i for i, p in enumerate(ps[:cap]) if p >= threshold]
    elif strategy == "expected_f1":
        candidates = [(k, _expected_f1(ps, k)) for k in range(0, cap + 1)]
        best_k, _ = max(candidates, key=lambda item: (item[1], -item[0]))
        chosen = list(range(best_k))
    elif strategy == "score_gap":
        chosen = [i for i, p in enumerate(ps[:cap]) if p >= threshold]
        if chosen:
            for i in range(len(chosen), cap):
                if ps[i - 1] - ps[i] >= score_gap: break
                if ps[i] >= low_threshold: chosen.append(i)
    elif strategy == "dual_threshold":
        mask = rescue_mask or [False] * len(ps)
        chosen = [i for i, p in enumerate(ps[:cap]) if p >= threshold]
        chosen.extend(i for i, p in enumerate(ps[:cap]) if low_threshold <= p < threshold and mask[i])
        chosen = sorted(set(chosen))[:cap]
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    if len(chosen) < min_results:
        chosen = list(range(min(min_results, cap)))
    selected = sorted(set(chosen))[:cap]
    supplementary = [i for i in range(cap) if i not in selected]
    return SetDecision(
        selected_indices=selected, supplementary_indices=supplementary, strategy=strategy,
        expected_f1=round(_expected_f1(ps, len(selected)), 6),
        reason=f"{strategy}: selected {len(selected)} of {len(ps)} candidates",
    )
