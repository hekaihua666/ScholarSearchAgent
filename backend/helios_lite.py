"""HELIOS-Lite: training-free, gap/cost-aware retrieval action control.

The controller never consumes benchmark gold.  Its priors are aggregate,
shrunken development statistics and its online state contains only production
signals available at search time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from backend.trace_rank import canonical_key


@dataclass
class HELIOSState:
    query_intent: str
    hard_constraints: list[str]
    soft_preferences: list[str]
    coverage_vector: dict[str, float]
    high_confidence_papers: list[str]
    unexpanded_papers: list[str]
    expanded_papers: list[str] = field(default_factory=list)
    past_queries: list[str] = field(default_factory=list)
    past_actions: list[str] = field(default_factory=list)
    channel_results: dict[str, int] = field(default_factory=dict)
    candidate_count: int = 0
    accepted_candidate_count: int = 0
    duplicate_ratio: float = 0.0
    estimated_drift: float = 0.0
    api_budget_remaining: int = 8
    llm_budget_remaining: int = 1
    elapsed_cost: float = 0.0
    consecutive_no_gain_actions: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionEstimate:
    action: str
    predicted_utility: float
    predicted_novel_relevant: float
    predicted_constraint_gap_closure: float
    predicted_duplicates: float
    predicted_drift: float
    normalized_latency: float
    api_cost: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Aggregate dev priors, shrunk toward a conservative global mean.  They are
# candidate-discovery priors, not online labels and not query-specific patches.
ACTION_PRIORS = {
    "citation_expand": {"novel": 0.045, "duplicates": 0.58, "drift": 0.22, "latency": 0.27, "api": 0.28},
    "second_round_search": {"novel": 0.075, "duplicates": 0.46, "drift": 0.34, "latency": 0.81, "api": 0.50},
    "semantic_search": {"novel": 0.11, "duplicates": 0.34, "drift": 0.18, "latency": 0.20, "api": 0.12},
    "arxiv_search": {"novel": 0.025, "duplicates": 0.42, "drift": 0.20, "latency": 0.32, "api": 0.25},
    "title_probe": {"novel": 0.04, "duplicates": 0.30, "drift": 0.16, "latency": 0.17, "api": 0.25},
    "recommend_from_seeds": {"novel": 0.055, "duplicates": 0.48, "drift": 0.25, "latency": 0.30, "api": 0.25},
}


def build_state(query: str, understanding: dict[str, Any], papers: list[dict[str, Any]], api_budget: int = 8) -> HELIOSState:
    top = papers[:20]
    facets = ("research_topic", "method_constraints", "task_constraints", "domain_constraints")
    coverage = {
        facet: round(sum(float((paper.get("constraint_coverage") or {}).get(facet, 0)) for paper in top) / max(len(top), 1), 4)
        for facet in facets if str(understanding.get(facet) or "").lower() not in {"", "not specified", "unknown", "n/a"}
    }
    high = [canonical_key(paper) for paper in top if float(paper.get("final_score") or 0) >= 0.35][:8]
    raw_count = sum(len(paper.get("retrieval_hits") or []) for paper in papers)
    duplicate_ratio = max(0.0, 1.0 - len(papers) / max(raw_count, len(papers), 1))
    drift = 1.0 - sum(float(paper.get("constraint_coverage_score") or 0) for paper in top[:10]) / max(min(len(top), 10), 1)
    hard = [str(understanding.get(key)) for key in facets if str(understanding.get(key) or "").lower() not in {"", "not specified", "unknown", "n/a"}]
    return HELIOSState(
        query_intent=str(understanding.get("search_intent") or "broad"), hard_constraints=hard,
        soft_preferences=[], coverage_vector=coverage, high_confidence_papers=high,
        unexpanded_papers=high.copy(), past_queries=[query], channel_results={"first_round": len(papers)},
        candidate_count=len(papers), accepted_candidate_count=len(papers), duplicate_ratio=round(duplicate_ratio, 4),
        estimated_drift=round(max(0.0, min(drift, 1.0)), 4), api_budget_remaining=api_budget,
    )


class HELIOSLiteController:
    def __init__(self, *, use_cost: bool = True, use_gap: bool = True, use_drift: bool = True):
        self.use_cost = use_cost
        self.use_gap = use_gap
        self.use_drift = use_drift

    def estimate(self, action: str, state: HELIOSState) -> ActionEstimate:
        if action == "stop":
            return ActionEstimate("stop", 0.0, 0, 0, 0, 0, 0, 0, "No remaining action has positive predicted utility.")
        prior = ACTION_PRIORS[action]
        gap = 1.0 - (sum(state.coverage_vector.values()) / max(len(state.coverage_vector), 1))
        intent_multiplier = 1.0
        if action == "citation_expand" and state.query_intent in {"foundation", "survey", "method_lineage"}:
            intent_multiplier = 1.8
        elif action == "second_round_search" and state.query_intent in {"comparison", "frontier", "broad"}:
            intent_multiplier = 1.45
        novelty = prior["novel"] * intent_multiplier * (0.65 + 0.35 * gap)
        gap_gain = novelty * gap if self.use_gap else 0.0
        drift = prior["drift"] * (0.6 + 0.4 * state.estimated_drift) if self.use_drift else 0.0
        duplicate = prior["duplicates"] * (0.7 + 0.3 * state.duplicate_ratio)
        # Gap estimates derived from title/abstract coverage are noisy.  Treat
        # them as a calibrated modifier rather than letting a large apparent
        # gap override weak novelty and high action cost.
        utility = 1.4 * novelty + 0.3 * gap_gain + 0.05 - 0.12 * duplicate - 0.16 * drift
        if self.use_cost:
            utility -= 0.10 * prior["latency"] + 0.10 * prior["api"]
        if state.api_budget_remaining <= 0:
            utility = -1.0
        return ActionEstimate(
            action, round(utility, 6), round(novelty, 6), round(gap_gain, 6),
            round(duplicate, 6), round(drift, 6), prior["latency"], prior["api"],
            f"intent={state.query_intent}; gap={gap:.3f}; cost={'on' if self.use_cost else 'off'}",
        )

    def choose(self, state: HELIOSState, allowed: list[str] | None = None) -> tuple[ActionEstimate, list[ActionEstimate]]:
        if state.consecutive_no_gain_actions >= 2 or state.api_budget_remaining <= 0:
            stop = self.estimate("stop", state)
            return stop, [stop]
        actions = allowed or ["citation_expand", "second_round_search"]
        estimates = [self.estimate(action, state) for action in actions]
        best = max(estimates, key=lambda item: (item.predicted_utility, item.action))
        if best.predicted_utility <= 0:
            stop = self.estimate("stop", state)
            return stop, estimates + [stop]
        return best, estimates


def admit_candidates(candidates: list[dict[str, Any]], *, enabled: bool = True, limit: int = 50) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not enabled:
        return candidates[:limit], []
    accepted=[]; rejected=[]
    for paper in candidates:
        evidence=max(float(paper.get("lexical_score") or 0),float(paper.get("idf_lexical_score") or 0),float(paper.get("constraint_coverage_score") or 0))
        if evidence < 0.10:
            rejected.append({"paper_id":canonical_key(paper),"reason":"insufficient_query_evidence"})
        else:
            accepted.append(paper)
    return accepted[:limit], rejected
