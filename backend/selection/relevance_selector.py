"""Deterministic, evidence-bounded relevance selector.

The selector is deliberately separate from retrieval/ranking.  It consumes
production-available query and paper metadata and emits an auditable record;
it never sees benchmark gold IDs or the expected number of relevant papers.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Label = Literal["fully_relevant", "partially_relevant", "irrelevant", "insufficient_evidence"]

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_.+-]*|[\u4e00-\u9fff]", re.I)
_EMPTY = {"", "none", "not specified", "n/a", "unknown", "未指定", "无", "不限"}
_STOP = {
    "a", "an", "and", "are", "about", "any", "can", "for", "from", "in", "is", "me",
    "of", "on", "or", "paper", "papers", "research", "some", "studies", "study", "tell",
    "that", "the", "their", "to", "use", "using", "what", "which", "with", "work",
}


def tokenize(value: Any) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(str(value or "")) if token.lower() not in _STOP}


def _constraint_text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    text = str(value or "").strip()
    return "" if text.lower() in _EMPTY else text


def _coverage(required: set[str], text_tokens: set[str]) -> float:
    return len(required & text_tokens) / len(required) if required else 1.0


def _bounded(value: Any) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


@dataclass
class SelectionRecord:
    paper_id: str
    label: Label
    raw_score: float
    calibrated_probability: float
    hard_constraint_pass: bool
    facet_scores: dict[str, float] = field(default_factory=dict)
    matched_constraints: list[str] = field(default_factory=list)
    missing_constraints: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    uncertainty_reason: str = ""
    judge_source: str = "deterministic"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RelevanceSelector:
    """Low-cost first-stage selector with explicit constraint gates.

    Scores are rank features, not probabilities.  A fitted calibrator can be
    supplied by the experiment/serving layer to populate comparable
    probabilities before final-set selection.
    """

    FACETS = {
        "topic": ("research_topic",),
        "method": ("method_constraints",),
        "task": ("task_constraints",),
        "dataset_or_domain": ("dataset_constraints", "domain_constraints"),
        "venue_or_type": ("venue_constraints", "publication_type"),
    }

    def __init__(
        self,
        *,
        hard_constraint_min: float = 0.34,
        fully_relevant_threshold: float = 0.62,
        partially_relevant_threshold: float = 0.42,
        calibrator: Any | None = None,
    ) -> None:
        self.hard_constraint_min = hard_constraint_min
        self.fully_relevant_threshold = fully_relevant_threshold
        self.partially_relevant_threshold = partially_relevant_threshold
        self.calibrator = calibrator

    @staticmethod
    def paper_id(paper: dict[str, Any]) -> str:
        return str(
            paper.get("arxiv_id") or paper.get("doi") or paper.get("openalex_id")
            or paper.get("semantic_scholar_id") or paper.get("title") or ""
        )

    def evaluate(self, query: str, understanding: dict[str, Any], paper: dict[str, Any]) -> SelectionRecord:
        title = str(paper.get("title") or "")
        abstract = str(paper.get("abstract") or "")
        searchable = tokenize(f"{title} {abstract}")
        query_tokens = tokenize(query)
        title_tokens = tokenize(title)

        facet_scores: dict[str, float] = {}
        matched: list[str] = []
        missing: list[str] = []
        hard_values: list[float] = []
        for facet, keys in self.FACETS.items():
            raw = " ".join(filter(None, (_constraint_text(understanding.get(key)) for key in keys)))
            required = tokenize(raw)
            value = _coverage(required, searchable)
            facet_scores[facet] = round(value, 6)
            if required:
                hard_values.append(value)
                (matched if value >= self.hard_constraint_min else missing).append(f"{facet}:{raw}")

        topic_coverage = _coverage(query_tokens, searchable)
        title_coverage = _coverage(query_tokens, title_tokens)
        facet_scores["query"] = round(topic_coverage, 6)
        facet_scores["title"] = round(title_coverage, 6)

        trace = (
            0.24 * _bounded(paper.get("constraint_coverage_score"))
            + 0.18 * _bounded(paper.get("idf_lexical_score"))
            + 0.14 * _bounded(paper.get("lexical_score"))
            + 0.10 * _bounded(paper.get("semantic_provenance_score"))
            + 0.08 * _bounded(paper.get("rrf_score"))
        ) / 0.74
        llm_score = paper.get("llm_relevance_score")
        has_llm = llm_score is not None
        llm = _bounded(llm_score) if has_llm else 0.5
        min_hard = min(hard_values) if hard_values else topic_coverage
        mean_hard = sum(hard_values) / len(hard_values) if hard_values else topic_coverage
        raw_score = (
            0.25 * topic_coverage + 0.12 * title_coverage + 0.20 * mean_hard
            + 0.13 * min_hard + 0.17 * trace + 0.13 * llm
        )
        if missing:
            raw_score *= 0.55
        raw_score = _bounded(raw_score)
        probability = (
            _bounded(self.calibrator.predict_one(raw_score)) if self.calibrator is not None else raw_score
        )

        hard_pass = not missing
        insufficient = not abstract.strip() and len(title_tokens) < 4
        if insufficient:
            label: Label = "insufficient_evidence"
            uncertainty = "abstract_missing_and_title_sparse"
        elif hard_pass and probability >= self.fully_relevant_threshold:
            label = "fully_relevant"; uncertainty = ""
        elif probability >= self.partially_relevant_threshold:
            label = "partially_relevant"; uncertainty = "below_full_relevance_threshold"
        else:
            label = "irrelevant"; uncertainty = "low_relevance_evidence"

        evidence = []
        overlap = sorted(query_tokens & searchable)
        if overlap:
            evidence.append("matched_terms=" + ", ".join(overlap[:12]))
        if has_llm:
            evidence.append(f"llm_relevance={llm:.3f}")
        return SelectionRecord(
            paper_id=self.paper_id(paper), label=label, raw_score=round(raw_score, 6),
            calibrated_probability=round(probability, 6), hard_constraint_pass=hard_pass,
            facet_scores=facet_scores, matched_constraints=matched, missing_constraints=missing,
            evidence=evidence, uncertainty_reason=uncertainty,
        )

    def evaluate_many(
        self, query: str, understanding: dict[str, Any], papers: list[dict[str, Any]]
    ) -> list[SelectionRecord]:
        return [self.evaluate(query, understanding, paper) for paper in papers]
