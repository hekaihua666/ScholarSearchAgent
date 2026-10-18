"""Lightweight evidence-calibrated selector for the precision-first result set.

The selector combines the ranking prior with TRACE, local semantic, graph,
constraint and metadata evidence.  Training happens offline; runtime scoring is
just a standardized linear projection and therefore adds no API or LLM calls.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


BASE_FEATURE_NAMES: tuple[str, ...] = (
    "v17_cross_raw",
    "v17_cross_pct",
    "v17_trace_pct",
    "v17_graph_raw",
    "v17_graph_pct",
    "v18_condition_average",
    "v18_condition_minimum",
    "v18_conjunction",
    "v18_relation",
    "v18_direction",
    "v18_seed_bridge",
    "lexical_score",
    "idf_lexical_score",
    "title_hypothesis_score",
    "generic_result_penalty",
    "citation_provenance_score",
    "semantic_provenance_score",
    "intent_bonus_score",
    "rrf_score",
    "constraint_coverage_score",
    "authority_score",
    "freshness_score",
    "metadata_score",
    "keyword_score",
    "openalex_score",
    "citation_score",
    "final_score",
    "relevance_score",
    "diversity_adjusted_score",
    "llm_relevance_score",
    "evidence_strength",
    "cited_by_log",
    "year_scaled",
)

DERIVED_FEATURE_NAMES: tuple[str, ...] = (
    "title_query_overlap",
    "abstract_query_overlap",
    "rank_prior",
    "incumbent_flag",
    "cross_graph_product",
    "cross_graph_minimum",
    "cross_graph_maximum",
)

FEATURE_NAMES: tuple[str, ...] = BASE_FEATURE_NAMES + DERIVED_FEATURE_NAMES

_STOPWORDS = {
    "which", "what", "papers", "paper", "studies", "study", "works", "work",
    "research", "are", "there", "first", "used", "using", "use", "about",
    "consider", "discuss", "focus", "focusing", "based", "methods", "method",
    "recently", "context", "provide", "examples",
}


def _float(value: Any) -> float:
    try:
        result = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _tokens(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(value or "").lower())) - _STOPWORDS


def _overlap(query_tokens: set[str], value: Any) -> float:
    return len(query_tokens & _tokens(value)) / max(1, len(query_tokens))


def selector_feature_vector(
    query: str,
    paper: dict[str, Any],
    *,
    rank: int,
    incumbent_count: int = 20,
) -> list[float]:
    """Build the fixed, label-free runtime feature vector."""

    cross = _float(paper.get("v17_cross_pct"))
    graph = _float(paper.get("v17_graph_pct"))
    year = _float(paper.get("year"))
    query_tokens = _tokens(query)
    numeric = [
        _float(paper.get(name))
        for name in BASE_FEATURE_NAMES[:-2]
    ]
    numeric.extend((
        math.log1p(max(0.0, _float(paper.get("cited_by_count")))) / 10.0,
        (year - 1990.0) / 40.0 if year else 0.0,
    ))
    numeric.extend((
        _overlap(query_tokens, paper.get("title")),
        _overlap(query_tokens, str(paper.get("abstract") or "")[:1800]),
        1.0 / math.sqrt(max(1, int(rank))),
        float(int(rank) <= max(0, int(incumbent_count))),
        cross * graph,
        min(cross, graph),
        max(cross, graph),
    ))
    if len(numeric) != len(FEATURE_NAMES):
        raise AssertionError((len(numeric), len(FEATURE_NAMES)))
    return numeric


@dataclass(frozen=True)
class EvidenceCalibratedSelectorV22:
    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    core_size: int = 1

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvidenceCalibratedSelectorV22":
        model = payload.get("model") or payload
        selector = cls(
            feature_names=tuple(model["feature_names"]),
            mean=tuple(float(value) for value in model["mean"]),
            scale=tuple(float(value) for value in model["scale"]),
            coefficients=tuple(float(value) for value in model["coefficients"]),
            intercept=float(model.get("intercept") or 0.0),
            core_size=max(1, int(model.get("core_size") or 1)),
        )
        selector._validate()
        return selector

    @classmethod
    def load(cls, path: Path) -> "EvidenceCalibratedSelectorV22":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def _validate(self) -> None:
        if self.feature_names != FEATURE_NAMES:
            raise ValueError("selector feature schema does not match runtime schema")
        lengths = {len(self.feature_names), len(self.mean), len(self.scale), len(self.coefficients)}
        if len(lengths) != 1:
            raise ValueError("selector parameter lengths are inconsistent")
        if any(value <= 0 or not math.isfinite(value) for value in self.scale):
            raise ValueError("selector scales must be finite and positive")

    def logit(self, vector: Sequence[float]) -> float:
        if len(vector) != len(self.feature_names):
            raise ValueError("selector feature vector has the wrong length")
        return self.intercept + sum(
            coefficient * ((float(value) - mean) / scale)
            for value, mean, scale, coefficient in zip(
                vector, self.mean, self.scale, self.coefficients
            )
        )

    @staticmethod
    def _sigmoid(value: float) -> float:
        value = max(-35.0, min(35.0, value))
        return 1.0 / (1.0 + math.exp(-value))

    def score_candidates(
        self,
        query: str,
        papers: Sequence[dict[str, Any]],
        *,
        incumbent_count: int = 20,
    ) -> list[tuple[int, float]]:
        scores: list[tuple[int, float]] = []
        for index, paper in enumerate(papers):
            vector = selector_feature_vector(
                query, paper, rank=index + 1, incumbent_count=incumbent_count
            )
            scores.append((index, self._sigmoid(self.logit(vector))))
        return sorted(scores, key=lambda item: (-item[1], item[0]))

    def select(
        self,
        query: str,
        papers: Sequence[dict[str, Any]],
        *,
        incumbent_count: int = 20,
        core_size: int | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not papers:
            return [], {"strategy": "evidence_calibrated_precision", "selected_count": 0}
        size = max(1, int(core_size or self.core_size))
        ordered = self.score_candidates(query, papers, incumbent_count=incumbent_count)
        chosen = ordered[:size]
        selected: list[dict[str, Any]] = []
        for index, probability in chosen:
            paper = dict(papers[index])
            paper["evidence_calibrated_probability"] = probability
            paper["evidence_calibrated_source_rank"] = index + 1
            selected.append(paper)
        return selected, {
            "strategy": "evidence_calibrated_precision",
            "selector": "ECS_v22",
            "selected_count": len(selected),
            "core_size": size,
            "runtime": "local_linear_projection",
        }


__all__ = [
    "BASE_FEATURE_NAMES",
    "DERIVED_FEATURE_NAMES",
    "FEATURE_NAMES",
    "EvidenceCalibratedSelectorV22",
    "selector_feature_vector",
]
