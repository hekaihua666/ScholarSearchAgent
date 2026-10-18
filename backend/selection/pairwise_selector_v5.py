"""Lightweight query-grouped pairwise selector used by short sprint v5.

The model only consumes information available on a query/candidate pair.  In
particular, benchmark labels, injected-gold flags, gold ranks and expected
answer cardinality are deliberately excluded from the feature schema.
"""

from __future__ import annotations

import math
import pickle
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .relevance_selector import tokenize


FEATURE_NAMES: tuple[str, ...] = (
    "query_title_coverage",
    "query_abstract_coverage",
    "title_jaccard",
    "all_text_jaccard",
    "required_constraint_coverage",
    "preferred_constraint_coverage",
    "relation_compatibility",
    "method_task_dataset_coverage",
    "time_validity",
    "venue_type_compatibility",
    "abstract_missing",
    "trace_relative",
    "lexical_relative",
    "llm_relevance",
    "llm_missing",
    "source_count_saturation",
    "best_channel_reciprocal_rank",
    "rank_percentile",
    "score_gap_to_query_top",
    "metadata_completeness",
)

FORBIDDEN_FEATURE_MARKERS: tuple[str, ...] = (
    "gold",
    "label",
    "matched_gold",
    "gold_injected",
    "expected_cardinality",
)

_RELATION_WORDS = {
    "adopt", "adopted", "align", "aligned", "apply", "applied", "compare",
    "extend", "extended", "improve", "improved", "introduce", "introduced",
    "propose", "proposed", "use", "used", "using",
}
_TYPE_WORDS = {"benchmark", "dataset", "review", "survey", "system", "framework"}


def _bounded(value: Any, default: float = 0.0) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _coverage(required: set[str], observed: set[str]) -> float:
    return len(required & observed) / len(required) if required else 1.0


def _paper_year(row: dict[str, Any]) -> int | None:
    value = str(row.get("year") or row.get("published") or "")[:4]
    return int(value) if value.isdigit() else None


def _cutoff_year(row: dict[str, Any]) -> int | None:
    value = str(row.get("cutoff") or "")[:4]
    return int(value) if value.isdigit() else None


def _source_count(row: dict[str, Any]) -> int:
    explicit = row.get("source_count")
    if explicit is not None:
        try:
            return max(0, int(explicit))
        except (TypeError, ValueError):
            pass
    sources = row.get("source_apis") or row.get("sources")
    if isinstance(sources, (list, tuple, set)):
        return len(sources)
    text = str(row.get("source_api") or "")
    return len({part.strip() for part in re.split(r"[,;+|]", text) if part.strip()}) or int(bool(text))


def _normalise(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    low, high = min(values), max(values)
    return low, max(high - low, 1e-9)


def _raw_score(row: dict[str, Any], name: str) -> float:
    value = row.get(name)
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def feature_rows(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    """Return production-safe, within-query normalised feature vectors."""
    if not rows:
        return np.empty((0, len(FEATURE_NAMES)), dtype=float)
    trace_values = [_raw_score(row, "trace_score") for row in rows]
    lexical_values = [_raw_score(row, "lexical_score") for row in rows]
    trace_low, trace_span = _normalise(trace_values)
    lexical_low, lexical_span = _normalise(lexical_values)
    top_trace = max(trace_values) if trace_values else 0.0
    group_size = max(len(rows), max((int(row.get("candidate_pool_size") or 0) for row in rows), default=0), 1)

    vectors: list[list[float]] = []
    for row, trace, lexical in zip(rows, trace_values, lexical_values):
        query = tokenize(row.get("query"))
        title = tokenize(row.get("title"))
        abstract = tokenize(row.get("abstract"))
        all_text = title | abstract
        informative = {token for token in query if len(token) >= 5 or any(char.isdigit() for char in token)}
        preferred = {token for token in query if token in _TYPE_WORDS}
        relations = query & _RELATION_WORDS
        relation_coverage = _coverage(relations, all_text) if relations else 1.0
        year, cutoff = _paper_year(row), _cutoff_year(row)
        time_validity = 1.0 if year is None or cutoff is None or year <= cutoff else 0.0
        requested_types = query & _TYPE_WORDS
        venue_type = _coverage(requested_types, all_text) if requested_types else 1.0
        llm_raw = row.get("llm_relevance_score")
        rank = max(1, int(row.get("candidate_rank") or row.get("live_rank") or group_size))
        metadata_fields = (
            bool(row.get("title")), bool(row.get("abstract")), bool(year),
            bool(row.get("doi") or row.get("arxiv_id") or row.get("openalex_id") or row.get("paper_id")),
        )
        vectors.append([
            _coverage(query, title),
            _coverage(query, abstract),
            _jaccard(query, title),
            _jaccard(query, all_text),
            _coverage(query, all_text),
            _coverage(preferred, all_text) if preferred else 1.0,
            relation_coverage,
            _coverage(informative, all_text) if informative else _coverage(query, all_text),
            time_validity,
            venue_type,
            float(not bool(str(row.get("abstract") or "").strip())),
            (trace - trace_low) / trace_span,
            (lexical - lexical_low) / lexical_span,
            _bounded(llm_raw, 0.5),
            float(llm_raw is None),
            min(1.0, math.log1p(_source_count(row)) / math.log(4.0)),
            1.0 / rank,
            1.0 - min(1.0, (rank - 1) / max(group_size - 1, 1)),
            max(0.0, top_trace - trace),
            sum(metadata_fields) / len(metadata_fields),
        ])
    return np.asarray(vectors, dtype=float)


def grouped_feature_matrix(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    """Vectorise mixed-query rows while preserving input order."""
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["query_id"])].append((index, row))
    result = np.zeros((len(rows), len(FEATURE_NAMES)), dtype=float)
    for items in grouped.values():
        matrix = feature_rows([row for _, row in items])
        for (index, _), vector in zip(items, matrix):
            result[index] = vector
    return result


def build_pairwise_examples(
    rows: Sequence[dict[str, Any]], *, negatives_per_positive: int = 10, seed: int = 20260830,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Build deterministic symmetric positive-negative difference samples."""
    rng = random.Random(seed)
    matrix = grouped_feature_matrix(rows)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["query_id"])].append(index)
    samples: list[np.ndarray] = []
    labels: list[int] = []
    pair_indices: list[tuple[int, int]] = []
    for qid in sorted(grouped):
        indices = grouped[qid]
        positives = [index for index in indices if int(rows[index].get("label") or 0) == 1]
        negatives = [index for index in indices if int(rows[index].get("label") or 0) == 0]
        negatives.sort(key=lambda index: (
            not bool(rows[index].get("mined_hard_negative")),
            int(rows[index].get("candidate_rank") or 10_000),
            str(rows[index].get("paper_id") or rows[index].get("title") or ""),
        ))
        for positive in positives:
            if len(negatives) <= negatives_per_positive:
                chosen = negatives
            else:
                head = negatives[: min(4, negatives_per_positive)]
                tail = negatives[len(head):]
                chosen = head + rng.sample(tail, negatives_per_positive - len(head))
            for negative in chosen:
                difference = matrix[positive] - matrix[negative]
                samples.extend((difference, -difference))
                labels.extend((1, 0))
                pair_indices.extend(((positive, negative), (negative, positive)))
    if not samples:
        raise ValueError("pairwise training requires at least one positive and one negative in a query group")
    return np.asarray(samples, dtype=float), np.asarray(labels, dtype=int), pair_indices


def _new_classifier(*, class_weight: str | None = None) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            C=0.5, class_weight=class_weight, max_iter=2000, random_state=20260830,
        )),
    ])


@dataclass
class PairwiseSelectorV5:
    mode: str
    estimator: Pipeline
    calibrator: LogisticRegression | None = None
    feature_names: tuple[str, ...] = FEATURE_NAMES

    @classmethod
    def fit_pointwise(cls, rows: Sequence[dict[str, Any]]) -> "PairwiseSelectorV5":
        x = grouped_feature_matrix(rows)
        y = np.asarray([int(row.get("label") or 0) for row in rows], dtype=int)
        estimator = _new_classifier(class_weight="balanced")
        estimator.fit(x, y)
        return cls(mode="pointwise", estimator=estimator)

    @classmethod
    def fit_pairwise(
        cls, rows: Sequence[dict[str, Any]], *, negatives_per_positive: int = 10, seed: int = 20260830,
    ) -> "PairwiseSelectorV5":
        x, y, _ = build_pairwise_examples(rows, negatives_per_positive=negatives_per_positive, seed=seed)
        estimator = _new_classifier()
        estimator.fit(x, y)
        return cls(mode="pairwise", estimator=estimator)

    def decision_function(self, rows: Sequence[dict[str, Any]]) -> np.ndarray:
        return np.asarray(self.estimator.decision_function(grouped_feature_matrix(rows)), dtype=float)

    def fit_calibrator(self, rows: Sequence[dict[str, Any]]) -> "PairwiseSelectorV5":
        raw = self.decision_function(rows).reshape(-1, 1)
        labels = np.asarray([int(row.get("label") or 0) for row in rows], dtype=int)
        calibrator = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=20260830)
        calibrator.fit(raw, labels)
        self.calibrator = calibrator
        return self

    def predict_proba(self, rows: Sequence[dict[str, Any]]) -> np.ndarray:
        raw = self.decision_function(rows)
        if self.calibrator is not None:
            return np.asarray(self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1], dtype=float)
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -30, 30)))

    def feature_weights(self) -> dict[str, float]:
        scale = self.estimator.named_steps["scale"]
        model = self.estimator.named_steps["model"]
        coefficients = model.coef_[0] / np.asarray(scale.scale_)
        return dict(zip(self.feature_names, map(float, coefficients)))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: Path) -> "PairwiseSelectorV5":
        with path.open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError(f"unexpected model type: {type(model)!r}")
        return model


def mine_hard_negative_indices(
    model: PairwiseSelectorV5, rows: Sequence[dict[str, Any]], *, per_query: int = 3,
) -> list[int]:
    """Return deterministic top-scoring benchmark-negative row indices."""
    scores = model.decision_function(rows)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if int(row.get("label") or 0) == 0:
            grouped[str(row["query_id"])].append(index)
    selected: list[int] = []
    for qid in sorted(grouped):
        ordered = sorted(
            grouped[qid], key=lambda index: (-scores[index], str(rows[index].get("paper_id") or rows[index].get("title") or "")),
        )
        selected.extend(ordered[:per_query])
    return selected


def assert_safe_feature_schema(names: Iterable[str] = FEATURE_NAMES) -> None:
    for name in names:
        lowered = name.lower()
        if any(marker in lowered for marker in FORBIDDEN_FEATURE_MARKERS):
            raise AssertionError(f"forbidden label-derived selector feature: {name}")


assert_safe_feature_schema()
