"""Precision-bounded Top2 core plus optional one-rescue selector."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES: tuple[str, ...] = (
    "candidate_rank_percentile", "trace_relative", "llm_relative", "pointwise_relative",
    "pairwise_relative", "fusion_relative", "trace_gap_rank2", "llm_gap_rank2",
    "pointwise_gap_rank2", "pairwise_gap_rank2", "fusion_gap_rank2", "fusion_ratio_rank2",
    "fusion_gap_rank3", "source_count_saturation", "lexical_support", "semantic_support",
    "title_probe_support", "constraint_coverage", "hard_constraint_fail", "constraint_unknown_ratio",
    "abstract_available", "metadata_completeness", "max_core_text_similarity", "query_length_scaled",
    "query_constraint_count_scaled", "intent_multi_constraint", "intent_survey", "intent_frontier",
    "intent_comparison", "intent_foundation",
)

FORBIDDEN = ("gold", "label", "query_id", "paper_id", "title_hash", "gold_injected")


def assert_safe_schema(names: Sequence[str] = FEATURE_NAMES) -> None:
    for name in names:
        if any(marker in name.lower() for marker in FORBIDDEN):
            raise AssertionError(f"forbidden core-rescue feature: {name}")


def matrix(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    return np.asarray([[float(row.get(name) or 0.0) for name in FEATURE_NAMES] for row in rows], dtype=float)


def new_estimator(family: str) -> Any:
    if family == "logistic":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, random_state=20260830)),
        ])
    if family == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.08, max_iter=120, max_leaf_nodes=7, max_depth=3,
            min_samples_leaf=8, l2_regularization=1.0, random_state=20260830,
        )
    raise ValueError(f"unknown model family: {family}")


@dataclass
class CoreRescueModelV6:
    family: str
    estimator: Any
    threshold: float = 1.1
    near_duplicate_threshold: float = 0.85
    feature_names: tuple[str, ...] = FEATURE_NAMES

    @classmethod
    def fit(
        cls, rows: Sequence[dict[str, Any]], *, family: str = "logistic", sample_weight: np.ndarray | None = None,
    ) -> "CoreRescueModelV6":
        estimator = new_estimator(family)
        y = np.asarray([int(row.get("label") or 0) for row in rows], dtype=int)
        kwargs = {"model__sample_weight": sample_weight} if family == "logistic" and sample_weight is not None else {}
        if family == "hist_gradient_boosting" and sample_weight is not None:
            kwargs = {"sample_weight": sample_weight}
        estimator.fit(matrix(rows), y, **kwargs)
        return cls(family=family, estimator=estimator)

    def predict_proba(self, rows: Sequence[dict[str, Any]]) -> np.ndarray:
        if not rows: return np.asarray([], dtype=float)
        return np.asarray(self.estimator.predict_proba(matrix(rows))[:, 1], dtype=float)

    @staticmethod
    def rescue_eligible(row: dict[str, Any], near_duplicate_threshold: float = 0.85) -> bool:
        return (
            3 <= int(row.get("candidate_rank") or 0) <= 8
            and float(row.get("hard_constraint_fail") or 0.0) < 0.5
            and float(row.get("max_core_text_similarity") or 0.0) < near_duplicate_threshold
        )

    def choose_rescue(self, rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
        eligible = [row for row in rows if self.rescue_eligible(row, self.near_duplicate_threshold)]
        if not eligible: return None
        probabilities = self.predict_proba(eligible)
        index = int(np.argmax(probabilities))
        if float(probabilities[index]) < self.threshold: return None
        return eligible[index]

    def select(self, core: Sequence[dict[str, Any]], boundary: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        selected = list(core[:2])
        rescue = self.choose_rescue(boundary)
        if rescue is not None: selected.append(rescue)
        return selected

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle: pickle.dump(self, handle)

    @classmethod
    def load(cls, path: Path) -> "CoreRescueModelV6":
        with path.open("rb") as handle: model = pickle.load(handle)
        if not isinstance(model, cls): raise TypeError(type(model))
        return model


assert_safe_schema()
