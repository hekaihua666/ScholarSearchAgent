"""Portable linear relevance model trained on production-safe text features."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .relevance_selector import tokenize


FEATURE_NAMES = [
    "bias", "query_title_coverage", "query_abstract_coverage", "query_all_coverage",
    "title_jaccard", "all_jaccard", "title_bigram_coverage", "abstract_available",
    "query_length_log", "title_length_log", "abstract_length_log",
]


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def _coverage(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a) if a else 0.0


def _bigrams(tokens: set[str]) -> set[str]:
    ordered = sorted(tokens)
    return {f"{ordered[i]}::{ordered[i + 1]}" for i in range(len(ordered) - 1)}


def text_features(query: str, paper: dict[str, Any]) -> list[float]:
    q = tokenize(query); title = tokenize(paper.get("title")); abstract = tokenize(paper.get("abstract"))
    all_text = title | abstract
    qb, tb = _bigrams(q), _bigrams(title)
    return [
        1.0, _coverage(q, title), _coverage(q, abstract), _coverage(q, all_text),
        _jaccard(q, title), _jaccard(q, all_text), _coverage(qb, tb), float(bool(abstract)),
        math.log1p(len(q)) / 4, math.log1p(len(title)) / 4, math.log1p(len(abstract)) / 8,
    ]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -30, 30)
    return 1 / (1 + np.exp(-values))


@dataclass
class LinearSelectorModel:
    weights: list[float]
    means: list[float]
    scales: list[float]
    feature_names: list[str]
    l2: float = 0.05

    @classmethod
    def fit(
        cls, rows: list[dict[str, Any]], *, steps: int = 2500, learning_rate: float = 0.08,
        l2: float = 0.05,
    ) -> "LinearSelectorModel":
        x = np.asarray([text_features(r["query"], r) for r in rows], dtype=float)
        y = np.asarray([r["label"] for r in rows], dtype=float)
        sample = np.asarray([r.get("sample_weight", 1.0) for r in rows], dtype=float)
        means = x.mean(axis=0); scales = x.std(axis=0); means[0] = 0; scales[0] = 1
        scales[scales < 1e-8] = 1
        z = (x - means) / scales
        w = np.zeros(z.shape[1], dtype=float)
        weighted_pos = max(1e-9, float((sample * y).sum()))
        weighted_neg = max(1e-9, float((sample * (1 - y)).sum()))
        class_weight = np.where(y > 0, (weighted_pos + weighted_neg) / (2 * weighted_pos),
                                (weighted_pos + weighted_neg) / (2 * weighted_neg))
        effective = sample * class_weight
        for step in range(steps):
            p = _sigmoid(z @ w)
            grad = z.T @ (effective * (p - y)) / effective.sum() + l2 * np.r_[0, w[1:]]
            rate = learning_rate / math.sqrt(1 + step / 250)
            w -= rate * grad
        return cls(w.tolist(), means.tolist(), scales.tolist(), list(FEATURE_NAMES), l2)

    def predict_one(self, query: str, paper: dict[str, Any]) -> float:
        x = np.asarray(text_features(query, paper), dtype=float)
        z = (x - np.asarray(self.means)) / np.asarray(self.scales)
        return float(_sigmoid(np.asarray([np.dot(z, np.asarray(self.weights))]))[0])

    def to_dict(self) -> dict[str, Any]:
        return {"model_type": "linear_logistic", "weights": self.weights, "means": self.means,
                "scales": self.scales, "feature_names": self.feature_names, "l2": self.l2}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LinearSelectorModel":
        return cls(list(raw["weights"]), list(raw["means"]), list(raw["scales"]),
                   list(raw["feature_names"]), float(raw.get("l2", 0.05)))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
