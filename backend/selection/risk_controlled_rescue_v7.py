"""Query-relative rescue ranking and query-level risk gate for v7."""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .core_rescue_selector_v6 import FEATURE_NAMES as V6_FEATURE_NAMES


RANK_FEATURE_NAMES: tuple[str, ...] = (
    *V6_FEATURE_NAMES,
    "original_rank",
    "reciprocal_rank",
    "fusion_percentile_rescue",
    "fusion_zscore_rescue",
    "fusion_minus_runner_up",
    "llm_trace_agreement",
    "constraint_coverage_relative",
)

GATE_FEATURE_NAMES: tuple[str, ...] = (
    "candidate_absolute_score",
    "candidate_relative_score",
    "candidate_minus_runner_up",
    "candidate_minus_rank2_core",
    "ranker_ensemble_agreement",
    "llm_trace_agreement",
    "high_scoring_candidate_count",
    "rescue_score_entropy",
    "core_rank2_confidence",
    "intent_multi_constraint",
    "intent_survey",
    "intent_frontier",
    "intent_comparison",
    "intent_foundation",
)

_FORBIDDEN = ("gold", "label", "query_id", "paper_id", "identity", "delta_f1")


def assert_safe_schemas() -> None:
    for schema in (RANK_FEATURE_NAMES, GATE_FEATURE_NAMES):
        for name in schema:
            if any(marker in name.lower() for marker in _FORBIDDEN):
                raise AssertionError(f"forbidden inference feature: {name}")


def _matrix(rows: Sequence[dict[str, Any]], names: Sequence[str]) -> np.ndarray:
    return np.asarray([[float(row.get(name) or 0.0) for name in names] for row in rows], dtype=float)


def _new_logistic(c: float = 0.5) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=20260830)),
    ])


def augment_query_relative(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add deterministic ranks-3--8 normalization without using labels."""
    output = [dict(row) for row in rows]
    by_query: dict[str, list[dict[str, Any]]] = {}
    for row in output: by_query.setdefault(str(row["query_id"]), []).append(row)
    for candidates in by_query.values():
        candidates.sort(key=lambda row: int(row.get("candidate_rank") or 10_000))
        fusion = np.asarray([float(row.get("fusion_score") or 0.0) for row in candidates])
        mean, std = float(fusion.mean()), max(float(fusion.std()), 1e-9)
        order = sorted(range(len(candidates)), key=lambda index: (-fusion[index], int(candidates[index].get("candidate_rank") or 10_000)))
        percentile = {index: 1.0 - rank / max(len(candidates)-1, 1) for rank, index in enumerate(order)}
        for index, row in enumerate(candidates):
            other = [fusion[j] for j in range(len(candidates)) if j != index]
            runner_up = max(other) if other else fusion[index]
            llm = float(row.get("llm_relative") or 0.0); trace = float(row.get("trace_relative") or 0.0)
            coverage = [float(item.get("constraint_coverage") or 0.0) for item in candidates]
            low, high = min(coverage), max(coverage); span = max(high-low, 1e-9)
            row.update({
                "original_rank": float(row.get("candidate_rank") or 8) / 8.0,
                "reciprocal_rank": 1.0 / max(int(row.get("candidate_rank") or 8), 1),
                "fusion_percentile_rescue": percentile[index],
                "fusion_zscore_rescue": (fusion[index]-mean)/std,
                "fusion_minus_runner_up": float(fusion[index]-runner_up),
                "llm_trace_agreement": 1.0-abs(llm-trace),
                "constraint_coverage_relative": (float(row.get("constraint_coverage") or 0.0)-low)/span if high>low else 0.5,
            })
    return output


@dataclass
class RescueRankerV7:
    variant: str
    estimator: Pipeline
    feature_names: tuple[str, ...]

    @classmethod
    def fit(cls, rows: Sequence[dict[str, Any]], *, variant: str = "query_relative", c: float = 0.5) -> "RescueRankerV7":
        names = V6_FEATURE_NAMES if variant == "v6" else RANK_FEATURE_NAMES
        estimator = _new_logistic(c); estimator.fit(_matrix(rows, names), [int(row.get("label") or 0) for row in rows])
        return cls(variant, estimator, tuple(names))

    def predict(self, rows: Sequence[dict[str, Any]]) -> np.ndarray:
        return np.asarray(self.estimator.predict_proba(_matrix(rows, self.feature_names))[:, 1], dtype=float)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle: pickle.dump(self, handle)

    @classmethod
    def load(cls, path: Path) -> "RescueRankerV7":
        with path.open("rb") as handle: value=pickle.load(handle)
        if not isinstance(value, cls): raise TypeError(type(value))
        return value


def exact_delta_f1(core_tp: int, gold_count: int, candidate_is_gold: bool) -> float:
    def f1(tp: int, predicted: int) -> float:
        precision = tp / predicted if predicted else 0.0
        recall = tp / gold_count if gold_count else 0.0
        return 2*precision*recall/(precision+recall) if precision+recall else 0.0
    return f1(core_tp + int(candidate_is_gold), 3) - f1(core_tp, 2)


def gate_row(
    candidates: Sequence[dict[str, Any]], scores: Sequence[float], selected_index: int,
    *, ensemble_agreement: float = 1.0, gold_count: int | None = None,
) -> dict[str, Any]:
    selected = candidates[selected_index]; values=np.asarray(scores,dtype=float)
    ordered=np.sort(values)[::-1]; best=float(values[selected_index]); runner=float(ordered[1]) if len(ordered)>1 else best
    low, high=float(values.min()),float(values.max()); span=max(high-low,1e-9)
    shifted=values-values.max(); probabilities=np.exp(shifted)/np.exp(shifted).sum()
    entropy=-float(np.sum(probabilities*np.log(np.maximum(probabilities,1e-12))))/max(math.log(len(values)),1e-9)
    relative=(best-low)/span if high>low else .5
    gate={
        "query_id": selected.get("query_id"), "selected_paper_id": selected.get("paper_id"),
        # Audit-only outcome context.  This is deliberately excluded from
        # GATE_FEATURE_NAMES, but is required to score the simulated final set.
        "core_tp": int(selected.get("core_tp") or 0),
        "candidate_absolute_score": best, "candidate_relative_score": relative,
        "candidate_minus_runner_up": best-runner,
        "candidate_minus_rank2_core": float(selected.get("fusion_gap_rank2") or 0.0),
        "ranker_ensemble_agreement": ensemble_agreement,
        "llm_trace_agreement": float(selected.get("llm_trace_agreement") or 0.0),
        "high_scoring_candidate_count": float(np.sum(values >= best-.10))/max(len(values),1),
        "rescue_score_entropy": entropy,
        "core_rank2_confidence": float(selected.get("fusion_score") or 0.0)-float(selected.get("fusion_gap_rank2") or 0.0),
        **{name: float(selected.get(name) or 0.0) for name in (
            "intent_multi_constraint","intent_survey","intent_frontier","intent_comparison","intent_foundation"
        )},
    }
    if "label" in selected:
        gate["y_gold"] = int(selected.get("label") or 0)
        if gold_count is not None:
            delta=exact_delta_f1(int(selected.get("core_tp") or 0),gold_count,bool(gate["y_gold"]))
            gate["exact_delta_f1"] = delta; gate["y_gate"] = int(delta>0)
    return gate


@dataclass
class QueryGateV7:
    estimator: Pipeline
    feature_names: tuple[str, ...] = GATE_FEATURE_NAMES

    @classmethod
    def fit(cls, rows: Sequence[dict[str, Any]], c: float = 0.5) -> "QueryGateV7":
        estimator=_new_logistic(c); estimator.fit(_matrix(rows,GATE_FEATURE_NAMES),[int(row.get("y_gate") or 0) for row in rows])
        return cls(estimator)

    def predict(self, rows: Sequence[dict[str, Any]]) -> np.ndarray:
        return np.asarray(self.estimator.predict_proba(_matrix(rows,self.feature_names))[:,1],dtype=float)

    def save(self,path:Path)->None:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("wb") as handle:pickle.dump(self,handle)

    @classmethod
    def load(cls,path:Path)->"QueryGateV7":
        with path.open("rb") as handle:value=pickle.load(handle)
        if not isinstance(value,cls):raise TypeError(type(value))
        return value


assert_safe_schemas()
