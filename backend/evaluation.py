"""Reproducible retrieval metrics used by the competition evaluation harness."""

from __future__ import annotations

import math
from typing import Iterable


def retrieval_metrics(predicted: Iterable[str], relevant: Iterable[str], k: int = 20) -> dict[str, float]:
    pred = list(dict.fromkeys(str(x) for x in predicted if x))[:k]
    gold = {str(x) for x in relevant if x}
    hits = [1 if item in gold else 0 for item in pred]
    tp = sum(hits)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hits))
    ideal_hits = min(len(gold), k)
    idcg = sum(1 / math.log2(i + 2) for i in range(ideal_hits))
    ndcg = dcg / idcg if idcg else 0.0
    return {
        f"precision@{k}": round(precision, 6),
        f"recall@{k}": round(recall, 6),
        f"f1@{k}": round(f1, 6),
        f"ndcg@{k}": round(ndcg, 6),
        "true_positive": tp,
        "predicted_count": len(pred),
        "relevant_count": len(gold),
    }


def macro_average(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    metric_keys = [k for k in rows[0] if k.startswith(("precision@", "recall@", "f1@", "ndcg@"))]
    return {k: round(sum(float(row[k]) for row in rows) / len(rows), 6) for k in metric_keys}


def set_f1_metrics(predicted: Iterable[str], relevant: Iterable[str]) -> dict[str, float]:
    """Candidate competition metric: F1 over the complete returned set.

    This is deliberately named as a candidate protocol until the organizer's
    private evaluator and submission schema are available.
    """
    pred = list(dict.fromkeys(str(x) for x in predicted if x))
    gold = {str(x) for x in relevant if x}
    tp = len(set(pred) & gold)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * tp / (len(pred) + len(gold)) if pred or gold else 0.0
    return {
        "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6),
        "true_positive": tp, "predicted_count": len(pred), "relevant_count": len(gold),
    }
