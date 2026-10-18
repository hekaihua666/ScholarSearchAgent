"""Query-grouped selector calibration and recall-constrained set decisions.

The two production captures are used as one development population.  Every
threshold is selected from 5-fold out-of-fold predictions.  ``dev20`` is only
evaluated after the choices have been made.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.selection.pasa_selector_adapter import runtime_status as pasa_status

OUT = ROOT / "reports/joint_optimization_v4"
DATASET = Path(r"D:\AICompetition\references\github\SPAR\benchmark\AutoScholarQuery_test.jsonl")
RECALL_FLOOR = 0.35


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def feature(row: dict[str, Any], names: tuple[str, ...]) -> list[float]:
    values = {
        "llm": float(row.get("llm_relevance_score") or 0),
        "cross": sigmoid(float(row.get("cross_encoder_raw") or -10)),
        "ace": float((row.get("ace") or {}).get("raw_probability") or 0),
        "trace": float(row.get("trace_score") or 0),
        "rank": 1.0 / max(int(row.get("live_rank") or row.get("candidate_rank") or 100), 1),
        "abstract": float(bool(row.get("abstract"))),
        "coverage": float(row.get("constraint_coverage_score") or 0),
    }
    return [values[name] for name in names]


def matrix(rows: list[dict[str, Any]], names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray([feature(row, names) for row in rows], dtype=float),
        np.asarray([int(row["label"]) for row in rows], dtype=int),
        np.asarray([row["query_id"] for row in rows]),
    )


def ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 8) -> float:
    edges = np.linspace(0, 1, bins + 1)
    result = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= low) & (probabilities < high if high < 1 else probabilities <= high)
        if mask.any():
            result += mask.mean() * abs(probabilities[mask].mean() - labels[mask].mean())
    return float(result)


def new_model() -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=20260830),
    )


def grouped_oof(rows: list[dict[str, Any]], names: tuple[str, ...], folds: int = 5) -> tuple[np.ndarray, dict[str, float]]:
    x, y, groups = matrix(rows, names)
    predictions = np.zeros(len(rows), dtype=float)
    splitter = GroupKFold(n_splits=folds)
    for train_idx, test_idx in splitter.split(x, y, groups):
        model = new_model()
        model.fit(x[train_idx], y[train_idx])
        predictions[test_idx] = model.predict_proba(x[test_idx])[:, 1]
    return predictions, {
        "pr_auc": float(average_precision_score(y, predictions)),
        "brier": float(brier_score_loss(y, predictions)),
        "ece": ece(y, predictions),
    }


def gold_counts() -> dict[str, int]:
    return {row["qid"]: len(row.get("answer") or []) for row in map(json.loads, DATASET.open(encoding="utf-8"))}


def set_metrics(rows: list[dict[str, Any]], selector: Callable[[list[dict[str, Any]]], list[dict[str, Any]]]) -> dict[str, float]:
    counts = gold_counts()
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["query_id"]].append(row)
    values = []
    for qid, candidates in groups.items():
        selected = selector(candidates)
        tp = sum(int(row["label"]) for row in selected)
        predicted, gold = len(selected), counts[qid]
        precision = tp / predicted if predicted else 0.0
        recall = tp / gold if gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        values.append((precision, recall, f1, predicted, tp))
    labels = ("precision", "recall", "f1", "mean_output_count", "mean_tp")
    return {name: round(sum(row[i] for row in values) / len(values), 6) for i, name in enumerate(labels)}


def attach(rows: list[dict[str, Any]], probabilities: np.ndarray, field: str) -> None:
    for row, probability in zip(rows, probabilities):
        row[field] = float(probability)


def threshold_selector(field: str, threshold: float, max_results: int = 8, min_results: int = 2):
    def select(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ordered = sorted(rows, key=lambda row: float(row.get(field) or 0), reverse=True)
        selected = [row for row in ordered if float(row.get(field) or 0) >= threshold][:max_results]
        if len(selected) < min_results:
            selected = ordered[:min_results]
        return selected
    return select


def top_live(k: int):
    def select(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        eligible = [row for row in rows if row.get("live_rank") and int(row["live_rank"]) <= 20]
        return sorted(eligible, key=lambda row: int(row["live_rank"]))[:k]
    return select


def llm090(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [row for row in rows if row.get("live_rank") and int(row["live_rank"]) <= 20]
    return [
        row for row in sorted(eligible, key=lambda item: float(item.get("llm_relevance_score") or 0), reverse=True)
        if float(row.get("llm_relevance_score") or 0) >= 0.9
    ]


def tune_threshold(rows: list[dict[str, Any]], field: str, recall_floor: float | None, max_results: int) -> tuple[float, dict[str, float], bool]:
    scored = []
    for threshold in np.linspace(0.01, 0.99, 99):
        metrics = set_metrics(rows, threshold_selector(field, float(threshold), max_results=max_results))
        if recall_floor is None or metrics["recall"] >= recall_floor:
            scored.append((metrics["f1"], metrics["precision"], -metrics["mean_output_count"], float(threshold), metrics))
    if scored:
        best = max(scored)
        return best[3], best[4], True
    # Highest-recall result is diagnostic only; it is not marked floor-compliant.
    fallback = max(
        ((set_metrics(rows, threshold_selector(field, float(t), max_results=max_results)), float(t)) for t in np.linspace(0.01, 0.99, 99)),
        key=lambda item: (item[0]["recall"], item[0]["f1"]),
    )
    return fallback[1], fallback[0], False


def support_count(row: dict[str, Any]) -> int:
    ace = row.get("ace") or {}
    return sum((
        float(row.get("llm_relevance_score") or 0) >= 0.8,
        float(row.get("cross_encoder_raw") or -10) >= 2.0,
        float(ace.get("raw_probability") or 0) >= 0.6,
    ))


def core_rescue_selector(field: str, high: float, low: float, rescue_limit: int, max_results: int = 10):
    def select(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ordered = sorted(rows, key=lambda row: float(row.get(field) or 0), reverse=True)
        core = [row for row in ordered if float(row.get(field) or 0) >= high][:max_results]
        core_ids = {row.get("paper_id") for row in core}
        rescue = []
        for row in ordered:
            ace = row.get("ace") or {}
            probability = float(row.get(field) or 0)
            if row.get("paper_id") in core_ids or not (low <= probability < high):
                continue
            if ace.get("hard_gate") == "fail" or float(ace.get("evidence_coverage") or 0) < 0.5 or support_count(row) < 2:
                continue
            rescue.append(row)
            if len(rescue) >= rescue_limit or len(core) + len(rescue) >= max_results:
                break
        selected = core + rescue
        if len(selected) < 2:
            selected = ordered[:2]
        return selected
    return select


def tune_core_rescue(rows: list[dict[str, Any]], field: str) -> tuple[dict[str, float], dict[str, float], bool]:
    trials = []
    for high in (0.75, 0.85, 0.95):
        for low in (0.35, 0.50, 0.65):
            if low >= high:
                continue
            for rescue_limit in (2, 4):
                metrics = set_metrics(rows, core_rescue_selector(field, high, low, rescue_limit))
                trials.append((metrics, {"high": high, "low": low, "rescue_limit": rescue_limit}))
    legal = [item for item in trials if item[0]["recall"] >= RECALL_FLOOR]
    pool = legal or trials
    metrics, params = max(pool, key=lambda item: (item[0]["f1"], item[0]["recall"], item[0]["precision"]))
    return params, metrics, bool(legal)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    args = parser.parse_args()
    development = load(args.train) + load(args.calibration)
    dev = load(args.dev)
    variants = {
        "S1_llm_calibrated": ("llm",),
        "S3_cross_encoder": ("cross",),
        "S4_ace": ("ace",),
        "S5_ace_cross": ("ace", "cross"),
        "S7_ace_llm": ("ace", "llm"),
        "S8_ace_cross_llm": ("ace", "cross", "llm"),
        "S9_full_consensus": ("llm", "cross", "ace", "trace", "rank", "abstract", "coverage"),
        "S9b_llm_cross_rank": ("llm", "cross", "rank"),
    }
    registry: list[dict[str, Any]] = []
    models: dict[str, Any] = {}
    for name, names in variants.items():
        oof, cv = grouped_oof(development, names)
        field = f"oof_{name}"
        attach(development, oof, field)
        x, y, _ = matrix(development, names)
        model = new_model(); model.fit(x, y)
        models[name] = model
        dev_probability = model.predict_proba(matrix(dev, names)[0])[:, 1]
        dev_field = f"prob_{name}"
        attach(dev, dev_probability, dev_field)
        threshold, cal_metrics, _ = tune_threshold(development, field, None, max_results=8)
        balanced_threshold, cal_balanced, floor_met = tune_threshold(development, field, RECALL_FLOOR, max_results=10)
        registry.extend([
            {"experiment": name, "decision": "f1_threshold", "threshold": threshold,
             "recall_floor_met": cal_metrics["recall"] >= RECALL_FLOOR, **cv,
             **{f"cal_{k}": v for k, v in cal_metrics.items()},
             **{f"dev_{k}": v for k, v in set_metrics(dev, threshold_selector(dev_field, threshold, 8)).items()}},
            {"experiment": name, "decision": "recall_floor_035", "threshold": balanced_threshold, "recall_floor_met": floor_met, **cv,
             **{f"cal_{k}": v for k, v in cal_balanced.items()},
             **{f"dev_{k}": v for k, v in set_metrics(dev, threshold_selector(dev_field, balanced_threshold, 10)).items()}},
        ])

    # S10: a bounded core/rescue policy over the strongest consensus probability.
    base_name = "S9_full_consensus"
    params, cal_s10, s10_floor = tune_core_rescue(development, f"oof_{base_name}")
    dev_s10 = set_metrics(dev, core_rescue_selector(f"prob_{base_name}", **params))
    registry.append({"experiment": "S10_core_rescue", "decision": "core_rescue", "threshold": json.dumps(params),
                     "recall_floor_met": s10_floor, **{f"cal_{k}": v for k, v in cal_s10.items()},
                     **{f"dev_{k}": v for k, v in dev_s10.items()}})

    references = [
        {"experiment": "S0_current_K2", "decision": "fixed_k2", "threshold": "", "recall_floor_met": False,
         **{f"cal_{k}": v for k, v in set_metrics(development, top_live(2)).items()},
         **{f"dev_{k}": v for k, v in set_metrics(dev, top_live(2)).items()}},
        {"experiment": "S1_current_LLM090", "decision": "fixed_llm090", "threshold": 0.9, "recall_floor_met": False,
         **{f"cal_{k}": v for k, v in set_metrics(development, llm090).items()},
         **{f"dev_{k}": v for k, v in set_metrics(dev, llm090).items()}},
        {"experiment": "S0_current_Top20", "decision": "fixed_top20", "threshold": "", "recall_floor_met": False,
         **{f"cal_{k}": v for k, v in set_metrics(development, top_live(20)).items()},
         **{f"dev_{k}": v for k, v in set_metrics(dev, top_live(20)).items()}},
        {"experiment": "S2_PaSa_7B", "decision": "not_run", **pasa_status()},
        {"experiment": "S3_BGE_v2_m3", "decision": "not_run_cpu_infeasible", "model": "BAAI/bge-reranker-v2-m3"},
        {"experiment": "S8_ACE_LLM_verifier", "decision": "rejected_prior_snapshot", "reason": "higher token cost and lower F1"},
    ]
    registry.extend(references)

    csv_path = OUT / "11_selector_ablation.csv"
    fields = sorted({key for row in registry for key in row})
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(registry)

    # Champions are chosen only from OOF development results.  The floor was not
    # reached by a practical <=20-result policy, so balanced/competition use the
    # highest-recall practical reference and record the unmet constraint.
    f1_rows = [row for row in registry if row.get("decision") in {"f1_threshold", "fixed_k2", "fixed_llm090"}]
    best_f1 = max(f1_rows, key=lambda row: (float(row.get("cal_f1") or 0), float(row.get("cal_recall") or 0)))
    floor_rows = [row for row in registry if row.get("recall_floor_met") is True and float(row.get("cal_recall") or 0) >= RECALL_FLOOR]
    if floor_rows:
        best_balanced = max(floor_rows, key=lambda row: (float(row.get("cal_f1") or 0), float(row.get("cal_precision") or 0)))
    else:
        best_balanced = next(row for row in references if row["experiment"] == "S0_current_Top20")

    champions: dict[str, dict[str, Any]] = {}
    for filename, row in (("champion_f1.json", best_f1), ("champion_balanced.json", best_balanced), ("champion_competition.json", best_balanced)):
        payload = {
            "selector": row["experiment"], "decision": row["decision"], "threshold": row.get("threshold"),
            "recall_floor": RECALL_FLOOR, "recall_floor_met": bool(row.get("recall_floor_met")),
            "calibration_protocol": "5-fold query-grouped OOF over frozen production50",
            "calibration_metrics": {k[4:]: v for k, v in row.items() if k.startswith("cal_")},
            "dev_metrics": {k[4:]: v for k, v in row.items() if k.startswith("dev_")},
            "feature_names": variants.get(row["experiment"], ()),
        }
        (OUT / filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        champions[filename] = payload
    with (OUT / "selector_model.pkl").open("wb") as handle:
        pickle.dump({"models": models, "variants": variants, "champions": champions}, handle)
    print(json.dumps(champions, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
