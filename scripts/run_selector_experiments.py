"""Train/calibrate a lightweight selector and run frozen, leakage-safe experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.paper_identity import match_identity
from backend.selection.calibration import (
    IsotonicCalibrator, PlattCalibrator, TemperatureCalibrator, calibration_metrics,
    calibrator_from_dict, save_calibrator,
)
from backend.selection.linear_model import LinearSelectorModel, text_features
from backend.selection.relevance_selector import RelevanceSelector
from backend.selection.set_decision import select_final_set
from backend.trace_rank import fuse_candidates
from scripts.audit_evaluation import aggregate, gold_papers, load_jsonl, metrics_from_papers, source_cutoff
from scripts.run_frozen_official_benchmark import module_candidates


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def classifier_metrics(probabilities: list[float], labels: list[int]) -> dict[str, float]:
    base = calibration_metrics(probabilities, labels)
    order = sorted(range(len(labels)), key=lambda i: probabilities[i], reverse=True)
    pos = sum(labels); neg = len(labels) - pos
    tp = fp = 0; previous_recall = 0.0; pr_auc = 0.0; roc_points = [(0.0, 0.0)]
    for i in order:
        if labels[i]: tp += 1
        else: fp += 1
        recall = tp / pos if pos else 0; precision = tp / (tp + fp)
        pr_auc += (recall - previous_recall) * precision; previous_recall = recall
        roc_points.append((fp / neg if neg else 0, recall))
    roc_auc = sum((roc_points[i][0] - roc_points[i - 1][0]) *
                  (roc_points[i][1] + roc_points[i - 1][1]) / 2 for i in range(1, len(roc_points)))
    return {**base, "pr_auc": round(pr_auc, 6), "roc_auc": round(roc_auc, 6)}


def fit_calibrator(name: str, scores: list[float], labels: list[int]) -> Any | None:
    if name == "identity": return None
    if name == "platt": return PlattCalibrator().fit(scores, labels)
    if name == "isotonic": return IsotonicCalibrator().fit(scores, labels)
    if name == "temperature": return TemperatureCalibrator().fit(scores, labels)
    raise ValueError(name)


def predict(calibrator: Any | None, score: float) -> float:
    return float(calibrator.predict_one(score)) if calibrator is not None else float(score)


def oof_calibration(rows: list[dict[str, Any]], scores: list[float], method: str, folds: int = 5) -> list[float]:
    groups = sorted({row["query_id"] for row in rows})
    random.Random(20260830).shuffle(groups)
    fold_by_group = {group: i % folds for i, group in enumerate(groups)}
    output = [0.0] * len(rows)
    for fold in range(folds):
        train_idx = [i for i, row in enumerate(rows) if fold_by_group[row["query_id"]] != fold]
        test_idx = [i for i, row in enumerate(rows) if fold_by_group[row["query_id"]] == fold]
        cal = fit_calibrator(method, [scores[i] for i in train_idx], [rows[i]["label"] for i in train_idx])
        for i in test_idx: output[i] = predict(cal, scores[i])
    return output


def set_metrics(labels: list[int], selected: list[int]) -> dict[str, float]:
    tp = sum(labels[i] for i in selected); pred = len(selected); gold = sum(labels)
    p = tp / pred if pred else 0.0; r = tp / gold if gold else 0.0
    return {"precision": p, "recall": r, "f1": 2 * tp / (pred + gold) if pred + gold else 0.0,
            "ndcg": 0.0, "tp": tp, "predicted": pred, "gold": gold}


def grouped_strategy(
    rows: list[dict[str, Any]], probabilities: list[float], strategy: str, **kwargs: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows): groups[row["query_id"]].append(i)
    per_query = []
    for qid, indices in groups.items():
        ordered = sorted(indices, key=lambda i: (probabilities[i], rows[i]["paper_key"]), reverse=True)
        ps = [probabilities[i] for i in ordered]; labels = [rows[i]["label"] for i in ordered]
        if strategy == "fixed_k":
            decision = select_final_set(ps, strategy="fixed_k", fixed_k=kwargs["k"], max_results=kwargs["k"])
        elif strategy == "threshold":
            decision = select_final_set(ps, strategy="global_threshold", threshold=kwargs["threshold"])
        elif strategy == "expected_f1":
            decision = select_final_set(ps, strategy="expected_f1")
        elif strategy == "score_gap":
            decision = select_final_set(ps, strategy="score_gap", threshold=kwargs["threshold"],
                                        low_threshold=kwargs["low_threshold"], score_gap=kwargs["score_gap"])
        elif strategy == "dual_threshold":
            masks = []
            for i in ordered:
                f = text_features(rows[i]["query"], rows[i])
                masks.append(bool(f[3] >= kwargs.get("coverage", 0.25) and rows[i].get("abstract")))
            decision = select_final_set(ps, strategy="dual_threshold", threshold=kwargs["threshold"],
                                        low_threshold=kwargs["low_threshold"], rescue_mask=masks)
        else: raise ValueError(strategy)
        metric = set_metrics(labels, decision.selected_indices); per_query.append({"query_id": qid, **metric})
    return aggregate(per_query), per_query


def choose_calibration_policy(rows: list[dict[str, Any]], by_method: dict[str, list[float]]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    experiments: list[dict[str, Any]] = []
    for method, probs in by_method.items():
        for k in [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]:
            metrics, _ = grouped_strategy(rows, probs, "fixed_k", k=k)
            experiments.append({"experiment_id": f"{method}_fixed_k{k}", "calibration": method,
                                "decision": "fixed_k", "params": {"k": k}, "metrics": metrics})
        for n in range(5, 96, 5):
            t = n / 100; metrics, _ = grouped_strategy(rows, probs, "threshold", threshold=t)
            experiments.append({"experiment_id": f"{method}_threshold_{n:02d}", "calibration": method,
                                "decision": "global_threshold", "params": {"threshold": t}, "metrics": metrics})
        metrics, _ = grouped_strategy(rows, probs, "expected_f1")
        experiments.append({"experiment_id": f"{method}_expected_f1", "calibration": method,
                            "decision": "expected_f1", "params": {}, "metrics": metrics})
        for high, low in [(0.6, 0.35), (0.7, 0.4), (0.8, 0.5)]:
            metrics, _ = grouped_strategy(rows, probs, "dual_threshold", threshold=high, low_threshold=low)
            experiments.append({"experiment_id": f"{method}_dual_{high}_{low}", "calibration": method,
                                "decision": "dual_threshold", "params": {"threshold": high, "low_threshold": low},
                                "metrics": metrics})
    baseline = next(x for x in experiments if x["calibration"] == "identity" and x["decision"] == "fixed_k" and x["params"]["k"] == 20)
    f1_champion = max(experiments, key=lambda x: (x["metrics"]["macro"]["f1"], x["metrics"]["macro"]["recall"], -x["metrics"]["micro"]["predicted"]))
    floor = max(baseline["metrics"]["macro"]["recall"] - 0.03, baseline["metrics"]["macro"]["recall"] * 0.95)
    eligible = [x for x in experiments if x["metrics"]["macro"]["recall"] >= floor]
    balanced = max(eligible, key=lambda x: (x["metrics"]["macro"]["f1"], -x["metrics"]["micro"]["predicted"]))
    return experiments, f1_champion, {**balanced, "recall_floor": floor}


def full_ranked_candidates(row: dict[str, Any], run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    qdir = run_dir / "queries" / row["qid"]
    response = json.loads((qdir / "live_response.json").read_text(encoding="utf-8"))
    audit = json.loads((qdir / "live_audit.json").read_text(encoding="utf-8"))
    snapshot = json.loads((qdir / "snapshot.json").read_text(encoding="utf-8"))
    cutoff = source_cutoff(row); assert cutoff is not None
    modules = module_candidates(snapshot, cutoff)
    merged = fuse_candidates([paper for name in ("lexical", "semantic", "arxiv", "title", "citation", "second") for paper in modules[name]])
    summarized = next(s["papers"] for s in audit["stages"] if s["name"] == "shortlist_post_llm")
    ranked = []
    for stub in summarized:
        full = next((paper for paper in merged if match_identity(paper, stub)), {})
        ranked.append({**full, **stub})
    return ranked, response.get("query_understanding") or {}


def evaluate_dev(
    rows: list[dict[str, Any]], run_dir: Path, model: LinearSelectorModel, calibrator: Any | None,
    policy: dict[str, Any], experiment_id: str,
) -> dict[str, Any]:
    per_query = []; output_counts = []
    for row in rows:
        papers, understanding = full_ranked_candidates(row, run_dir)
        scored = [(predict(calibrator, model.predict_one(row["question"], paper)), paper) for paper in papers]
        if policy.get("ranking", "model") != "current":
            scored.sort(key=lambda x: (x[0], str(x[1].get("title") or "")), reverse=True)
        ps = [x[0] for x in scored]; ordered = [x[1] for x in scored]
        decision_name = policy["decision"]; params = policy.get("params") or {}
        if decision_name == "fixed_k": decision = select_final_set(ps, strategy="fixed_k", fixed_k=params["k"], max_results=params["k"])
        elif decision_name == "global_threshold": decision = select_final_set(ps, strategy="global_threshold", threshold=params["threshold"])
        elif decision_name == "expected_f1": decision = select_final_set(ps, strategy="expected_f1")
        elif decision_name == "dual_threshold":
            selector = RelevanceSelector(); records = selector.evaluate_many(row["question"], understanding, ordered)
            mask = [record.hard_constraint_pass and (paper.get("llm_relevance_score") or 0) >= 0.7 for record, paper in zip(records, ordered)]
            decision = select_final_set(ps, strategy="dual_threshold", threshold=params["threshold"], low_threshold=params["low_threshold"], rescue_mask=mask)
        else: raise ValueError(decision_name)
        selected = [ordered[i] for i in decision.selected_indices]
        metric = metrics_from_papers(selected, gold_papers(row), max(1, len(selected)))
        metric["predicted"] = len(selected); metric["gold"] = len(gold_papers(row)); metric["tp"] = metric.pop("tp")
        per_query.append(metric); output_counts.append(len(selected))
    value = aggregate(per_query)
    return {"experiment_id": experiment_id, "query_set": "dev20", "metrics": value,
            "mean_output_count": round(sum(output_counts) / len(output_counts), 4),
            "median_output_count": sorted(output_counts)[len(output_counts)//2],
            "empty_output_rate": sum(x == 0 for x in output_counts) / len(output_counts)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "reports/f1_selector_v3")
    ap.add_argument("--dataset", type=Path, default=Path(r"D:\AICompetition\references\github\SPAR\benchmark\AutoScholarQuery_test.jsonl"))
    ap.add_argument("--run-dir", type=Path, default=ROOT / "reports/frozen_eval/autoscholar20_trace_v4_official_20260824_v2")
    args = ap.parse_args()
    train = load_jsonl(args.data_dir / "selector_train_pairs.jsonl")
    calibration = load_jsonl(args.data_dir / "selector_calibration_pairs.jsonl")
    model = LinearSelectorModel.fit(train); model_path = ROOT / "configs/f1_selector_v3/linear_selector.json"; model.save(model_path)
    raw = [model.predict_one(row["query"], row) for row in calibration]
    labels = [row["label"] for row in calibration]
    methods = ["identity", "platt", "isotonic", "temperature"]
    oof = {method: oof_calibration(calibration, raw, method) for method in methods}
    calibration_report = {method: classifier_metrics(oof[method], labels) for method in methods}
    experiments, f1_champion, balanced = choose_calibration_policy(calibration, oof)
    best_calibration = min(methods, key=lambda m: (calibration_report[m]["brier"], calibration_report[m]["ece"]))
    final_cal = fit_calibrator(best_calibration, raw, labels)
    if final_cal is not None: save_calibrator(final_cal, ROOT / "configs/f1_selector_v3/calibrator.json")
    else: (ROOT / "configs/f1_selector_v3/calibrator.json").write_text('{"method":"identity"}', encoding="utf-8")

    by_id = {row["qid"]: row for row in load_jsonl(args.dataset)}
    query_sets = json.loads((ROOT / "configs/query_sets.json").read_text(encoding="utf-8"))
    dev_rows = [by_id[qid] for qid in query_sets["dev20"]]
    # Evaluate only pre-selected policies; dev results are diagnostics, not threshold fitting.
    selected_policies = [
        {"experiment_id": "S0_fixed_top20_baseline", "calibration": best_calibration, "decision": "fixed_k", "params": {"k": 20}, "ranking": "current"},
        {"experiment_id": "S1_fixed_k2_diagnostic", "calibration": best_calibration, "decision": "fixed_k", "params": {"k": 2}, "ranking": "current"},
        {"experiment_id": "S1_fixed_k8_recall_guard", "calibration": best_calibration, "decision": "fixed_k", "params": {"k": 8}, "ranking": "current"},
        {"experiment_id": "f1_champion", **{k: f1_champion[k] for k in ("calibration", "decision", "params")}},
        {"experiment_id": "balanced_champion", **{k: balanced[k] for k in ("calibration", "decision", "params")}},
    ]
    dev_results = []
    for policy in selected_policies:
        cal = fit_calibrator(policy["calibration"], raw, labels)
        dev_results.append(evaluate_dev(dev_rows, args.run_dir, model, cal, policy, policy["experiment_id"]))

    registry_rows = []
    for exp in experiments:
        m = exp["metrics"]
        registry_rows.append({"experiment_id": exp["experiment_id"], "query_set": "selector_calibration_oof",
                              "calibration_method": exp["calibration"], "set_decision_method": exp["decision"],
                              "params": json.dumps(exp["params"], sort_keys=True),
                              "precision": m["macro"]["precision"], "recall": m["macro"]["recall"],
                              "f1": m["macro"]["f1"], "tp": m["micro"]["true_positive"],
                              "predicted": m["micro"]["predicted"], "gold": m["micro"]["gold"],
                              "candidate_recall": 1.0, "accept_or_reject": "candidate" if exp in (f1_champion, balanced) else "screened",
                              "reason": "OOF grouped calibration experiment"})
    for exp in dev_results:
        m = exp["metrics"]
        registry_rows.append({"experiment_id": exp["experiment_id"], "query_set": "dev20_frozen",
                              "calibration_method": "locked_from_calibration", "set_decision_method": "locked",
                              "params": "{}", "precision": m["macro"]["precision"], "recall": m["macro"]["recall"],
                              "f1": m["macro"]["f1"], "tp": m["micro"]["true_positive"],
                              "predicted": m["micro"]["predicted"], "gold": m["micro"]["gold"],
                              "candidate_recall": "unchanged", "accept_or_reject": "diagnostic", "reason": "no dev threshold fitting"})
    with (args.data_dir / "experiment_registry.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(registry_rows[0])); writer.writeheader(); writer.writerows(registry_rows)

    champion_common = {"model": str(model_path), "calibrator": str(ROOT / "configs/f1_selector_v3/calibrator.json"),
                       "calibration_metrics": calibration_report, "selection_basis": "grouped OOF selector_calibration",
                       "validation30_used": False}
    (args.data_dir / "f1_champion.json").write_text(json.dumps({**champion_common, **f1_champion}, indent=2), encoding="utf-8")
    (args.data_dir / "balanced_champion.json").write_text(json.dumps({**champion_common, **balanced}, indent=2), encoding="utf-8")
    report = "# 概率校准报告\n\n" + "\n".join(
        f"- **{name}**: Brier={value['brier']:.4f}, ECE={value['ece']:.4f}, PR-AUC={value['pr_auc']:.4f}, ROC-AUC={value['roc_auc']:.4f}"
        for name, value in calibration_report.items()) + f"\n\n选择用于冻结模型的校准器：**{best_calibration}**。所有数值为 query-grouped 5-fold OOF；validation30 未参与。\n"
    (args.data_dir / "05_calibration_report.md").write_text(report, encoding="utf-8")
    exp_md = f"""# Selector 实验报告

## 校准集 champion

- F1 champion：`{f1_champion['experiment_id']}`，macro P/R/F1 = {f1_champion['metrics']['macro']['precision']:.4f}/{f1_champion['metrics']['macro']['recall']:.4f}/{f1_champion['metrics']['macro']['f1']:.4f}。
- Balanced champion：`{balanced['experiment_id']}`，macro P/R/F1 = {balanced['metrics']['macro']['precision']:.4f}/{balanced['metrics']['macro']['recall']:.4f}/{balanced['metrics']['macro']['f1']:.4f}，recall floor={balanced['recall_floor']:.4f}。
- 模型只使用线上可得 query/title/abstract 文本特征；没有 gold ID、gold 数量或 validation30 特征。

## 冻结 dev20 诊断（策略未在 dev 上重新调阈值）

""" + "\n".join(f"- `{x['experiment_id']}`: P={x['metrics']['macro']['precision']:.4f}, R={x['metrics']['macro']['recall']:.4f}, F1={x['metrics']['macro']['f1']:.4f}, mean outputs={x['mean_output_count']:.2f}" for x in dev_results) + "\n\n完整矩阵见 `experiment_registry.csv`。本轮尚未把 cross-encoder 写成已运行组件；当前环境无 torch/transformers/sentence-transformers，因此明确留空，不伪造结果。\n"
    (args.data_dir / "06_selector_experiments.md").write_text(exp_md, encoding="utf-8")
    with (args.data_dir / "07_output_cardinality.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["experiment_id", "mean_output_count", "median_output_count", "empty_output_rate"])
        writer.writeheader(); writer.writerows([{k: x[k] for k in writer.fieldnames} for x in dev_results])
    print(json.dumps({"calibration": calibration_report, "f1_champion": f1_champion,
                      "balanced_champion": balanced, "dev": dev_results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
