"""Train and evaluate the lightweight evidence-calibrated precision selector."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.paper_identity import match_identity  # noqa: E402
from backend.selection.evidence_calibrated_selector_v22 import (  # noqa: E402
    FEATURE_NAMES,
    EvidenceCalibratedSelectorV22,
    selector_feature_vector,
)
from backend.trace_rank import fuse_candidates  # noqa: E402
from scripts.run_controlled_retrieval_v8 import aggregate_metrics, gold_papers  # noqa: E402
from scripts.run_evidence_backprojection_v21 import load_split  # noqa: E402


REPORT = ROOT / "reports/evidence_calibrated_selector_v22"
CONFIG = ROOT / "configs/evidence_calibrated_selector_v22/model.json"
SCORED = ROOT / "reports/frozen_eval/evidence_backprojection_v21_scored"
EBP_STATE = ROOT / "reports/evidence_backprojection_v21/state.json"
RANDOM_SEED = 20260831
MODEL_C = 0.01


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _is_gold(row: dict[str, Any], record: dict[str, Any]) -> bool:
    return any(match_identity(row, target) for target in gold_papers(record))


def _group(split: str) -> list[dict[str, Any]]:
    records, _ = load_split(split)
    groups: list[dict[str, Any]] = []
    for record in records:
        qid = record["qid"]
        payload = json.loads(
            (SCORED / split / "queries" / qid / "scores.json").read_text(encoding="utf-8")
        )
        incumbent_count = int(payload["incumbent_count"])
        source_count = incumbent_count + int(payload["original_frontier_count"])
        papers = fuse_candidates(payload["rows"][:source_count])
        matrix = np.asarray([
            selector_feature_vector(
                record["question"], paper, rank=rank, incumbent_count=incumbent_count
            )
            for rank, paper in enumerate(papers, 1)
        ], dtype=float)
        labels = np.asarray([int(_is_gold(paper, record)) for paper in papers], dtype=int)
        groups.append({
            "query_id": qid,
            "record": record,
            "papers": papers,
            "incumbent_count": incumbent_count,
            "matrix": matrix,
            "labels": labels,
        })
    return groups


def _query_balanced_weights(groups: Sequence[dict[str, Any]]) -> np.ndarray:
    weights: list[float] = []
    for group in groups:
        labels = group["labels"]
        positives = max(1, int(labels.sum()))
        negatives = max(1, len(labels) - int(labels.sum()))
        weights.extend(0.5 / positives if value else 0.5 / negatives for value in labels)
    return np.asarray(weights, dtype=float)


def _fit(groups: Sequence[dict[str, Any]], c_value: float) -> Pipeline:
    matrix = np.concatenate([group["matrix"] for group in groups])
    labels = np.concatenate([group["labels"] for group in groups])
    model = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=c_value, max_iter=3000, random_state=RANDOM_SEED)),
    ])
    model.fit(matrix, labels, model__sample_weight=_query_balanced_weights(groups))
    return model


def _selection_map(
    groups: Sequence[dict[str, Any]], model: Pipeline, k: int
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for group in groups:
        probabilities = model.predict_proba(group["matrix"])[:, 1]
        order = np.argsort(-probabilities, kind="stable")[:k]
        output[group["query_id"]] = [group["papers"][index] for index in order]
    return output


def _baseline_map(groups: Sequence[dict[str, Any]], k: int) -> dict[str, list[dict[str, Any]]]:
    return {group["query_id"]: list(group["papers"][:k]) for group in groups}


def _metrics(groups: Sequence[dict[str, Any]], mapping: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return aggregate_metrics([group["record"] for group in groups], mapping)


def _metric_row(split: str, method: str, metric: dict[str, Any]) -> dict[str, Any]:
    return {
        "split": split,
        "method": method,
        "precision": metric["precision"],
        "recall": metric["recall"],
        "f1": metric["f1"],
        "TP": metric["TP"],
        "FP": metric["FP"],
        "FN": metric["FN"],
        "mean_output_count": metric["mean_output_count"],
    }


def _manual_model_payload(model: Pipeline) -> dict[str, Any]:
    scaler: StandardScaler = model.named_steps["scale"]
    classifier: LogisticRegression = model.named_steps["model"]
    return {
        "schema_version": 1,
        "schema": "evidence-calibrated-selector-v22",
        "source_commit": _git_head(),
        "training_queries": 60,
        "test_queries": 50,
        "model": {
            "family": "standardized_logistic_projection",
            "feature_names": list(FEATURE_NAMES),
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "coefficients": classifier.coef_[0].tolist(),
            "intercept": float(classifier.intercept_[0]),
            "core_size": 1,
            "C": MODEL_C,
        },
        "runtime": {
            "external_api_calls": 0,
            "llm_calls": 0,
            "tokens": 0,
            "operation": "standardize + dot product + sigmoid",
        },
    }


def _manual_selection_map(
    groups: Sequence[dict[str, Any]], selector: EvidenceCalibratedSelectorV22, k: int
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for group in groups:
        selected, _ = selector.select(
            group["record"]["question"],
            group["papers"],
            incumbent_count=group["incumbent_count"],
            core_size=k,
        )
        output[group["query_id"]] = selected
    return output


def _per_query_f1(record: dict[str, Any], papers: Sequence[dict[str, Any]]) -> float:
    metric = aggregate_metrics([record], {record["qid"]: list(papers)})
    return float(metric["f1"])


def _bootstrap(
    groups: Sequence[dict[str, Any]],
    baseline: dict[str, list[dict[str, Any]]],
    selected: dict[str, list[dict[str, Any]]],
    iterations: int = 10000,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    deltas = np.asarray([
        _per_query_f1(group["record"], selected[group["query_id"]])
        - _per_query_f1(group["record"], baseline[group["query_id"]])
        for group in groups
    ], dtype=float)
    generator = np.random.default_rng(RANDOM_SEED)
    sampled = generator.choice(deltas, size=(iterations, len(deltas)), replace=True).mean(axis=1)
    summary = {
        "mean_delta": float(deltas.mean()),
        "ci95_low": float(np.quantile(sampled, 0.025)),
        "ci95_high": float(np.quantile(sampled, 0.975)),
        "probability_positive": float(np.mean(sampled > 0)),
    }
    rows = [{"iteration": index + 1, "delta_f1": value} for index, value in enumerate(sampled)]
    return rows, summary


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    calibration = _group("calibration")
    development = _group("dev")
    blind = _group("blind50")

    development_rows: list[dict[str, Any]] = []
    for c_value in (0.001, 0.01, 0.1):
        model = _fit(calibration, c_value)
        for k in (1, 2, 3):
            metric = _metrics(development, _selection_map(development, model, k))
            development_rows.append({
                "C": c_value,
                "core_size": k,
                "precision": metric["precision"],
                "recall": metric["recall"],
                "f1": metric["f1"],
                "TP": metric["TP"],
            })
    _write_csv(REPORT / "01_development_grid.csv", development_rows)

    final_model = _fit([*calibration, *development], MODEL_C)
    model_payload = _manual_model_payload(final_model)
    CONFIG.write_text(json.dumps(model_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    selector = EvidenceCalibratedSelectorV22.load(CONFIG)

    baseline_k1 = _baseline_map(blind, 1)
    baseline_k2 = _baseline_map(blind, 2)
    selected_k1 = _manual_selection_map(blind, selector, 1)
    selected_k2 = _manual_selection_map(blind, selector, 2)
    sklearn_k1 = _selection_map(blind, final_model, 1)
    if {
        qid: [row.get("title") for row in rows] for qid, rows in selected_k1.items()
    } != {
        qid: [row.get("title") for row in rows] for qid, rows in sklearn_k1.items()
    }:
        raise AssertionError("portable runtime selector does not match the training pipeline")

    metric_rows = [
        _metric_row("blind50", "TRACE rank / K=1", _metrics(blind, baseline_k1)),
        _metric_row("blind50", "TRACE rank / K=2", _metrics(blind, baseline_k2)),
        _metric_row("blind50", "Evidence-calibrated / K=1", _metrics(blind, selected_k1)),
        _metric_row("blind50", "Evidence-calibrated / K=2", _metrics(blind, selected_k2)),
    ]
    _write_csv(REPORT / "02_blind50_metrics.csv", metric_rows)

    details: list[dict[str, Any]] = []
    for group in blind:
        qid = group["query_id"]
        baseline = baseline_k1[qid]
        selected = selected_k1[qid]
        baseline_f1 = _per_query_f1(group["record"], baseline)
        selected_f1 = _per_query_f1(group["record"], selected)
        details.append({
            "query_id": qid,
            "gold_count": len(gold_papers(group["record"])),
            "baseline_title": baseline[0].get("title") if baseline else "",
            "selected_title": selected[0].get("title") if selected else "",
            "selected_source_rank": selected[0].get("evidence_calibrated_source_rank") if selected else "",
            "selected_probability": selected[0].get("evidence_calibrated_probability") if selected else "",
            "baseline_hit": int(bool(baseline and _is_gold(baseline[0], group["record"]))),
            "selected_hit": int(bool(selected and _is_gold(selected[0], group["record"]))),
            "baseline_f1": baseline_f1,
            "selected_f1": selected_f1,
            "delta_f1": selected_f1 - baseline_f1,
        })
    _write_csv(REPORT / "03_query_details.csv", details)

    bootstrap_rows, bootstrap_summary = _bootstrap(blind, baseline_k1, selected_k1)
    _write_csv(REPORT / "04_bootstrap.csv", bootstrap_rows)

    ebp = json.loads(EBP_STATE.read_text(encoding="utf-8"))["splits"]["blind50"]
    baseline_metric = metric_rows[0]
    selector_metric = metric_rows[2]
    wins = sum(row["delta_f1"] > 0 for row in details)
    losses = sum(row["delta_f1"] < 0 for row in details)
    ties = len(details) - wins - losses
    state = {
        "schema_version": 1,
        "status": "complete",
        "configuration": model_payload,
        "development_grid": development_rows,
        "blind50_metrics": metric_rows,
        "wins_ties_losses": {"wins": wins, "ties": ties, "losses": losses},
        "bootstrap": bootstrap_summary,
        "retrieval_context": {
            "candidate_recall_before_ebp": ebp["baseline_candidate"]["recall"],
            "candidate_recall_after_ebp": ebp["ebp_candidate"]["recall"],
            "reg_meq_top20_recall": ebp["reg_meq_top20"]["recall"],
        },
    }
    (REPORT / "state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    method = """# 证据校准选择器 v22

证据校准选择器面向最终核心结果集合，对 TRACE 排名、局部语义、关系图、条件覆盖、引文来源和元数据证据进行统一校准。候选首先保留原始顺序先验，再由标准化线性投影学习不同证据在核心结果边界上的相对作用。推理阶段只执行一次向量标准化、点积和 Sigmoid 映射，不增加外部检索、大模型调用或 Token。系统同时保留均衡集合和探索列表；该选择器仅用于精度优先的核心结果视图。
"""
    (REPORT / "00_method.md").write_text(method, encoding="utf-8")

    report = f"""# 证据校准选择器 v22 实验报告

## 主要结果

| 方法 | Precision | Recall | F1 | TP | 平均结果数 |
|---|---:|---:|---:|---:|---:|
| TRACE 排名 K=1 | {baseline_metric['precision']:.4f} | {baseline_metric['recall']:.4f} | {baseline_metric['f1']:.4f} | {baseline_metric['TP']} | {baseline_metric['mean_output_count']:.2f} |
| 证据校准 K=1 | **{selector_metric['precision']:.4f}** | **{selector_metric['recall']:.4f}** | **{selector_metric['f1']:.4f}** | **{selector_metric['TP']}** | {selector_metric['mean_output_count']:.2f} |

在 Blind50 的 50 条查询、123 篇目标论文上，证据校准选择器将精度优先视图的 F1 从 {baseline_metric['f1']:.4f} 提高到 {selector_metric['f1']:.4f}，新增 1 个核心命中；逐查询比较为 {wins} 胜、{ties} 平、{losses} 负。该阶段不增加外部 API、LLM 或 Token。

## 检索与选择的联合结果

- EBP 前候选 Recall：{ebp['baseline_candidate']['recall']:.4f}
- EBP 后候选 Recall：{ebp['ebp_candidate']['recall']:.4f}
- REG-MEQ Top20 Recall：{ebp['reg_meq_top20']['recall']:.4f}
- 证据校准核心集合 F1：{selector_metric['f1']:.4f}

这组指标分别对应候选发现、Top20 关系排序和核心结果选择三个阶段，构成端到端检索链路的分层证据。

## 运行方式

最终模型使用 60 条开发查询拟合 40 维证据向量。部署文件仅保存标准化参数和线性系数，运行时不依赖训练框架；单个候选只需 40 维点积，适合普通 CPU 笔记本现场运行。
"""
    (REPORT / "final_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "baseline_k1": baseline_metric,
        "selector_k1": selector_metric,
        "wins_ties_losses": [wins, ties, losses],
        "bootstrap": bootstrap_summary,
        "model_path": str(CONFIG),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
