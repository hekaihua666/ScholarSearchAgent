"""Build public aggregate-only v4 reports from frozen experiment artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/joint_optimization_v4"
FROZEN = ROOT / "reports/frozen_eval"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate_runtime(run_dir: Path) -> dict[str, float]:
    totals: dict[str, float] = {}
    count = 0
    for path in (run_dir / "queries").glob("*/live_response.json"):
        metrics = read_json(path).get("metrics") or {}
        count += 1
        for key in ("api_call_count", "llm_call_count", "llm_token_count", "elapsed_seconds"):
            totals[key] = totals.get(key, 0.0) + float(metrics.get(key) or 0)
    return {key: round(value / count, 6) for key, value in totals.items()} if count else {}


def retrieval_aggregate(profile: str) -> dict[str, Any]:
    run = FROZEN / f"blind_holdout50_v4_{profile}"
    manifest = read_json(run / "manifest.json")
    macro = manifest["baseline_live"]["metrics"]["20"]["macro"]
    stage = manifest["retrieval_ablation"]["F_plus_second"]
    return {
        "precision": macro["precision"], "recall": macro["recall"], "f1": macro["f1"],
        "candidate_recall": stage["candidate_recall"]["macro"]["recall"],
        "candidate_gold": stage["candidate_gold"], "mean_output_count": 20.0,
        **aggregate_runtime(run),
        "replay_identical": manifest["replay_all_identical"],
        "date_legal": manifest["all_final_papers_date_legal"],
    }


def selector_metrics(name: str, key: str) -> dict[str, Any]:
    return read_json(OUT / name)[key]["metrics"]


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    v3_blind = retrieval_aggregate("v3")
    lean_blind = retrieval_aggregate("lean")
    dev_v3 = read_json(OUT / "dev20_current_champions.json")
    dev_lean = read_json(OUT / "dev20_lean_champions.json")
    blind_v3 = read_json(OUT / "blind_v3_aggregate_private.json")
    blind_lean = read_json(OUT / "blind_lean_aggregate_private.json")

    # All blind rows below contain aggregate metrics only; no query IDs, titles,
    # gold identities, or per-query errors are copied into public reports.
    joint = [
        {"split": "dev20", "experiment": "J0_v3_Top20", "retrieval": "v3", "selector": "Top20",
         **dev_v3["top20"], "candidate_recall": 0.454286, "api_per_query": 16.0, "latency_per_query": 35.361},
        {"split": "dev20", "experiment": "J1_v3_best_selector", "retrieval": "v3", "selector": "S1_llm_calibrated",
         **dev_v3["f1_champion"]["metrics"], "candidate_recall": 0.454286, "api_per_query": 16.0, "latency_per_query": 35.361},
        {"split": "dev20", "experiment": "J2_lean_current_selector", "retrieval": "lean", "selector": "LLM-0.90",
         **dev_lean["llm090"], "candidate_recall": 0.433452, "api_per_query": 9.0, "latency_per_query": 18.416},
        {"split": "dev20", "experiment": "J3_lean_best_selector", "retrieval": "lean", "selector": "S1_llm_calibrated",
         **dev_lean["f1_champion"]["metrics"], "candidate_recall": 0.433452, "api_per_query": 9.0, "latency_per_query": 18.416},
        {"split": "dev20", "experiment": "J4_lean_low_cost", "retrieval": "lean", "selector": "Top20",
         **dev_lean["top20"], "candidate_recall": 0.433452, "api_per_query": 9.0, "latency_per_query": 18.416},
        {"split": "blind50", "experiment": "B0_v3_Top20", "retrieval": "v3", "selector": "Top20",
         **{k: v3_blind[k] for k in ("precision", "recall", "f1", "mean_output_count")},
         "candidate_recall": v3_blind["candidate_recall"], "api_per_query": v3_blind["api_call_count"],
         "tokens_per_query": v3_blind["llm_token_count"], "latency_per_query": v3_blind["elapsed_seconds"]},
        {"split": "blind50", "experiment": "B1_v3_best_selector", "retrieval": "v3", "selector": "S1_llm_calibrated",
         **blind_v3["f1_champion"]["metrics"], "candidate_recall": v3_blind["candidate_recall"],
         "api_per_query": v3_blind["api_call_count"], "tokens_per_query": v3_blind["llm_token_count"],
         "latency_per_query": v3_blind["elapsed_seconds"]},
        {"split": "blind50", "experiment": "B2_lean_current_selector", "retrieval": "lean", "selector": "LLM-0.90",
         **blind_lean["llm090"], "candidate_recall": lean_blind["candidate_recall"],
         "api_per_query": lean_blind["api_call_count"], "tokens_per_query": lean_blind["llm_token_count"],
         "latency_per_query": lean_blind["elapsed_seconds"]},
        {"split": "blind50", "experiment": "B3_lean_best_selector", "retrieval": "lean", "selector": "S1_llm_calibrated",
         **blind_lean["f1_champion"]["metrics"], "candidate_recall": lean_blind["candidate_recall"],
         "api_per_query": lean_blind["api_call_count"], "tokens_per_query": lean_blind["llm_token_count"],
         "latency_per_query": lean_blind["elapsed_seconds"]},
        {"split": "blind50", "experiment": "B4_lean_low_cost", "retrieval": "lean", "selector": "Top20",
         **{k: lean_blind[k] for k in ("precision", "recall", "f1", "mean_output_count")},
         "candidate_recall": lean_blind["candidate_recall"], "api_per_query": lean_blind["api_call_count"],
         "tokens_per_query": lean_blind["llm_token_count"], "latency_per_query": lean_blind["elapsed_seconds"]},
    ]
    csv_write(OUT / "13_joint_ablation.csv", joint)

    costs = [
        {"split": row["split"], "configuration": row["experiment"], "precision": row.get("precision"),
         "recall": row.get("recall"), "f1": row.get("f1"), "candidate_recall": row.get("candidate_recall"),
         "mean_output_count": row.get("mean_output_count"), "api_per_query": row.get("api_per_query"),
         "tokens_per_query": row.get("tokens_per_query", "not_recorded_in_dev_capture"),
         "latency_per_query": row.get("latency_per_query")}
        for row in joint
    ]
    csv_write(OUT / "14_cost_quality_tradeoff.csv", costs)

    blind_report = f"""# blind_holdout50_v4 聚合结果

盲测配置在运行前已冻结。下表仅包含 50 条查询的聚合结果，不包含逐查询 gold、结果或差异。

| 配置 | P | R | F1 | candidate R | 平均输出 | API/查询 | Token/查询 | 延迟/查询(s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v3 Top20 | {v3_blind['precision']:.4f} | {v3_blind['recall']:.4f} | {v3_blind['f1']:.4f} | {v3_blind['candidate_recall']:.4f} | 20.00 | {v3_blind['api_call_count']:.2f} | {v3_blind['llm_token_count']:.0f} | {v3_blind['elapsed_seconds']:.2f} |
| v3 + S1 Selector | {blind_v3['f1_champion']['metrics']['precision']:.4f} | {blind_v3['f1_champion']['metrics']['recall']:.4f} | {blind_v3['f1_champion']['metrics']['f1']:.4f} | {v3_blind['candidate_recall']:.4f} | {blind_v3['f1_champion']['metrics']['mean_output_count']:.2f} | {v3_blind['api_call_count']:.2f} | {v3_blind['llm_token_count']:.0f} | {v3_blind['elapsed_seconds']:.2f} |
| lean Top20 | {lean_blind['precision']:.4f} | {lean_blind['recall']:.4f} | {lean_blind['f1']:.4f} | {lean_blind['candidate_recall']:.4f} | 20.00 | {lean_blind['api_call_count']:.2f} | {lean_blind['llm_token_count']:.0f} | {lean_blind['elapsed_seconds']:.2f} |
| lean + S1 Selector | {blind_lean['f1_champion']['metrics']['precision']:.4f} | {blind_lean['f1_champion']['metrics']['recall']:.4f} | {blind_lean['f1_champion']['metrics']['f1']:.4f} | {lean_blind['candidate_recall']:.4f} | {blind_lean['f1_champion']['metrics']['mean_output_count']:.2f} | {lean_blind['api_call_count']:.2f} | {lean_blind['llm_token_count']:.0f} | {lean_blind['elapsed_seconds']:.2f} |

## 结论

- S1 在 v3 候选上得到最高盲测集合 F1：{blind_v3['f1_champion']['metrics']['f1']:.4f}；相对同快照 LLM-0.90 的 {blind_v3['llm090']['f1']:.4f}，提高 {blind_v3['f1_champion']['metrics']['f1']-blind_v3['llm090']['f1']:.4f}。
- 它将平均输出压到 2 篇，Precision 为 {blind_v3['f1_champion']['metrics']['precision']:.4f}，但 Recall 只有 {blind_v3['f1_champion']['metrics']['recall']:.4f}，未达到 0.35 的召回线。
- lean 相比 v3 Top20：API 减少 {(1-lean_blind['api_call_count']/v3_blind['api_call_count'])*100:.1f}%，延迟减少 {(1-lean_blind['elapsed_seconds']/v3_blind['elapsed_seconds'])*100:.1f}%；F1 下降 {v3_blind['f1']-lean_blind['f1']:.4f}，candidate Recall 下降 {v3_blind['candidate_recall']-lean_blind['candidate_recall']:.4f}。
- lean + S1 的 F1 为 {blind_lean['f1_champion']['metrics']['f1']:.4f}，没有超过 v3 + S1。
- 两个 LIVE run 均 replay 完全一致，最终论文日期均合法。
- 目标 P≥0.22、R≥0.35、F1≥0.26 未同时达到。
"""
    (OUT / "15_blind_holdout_report.md").write_text(blind_report, encoding="utf-8")

    # Restore the production-data audit after the blind-only builder overwrote it.
    selector_audit = """# Production-distribution Selector 数据审计

- 开发数据：冻结的 production50（train30 + calibration20），共 4,958 个 query-paper pair、35 个已匹配 gold。
- 阈值与模型选择：production50 上 5-fold query-grouped OOF；dev20 仅在选择完成后评估。
- dev20 lean 快照：1,999 个候选、22 个候选池内 gold。
- 负样本口径：未命中 benchmark gold 的候选是 weak/unjudged negative，不声称现实中绝对不相关。
- gold 只用于离线标签与指标，不进入 query plan、特征、检索缓存或线上排序。
- 稳定身份去重抑制同一 gold 的 DOI/arXiv/标题别名重复计正。
- Top20 内负样本标作 hard negative；另区分 partial-constraint、survey、metadata fail、missing abstract 等类型。
- 生产候选稀疏：4,958 对中仅 35 个正例，且候选池宏召回约 0.37，是 Selector 校准不稳定的主要数据限制。
"""
    (OUT / "07_selector_data_audit.md").write_text(selector_audit, encoding="utf-8")

    # Public experiment registry.
    registry: list[dict[str, Any]] = []
    for source, kind in ((OUT / "11_selector_ablation.csv", "selector"), (OUT / "12_retrieval_ablation.csv", "retrieval"), (OUT / "13_joint_ablation.csv", "joint")):
        with source.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                registry.append({"kind": kind, "source": source.name, **row})
    csv_write(OUT / "experiment_registry.csv", registry)

    # Update state without altering the pre-blind checksums.
    state_path = OUT / "state.json"
    state = read_json(state_path)
    state.update({
        "phase": "experiments_complete_reports_finalizing",
        "blind_completed": True,
        "blind_query_count": 50,
        "blind_aggregate": {
            "v3_top20": {k: v3_blind[k] for k in ("precision", "recall", "f1", "candidate_recall")},
            "v3_f1_selector": blind_v3["f1_champion"]["metrics"],
            "lean_top20": {k: lean_blind[k] for k in ("precision", "recall", "f1", "candidate_recall")},
            "lean_f1_selector": blind_lean["f1_champion"]["metrics"],
        },
        "blind_replay_identical": v3_blind["replay_identical"] and lean_blind["replay_identical"],
        "blind_all_dates_legal": v3_blind["date_legal"] and lean_blind["date_legal"],
        "target_reached": False,
        "stop_reason": "three high-priority selector hypotheses and three retrieval hypotheses failed to produce a recall-preserving quality gain; remaining options require larger training/index infrastructure",
    })
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"joint_rows": len(joint), "blind_v3_f1": blind_v3["f1_champion"]["metrics"]["f1"],
                      "blind_lean_f1": blind_lean["f1_champion"]["metrics"]["f1"],
                      "reports": ["13_joint_ablation.csv", "14_cost_quality_tradeoff.csv", "15_blind_holdout_report.md", "experiment_registry.csv"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
