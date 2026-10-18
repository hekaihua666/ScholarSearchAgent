"""One-shot validation of policies frozen before reading validation30 outcomes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_evaluation import aggregate, gold_papers, load_jsonl, metrics_from_papers


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap(values: list[float], iterations: int = 3000) -> list[float]:
    rng = random.Random(20260830); means = []
    for _ in range(iterations):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return [round(means[int(0.025 * iterations)], 6), round(means[int(0.975 * iterations)], 6)]


def select(papers: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    if policy["strategy"] == "fixed_k": return papers[: int(policy["k"])]
    if policy["strategy"] == "llm_threshold":
        threshold = float(policy["threshold"])
        return [paper for paper in papers if paper.get("llm_relevance_score") is not None
                and float(paper["llm_relevance_score"]) >= threshold]
    raise ValueError(policy["strategy"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", type=Path, default=ROOT / "configs/f1_selector_v3/final_validation_lock.json")
    ap.add_argument("--run-dir", type=Path, default=ROOT / "reports/frozen_eval/validation30_balanced_final")
    ap.add_argument("--dataset", type=Path, default=ROOT / "reports/frozen_eval/query_sets/validation30_private.jsonl")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "reports/f1_selector_v3")
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_bytes = args.lock.read_bytes(); lock = json.loads(lock_bytes)
    rows = load_jsonl(args.dataset)
    results: dict[str, Any] = {}
    detail_rows: list[dict[str, Any]] = []
    baseline_f1: dict[str, float] = {}
    for policy in lock["policies"]:
        metrics = []; counts = []
        for row in rows:
            response = json.loads((args.run_dir / "queries" / row["qid"] / "live_response.json").read_text(encoding="utf-8"))
            papers = response.get("papers") or []
            chosen = select(papers, policy); value = metrics_from_papers(chosen, gold_papers(row), max(1, len(chosen)))
            metrics.append(value); counts.append(len(chosen))
            detail_rows.append({"experiment_id": policy["experiment_id"], "query_id_hash": hashlib.sha256(row["qid"].encode()).hexdigest()[:12],
                                "precision": value["precision"], "recall": value["recall"], "f1": value["f1"],
                                "tp": value["tp"], "predicted": value["predicted"], "gold": value["gold"]})
        agg = aggregate(metrics); f1s = [x["f1"] for x in metrics]
        results[policy["experiment_id"]] = {"policy": policy, "metrics": agg,
            "mean_output_count": round(sum(counts) / len(counts), 6),
            "median_output_count": sorted(counts)[len(counts)//2],
            "p90_output_count": sorted(counts)[min(len(counts)-1, int(0.9*len(counts)))],
            "empty_output_rate": round(sum(x == 0 for x in counts) / len(counts), 6),
            "macro_f1_bootstrap_95": bootstrap(f1s)}
        if policy["experiment_id"].startswith("S0_"):
            baseline_f1 = {d["query_id_hash"]: d["f1"] for d in detail_rows if d["experiment_id"] == policy["experiment_id"]}
    for eid, result in results.items():
        values = [d for d in detail_rows if d["experiment_id"] == eid]
        result["wins_ties_losses_vs_s0"] = {
            "wins": sum(d["f1"] > baseline_f1[d["query_id_hash"]] for d in values),
            "ties": sum(d["f1"] == baseline_f1[d["query_id_hash"]] for d in values),
            "losses": sum(d["f1"] < baseline_f1[d["query_id_hash"]] for d in values),
        }
    payload = {"validation_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
               "dataset_sha256": sha256(args.dataset), "candidate_manifest_sha256": sha256(args.run_dir / "manifest.json"),
               "query_count": len(rows), "validation_used_for_tuning": False, "results": results}
    (args.output_dir / "validation_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "validation_per_query_hashed.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(detail_rows[0])); writer.writeheader(); writer.writerows(detail_rows)
    with (args.output_dir / "08_pareto_frontier.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["experiment_id", "precision", "recall", "f1", "mean_output_count", "empty_output_rate"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for eid, value in results.items():
            m = value["metrics"]["macro"]; writer.writerow({"experiment_id": eid, **{k:m[k] for k in ("precision","recall","f1")},
                "mean_output_count": value["mean_output_count"], "empty_output_rate": value["empty_output_rate"]})
    lines = ["# validation30 动态集合最终验证", "", f"- 预注册配置哈希：`{payload['validation_lock_sha256']}`。",
             "- 所有策略在读取本轮 validation30 结果前已写入 lock；本脚本不搜索 K 或阈值。",
             "- 使用同一 balanced frozen response，检索候选、身份匹配和日期规则完全一致；仅改变最终集合选择。", "",
             "| 配置 | P | R | F1 | 平均结果数 | 95% bootstrap CI | W/T/L |", "|---|---:|---:|---:|---:|---:|---:|"]
    for eid, value in results.items():
        m=value["metrics"]["macro"]; w=value["wins_ties_losses_vs_s0"]; ci=value["macro_f1_bootstrap_95"]
        lines.append(f"| {eid} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {value['mean_output_count']:.2f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {w['wins']}/{w['ties']}/{w['losses']} |")
    lines += ["", "## 解释边界", "", "该结果是 `official_set_f1_candidate` 兼容模式，不得在主办方私有 evaluator 未发布前写成官方排行榜分数。K=2 是 F1 型配置；K=8 是在 dev20 recall floor 下预注册的 balanced 配置。validation30 结果生成后不再修改 K 或阈值。", ""]
    (args.output_dir / "10_validation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
