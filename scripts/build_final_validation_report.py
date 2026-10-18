"""Build aggregate final-evaluation tables without exposing holdout query gold."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "reports/final_evaluation"
OVERNIGHT = ROOT / "reports/overnight_iteration"


RUNS = {
    "dev20_balanced_v2": ROOT / "reports/frozen_eval/autoscholar20_trace_v4_official_20260824_v2/manifest.json",
    "validation30_balanced": ROOT / "reports/frozen_eval/validation30_balanced_final/manifest.json",
    "validation30_efficient": ROOT / "reports/frozen_eval/validation30_efficient_final/manifest.json",
    "validation30_helios": ROOT / "reports/frozen_eval/validation30_helios_final/manifest.json",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def run_stats(name: str, path: Path) -> dict:
    data = load(path)
    run_dir = path.parent
    actions: Counter[str] = Counter()
    provider_seconds = 0.0
    llm_tokens = 0
    for query in data["queries"]:
        query_dir = run_dir / "queries" / query["query_id"]
        response = load(query_dir / "live_response.json")
        trace = (response.get("search_process") or {}).get("helios_trace") or []
        if isinstance(trace, dict):
            trace = [trace]
        actions.update(step.get("action", "unknown") for step in trace)
        snapshot = load(query_dir / "snapshot.json")
        for call in (snapshot.get("calls") or {}).values():
            provider_seconds += float(call.get("elapsed_seconds") or call.get("duration_seconds") or 0)
            llm_tokens += int(call.get("total_tokens") or 0)
    at20 = data["baseline_live"]["metrics"]["20"]
    return {
        "label": name,
        "run_id": data["run_id"],
        "query_count": data["query_count"],
        "query_set": "dev20" if name.startswith("dev20") else "validation30",
        "search_mode": data.get("search_mode", "balanced"),
        "macro_precision_at20": at20["macro"]["precision"],
        "macro_recall_at20": at20["macro"]["recall"],
        "macro_f1_at20": at20["macro"]["f1"],
        "macro_ndcg_at20": at20["macro"]["ndcg"],
        "micro_true_positive_at20": at20["micro"]["true_positive"],
        "micro_recall_at20": at20["micro"]["recall"],
        "api_calls": sum(q["api_calls_recorded"] for q in data["queries"]),
        "external_errors": sum(len(q["external_call_errors"]) for q in data["queries"]),
        "provider_call_seconds": round(provider_seconds, 3),
        "llm_tokens_if_recorded": llm_tokens,
        "helios_actions": json.dumps(dict(actions), ensure_ascii=False),
        "replay_identical": data["replay_all_identical"],
        "date_legal": data["all_final_papers_date_legal"],
        "manifest_sha256": file_hash(path),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    stats = [run_stats(name, path) for name, path in RUNS.items()]
    write_csv(FINAL / "baseline_vs_champion.csv", stats)
    write_csv(FINAL / "cost_quality_tradeoff.csv", stats)

    balanced = stats[1]
    efficient = stats[2]
    helios = stats[3]
    commit = git_commit()
    text = f"""# validation30 最终验证报告

## 协议

- 固定查询集：`validation30`，30 条，构造 seed `20260830`，与 dev20 不重叠。
- 数据集哈希：`a6f1ad2ed677b11aef59b80229f4c8e953f54bb73e2b88e6e7319b9d3b5a368a`。
- 日期规则：`source_meta.published_time - 7 days`；所有最终论文日期合法。
- 身份匹配：稳定 ID 优先，标准化标题仅作 fallback。
- 三个配置在评测前已冻结；未根据 validation30 的逐查询 gold 调参。
- 当前 Git 基点：`{commit}`（本报告生成前）。

## 聚合结果（含 LLM rerank）

| 配置 | P@20 | R@20 | F1@20 | NDCG@20 | TP@20 | API 调用 | 外部错误 |
|---|---:|---:|---:|---:|---:|---:|---:|
| balanced | {balanced['macro_precision_at20']:.4f} | {balanced['macro_recall_at20']:.4f} | {balanced['macro_f1_at20']:.4f} | {balanced['macro_ndcg_at20']:.4f} | {balanced['micro_true_positive_at20']} | {balanced['api_calls']} | {balanced['external_errors']} |
| efficient | {efficient['macro_precision_at20']:.4f} | {efficient['macro_recall_at20']:.4f} | {efficient['macro_f1_at20']:.4f} | {efficient['macro_ndcg_at20']:.4f} | {efficient['micro_true_positive_at20']} | {efficient['api_calls']} | {efficient['external_errors']} |
| HELIOS-Lite | {helios['macro_precision_at20']:.4f} | {helios['macro_recall_at20']:.4f} | {helios['macro_f1_at20']:.4f} | {helios['macro_ndcg_at20']:.4f} | {helios['micro_true_positive_at20']} | {helios['api_calls']} | {helios['external_errors']} |

## 判定

1. **最终质量 champion 保持 balanced。** 它在 validation30 的 Recall@20、F1@20、NDCG@20 均最高。
2. efficient 将 API 调用从 {balanced['api_calls']} 降至 {efficient['api_calls']}（{(1-efficient['api_calls']/balanced['api_calls'])*100:.1f}%），但 Recall@20 下降 {balanced['macro_recall_at20']-efficient['macro_recall_at20']:.4f}，只能作为显式低成本模式，不能替代默认质量模式。
3. HELIOS-Lite 在 30 条上全部选择 `stop`，API 成本与 efficient 相同，但没有保持质量；它仍是可运行、可解释的实验控制器，**不作为已验证提升写入主结论**。
4. balanced 冻结快照的通道消融显示 citation 与 second round 分别增加候选 gold，但固定排序下没有同步提高 Top20，说明扩展有真实候选收益，也有排序噪声。

## 重要限制

- 三个模式分别进行了 LIVE 抓取，外部索引与 LLM 响应存在时间漂移；因此跨 run 差值同时包含策略差异与提供方波动。每个 run 内 LIVE/FROZEN 完全一致，但跨模式并非同一候选快照的严格配对试验。
- 严格策略因果结论应优先引用同一 balanced 快照上的通道消融；三次完整 LIVE 结果用于系统级鲁棒性验证。
- validation30 只用于本次冻结配置的最终判断，不据其逐查询结果继续调参。
"""
    (FINAL / "validation_report.md").write_text(text, encoding="utf-8")

    champion = load(OVERNIGHT / "champion.json")
    champion["final_validation"] = {
        "decision": "balanced_remains_quality_champion",
        "quality_champion": balanced,
        "low_cost_mode": efficient,
        "helios_lite": helios,
        "claim_boundary": "HELIOS-Lite is implemented and explainable but did not generalize as a quality improvement on validation30.",
    }
    (OVERNIGHT / "champion.json").write_text(json.dumps(champion, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
