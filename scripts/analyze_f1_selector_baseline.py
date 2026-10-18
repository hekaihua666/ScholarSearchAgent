"""Audit fixed-Top20 cardinality and selector oracle gaps on frozen dev data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.paper_identity import match_identity
from scripts.audit_evaluation import aggregate, gold_papers, load_jsonl, metrics_from_papers, source_cutoff
from scripts.run_frozen_official_benchmark import module_candidates
from backend.trace_rank import fuse_candidates


def stage(audit: dict[str, Any], name: str) -> list[dict[str, Any]]:
    return next((item.get("papers") or [] for item in audit.get("stages", []) if item.get("name") == name), [])


def matched_count(papers: list[dict[str, Any]], gold: list[dict[str, Any]]) -> int:
    used: set[int] = set()
    for paper in papers:
        found = next((i for i, target in enumerate(gold) if i not in used and match_identity(paper, target)), None)
        if found is not None:
            used.add(found)
    return len(used)


def perfect_fixed_f1(k: int, gold_count: int) -> float:
    tp = min(k, gold_count)
    return 2 * tp / (k + gold_count) if k + gold_count else 0.0


def build(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_jsonl(args.dataset)[: args.query_count]
    per_query: list[dict[str, Any]] = []
    metrics_by_k: dict[int, list[dict[str, float]]] = {k: [] for k in range(1, 21)}
    candidate_oracle_rows: list[dict[str, float]] = []
    metadata_oracle_rows: list[dict[str, float]] = []
    dynamic_rows: list[dict[str, float]] = []
    gold_dist = Counter()

    for row in rows:
        qdir = args.run_dir / "queries" / row["qid"]
        response = json.loads((qdir / "live_response.json").read_text(encoding="utf-8"))
        audit = json.loads((qdir / "live_audit.json").read_text(encoding="utf-8"))
        snapshot = json.loads((qdir / "snapshot.json").read_text(encoding="utf-8"))
        gold = gold_papers(row); gold_dist[len(gold)] += 1
        final = response.get("papers") or []
        ranked32 = stage(audit, "shortlist_post_llm") or final
        merged_audit = stage(audit, "merged_candidates")
        cutoff = source_cutoff(row)
        assert cutoff is not None
        modules = module_candidates(snapshot, cutoff)
        merged_full = fuse_candidates([
            paper for name in ("lexical", "semantic", "arxiv", "title", "citation", "second")
            for paper in modules[name]
        ])

        current_by_k: dict[int, float] = {}
        for k in range(1, 21):
            value = metrics_from_papers(final, gold, k)
            metrics_by_k[k].append(value); current_by_k[k] = value["f1"]
        dynamic_candidates = [(k, metrics_from_papers(ranked32, gold, k)) for k in range(0, len(ranked32) + 1)]
        best_k, best_metrics = max(dynamic_candidates, key=lambda item: (item[1]["f1"], -item[0]))
        dynamic_rows.append(best_metrics)

        candidate_tp = matched_count(merged_audit or merged_full, gold)
        complete_gold = []
        for target in gold:
            hit = next((paper for paper in merged_full if match_identity(paper, target)), None)
            if hit and str(hit.get("abstract") or "").strip(): complete_gold.append(hit)
        metadata_tp = len(complete_gold)
        g = len(gold)
        def oracle(tp: int) -> dict[str, float]:
            return {"precision": 1.0 if tp else 0.0, "recall": tp / g if g else 0.0,
                    "f1": 2 * tp / (tp + g) if tp + g else 0.0, "ndcg": tp / g if g else 0.0,
                    "tp": tp, "predicted": tp, "gold": g}
        candidate_oracle_rows.append(oracle(candidate_tp)); metadata_oracle_rows.append(oracle(metadata_tp))
        base20 = metrics_from_papers(final, gold, 20)
        per_query.append({
            "query_id": row["qid"], "gold_count": g, "baseline_output_count": len(final),
            "baseline_tp": base20["tp"], "baseline_f1": round(base20["f1"], 6),
            "candidate_count": len(merged_audit or merged_full), "candidate_gold": candidate_tp,
            "metadata_complete_candidate_gold": metadata_tp, "oracle_dynamic_k": best_k,
            "oracle_dynamic_f1": round(best_metrics["f1"], 6),
            "fixed20_perfect_f1": round(perfect_fixed_f1(20, g), 6),
            **{f"current_f1_k{k}": round(current_by_k[k], 6) for k in range(1, 21)},
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "01_cardinality_details.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_query[0])); writer.writeheader(); writer.writerows(per_query)

    k_summary = {
        str(k): {
            "current": aggregate(values),
            "perfect_fixed_k_macro_f1": round(sum(perfect_fixed_f1(k, row["gold_count"]) for row in per_query) / len(per_query), 6),
        }
        for k, values in metrics_by_k.items()
    }
    result = {
        "query_set": f"dev{len(rows)}", "run_dir": str(args.run_dir),
        "gold_count_distribution": dict(sorted(gold_dist.items())),
        "mean_gold_count": round(sum(r["gold_count"] for r in per_query) / len(per_query), 6),
        "baseline_fixed20": aggregate(metrics_by_k[20]),
        "current_ranking_oracle_dynamic_k": aggregate(dynamic_rows),
        "candidate_oracle": aggregate(candidate_oracle_rows),
        "metadata_complete_candidate_oracle": aggregate(metadata_oracle_rows),
        "k_summary": k_summary, "per_query": per_query,
    }
    (args.output_dir / "baseline_oracles.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    fixed20_ceiling = sum(row["fixed20_perfect_f1"] for row in per_query) / len(per_query)
    best_global_k = max(range(1, 21), key=lambda k: k_summary[str(k)]["current"]["macro"]["f1"])
    cardinality = f"""# 固定 Top20 与结果数量上限审计

## 审计范围

- 只使用冻结的 `dev{len(rows)}`，不使用 validation30 调整策略。
- 基线快照：`{args.run_dir}`。
- 论文身份：稳定 ID 优先、标准化标题 fallback（沿用当前 `match_identity`）。
- 本页的集合 F1 使用 `2TP/(|R|+|G|)`；它是**候选官方协议**，尚不能称为主办方私有 evaluator 的最终口径。

## 基本数量事实

- gold 数量分布：`{dict(sorted(gold_dist.items()))}`。
- 平均 gold 数：**{result['mean_gold_count']:.3f}**。
- 当前输出数量：每条固定 20 篇。
- 即使 Top20 中全部先放 gold，固定 K=20 的 macro F1 理论上限也只有 **{fixed20_ceiling:.4f}**。

## 当前排序与动态数量

- 当前 fixed-Top20 macro F1：**{result['baseline_fixed20']['macro']['f1']:.4f}**。
- 当前排序下，逐查询 oracle dynamic-k macro F1：**{result['current_ranking_oracle_dynamic_k']['macro']['f1']:.4f}**。
- 不使用 gold 的单一固定 K 扫描中，当前排序最佳 K={best_global_k}，macro F1={k_summary[str(best_global_k)]['current']['macro']['f1']:.4f}。

逐查询与 K=1…20 明细见 `01_cardinality_details.csv` 和 `baseline_oracles.json`。oracle dynamic-k 只用于上限分析，绝不进入线上逻辑。
"""
    (args.output_dir / "01_cardinality_ceiling.md").write_text(cardinality, encoding="utf-8")

    cand = result["candidate_oracle"]; dyn = result["current_ranking_oracle_dynamic_k"]
    gap = f"""# Candidate / Ranking / Selector Oracle Gap

## 冻结 dev{len(rows)} 结果

| 层级 | Macro Precision | Macro Recall | Macro F1 | Micro TP |
|---|---:|---:|---:|---:|
| 当前 fixed Top20 | {result['baseline_fixed20']['macro']['precision']:.4f} | {result['baseline_fixed20']['macro']['recall']:.4f} | {result['baseline_fixed20']['macro']['f1']:.4f} | {result['baseline_fixed20']['micro']['true_positive']} |
| 当前排序 oracle dynamic-k | {dyn['macro']['precision']:.4f} | {dyn['macro']['recall']:.4f} | {dyn['macro']['f1']:.4f} | {dyn['micro']['true_positive']} |
| 候选池 oracle set selector | {cand['macro']['precision']:.4f} | {cand['macro']['recall']:.4f} | {cand['macro']['f1']:.4f} | {cand['micro']['true_positive']} |
| 摘要完整候选 oracle | {result['metadata_complete_candidate_oracle']['macro']['precision']:.4f} | {result['metadata_complete_candidate_oracle']['macro']['recall']:.4f} | {result['metadata_complete_candidate_oracle']['macro']['f1']:.4f} | {result['metadata_complete_candidate_oracle']['micro']['true_positive']} |

## 解释边界

- `candidate oracle` 证明候选池中已有多少可识别 gold，不代表可由当前特征完美区分。
- `ranking oracle dynamic-k` 使用真实 gold 选择 k，只是回答“当前排序若数量决策完美，最多能到哪里”。
- 摘要完整 oracle 单列是为了量化 metadata recovery 的潜在空间。
- 是否能达到 0.15/0.20/0.30，应先看上述 oracle，再由未见校准集上的真实 selector 结果判定，不能从 oracle 直接宣称。
"""
    (args.output_dir / "04_oracle_gap_report.md").write_text(gap, encoding="utf-8")
    return result


def write_protocol(output_dir: Path) -> None:
    text = """# 官方 F1 协议核验（v3）

## 已由赛题材料确认

1. 自动评分权重为 F1 Score 70%、运行效率 20%、结构化输出 10%。
2. 系统需完成查询理解与分解、自主搜索策略迭代、论文综合排序和结构化归纳。
3. 至少对接一种学术搜索 API，并记录 API、Token 与延时成本。
4. 赛题引用 PaSa、SPAR、AstaBench 作为参考，但没有说明竞赛私有 evaluator 与其中任一开源 evaluator 完全一致。

来源：参赛者提供的官方赛题全文及[赛事页面](https://www.chaspark.com/#/races/competitions/1264753432170905600)。赛事页面为动态页面，当前公开抓取未暴露附件内容。

## 尚未由官方材料确认

- 最终结果是否固定 TopK、是否允许逐查询动态数量、最大返回数和空结果规则；
- F1 是完整集合 F1、F1@K、macro 还是 micro；
- 重复项、部分相关、无效论文及缺失元数据的计分规则；
- DOI/arXiv/OpenAlex/标题的官方身份匹配优先级；
- 查询截止日期、未来论文、版本号处理；
- 提交 API/JSON schema 与隐藏评测脚本。

因此，在拿到主办方接口说明前，项目不得把当前 `F1@20` 称为“官方 F1”。

## 参考实现核验

PaSa 的公开 `metrics.py` 将 `select_score > 0.5` 的论文组成**可变长度 selected set**，按字母数字归一化标题匹配，逐查询计算 selector precision/recall 后求平均；Crawler 另报 Recall@20/50/100。它证明“高召回 Crawler + Selector 最终集合”是参考系统真实结构，但不能证明本赛事私有协议与 PaSa 相同：

- https://github.com/bytedance/pasa/blob/main/metrics.py
- https://github.com/bytedance/pasa

## 项目当前实现

- 当前 `backend/evaluation.py::retrieval_metrics`：固定 `k` 截断、去重、计算 P/R/F1/NDCG@K。
- 历史 benchmark runner：固定 Top20，macro 聚合，同时另报 micro。
- 身份匹配：ID 优先、标题 fallback；比 PaSa 单纯标题匹配更稳，但并非官方确认。
- 日期：AutoScholarQuery `source_meta.published_time - 7 days`；是项目的防未来泄漏策略，不是已确认竞赛规则。

## 本轮兼容模式

1. `official_set_f1_candidate`：对完整 `selected_results` 计算 `2TP/(|R|+|G|)`，允许动态数量；名称明确标注 candidate。
2. `f1_at_20_compat`：保留当前固定 Top20 P/R/F1/NDCG，确保能与既有 frozen 报告复核。

两个模式同时保留，直到主办方 evaluator 与提交 schema 被正式确认。
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "00_official_metric_protocol.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\AICompetition\references\github\SPAR\benchmark\AutoScholarQuery_test.jsonl"))
    parser.add_argument("--run-dir", type=Path, default=ROOT / "reports/frozen_eval/autoscholar20_trace_v4_official_20260824_v2")
    parser.add_argument("--query-count", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/f1_selector_v3")
    args = parser.parse_args()
    write_protocol(args.output_dir)
    print(json.dumps(build(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
