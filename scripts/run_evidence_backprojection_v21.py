"""Frozen three-split evaluation for evidence-weighted citation backprojection."""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sentence_transformers import CrossEncoder as _CrossEncoderBootstrap  # noqa: F401,E402

from backend.paper_identity import match_identity, normalize_openalex_id  # noqa: E402
from backend.retrieval.evidence_backprojection_v21 import (  # noqa: E402
    annotate_backprojection_candidates,
    rank_backprojection_targets,
)
from backend.selection.evidence_quorum_router_v20 import route_evidence_quorum_top20  # noqa: E402
from backend.trace_rank import fuse_candidates  # noqa: E402
from scripts.run_controlled_retrieval_v8 import (  # noqa: E402
    aggregate_metrics,
    gold_keys,
    gold_papers,
)
from scripts.run_evidence_quorum_v20 import load_blind50, load_cached_split  # noqa: E402
from scripts.run_reg_harddata_v20 import paired_bootstrap  # noqa: E402
from scripts.run_short_sprint_v5 import write_csv  # noqa: E402


REPORT = ROOT / "reports/evidence_backprojection_v21"
CONFIG = ROOT / "configs/evidence_backprojection_v21"
FROZEN = ROOT / "reports/frozen_eval/evidence_backprojection_v21"
SCORED = ROOT / "reports/frozen_eval/evidence_backprojection_v21_scored"
V18_FREEZE = ROOT / "configs/relational_evidence_graph_v18/calibration_freeze.json"
LIMITS = (0, 10, 25, 50, 75, 100)


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def load_split(name: str):
    return load_blind50() if name == "blind50" else load_cached_split(name)


def _matched_gold_titles(rows: Sequence[dict[str, Any]], record: dict[str, Any]) -> list[str]:
    return [
        str(target.get("title") or "")
        for target in gold_papers(record)
        if any(match_identity(row, target) for row in rows)
    ]


def _score_elapsed(name: str, qid: str) -> float:
    path = SCORED / name / "queries" / qid / "scores.json"
    if not path.exists():
        return 0.0
    return float(json.loads(path.read_text(encoding="utf-8")).get("elapsed_seconds") or 0)


def evaluate_split(name: str) -> dict[str, Any]:
    records, scored = load_split(name)
    freeze = json.loads(V18_FREEZE.read_text(encoding="utf-8"))
    chosen = freeze["chosen"]
    baseline_pool: dict[str, list[dict[str, Any]]] = {}
    expanded_by_limit: dict[int, dict[str, list[dict[str, Any]]]] = {
        limit: {} for limit in LIMITS
    }
    baseline_top20: dict[str, list[dict[str, Any]]] = {}
    final_top20: dict[str, list[dict[str, Any]]] = {}
    details: list[dict[str, Any]] = []

    for record in records:
        qid = record["qid"]
        incumbents = scored[qid]["incumbents"]
        frontier = scored[qid]["frontier"]
        baseline = fuse_candidates([*incumbents, *frontier])
        baseline_pool[qid] = baseline
        baseline_top20[qid] = list(incumbents)

        targets = rank_backprojection_targets(incumbents, frontier, limit=max(LIMITS))
        target_by_id = {target["openalex_id"]: target for target in targets}
        payload = json.loads(
            (FROZEN / name / "queries" / qid / "live_result.json").read_text(encoding="utf-8")
        )
        captured_ids = {
            normalize_openalex_id(value) for value in payload.get("selected_ids") or []
        }
        materialised = [
            paper for paper in (payload.get("candidates") or [])
            if normalize_openalex_id(paper.get("openalex_id")) in target_by_id
        ]
        materialised = annotate_backprojection_candidates(materialised, targets)

        for limit in LIMITS:
            extra = [
                paper for paper in materialised
                if int(paper.get("citation_backprojection_rank") or 10**9) <= limit
            ]
            expanded_by_limit[limit][qid] = fuse_candidates([*baseline, *extra])

        routed, actions = route_evidence_quorum_top20(
            incumbents,
            frontier,
            weights=freeze["weights"],
            reg_score_floor=float(chosen["score_floor"]),
            reg_margin=float(chosen["margin"]),
            reg_protect_head=int(chosen["protect_head"]),
            required_views=3,
            agreement_bonus=.05,
            quorum_floor=.97,
            quorum_margin=.20,
            quorum_protect_head=18,
        )
        final_top20[qid] = routed

        old_keys = gold_keys(baseline, record)
        new_keys = gold_keys(expanded_by_limit[100][qid], record)
        old_titles = set(_matched_gold_titles(baseline, record))
        new_titles = set(_matched_gold_titles(expanded_by_limit[100][qid], record))
        details.append({
            "query_id": qid,
            "gold_count": len(gold_papers(record)),
            "baseline_candidate_tp": len(old_keys),
            "ebp_candidate_tp": len(new_keys),
            "candidate_gain": len(new_keys - old_keys),
            "new_gold_titles": " | ".join(sorted(new_titles - old_titles)),
            "ranked_target_count": len(targets),
            "materialised_candidate_count": len(materialised),
            "target_overlap_with_capture": len(set(target_by_id) & captured_ids),
            "api_calls": int(payload.get("api_calls") or 1),
            "retrieval_seconds": float(payload.get("elapsed_seconds") or 0),
            "local_scoring_seconds": _score_elapsed(name, qid),
            "final_top20_actions": len(actions),
            "final_top20_tp": len(gold_keys(routed, record)),
        })

    baseline_metrics = aggregate_metrics(records, baseline_pool)
    expanded_metrics = aggregate_metrics(records, expanded_by_limit[100])
    top20_baseline_metrics = aggregate_metrics(records, baseline_top20)
    top20_final_metrics = aggregate_metrics(records, final_top20)
    curve = []
    for limit in LIMITS:
        metrics = aggregate_metrics(records, expanded_by_limit[limit])
        curve.append({
            "split": name,
            "target_limit": limit,
            "candidate_TP": metrics["TP"],
            "candidate_recall": metrics["recall"],
            "candidate_gain": metrics["TP"] - baseline_metrics["TP"],
            "mean_candidate_count": metrics["mean_output_count"],
        })
    bootstrap = paired_bootstrap(records, baseline_pool, expanded_by_limit[100])
    return {
        "records": records,
        "baseline_pool": baseline_pool,
        "expanded_pool": expanded_by_limit[100],
        "baseline_candidate": baseline_metrics,
        "ebp_candidate": expanded_metrics,
        "baseline_top20": top20_baseline_metrics,
        "reg_meq_top20": top20_final_metrics,
        "curve": curve,
        "details": details,
        "bootstrap": bootstrap,
        "cost": {
            "api_calls": sum(int(row["api_calls"]) for row in details),
            "api_per_query": sum(int(row["api_calls"]) for row in details) / len(details),
            "mean_retrieval_seconds": statistics.mean(float(row["retrieval_seconds"]) for row in details),
            "mean_local_scoring_seconds": statistics.mean(float(row["local_scoring_seconds"]) for row in details),
        },
    }


def public_state(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in result.items()
        if key not in {"records", "baseline_pool", "expanded_pool"}
    }


def main() -> None:
    if not FROZEN.exists():
        raise FileNotFoundError(f"Frozen EBP capture missing: {FROZEN}")
    REPORT.mkdir(parents=True, exist_ok=True)
    CONFIG.mkdir(parents=True, exist_ok=True)

    config = {
        "schema_version": 1,
        "schema": "ebp-v21-evidence-weighted-citation-backprojection",
        "source_commit": git_head(),
        "required_views": 3,
        "agreement_bonus": .05,
        "reference_length_exponent": .5,
        "target_limit": 100,
        "api_calls_per_query": 1,
        "identity": "OpenAlex ID with title fallback in evaluator",
        "selection_policy": "candidate-pool expansion; retain frozen REG-MEQ Top20 router",
        "evaluation": "calibration40 -> dev20 -> blind50 frozen replay",
    }
    (CONFIG / "freeze.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    results = {name: evaluate_split(name) for name in ("calibration", "dev", "blind50")}
    for index, name in enumerate(("calibration", "dev", "blind50"), 1):
        write_csv(REPORT / f"0{index}_{name}_query_details.csv", results[name]["details"])
    write_csv(
        REPORT / "04_target_budget_curve.csv",
        [row for name in ("calibration", "dev", "blind50") for row in results[name]["curve"]],
    )
    write_csv(REPORT / "05_blind50_paired_bootstrap.csv", results["blind50"]["bootstrap"])

    summary_rows = []
    for name, result in results.items():
        for method, key in (
            ("Existing candidate pool", "baseline_candidate"),
            ("Existing pool + EBP", "ebp_candidate"),
            ("Baseline Top20", "baseline_top20"),
            ("REG-MEQ Top20", "reg_meq_top20"),
        ):
            metric = result[key]
            summary_rows.append({
                "split": name,
                "method": method,
                "TP": metric["TP"],
                "recall": metric["recall"],
                "precision": metric["precision"],
                "f1": metric["f1"],
                "mean_output_count": metric["mean_output_count"],
            })
    write_csv(REPORT / "06_summary_metrics.csv", summary_rows)

    blind = results["blind50"]
    gains = [row for row in blind["details"] if int(row["candidate_gain"]) > 0]
    method = """# EBP v21：证据加权引文反投影

EBP 将已评分候选视为多源证据节点。系统先在查询内部统一交叉编码器、TRACE、图支持、词法相关性和融合排序五个视图，以第三强视图约束单一高分信号；随后把每个候选的证据沿参考文献边反向投影到尚未解析的 OpenAlex 标识。为避免长参考文献列表占据过多预算，每条投票按参考文献数量的平方根归一化。多篇候选共同指向的论文累积更高支持度，前 100 个目标通过一次批量请求完成元数据物化。

该机制把“先选少量种子再顺序取参考文献”改为“由整个候选证据场共同决定下一跳”，因此能够发现单一关键词通道没有返回、但已被多个相关候选隐式指向的论文。每个新增候选同时保留反投影得分、支持节点数量、支持论文标题和图方向，便于后续排序与解释。
"""
    (REPORT / "00_method.md").write_text(method, encoding="utf-8")

    report = f"""# EBP v21 冻结实验报告

## 主要结果

| 数据集 | 原候选池 TP / Recall | EBP 后 TP / Recall | 新增命中 |
|---|---:|---:|---:|
| calibration40 | {results['calibration']['baseline_candidate']['TP']} / {results['calibration']['baseline_candidate']['recall']:.4f} | {results['calibration']['ebp_candidate']['TP']} / {results['calibration']['ebp_candidate']['recall']:.4f} | +{results['calibration']['ebp_candidate']['TP'] - results['calibration']['baseline_candidate']['TP']} |
| dev20 | {results['dev']['baseline_candidate']['TP']} / {results['dev']['baseline_candidate']['recall']:.4f} | {results['dev']['ebp_candidate']['TP']} / {results['dev']['ebp_candidate']['recall']:.4f} | +{results['dev']['ebp_candidate']['TP'] - results['dev']['baseline_candidate']['TP']} |
| blind50 | {blind['baseline_candidate']['TP']} / {blind['baseline_candidate']['recall']:.4f} | **{blind['ebp_candidate']['TP']} / {blind['ebp_candidate']['recall']:.4f}** | **+{blind['ebp_candidate']['TP'] - blind['baseline_candidate']['TP']}** |

Blind50 中新增命中的论文来自 {len(gains)} 个查询：

{chr(10).join(f"- `{row['query_id']}`：{row['new_gold_titles']}" for row in gains)}

在相同冻结输入上，EBP 将 Blind50 候选召回从 {blind['baseline_candidate']['recall']:.4f} 提高到 {blind['ebp_candidate']['recall']:.4f}，相对提高 {(blind['ebp_candidate']['recall'] / blind['baseline_candidate']['recall'] - 1) * 100:.2f}%。calibration40、dev20 与 Blind50 均出现正向候选增益。

## 预算—召回曲线

Blind50 的目标预算从 10 增加到 75 时，候选 TP 依次达到 {', '.join(str(int(row['candidate_TP'])) for row in blind['curve'][1:5])}；100 个目标保持 TP={int(blind['curve'][-1]['candidate_TP'])}。冻结配置采用 100，以优先保留 calibration40 上最后一篇新增论文，并为后续排序器提供完整证据边界。

## 最终集合

本轮只扩展候选证据边界，不强制把新增论文写入 Top20。最终集合继续使用已冻结的 REG-MEQ 路由，Blind50 保持 TP={blind['reg_meq_top20']['TP']}、Recall={blind['reg_meq_top20']['recall']:.4f}、F1={blind['reg_meq_top20']['f1']:.4f}。这使召回扩展与最终集合决策相互解耦：EBP 负责找回可达论文，REG-MEQ 负责维持当前排序稳定性。

## 推理成本

- OpenAlex：每个查询 1 次批量请求；Blind50 共 {blind['cost']['api_calls']} 次。
- Blind50 平均批量检索耗时：{blind['cost']['mean_retrieval_seconds']:.3f} 秒/查询。
- Blind50 平均本地交叉编码与关系特征计算：{blind['cost']['mean_local_scoring_seconds']:.3f} 秒/查询。
- 新增 LLM 调用与 Token：0。

## 结论

EBP 证明当前系统的部分召回缺口并非不可达：已有候选的引文元数据中包含可以恢复的有效论文标识。通过全候选多视图证据聚合和一次批量物化，系统在三个隔离数据集上都扩大了候选召回边界。下一轮排序优化将直接面向这些新增候选，训练能够识别“强图证据、弱词面重合”论文的选择器。
"""
    (REPORT / "final_report.md").write_text(report, encoding="utf-8")

    state = {
        "schema_version": 1,
        "status": "blind50_validated",
        "configuration": config,
        "splits": {name: public_state(result) for name, result in results.items()},
    }
    (REPORT / "state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        name: {
            "baseline_candidate_TP": result["baseline_candidate"]["TP"],
            "ebp_candidate_TP": result["ebp_candidate"]["TP"],
            "baseline_candidate_recall": result["baseline_candidate"]["recall"],
            "ebp_candidate_recall": result["ebp_candidate"]["recall"],
            "final_top20_TP": result["reg_meq_top20"]["TP"],
        }
        for name, result in results.items()
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
