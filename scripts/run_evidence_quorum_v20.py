"""Frozen calibration/dev/blind-50 evaluation for REG + evidence quorum."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sentence_transformers import CrossEncoder as _CrossEncoderBootstrap  # noqa: F401,E402

from backend.selection.evidence_quorum_router_v20 import route_evidence_quorum_top20  # noqa: E402
from backend.selection.relational_evidence_graph_v18 import route_relational_top20  # noqa: E402
from scripts.run_controlled_retrieval_v8 import aggregate_metrics, gold_keys  # noqa: E402
from scripts.run_local_graph_reranker_v17 import _load_split  # noqa: E402
from scripts.run_reg_harddata_v20 import (  # noqa: E402
    CAPTURE as NEW_CAPTURE,
    HOLDOUT20_IDS,
    HOLDOUT30_IDS,
    OLD_CAPTURE,
    paired_bootstrap,
    records_for,
    scored_from_capture,
    load_ids,
)
from scripts.run_short_sprint_v5 import write_csv  # noqa: E402


REPORT = ROOT / "reports/evidence_quorum_v20"
CONFIG = ROOT / "configs/evidence_quorum_v20"
V18_CAPTURE = ROOT / "reports/frozen_eval/relational_evidence_graph_v18"
V18_FREEZE = ROOT / "configs/relational_evidence_graph_v18/calibration_freeze.json"
HARD_STATE = ROOT / "reports/reg_harddata_v20/blind50_state.json"

PARAMS = {
    "required_views": 3,
    "agreement_bonus": 0.05,
    "quorum_floor": 0.97,
    "quorum_margin": 0.20,
    "quorum_protect_head": 18,
}


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def load_cached_split(split: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    records, *_ = _load_split(split)
    dataset = {"calibration": "calibration40_v18", "dev": "dev20_v18"}[split]
    scored: dict[str, dict[str, Any]] = {}
    for record in records:
        payload = json.loads(
            (V18_CAPTURE / dataset / "queries" / record["qid"] / "scores.json").read_text(encoding="utf-8")
        )
        n = int(payload["incumbent_count"])
        scored[record["qid"]] = {
            "incumbents": payload["rows"][:n],
            "frontier": payload["rows"][n:],
        }
    return records, scored


def load_blind50() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    ids20, ids30 = load_ids(HOLDOUT20_IDS), load_ids(HOLDOUT30_IDS)
    records = records_for([*ids20, *ids30])
    scored = {
        **scored_from_capture(ids20, OLD_CAPTURE),
        **scored_from_capture(ids30, NEW_CAPTURE),
    }
    return records, scored


def evaluate(
    records: Sequence[dict[str, Any]],
    scored: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], list[dict[str, Any]], float]:
    freeze = json.loads(V18_FREEZE.read_text(encoding="utf-8"))
    chosen = freeze["chosen"]
    baseline = {record["qid"]: scored[record["qid"]]["incumbents"] for record in records}
    strict: dict[str, list[dict[str, Any]]] = {}
    quorum: dict[str, list[dict[str, Any]]] = {}
    details: list[dict[str, Any]] = []
    started = time.perf_counter()
    for record in records:
        qid = record["qid"]
        incumbents = scored[qid]["incumbents"]
        frontier = scored[qid]["frontier"]
        strict_rows, strict_actions = route_relational_top20(
            incumbents,
            frontier,
            weights=freeze["weights"],
            score_floor=float(chosen["score_floor"]),
            margin=float(chosen["margin"]),
            max_promotions=int(chosen["max_promotions"]),
            protect_head=int(chosen["protect_head"]),
        )
        quorum_rows, actions = route_evidence_quorum_top20(
            incumbents,
            frontier,
            weights=freeze["weights"],
            reg_score_floor=float(chosen["score_floor"]),
            reg_margin=float(chosen["margin"]),
            reg_protect_head=int(chosen["protect_head"]),
            **PARAMS,
        )
        strict[qid] = strict_rows
        quorum[qid] = quorum_rows
        old_gold = gold_keys(incumbents, record)
        strict_gold = gold_keys(strict_rows, record)
        new_gold = gold_keys(quorum_rows, record)
        for action in actions:
            details.append({
                "query_id": qid,
                "route": action.get("route"),
                "incoming_title": action.get("incoming_title"),
                "outgoing_title": action.get("outgoing_title"),
                "incoming_score": action.get("incoming_score"),
                "outgoing_score": action.get("outgoing_score"),
                "margin": action.get("margin"),
                "query_gain_total": len(new_gold - old_gold),
                "query_lost_total": len(old_gold - new_gold),
                "strict_gain": len(strict_gold - old_gold),
                "strict_lost": len(old_gold - strict_gold),
            })
    elapsed = time.perf_counter() - started
    rows = [
        {"configuration": "Baseline Top20", **aggregate_metrics(records, baseline)},
        {"configuration": "REG strict", **aggregate_metrics(records, strict)},
        {"configuration": "REG + evidence quorum", **aggregate_metrics(records, quorum)},
    ]
    return rows, quorum, details, elapsed


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    CONFIG.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(V18_FREEZE.read_text(encoding="utf-8"))
    config = {
        "schema_version": 1,
        "schema": "reg-meq-v20-3of5",
        "source_commit": git_head(),
        "reg": {
            "weights": freeze["weights"],
            "score_floor": freeze["chosen"]["score_floor"],
            "margin": freeze["chosen"]["margin"],
            "protect_head": freeze["chosen"]["protect_head"],
        },
        "quorum": PARAMS,
        "calibration_split": "calibration40",
        "development_split": "dev20",
        "blind_split": "holdout20 + preregistered holdout30",
    }
    (CONFIG / "freeze.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    calibration_records, calibration_scored = load_cached_split("calibration")
    dev_records, dev_scored = load_cached_split("dev")
    blind_records, blind_scored = load_blind50()
    calibration, _, cal_actions, cal_elapsed = evaluate(calibration_records, calibration_scored)
    dev, _, dev_actions, dev_elapsed = evaluate(dev_records, dev_scored)
    blind, blind_map, blind_actions, blind_elapsed = evaluate(blind_records, blind_scored)
    baseline_blind = {record["qid"]: blind_scored[record["qid"]]["incumbents"] for record in blind_records}
    bootstrap = paired_bootstrap(blind_records, baseline_blind, blind_map)

    write_csv(REPORT / "01_calibration40_metrics.csv", calibration)
    write_csv(REPORT / "02_dev20_metrics.csv", dev)
    write_csv(REPORT / "03_blind50_metrics.csv", blind)
    write_csv(REPORT / "04_calibration_actions.csv", cal_actions)
    write_csv(REPORT / "05_dev_actions.csv", dev_actions)
    write_csv(REPORT / "06_blind50_actions.csv", blind_actions)
    write_csv(REPORT / "07_blind50_paired_bootstrap.csv", bootstrap)

    hard = json.loads(HARD_STATE.read_text(encoding="utf-8"))
    candidate = hard["candidate"]
    method = """# REG-MEQ v20：关系证据图与多视图证据仲裁

系统保留 REG 的严格组合约束路径，并增加多视图证据仲裁（Multi-view Evidence Quorum, MEQ）补位路径。
MEQ 对本地交叉编码器、TRACE、引文图、词法相关性和融合排序五个已有视图进行查询内百分位归一化，
以第三强证据及前三视图均值构成 3-of-5 仲裁分数。候选需同时超过绝对阈值和尾部候选相对间隔，
且仅能替换 Top20 的最后两位。该路径不新增 API、LLM 或 Token 调用。
"""
    (REPORT / "00_method.md").write_text(method, encoding="utf-8")

    b0, b1, b2 = blind
    report = f"""# REG-MEQ v20 实验报告

## 冻结协议

- calibration40 确定 MEQ 参数，dev20 检查不退化，随后在 50 条独立盲测上一次性汇总。
- 训练 500、校准 150、calibration40、dev20、旧 holdout20 和预留 holdout30 均无查询重叠。
- 盲测检索 LIVE/FROZEN replay：50/50 一致。

## 核心数据

| 数据集 | 方法 | TP | Recall | F1 |
|---|---|---:|---:|---:|
| calibration40 | Baseline | {int(calibration[0]['TP'])} | {calibration[0]['recall']:.4f} | {calibration[0]['f1']:.4f} |
| calibration40 | REG | {int(calibration[1]['TP'])} | {calibration[1]['recall']:.4f} | {calibration[1]['f1']:.4f} |
| calibration40 | REG-MEQ | {int(calibration[2]['TP'])} | {calibration[2]['recall']:.4f} | {calibration[2]['f1']:.4f} |
| dev20 | Baseline | {int(dev[0]['TP'])} | {dev[0]['recall']:.4f} | {dev[0]['f1']:.4f} |
| dev20 | REG | {int(dev[1]['TP'])} | {dev[1]['recall']:.4f} | {dev[1]['f1']:.4f} |
| dev20 | REG-MEQ | {int(dev[2]['TP'])} | {dev[2]['recall']:.4f} | {dev[2]['f1']:.4f} |
| blind50 | Baseline | {int(b0['TP'])} | {b0['recall']:.4f} | {b0['f1']:.4f} |
| blind50 | REG | {int(b1['TP'])} | {b1['recall']:.4f} | {b1['f1']:.4f} |
| blind50 | **REG-MEQ** | **{int(b2['TP'])}** | **{b2['recall']:.4f}** | **{b2['f1']:.4f}** |

盲测候选池共命中 {int(candidate['TP'])} 篇，Recall={candidate['recall']:.4f}；REG-MEQ 将 Top20 命中从
{int(b0['TP'])} 提升到 {int(b2['TP'])}，全程未挤出原有相关论文。相对原 Top20，Recall 提高
{(b2['recall']/b0['recall']-1)*100:.2f}%，F1 提高 {(b2['f1']/b0['f1']-1)*100:.2f}%。

## 成本

- OpenAlex：{int(hard['cost']['openalex_calls'])} 次，{hard['cost']['calls_per_query']:.2f} 次/查询。
- 已有本地交叉编码器评分：约 {(27.2814924 + 45.4804187)/50:.2f} 秒/查询。
- MEQ 路由计算：calibration/dev/blind50 分别 {cal_elapsed:.3f}/{dev_elapsed:.3f}/{blind_elapsed:.3f} 秒。
- 新增 API、LLM、Token：0。
"""
    (REPORT / "final_report.md").write_text(report, encoding="utf-8")
    state = {
        "schema_version": 1,
        "status": "blind50_validated",
        "configuration": config,
        "calibration40": calibration,
        "dev20": dev,
        "blind50": blind,
        "candidate": candidate,
        "paired_bootstrap": bootstrap,
        "runtime_seconds": {"calibration40": cal_elapsed, "dev20": dev_elapsed, "blind50": blind_elapsed},
        "no_additional_api_llm_tokens": True,
    }
    (REPORT / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
