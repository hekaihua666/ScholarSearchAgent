"""Frozen runner for Coverage-Adaptive Contract Verification (COVR) v10."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.eval_runtime import EvaluationSession, evaluation_session, frozen_call
from backend.retrieval.controlled_facet_retrieval_v8 import gap_triggered
from backend.selection.coverage_adaptive_contract_v10 import (
    build_verification_cards,
    request_overflow_verification,
    score_verification_cards,
    should_verify_overflow,
)
from backend.selection.evidence_contract_ranker_v9 import (
    ContractCard,
    route_contract_core,
    route_top20,
    score_cards,
)
from scripts.audit_evaluation import source_cutoff
from scripts.run_controlled_retrieval_v8 import (
    aggregate_metrics,
    baseline_configuration,
    baseline_pool,
    baseline_top20_map,
    filter_sources,
    gold_keys,
    records_for,
    understanding_for,
    v6_selection,
)
from scripts.run_evidence_contract_rank_v9 import (
    baseline_metrics,
    candidate_pool,
    evaluate_profile as evaluate_v9_profile,
    restore_cards,
)
from scripts.run_frozen_official_benchmark import json_hash
from scripts.run_short_sprint_v5 import write_csv


REPORT = ROOT / "reports/coverage_adaptive_contract_v10"
CONFIG = ROOT / "configs/coverage_adaptive_contract_v10"
CAPTURE = ROOT / "reports/frozen_eval/coverage_adaptive_contract_v10"
V9_CONFIG = ROOT / "configs/evidence_contract_rank_v9/calibration_freeze.json"
V8_ROOTS = {
    "calibration": ROOT / "reports/frozen_eval/controlled_retrieval_v8/calibration40_v8/queries",
    "dev": ROOT / "reports/frozen_eval/controlled_retrieval_v8/dev20_v9_dsfr/queries",
}
V9_ROOTS = {
    "calibration": ROOT / "reports/frozen_eval/evidence_contract_rank_v9/calibration40_v9/queries",
    "dev": ROOT / "reports/frozen_eval/evidence_contract_rank_v9/dev20_v9_contract/queries",
}
ANCHOR_PROFILES = (False, True)
OVERFLOW_MARGINS = (0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0)


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_dirs() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    CONFIG.mkdir(parents=True, exist_ok=True)
    CAPTURE.mkdir(parents=True, exist_ok=True)


def frozen_v9_profile() -> dict[str, Any]:
    return json.loads(V9_CONFIG.read_text(encoding="utf-8"))["chosen"]


def load_inputs(
    records: Sequence[dict[str, Any]], split: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    dsfr: dict[str, dict[str, Any]] = {}
    contract: dict[str, dict[str, Any]] = {}
    for record in records:
        qid = record["qid"]
        dsfr[qid] = json.loads((V8_ROOTS[split] / qid / "live_result.json").read_text(encoding="utf-8"))
        contract[qid] = json.loads((V9_ROOTS[split] / qid / "live_result.json").read_text(encoding="utf-8"))
    return dsfr, contract


def raw_newcomers(record: dict[str, Any], dsfr: dict[str, Any]) -> list[dict[str, Any]]:
    return filter_sources(dsfr, {"arxiv", "openalex"}, record).get("candidates") or []


async def _safe_verify(
    query: str, contract: dict[str, Any], cards: Sequence[ContractCard],
) -> dict[str, Any]:
    try:
        return await request_overflow_verification(query, contract, cards)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {str(exc)[:300]}", "parsed": {}, "tokens": 0}


async def execute_verification(
    record: dict[str, Any],
    dsfr: dict[str, Any],
    contract_result: dict[str, Any],
    dataset_name: str,
    mode: str,
) -> tuple[dict[str, Any], EvaluationSession]:
    profile = frozen_v9_profile()
    threshold = float(profile["threshold"])
    margin = float(profile["margin"])
    max_replacements = int(profile["max_replacements"])
    qid = record["qid"]
    path = CAPTURE / dataset_name / "queries" / qid / "snapshot.json"
    session = EvaluationSession.create(dataset_name, qid, source_cutoff(record), mode, path)
    top20 = baseline_top20_map([record])[qid]
    initial_cards = restore_cards(contract_result)
    parsed = (contract_result.get("contract_result") or {}).get("parsed") or {}
    initial_scored = score_cards(initial_cards, parsed) if initial_cards and parsed else []
    if initial_scored:
        _, initial_actions, contract_mode = route_contract_core(
            record["question"], top20, initial_scored, parsed,
            threshold=threshold, margin=margin, max_replacements=max_replacements,
        )
    else:
        initial_actions, contract_mode = [], "not_evaluated"
    cards = build_verification_cards(
        record["question"], top20, raw_newcomers(record, dsfr), initial_cards
    ) if initial_scored else []
    fire = should_verify_overflow(
        initial_scored, initial_actions, cards, threshold=threshold
    ) if cards else False
    started = time.perf_counter()
    if fire:
        card_signature = hashlib.sha256(json.dumps([
            {"id": card.card_id, "title": card.paper.get("title"), "doi": card.paper.get("doi"),
             "arxiv": card.paper.get("arxiv_id"), "openalex": card.paper.get("openalex_id")}
            for card in cards
        ], ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        contract_signature = hashlib.sha256(
            json.dumps(parsed.get("contract") or {}, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        arguments = {
            "schema": "covr-v10-overflow-point-1",
            "query": record["question"],
            "contract_signature": contract_signature,
            "card_signature": card_signature,
            "card_count": len(cards),
        }
        with evaluation_session(session):
            payload = await frozen_call(
                "covr_overflow_point_v10", arguments,
                lambda: _safe_verify(record["question"], parsed.get("contract") or {}, cards),
            )
    else:
        payload = {"skipped": True, "reason": "resolved_or_no_overflow", "parsed": {}, "tokens": 0}
    result = {
        "query_id": qid,
        "triggered": fire,
        "contract_mode": contract_mode,
        "initial_action_count": len(initial_actions),
        "elapsed_seconds": time.perf_counter() - started,
        "cards": [
            {"card_id": card.card_id, "origin": card.origin,
             "original_rank": card.original_rank, "paper": card.paper}
            for card in cards
        ],
        "verification_result": payload,
    }
    if mode == "live":
        session.save({"source_commit": git_head(), "schema": "covr-v10-overflow-point-1"})
        path.parent.mkdir(parents=True, exist_ok=True)
        (path.parent / "live_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return result, session


def result_projection(result: dict[str, Any]) -> dict[str, Any]:
    payload = result.get("verification_result") or {}
    return {
        "triggered": result.get("triggered"),
        "cards": [(row["card_id"], row["origin"], row["paper"].get("title")) for row in result.get("cards") or []],
        "parsed": payload.get("parsed"),
        "error": payload.get("error"),
    }


async def capture_dataset(
    records: Sequence[dict[str, Any]],
    dsfr_results: dict[str, dict[str, Any]],
    contract_results: dict[str, dict[str, Any]],
    dataset_name: str,
) -> dict[str, Any]:
    ensure_dirs()
    results: dict[str, dict[str, Any]] = {}
    manifest: list[dict[str, Any]] = []
    for index, record in enumerate(records, 1):
        qid = record["qid"]
        root = CAPTURE / dataset_name / "queries" / qid
        live_path, snapshot_path = root / "live_result.json", root / "snapshot.json"
        if live_path.exists() and snapshot_path.exists():
            live = json.loads(live_path.read_text(encoding="utf-8"))
            live_session = EvaluationSession.create(dataset_name, qid, source_cutoff(record), "replay", snapshot_path)
        else:
            live, live_session = await execute_verification(
                record, dsfr_results[qid], contract_results[qid], dataset_name, "live"
            )
        replay, _ = await execute_verification(
            record, dsfr_results[qid], contract_results[qid], dataset_name, "replay"
        )
        identical = json_hash(result_projection(live)) == json_hash(result_projection(replay))
        results[qid] = live
        payload = live.get("verification_result") or {}
        calls = list(live_session.calls.values())
        manifest.append({
            "query_id": qid,
            "triggered": live.get("triggered"),
            "card_count": len(live.get("cards") or []),
            "overflow_count": sum(row.get("origin") == "overflow" for row in live.get("cards") or []),
            "tokens": int(payload.get("tokens") or 0),
            "elapsed_seconds": live.get("elapsed_seconds"),
            "llm_calls": sum(call.get("label") == "covr_overflow_point_v10" for call in calls),
            "error": payload.get("error"),
            "replay_identical": identical,
            "snapshot_sha256": sha(snapshot_path) if snapshot_path.exists() else None,
        })
        print(
            f"[{index}/{len(records)}] {qid} trigger={live.get('triggered')} "
            f"cards={len(live.get('cards') or [])} replay={'OK' if identical else 'DIFF'}",
            flush=True,
        )
    summary = {
        "dataset": dataset_name,
        "query_count": len(records),
        "source_commit": git_head(),
        "replay_all_identical": all(row["replay_identical"] for row in manifest),
        "queries": manifest,
    }
    (CAPTURE / dataset_name / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"manifest": summary, "results": results}


def restore_verification_cards(result: dict[str, Any]) -> list[ContractCard]:
    return [
        ContractCard(row["card_id"], row["origin"], row.get("original_rank"), row["paper"])
        for row in result.get("cards") or []
    ]


def evaluate_variant(
    records: Sequence[dict[str, Any]],
    dsfr_results: dict[str, dict[str, Any]],
    contract_results: dict[str, dict[str, Any]],
    verification_results: dict[str, dict[str, Any]],
    *,
    anchor_calibration: bool,
    overflow_margin: float,
) -> tuple[dict[str, Any], dict[str, dict[str, list[dict[str, Any]]]], list[dict[str, Any]]]:
    profile = frozen_v9_profile()
    threshold, margin = float(profile["threshold"]), float(profile["margin"])
    max_replacements = int(profile["max_replacements"])
    baseline_top = baseline_top20_map(records)
    core_map: dict[str, list[dict[str, Any]]] = {}
    top20_map: dict[str, list[dict[str, Any]]] = {}
    candidate_map: dict[str, list[dict[str, Any]]] = {}
    details: list[dict[str, Any]] = []
    core_gain = core_lost = top_gain = top_lost = actions_total = 0
    llm_calls = tokens = 0
    offsets: list[float] = []
    for record in records:
        qid = record["qid"]
        base = baseline_top[qid]
        candidate_map[qid] = candidate_pool(record, dsfr_results[qid])
        contract_result = contract_results[qid]
        initial_cards = restore_cards(contract_result)
        parsed = (contract_result.get("contract_result") or {}).get("parsed") or {}
        initial_scored = score_cards(initial_cards, parsed) if initial_cards and parsed else []
        verification = verification_results[qid]
        verify_cards = restore_verification_cards(verification)
        verify_parsed = (verification.get("verification_result") or {}).get("parsed") or {}
        overflow_scored, offset = score_verification_cards(
            verify_cards, verify_parsed, initial_scored,
            calibrate_with_anchors=anchor_calibration,
        ) if verify_cards and verify_parsed else ([], 0.0)
        combined = sorted(
            [*initial_scored, *overflow_scored],
            key=lambda row: (-float(row.get("contract_score") or 0), int(row.get("contract_rank") or 999)),
        )
        if initial_scored:
            initial_selected, initial_actions, contract_mode = route_contract_core(
                record["question"], base, initial_scored, parsed,
                threshold=threshold, margin=margin, max_replacements=max_replacements,
            )
            initial_routed = route_top20(base, initial_selected, initial_scored, threshold=threshold)
            # The overflow margin is calibrated independently, but it must not
            # change decisions among first-pass newcomers. COVR therefore
            # exposes only the frozen Top2 incumbents and overflow rows to the
            # second router. The trigger guarantees initial_actions is empty.
            if verification.get("triggered") and overflow_scored and not initial_actions:
                incumbents = [
                    row for row in initial_scored
                    if row.get("origin") == "baseline" and int(row.get("original_rank") or 99) <= 2
                ]
                selected, overflow_actions, contract_mode = route_contract_core(
                    record["question"], base, [*incumbents, *overflow_scored], parsed,
                    threshold=threshold, margin=overflow_margin,
                    max_replacements=max_replacements,
                )
                actions = [*initial_actions, *overflow_actions]
                routed = route_top20(initial_routed, selected, overflow_scored, threshold=threshold)
            else:
                selected, actions, routed = initial_selected, initial_actions, initial_routed
        else:
            selected, actions, contract_mode = [copy.deepcopy(row) for row in base[:2]], [], "not_evaluated"
            routed = [copy.deepcopy(row) for row in base]
        core_map[qid], top20_map[qid] = selected, routed
        actions_total += len(actions)
        offsets.append(offset)
        base_core_gold, new_core_gold = gold_keys(base[:2], record), gold_keys(selected, record)
        base_top_gold, new_top_gold = gold_keys(base, record), gold_keys(routed, record)
        core_gain += len(new_core_gold - base_core_gold)
        core_lost += len(base_core_gold - new_core_gold)
        top_gain += len(new_top_gold - base_top_gold)
        top_lost += len(base_top_gold - new_top_gold)
        payload = verification.get("verification_result") or {}
        llm_calls += int(bool(verification.get("triggered")))
        tokens += int(payload.get("tokens") or 0)
        overflow_ids = {row["card_id"] for row in overflow_scored}
        details.append({
            "query_id": qid,
            "triggered": bool(verification.get("triggered")),
            "contract_mode": contract_mode,
            "anchor_offset": offset,
            "overflow_candidates": len(overflow_scored),
            "actions": len(actions),
            "overflow_actions": sum(action.get("incoming_id") in overflow_ids for action in actions),
            "core_gain": len(new_core_gold - base_core_gold),
            "core_lost": len(base_core_gold - new_core_gold),
            "top20_gain": len(new_top_gold - base_top_gold),
            "top20_lost": len(base_top_gold - new_top_gold),
            "candidate_gain": len(gold_keys(candidate_map[qid], record) - gold_keys(baseline_pool(record), record)),
            "incoming_ids": " | ".join(str(action.get("incoming_id")) for action in actions),
            "incoming_titles": " | ".join(
                next((row["paper"].get("title") or "" for row in combined if row["card_id"] == action.get("incoming_id")), "")
                for action in actions
            ),
        })
    candidate = aggregate_metrics(records, candidate_map)
    top = aggregate_metrics(records, top20_map)
    core = aggregate_metrics(records, core_map)
    metrics = {
        "configuration": "COVR_anchor" if anchor_calibration else "COVR_no_anchor",
        "anchor_calibration": anchor_calibration,
        "overflow_margin": overflow_margin,
        "candidate_TP": candidate["TP"],
        "candidate_recall": candidate["recall"],
        "top20_TP": top["TP"],
        "top20_recall": top["recall"],
        "top20_gain": top_gain,
        "top20_lost": top_lost,
        "selected_TP": core["TP"],
        "selected_FP": core["FP"],
        "selected_FN": core["FN"],
        "precision": core["precision"],
        "recall": core["recall"],
        "f1": core["f1"],
        "mean_output_count": core["mean_output_count"],
        "core_gain": core_gain,
        "core_lost": core_lost,
        "replacement_actions": actions_total,
        "llm_calls": llm_calls,
        "tokens": tokens,
        "tokens_per_query": tokens / len(records),
        "mean_anchor_offset": sum(offsets) / len(offsets),
    }
    return metrics, {"candidate": candidate_map, "top20": top20_map, "core": core_map}, details


def select_profile(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda row: (
        row["f1"], row["selected_TP"], -row["selected_FP"],
        row["top20_gain"] - row["top20_lost"], -row["replacement_actions"],
        int(bool(row["anchor_calibration"])), -float(row["overflow_margin"]),
    ))


async def run_calibration() -> dict[str, Any]:
    ensure_dirs()
    records = records_for("calibration")
    dsfr, contract = load_inputs(records, "calibration")
    capture = await capture_dataset(records, dsfr, contract, "calibration40_v10")
    if not capture["manifest"]["replay_all_identical"]:
        raise RuntimeError("calibration replay mismatch")
    v9_profile = frozen_v9_profile()
    v9_metrics, _, _ = evaluate_v9_profile(
        records, dsfr, contract,
        threshold=float(v9_profile["threshold"]), margin=float(v9_profile["margin"]),
        max_replacements=int(v9_profile["max_replacements"]),
    )
    rows, products_by, details_by = [], {}, {}
    for anchor in ANCHOR_PROFILES:
        for overflow_margin in OVERFLOW_MARGINS:
            metrics, products, details = evaluate_variant(
                records, dsfr, contract, capture["results"],
                anchor_calibration=anchor, overflow_margin=overflow_margin,
            )
            rows.append(metrics)
            products_by[(anchor, overflow_margin)] = products
            details_by[(anchor, overflow_margin)] = details
    chosen = select_profile(rows)
    key = (bool(chosen["anchor_calibration"]), float(chosen["overflow_margin"]))
    write_csv(REPORT / "01_calibration_profiles.csv", [{"configuration": "ECRank_v9", **v9_metrics}, *rows])
    write_csv(REPORT / "02_calibration_query_actions.csv", details_by[key])
    gate = (
        chosen["selected_TP"] >= v9_metrics["selected_TP"]
        and chosen["selected_FP"] <= v9_metrics["selected_FP"] + 1
        and chosen["top20_lost"] <= v9_metrics["top20_lost"]
        and chosen["core_lost"] <= v9_metrics["core_lost"]
    )
    freeze = {
        "schema_version": 1,
        "source_commit": git_head(),
        "prompt_schema": "covr-v10-overflow-point-1",
        "v9_profile": v9_profile,
        "v9_baseline": v9_metrics,
        "chosen": chosen,
        "calibration_gate_passed": gate,
        "dev_evaluated": False,
        "anchor_profiles": list(ANCHOR_PROFILES),
        "overflow_margin_profiles": list(OVERFLOW_MARGINS),
    }
    (CONFIG / "calibration_freeze.json").write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return freeze


async def run_dev() -> dict[str, Any]:
    ensure_dirs()
    freeze = json.loads((CONFIG / "calibration_freeze.json").read_text(encoding="utf-8"))
    if not freeze["calibration_gate_passed"]:
        state = {"status": "rejected_on_calibration", "dev_live_runs": 0, "holdout_used": False}
        write_csv(REPORT / "03_dev20_replay.csv", [{"configuration": "SKIPPED", "reason": "calibration gate not met"}])
        (REPORT / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        return state
    records = records_for("dev")
    dsfr, contract = load_inputs(records, "dev")
    capture = await capture_dataset(records, dsfr, contract, "dev20_v10")
    if not capture["manifest"]["replay_all_identical"]:
        raise RuntimeError("dev replay mismatch")
    anchor = bool(freeze["chosen"]["anchor_calibration"])
    overflow_margin = float(freeze["chosen"]["overflow_margin"])
    metrics, products, details = evaluate_variant(
        records, dsfr, contract, capture["results"],
        anchor_calibration=anchor, overflow_margin=overflow_margin,
    )
    base_k2 = baseline_metrics(records)
    base_v6 = baseline_configuration(records)
    v9_profile = frozen_v9_profile()
    v9_metrics, v9_products, _ = evaluate_v9_profile(
        records, dsfr, contract,
        threshold=float(v9_profile["threshold"]), margin=float(v9_profile["margin"]),
        max_replacements=int(v9_profile["max_replacements"]),
    )
    v9_v6 = aggregate_metrics(records, v6_selection(records, v9_products["top20"]))
    covr_v6 = aggregate_metrics(records, v6_selection(records, products["top20"]))
    rows = [
        base_k2,
        {"configuration": "B0_v6", **base_v6},
        {"configuration": "ECRank_v9_K2", **v9_metrics},
        {"configuration": "ECRank_v9_plus_v6", **v9_v6},
        metrics,
        {"configuration": "COVR_plus_v6", **covr_v6},
    ]
    write_csv(REPORT / "03_dev20_replay.csv", rows)
    write_csv(REPORT / "04_dev_query_actions.csv", details)
    accepted_k2 = (
        metrics["selected_TP"] >= base_v6["selected_TP"]
        and metrics["selected_FP"] <= base_v6["selected_FP"]
        and metrics["f1"] >= base_v6["f1"] + .005
        and metrics["recall"] >= base_v6["recall"]
    )
    accepted_v6 = (
        covr_v6["TP"] >= base_v6["selected_TP"] + 1
        and covr_v6["FP"] <= base_v6["selected_FP"] + 2
        and covr_v6["f1"] >= base_v6["f1"] + .01
        and covr_v6["recall"] >= base_v6["recall"]
    )
    state = {
        "schema_version": 1,
        "status": "development_candidate_v10" if accepted_k2 or accepted_v6 else "rejected_on_dev",
        "branch": "coverage-adaptive-contract-v10",
        "source_commit": freeze["source_commit"],
        "holdout_used": False,
        "dev_live_runs": 1,
        "verification_replay_identical": capture["manifest"]["replay_all_identical"],
        "baseline_k2": base_k2,
        "baseline_v6": base_v6,
        "ecrank_v9_k2": v9_metrics,
        "ecrank_v9_plus_v6": v9_v6,
        "covr_k2": metrics,
        "covr_plus_v6": covr_v6,
        "accepted_covr_k2": accepted_k2,
        "accepted_covr_plus_v6": accepted_v6,
    }
    (REPORT / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    if accepted_k2 or accepted_v6:
        (CONFIG / "development_candidate_v10.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return state


def write_method_report() -> dict[str, Any]:
    ensure_dirs()
    text = """# COVR v10：覆盖自适应契约验证

## 问题

ECRank v9 将 LLM 列表限制在 4 篇 incumbent 与 12 篇 newcomer。dev20 中 DSFR 唯一新增的 gold（PaLM）在 residual source view 中仍处于前列，却因固定 shortlist 截断而没有接受契约判断。扩大整张 listwise 列表会线性增加 token、位置偏差与噪声，因此 v10 不直接扩大列表。

## 方法

Coverage-Adaptive Contract Verification（COVR）包含四个步骤：

1. **残余覆盖检测**：仅当第一阶段没有安全替换动作、也没有达到阈值的 evidence-complete newcomer 时，检测 source-ranked residual mass。
2. **预算化 overflow**：从第一阶段未观察的 source view 中取 8 篇，避免再次选取 TRACE/facet shortlist 已覆盖的候选。
3. **冻结契约 point verification**：复用 v9 生成的 query contract，不再调用一次 query planning；把 8 篇 overflow 与 2 篇 Top2 anchor 放入一次批量验证调用。
4. **跨阶段 anchor alignment**：用两篇 anchor 在 listwise 与 point 两阶段的分数差中位数校正 overflow 分数，偏移截断在 ±5；只有满足全部条件、关系支持与替换 margin 的候选才能进入 Top2/Top20。

该机制的创新点不在于“多调一次 LLM”，而在于将有限 reranking 预算建模成可观测覆盖过程：第一阶段负责高概率候选的比较式排序，第二阶段只在未解决状态下审计互补检索视图，并通过共享 anchor 对齐两个评分空间。

## 可证伪假设

- H1：overflow verification 能找回固定 shortlist 截断的有效论文，而不损失已有 Top20 gold。
- H2：共享 anchor 比直接拼接两个提示词的原始分数更少产生错误替换。
- H3：自适应触发的调用数低于对所有查询统一扩大 listwise reranking。

## 消融

- ECRank v9（无 overflow）；
- COVR no-anchor（overflow 原始 point score）；
- COVR anchor（±5 跨阶段对齐）；
- 查询级 triggered / skipped 成本与增益。
"""
    (REPORT / "00_method.md").write_text(text, encoding="utf-8")
    return {"written": str(REPORT / "00_method.md")}


def final_report() -> dict[str, Any]:
    freeze = json.loads((CONFIG / "calibration_freeze.json").read_text(encoding="utf-8"))
    state_path = REPORT / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"status": "calibration_only"}
    v9, chosen = freeze["v9_baseline"], freeze["chosen"]
    lines = [
        "# Coverage-Adaptive Contract Verification v10 最终报告", "",
        "## calibration40", "",
        "| 配置 | TP/FP/FN | P | R | F1 | Top20 gain/lost | 调用 | tokens/query |", "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| ECRank v9 | {v9['selected_TP']}/{v9['selected_FP']}/{v9['selected_FN']} | {v9['precision']:.4f} | {v9['recall']:.4f} | {v9['f1']:.4f} | {v9['top20_gain']}/{v9['top20_lost']} | 0 | 0 |",
        f"| COVR | {chosen['selected_TP']}/{chosen['selected_FP']}/{chosen['selected_FN']} | {chosen['precision']:.4f} | {chosen['recall']:.4f} | {chosen['f1']:.4f} | {chosen['top20_gain']}/{chosen['top20_lost']} | {chosen['llm_calls']} | {chosen['tokens_per_query']:.1f} |",
        "", f"- 选择 anchor calibration：{chosen['anchor_calibration']}；overflow margin：{chosen['overflow_margin']}。", f"- calibration gate：{freeze['calibration_gate_passed']}。",
        "", "## dev20", "", f"- 状态：{state.get('status')}。",
    ]
    if state.get("covr_k2"):
        b, k, v = state["baseline_v6"], state["covr_k2"], state["covr_plus_v6"]
        lines += [
            f"- v6：P/R/F1={b['precision']:.4f}/{b['recall']:.4f}/{b['f1']:.4f}，TP/FP={b['selected_TP']}/{b['selected_FP']}。",
            f"- COVR K2：P/R/F1={k['precision']:.4f}/{k['recall']:.4f}/{k['f1']:.4f}，TP/FP={k['selected_TP']}/{k['selected_FP']}。",
            f"- COVR+v6：P/R/F1={v['precision']:.4f}/{v['recall']:.4f}/{v['f1']:.4f}，TP/FP={v['TP']}/{v['FP']}。",
            f"- overflow 调用={k['llm_calls']}，tokens/query={k['tokens_per_query']:.1f}，Top20 gain/lost={k['top20_gain']}/{k['top20_lost']}。",
        ]
    lines += [
        "", "## 结论", "",
        "- COVR 是否进入后续 holdout，以 dev20 的冻结回放结果和 state.json 为准。",
        "- 所有动作由查询、论文元数据、冻结 contract 与冻结 LLM 输出产生；运行时不读取 gold、query ID 或答案基数。",
    ]
    (REPORT / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"freeze": freeze, "state": state}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("method", "calibration", "dev", "report"))
    args = parser.parse_args()
    if args.phase == "method":
        result = write_method_report()
    elif args.phase == "calibration":
        result = asyncio.run(run_calibration())
    elif args.phase == "dev":
        result = asyncio.run(run_dev())
    else:
        result = final_report()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
