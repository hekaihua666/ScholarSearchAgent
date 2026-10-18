"""Frozen two-stage contract verification for the SCGQ citation frontier."""

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
from backend.selection.citation_frontier_verifier_v16 import (
    build_title_cards,
    cards_from_title_screen,
    request_title_screen,
    should_verify_frontier,
)
from backend.selection.coverage_adaptive_contract_v10 import (
    request_overflow_verification,
    score_verification_cards,
)
from backend.selection.evidence_contract_ranker_v9 import (
    ContractCard,
    route_contract_core,
    route_top20,
    score_cards,
)
from backend.trace_rank import fuse_candidates
from scripts.audit_evaluation import source_cutoff
from scripts.run_controlled_retrieval_v8 import (
    aggregate_metrics,
    baseline_configuration,
    baseline_top20_map,
    gold_keys,
    records_for,
    v6_selection,
)
from scripts.run_coverage_adaptive_contract_v10 import (
    evaluate_variant as evaluate_v10_variant,
    load_inputs,
    restore_verification_cards,
)
from scripts.run_evidence_contract_rank_v9 import candidate_pool, restore_cards
from scripts.run_frozen_official_benchmark import json_hash
from scripts.run_short_sprint_v5 import write_csv


REPORT = ROOT / "reports/citation_frontier_verifier_v16"
CONFIG = ROOT / "configs/citation_frontier_verifier_v16"
CAPTURE = ROOT / "reports/frozen_eval/citation_frontier_verifier_v16"
V9_ROOTS = {
    "calibration": ROOT / "reports/frozen_eval/evidence_contract_rank_v9/calibration40_v9/queries",
    "dev": ROOT / "reports/frozen_eval/evidence_contract_rank_v9/dev20_v9_contract/queries",
}
V10_ROOTS = {
    "calibration": ROOT / "reports/frozen_eval/coverage_adaptive_contract_v10/calibration40_v10/queries",
    "dev": ROOT / "reports/frozen_eval/coverage_adaptive_contract_v10/dev20_v10/queries",
}
V15_ROOTS = {
    "calibration": ROOT / "reports/frozen_eval/seed_conditioned_citation_v15/calibration40_v15/queries",
    "dev": ROOT / "reports/frozen_eval/seed_conditioned_citation_v15/dev20_v15/queries",
}
V11_ROOTS = {
    "calibration": ROOT / "reports/frozen_eval/query_conditioned_citation_v11/calibration40_v11/queries",
    "dev": ROOT / "reports/frozen_eval/query_conditioned_citation_v11/dev20_v11/queries",
}
THRESHOLD = 95.0
MARGIN = 2.0
MAX_REPLACEMENTS = 2


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_dirs() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    CONFIG.mkdir(parents=True, exist_ok=True)
    CAPTURE.mkdir(parents=True, exist_ok=True)


def load_json_results(records: Sequence[dict[str, Any]], roots: dict[str, Path], split: str) -> dict[str, dict[str, Any]]:
    return {
        record["qid"]: json.loads((roots[split] / record["qid"] / "live_result.json").read_text(encoding="utf-8"))
        for record in records
    }


def frontier_for(qid: str, v15: dict[str, Any], v11: dict[str, Any]) -> list[dict[str, Any]]:
    del qid
    return fuse_candidates([*(v15.get("candidates") or []), *(v11.get("candidates") or [])])


def v10_products(
    records: Sequence[dict[str, Any]], dsfr: dict[str, dict[str, Any]], contracts: dict[str, dict[str, Any]],
    verification: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, dict[str, Any]]]:
    _, products, details = evaluate_v10_variant(
        records, dsfr, contracts, verification, anchor_calibration=True, overflow_margin=2.0
    )
    return products, {row["query_id"]: row for row in details}


async def _safe_title_screen(query: str, contract: dict[str, Any], cards: Sequence[ContractCard]) -> dict[str, Any]:
    try:
        return await request_title_screen(query, contract, cards)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {str(exc)[:300]}", "parsed": {}, "tokens": 0}


async def _safe_evidence_verify(query: str, contract: dict[str, Any], cards: Sequence[ContractCard]) -> dict[str, Any]:
    try:
        return await request_overflow_verification(query, contract, cards)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {str(exc)[:300]}", "parsed": {}, "tokens": 0}


async def execute_funnel(
    record: dict[str, Any],
    dsfr: dict[str, Any],
    contract_result: dict[str, Any],
    current_core: Sequence[dict[str, Any]],
    prior_actions: int,
    frontier: Sequence[dict[str, Any]],
    dataset_name: str,
    mode: str,
) -> tuple[dict[str, Any], EvaluationSession]:
    qid = record["qid"]
    path = CAPTURE / dataset_name / "queries" / qid / "snapshot.json"
    session = EvaluationSession.create(dataset_name, qid, source_cutoff(record), mode, path)
    parsed_contract = (contract_result.get("contract_result") or {}).get("parsed") or {}
    contract = parsed_contract.get("contract") or {}
    existing = candidate_pool(record, dsfr)
    title_cards = build_title_cards(record["question"], frontier, existing)
    provisional = [
        ContractCard("A01", "anchor", 1, current_core[0]),
        ContractCard("A02", "anchor", 2, current_core[1]),
        *title_cards[:1],
    ] if len(current_core) >= 2 and title_cards else []
    fire = should_verify_frontier(provisional, prior_actions, contract) if provisional else False
    started = time.perf_counter()
    title_payload: dict[str, Any]
    evidence_payload: dict[str, Any]
    evidence_cards: list[ContractCard] = []
    if fire:
        title_signature = hashlib.sha256(json.dumps([
            {"id": card.card_id, "title": card.paper.get("title"), "openalex": card.paper.get("openalex_id")}
            for card in title_cards
        ], ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        with evaluation_session(session):
            title_payload = await frozen_call(
                "cfv16_title_funnel",
                {"schema": "cfv16-title-1", "query": record["question"],
                 "contract": contract, "title_signature": title_signature, "card_count": len(title_cards)},
                lambda: _safe_title_screen(record["question"], contract, title_cards),
            )
        evidence_cards = cards_from_title_screen(
            record["question"], current_core, title_cards, title_payload.get("parsed") or {}
        )
        if any(card.origin == "overflow" for card in evidence_cards):
            evidence_signature = hashlib.sha256(json.dumps([
                {"id": card.card_id, "title": card.paper.get("title"), "openalex": card.paper.get("openalex_id")}
                for card in evidence_cards
            ], ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
            with evaluation_session(session):
                evidence_payload = await frozen_call(
                    "cfv16_evidence_verify",
                    {"schema": "cfv16-evidence-1", "query": record["question"],
                     "contract": contract, "evidence_signature": evidence_signature,
                     "card_count": len(evidence_cards)},
                    lambda: _safe_evidence_verify(record["question"], contract, evidence_cards),
                )
        else:
            evidence_payload = {"skipped": True, "reason": "title_funnel_empty", "parsed": {}, "tokens": 0}
    else:
        title_payload = {"skipped": True, "reason": "prior_action_or_no_contract", "parsed": {}, "tokens": 0}
        evidence_payload = {"skipped": True, "reason": "title_funnel_not_run", "parsed": {}, "tokens": 0}
    result = {
        "query_id": qid,
        "triggered": fire,
        "prior_actions": prior_actions,
        "elapsed_seconds": time.perf_counter() - started,
        "title_cards": [{"card_id": card.card_id, "paper": card.paper} for card in title_cards],
        "title_result": title_payload,
        "evidence_cards": [{"card_id": card.card_id, "origin": card.origin,
                            "original_rank": card.original_rank, "paper": card.paper} for card in evidence_cards],
        "evidence_result": evidence_payload,
    }
    if mode == "live":
        session.save({"source_commit": git_head(), "schema": "cfv16-funnel-1"})
        path.parent.mkdir(parents=True, exist_ok=True)
        (path.parent / "live_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return result, session


def projection(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "triggered": result.get("triggered"),
        "title_cards": [(row["card_id"], row["paper"].get("title")) for row in result.get("title_cards") or []],
        "title_parsed": (result.get("title_result") or {}).get("parsed"),
        "evidence_cards": [(row["card_id"], row["origin"], row["paper"].get("title")) for row in result.get("evidence_cards") or []],
        "evidence_parsed": (result.get("evidence_result") or {}).get("parsed"),
    }


async def capture_dataset(
    records: Sequence[dict[str, Any]], dsfr: dict[str, dict[str, Any]], contracts: dict[str, dict[str, Any]],
    current_products: dict[str, dict[str, list[dict[str, Any]]]], prior_details: dict[str, dict[str, Any]],
    v15: dict[str, dict[str, Any]], v11: dict[str, dict[str, Any]], dataset_name: str,
) -> dict[str, Any]:
    ensure_dirs(); results = {}; manifest = []
    for index, record in enumerate(records, 1):
        qid = record["qid"]; root = CAPTURE / dataset_name / "queries" / qid
        live_path, snapshot_path = root / "live_result.json", root / "snapshot.json"
        frontier = frontier_for(qid, v15[qid], v11[qid])
        if live_path.exists() and snapshot_path.exists():
            live = json.loads(live_path.read_text(encoding="utf-8"))
            live_session = EvaluationSession.create(dataset_name, qid, source_cutoff(record), "replay", snapshot_path)
        else:
            live, live_session = await execute_funnel(
                record, dsfr[qid], contracts[qid], current_products["core"][qid],
                int(prior_details[qid].get("actions") or 0), frontier, dataset_name, "live",
            )
        replay, _ = await execute_funnel(
            record, dsfr[qid], contracts[qid], current_products["core"][qid],
            int(prior_details[qid].get("actions") or 0), frontier, dataset_name, "replay",
        )
        identical = json_hash(projection(live)) == json_hash(projection(replay)); results[qid] = live
        title_tokens = int((live.get("title_result") or {}).get("tokens") or 0)
        evidence_tokens = int((live.get("evidence_result") or {}).get("tokens") or 0)
        manifest.append({
            "query_id": qid, "triggered": live.get("triggered"),
            "title_count": len(live.get("title_cards") or []),
            "evidence_count": len(live.get("evidence_cards") or []),
            "llm_calls": len(live_session.calls), "tokens": title_tokens + evidence_tokens,
            "elapsed_seconds": live.get("elapsed_seconds"), "replay_identical": identical,
            "snapshot_sha256": sha(snapshot_path),
        })
        print(f"[{index}/{len(records)}] {qid} trigger={live.get('triggered')} titles={len(live.get('title_cards') or [])} evidence={len(live.get('evidence_cards') or [])} replay={'OK' if identical else 'DIFF'}", flush=True)
    payload = {"dataset": dataset_name, "query_count": len(records), "source_commit": git_head(),
               "replay_all_identical": all(row["replay_identical"] for row in manifest), "queries": manifest}
    (CAPTURE / dataset_name / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"manifest": payload, "results": results}


def evaluate(
    records: Sequence[dict[str, Any]], dsfr: dict[str, dict[str, Any]], contracts: dict[str, dict[str, Any]],
    current_products: dict[str, dict[str, list[dict[str, Any]]]], funnel: dict[str, dict[str, Any]],
    v15: dict[str, dict[str, Any]], v11: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, list[dict[str, Any]]]], list[dict[str, Any]]]:
    core_map = {}; top_map = {}; candidate_map = {}; details = []
    gain = lost = top_gain = top_lost = actions_total = tokens = calls = 0
    for record in records:
        qid = record["qid"]; base_core = current_products["core"][qid]; base_top = current_products["top20"][qid]
        frontier = frontier_for(qid, v15[qid], v11[qid])
        candidate_map[qid] = fuse_candidates([*candidate_pool(record, dsfr[qid]), *frontier])
        result = funnel[qid]
        evidence_cards = [
            ContractCard(row["card_id"], row["origin"], row.get("original_rank"), row["paper"])
            for row in result.get("evidence_cards") or []
        ]
        evidence_parsed = (result.get("evidence_result") or {}).get("parsed") or {}
        initial_cards = restore_cards(contracts[qid]); contract_parsed = (contracts[qid].get("contract_result") or {}).get("parsed") or {}
        initial_scored = score_cards(initial_cards, contract_parsed) if initial_cards and contract_parsed else []
        overflow_scored, offset = score_verification_cards(
            evidence_cards, evidence_parsed, initial_scored, calibrate_with_anchors=True
        ) if evidence_cards and evidence_parsed else ([], 0.0)
        if overflow_scored:
            incumbents = [row for row in initial_scored if row["origin"] == "baseline" and int(row.get("original_rank") or 99) <= 2]
            selected, actions, mode = route_contract_core(
                record["question"], base_top, [*incumbents, *overflow_scored], contract_parsed,
                threshold=THRESHOLD, margin=MARGIN, max_replacements=MAX_REPLACEMENTS,
            )
            routed = route_top20(base_top, selected, overflow_scored, threshold=THRESHOLD)
        else:
            selected, routed, actions, mode = [copy.deepcopy(row) for row in base_core], [copy.deepcopy(row) for row in base_top], [], "not_evaluated"
        core_map[qid], top_map[qid] = selected, routed; actions_total += len(actions)
        old_core, new_core = gold_keys(base_core, record), gold_keys(selected, record)
        old_top, new_top = gold_keys(base_top, record), gold_keys(routed, record)
        gain += len(new_core - old_core); lost += len(old_core - new_core)
        top_gain += len(new_top - old_top); top_lost += len(old_top - new_top)
        title_selected = {row.get("id") for row in ((result.get("title_result") or {}).get("parsed") or {}).get("selected") or []}
        title_gold = 0
        for row in result.get("title_cards") or []:
            if row["card_id"] in title_selected and gold_keys([row["paper"]], record): title_gold += 1
        t = int((result.get("title_result") or {}).get("tokens") or 0) + int((result.get("evidence_result") or {}).get("tokens") or 0)
        tokens += t; calls += int(bool(result.get("triggered"))) * 2
        details.append({
            "query_id": qid, "triggered": result.get("triggered"), "title_gold_selected": title_gold,
            "anchor_offset": offset, "actions": len(actions), "core_gain": len(new_core-old_core),
            "core_lost": len(old_core-new_core), "top20_gain": len(new_top-old_top), "top20_lost": len(old_top-new_top),
            "incoming_titles": " | ".join(next((row["paper"].get("title") or "" for row in overflow_scored if row["card_id"] == action.get("incoming_id")), "") for action in actions),
        })
    candidate = aggregate_metrics(records, candidate_map); core = aggregate_metrics(records, core_map); top = aggregate_metrics(records, top_map)
    metrics = {
        "configuration": "CFV16", "candidate_TP": candidate["TP"], "candidate_recall": candidate["recall"],
        "top20_TP": top["TP"], "top20_recall": top["recall"], "top20_gain": top_gain, "top20_lost": top_lost,
        "selected_TP": core["TP"], "selected_FP": core["FP"], "selected_FN": core["FN"],
        "precision": core["precision"], "recall": core["recall"], "f1": core["f1"],
        "mean_output_count": core["mean_output_count"], "core_gain": gain, "core_lost": lost,
        "replacement_actions": actions_total, "llm_calls": calls, "tokens": tokens,
        "tokens_per_query": tokens / len(records),
    }
    return metrics, {"candidate": candidate_map, "core": core_map, "top20": top_map}, details


async def run_split(split: str) -> dict[str, Any]:
    ensure_dirs(); records = records_for(split); dsfr, contracts = load_inputs(records, split)
    v10 = load_json_results(records, V10_ROOTS, split); v15 = load_json_results(records, V15_ROOTS, split); v11 = load_json_results(records, V11_ROOTS, split)
    current_products, prior_details = v10_products(records, dsfr, contracts, v10)
    dataset_name = "calibration40_v16" if split == "calibration" else "dev20_v16"
    capture = await capture_dataset(records, dsfr, contracts, current_products, prior_details, v15, v11, dataset_name)
    if not capture["manifest"]["replay_all_identical"]: raise RuntimeError(f"{split} replay mismatch")
    metrics, products, details = evaluate(records, dsfr, contracts, current_products, capture["results"], v15, v11)
    current_core = aggregate_metrics(records, current_products["core"]); current_top = aggregate_metrics(records, current_products["top20"])
    base = {"configuration": "COVR_v10", "selected_TP": current_core["TP"], "selected_FP": current_core["FP"], "selected_FN": current_core["FN"],
            "precision": current_core["precision"], "recall": current_core["recall"], "f1": current_core["f1"], "mean_output_count": current_core["mean_output_count"],
            "top20_TP": current_top["TP"], "top20_recall": current_top["recall"]}
    if split == "calibration":
        write_csv(REPORT / "01_calibration_metrics.csv", [base, metrics]); write_csv(REPORT / "02_calibration_actions.csv", details)
        gate = metrics["selected_TP"] >= base["selected_TP"] and metrics["selected_FP"] <= base["selected_FP"] + 1 and metrics["top20_lost"] == 0 and metrics["core_lost"] == 0
        freeze = {"schema_version": 1, "source_commit": git_head(), "schema": "cfv16-funnel-1",
                  "threshold": THRESHOLD, "margin": MARGIN, "max_replacements": MAX_REPLACEMENTS,
                  "baseline": base, "chosen": metrics, "calibration_gate_passed": gate, "dev_evaluated": False}
        (CONFIG / "calibration_freeze.json").write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")
        return freeze
    freeze = json.loads((CONFIG / "calibration_freeze.json").read_text(encoding="utf-8"))
    if not freeze["calibration_gate_passed"]:
        state = {"status": "rejected_on_calibration", "dev_live_runs": 0}; (REPORT / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"); return state
    base_v6 = baseline_configuration(records)
    current_v6 = aggregate_metrics(records, v6_selection(records, current_products["top20"]))
    routed_v6 = aggregate_metrics(records, v6_selection(records, products["top20"]))
    rows = [base, {"configuration": "B0_v6", **base_v6}, {"configuration": "COVR_v10_plus_v6", **current_v6}, metrics, {"configuration": "CFV16_plus_v6", **routed_v6}]
    write_csv(REPORT / "03_dev20_metrics.csv", rows); write_csv(REPORT / "04_dev20_actions.csv", details)
    accepted = routed_v6["TP"] >= current_v6["TP"] + 1 and routed_v6["FP"] <= current_v6["FP"] + 2 and routed_v6["f1"] >= current_v6["f1"] + .01
    state = {"schema_version": 1, "status": "development_candidate_v16" if accepted else "rejected_on_dev",
             "branch": "citation-frontier-verifier-v16", "source_commit": freeze["source_commit"], "dev_live_runs": 1,
             "replay_identical": capture["manifest"]["replay_all_identical"], "baseline_covr": base, "baseline_v6": base_v6,
             "baseline_covr_plus_v6": current_v6,
             "cfv16": metrics, "cfv16_plus_v6": routed_v6, "accepted": accepted}
    (REPORT / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return state


def write_method() -> dict[str, Any]:
    ensure_dirs()
    text = """# CFV16: Citation Frontier Coarse-to-Fine Evidence Funnel

SCGQ v15 raises dev20 candidate TP from 23 to 27 but produces hundreds of frontier papers per query. CFV16 tests a fixed-budget two-stage evidence funnel that converts the additional recall into safely routable candidates:

1. Multi-view prefiltering retains at most 100 title cards using TRACE, citation seed count, and source rank.
2. A title-only funnel selects at most 16 papers that may satisfy the query contract.
3. A point verifier reads only those abstracts plus two incumbent anchors and emits criterion-level and relational evidence.
4. Routing reuses the frozen COVR v10 anchor alignment, threshold 95, and replacement margin 2.
5. Queries already modified by COVR v10 do not trigger CFV16, avoiding stacked rescue risk.

The method separates graph recall, low-cost semantic filtering, and high-cost evidence verification while recording funnel recall, Top20 gain/loss, and token cost.
"""
    (REPORT / "00_method.md").write_text(text, encoding="utf-8")
    return {"written": str(REPORT / "00_method.md")}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("phase", choices=("method", "calibration", "dev")); args = parser.parse_args()
    result = write_method() if args.phase == "method" else asyncio.run(run_split("calibration" if args.phase == "calibration" else "dev"))
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__": main()
