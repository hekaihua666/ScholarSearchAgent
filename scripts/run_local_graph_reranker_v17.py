"""Frozen calibration and one-shot dev evaluation for LGR v17."""

from __future__ import annotations

import argparse
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

# Torch must initialize before the legacy selector stack imports sklearn's
# OpenMP runtime on Windows; otherwise c10.dll can fail with WinError 1114.
from sentence_transformers import CrossEncoder as _CrossEncoderBootstrap  # noqa: F401,E402

from backend.paper_identity import match_identity
from backend.selection.coverage_adaptive_contract_v10 import identity_key
from backend.selection.local_graph_reranker_v17 import (
    make_selector,
    route_final_set,
    route_frontier_top20,
    score_query_pool,
    select_multiview_frontier_pool,
)
from backend.trace_rank import fuse_candidates
from scripts.run_citation_frontier_verifier_v16 import (
    V10_ROOTS,
    V11_ROOTS,
    V15_ROOTS,
    frontier_for,
    load_json_results,
    v10_products,
)
from scripts.run_controlled_retrieval_v8 import (
    aggregate_metrics,
    gold_keys,
    records_for,
    v6_selection,
)
from scripts.run_coverage_adaptive_contract_v10 import load_inputs
from scripts.run_short_sprint_v5 import write_csv


REPORT = ROOT / "reports/local_graph_reranker_v17"
CONFIG = ROOT / "configs/local_graph_reranker_v17"
CAPTURE = ROOT / "reports/frozen_eval/local_graph_reranker_v17"
CACHE = ROOT / "cache/v17_cross_encoder_scores_v3.json"
SCHEMA = "lgr-v17-local-cross-graph-3"

WEIGHT_PROFILES = {
    "cross_only": (1.0, 0.0, 0.0),
    "cross_graph": (0.80, 0.20, 0.0),
    "cross_graph_trace": (0.70, 0.20, 0.10),
    "graph_balanced": (0.60, 0.30, 0.10),
}
TOP_MARGINS = (0.00, 0.03, 0.06, 0.10, 0.15)
TOP_PROMOTIONS = (1, 2)
FINAL_MARGINS = (0.03, 0.06, 0.10, 0.15, 0.20)
FINAL_CROSS_FLOORS = (0.90, 0.95, 0.975, 1.10)
FINAL_GRAPH_FLOORS = (0.0, 0.50, 0.75)


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def ensure_dirs() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    CONFIG.mkdir(parents=True, exist_ok=True)
    CAPTURE.mkdir(parents=True, exist_ok=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)


def _hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _contract(contract_result: dict[str, Any]) -> dict[str, Any]:
    parsed = (contract_result.get("contract_result") or {}).get("parsed") or {}
    return parsed.get("contract") or {}


def _paper_signature(paper: dict[str, Any]) -> tuple[str, str, str]:
    return (
        identity_key(paper),
        str(paper.get("title") or ""),
        str(paper.get("abstract") or "")[:700],
    )


def _load_split(split: str) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, list[dict[str, Any]]]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    records = records_for(split)
    dsfr, contracts = load_inputs(records, split)
    v10 = load_json_results(records, V10_ROOTS, split)
    v15 = load_json_results(records, V15_ROOTS, split)
    v11 = load_json_results(records, V11_ROOTS, split)
    products, _ = v10_products(records, dsfr, contracts, v10)
    return records, dsfr, contracts, products, v15, v11


def score_split(
    split: str,
    records: Sequence[dict[str, Any]],
    contracts: dict[str, dict[str, Any]],
    current_products: dict[str, dict[str, list[dict[str, Any]]]],
    v15: dict[str, dict[str, Any]],
    v11: dict[str, dict[str, Any]],
    selector: Any,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    scored: dict[str, dict[str, Any]] = {}
    manifest_rows: list[dict[str, Any]] = []
    split_root = CAPTURE / ("calibration40_v17" if split == "calibration" else "dev20_v17")
    started_all = time.perf_counter()
    for index, record in enumerate(records, 1):
        qid = record["qid"]
        incumbents = current_products["top20"][qid]
        frontier_all = frontier_for(qid, v15[qid], v11[qid])
        frontier = select_multiview_frontier_pool(frontier_all, incumbents, limit=120)
        papers = [*incumbents, *frontier]
        contract = _contract(contracts[qid])
        signature = _hash({
            "schema": SCHEMA,
            "question": record["question"],
            "contract": contract,
            "incumbent_count": len(incumbents),
            "papers": [_paper_signature(paper) for paper in papers],
        })
        path = split_root / "queries" / qid / "scores.json"
        payload: dict[str, Any] | None = None
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("signature") == signature and cached.get("schema") == SCHEMA:
                payload = cached
        if payload is None:
            started = time.perf_counter()
            rows, runtime = score_query_pool(record["question"], contract, papers, selector)
            payload = {
                "schema": SCHEMA,
                "signature": signature,
                "query_id": qid,
                "incumbent_count": len(incumbents),
                "frontier_count": len(frontier),
                "rows": rows,
                "runtime": runtime,
                "elapsed_seconds": time.perf_counter() - started,
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        rows = payload["rows"]
        incumbent_count = int(payload["incumbent_count"])
        scored[qid] = {
            "incumbents": rows[:incumbent_count],
            "frontier": rows[incumbent_count:],
            "all_frontier": frontier_all,
            "runtime": payload.get("runtime") or {},
        }
        manifest_rows.append({
            "query_id": qid,
            "signature": signature,
            "incumbent_count": incumbent_count,
            "frontier_count": len(rows) - incumbent_count,
            "runtime": (payload.get("runtime") or {}).get("runtime"),
            "cache_hits": (payload.get("runtime") or {}).get("cache_hits"),
            "elapsed_seconds": payload.get("elapsed_seconds"),
        })
        print(
            f"[{index}/{len(records)}] {qid} incumbents={incumbent_count} "
            f"frontier={len(rows)-incumbent_count} runtime={manifest_rows[-1]['runtime']}",
            flush=True,
        )
    manifest = {
        "schema": SCHEMA,
        "split": split,
        "query_count": len(records),
        "source_commit": git_head(),
        "wall_seconds": time.perf_counter() - started_all,
        "queries": manifest_rows,
    }
    split_root.mkdir(parents=True, exist_ok=True)
    (split_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return scored, manifest


def _same(paper: dict[str, Any], other: dict[str, Any]) -> bool:
    return match_identity(paper, other) or match_identity(other, paper)


def _candidate_map(
    records: Sequence[dict[str, Any]],
    current_products: dict[str, dict[str, list[dict[str, Any]]]],
    scored: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        record["qid"]: fuse_candidates([
            *current_products["candidate"][record["qid"]],
            *scored[record["qid"]]["all_frontier"],
        ])
        for record in records
    }


def evaluate_top_profile(
    records: Sequence[dict[str, Any]],
    current_products: dict[str, dict[str, list[dict[str, Any]]]],
    scored: dict[str, dict[str, Any]],
    *,
    profile_name: str,
    weights: Sequence[float],
    margin: float,
    max_promotions: int,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    routed: dict[str, list[dict[str, Any]]] = {}
    details: list[dict[str, Any]] = []
    gain = lost = actions_total = 0
    for record in records:
        qid = record["qid"]
        base = scored[qid]["incumbents"]
        result, actions = route_frontier_top20(
            base,
            scored[qid]["frontier"],
            weights=weights,
            margin=margin,
            max_promotions=max_promotions,
            protect_head=2,
        )
        routed[qid] = result
        old_gold, new_gold = gold_keys(base, record), gold_keys(result, record)
        qgain, qlost = len(new_gold - old_gold), len(old_gold - new_gold)
        gain += qgain; lost += qlost; actions_total += len(actions)
        details.append({
            "query_id": qid,
            "actions": len(actions),
            "top20_gain": qgain,
            "top20_lost": qlost,
            "incoming_titles": " | ".join(str(action.get("incoming_title") or "") for action in actions),
            "outgoing_titles": " | ".join(str(action.get("outgoing_title") or "") for action in actions),
        })
    top = aggregate_metrics(records, routed)
    metrics = {
        "configuration": profile_name,
        "weights": "/".join(str(value) for value in weights),
        "margin": margin,
        "max_promotions": max_promotions,
        "top20_TP": top["TP"],
        "top20_FP": top["FP"],
        "top20_recall": top["recall"],
        "top20_gain": gain,
        "top20_lost": lost,
        "actions": actions_total,
    }
    return metrics, routed, details


def evaluate_final_profile(
    records: Sequence[dict[str, Any]],
    top20: dict[str, list[dict[str, Any]]],
    base: dict[str, list[dict[str, Any]]],
    *,
    weights: Sequence[float],
    margin: float,
    cross_floor: float,
    graph_floor: float,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    routed: dict[str, list[dict[str, Any]]] = {}
    details: list[dict[str, Any]] = []
    gain = lost = actions_total = 0
    for record in records:
        qid = record["qid"]
        result, actions = route_final_set(
            base[qid], top20[qid], weights=weights, margin=margin,
            cross_floor=cross_floor, graph_floor=graph_floor,
        )
        routed[qid] = result
        old_gold, new_gold = gold_keys(base[qid], record), gold_keys(result, record)
        qgain, qlost = len(new_gold-old_gold), len(old_gold-new_gold)
        gain += qgain; lost += qlost; actions_total += len(actions)
        details.append({
            "query_id": qid,
            "actions": len(actions),
            "final_gain": qgain,
            "final_lost": qlost,
            "incoming_title": (actions[0].get("incoming_title") if actions else ""),
            "outgoing_title": (actions[0].get("outgoing_title") if actions else ""),
        })
    metrics = aggregate_metrics(records, routed)
    return {
        "configuration": "final_residual",
        "weights": "/".join(str(value) for value in weights),
        "margin": margin,
        "cross_floor": cross_floor,
        "graph_floor": graph_floor,
        "TP": metrics["TP"], "FP": metrics["FP"], "FN": metrics["FN"],
        "precision": metrics["precision"], "recall": metrics["recall"], "f1": metrics["f1"],
        "mean_output_count": metrics["mean_output_count"],
        "gain": gain, "lost": lost, "actions": actions_total,
    }, routed, details


def calibrate() -> dict[str, Any]:
    ensure_dirs()
    selector = make_selector(CACHE)
    records, _, contracts, current, v15, v11 = _load_split("calibration")
    scored, manifest = score_split("calibration", records, contracts, current, v15, v11, selector)
    candidate = aggregate_metrics(records, _candidate_map(records, current, scored))
    baseline_top = aggregate_metrics(records, current["top20"])
    baseline_v6 = aggregate_metrics(records, v6_selection(records, current["top20"]))
    top_rows: list[dict[str, Any]] = []
    top_products: dict[tuple[str, float, int], dict[str, list[dict[str, Any]]]] = {}
    top_details: dict[tuple[str, float, int], list[dict[str, Any]]] = {}
    for name, weights in WEIGHT_PROFILES.items():
        for margin in TOP_MARGINS:
            for max_promotions in TOP_PROMOTIONS:
                row, product, details = evaluate_top_profile(
                    records, current, scored, profile_name=name, weights=weights,
                    margin=margin, max_promotions=max_promotions,
                )
                key = (name, margin, max_promotions)
                top_rows.append(row); top_products[key] = product; top_details[key] = details
    eligible_top = [row for row in top_rows if row["top20_lost"] == 0]
    chosen_top = max(eligible_top or top_rows, key=lambda row: (
        row["top20_TP"], -row["actions"],
        row["margin"], -row["max_promotions"], row["configuration"],
    ))
    top_key = (chosen_top["configuration"], float(chosen_top["margin"]), int(chosen_top["max_promotions"]))
    chosen_top_map = top_products[top_key]
    chosen_weights = WEIGHT_PROFILES[chosen_top["configuration"]]
    chosen_top_v6_map = v6_selection(records, chosen_top_map)
    chosen_top_v6 = aggregate_metrics(records, chosen_top_v6_map)
    chosen_top["v6_TP"] = chosen_top_v6["TP"]
    chosen_top["v6_FP"] = chosen_top_v6["FP"]
    chosen_top["v6_f1"] = chosen_top_v6["f1"]

    final_rows: list[dict[str, Any]] = []
    final_products: dict[tuple[float, float, float], dict[str, list[dict[str, Any]]]] = {}
    final_details: dict[tuple[float, float, float], list[dict[str, Any]]] = {}
    for margin in FINAL_MARGINS:
        for cross_floor in FINAL_CROSS_FLOORS:
            for graph_floor in FINAL_GRAPH_FLOORS:
                row, product, details = evaluate_final_profile(
                    records, chosen_top_map, chosen_top_v6_map,
                    weights=chosen_weights, margin=margin,
                    cross_floor=cross_floor, graph_floor=graph_floor,
                )
                key = (margin, cross_floor, graph_floor)
                final_rows.append(row); final_products[key] = product; final_details[key] = details
    eligible_final = [row for row in final_rows if row["lost"] == 0]
    chosen_final = max(eligible_final or final_rows, key=lambda row: (
        row["f1"], row["TP"], -row["FP"], -row["actions"],
        row["cross_floor"], row["graph_floor"], row["margin"],
    ))
    final_key = (
        float(chosen_final["margin"]),
        float(chosen_final["cross_floor"]),
        float(chosen_final["graph_floor"]),
    )
    write_csv(REPORT / "01_calibration_top20_profiles.csv", top_rows)
    write_csv(REPORT / "02_calibration_top20_actions.csv", top_details[top_key])
    write_csv(REPORT / "03_calibration_final_profiles.csv", final_rows)
    write_csv(REPORT / "04_calibration_final_actions.csv", final_details[final_key])
    gate = (
        chosen_top["top20_TP"] >= baseline_top["TP"]
        and chosen_top["top20_lost"] == 0
        and chosen_final["TP"] >= baseline_v6["TP"]
        and chosen_final["f1"] >= baseline_v6["f1"]
    )
    freeze = {
        "schema_version": 1,
        "schema": SCHEMA,
        "source_commit": git_head(),
        "candidate": candidate,
        "baseline_top20": baseline_top,
        "baseline_v6": baseline_v6,
        "chosen_top": chosen_top,
        "top_weights": list(chosen_weights),
        "chosen_final": chosen_final,
        "calibration_gate_passed": gate,
        "dev_evaluated": False,
        "scoring_manifest": manifest,
    }
    (CONFIG / "calibration_freeze.json").write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")
    return freeze


def evaluate_dev() -> dict[str, Any]:
    ensure_dirs()
    freeze = json.loads((CONFIG / "calibration_freeze.json").read_text(encoding="utf-8"))
    if not freeze.get("calibration_gate_passed"):
        state = {"status": "rejected_on_calibration", "dev_runs": 0}
        (REPORT / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        return state
    selector = make_selector(CACHE)
    records, _, contracts, current, v15, v11 = _load_split("dev")
    scored, manifest = score_split("dev", records, contracts, current, v15, v11, selector)
    candidate = aggregate_metrics(records, _candidate_map(records, current, scored))
    baseline_top = aggregate_metrics(records, current["top20"])
    baseline_v6 = aggregate_metrics(records, v6_selection(records, current["top20"]))
    chosen_top = freeze["chosen_top"]
    top_metrics, top_map, top_details = evaluate_top_profile(
        records, current, scored,
        profile_name=str(chosen_top["configuration"]),
        weights=freeze["top_weights"],
        margin=float(chosen_top["margin"]),
        max_promotions=int(chosen_top["max_promotions"]),
    )
    chosen_final = freeze["chosen_final"]
    final_metrics, final_map, final_details = evaluate_final_profile(
        records, top_map, v6_selection(records, top_map), weights=freeze["top_weights"],
        margin=float(chosen_final["margin"]),
        cross_floor=float(chosen_final["cross_floor"]),
        graph_floor=float(chosen_final["graph_floor"]),
    )
    write_csv(REPORT / "05_dev20_metrics.csv", [
        {"configuration": "COVR_v10_top20", **baseline_top},
        {"configuration": "COVR_v10_plus_v6", **baseline_v6},
        {"configuration": "LGR_v17_candidate", **candidate},
        top_metrics,
        final_metrics,
    ])
    write_csv(REPORT / "06_dev20_top20_actions.csv", top_details)
    write_csv(REPORT / "07_dev20_final_actions.csv", final_details)
    accepted_ranking = (
        top_metrics["top20_TP"] >= baseline_top["TP"] + 1
        and top_metrics["top20_lost"] == 0
    )
    final_not_degraded = (
        final_metrics["TP"] >= baseline_v6["TP"]
        and final_metrics["f1"] >= baseline_v6["f1"]
    )
    accepted = accepted_ranking and final_not_degraded
    state = {
        "schema_version": 1,
        "status": "development_candidate_v17" if accepted else "rejected_on_dev",
        "branch": "local-graph-reranker-v17",
        "source_commit": freeze["source_commit"],
        "dev_runs": 1,
        "candidate": candidate,
        "baseline_top20": baseline_top,
        "baseline_v6": baseline_v6,
        "top20": top_metrics,
        "final": final_metrics,
        "accepted_ranking_component": accepted_ranking,
        "final_set_not_degraded": final_not_degraded,
        "accepted": accepted,
        "scoring_manifest": manifest,
    }
    (REPORT / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return state


def write_method() -> dict[str, Any]:
    ensure_dirs()
    text = """# LGR v17: Local Graph-Residual Reranking

LGR converts SCGQ citation-frontier recall into ranking decisions without an online LLM. It builds a deterministic 160-paper multiview pool, scores query-contract/document pairs with a cached local cross encoder, derives independent graph support from citation-seed agreement, and retains TRACE as a retrieval prior. Per-query empirical CDF normalization makes the three signals comparable. A residual router then promotes only frontier candidates that beat the weakest unprotected incumbent by a frozen margin; a stricter second router can replace at most one item in the final set.

Calibration freezes signal weights, pairwise margins, promotion budget, and confidence floors. Development evaluation reuses those values once.
"""
    path = REPORT / "00_method.md"
    path.write_text(text, encoding="utf-8")
    return {"written": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("method", "calibration", "dev"))
    args = parser.parse_args()
    result = write_method() if args.phase == "method" else (calibrate() if args.phase == "calibration" else evaluate_dev())
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
