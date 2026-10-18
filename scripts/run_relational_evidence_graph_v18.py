"""Calibration and diagnostic evaluation for Relational Evidence Graph v18."""

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

from sentence_transformers import CrossEncoder as _CrossEncoderBootstrap  # noqa: F401,E402

from backend.selection.coverage_adaptive_contract_v10 import identity_key
from backend.eval_runtime import EvaluationSession, evaluation_session
from backend.retrieval.query_conditioned_citation_v11 import retrieve_query_conditioned_frontier
from backend.retrieval.seed_conditioned_citation_v15 import retrieve_seed_conditioned_reverse_frontier
from backend.selection.local_graph_reranker_v17 import make_selector, score_query_pool
from backend.selection.relational_evidence_graph_v18 import (
    add_consensus_to_final,
    attach_relational_features,
    dual_view_consensus,
    route_relational_top20,
    select_quota_frontier_pool,
)
from backend.trace_rank import fuse_candidates
from scripts.run_citation_frontier_verifier_v16 import frontier_for
from scripts.audit_evaluation import load_jsonl, source_cutoff
from scripts.run_controlled_retrieval_v8 import aggregate_metrics, gold_keys, v6_selection
from scripts.run_frozen_official_benchmark import json_hash, legalize
from scripts.run_local_graph_reranker_v17 import CACHE as CROSS_CACHE
from scripts.run_local_graph_reranker_v17 import _contract, _load_split
from scripts.run_short_sprint_v5 import write_csv


REPORT = ROOT / "reports/relational_evidence_graph_v18"
CONFIG = ROOT / "configs/relational_evidence_graph_v18"
CAPTURE = ROOT / "reports/frozen_eval/relational_evidence_graph_v18"
HOLDOUT_SOURCE = ROOT / "reports/frozen_eval/query_sets/blind_holdout50_v4_private.jsonl"
HOLDOUT_BASE = ROOT / "reports/frozen_eval/blind_holdout50_v4_lean/queries"
SCHEMA = "reg-v18-condition-relation-direction-1"

WEIGHT_PROFILES = {
    "contract_strict": (0.45, 0.15, 0.10, 0.05, 0.10, 0.05, 0.10),
    "contract_graph": (0.35, 0.15, 0.10, 0.10, 0.15, 0.10, 0.05),
    "predicate_heavy": (0.35, 0.25, 0.10, 0.05, 0.10, 0.10, 0.05),
    "balanced": (0.30, 0.15, 0.10, 0.10, 0.15, 0.10, 0.10),
    "family_adaptive": {
        "default": (0.45, 0.15, 0.10, 0.05, 0.10, 0.05, 0.10),
        "taxonomy": (0.35, 0.15, 0.10, 0.10, 0.15, 0.10, 0.05),
        "survey": (0.35, 0.15, 0.10, 0.10, 0.15, 0.10, 0.05),
        "usage": (0.35, 0.15, 0.10, 0.10, 0.15, 0.10, 0.05),
    },
}
SCORE_FLOORS = (0.45, 0.55, 0.62, 0.68, 0.74)
MARGINS = (0.00, 0.03, 0.06, 0.10)
MAX_PROMOTIONS = (1, 2)
PROTECT_HEADS = (5, 10, 15)
CONSENSUS_STRICT_FLOORS = (0.85, 0.90, 0.92)
CONSENSUS_GRAPH_FLOORS = (0.80, 0.85, 0.88)


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def ensure_dirs() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    CONFIG.mkdir(parents=True, exist_ok=True)
    CAPTURE.mkdir(parents=True, exist_ok=True)


def _hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _paper_signature(paper: dict[str, Any]) -> tuple[str, str, str]:
    return identity_key(paper), str(paper.get("title") or ""), str(paper.get("abstract") or "")[:700]


def score_split(
    split: str,
    records: Sequence[dict[str, Any]],
    contracts: dict[str, dict[str, Any]],
    current: dict[str, dict[str, list[dict[str, Any]]]],
    v15: dict[str, dict[str, Any]],
    v11: dict[str, dict[str, Any]],
    selector: Any,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    scored: dict[str, dict[str, Any]] = {}
    manifest_rows: list[dict[str, Any]] = []
    dataset_name = {
        "calibration": "calibration40_v18",
        "dev": "dev20_v18",
        "holdout": "blind_holdout20_v18",
    }[split]
    root = CAPTURE / dataset_name
    started_all = time.perf_counter()
    for index, record in enumerate(records, 1):
        qid = record["qid"]
        incumbents = current["top20"][qid]
        all_frontier = frontier_for(qid, v15[qid], v11[qid])
        frontier = select_quota_frontier_pool(all_frontier, incumbents, limit=200)
        contract = _contract(contracts[qid])
        papers = [*incumbents, *frontier]
        signature = _hash({
            "schema": SCHEMA,
            "question": record["question"],
            "contract": contract,
            "incumbents": len(incumbents),
            "papers": [_paper_signature(paper) for paper in papers],
        })
        path = root / "queries" / qid / "scores.json"
        payload = None
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("schema") == SCHEMA and cached.get("signature") == signature:
                payload = cached
        if payload is None:
            started = time.perf_counter()
            cross_rows, runtime = score_query_pool(record["question"], contract, papers, selector, batch_size=64)
            rows = attach_relational_features(cross_rows, record["question"], contract)
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
            "all_frontier": all_frontier,
        }
        manifest_rows.append({
            "query_id": qid,
            "incumbent_count": incumbent_count,
            "frontier_count": len(rows) - incumbent_count,
            "runtime": (payload.get("runtime") or {}).get("runtime"),
            "cache_hits": (payload.get("runtime") or {}).get("cache_hits"),
            "elapsed_seconds": payload.get("elapsed_seconds"),
        })
        print(f"[{index}/{len(records)}] {qid} frontier={len(rows)-incumbent_count}", flush=True)
    manifest = {
        "schema": SCHEMA,
        "split": split,
        "query_count": len(records),
        "source_commit": git_head(),
        "wall_seconds": time.perf_counter() - started_all,
        "queries": manifest_rows,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return scored, manifest


def candidate_map(
    records: Sequence[dict[str, Any]],
    current: dict[str, dict[str, list[dict[str, Any]]]],
    scored: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        record["qid"]: fuse_candidates([
            *current["candidate"][record["qid"]],
            *scored[record["qid"]]["all_frontier"],
        ])
        for record in records
    }


def evaluate_profile(
    records: Sequence[dict[str, Any]],
    scored: dict[str, dict[str, Any]],
    *,
    name: str,
    weights: Any,
    score_floor: float,
    margin: float,
    max_promotions: int,
    protect_head: int,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    routed: dict[str, list[dict[str, Any]]] = {}
    details: list[dict[str, Any]] = []
    gain = lost = actions_total = gain_queries = loss_queries = 0
    fold_gain = [0, 0, 0, 0]
    fold_lost = [0, 0, 0, 0]
    for record in records:
        qid = record["qid"]
        result, actions = route_relational_top20(
            scored[qid]["incumbents"], scored[qid]["frontier"],
            weights=weights, margin=margin, score_floor=score_floor,
            max_promotions=max_promotions, protect_head=protect_head,
        )
        routed[qid] = result
        old_gold = gold_keys(scored[qid]["incumbents"], record)
        new_gold = gold_keys(result, record)
        qgain, qlost = len(new_gold-old_gold), len(old_gold-new_gold)
        fold = int(qid.rsplit("_", 1)[-1]) % 4
        fold_gain[fold] += qgain; fold_lost[fold] += qlost
        gain += qgain; lost += qlost; actions_total += len(actions)
        gain_queries += int(qgain > 0); loss_queries += int(qlost > 0)
        details.append({
            "query_id": qid, "fold": fold, "actions": len(actions),
            "top20_gain": qgain, "top20_lost": qlost,
            "incoming_titles": " | ".join(str(action.get("incoming_title") or "") for action in actions),
            "outgoing_titles": " | ".join(str(action.get("outgoing_title") or "") for action in actions),
            "incoming_scores": " | ".join(f"{float(action.get('incoming_score') or 0):.4f}" for action in actions),
            "families": " | ".join(str(action.get("family") or "") for action in actions),
        })
    metrics = aggregate_metrics(records, routed)
    return {
        "configuration": name,
        "weights": json.dumps(weights, sort_keys=True) if isinstance(weights, dict) else "/".join(str(value) for value in weights),
        "score_floor": score_floor,
        "margin": margin,
        "max_promotions": max_promotions,
        "protect_head": protect_head,
        "top20_TP": metrics["TP"], "top20_FP": metrics["FP"],
        "top20_recall": metrics["recall"], "top20_f1": metrics["f1"],
        "top20_gain": gain, "top20_lost": lost,
        "gain_queries": gain_queries, "loss_queries": loss_queries,
        "actions": actions_total,
        "action_precision": gain / actions_total if actions_total else 0.0,
        "fold_gain": "/".join(str(value) for value in fold_gain),
        "fold_lost": "/".join(str(value) for value in fold_lost),
        "gain_folds": sum(value > 0 for value in fold_gain),
    }, routed, details


def evaluate_consensus_final(
    records: Sequence[dict[str, Any]],
    scored: dict[str, dict[str, Any]],
    base_selected: dict[str, list[dict[str, Any]]],
    *,
    strict_floor: float,
    graph_floor: float,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    routed: dict[str, list[dict[str, Any]]] = {}
    details: list[dict[str, Any]] = []
    gain = additions = 0
    strict_weights = WEIGHT_PROFILES["contract_strict"]
    graph_weights = WEIGHT_PROFILES["contract_graph"]
    for record in records:
        qid = record["qid"]
        candidate, audit = dual_view_consensus(
            scored[qid]["frontier"], strict_weights, graph_weights,
            strict_floor=strict_floor, graph_floor=graph_floor,
        )
        result, added = add_consensus_to_final(base_selected[qid], candidate)
        routed[qid] = result
        old_gold, new_gold = gold_keys(base_selected[qid], record), gold_keys(result, record)
        qgain = len(new_gold-old_gold)
        gain += qgain; additions += int(added)
        details.append({
            "query_id": qid, "added": int(added), "final_gain": qgain,
            "candidate_title": candidate.get("title") if candidate else "",
            "strict_title": audit.get("strict_title"), "graph_title": audit.get("graph_title"),
            "strict_score": audit.get("strict_score"), "graph_score": audit.get("graph_score"),
            "reason": audit.get("reason"),
        })
    metrics = aggregate_metrics(records, routed)
    return {
        "configuration": "dual_view_consensus_add",
        "strict_floor": strict_floor, "graph_floor": graph_floor,
        "TP": metrics["TP"], "FP": metrics["FP"], "FN": metrics["FN"],
        "precision": metrics["precision"], "recall": metrics["recall"], "f1": metrics["f1"],
        "mean_output_count": metrics["mean_output_count"],
        "gain": gain, "additions": additions,
        "addition_precision": gain / additions if additions else 0.0,
    }, routed, details


def calibrate() -> dict[str, Any]:
    ensure_dirs()
    selector = make_selector(CROSS_CACHE)
    records, _, contracts, current, v15, v11 = _load_split("calibration")
    scored, manifest = score_split("calibration", records, contracts, current, v15, v11, selector)
    baseline_top = aggregate_metrics(records, current["top20"])
    baseline_v6 = aggregate_metrics(records, v6_selection(records, current["top20"]))
    candidate = aggregate_metrics(records, candidate_map(records, current, scored))
    rows: list[dict[str, Any]] = []
    products: dict[tuple[Any, ...], dict[str, list[dict[str, Any]]]] = {}
    details_by: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for name, weights in WEIGHT_PROFILES.items():
        for score_floor in SCORE_FLOORS:
            for margin in MARGINS:
                for max_promotions in MAX_PROMOTIONS:
                    for protect_head in PROTECT_HEADS:
                        row, product, details = evaluate_profile(
                            records, scored, name=name, weights=weights,
                            score_floor=score_floor, margin=margin,
                            max_promotions=max_promotions, protect_head=protect_head,
                        )
                        key = (name, score_floor, margin, max_promotions, protect_head)
                        rows.append(row); products[key] = product; details_by[key] = details
    eligible = [row for row in rows if row["top20_lost"] == 0]
    chosen = max(eligible or rows, key=lambda row: (
        row["top20_TP"], row["gain_folds"],
        int(row["configuration"] == "family_adaptive"), row["action_precision"],
        -row["actions"], row["protect_head"], row["score_floor"], row["margin"],
    ))
    key = (
        chosen["configuration"], float(chosen["score_floor"]), float(chosen["margin"]),
        int(chosen["max_promotions"]), int(chosen["protect_head"]),
    )
    chosen_top = products[key]
    chosen_v6 = aggregate_metrics(records, v6_selection(records, chosen_top))
    chosen["v6_TP"] = chosen_v6["TP"]
    chosen["v6_FP"] = chosen_v6["FP"]
    chosen["v6_f1"] = chosen_v6["f1"]
    write_csv(REPORT / "01_calibration_profiles.csv", rows)
    write_csv(REPORT / "02_calibration_actions.csv", details_by[key])
    base_selected = v6_selection(records, current["top20"])
    consensus_rows: list[dict[str, Any]] = []
    consensus_products: dict[tuple[float, float], dict[str, list[dict[str, Any]]]] = {}
    consensus_details: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for strict_floor in CONSENSUS_STRICT_FLOORS:
        for graph_floor in CONSENSUS_GRAPH_FLOORS:
            row, product, details = evaluate_consensus_final(
                records, scored, base_selected,
                strict_floor=strict_floor, graph_floor=graph_floor,
            )
            ckey = (strict_floor, graph_floor)
            consensus_rows.append(row); consensus_products[ckey] = product; consensus_details[ckey] = details
    precision_eligible = [
        row for row in consensus_rows
        if row["gain"] >= 1 and row["f1"] >= baseline_v6["f1"] + 0.01 - 1e-12
    ]
    chosen_consensus = max(precision_eligible or consensus_rows, key=lambda row: (
        row["addition_precision"], row["f1"], row["TP"], -row["FP"],
        -row["additions"], row["strict_floor"], row["graph_floor"],
    ))
    consensus_key = (float(chosen_consensus["strict_floor"]), float(chosen_consensus["graph_floor"]))
    write_csv(REPORT / "03_calibration_consensus_profiles.csv", consensus_rows)
    write_csv(REPORT / "04_calibration_consensus_actions.csv", consensus_details[consensus_key])
    gate = chosen["top20_gain"] >= 2 and chosen["top20_lost"] == 0 and chosen["gain_folds"] >= 2
    freeze = {
        "schema_version": 1, "schema": SCHEMA, "source_commit": git_head(),
        "candidate": candidate, "baseline_top20": baseline_top, "baseline_v6": baseline_v6,
        "chosen": chosen, "weights": WEIGHT_PROFILES[chosen["configuration"]],
        "chosen_v6": chosen_v6, "chosen_consensus": chosen_consensus,
        "calibration_gate_passed": gate,
        "dev_evaluated": False, "manifest": manifest,
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
    selector = make_selector(CROSS_CACHE)
    records, _, contracts, current, v15, v11 = _load_split("dev")
    scored, manifest = score_split("dev", records, contracts, current, v15, v11, selector)
    baseline_top = aggregate_metrics(records, current["top20"])
    baseline_v6 = aggregate_metrics(records, v6_selection(records, current["top20"]))
    chosen = freeze["chosen"]
    metrics, top20, details = evaluate_profile(
        records, scored, name=str(chosen["configuration"]), weights=freeze["weights"],
        score_floor=float(chosen["score_floor"]), margin=float(chosen["margin"]),
        max_promotions=int(chosen["max_promotions"]), protect_head=int(chosen["protect_head"]),
    )
    routed_v6 = aggregate_metrics(records, v6_selection(records, top20))
    base_selected = v6_selection(records, current["top20"])
    consensus = freeze["chosen_consensus"]
    consensus_metrics, consensus_final, consensus_details = evaluate_consensus_final(
        records, scored, base_selected,
        strict_floor=float(consensus["strict_floor"]),
        graph_floor=float(consensus["graph_floor"]),
    )
    candidate = aggregate_metrics(records, candidate_map(records, current, scored))
    write_csv(REPORT / "05_dev20_metrics.csv", [
        {"configuration": "COVR_v10_top20", **baseline_top},
        {"configuration": "COVR_v10_plus_v6", **baseline_v6},
        {"configuration": "REG_v18_candidate", **candidate},
        metrics,
        {"configuration": "REG_v18_plus_v6", **routed_v6},
        consensus_metrics,
    ])
    write_csv(REPORT / "06_dev20_actions.csv", details)
    write_csv(REPORT / "07_dev20_consensus_actions.csv", consensus_details)
    accepted = (
        metrics["top20_gain"] >= 1 and metrics["top20_lost"] == 0
        and consensus_metrics["f1"] >= baseline_v6["f1"]
    )
    state = {
        "schema_version": 1,
        "status": "development_candidate_v18" if accepted else "rejected_on_dev",
        "branch": "relational-evidence-graph-v18", "source_commit": freeze["source_commit"],
        "dev_runs": 1, "candidate": candidate,
        "baseline_top20": baseline_top, "baseline_v6": baseline_v6,
        "top20": metrics, "routed_v6": routed_v6, "consensus_final": consensus_metrics,
        "accepted": accepted, "manifest": manifest,
    }
    (REPORT / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return state


def holdout_records() -> list[dict[str, Any]]:
    dataset = {row["qid"]: row for row in load_jsonl(HOLDOUT_SOURCE)}
    ids_path = CONFIG / "holdout20_ids.json"
    if ids_path.exists():
        ids = json.loads(ids_path.read_text(encoding="utf-8"))["query_ids"]
    else:
        ids = sorted(dataset, key=lambda value: int(value.rsplit("_", 1)[-1]))[:20]
        ids_path.parent.mkdir(parents=True, exist_ok=True)
        ids_path.write_text(json.dumps({"query_ids": ids}, ensure_ascii=False, indent=2), encoding="utf-8")
    records = []
    for qid in ids:
        row = copy.deepcopy(dataset[qid]); row["_v8_split"] = "v18_holdout"; records.append(row)
    return records


def _holdout_response(qid: str) -> dict[str, Any]:
    return json.loads((HOLDOUT_BASE / qid / "live_response.json").read_text(encoding="utf-8"))


def _holdout_top20(records: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {record["qid"]: copy.deepcopy((_holdout_response(record["qid"]).get("papers") or [])[:20]) for record in records}


def _holdout_pool(record: dict[str, Any]) -> list[dict[str, Any]]:
    qid = record["qid"]
    snapshot = json.loads((HOLDOUT_BASE / qid / "snapshot.json").read_text(encoding="utf-8"))
    papers: list[dict[str, Any]] = []
    for entry in (snapshot.get("calls") or {}).values():
        output = entry.get("output")
        if not isinstance(output, list) or str(entry.get("label") or "").startswith("llm_"):
            continue
        papers.extend(item for item in output if isinstance(item, dict) and item.get("title"))
    papers.extend((_holdout_response(qid).get("papers") or [])[:20])
    return fuse_candidates(legalize(papers, source_cutoff(record)))


def _contract_from_understanding(understanding: dict[str, Any]) -> dict[str, Any]:
    graph = understanding.get("constraint_graph") or {}
    conditions = []
    relation = ""
    exclusions = list(graph.get("negative_constraints") or [])
    for criterion in graph.get("criteria") or []:
        kind = str(criterion.get("type") or "topic")
        description = str(criterion.get("description") or "").strip()
        if not description:
            continue
        if kind == "relation":
            relation = relation or description
        elif kind == "negative":
            exclusions.append(description)
        elif kind not in {"time", "paper_type"}:
            conditions.append({
                "id": str(criterion.get("criterion_id") or f"c{len(conditions)+1}"),
                "description": description,
                "aliases": [],
            })
    relation = relation or str((graph.get("relation_constraints") or [""])[0])
    if not conditions:
        for field in ("research_topic", "method_constraints", "task_constraints", "domain_constraints"):
            value = str(understanding.get(field) or "").strip()
            if value and value.lower() != "not specified":
                conditions.append({"id": f"c{len(conditions)+1}", "description": value, "aliases": []})
    return {"conditions": conditions[:6], "relation": relation, "exclusions": exclusions[:8]}


async def _capture_holdout_retrieval(
    records: Sequence[dict[str, Any]],
    top20: dict[str, list[dict[str, Any]]],
    pools: dict[str, list[dict[str, Any]]],
    understandings: dict[str, dict[str, Any]],
    contracts: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    root = CAPTURE / "blind_holdout20_v18_retrieval"
    results: dict[str, dict[str, Any]] = {}
    manifest_rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, 1):
        qid = record["qid"]
        query_root = root / "queries" / qid
        live_path, snapshot_path = query_root / "live_result.json", query_root / "snapshot.json"

        async def execute(mode: str) -> tuple[dict[str, Any], EvaluationSession]:
            session = EvaluationSession.create("blind_holdout20_v18_retrieval", qid, source_cutoff(record), mode, snapshot_path)
            with evaluation_session(session):
                reverse = await retrieve_seed_conditioned_reverse_frontier(
                    record["question"], understandings[qid], contracts[qid], top20[qid], pools[qid]
                )
                qcf = await retrieve_query_conditioned_frontier(
                    record["question"], understandings[qid], contracts[qid], top20[qid], pools[qid]
                )
            payload = {"query_id": qid, "reverse": reverse, "qcf": qcf}
            if mode == "live":
                session.save({"source_commit": git_head(), "schema": "v18-holdout-retrieval-1"})
                query_root.mkdir(parents=True, exist_ok=True)
                live_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return payload, session

        if live_path.exists() and snapshot_path.exists():
            live = json.loads(live_path.read_text(encoding="utf-8"))
        else:
            live, _ = await execute("live")
        replay, _ = await execute("replay")
        projection = lambda payload: {
            "reverse": [(paper.get("openalex_id"), paper.get("title")) for paper in (payload["reverse"].get("candidates") or [])],
            "qcf": [(paper.get("openalex_id"), paper.get("title")) for paper in (payload["qcf"].get("candidates") or [])],
        }
        identical = json_hash(projection(live)) == json_hash(projection(replay))
        results[qid] = live
        manifest_rows.append({
            "query_id": qid, "replay_identical": identical,
            "reverse_candidates": len(live["reverse"].get("candidates") or []),
            "qcf_candidates": len(live["qcf"].get("candidates") or []),
            "api_calls": int(live["reverse"].get("api_calls") or 0) + int(live["qcf"].get("api_calls") or 0),
            "errors": [*(live["reverse"].get("errors") or []), *(live["qcf"].get("errors") or [])],
        })
        print(f"[{index}/{len(records)}] {qid} holdout retrieval replay={'OK' if identical else 'DIFF'}", flush=True)
    manifest = {
        "schema": "v18-holdout-retrieval-1", "query_count": len(records),
        "source_commit": git_head(), "replay_all_identical": all(row["replay_identical"] for row in manifest_rows),
        "queries": manifest_rows,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return results, manifest


async def evaluate_holdout() -> dict[str, Any]:
    ensure_dirs()
    freeze = json.loads((CONFIG / "calibration_freeze.json").read_text(encoding="utf-8"))
    if not freeze.get("calibration_gate_passed"):
        return {"status": "rejected_on_calibration", "holdout_runs": 0}
    selector = make_selector(CROSS_CACHE)
    records = holdout_records()
    top20 = _holdout_top20(records)
    pools = {record["qid"]: _holdout_pool(record) for record in records}
    understandings = {record["qid"]: copy.deepcopy(_holdout_response(record["qid"]).get("query_understanding") or {}) for record in records}
    contracts_plain = {qid: _contract_from_understanding(value) for qid, value in understandings.items()}
    retrieval, retrieval_manifest = await _capture_holdout_retrieval(records, top20, pools, understandings, contracts_plain)
    if not retrieval_manifest["replay_all_identical"]:
        raise RuntimeError("holdout retrieval replay mismatch")
    current = {"top20": top20, "candidate": pools}
    v15 = {qid: payload["reverse"] for qid, payload in retrieval.items()}
    v11 = {qid: payload["qcf"] for qid, payload in retrieval.items()}
    contract_results = {
        qid: {"contract_result": {"parsed": {"contract": contract}}}
        for qid, contract in contracts_plain.items()
    }
    scored, score_manifest = score_split("holdout", records, contract_results, current, v15, v11, selector)
    baseline_top = aggregate_metrics(records, top20)
    base_selected = v6_selection(records, top20)
    baseline_v6 = aggregate_metrics(records, base_selected)
    chosen = freeze["chosen"]
    top_metrics, routed_top, top_details = evaluate_profile(
        records, scored, name=str(chosen["configuration"]), weights=freeze["weights"],
        score_floor=float(chosen["score_floor"]), margin=float(chosen["margin"]),
        max_promotions=int(chosen["max_promotions"]), protect_head=int(chosen["protect_head"]),
    )
    consensus = freeze["chosen_consensus"]
    final_metrics, final_map, final_details = evaluate_consensus_final(
        records, scored, base_selected,
        strict_floor=float(consensus["strict_floor"]), graph_floor=float(consensus["graph_floor"]),
    )
    candidate = aggregate_metrics(records, candidate_map(records, current, scored))
    write_csv(REPORT / "08_holdout20_metrics.csv", [
        {"configuration": "holdout_baseline_top20", **baseline_top},
        {"configuration": "holdout_baseline_v6", **baseline_v6},
        {"configuration": "holdout_v18_candidate", **candidate},
        top_metrics, final_metrics,
    ])
    write_csv(REPORT / "09_holdout20_top20_actions.csv", top_details)
    write_csv(REPORT / "10_holdout20_consensus_actions.csv", final_details)
    accepted = (
        top_metrics["top20_TP"] >= baseline_top["TP"]
        and top_metrics["top20_lost"] == 0
        and final_metrics["f1"] >= baseline_v6["f1"]
    )
    state = {
        "schema_version": 1, "status": "holdout_validated_v18" if accepted else "rejected_on_holdout",
        "branch": "relational-evidence-graph-v18", "source_commit": freeze["source_commit"],
        "holdout_runs": 1, "baseline_top20": baseline_top, "baseline_v6": baseline_v6,
        "candidate": candidate, "top20": top_metrics, "consensus_final": final_metrics,
        "accepted": accepted, "retrieval_manifest": retrieval_manifest, "score_manifest": score_manifest,
    }
    (REPORT / "holdout_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return state


def write_method() -> dict[str, Any]:
    ensure_dirs()
    text = """# REG v18: Relational Evidence Graph

REG represents each complex query as required condition nodes plus a relational predicate. Candidate evidence is decomposed into condition-average and minimum-condition coverage, predicate cues, citation-direction compatibility, seed-path coherence, graph agreement, TRACE, and local cross-encoder relevance. The geometric conjunction term penalizes partial matches, while direction compatibility distinguishes foundational/proposal queries from extension/analysis queries. A quota-based 200-paper frontier preserves deep candidates from every citation lane. Routing protects a frozen head of the incumbent Top20 and requires both an absolute evidence floor and an incumbent-relative margin.
"""
    path = REPORT / "00_method.md"; path.write_text(text, encoding="utf-8")
    return {"written": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("phase", choices=("method", "calibration", "dev", "holdout")); args = parser.parse_args()
    if args.phase == "method":
        result = write_method()
    elif args.phase == "calibration":
        result = calibrate()
    elif args.phase == "dev":
        result = evaluate_dev()
    else:
        result = asyncio.run(evaluate_holdout())
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
