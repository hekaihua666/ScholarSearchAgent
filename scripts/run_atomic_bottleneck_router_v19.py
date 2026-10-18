"""Frozen calibration and evaluation for Atomic Bottleneck Evidence Routing."""

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

# Import torch before the legacy sklearn/OpenMP stack on Windows.
from sentence_transformers import CrossEncoder as _CrossEncoderBootstrap  # noqa: F401,E402

from backend.selection.atomic_bottleneck_router_v19 import (
    attach_atomic_cross_evidence,
    route_atomic_bottleneck_top20,
    select_atomic_screen_pool,
)
from backend.selection.relational_evidence_graph_v18 import route_relational_top20
from scripts.run_controlled_retrieval_v8 import aggregate_metrics, gold_keys
from scripts.run_local_graph_reranker_v17 import _contract, _load_split
from scripts.run_relational_evidence_graph_v18 import WEIGHT_PROFILES as V18_WEIGHTS
from scripts.run_short_sprint_v5 import write_csv
from backend.selection.local_graph_reranker_v17 import make_selector


REPORT = ROOT / "reports/atomic_bottleneck_routing_v19"
CONFIG = ROOT / "configs/atomic_bottleneck_routing_v19"
CAPTURE = ROOT / "reports/frozen_eval/atomic_bottleneck_routing_v19"
V18_CAPTURE = ROOT / "reports/frozen_eval/relational_evidence_graph_v18"
V18_FREEZE = ROOT / "configs/relational_evidence_graph_v18/calibration_freeze.json"
SCHEMA = "aber-v19-atomic-condition-bottleneck-1"
ATOMIC_CACHE = ROOT / "cache/v19_atomic_cross_scores.json"

WEIGHT_PROFILES = {
    "bottleneck": (0.35, 0.15, 0.10, 0.15, 0.10, 0.10, 0.05),
    "bottleneck_graph": (0.30, 0.15, 0.10, 0.20, 0.10, 0.10, 0.05),
    "bottleneck_trace": (0.35, 0.15, 0.10, 0.10, 0.15, 0.10, 0.05),
    "semantic_strict": (0.40, 0.20, 0.10, 0.10, 0.05, 0.10, 0.05),
    "balanced": (0.25, 0.15, 0.10, 0.15, 0.15, 0.15, 0.05),
}
SCORE_FLOORS = (0.58, 0.66, 0.74)
MARGINS = (0.05, 0.10)
MINIMUM_DELTAS = (-0.05, 0.00)
DOMINANCE_REQUIRED = (3, 4)


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def ensure_dirs() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    CONFIG.mkdir(parents=True, exist_ok=True)
    CAPTURE.mkdir(parents=True, exist_ok=True)


def _hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _v18_dataset(split: str) -> str:
    return {"calibration": "calibration40_v18", "dev": "dev20_v18"}[split]


def score_atomic_split(
    split: str,
    records: Sequence[dict[str, Any]],
    contracts: dict[str, dict[str, Any]],
    selector: Any,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    manifest_rows: list[dict[str, Any]] = []
    root = CAPTURE / {"calibration": "calibration40_v19", "dev": "dev20_v19"}[split]
    started_all = time.perf_counter()
    for index, record in enumerate(records, 1):
        qid = record["qid"]
        v18_path = V18_CAPTURE / _v18_dataset(split) / "queries" / qid / "scores.json"
        v18 = json.loads(v18_path.read_text(encoding="utf-8"))
        v18_rows = v18["rows"]
        incumbent_count = int(v18["incumbent_count"])
        rows = [
            *v18_rows[:incumbent_count],
            *select_atomic_screen_pool(v18_rows[incumbent_count:], limit=32),
        ]
        contract = _contract(contracts[qid])
        signature = _hash({
            "schema": SCHEMA, "question": record["question"], "contract": contract,
            "v18_signature": v18.get("signature"), "screen_limit": 32,
            "papers": [(row.get("openalex_id"), row.get("doi"), row.get("title")) for row in rows],
        })
        path = root / "queries" / qid / "scores.json"
        payload = None
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("schema") == SCHEMA and cached.get("signature") == signature:
                payload = cached
        if payload is None:
            started = time.perf_counter()
            scored, runtime = attach_atomic_cross_evidence(
                rows, record["question"], contract, selector, batch_size=128,
            )
            payload = {
                "schema": SCHEMA, "signature": signature, "query_id": qid,
                "incumbent_count": incumbent_count, "rows": scored,
                "runtime": runtime, "elapsed_seconds": time.perf_counter() - started,
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        n = int(payload["incumbent_count"])
        output[qid] = {"incumbents": payload["rows"][:n], "frontier": payload["rows"][n:]}
        manifest_rows.append({
            "query_id": qid, "incumbent_count": n,
            "frontier_count": len(payload["rows"]) - n,
            "condition_views": len([v for v in (payload.get("runtime") or {}).get("atomic_views") or [] if v.get("kind") == "condition"]),
            "pair_count": (payload.get("runtime") or {}).get("pair_count"),
            "runtime": (payload.get("runtime") or {}).get("runtime"),
            "cache_hits": (payload.get("runtime") or {}).get("cache_hits"),
            "elapsed_seconds": payload.get("elapsed_seconds"),
        })
        print(f"[{index}/{len(records)}] {qid} atomic views={manifest_rows[-1]['condition_views']}", flush=True)
    manifest = {
        "schema": SCHEMA, "split": split, "query_count": len(records),
        "source_commit": git_head(), "wall_seconds": time.perf_counter() - started_all,
        "queries": manifest_rows,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return output, manifest


def evaluate_profile(
    records: Sequence[dict[str, Any]],
    scored: dict[str, dict[str, Any]],
    *,
    name: str,
    weights: Sequence[float],
    score_floor: float,
    margin: float,
    minimum_delta: float,
    dominance_required: int,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    routed: dict[str, list[dict[str, Any]]] = {}
    details: list[dict[str, Any]] = []
    gain = lost = actions_total = 0
    fold_gain = [0, 0, 0, 0]
    fold_lost = [0, 0, 0, 0]
    for record in records:
        qid = record["qid"]
        result, actions = route_atomic_bottleneck_top20(
            scored[qid]["incumbents"], scored[qid]["frontier"],
            weights=weights, score_floor=score_floor, margin=margin,
            minimum_delta=minimum_delta, dominance_required=dominance_required,
            protect_head=15, max_promotions=1,
        )
        routed[qid] = result
        old_gold = gold_keys(scored[qid]["incumbents"], record)
        new_gold = gold_keys(result, record)
        qgain, qlost = len(new_gold - old_gold), len(old_gold - new_gold)
        fold = int(qid.rsplit("_", 1)[-1]) % 4
        gain += qgain; lost += qlost; actions_total += len(actions)
        fold_gain[fold] += qgain; fold_lost[fold] += qlost
        action = actions[0] if actions else {}
        details.append({
            "query_id": qid, "fold": fold, "actions": len(actions),
            "top20_gain": qgain, "top20_lost": qlost,
            "incoming_title": action.get("incoming_title", ""),
            "outgoing_title": action.get("outgoing_title", ""),
            "incoming_score": action.get("incoming_score", ""),
            "atomic_min": action.get("atomic_min", ""),
            "atomic_geo": action.get("atomic_geo", ""),
            "relation": action.get("relation", ""),
            "weak_delta": action.get("weak_delta", ""),
            "dominance": action.get("dominance", ""),
        })
    metrics = aggregate_metrics(records, routed)
    row = {
        "configuration": name,
        "weights": "/".join(str(value) for value in weights),
        "score_floor": score_floor, "margin": margin,
        "minimum_delta": minimum_delta, "dominance_required": dominance_required,
        "top20_TP": metrics["TP"], "top20_FP": metrics["FP"],
        "top20_recall": metrics["recall"], "top20_f1": metrics["f1"],
        "top20_gain": gain, "top20_lost": lost, "actions": actions_total,
        "action_precision": gain / actions_total if actions_total else 0.0,
        "fold_gain": "/".join(str(value) for value in fold_gain),
        "fold_lost": "/".join(str(value) for value in fold_lost),
        "gain_folds": sum(value > 0 for value in fold_gain),
    }
    return row, routed, details


def v18_baseline(records: Sequence[dict[str, Any]], scored: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    freeze = json.loads(V18_FREEZE.read_text(encoding="utf-8"))
    chosen = freeze["chosen"]
    routed = {}
    gain = lost = actions = 0
    for record in records:
        qid = record["qid"]
        result, audit = route_relational_top20(
            scored[qid]["incumbents"], scored[qid]["frontier"],
            weights=freeze["weights"], score_floor=float(chosen["score_floor"]),
            margin=float(chosen["margin"]), max_promotions=int(chosen["max_promotions"]),
            protect_head=int(chosen["protect_head"]),
        )
        routed[qid] = result
        old, new = gold_keys(scored[qid]["incumbents"], record), gold_keys(result, record)
        gain += len(new-old); lost += len(old-new); actions += len(audit)
    metrics = aggregate_metrics(records, routed)
    return {"configuration": "REG_v18", **metrics, "top20_gain": gain, "top20_lost": lost, "actions": actions}, routed


def calibrate() -> dict[str, Any]:
    ensure_dirs()
    selector = make_selector(ATOMIC_CACHE)
    records, _, contracts, current, _, _ = _load_split("calibration")
    scored, manifest = score_atomic_split("calibration", records, contracts, selector)
    baseline = aggregate_metrics(records, current["top20"])
    reg, _ = v18_baseline(records, scored)
    rows: list[dict[str, Any]] = []
    products: dict[tuple[Any, ...], dict[str, list[dict[str, Any]]]] = {}
    details_by: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for name, weights in WEIGHT_PROFILES.items():
        for score_floor in SCORE_FLOORS:
            for margin in MARGINS:
                for minimum_delta in MINIMUM_DELTAS:
                    for dominance in DOMINANCE_REQUIRED:
                        row, routed, details = evaluate_profile(
                            records, scored, name=name, weights=weights,
                            score_floor=score_floor, margin=margin,
                            minimum_delta=minimum_delta, dominance_required=dominance,
                        )
                        key = (name, score_floor, margin, minimum_delta, dominance)
                        rows.append(row); products[key] = routed; details_by[key] = details
    eligible = [row for row in rows if row["top20_lost"] == 0 and row["top20_gain"] >= 2 and row["gain_folds"] >= 2]
    chosen = max(eligible or rows, key=lambda row: (
        row["top20_TP"] - 2 * row["top20_lost"], row["gain_folds"],
        row["action_precision"], -row["actions"], row["dominance_required"],
        row["minimum_delta"], row["score_floor"], row["margin"],
    ))
    key = (chosen["configuration"], float(chosen["score_floor"]), float(chosen["margin"]),
           float(chosen["minimum_delta"]), int(chosen["dominance_required"]))
    write_csv(REPORT / "01_calibration_profiles.csv", rows)
    write_csv(REPORT / "02_calibration_actions.csv", details_by[key])
    gate = chosen["top20_gain"] >= 2 and chosen["top20_lost"] == 0 and chosen["gain_folds"] >= 2
    state = {
        "schema_version": 1, "schema": SCHEMA, "source_commit": git_head(),
        "baseline_top20": baseline, "reg_v18": reg, "chosen": chosen,
        "weights": WEIGHT_PROFILES[chosen["configuration"]],
        "calibration_gate_passed": gate, "dev_evaluated": False,
        "manifest": manifest,
    }
    (CONFIG / "calibration_freeze.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return state


def evaluate_dev() -> dict[str, Any]:
    ensure_dirs()
    freeze = json.loads((CONFIG / "calibration_freeze.json").read_text(encoding="utf-8"))
    if not freeze.get("calibration_gate_passed"):
        state = {"status": "rejected_on_calibration", "dev_runs": 0}
        (REPORT / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
        return state
    selector = make_selector(ATOMIC_CACHE)
    records, _, contracts, current, _, _ = _load_split("dev")
    scored, manifest = score_atomic_split("dev", records, contracts, selector)
    baseline = aggregate_metrics(records, current["top20"])
    reg, _ = v18_baseline(records, scored)
    chosen = freeze["chosen"]
    metrics, routed, details = evaluate_profile(
        records, scored, name=str(chosen["configuration"]), weights=freeze["weights"],
        score_floor=float(chosen["score_floor"]), margin=float(chosen["margin"]),
        minimum_delta=float(chosen["minimum_delta"]),
        dominance_required=int(chosen["dominance_required"]),
    )
    write_csv(REPORT / "03_dev20_metrics.csv", [
        {"configuration": "COVR_v10_top20", **baseline}, reg, metrics,
    ])
    write_csv(REPORT / "04_dev20_actions.csv", details)
    accepted = metrics["top20_gain"] >= 1 and metrics["top20_lost"] == 0 and metrics["top20_TP"] >= reg["TP"]
    state = {
        "schema_version": 1,
        "status": "development_candidate_v19" if accepted else "rejected_on_dev",
        "branch": "atomic-bottleneck-routing-v19", "source_commit": freeze["source_commit"],
        "dev_runs": 1, "baseline_top20": baseline, "reg_v18": reg,
        "top20": metrics, "accepted": accepted, "manifest": manifest,
    }
    (REPORT / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("calibration", "dev"))
    args = parser.parse_args()
    result = calibrate() if args.phase == "calibration" else evaluate_dev()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
