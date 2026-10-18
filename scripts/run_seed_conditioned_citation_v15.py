"""Frozen retrieval evaluation for Seed-Conditioned Citation Frontier v15."""

from __future__ import annotations

import argparse
import asyncio
import csv
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

from backend.eval_runtime import EvaluationSession, evaluation_session
from backend.paper_identity import match_identity
from backend.retrieval.seed_conditioned_citation_v15 import retrieve_seed_conditioned_reverse_frontier
from backend.trace_rank import fuse_candidates
from scripts.audit_evaluation import source_cutoff
from scripts.run_controlled_retrieval_v8 import (
    aggregate_metrics,
    baseline_pool,
    baseline_top20_map,
    gold_keys,
    records_for,
    understanding_for,
)
from scripts.run_coverage_adaptive_contract_v10 import load_inputs
from scripts.run_evidence_contract_rank_v9 import candidate_pool
from scripts.run_frozen_official_benchmark import json_hash
from scripts.run_short_sprint_v5 import write_csv


REPORT = ROOT / "reports/seed_conditioned_citation_v15"
CONFIG = ROOT / "configs/seed_conditioned_citation_v15"
CAPTURE = ROOT / "reports/frozen_eval/seed_conditioned_citation_v15"
V9_ROOTS = {
    "calibration": ROOT / "reports/frozen_eval/evidence_contract_rank_v9/calibration40_v9/queries",
    "dev": ROOT / "reports/frozen_eval/evidence_contract_rank_v9/dev20_v9_contract/queries",
}
V11_FORWARD_ROOTS = {
    "calibration": ROOT / "reports/frozen_eval/query_conditioned_citation_v11/calibration40_v11/queries",
    "dev": ROOT / "reports/frozen_eval/query_conditioned_citation_v11/dev20_v11/queries",
}
LANE_PROFILES = {
    "reverse_group_1": ("reverse_group_1",),
    "reverse_group_2": ("reverse_group_2",),
    "reverse_group_3": ("reverse_group_3",),
    "reverse_group_4": ("reverse_group_4",),
    "reverse_grouped": ("reverse_group_1", "reverse_group_2", "reverse_group_3", "reverse_group_4"),
    "forward_reference": ("forward_reference",),
    "grouped_plus_forward": ("reverse_group_1", "reverse_group_2", "reverse_group_3", "reverse_group_4", "forward_reference"),
}


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_dirs() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    CONFIG.mkdir(parents=True, exist_ok=True)
    CAPTURE.mkdir(parents=True, exist_ok=True)


def contract_for(qid: str, split: str) -> dict[str, Any]:
    result = json.loads((V9_ROOTS[split] / qid / "live_result.json").read_text(encoding="utf-8"))
    return ((result.get("contract_result") or {}).get("parsed") or {}).get("contract") or {}


async def execute_qcf(
    record: dict[str, Any], dsfr: dict[str, Any], split: str, dataset_name: str, mode: str,
) -> tuple[dict[str, Any], EvaluationSession]:
    qid = record["qid"]
    path = CAPTURE / dataset_name / "queries" / qid / "snapshot.json"
    session = EvaluationSession.create(dataset_name, qid, source_cutoff(record), mode, path)
    top20 = baseline_top20_map([record])[qid]
    pool = candidate_pool(record, dsfr)
    contract = contract_for(qid, split)
    started = time.perf_counter()
    with evaluation_session(session):
        result = await retrieve_seed_conditioned_reverse_frontier(
            record["question"], understanding_for(record), contract, top20, pool
        )
    result["elapsed_seconds"] = time.perf_counter() - started
    if mode == "live":
        session.save({"source_commit": git_head(), "schema": "scgq-v15-1"})
        path.parent.mkdir(parents=True, exist_ok=True)
        (path.parent / "live_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return result, session


def projection(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "seeds": [(row.get("openalex_id"), row.get("title")) for row in result.get("seeds") or []],
        "queries": result.get("queries"),
        "lanes": {
            lane: [(row.get("openalex_id"), row.get("doi"), row.get("title")) for row in papers]
            for lane, papers in (result.get("lanes") or {}).items()
        },
        "errors": result.get("errors"),
    }


async def capture_dataset(
    records: Sequence[dict[str, Any]], dsfr: dict[str, dict[str, Any]], split: str, dataset_name: str,
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
            live, live_session = await execute_qcf(record, dsfr[qid], split, dataset_name, "live")
        replay, _ = await execute_qcf(record, dsfr[qid], split, dataset_name, "replay")
        identical = json_hash(projection(live)) == json_hash(projection(replay))
        results[qid] = live
        calls = list(live_session.calls.values())
        manifest.append({
            "query_id": qid,
            "seed_count": len(live.get("seeds") or []),
            "candidate_count": len(live.get("candidates") or []),
            "api_calls": sum(str(call.get("label") or "").startswith(("qcf", "scgq")) for call in calls),
            "elapsed_seconds": live.get("elapsed_seconds"),
            "errors": live.get("errors"),
            "replay_identical": identical,
            "snapshot_sha256": sha(snapshot_path),
        })
        print(
            f"[{index}/{len(records)}] {qid} seeds={len(live.get('seeds') or [])} "
            f"candidates={len(live.get('candidates') or [])} replay={'OK' if identical else 'DIFF'}",
            flush=True,
        )
    payload = {
        "dataset": dataset_name,
        "query_count": len(records),
        "source_commit": git_head(),
        "replay_all_identical": all(row["replay_identical"] for row in manifest),
        "queries": manifest,
    }
    (CAPTURE / dataset_name / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"manifest": payload, "results": results}


def lane_papers(result: dict[str, Any], lanes: Sequence[str]) -> list[dict[str, Any]]:
    return fuse_candidates([
        paper for lane in lanes for paper in ((result.get("lanes") or {}).get(lane) or [])
    ])


def evaluate_profile(
    records: Sequence[dict[str, Any]],
    dsfr: dict[str, dict[str, Any]],
    results: dict[str, dict[str, Any]],
    name: str,
    lanes: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    baseline_map: dict[str, list[dict[str, Any]]] = {}
    candidate_map: dict[str, list[dict[str, Any]]] = {}
    frontier_map: dict[str, list[dict[str, Any]]] = {}
    details: list[dict[str, Any]] = []
    total_calls = total_seconds = 0.0
    for record in records:
        qid = record["qid"]
        base = candidate_pool(record, dsfr[qid])
        frontier = lane_papers(results[qid], lanes)
        merged = fuse_candidates([*base, *frontier])
        baseline_map[qid], candidate_map[qid], frontier_map[qid] = base, merged, frontier
        base_gold, merged_gold = gold_keys(base, record), gold_keys(merged, record)
        gained = merged_gold - base_gold
        matched_titles = [
            paper.get("title") for paper in frontier
            if any(match_identity(paper, target) for target in __import__(
                "scripts.audit_evaluation", fromlist=["gold_papers"]
            ).gold_papers(record))
        ]
        calls = len(lanes)
        total_calls += calls
        total_seconds += float(results[qid].get("elapsed_seconds") or 0)
        details.append({
            "query_id": qid,
            "baseline_candidate_tp": len(base_gold),
            "candidate_tp": len(merged_gold),
            "candidate_gain": len(gained),
            "frontier_count": len(frontier),
            "frontier_gold_count": len(gold_keys(frontier, record)),
            "frontier_gold_titles": " | ".join(str(title) for title in matched_titles),
            "lanes": "|".join(lanes),
        })
    baseline = aggregate_metrics(records, baseline_map)
    candidate = aggregate_metrics(records, candidate_map)
    frontier_metrics = aggregate_metrics(records, frontier_map)
    metrics = {
        "configuration": name,
        "lanes": "|".join(lanes),
        "baseline_candidate_TP": baseline["TP"],
        "candidate_TP": candidate["TP"],
        "candidate_FP": candidate["FP"],
        "candidate_FN": candidate["FN"],
        "candidate_recall": candidate["recall"],
        "candidate_gain": candidate["TP"] - baseline["TP"],
        "frontier_TP": frontier_metrics["TP"],
        "frontier_FP": frontier_metrics["FP"],
        "frontier_precision": frontier_metrics["precision"],
        "mean_frontier_count": sum(len(rows) for rows in frontier_map.values()) / len(records),
        "api_per_query": total_calls / len(records),
        "elapsed_seconds_per_query": total_seconds / len(records),
    }
    return metrics, details, candidate_map


def choose_profile(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda row: (
        row["candidate_TP"], row["candidate_recall"],
        -row["api_per_query"], -row["mean_frontier_count"], row["configuration"],
    ))


async def run_split(split: str) -> dict[str, Any]:
    ensure_dirs()
    records = records_for(split)
    dsfr, _ = load_inputs(records, split)
    dataset_name = "calibration40_v15" if split == "calibration" else "dev20_v15"
    capture = await capture_dataset(records, dsfr, split, dataset_name)
    if not capture["manifest"]["replay_all_identical"]:
        raise RuntimeError(f"{split} replay mismatch")
    # Reuse the v11 frozen head-weighted forward lane. v15 changes only
    # reverse graph-query construction; imported files are immutable captures.
    for record in records:
        qid = record["qid"]
        prior = json.loads((V11_FORWARD_ROOTS[split] / qid / "live_result.json").read_text(encoding="utf-8"))
        capture["results"][qid].setdefault("lanes", {})["forward_reference"] = (
            (prior.get("lanes") or {}).get("forward_reference") or []
        )
    if split == "calibration":
        rows, details_by = [], {}
        for name, lanes in LANE_PROFILES.items():
            metrics, details, _ = evaluate_profile(records, dsfr, capture["results"], name, lanes)
            rows.append(metrics); details_by[name] = details
        chosen = choose_profile(rows)
        write_csv(REPORT / "01_calibration_lane_ablation.csv", rows)
        write_csv(REPORT / "02_calibration_candidate_gain.csv", details_by[chosen["configuration"]])
        gate = chosen["candidate_gain"] >= 3 and chosen["api_per_query"] <= 5
        freeze = {
            "schema_version": 1,
            "source_commit": git_head(),
            "prompt_schema": "scgq-v15-1",
            "chosen": chosen,
            "lanes": list(LANE_PROFILES[chosen["configuration"]]),
            "calibration_gate_passed": gate,
            "dev_evaluated": False,
        }
        (CONFIG / "calibration_freeze.json").write_text(
            json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return freeze
    freeze = json.loads((CONFIG / "calibration_freeze.json").read_text(encoding="utf-8"))
    if not freeze["calibration_gate_passed"]:
        state = {"status": "rejected_on_calibration", "dev_live_runs": 0}
        (REPORT / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        return state
    metrics, details, _ = evaluate_profile(
        records, dsfr, capture["results"], "QCF_frozen", freeze["lanes"]
    )
    write_csv(REPORT / "03_dev20_candidate_recall.csv", [metrics])
    write_csv(REPORT / "04_dev20_candidate_gain.csv", details)
    accepted = metrics["candidate_gain"] >= 3 and metrics["api_per_query"] <= 5
    state = {
        "schema_version": 1,
        "status": "retrieval_candidate_v15" if accepted else "rejected_on_dev",
        "branch": "seed-conditioned-citation-v15",
        "source_commit": freeze["source_commit"],
        "dev_live_runs": 1,
        "replay_identical": capture["manifest"]["replay_all_identical"],
        "metrics": metrics,
        "accepted_retrieval_component": accepted,
        "final_selection_evaluated": False,
    }
    (REPORT / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return state


def write_method() -> dict[str, Any]:
    ensure_dirs()
    text = """# SCGQ v15?seed-conditioned graph query

## ??

v14 ? `research_topic + method + task` ??? OpenAlex ?? lexical search ?????SCGQ ???????????????? seed title ? graph query ????????

1. ? `research_topic` ?????????????????
2. ??? 12 ? seed title ???????
3. ??? topic ???????? seed title?? topic ????????????????? 6 ???
4. ?? `unsupervised segmentation classification` ? seed ????? `unsupervised semantic segmentation`??? `semantic` ????????????
5. ??? 10-seed group ???? citing search?forward lane ???? v11 ???? head-weighted snapshot?

???? query planning ?????????????????????????????????????? gold???? LLM ???
"""
    (REPORT / "00_method.md").write_text(text, encoding="utf-8")
    return {"written": str(REPORT / "00_method.md")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("method", "calibration", "dev"))
    args = parser.parse_args()
    if args.phase == "method":
        result = write_method()
    else:
        result = asyncio.run(run_split("calibration" if args.phase == "calibration" else "dev"))
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
