"""Capture the bounded production-query additions for core-rescue v6."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_evaluation import load_jsonl, source_cutoff
from scripts.run_frozen_official_benchmark import (
    deterministic_projection, execute_query, json_hash, legal_for_cutoff, sha256_file,
)


DATASET = Path(r"D:\AICompetition\references\github\SPAR\benchmark\AutoScholarQuery_test.jsonl")
SPLIT = ROOT / "reports/f1_selector_v3/02_selector_split_manifest.json"
EXISTING = ROOT / "reports/joint_optimization_v4/07_production_split_manifest.json"
QUERY_SET = ROOT / "reports/frozen_eval/query_sets/core_rescue_v6_additions_private.jsonl"
RUN_DIR = ROOT / "reports/frozen_eval/core_rescue_v6_production_lean"
REPORT_MANIFEST = ROOT / "reports/core_rescue_v6/01_production_manifest.json"
SEED = 20260830


def _tokens(value: str) -> list[str]:
    return [part for part in value.lower().replace("/", " ").replace("-", " ").split() if part]


def stratum(item: dict[str, Any], row: dict[str, Any]) -> tuple[str, str, str, str]:
    query = str(row.get("question") or "")
    length = "long" if len(_tokens(query)) >= 12 else "short"
    constrained = "constrained" if any(word in query.lower() for word in (
        "using", "with", "dataset", "method", "task", "based on", "under", "for ", "on ",
    )) else "open"
    return str(item.get("gold_bin")), str(item.get("intent")), length, constrained


def stratified_additions(
    items: list[dict[str, Any]], existing: set[str], rows: dict[str, dict[str, Any]], count: int, seed: int,
) -> list[str]:
    rng = random.Random(seed)
    buckets: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    for item in items:
        qid = str(item["query_id"])
        if qid not in existing:
            buckets[stratum(item, rows[qid])].append(qid)
    for values in buckets.values(): rng.shuffle(values)
    selected: list[str] = []
    keys = sorted(buckets)
    while len(selected) < count:
        progressed = False
        for key in keys:
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].pop()); progressed = True
        if not progressed:
            raise RuntimeError(f"not enough eligible queries for {count} additions")
    return selected


def prepare_query_set() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    existing = json.loads(EXISTING.read_text(encoding="utf-8"))
    dataset_rows = load_jsonl(DATASET); by_id = {row["qid"]: row for row in dataset_rows}
    train_existing = set(existing["train_query_ids"]); calibration_existing = set(existing["calibration_query_ids"])
    train_add = stratified_additions(split["selector_train"], train_existing, by_id, 70, SEED + 1)
    calibration_add = stratified_additions(split["selector_calibration"], calibration_existing, by_id, 20, SEED + 2)
    selected = []
    for qid in train_add:
        selected.append({**by_id[qid], "_v6_split": "train"})
    for qid in calibration_add:
        selected.append({**by_id[qid], "_v6_split": "calibration"})
    QUERY_SET.parent.mkdir(parents=True, exist_ok=True)
    with QUERY_SET.open("w", encoding="utf-8") as handle:
        for row in selected: handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {
        "seed": SEED, "train_existing": sorted(train_existing), "calibration_existing": sorted(calibration_existing),
        "train_additions": train_add, "calibration_additions": calibration_add,
        "train_total": len(train_existing) + len(train_add), "calibration_total": len(calibration_existing) + len(calibration_add),
        "query_set": str(QUERY_SET), "query_set_sha256": sha256_file(QUERY_SET),
    }
    return selected, metadata


async def capture_one(row: dict[str, Any], semaphore: asyncio.Semaphore, index: int, total: int) -> dict[str, Any]:
    qid = row["qid"]; cutoff = source_cutoff(row)
    if cutoff is None: raise ValueError(f"missing cutoff: {qid}")
    query_dir = RUN_DIR / "queries" / qid
    snapshot = query_dir / "snapshot.json"; live_path = query_dir / "live_response.json"
    async with semaphore:
        if snapshot.exists() and live_path.exists():
            live = json.loads(live_path.read_text(encoding="utf-8"))
            print(f"[{index}/{total}] {qid} reuse LIVE", flush=True)
        else:
            live, _, _ = await execute_query(row, "AutoScholarQuery_core_rescue_v6", RUN_DIR, "live", "balanced", "lean")
            print(f"[{index}/{total}] {qid} LIVE complete", flush=True)
        replay, _, _ = await execute_query(row, "AutoScholarQuery_core_rescue_v6", RUN_DIR, "replay", "balanced", "lean")
    live_hash = json_hash(deterministic_projection(live)); replay_hash = json_hash(deterministic_projection(replay))
    legal = [legal_for_cutoff(paper, cutoff)[0] for paper in live.get("papers") or []]
    return {
        "query_id": qid, "split": row["_v6_split"], "cutoff": cutoff.isoformat(),
        "snapshot_sha256": sha256_file(snapshot), "live_result_hash": live_hash, "replay_result_hash": replay_hash,
        "replay_identical": live_hash == replay_hash, "all_final_papers_date_legal": all(legal),
        "top20_count": len(live.get("papers") or []),
    }


async def run(concurrency: int) -> dict[str, Any]:
    rows, metadata = prepare_query_set(); RUN_DIR.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [capture_one(row, semaphore, index, len(rows)) for index, row in enumerate(rows, 1)]
    details = await asyncio.gather(*tasks)
    manifest = {
        "schema_version": 1, "profile": "lean", "search_mode": "balanced", "concurrency": concurrency,
        **metadata, "queries": details,
        "replay_all_identical": all(row["replay_identical"] for row in details),
        "all_final_papers_date_legal": all(row["all_final_papers_date_legal"] for row in details),
        "gold_online_features": False,
    }
    REPORT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (RUN_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args(); result = asyncio.run(run(max(1, min(args.concurrency, 3))))
    print(json.dumps({key: result[key] for key in (
        "train_total", "calibration_total", "replay_all_identical", "all_final_papers_date_legal", "query_set_sha256"
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
