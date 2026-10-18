"""Create a deterministic holdout manifest without exposing per-query gold."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_evaluation import load_jsonl, source_cutoff


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\AICompetition\references\github\SPAR\benchmark\AutoScholarQuery_test.jsonl"))
    parser.add_argument("--exclude-first", type=int, default=20)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--manifest", type=Path, default=ROOT / "reports/overnight_iteration/validation30_manifest.json")
    parser.add_argument("--private-subset", type=Path, default=ROOT / "reports/frozen_eval/query_sets/validation30_private.jsonl")
    args = parser.parse_args()
    rows = load_jsonl(args.dataset)
    excluded = {row["qid"] for row in rows[:args.exclude_first]}
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["qid"] in excluded or source_cutoff(row) is None:
            continue
        gold_count = len(row.get("answer") or [])
        gold_bin = "g1" if gold_count <= 1 else "g2" if gold_count == 2 else "g3plus"
        length = len(str(row.get("question") or "").split())
        length_bin = "short" if length < 16 else "long"
        strata[(gold_bin, length_bin)].append(row)
    rng = random.Random(args.seed)
    for values in strata.values():
        rng.shuffle(values)
    selected = []
    keys = sorted(strata)
    while len(selected) < args.count:
        progressed = False
        for key in keys:
            if strata[key] and len(selected) < args.count:
                selected.append(strata[key].pop()); progressed = True
        if not progressed:
            raise RuntimeError("not enough eligible rows")
    # Randomize evaluation order after stratified selection.
    rng.shuffle(selected)
    args.private_subset.parent.mkdir(parents=True, exist_ok=True)
    with args.private_subset.open("w", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    public_queries = []
    for row in selected:
        cutoff = source_cutoff(row)
        assert cutoff is not None
        public_queries.append({"query_id": row["qid"], "query": row["question"], "cutoff": cutoff.isoformat()})
    manifest = {
        "query_set": f"validation{args.count}", "count": args.count, "sampling_seed": args.seed,
        "dataset": str(args.dataset), "dataset_sha256": sha256(args.dataset),
        "excluded_query_set": "dev20", "excluded_ids": sorted(excluded),
        "stratification": ["gold_count_bin", "query_length_bin"],
        "gold_visibility_policy": "Per-query gold is stored only in the ignored private subset and must not be inspected during iteration.",
        "private_subset_sha256": sha256(args.private_subset), "queries": public_queries,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    config_path = ROOT / "configs/query_sets.json"
    config = {"dev20": [row["qid"] for row in rows[:20]], f"validation{args.count}": [row["qid"] for row in selected]}
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(args.manifest), "query_set": manifest["query_set"], "count": args.count,
                      "dataset_sha256": manifest["dataset_sha256"], "private_subset_sha256": manifest["private_subset_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
