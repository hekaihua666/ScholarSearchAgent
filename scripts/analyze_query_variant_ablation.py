"""Measure per-variant and cumulative recall from a frozen query-portfolio run."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.paper_identity import match_identity
from backend.trace_rank import diversify, fuse_candidates, score_candidates
from scripts.audit_evaluation import aggregate, gold_papers, load_jsonl, metrics_from_papers, source_cutoff
from scripts.run_frozen_official_benchmark import legalize, output_papers, snapshot_entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\AICompetition\references\github\SPAR\benchmark\AutoScholarQuery_test.jsonl"))
    args = parser.parse_args()
    dataset = {row["qid"]: row for row in load_jsonl(args.dataset)}
    accum: dict[str, dict[str, Any]] = defaultdict(lambda: {"candidate": [], "top20": [], "counts": [], "gold": set(), "queries": 0})
    for query_dir in sorted((args.run_dir / "queries").glob("AutoScholarQuery_*")):
        qid = query_dir.name
        if qid not in dataset:
            continue
        response = json.loads((query_dir / "live_response.json").read_text(encoding="utf-8"))
        snapshot = json.loads((query_dir / "snapshot.json").read_text(encoding="utf-8"))
        understanding = response.get("query_understanding") or {}
        portfolio = understanding.get("query_portfolio") or []
        cutoff = source_cutoff(dataset[qid]); assert cutoff is not None
        gold = gold_papers(dataset[qid])
        by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for index, item in enumerate(portfolio):
            variant_id = str(item.get("variant_id") or f"Q{index}")
            form = str(item.get("form") or "unknown")
            key = f"{variant_id}:{form}"
            for prefix in (f"first_lexical_{index}", f"first_arxiv_{index}"):
                for entry in snapshot_entries(snapshot, prefix):
                    by_variant[key].extend(legalize(output_papers(entry), cutoff))
            if form == "pseudo_document":
                for entry in snapshot_entries(snapshot, "first_semantic"):
                    by_variant[key].extend(legalize(output_papers(entry), cutoff))
        cumulative: list[dict[str, Any]] = []
        for key, papers in by_variant.items():
            fused = fuse_candidates(papers)
            ranked = score_candidates(copy.deepcopy(fused), understanding)
            top20 = diversify(copy.deepcopy(ranked), limit=20, lambda_relevance=0.88)
            bucket = accum[key]
            bucket["candidate"].append(metrics_from_papers(fused, gold, len(fused)))
            bucket["top20"].append(metrics_from_papers(top20, gold, 20))
            bucket["counts"].append(len(fused)); bucket["queries"] += 1
            for target_index, target in enumerate(gold):
                if any(match_identity(paper, target) for paper in fused):
                    bucket["gold"].add(f"{qid}:{target_index}")
            cumulative.extend(papers)
            ckey = f"CUMULATIVE_THROUGH_{key}"
            cfused = fuse_candidates(cumulative)
            cbucket = accum[ckey]
            cbucket["candidate"].append(metrics_from_papers(cfused, gold, len(cfused)))
            cbucket["counts"].append(len(cfused)); cbucket["queries"] += 1
            for target_index, target in enumerate(gold):
                if any(match_identity(paper, target) for paper in cfused):
                    cbucket["gold"].add(f"{qid}:{target_index}")
    rows = []
    for name, bucket in sorted(accum.items()):
        candidate = aggregate(bucket["candidate"])
        top20 = aggregate(bucket["top20"]) if bucket["top20"] else {"macro": {"recall": None, "f1": None}}
        rows.append({
            "variant": name,
            "queries": bucket["queries"],
            "mean_candidate_count": round(sum(bucket["counts"]) / max(len(bucket["counts"]), 1), 3),
            "unique_gold": len(bucket["gold"]),
            "candidate_recall_macro": candidate["macro"]["recall"],
            "candidate_recall_micro": candidate["micro"]["recall"],
            "top20_recall_macro": top20["macro"]["recall"],
            "top20_f1_macro": top20["macro"]["f1"],
        })
    output = ROOT / "reports/joint_optimization_v4/05_query_variant_ablation.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
