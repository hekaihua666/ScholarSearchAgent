"""Explain fixed LLM-0.90 FP/FN on the dev frozen candidate pool."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/joint_optimization_v4"
DATASET = Path(r"D:\AICompetition\references\github\SPAR\benchmark\AutoScholarQuery_test.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.open(encoding="utf-8") if line.strip()]
    dataset = {row["qid"]: row for row in map(json.loads, DATASET.open(encoding="utf-8"))}
    groups = defaultdict(list)
    for row in rows:
        groups[row["query_id"]].append(row)
    cases = []
    fp_counts = Counter(); fn_counts = Counter()
    for qid, candidates in groups.items():
        live = [row for row in candidates if row.get("live_rank") and int(row["live_rank"]) <= 20]
        selected = [row for row in live if float(row.get("llm_relevance_score") or 0) >= 0.90]
        for row in selected:
            if not int(row["label"]):
                category = row.get("negative_type") or "top20_hard_negative"
                fp_counts[category] += 1
                cases.append({"query_id": qid, "error": "FP", "category": category, "title": row["title"], "live_rank": row.get("live_rank"), "llm_score": row.get("llm_relevance_score"), "candidate_rank": row.get("candidate_rank")})
        positive = [row for row in candidates if int(row["label"])]
        selected_ids = {row["paper_id"] for row in selected if int(row["label"])}
        for row in positive:
            if row["paper_id"] in selected_ids:
                continue
            if not row.get("abstract"):
                category = "missing_abstract"
            elif not row.get("live_rank") or int(row["live_rank"]) > 20:
                category = "ranking_below_shortlist"
            elif float(row.get("llm_relevance_score") or 0) < 0.90:
                category = "llm_overconservative"
            else:
                category = "identity_error"
            fn_counts[category] += 1
            cases.append({"query_id": qid, "error": "FN", "category": category, "title": row["title"], "live_rank": row.get("live_rank"), "llm_score": row.get("llm_relevance_score"), "candidate_rank": row.get("candidate_rank")})
        absent = len(dataset[qid].get("answer") or []) - len(positive)
        if absent > 0:
            fn_counts["candidate_absent"] += absent
            cases.append({"query_id": qid, "error": "FN", "category": "candidate_absent", "title": f"{absent} gold paper(s)", "live_rank": "", "llm_score": "", "candidate_rank": ""})
    csv_path = OUT / "09_fp_fn_cases.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cases[0]))
        writer.writeheader(); writer.writerows(cases)
    report = f"""# Selector 错误分类

对象：dev20 当前 LIVE candidate pool，固定 LLM threshold=0.90。

## FP

`{dict(fp_counts)}`

## FN

`{dict(fn_counts)}`

candidate-absent 与 Selector 无关；ranking-below-shortlist 属于排序/候选截断；LLM-overconservative 才是当前阈值直接造成的 Selector FN。逐例见 `09_fp_fn_cases.csv`。
"""
    (OUT / "09_selector_error_taxonomy.md").write_text(report, encoding="utf-8")
    print(json.dumps({"fp": fp_counts, "fn": fn_counts}, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
