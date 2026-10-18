"""Materialize selector pairs from real frozen production candidate generators."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.paper_identity import match_identity
from backend.query_portfolio import atomicize_query
from backend.selection.atomic_constraint_selector import judge_paper
from backend.trace_rank import canonical_key, fuse_candidates, score_candidates
from scripts.audit_evaluation import gold_papers, load_jsonl, source_cutoff
from scripts.run_frozen_official_benchmark import module_candidates


def stage(audit: dict[str, Any], name: str) -> list[dict[str, Any]]:
    for item in audit.get("stages") or []:
        if item.get("name") == name:
            return item.get("papers") or []
    return []


def negative_type(query: str, paper: dict[str, Any], ace: dict[str, Any], rank: int) -> str:
    title = str(paper.get("title") or "").lower()
    source_query = str(paper.get("source_query") or "")
    if not paper.get("abstract"):
        return "missing_abstract"
    if source_query.startswith("citation:"):
        return "citation_drift"
    asks_survey = any(word in query.lower() for word in ("survey", "review", "overview"))
    is_survey = any(word in title for word in ("survey", "review", "overview"))
    if asks_survey != is_survey and (asks_survey or is_survey):
        return "survey_vs_original"
    if any(item.get("type") == "relation" and item.get("state") == "CONTRADICTED" for item in ace["criteria"]):
        return "relation_mismatch"
    if ace["hard_gate"] == "fail":
        return "metadata_constraint_fail"
    if float(paper.get("lexical_score") or 0) >= 0.35 and ace["required_unknown_ratio"] >= 0.5:
        return "partial_constraint_match"
    if rank <= 20:
        return "top20_hard_negative"
    return "production_weak_negative"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\AICompetition\references\github\SPAR\benchmark\AutoScholarQuery_test.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--allowed-query-ids", type=Path)
    args = parser.parse_args()
    rows = {row["qid"]: row for row in load_jsonl(args.dataset)}
    allowed = None
    if args.allowed_query_ids:
        allowed = set(json.loads(args.allowed_query_ids.read_text(encoding="utf-8")))
    pairs: list[dict[str, Any]] = []
    query_dirs = sorted((args.run_dir / "queries").glob("AutoScholarQuery_*"))
    for query_dir in query_dirs:
        qid = query_dir.name
        if qid not in rows or (allowed is not None and qid not in allowed):
            continue
        response_path = query_dir / "live_response.json"
        audit_path = query_dir / "live_audit.json"
        snapshot_path = query_dir / "snapshot.json"
        if not response_path.exists() or not audit_path.exists() or not snapshot_path.exists():
            continue
        response = json.loads(response_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        understanding = response.get("query_understanding") or {}
        # Re-materialize with the current frozen selector schema. Retrieval
        # snapshots remain unchanged; this avoids stale parser fields after a
        # parser bug fix and still uses no gold information.
        graph = atomicize_query(rows[qid]["question"], understanding)
        cutoff = source_cutoff(rows[qid])
        assert cutoff is not None
        modules = module_candidates(snapshot, cutoff)
        merged = fuse_candidates([paper for papers in modules.values() for paper in papers])
        ranked = score_candidates(merged, understanding)[: args.top_n]
        post_llm = {}
        for live_rank, live_paper in enumerate(stage(audit, "shortlist_post_llm"), 1):
            post_llm[canonical_key(live_paper)] = {**live_paper, "_live_rank": live_rank}
        gold = gold_papers(rows[qid])
        seen_gold_indices: set[int] = set()
        for rank, paper in enumerate(ranked, 1):
            llm = post_llm.get(canonical_key(paper)) or {}
            if llm.get("_live_rank"):
                paper["_live_rank"] = llm["_live_rank"]
            for key in ("llm_relevance_score", "llm_relevant", "recommendation_reason"):
                if key in llm:
                    paper[key] = llm[key]
            matched_gold = next((index for index, target in enumerate(gold) if match_identity(paper, target)), None)
            if matched_gold is not None and matched_gold in seen_gold_indices:
                # Identity aliases of one relevant paper must not become two TP
                # training rows or inflate set metrics.
                continue
            if matched_gold is not None:
                seen_gold_indices.add(matched_gold)
            label = int(matched_gold is not None)
            ace = judge_paper(paper, graph)
            ntype = "positive" if label else negative_type(rows[qid]["question"], paper, ace, rank)
            pairs.append(
                {
                    "query_id": qid,
                    "query": rows[qid]["question"],
                    "cutoff": (response.get("query_understanding") or {}).get("_evaluation_cutoff"),
                    "candidate_rank": rank,
                    "live_rank": paper.get("_live_rank"),
                    "candidate_pool_size": len(ranked),
                    "paper_id": paper.get("openalex_id") or paper.get("doi") or canonical_key(paper),
                    "title": paper.get("title"),
                    "abstract": paper.get("abstract") or "",
                    "year": paper.get("year"),
                    "doi": paper.get("doi"),
                    "arxiv_id": paper.get("arxiv_id"),
                    "source_api": paper.get("source_api"),
                    "source_round": paper.get("source_round"),
                    "trace_score": paper.get("final_score"),
                    "lexical_score": paper.get("lexical_score"),
                    "constraint_coverage_score": paper.get("constraint_coverage_score"),
                    "llm_relevance_score": paper.get("llm_relevance_score"),
                    "label": label,
                    "matched_gold_index": matched_gold,
                    "label_scope": "benchmark_gold_positive_else_unjudged_weak_negative",
                    "negative_type": ntype,
                    "ace": ace,
                    "constraint_graph": graph,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    counts = Counter(pair["negative_type"] for pair in pairs)
    stats_path = ROOT / "reports/joint_optimization_v4/07_hard_negative_stats.csv"
    with stats_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["negative_type", "count"])
        writer.writerows(sorted(counts.items()))
    queries = len({pair["query_id"] for pair in pairs})
    positives = sum(pair["label"] for pair in pairs)
    report = f"""# Production-distribution Selector 数据审计

- 冻结生成器：`{args.run_dir}`
- 查询数：{queries}
- 每查询候选上限：{args.top_n}
- query-paper pairs：{len(pairs)}
- 命中 benchmark gold：{positives}
- 其余样本不是现实世界的确定负例，标签口径为 weak/unjudged negative。
- 候选来自真实 `merged_candidates`，再用同版本 TRACE 排到 Top{args.top_n}；Top32 内回填已有 LLM 分数。
- gold 只在本脚本中用于离线标签，不进入候选特征、query plan 或生产缓存。

## Hard-negative 组成

`{dict(counts)}`
"""
    (ROOT / "reports/joint_optimization_v4/07_selector_data_audit.md").write_text(report, encoding="utf-8")
    print(json.dumps({"queries": queries, "pairs": len(pairs), "positives": positives, "negative_types": counts}, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
