"""Run a small, reproducible AutoScholarQuery retrieval benchmark.

This runner compares a single-pass OpenAlex baseline with the complete
YanwenZhiso pipeline. Results are checkpointed after every query.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.evaluation import macro_average
from backend.paper_identity import match_identity
from backend.main import SearchRequest, search
from backend.openalex_client import search_openalex
from backend import response_cache


def norm_title(value: str) -> str:
    value = value.replace("\\n", " ")
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def arxiv_id(value: str) -> str:
    match = re.search(r"(?:abs/|pdf/|arxiv:)?(\d{4}\.\d{4,5})(?:v\d+)?", value or "", re.I)
    return match.group(1) if match else ""


def paper_metrics(predicted: list[dict], gold_titles: list[str], gold_ids: list[str], k: int = 20) -> dict[str, float]:
    gold = [
        {"title": title, "arxiv_id": gold_ids[i] if i < len(gold_ids) else ""}
        for i, title in enumerate(gold_titles)
    ]
    matched: set[int] = set()
    hits: list[int] = []
    for paper in predicted[:k]:
        found = None
        for i, item in enumerate(gold):
            if i in matched:
                continue
            if match_identity(paper, item):
                found = i
                break
        hits.append(1 if found is not None else 0)
        if found is not None:
            matched.add(found)
    tp = len(matched)
    precision = tp / len(predicted[:k]) if predicted[:k] else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hits))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return {f"precision@{k}": round(precision, 6), f"recall@{k}": round(recall, 6),
            f"f1@{k}": round(f1, 6), f"ndcg@{k}": round(dcg / idcg if idcg else 0.0, 6),
            "true_positive": tp, "predicted_count": len(predicted[:k]), "relevant_count": len(gold)}


async def run(args: argparse.Namespace) -> dict:
    rows = [json.loads(x) for x in args.dataset.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = rows[args.offset: args.offset + args.limit]
    details = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows, start=1):
        query = row["question"]
        gold = row.get("answer") or []
        gold_ids = row.get("answer_arxiv_id") or []

        t0 = time.perf_counter()
        try:
            baseline_papers = await search_openalex(query, per_page=args.k)
            baseline_error = None
        except Exception as exc:
            baseline_papers, baseline_error = [], type(exc).__name__
        baseline_seconds = time.perf_counter() - t0

        t1 = time.perf_counter()
        try:
            if args.refresh:
                response_cache.invalidate(query)
            response = await search(SearchRequest(query=query))
            payload = response.model_dump()
            trace_papers = payload["papers"]
            trace_error = None
        except Exception as exc:
            payload, trace_papers, trace_error = {}, [], f"{type(exc).__name__}: {exc}"
        trace_seconds = time.perf_counter() - t1

        detail = {
            "query_id": row.get("qid") or f"offset_{args.offset + index - 1}",
            "query": query,
            "gold_titles": gold,
            "baseline_titles": [p.get("title", "") for p in baseline_papers[:args.k]],
            "trace_titles": [p.get("title", "") for p in trace_papers[:args.k]],
            "baseline": paper_metrics(baseline_papers, gold, gold_ids, args.k),
            "trace": paper_metrics(trace_papers, gold, gold_ids, args.k),
            "trace_at_5": paper_metrics(trace_papers, gold, gold_ids, 5),
            "trace_at_10": paper_metrics(trace_papers, gold, gold_ids, 10),
            "baseline_seconds": round(baseline_seconds, 3),
            "trace_seconds": round(trace_seconds, 3),
            "trace_runtime_metrics": payload.get("metrics", {}),
            "baseline_error": baseline_error,
            "trace_error": trace_error,
        }
        details.append(detail)
        report = build_report(args, details)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{index}/{len(rows)}] {detail['query_id']} baseline_f1={detail['baseline'][f'f1@{args.k}']:.4f} trace_f1={detail['trace'][f'f1@{args.k}']:.4f} trace_s={trace_seconds:.1f}", flush=True)
    return build_report(args, details)


def build_report(args: argparse.Namespace, details: list[dict]) -> dict:
    baseline_rows = [x["baseline"] for x in details]
    trace_rows = [x["trace"] for x in details]
    trace_rows_5 = [x["trace_at_5"] for x in details]
    trace_rows_10 = [x["trace_at_10"] for x in details]
    return {
        "benchmark": "AutoScholarQuery_test",
        "dataset": str(args.dataset),
        "offset": args.offset,
        "requested_limit": args.limit,
        "completed": len(details),
        "k": args.k,
        "matching": "stable IDs (arXiv/DOI/OpenAlex/S2) then exact normalized title",
        "baseline_name": "single-pass OpenAlex raw query",
        "baseline": macro_average(baseline_rows),
        "trace_rank": macro_average(trace_rows),
        "trace_rank_at_5": macro_average(trace_rows_5),
        "trace_rank_at_10": macro_average(trace_rows_10),
        "mean_baseline_seconds": round(sum(x["baseline_seconds"] for x in details) / len(details), 3) if details else 0,
        "mean_trace_seconds": round(sum(x["trace_seconds"] for x in details) / len(details), 3) if details else 0,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\AICompetition\references\github\SPAR\benchmark\AutoScholarQuery_test.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/autoscholar_pilot.json"))
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    report = asyncio.run(run(args))
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
