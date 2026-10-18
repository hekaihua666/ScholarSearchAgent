"""Audit AutoScholarQuery evaluation without changing TRACE ranking weights.

Default mode analyzes saved v3/v4 reports and current response caches.  With
``--live-limit`` it additionally reruns a fixed prefix while collecting opt-in
stage snapshots from the unchanged search pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import response_cache
from backend.audit_trace import finish_audit, start_audit
from backend.main import SearchRequest, search
from backend.paper_identity import (
    fuzzy_title_similarity, match_identity, normalize_title, official_pasa_title,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def gold_papers(row: dict[str, Any]) -> list[dict[str, Any]]:
    titles = row.get("answer") or []
    ids = row.get("answer_arxiv_id") or []
    return [
        {"title": title, "arxiv_id": ids[i] if i < len(ids) else ""}
        for i, title in enumerate(titles)
    ]


def match_rank(papers: list[dict[str, Any]], gold: dict[str, Any]) -> tuple[int | None, str | None]:
    for rank, paper in enumerate(papers, 1):
        route = match_identity(paper, gold)
        if route:
            return rank, route
    return None, None


def metrics_from_papers(papers: list[dict[str, Any]], gold: list[dict[str, Any]], k: int) -> dict[str, float]:
    matched: set[int] = set()
    hits: list[int] = []
    for paper in papers[:k]:
        found = None
        for index, target in enumerate(gold):
            if index not in matched and match_identity(paper, target):
                found = index
                break
        hits.append(int(found is not None))
        if found is not None:
            matched.add(found)
    tp = len(matched)
    predicted = len(papers[:k])
    precision = tp / predicted if predicted else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hits))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return {"precision": precision, "recall": recall, "f1": f1,
            "ndcg": dcg / idcg if idcg else 0.0, "tp": tp,
            "predicted": predicted, "gold": len(gold)}


def aggregate(rows: list[dict[str, float]]) -> dict[str, Any]:
    if not rows:
        return {}
    macro = {
        key: round(sum(row[key] for row in rows) / len(rows), 6)
        for key in ("precision", "recall", "f1", "ndcg")
    }
    tp = sum(int(row["tp"]) for row in rows)
    predicted = sum(int(row["predicted"]) for row in rows)
    gold = sum(int(row["gold"]) for row in rows)
    p = tp / predicted if predicted else 0.0
    r = tp / gold if gold else 0.0
    return {
        "macro": macro,
        "micro": {
            "precision": round(p, 6), "recall": round(r, 6),
            "f1": round(2 * p * r / (p + r) if p + r else 0.0, 6),
            "true_positive": tp, "predicted": predicted, "gold": gold,
        },
    }


def papers_from_saved(detail: dict[str, Any], key: str) -> list[dict[str, Any]]:
    return [{"title": title} for title in detail.get(key, [])]


def analyze_saved(report_path: Path, dataset_by_qid: dict[str, dict[str, Any]]) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    details: list[dict[str, Any]] = []
    all_metrics: dict[int, list[dict[str, float]]] = defaultdict(list)
    likely_fuzzy: list[dict[str, Any]] = []
    for item in report.get("details", []):
        row = dataset_by_qid.get(item["query_id"], {
            "answer": item.get("gold_titles", []), "answer_arxiv_id": []
        })
        gold = gold_papers(row)
        papers = papers_from_saved(item, "trace_titles")
        per_k = {}
        for k in (5, 10, 20):
            value = metrics_from_papers(papers, gold, k)
            all_metrics[k].append(value)
            per_k[str(k)] = value
        ranks = []
        for target in gold:
            rank, route = match_rank(papers, target)
            ranks.append({"title": target["title"], "rank": rank, "match_route": route})
            if rank is None:
                best = max(
                    ((fuzzy_title_similarity(p.get("title"), target["title"]), p.get("title")) for p in papers),
                    default=(0.0, ""),
                )
                if best[0] >= 0.90:
                    likely_fuzzy.append({"query_id": item["query_id"], "gold": target["title"],
                                         "candidate": best[1], "similarity": round(best[0], 4)})
        details.append({"query_id": item["query_id"], "query": item["query"],
                        "metrics": per_k, "gold_ranks": ranks})
    return {
        "source": str(report_path),
        "query_count": len(details),
        "recomputed": {str(k): aggregate(values) for k, values in all_metrics.items()},
        "likely_fuzzy_title_misses": likely_fuzzy,
        "details": details,
    }


def cache_payloads() -> dict[str, dict[str, Any]]:
    result = {}
    for path in (ROOT / "cache").glob("*.json"):
        try:
            wrapper = json.loads(path.read_text(encoding="utf-8"))
            payload = wrapper.get("payload") or {}
            query = str(payload.get("query") or wrapper.get("query") or "")
            if query:
                result[query] = payload
        except Exception:
            continue
    return result


def source_cutoff(row: dict[str, Any]) -> datetime | None:
    value = str((row.get("source_meta") or {}).get("published_time") or "")
    try:
        return datetime.strptime(value, "%Y%m%d") - timedelta(days=7)
    except ValueError:
        return None


def analyze_cache(rows: list[dict[str, Any]]) -> dict[str, Any]:
    caches = cache_payloads()
    future = []
    counts = Counter()
    channel_gold = Counter()
    route_counts = Counter()
    metrics_by_k: dict[int, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        payload = caches.get(row["question"])
        if not payload:
            continue
        papers = payload.get("papers") or []
        gold = gold_papers(row)
        counts["queries_with_cache"] += 1
        for k in (5, 10, 20):
            metrics_by_k[k].append(metrics_from_papers(papers, gold, k))
        cutoff = source_cutoff(row)
        for rank, paper in enumerate(papers, 1):
            year = paper.get("year")
            if cutoff and year and int(year) > cutoff.year:
                future.append({"qid": row.get("qid"), "rank": rank, "year": year,
                               "cutoff": cutoff.date().isoformat(), "title": paper.get("title")})
        for target in gold:
            rank, route = match_rank(papers, target)
            if rank is None:
                continue
            route_counts[route or "unknown"] += 1
            paper = papers[rank - 1]
            hits = paper.get("retrieval_hits") or []
            sources = {str(hit.get("source") or "unknown") for hit in hits}
            rounds = {str(hit.get("round") or "unknown") for hit in hits}
            for source in sources:
                channel_gold[f"source:{source}"] += 1
            for round_name in rounds:
                channel_gold[f"round:{round_name}"] += 1
    return {
        "cache_query_count": counts["queries_with_cache"],
        "metrics": {str(k): aggregate(values) for k, values in metrics_by_k.items()},
        "match_routes": dict(route_counts),
        "gold_channel_presence_nonexclusive": dict(channel_gold),
        "future_dated_top20_count": len(future),
        "future_dated_top20_examples": future[:100],
    }


def stage_map(audit: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {stage["name"]: stage.get("papers") or [] for stage in audit.get("stages", [])}


def diagnose_live(row: dict[str, Any], audit: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    stages = stage_map(audit)
    order = ["first_raw", "first_fused_ranked", "post_citation", "second_raw",
             "merged_candidates", "shortlist_pre_llm", "shortlist_post_llm", "final_top20"]
    diagnostics = []
    loss_counts = Counter()
    source_contributions = Counter()
    for target in gold_papers(row):
        found = {}
        for name in order:
            rank, route = match_rank(stages.get(name, []), target)
            found[name] = {"rank": rank, "route": route}
        if found["final_top20"]["rank"] is not None:
            classification = "hit_top20"
        elif found["shortlist_pre_llm"]["rank"] is not None and found["shortlist_post_llm"]["rank"] is None:
            classification = "llm_rerank_loss"
        elif found["merged_candidates"]["rank"] is not None:
            classification = "ranking_outside_top20"
        elif found["first_raw"]["rank"] is not None:
            classification = "fusion_loss"
        elif found["post_citation"]["rank"] is not None:
            classification = "citation_retrieval_only"
        elif found["second_raw"]["rank"] is not None:
            classification = "second_round_retrieval_only"
        else:
            classification = "not_retrieved"
        loss_counts[classification] += 1
        for stage_name in ("first_raw", "second_raw", "post_citation"):
            rank = found[stage_name]["rank"]
            if rank is not None:
                paper = stages[stage_name][rank - 1]
                source_contributions[str(paper.get("source_api") or stage_name)] += 1
        diagnostics.append({"gold": target, "classification": classification, "stages": found})
    return {
        "query_id": row.get("qid"), "query": row["question"],
        "gold_diagnostics": diagnostics,
        "loss_counts": dict(loss_counts),
        "source_contributions_nonexclusive": dict(source_contributions),
        "runtime_metrics": payload.get("metrics", {}),
        "audit_total_seconds": audit.get("total_seconds"),
        "stage_checkpoints": [
            {k: stage.get(k) for k in ("name", "elapsed_seconds", "count", "metadata")}
            for stage in audit.get("stages", [])
        ],
    }


async def run_live(rows: list[dict[str, Any]], limit: int, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    details = []
    totals = Counter()
    caches = cache_payloads()
    for index, row in enumerate(rows[:limit], 1):
        audit_path = output_dir / f"{row['qid']}.audit.json"
        payload = caches.get(row["question"])
        error = None
        if audit_path.exists() and payload:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
        else:
            response_cache.invalidate(row["question"])
            token = start_audit(row["question"])
            try:
                response = await search(SearchRequest(query=row["question"], mode="balanced"))
                payload = response.model_dump()
            except Exception as exc:
                payload = {}
                error = f"{type(exc).__name__}: {exc}"
            finally:
                audit = finish_audit(token)
            audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        if not payload:
            details.append({"query_id": row.get("qid"), "query": row["question"],
                            "error": error or "missing cached payload", "stage_checkpoints": [
                                {k: stage.get(k) for k in ("name", "elapsed_seconds", "count", "metadata")}
                                for stage in audit.get("stages", [])
                            ]})
            print(f"[{index}/{limit}] {row['qid']} ERROR {error}", flush=True)
            continue
        detail = diagnose_live(row, audit, payload)
        details.append(detail)
        totals.update(detail["loss_counts"])
        print(f"[{index}/{limit}] {row['qid']} {detail['loss_counts']} {detail['audit_total_seconds']:.1f}s", flush=True)
    return {"query_count": len(details), "successful_query_count": sum("error" not in x for x in details),
            "gold_outcome_counts": dict(totals), "details": details}


def dataset_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    title_keys, id_keys = Counter(), Counter()
    empty = 0
    for row in rows:
        gold = gold_papers(row)
        empty += int(not gold)
        title_keys.update(official_pasa_title(p["title"]) for p in gold)
        id_keys.update(str(p.get("arxiv_id") or "") for p in gold if p.get("arxiv_id"))
    return {
        "queries": len(rows),
        "gold_papers": sum(len(gold_papers(row)) for row in rows),
        "mean_gold_per_query": round(sum(len(gold_papers(row)) for row in rows) / len(rows), 4),
        "empty_gold_queries": empty,
        "duplicate_gold_title_keys": sum(1 for count in title_keys.values() if count > 1),
        "duplicate_gold_arxiv_ids": sum(1 for count in id_keys.values() if count > 1),
        "missing_gold_arxiv_ids": sum(
            max(0, len(row.get("answer") or []) - len(row.get("answer_arxiv_id") or [])) for row in rows
        ),
    }


def compare_versions(v3: dict[str, Any], v4: dict[str, Any]) -> list[dict[str, Any]]:
    left = {row["query_id"]: row for row in v3["details"]}
    right = {row["query_id"]: row for row in v4["details"]}
    changes = []
    for qid in sorted(left.keys() & right.keys()):
        m3 = left[qid]["metrics"]["20"]
        m4 = right[qid]["metrics"]["20"]
        if any(abs(m4[key] - m3[key]) > 1e-12 for key in ("tp", "recall", "ndcg")):
            changes.append({"query_id": qid, "query": right[qid]["query"],
                            "v3": m3, "v4": m4,
                            "v3_ranks": left[qid]["gold_ranks"], "v4_ranks": right[qid]["gold_ranks"]})
    return changes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\AICompetition\references\github\SPAR\benchmark\AutoScholarQuery_test.jsonl"))
    parser.add_argument("--v3", type=Path, default=ROOT / "reports/autoscholar_trace_v3_semantic_20.json")
    parser.add_argument("--v4", type=Path, default=ROOT / "reports/autoscholar_trace_v4_intent_semantic_20.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/evaluation_audit.json")
    parser.add_argument("--live-limit", type=int, default=0)
    args = parser.parse_args()
    rows = load_jsonl(args.dataset)
    by_qid = {row["qid"]: row for row in rows}
    v3 = analyze_saved(args.v3, by_qid)
    v4 = analyze_saved(args.v4, by_qid)
    result: dict[str, Any] = {
        "dataset": dataset_audit(rows),
        "saved_v3": v3,
        "saved_v4": v4,
        "v3_v4_changed_queries": compare_versions(v3, v4),
        "current_response_cache": analyze_cache(rows[:20]),
        "protocol_notes": {
            "local_aggregation": "macro average of per-query metrics; micro is now also reported diagnostically",
            "pasa_released_matching": "case-folded alphabetic-only exact title key (keep_letters)",
            "pasa_date_rule": "source_meta.published_time minus 7 days, passed as search end_date",
            "current_system_date_rule": "only user-expressed time_range; AutoScholarQuery source date is not passed",
            "competition_hidden_evaluator": "尚未确认",
        },
    }
    if args.live_limit:
        result["live_stage_audit"] = asyncio.run(
            run_live(rows, args.live_limit, ROOT / "reports/audit_traces")
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "dataset": result["dataset"],
        "v3_at20": v3["recomputed"]["20"],
        "v4_at20": v4["recomputed"]["20"],
        "changed_queries": len(result["v3_v4_changed_queries"]),
        "cache": result["current_response_cache"],
        "live": result.get("live_stage_audit", {}).get("gold_outcome_counts"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
