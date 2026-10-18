"""Create/replay an official-date AutoScholarQuery frozen benchmark.

This is evaluation infrastructure. It does not tune TRACE weights, prompts or
MMR parameters. Component ablations reuse the exact frozen external outputs.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.audit_trace import finish_audit, start_audit
from backend.eval_runtime import EvaluationSession, evaluation_session, parse_publication_date
from backend.main import SearchRequest, search
from backend.paper_identity import match_identity, paper_identity
from backend.trace_rank import diversify, fuse_candidates, score_candidates
from scripts.audit_evaluation import (
    aggregate, gold_papers, load_jsonl, metrics_from_papers, source_cutoff,
)


EVALUATOR_VERSION = "stable-identity-v2+autoscholar-cutoff-v1"
TRACE_RANK_VERSION = "TRACE-Rank v4"
TRACE_LOOP_VERSION = "TRACE-Loop v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def deterministic_projection(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(payload.get("metrics") or {})
    for key in ("elapsed_seconds", "from_cache"):
        metrics.pop(key, None)
    projection = {
        "query": payload.get("query"),
        "query_understanding": payload.get("query_understanding"),
        "sub_queries": payload.get("sub_queries"),
        "papers": payload.get("papers"),
        "summary": payload.get("summary"),
        "search_process": payload.get("search_process"),
        "metrics": metrics,
        "relation_graph": payload.get("relation_graph"),
        "evidence_matrix": payload.get("evidence_matrix"),
        "research_landscape": payload.get("research_landscape"),
    }
    # Older LIVE captures may preserve arbitrary first-seen order for terms
    # with the same frequency. It is presentation-only, so normalize that
    # tie order before reproducibility hashing. New runs are sorted at source.
    landscape = projection.get("research_landscape")
    if isinstance(landscape, dict) and isinstance(landscape.get("top_terms"), list):
        landscape = copy.deepcopy(landscape)
        landscape["top_terms"] = sorted(
            landscape["top_terms"],
            key=lambda item: (-int(item.get("count") or 0), str(item.get("term") or "")),
        )
        projection["research_landscape"] = landscape
    return projection


def legal_for_cutoff(paper: dict[str, Any], cutoff: datetime) -> tuple[bool, str]:
    exact = parse_publication_date(paper.get("publication_date"))
    if exact:
        return exact <= cutoff.date(), "exact"
    try:
        year = int(paper.get("year"))
    except (TypeError, ValueError):
        return False, "missing"
    return datetime(year, 1, 1).date() <= cutoff.date(), "year_as_jan1"


async def execute_query(
    row: dict[str, Any], benchmark: str, run_dir: Path, mode: str,
    search_mode: str = "balanced", retrieval_profile: str = "v3",
) -> tuple[dict[str, Any], dict[str, Any], EvaluationSession]:
    cutoff = source_cutoff(row)
    if cutoff is None:
        raise ValueError(f"missing source cutoff: {row.get('qid')}")
    query_dir = run_dir / "queries" / row["qid"]
    query_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = query_dir / "snapshot.json"
    session = EvaluationSession.create(
        benchmark, row["qid"], cutoff, mode, snapshot_path
    )
    token = start_audit(row["question"])
    with evaluation_session(session):
        try:
            response = await search(SearchRequest(
                query=row["question"], mode=search_mode,
                selection_profile="exploratory", retrieval_profile=retrieval_profile,
            ))
            payload = response.model_dump()
        finally:
            audit = finish_audit(token)
    if mode == "live":
        session.save({"query": row["question"], "response": payload, "audit": audit})
    (query_dir / f"{mode}_response.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (query_dir / f"{mode}_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload, audit, session


def metrics_for_queries(
    rows: list[dict[str, Any]], payloads: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    by_k: dict[int, list[dict[str, float]]] = defaultdict(list)
    match_reasons = Counter()
    query_details = []
    for row in rows:
        payload = payloads[row["qid"]]
        papers = payload.get("papers") or []
        gold = gold_papers(row)
        per_k = {}
        for k in (5, 10, 20):
            value = metrics_from_papers(papers, gold, k)
            by_k[k].append(value)
            per_k[str(k)] = value
        matches = []
        for target in gold:
            found = None
            for rank, paper in enumerate(papers, 1):
                reason = match_identity(paper, target)
                if reason:
                    found = {"rank": rank, "match_type": reason, "predicted_title": paper.get("title")}
                    match_reasons[reason] += 1
                    break
            matches.append({"gold": target, **(found or {"rank": None, "match_type": "no_match"})})
            if found is None:
                match_reasons["no_match"] += 1
        query_details.append({"query_id": row["qid"], "metrics": per_k, "matches": matches})
    return {
        "metrics": {str(k): aggregate(values) for k, values in by_k.items()},
        "match_reasons": dict(match_reasons),
        "details": query_details,
    }


def snapshot_entries(snapshot: dict[str, Any], prefix: str) -> list[dict[str, Any]]:
    return sorted(
        [entry for entry in (snapshot.get("calls") or {}).values() if str(entry.get("label", "")).startswith(prefix)],
        key=lambda entry: str(entry.get("label")),
    )


def output_papers(entry: dict[str, Any]) -> list[dict[str, Any]]:
    output = entry.get("output")
    if isinstance(output, list) and len(output) == 2 and isinstance(output[0], list) and isinstance(output[1], str):
        return copy.deepcopy(output[0])
    return copy.deepcopy(output) if isinstance(output, list) else []


def add_stage(papers: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    result = []
    for paper in papers:
        item = copy.deepcopy(paper)
        if stage in {"lexical", "semantic", "arxiv", "title"}:
            item["source_round"] = "first_round"
        elif stage == "citation":
            item["source_round"] = "citation_expansion"
        else:
            item["source_round"] = "second_round"
        if stage == "title":
            item["retrieval_lane"] = "title_hypothesis"
        result.append(item)
    return result


def legalize(papers: list[dict[str, Any]], cutoff: datetime) -> list[dict[str, Any]]:
    return [paper for paper in papers if legal_for_cutoff(paper, cutoff)[0]]


def module_candidates(snapshot: dict[str, Any], cutoff: datetime) -> dict[str, list[dict[str, Any]]]:
    mapping = {
        "lexical": ["first_lexical_"],
        "semantic": ["first_semantic"],
        "arxiv": ["first_arxiv_"],
        "title": ["title_probe_"],
        "citation": ["citation_references", "citation_citing_"],
        "second": ["second_lexical_", "second_arxiv", "second_semantic_"],
    }
    result: dict[str, list[dict[str, Any]]] = {}
    for module, prefixes in mapping.items():
        papers: list[dict[str, Any]] = []
        for prefix in prefixes:
            for entry in snapshot_entries(snapshot, prefix):
                papers.extend(output_papers(entry))
        result[module] = add_stage(legalize(papers, cutoff), module)
    return result


def module_cost(snapshot: dict[str, Any], module: str) -> dict[str, float]:
    prefixes = {
        "lexical": ["first_lexical_"], "semantic": ["first_semantic"],
        "arxiv": ["first_arxiv_"], "title": ["title_probe_"],
        "citation": ["citation_references", "citation_citing_"],
        "second": ["second_lexical_", "second_arxiv", "second_semantic_", "llm_query_expansion"],
    }[module]
    entries = [entry for prefix in prefixes for entry in snapshot_entries(snapshot, prefix)]
    api_entries = [entry for entry in entries if not str(entry.get("label", "")).startswith("llm_")]
    # Calls inside lexical/arXiv/title groups are concurrent; max is a closer
    # estimate of wall time. Citation and second substeps are mostly sequential.
    if module in {"lexical", "arxiv", "title"}:
        seconds = max((float(entry.get("elapsed_seconds") or 0) for entry in entries), default=0.0)
    else:
        seconds = sum(float(entry.get("elapsed_seconds") or 0) for entry in entries)
    tokens = 0
    for entry in entries:
        if str(entry.get("label")) == "llm_query_expansion":
            output = entry.get("output") or []
            if isinstance(output, list) and output:
                try:
                    tokens += int(output[-1])
                except (TypeError, ValueError):
                    pass
    return {"seconds": seconds, "api_calls": float(len(api_entries)), "llm_tokens": float(tokens)}


def gold_key(query_id: str, gold: dict[str, Any]) -> str:
    identity = paper_identity(gold)
    stable = sorted(identity.arxiv_ids or identity.dois or {identity.title})
    return f"{query_id}:{stable[0] if stable else gold.get('title')}"


def evaluate_ablation(rows: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    stages = [
        ("A_lexical", ["lexical"]),
        ("B_plus_semantic", ["lexical", "semantic"]),
        ("C_plus_arxiv", ["lexical", "semantic", "arxiv"]),
        ("D_plus_title", ["lexical", "semantic", "arxiv", "title"]),
        ("E_plus_citation", ["lexical", "semantic", "arxiv", "title", "citation"]),
        ("F_plus_second", ["lexical", "semantic", "arxiv", "title", "citation", "second"]),
    ]
    accum: dict[str, dict[str, Any]] = {
        name: {"m20": [], "m50": [], "m100": [], "candidate": [], "counts": [], "gold_keys": set(),
               "seconds": 0.0, "api_calls": 0.0, "llm_tokens": 0.0}
        for name, _ in stages
    }
    for row in rows:
        snapshot = json.loads(
            (run_dir / "queries" / row["qid"] / "snapshot.json").read_text(encoding="utf-8")
        )
        cutoff = source_cutoff(row)
        assert cutoff is not None
        modules = module_candidates(snapshot, cutoff)
        response = (snapshot.get("metadata") or {}).get("response") or {}
        understanding = response.get("query_understanding") or {}
        gold = gold_papers(row)
        module_costs = {name: module_cost(snapshot, name) for name in modules}
        for stage_name, included in stages:
            candidates = [paper for name in included for paper in modules[name]]
            fused = fuse_candidates(candidates)
            scored = score_candidates(copy.deepcopy(fused), understanding)
            top20 = diversify(copy.deepcopy(scored), limit=20, lambda_relevance=0.88)
            bucket = accum[stage_name]
            bucket["m20"].append(metrics_from_papers(top20, gold, 20))
            bucket["m50"].append(metrics_from_papers(scored, gold, 50))
            bucket["m100"].append(metrics_from_papers(scored, gold, 100))
            bucket["candidate"].append(metrics_from_papers(fused, gold, len(fused)))
            bucket["counts"].append(len(fused))
            for target in gold:
                if any(match_identity(paper, target) for paper in fused):
                    bucket["gold_keys"].add(gold_key(row["qid"], target))
            for module in included:
                cost = module_costs[module]
                bucket["seconds"] += cost["seconds"]
                bucket["api_calls"] += cost["api_calls"]
                bucket["llm_tokens"] += cost["llm_tokens"]
    output = {}
    previous_gold: set[str] = set()
    previous_seconds = previous_api = 0.0
    for stage_name, included in stages:
        bucket = accum[stage_name]
        current_gold = set(bucket["gold_keys"])
        gain = len(current_gold - previous_gold)
        added_seconds = bucket["seconds"] - previous_seconds
        added_api = bucket["api_calls"] - previous_api
        output[stage_name] = {
            "modules": included,
            "mean_candidate_count": round(sum(bucket["counts"]) / len(bucket["counts"]), 3),
            "candidate_gold": len(current_gold),
            "incremental_unique_gold": gain,
            "candidate_recall": aggregate(bucket["candidate"]),
            "recall_at_50": aggregate(bucket["m50"]),
            "recall_at_100": aggregate(bucket["m100"]),
            "trace_mmr_top20": aggregate(bucket["m20"]),
            "cumulative_external_seconds": round(bucket["seconds"], 3),
            "incremental_external_seconds": round(added_seconds, 3),
            "cumulative_api_calls": int(bucket["api_calls"]),
            "incremental_api_calls": int(added_api),
            "cumulative_llm_tokens": int(bucket["llm_tokens"]),
            "gold_gain_per_second": round(gain / added_seconds, 6) if added_seconds > 0 else None,
            "gold_gain_per_api_call": round(gain / added_api, 6) if added_api > 0 else None,
        }
        previous_gold = current_gold
        previous_seconds = bucket["seconds"]
        previous_api = bucket["api_calls"]
    return output


async def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_jsonl(args.dataset)[args.offset: args.offset + args.limit]
    run_dir = args.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    live_payloads: dict[str, dict[str, Any]] = {}
    replay_payloads: dict[str, dict[str, Any]] = {}
    query_manifest = []
    for index, row in enumerate(rows, 1):
        cutoff = source_cutoff(row)
        assert cutoff is not None
        query_dir = run_dir / "queries" / row["qid"]
        snapshot_path = query_dir / "snapshot.json"
        live_response_path = query_dir / "live_response.json"
        if args.resume and snapshot_path.exists() and live_response_path.exists():
            live = json.loads(live_response_path.read_text(encoding="utf-8"))
            raw_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            live_session = EvaluationSession.create(
                args.benchmark, row["qid"], cutoff, "replay", snapshot_path
            )
            live_session.date_filter_stats = raw_snapshot.get("date_filter_stats") or {}
            print(f"[{index}/{len(rows)}] {row['qid']} reusing LIVE snapshot", flush=True)
        else:
            live, _, live_session = await execute_query(
                row, args.benchmark, run_dir, "live", args.search_mode, args.retrieval_profile
            )
        replay, _, replay_session = await execute_query(
            row, args.benchmark, run_dir, "replay", args.search_mode, args.retrieval_profile
        )
        live_payloads[row["qid"]] = live
        replay_payloads[row["qid"]] = replay
        live_hash = json_hash(deterministic_projection(live))
        replay_hash = json_hash(deterministic_projection(replay))
        legal = [legal_for_cutoff(paper, cutoff) for paper in live.get("papers") or []]
        snapshot_path = run_dir / "queries" / row["qid"] / "snapshot.json"
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        calls = list((snapshot.get("calls") or {}).values())
        query_manifest.append({
            "query_id": row["qid"], "query": row["question"],
            "gold": gold_papers(row), "cutoff": cutoff.isoformat(),
            "snapshot": str(snapshot_path), "snapshot_sha256": sha256_file(snapshot_path),
            "live_result_hash": live_hash, "replay_result_hash": replay_hash,
            "replay_identical": live_hash == replay_hash,
            "all_final_papers_date_legal": all(flag for flag, _ in legal),
            "final_date_modes": dict(Counter(mode for _, mode in legal)),
            "date_filter_stats": live_session.date_filter_stats,
            "missing_date_count": sum(
                stats.get("filtered_missing_date", 0) for stats in live_session.date_filter_stats.values()
            ),
            "api_calls_recorded": sum(1 for call in calls if not str(call.get("label", "")).startswith("llm_")),
            "external_call_errors": [call for call in calls if call.get("error")],
            "fallback_queries": (live.get("search_process") or {}).get("fallback_queries") or [],
            "query_plan": {
                "understanding": live.get("query_understanding"),
                "sub_queries": live.get("sub_queries"),
                "second_round_queries": (live.get("search_process") or {}).get("second_round_queries"),
            },
        })
        print(
            f"[{index}/{len(rows)}] {row['qid']} cutoff={cutoff.date()} "
            f"replay={'OK' if live_hash == replay_hash else 'DIFF'} "
            f"top20={len(live.get('papers') or [])}", flush=True,
        )
    baseline = metrics_for_queries(rows, live_payloads)
    replay_baseline = metrics_for_queries(rows, replay_payloads)
    ablation = evaluate_ablation(rows, run_dir)
    result = {
        "benchmark": args.benchmark,
        "run_id": run_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "dataset_sha256": sha256_file(args.dataset),
        "offset": args.offset, "query_count": len(rows),
        "versions": {
            "trace_rank": TRACE_RANK_VERSION, "trace_loop": TRACE_LOOP_VERSION,
            "evaluator": EVALUATOR_VERSION,
            "prompt_file_sha256": sha256_file(ROOT / "backend/llm_client.py"),
            "ranking_file_sha256": sha256_file(ROOT / "backend/trace_rank.py"),
            "python": platform.python_version(),
        },
        "mode": "LIVE external capture + deterministic materialization + FROZEN replay",
        "search_mode": args.search_mode,
        "retrieval_profile": args.retrieval_profile,
        "materialization_note": (
            "External API/LLM outputs are the immutable LIVE snapshot. Existing captures were "
            "re-materialized once after fixing a presentation-only tied-term ordering bug; the "
            "original responses are preserved as live_response_original.json."
        ),
        "date_protocol": {
            "rule": "source_meta.published_time minus 7 days",
            "exact_date": "keep when publication_date <= cutoff",
            "year_only": "interpret as January 1 of known year",
            "missing_year_and_date": "exclude before ranking",
        },
        "replay_all_identical": all(item["replay_identical"] for item in query_manifest),
        "all_final_papers_date_legal": all(item["all_final_papers_date_legal"] for item in query_manifest),
        "baseline_live": baseline,
        "baseline_replay": replay_baseline,
        "ablation_note": "Component Top20 uses deterministic TRACE+MMR before LLM rerank to isolate retrieval modules.",
        "retrieval_ablation": ablation,
        "queries": query_manifest,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\AICompetition\references\github\SPAR\benchmark\AutoScholarQuery_test.jsonl"))
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--benchmark", default="AutoScholarQuery_test_official_cutoff_20")
    parser.add_argument("--search-mode", choices=["quick", "efficient", "helios", "balanced", "deep"], default="balanced")
    parser.add_argument("--retrieval-profile", choices=["v3", "portfolio", "semantic_portfolio", "citation", "joint", "lean"], default="v3")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/frozen_eval/autoscholar20_trace_v4_official_20260824")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="ignore existing per-query LIVE snapshots")
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    result = asyncio.run(run(args))
    print(json.dumps({
        "manifest": str(args.output_dir / "manifest.json"),
        "replay_all_identical": result["replay_all_identical"],
        "all_final_papers_date_legal": result["all_final_papers_date_legal"],
        "baseline_at20": result["baseline_live"]["metrics"]["20"],
        "ablation": result["retrieval_ablation"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
