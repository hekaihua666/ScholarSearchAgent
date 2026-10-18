"""Run cached ACE criterion judgments over frozen candidates without gold in prompts."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.llm_client import _chat
from backend.selection.evidence_verifier import validate_evidence_span, verify_with_llm


def key(query: str, graph: dict[str, Any], papers: list[dict[str, Any]]) -> str:
    payload = [query, graph, [(p.get("title"), str(p.get("abstract") or "")[:1600], p.get("year")) for p in papers]]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def aggregate(graph: dict[str, Any], paper: dict[str, Any], judgments: list[dict[str, Any]], unknown: float = 0.4) -> dict[str, Any]:
    by_id = {str(item.get("criterion_id")): item for item in judgments if isinstance(item, dict)}
    resolved = []
    for criterion in graph.get("criteria") or []:
        raw = by_id.get(str(criterion.get("criterion_id"))) or {}
        state = str(raw.get("state") or "UNKNOWN").upper()
        if state not in {"ENTAILED", "CONTRADICTED", "UNKNOWN"}:
            state = "UNKNOWN"
        span = str(raw.get("evidence_span") or "").strip()
        if criterion.get("type") == "time":
            valid_span = bool(span) and str(paper.get("year") or "") in span
        else:
            valid_span = validate_evidence_span(span, paper)
        if state != "UNKNOWN" and not valid_span:
            state = "UNKNOWN"
            span = ""
        resolved.append({**criterion, "state": state, "evidence_span": span, "evidence_valid": valid_span})
    required = [item for item in resolved if item.get("importance") == "required"]
    hard_gate = "fail" if any(item["state"] == "CONTRADICTED" for item in required) else "unknown" if any(item["state"] == "UNKNOWN" for item in required) else "pass"
    weighted = []
    for item in resolved:
        score = 1.0 if item["state"] == "ENTAILED" else 0.0 if item["state"] == "CONTRADICTED" else unknown
        weight = 1.4 if item.get("importance") == "required" else 1.0
        if item.get("type") == "relation":
            weight *= 1.8
        weighted.append((score, weight))
    value = math.exp(sum(weight * math.log(1e-5 + score) for score, weight in weighted) / sum(weight for _, weight in weighted)) if weighted else unknown
    evidence_coverage = sum(item["evidence_valid"] for item in resolved) / max(len(resolved), 1)
    entailed = sum(item["state"] == "ENTAILED" for item in required) / max(len(required), 1)
    probability = 0.0 if hard_gate == "fail" else min(1.0, 0.72 * value + 0.18 * entailed + 0.10 * evidence_coverage)
    return {
        "criteria": resolved,
        "hard_gate": hard_gate,
        "criterion_score": round(value, 6),
        "evidence_coverage": round(evidence_coverage, 6),
        "required_entailed_ratio": round(entailed, 6),
        "probability": round(probability, 6),
    }


async def main_async(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in args.input.open(encoding="utf-8") if line.strip()]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["candidate_rank"]) <= args.top_n:
            groups[row["query_id"]].append(row)
    cache_path = ROOT / "reports/joint_optimization_v4/model_cache/ace_llm_judgments.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    total_tokens = 0
    failures = 0
    for index, (qid, group) in enumerate(sorted(groups.items()), 1):
        group.sort(key=lambda row: int(row["candidate_rank"]))
        query = group[0]["query"]
        graph = group[0]["constraint_graph"]
        papers = [{"title": row["title"], "abstract": row["abstract"], "year": row.get("year")} for row in group]
        digest = key(query, graph, papers)
        if digest not in cache:
            try:
                judgments, tokens = await verify_with_llm(query, graph, papers, _chat)
                cache[digest] = {"judgments": judgments, "tokens": tokens}
                total_tokens += tokens
            except Exception as exc:
                cache[digest] = {"judgments": [], "tokens": 0, "error": type(exc).__name__}
                failures += 1
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        payload = cache[digest]
        per_paper = {int(item.get("paper_index")): item for item in payload.get("judgments") or [] if isinstance(item, dict) and str(item.get("paper_index", "")).isdigit()}
        for paper_index, row in enumerate(group):
            item = per_paper.get(paper_index) or {}
            row["ace_llm"] = aggregate(graph, papers[paper_index], item.get("judgments") or [])
        print(f"[{index}/{len(groups)}] {qid} judgments={len(per_paper)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for qid in sorted(groups):
            for row in groups[qid]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    runtime = {"queries": len(groups), "papers": sum(len(x) for x in groups.values()), "new_tokens": total_tokens, "failures": failures, "top_n": args.top_n, "cache_entries": len(cache)}
    (ROOT / "reports/joint_optimization_v4/ace_llm_runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    print(json.dumps(runtime, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=12)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
