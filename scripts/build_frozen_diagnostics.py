"""Build protocol, failure-taxonomy and channel reports from a frozen run.

This script is offline-only: it reads dataset rows, snapshots and audit traces;
it never calls an API/LLM and never feeds gold labels into retrieval/ranking.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.paper_identity import match_identity, paper_identity
from scripts.audit_evaluation import gold_papers, load_jsonl
from scripts.run_frozen_official_benchmark import module_candidates, module_cost


MODULES = ("lexical", "semantic", "arxiv", "title", "citation", "second")


def _hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _find_rank(papers: list[dict[str, Any]], gold: dict[str, Any]) -> tuple[int | None, str]:
    for rank, paper in enumerate(papers, 1):
        reason = match_identity(paper, gold)
        if reason:
            return rank, reason
    return None, "no_match"


def _stage(audit: dict[str, Any], name: str) -> list[dict[str, Any]]:
    for item in audit.get("stages") or []:
        if item.get("name") == name:
            return item.get("papers") or []
    return []


def _raw_module_candidates(snapshot: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    prefix_map = {
        "lexical": ("first_lexical_",),
        "semantic": ("first_semantic",),
        "arxiv": ("first_arxiv_",),
        "title": ("title_probe_",),
        "citation": ("citation_references", "citation_citing_"),
        "second": ("second_lexical_", "second_arxiv", "second_semantic_"),
    }
    output: dict[str, list[dict[str, Any]]] = {name: [] for name in MODULES}
    for call in (snapshot.get("calls") or {}).values():
        label = str(call.get("label") or "")
        value = call.get("output")
        if isinstance(value, list) and len(value) == 2 and isinstance(value[0], list):
            papers = value[0]
        else:
            papers = value if isinstance(value, list) else []
        for module, prefixes in prefix_map.items():
            if label.startswith(prefixes):
                output[module].extend(papers)
                break
    return output


def _identity_key(paper: dict[str, Any]) -> str:
    ident = paper_identity(paper)
    for values in (ident.arxiv_ids, ident.dois, ident.openalex_ids, ident.semantic_scholar_ids):
        if values:
            return sorted(values)[0]
    return ident.title or str(paper.get("title") or "")


def build(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    rows_all = load_jsonl(args.dataset)
    row_by_id = {row["qid"]: row for row in rows_all}
    rows = [row_by_id[item["query_id"]] for item in manifest["queries"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_gold: list[dict[str, Any]] = []
    per_query: list[dict[str, Any]] = []
    channel = {
        name: {"candidate_count": 0, "unique_candidate_keys": set(), "gold_keys": set(),
               "top20_gold_keys": set(), "seconds": 0.0, "api_calls": 0.0, "tokens": 0.0}
        for name in MODULES
    }
    loss_counts = Counter()
    snapshot_benchmarks = Counter()
    raw_output_hashes: dict[str, str] = {}
    candidate_pool_hashes: dict[str, str] = {}

    for row in rows:
        qid = row["qid"]
        qdir = args.run_dir / "queries" / qid
        snapshot = json.loads((qdir / "snapshot.json").read_text(encoding="utf-8"))
        audit = json.loads((qdir / "live_audit.json").read_text(encoding="utf-8"))
        response = json.loads((qdir / "live_response.json").read_text(encoding="utf-8"))
        snapshot_benchmarks[str(snapshot.get("benchmark") or "missing")] += 1
        raw_output_hashes[qid] = _hash({
            key: {"label": call.get("label"), "arguments": call.get("arguments"),
                  "output": call.get("output"), "error": call.get("error")}
            for key, call in sorted((snapshot.get("calls") or {}).items())
        })
        cutoff_text = next(item["cutoff"] for item in manifest["queries"] if item["query_id"] == qid)
        from datetime import datetime
        cutoff = datetime.fromisoformat(cutoff_text)
        legal_modules = module_candidates(snapshot, cutoff)
        raw_modules = _raw_module_candidates(snapshot)
        merged = _stage(audit, "merged_candidates")
        pre = _stage(audit, "shortlist_pre_llm")
        post = _stage(audit, "shortlist_post_llm")
        final = response.get("papers") or _stage(audit, "final_top20")
        candidate_pool_hashes[qid] = _hash([_identity_key(p) for p in merged])
        golds = gold_papers(row)

        for module in MODULES:
            papers = legal_modules[module]
            channel[module]["candidate_count"] += len(papers)
            channel[module]["unique_candidate_keys"].update(_identity_key(p) for p in papers)
            cost = module_cost(snapshot, module)
            channel[module]["seconds"] += cost["seconds"]
            channel[module]["api_calls"] += cost["api_calls"]
            channel[module]["tokens"] += cost["llm_tokens"]

        q_missing = q_ranking = q_final = 0
        for gold_index, gold in enumerate(golds):
            gold_id = f"{qid}:{gold_index}:{_identity_key(gold)}"
            legal_hits = [name for name in MODULES if _find_rank(legal_modules[name], gold)[0]]
            raw_hits = [name for name in MODULES if _find_rank(raw_modules[name], gold)[0]]
            merged_rank, merged_match = _find_rank(merged, gold)
            pre_rank, _ = _find_rank(pre, gold)
            post_rank, _ = _find_rank(post, gold)
            final_rank, final_match = _find_rank(final, gold)
            if final_rank:
                loss = "top20_hit"
                q_final += 1
            elif not legal_hits and raw_hits:
                loss = "date_or_missing_metadata_filtered"
                q_missing += 1
            elif not legal_hits:
                loss = "retrieval_miss_all_channels"
                q_missing += 1
            elif not pre_rank:
                loss = "retrieved_but_trace_mmr_below_shortlist"
                q_ranking += 1
            elif not post_rank:
                loss = "pre_llm_hit_removed_by_llm_stage"
                q_ranking += 1
            elif not final_rank:
                loss = "post_llm_hit_below_top20"
                q_ranking += 1
            else:
                loss = "unclassified"
            loss_counts[loss] += 1
            for name in legal_hits:
                channel[name]["gold_keys"].add(gold_id)
                if final_rank:
                    channel[name]["top20_gold_keys"].add(gold_id)
            per_gold.append({
                "query_id": qid,
                "gold_index": gold_index,
                "gold_title": gold.get("title"),
                "gold_arxiv_id": gold.get("arxiv_id"),
                "channels_legal": ";".join(legal_hits),
                "channels_raw": ";".join(raw_hits),
                "merged_rank": merged_rank,
                "pre_llm_rank": pre_rank,
                "post_llm_rank": post_rank,
                "final_rank": final_rank,
                "match_type": final_match if final_rank else merged_match,
                "loss_class": loss,
            })
        per_query.append({
            "query_id": qid, "gold_count": len(golds), "candidate_count": len(merged),
            "candidate_gold": sum(_find_rank(merged, g)[0] is not None for g in golds),
            "pre_llm_gold": sum(_find_rank(pre, g)[0] is not None for g in golds),
            "final_top20_gold": q_final, "retrieval_misses": q_missing,
            "ranking_losses": q_ranking, "raw_output_hash": raw_output_hashes[qid],
            "candidate_pool_hash": candidate_pool_hashes[qid],
        })

    def write_csv(path: Path, rows_out: list[dict[str, Any]]) -> None:
        if not rows_out:
            return
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows_out[0]))
            writer.writeheader(); writer.writerows(rows_out)

    write_csv(args.output_dir / "per_gold_diagnostics.csv", per_gold)
    write_csv(args.output_dir / "per_query_diagnostics.csv", per_query)
    channel_rows = []
    previous_gold: set[str] = set()
    for name in MODULES:
        item = channel[name]
        gold_set = set(item["gold_keys"])
        gain = gold_set - previous_gold
        channel_rows.append({
            "channel": name, "raw_legal_candidates": item["candidate_count"],
            "unique_candidates": len(item["unique_candidate_keys"]), "gold": len(gold_set),
            "incremental_gold_in_order": len(gain), "top20_gold_supported": len(item["top20_gold_keys"]),
            "external_seconds": round(item["seconds"], 3), "api_calls": int(item["api_calls"]),
            "llm_tokens": int(item["tokens"]),
            "gold_per_second": round(len(gain) / item["seconds"], 6) if item["seconds"] else None,
            "gold_per_api": round(len(gain) / item["api_calls"], 6) if item["api_calls"] else None,
        })
        previous_gold |= gold_set
    write_csv(args.output_dir / "channel_marginal_gain.csv", channel_rows)

    total_gold = len(per_gold)
    candidate_gold = sum(row["merged_rank"] is not None for row in per_gold)
    final_gold = sum(row["final_rank"] is not None for row in per_gold)
    report = f"""# Frozen dev20 Failure Taxonomy

- Run: `{manifest.get('run_id')}`
- Manifest benchmark alias: `{manifest.get('benchmark')}`
- Queries: {len(rows)}
- Gold papers: {total_gold}
- Candidate-pool gold: {candidate_gold}
- Final Top20 gold: {final_gold}
- Retrieval misses: {total_gold - candidate_gold}
- Ranking/selection gap: {candidate_gold - final_gold}
- Replay identical: `{manifest.get('replay_all_identical')}`
- Final dates legal: `{manifest.get('all_final_papers_date_legal')}`
- Source snapshot benchmark labels: `{dict(snapshot_benchmarks)}`

## Loss classes

""" + "\n".join(f"- `{name}`: {count}" for name, count in loss_counts.most_common()) + "\n"
    (args.output_dir / "failure_taxonomy.md").write_text(report, encoding="utf-8")
    oracle = f"""# Oracle Gap Report

The frozen union candidate pool contains **{candidate_gold}/{total_gold}** gold papers. A perfect
ranker over the existing pool could therefore recover at most {candidate_gold} gold papers across
the 20 Top20 lists. The current final output recovers **{final_gold}**, leaving a ranking/selection
gap of **{candidate_gold - final_gold}** and a retrieval gap of **{total_gold - candidate_gold}**.

Retrieval remains the larger bottleneck whenever the retrieval gap exceeds the ranking gap. This
report does not infer relevance from LLM self-scores; all counts use stable identity matching.
"""
    (args.output_dir / "oracle_gap_report.md").write_text(oracle, encoding="utf-8")
    summary = {
        "run_id": manifest.get("run_id"), "query_count": len(rows), "gold": total_gold,
        "candidate_gold": candidate_gold, "final_gold": final_gold,
        "retrieval_gap": total_gold - candidate_gold, "ranking_gap": candidate_gold - final_gold,
        "loss_counts": dict(loss_counts), "snapshot_benchmarks": dict(snapshot_benchmarks),
        "raw_output_hashes": raw_output_hashes, "candidate_pool_hashes": candidate_pool_hashes,
    }
    (args.output_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\AICompetition\references\github\SPAR\benchmark\AutoScholarQuery_test.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/overnight_iteration")
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
