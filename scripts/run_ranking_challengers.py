"""Run interpretable ranking-only challengers on immutable frozen candidates."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.paper_identity import match_identity
from backend.trace_rank import fuse_candidates, score_candidates, tokenize
from scripts.audit_evaluation import aggregate, gold_papers, load_jsonl, metrics_from_papers, source_cutoff
from scripts.run_frozen_official_benchmark import module_candidates


DEFAULT_WEIGHTS = {
    "lexical": 0.25, "idf": 0.18, "rrf": 0.20, "coverage": 0.14,
    "hypothesis": 0.08, "authority": 0.07, "freshness": 0.04,
    "metadata": 0.04, "semantic": 0.10, "citation_provenance": 1.0,
    "intent_bonus": 1.0, "generic_penalty": 1.0,
}


def rescore(papers: list[dict[str, Any]], understanding: dict[str, Any], weights: dict[str, float]) -> list[dict[str, Any]]:
    items = score_candidates(copy.deepcopy(papers), understanding)
    w = {**DEFAULT_WEIGHTS, **weights}
    fields = {
        "lexical": "lexical_score", "idf": "idf_lexical_score", "rrf": "rrf_score",
        "coverage": "constraint_coverage_score", "hypothesis": "title_hypothesis_score",
        "authority": "authority_score", "freshness": "freshness_score", "metadata": "metadata_score",
        "semantic": "semantic_provenance_score", "citation_provenance": "citation_provenance_score",
        "intent_bonus": "intent_bonus_score",
    }
    for paper in items:
        value = sum(w[name] * float(paper.get(field) or 0) for name, field in fields.items())
        value -= w["generic_penalty"] * float(paper.get("generic_result_penalty") or 0)
        paper["experiment_score"] = round(value, 8)
        paper["final_score"] = round(value, 8)
    return sorted(items, key=lambda paper: (-float(paper["experiment_score"]), str(paper.get("title") or "")))


def similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    left = tokenize(f"{a.get('title', '')} {a.get('abstract', '')}")
    right = tokenize(f"{b.get('title', '')} {b.get('abstract', '')}")
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def select(papers: list[dict[str, Any]], cfg: dict[str, Any], limit: int = 20) -> list[dict[str, Any]]:
    strategy = cfg.get("selection", "mmr")
    if strategy == "none":
        return papers[:limit]
    locked = int(cfg.get("locked_prefix") or 0)
    selected = list(papers[:locked])
    remaining = list(papers[locked:])
    token_cache = {
        id(paper): tokenize(f"{paper.get('title', '')} {paper.get('abstract', '')}")
        for paper in papers
    }
    lam = float(cfg.get("mmr_lambda", 0.88))
    threshold = float(cfg.get("redundancy_threshold", 0.0))
    while remaining and len(selected) < limit:
        best = None; best_key = None
        for paper in remaining:
            left = token_cache[id(paper)]
            redundancy = 0.0
            for prior in selected:
                right = token_cache[id(prior)]
                union = left | right
                redundancy = max(redundancy, len(left & right) / len(union) if union else 0.0)
            if redundancy < threshold:
                redundancy = 0.0
            value = lam * float(paper.get("experiment_score") or 0) - (1 - lam) * redundancy
            key = (value, float(paper.get("experiment_score") or 0), str(paper.get("title") or ""))
            if best_key is None or key > best_key:
                best, best_key = paper, key
        assert best is not None
        selected.append(best); remaining.remove(best)
    return selected


def gold_ranks(papers: list[dict[str, Any]], gold: list[dict[str, Any]]) -> list[int | None]:
    return [next((i for i, paper in enumerate(papers, 1) if match_identity(paper, target)), None) for target in gold]


def evaluate(rows: list[dict[str, Any]], run_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[int, list[dict[str, float]]] = {5: [], 10: [], 20: []}
    details = []
    for row in rows:
        qdir = run_dir / "queries" / row["qid"]
        snapshot = json.loads((qdir / "snapshot.json").read_text(encoding="utf-8"))
        response = json.loads((qdir / "live_response.json").read_text(encoding="utf-8"))
        cutoff = source_cutoff(row)
        assert cutoff is not None
        modules = module_candidates(snapshot, cutoff)
        merged = fuse_candidates([paper for name in ("lexical", "semantic", "arxiv", "title", "citation", "second") for paper in modules[name]])
        understanding = response.get("query_understanding") or {}
        ranked = rescore(merged, understanding, cfg.get("weights") or {})
        top20 = select(ranked, cfg, 20)
        gold = gold_papers(row)
        for k in metrics:
            metrics[k].append(metrics_from_papers(top20, gold, k))
        details.append({"query_id": row["qid"], "gold_count": len(gold),
                        "gold_ranks": gold_ranks(top20, gold),
                        "top20_titles": [paper.get("title") for paper in top20]})
    return {"metrics": {str(k): aggregate(v) for k, v in metrics.items()}, "details": details}


def tp_by_query(result: dict[str, Any]) -> dict[str, int]:
    return {item["query_id"]: sum(rank is not None for rank in item["gold_ranks"]) for item in result["details"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\AICompetition\references\github\SPAR\benchmark\AutoScholarQuery_test.jsonl"))
    parser.add_argument("--query-count", type=int, default=20)
    parser.add_argument("--configs", type=Path, default=ROOT / "configs/experiments/ranking_dev20.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/overnight_iteration")
    args = parser.parse_args()
    rows = load_jsonl(args.dataset)[:args.query_count]
    configs = json.loads(args.configs.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for cfg in configs:
        result = evaluate(rows, args.run_dir, cfg)
        result["config"] = cfg
        results[cfg["experiment_id"]] = result
        print(cfg["experiment_id"], result["metrics"]["20"]["macro"], flush=True)
    baseline_id = configs[0]["experiment_id"]
    baseline = results[baseline_id]
    base_tp = tp_by_query(baseline)
    base_m = baseline["metrics"]["20"]
    registry_rows = []
    for cfg in configs:
        eid = cfg["experiment_id"]; result = results[eid]; m = result["metrics"]["20"]
        current_tp = tp_by_query(result)
        wins = sum(current_tp[q] > base_tp[q] for q in base_tp)
        ties = sum(current_tp[q] == base_tp[q] for q in base_tp)
        losses = sum(current_tp[q] < base_tp[q] for q in base_tp)
        delta_tp = m["micro"]["true_positive"] - base_m["micro"]["true_positive"]
        delta_ndcg = m["macro"]["ndcg"] - base_m["macro"]["ndcg"]
        if eid == baseline_id:
            decision, reason = "baseline", "deterministic frozen parent"
        elif delta_tp >= 1 and delta_ndcg >= -0.005 and losses <= max(2, wins):
            decision, reason = "candidate", "TP gain with bounded NDCG/query regression"
        elif delta_tp == 0 and delta_ndcg > 0.005 and losses <= wins:
            decision, reason = "candidate", "stable TP with NDCG gain"
        else:
            decision, reason = "reject", "automatic dev20 acceptance rule not met"
        registry_rows.append({
            "experiment_id": eid, "hypothesis": cfg.get("hypothesis"), "selection": cfg.get("selection"),
            "weights": json.dumps(cfg.get("weights") or {}, sort_keys=True),
            "tp20": m["micro"]["true_positive"], "macro_recall20": m["macro"]["recall"],
            "macro_f1_20": m["macro"]["f1"], "macro_ndcg20": m["macro"]["ndcg"],
            "delta_tp": delta_tp, "delta_ndcg": round(delta_ndcg, 6),
            "wins": wins, "ties": ties, "losses": losses,
            "decision": decision, "reason": reason,
        })
    candidates = [row for row in registry_rows if row["decision"] in {"baseline", "candidate"}]
    best = max(candidates, key=lambda row: (row["tp20"], row["macro_ndcg20"], row["macro_f1_20"], -len(row["weights"])))
    with (args.output_dir / "ranking_challengers.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(registry_rows[0])); writer.writeheader(); writer.writerows(registry_rows)
    summary = {
        "run_mode": "frozen_ranking_only", "query_set": "dev20", "query_count": len(rows),
        "parent": baseline_id, "best": best["experiment_id"], "best_result": best,
        "results": results, "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    (args.output_dir / "ranking_challengers.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("BEST", json.dumps(best, ensure_ascii=False))


if __name__ == "__main__":
    main()
