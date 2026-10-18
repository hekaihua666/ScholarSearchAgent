"""Offline admission-gate experiments over frozen citation/second-round outputs."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.paper_identity import match_identity
from backend.trace_rank import canonical_key, diversify, fuse_candidates, score_candidates
from scripts.audit_evaluation import aggregate, gold_papers, load_jsonl, metrics_from_papers, source_cutoff
from scripts.run_frozen_official_benchmark import module_candidates


def gate_expansion(papers: list[dict[str, Any]], understanding: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    if not cfg.get("enabled", True):
        return []
    if not any(key in cfg for key in ("min_score", "min_coverage", "min_any_evidence", "top_n")):
        return papers
    fused = fuse_candidates(papers)
    scored = score_candidates(copy.deepcopy(fused), understanding)
    threshold = float(cfg.get("min_score", -1))
    coverage = float(cfg.get("min_coverage", -1))
    evidence = float(cfg.get("min_any_evidence", -1))
    kept = []
    for paper in scored:
        any_evidence = max(float(paper.get("lexical_score") or 0), float(paper.get("idf_lexical_score") or 0),
                           float(paper.get("constraint_coverage_score") or 0))
        if float(paper.get("final_score") or 0) < threshold:
            continue
        if float(paper.get("constraint_coverage_score") or 0) < coverage:
            continue
        if any_evidence < evidence:
            continue
        kept.append(paper)
    top_n = int(cfg.get("top_n") or 0)
    kept = kept[:top_n] if top_n else kept
    accepted = {canonical_key(paper) for paper in kept}
    # Return original route-level papers so the final fusion retains every
    # query/source/rank hit instead of collapsing provenance in the gate.
    return [paper for paper in papers if canonical_key(paper) in accepted]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\AICompetition\references\github\SPAR\benchmark\AutoScholarQuery_test.jsonl"))
    parser.add_argument("--configs", type=Path, default=ROOT / "configs/experiments/admission_dev20.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/overnight_iteration/admission_challengers.csv")
    args = parser.parse_args()
    rows = load_jsonl(args.dataset)[:20]
    configs = json.loads(args.configs.read_text(encoding="utf-8"))
    buckets = {cfg["experiment_id"]: {"m20": [], "candidate": [], "counts": [], "details": []} for cfg in configs}
    for row in rows:
        qdir = args.run_dir / "queries" / row["qid"]
        snapshot = json.loads((qdir / "snapshot.json").read_text(encoding="utf-8"))
        response = json.loads((qdir / "live_response.json").read_text(encoding="utf-8"))
        understanding = response.get("query_understanding") or {}
        cutoff = source_cutoff(row); assert cutoff is not None
        modules = module_candidates(snapshot, cutoff)
        base = [paper for name in ("lexical", "semantic", "arxiv", "title") for paper in modules[name]]
        gold = gold_papers(row)
        for cfg in configs:
            citation = gate_expansion(modules["citation"], understanding, cfg.get("citation") or {})
            second = gate_expansion(modules["second"], understanding, cfg.get("second") or {})
            fused = fuse_candidates(base + citation + second)
            ranked = score_candidates(copy.deepcopy(fused), understanding)
            top20 = diversify(ranked, limit=20, lambda_relevance=0.88)
            bucket = buckets[cfg["experiment_id"]]
            bucket["m20"].append(metrics_from_papers(top20, gold, 20))
            bucket["candidate"].append(metrics_from_papers(fused, gold, len(fused)))
            bucket["counts"].append(len(fused))
            bucket["details"].append({"query_id": row["qid"], "top20_gold": sum(any(match_identity(p,g) for p in top20) for g in gold),
                                      "candidate_gold": sum(any(match_identity(p,g) for p in fused) for g in gold),
                                      "citation_admitted": len(citation), "second_admitted": len(second)})
    parent = buckets[configs[0]["experiment_id"]]
    parent_metrics = aggregate(parent["m20"])
    parent_tp = {d["query_id"]: d["top20_gold"] for d in parent["details"]}
    output_rows=[]; full={}
    for cfg in configs:
        bucket=buckets[cfg["experiment_id"]]; m=aggregate(bucket["m20"]); c=aggregate(bucket["candidate"])
        current={d["query_id"]:d["top20_gold"] for d in bucket["details"]}
        wins=sum(current[q]>parent_tp[q] for q in parent_tp); losses=sum(current[q]<parent_tp[q] for q in parent_tp)
        row={"experiment_id":cfg["experiment_id"],"hypothesis":cfg.get("hypothesis"),"mean_candidates":round(sum(bucket['counts'])/len(bucket['counts']),3),
             "candidate_gold":c['micro']['true_positive'],"tp20":m['micro']['true_positive'],"macro_recall20":m['macro']['recall'],
             "macro_f1_20":m['macro']['f1'],"macro_ndcg20":m['macro']['ndcg'],"delta_tp":m['micro']['true_positive']-parent_metrics['micro']['true_positive'],
             "delta_ndcg":round(m['macro']['ndcg']-parent_metrics['macro']['ndcg'],6),"wins":wins,"ties":20-wins-losses,"losses":losses}
        output_rows.append(row); full[cfg['experiment_id']]={"config":cfg,"summary":row,"details":bucket['details']}
        print(cfg['experiment_id'],row,flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(output_rows[0]));w.writeheader();w.writerows(output_rows)
    args.output.with_suffix('.json').write_text(json.dumps(full,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__ == "__main__":
    main()
