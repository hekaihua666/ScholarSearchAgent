"""Compare LIVE retrieval profiles with one fixed evaluator and selectors."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_evaluation import aggregate, gold_papers, load_jsonl, metrics_from_papers

DATASET = Path(r"D:\AICompetition\references\github\SPAR\benchmark\AutoScholarQuery_test.jsonl")
OUT = ROOT / "reports/joint_optimization_v4"


def selected_metrics(run_dir: Path, rows: list[dict[str, Any]], strategy: str) -> dict[str, Any]:
    values = []
    counts = []
    tp = 0
    for row in rows:
        payload = json.loads((run_dir / "queries" / row["qid"] / "live_response.json").read_text(encoding="utf-8"))
        papers = payload.get("ranked_candidates") or payload.get("papers") or []
        if strategy == "k2":
            chosen = papers[:2]
        elif strategy == "llm090":
            chosen = [paper for paper in papers if float(paper.get("llm_relevance_score") or 0) >= 0.90]
        else:
            chosen = papers[:20]
        metric = metrics_from_papers(chosen, gold_papers(row), max(len(chosen), 1))
        values.append(metric); counts.append(len(chosen)); tp += int(metric["tp"])
    result = aggregate(values)
    return {"macro": result["macro"], "micro": result["micro"], "mean_output_count": sum(counts) / len(counts), "unique_tp": tp}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", nargs=2, metavar=("NAME", "DIR"), required=True)
    args = parser.parse_args()
    rows = load_jsonl(DATASET)[:20]
    output = []
    details = {}
    for name, raw_dir in args.run:
        run_dir = Path(raw_dir)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        final = manifest["retrieval_ablation"]["F_plus_second"]
        top20 = selected_metrics(run_dir, rows, "top20")
        k2 = selected_metrics(run_dir, rows, "k2")
        llm = selected_metrics(run_dir, rows, "llm090")
        qcount = len(rows)
        row = {
            "profile": name,
            "retrieval_profile": manifest.get("retrieval_profile", "v3"),
            "candidate_gold": final["candidate_gold"],
            "candidate_recall_macro": final["candidate_recall"]["macro"]["recall"],
            "candidate_recall_micro": final["candidate_recall"]["micro"]["recall"],
            "top20_precision": top20["macro"]["precision"],
            "top20_recall": top20["macro"]["recall"],
            "top20_f1": top20["macro"]["f1"],
            "k2_precision": k2["macro"]["precision"],
            "k2_recall": k2["macro"]["recall"],
            "k2_f1": k2["macro"]["f1"],
            "llm090_precision": llm["macro"]["precision"],
            "llm090_recall": llm["macro"]["recall"],
            "llm090_f1": llm["macro"]["f1"],
            "llm090_mean_output": round(llm["mean_output_count"], 3),
            "api_calls_per_query": round(final["cumulative_api_calls"] / qcount, 3),
            "external_seconds_per_query": round(final["cumulative_external_seconds"] / qcount, 3),
            "expansion_tokens_per_query": round(final["cumulative_llm_tokens"] / qcount, 3),
            "replay_identical": manifest["replay_all_identical"],
            "date_legal": manifest["all_final_papers_date_legal"],
        }
        output.append(row); details[name] = {"top20": top20, "k2": k2, "llm090": llm}
    baseline = output[0]
    for row in output:
        candidate_gain = row["candidate_recall_macro"] - baseline["candidate_recall_macro"]
        recall_gain = row["llm090_recall"] - baseline["llm090_recall"]
        f1_gain = row["llm090_f1"] - baseline["llm090_f1"]
        low_cost = row["external_seconds_per_query"] <= 0.8 * baseline["external_seconds_per_query"] and f1_gain >= -0.005 and recall_gain >= -0.005
        row["candidate_recall_gain"] = round(candidate_gain, 6)
        row["selected_recall_gain"] = round(recall_gain, 6)
        row["selected_f1_gain"] = round(f1_gain, 6)
        row["accepted"] = bool(
            (f1_gain >= 0.01 and recall_gain >= 0)
            or (candidate_gain >= 0.05 and recall_gain >= 0.03 and f1_gain >= -0.005)
            or low_cost
        )
    csv_path = OUT / "12_retrieval_ablation.csv"
    fields = list(output[0])
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(output)
    citation = next((row for row in output if "citation" in row["profile"]), None)
    report = [
        "# 双前沿引用扩展消融",
        "",
        "所有行均为新的 LIVE snapshot，并使用相同 dev20、cutoff、身份匹配和 LLM-0.90 固定 Selector。",
        "",
        "```json",
        json.dumps(output, ensure_ascii=False, indent=2),
        "```",
    ]
    if citation:
        report.extend([
            "",
            f"双前沿 candidate recall 变化：{citation['candidate_recall_gain']:+.4f}；",
            f"固定 Selector 最终 Recall 变化：{citation['selected_recall_gain']:+.4f}；",
            f"F1 变化：{citation['selected_f1_gain']:+.4f}；接受：{citation['accepted']}。",
        ])
    (OUT / "06_dual_frontier_ablation.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
