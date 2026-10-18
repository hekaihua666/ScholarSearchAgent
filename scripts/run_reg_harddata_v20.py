"""Independent holdout-30 and combined blind-50 audit for frozen REG v18.

This script does not tune REG.  It applies the calibration-frozen v18 route to
the 30 blind queries that were preregistered but never evaluated by v19, then
combines those results with the existing one-shot holdout-20 capture.  It also
computes paired query-level bootstrap intervals and fixed-route component
ablations for paper-ready reporting.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import the v18 runner first.  It intentionally bootstraps torch before the
# legacy sklearn/OpenMP stack on Windows.
from scripts import run_relational_evidence_graph_v18 as reg  # noqa: E402
from scripts.run_controlled_retrieval_v8 import aggregate_metrics, v6_selection  # noqa: E402
from scripts.run_short_sprint_v5 import write_csv  # noqa: E402


REPORT = ROOT / "reports/reg_harddata_v20"
CAPTURE = ROOT / "reports/frozen_eval/reg_harddata_v20"
HOLDOUT30_IDS = ROOT / "configs/atomic_bottleneck_routing_v19/holdout30_ids.json"
HOLDOUT20_IDS = ROOT / "configs/relational_evidence_graph_v18/holdout20_ids.json"
FREEZE_PATH = ROOT / "configs/relational_evidence_graph_v18/calibration_freeze.json"
OLD_CAPTURE = ROOT / "reports/frozen_eval/relational_evidence_graph_v18"
SEED = 20260831


def ensure_dirs() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    CAPTURE.mkdir(parents=True, exist_ok=True)


def load_ids(path: Path) -> list[str]:
    return list(json.loads(path.read_text(encoding="utf-8"))["query_ids"])


def records_for(ids: Sequence[str]) -> list[dict[str, Any]]:
    dataset = {row["qid"]: row for row in reg.load_jsonl(reg.HOLDOUT_SOURCE)}
    records: list[dict[str, Any]] = []
    for qid in ids:
        row = copy.deepcopy(dataset[qid])
        row["_v8_split"] = "reg_harddata_v20"
        records.append(row)
    return records


def scored_from_capture(ids: Sequence[str], capture: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    score_root = capture / "blind_holdout20_v18" / "queries"
    retrieval_root = capture / "blind_holdout20_v18_retrieval" / "queries"
    for qid in ids:
        payload = json.loads((score_root / qid / "scores.json").read_text(encoding="utf-8"))
        retrieval = json.loads((retrieval_root / qid / "live_result.json").read_text(encoding="utf-8"))
        n = int(payload["incumbent_count"])
        output[qid] = {
            "incumbents": payload["rows"][:n],
            "frontier": payload["rows"][n:],
            "all_frontier": reg.frontier_for(qid, retrieval["reverse"], retrieval["qcf"]),
        }
    return output


def baseline_inputs(records: Sequence[dict[str, Any]]) -> tuple[
    dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]], dict[str, dict[str, Any]],
]:
    top20 = reg._holdout_top20(records)
    pools = {record["qid"]: reg._holdout_pool(record) for record in records}
    understandings = {
        record["qid"]: copy.deepcopy(
            reg._holdout_response(record["qid"]).get("query_understanding") or {}
        )
        for record in records
    }
    contracts = {
        qid: reg._contract_from_understanding(value)
        for qid, value in understandings.items()
    }
    return top20, pools, understandings, contracts


def frozen_products(
    records: Sequence[dict[str, Any]],
    scored: dict[str, dict[str, Any]],
    top20: dict[str, list[dict[str, Any]]],
    pools: dict[str, list[dict[str, Any]]],
    *,
    weights: Sequence[float] | Mapping[str, Sequence[float]] | None = None,
) -> dict[str, Any]:
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    chosen = freeze["chosen"]
    baseline_top = aggregate_metrics(records, top20)
    base_selected = v6_selection(records, top20)
    baseline_v6 = aggregate_metrics(records, base_selected)
    top_metrics, routed_top, top_details = reg.evaluate_profile(
        records,
        scored,
        name=str(chosen["configuration"]),
        weights=weights if weights is not None else freeze["weights"],
        score_floor=float(chosen["score_floor"]),
        margin=float(chosen["margin"]),
        max_promotions=int(chosen["max_promotions"]),
        protect_head=int(chosen["protect_head"]),
    )
    consensus = freeze["chosen_consensus"]
    final_metrics, final_map, final_details = reg.evaluate_consensus_final(
        records,
        scored,
        base_selected,
        strict_floor=float(consensus["strict_floor"]),
        graph_floor=float(consensus["graph_floor"]),
    )
    current = {"top20": top20, "candidate": pools}
    candidate = aggregate_metrics(records, reg.candidate_map(records, current, scored))
    return {
        "baseline_top20": baseline_top,
        "baseline_v6": baseline_v6,
        "candidate": candidate,
        "reg_top20": top_metrics,
        "reg_top20_map": routed_top,
        "top_details": top_details,
        "consensus_final": final_metrics,
        "consensus_final_map": final_map,
        "final_details": final_details,
    }


def per_query_metrics(
    records: Sequence[dict[str, Any]],
    products: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for record in records:
        metric = aggregate_metrics([record], {record["qid"]: products[record["qid"]]})
        output[record["qid"]] = {
            key: float(metric[key]) for key in ("precision", "recall", "f1")
        }
    return output


def paired_bootstrap(
    records: Sequence[dict[str, Any]],
    baseline: dict[str, list[dict[str, Any]]],
    challenger: dict[str, list[dict[str, Any]]],
    *,
    draws: int = 10_000,
) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    qids = [record["qid"] for record in records]
    base = per_query_metrics(records, baseline)
    new = per_query_metrics(records, challenger)
    rows: list[dict[str, Any]] = []
    for metric in ("precision", "recall", "f1"):
        delta = [new[qid][metric] - base[qid][metric] for qid in qids]
        samples = []
        for _ in range(draws):
            samples.append(sum(delta[rng.randrange(len(delta))] for _ in delta) / len(delta))
        samples.sort()
        rows.append({
            "metric": metric,
            "baseline_macro": sum(base[qid][metric] for qid in qids) / len(qids),
            "reg_macro": sum(new[qid][metric] for qid in qids) / len(qids),
            "paired_delta": sum(delta) / len(delta),
            "ci95_low": samples[int(0.025 * draws)],
            "ci95_high": samples[int(0.975 * draws) - 1],
            "wins": sum(value > 0 for value in delta),
            "ties": sum(value == 0 for value in delta),
            "losses": sum(value < 0 for value in delta),
        })
    return rows


def ablated_weights(
    weights: Mapping[str, Sequence[float]], disabled: set[int]
) -> dict[str, list[float]]:
    output: dict[str, list[float]] = {}
    for name, vector in weights.items():
        values = [0.0 if index in disabled else float(value) for index, value in enumerate(vector)]
        total = sum(values)
        output[name] = [value / total for value in values]
    return output


def fixed_ablation_rows(
    records: Sequence[dict[str, Any]],
    scored: dict[str, dict[str, Any]],
    top20: dict[str, list[dict[str, Any]]],
    pools: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    weights = freeze["weights"]
    definitions = [
        ("Full REG", set()),
        ("w/o condition conjunction", {0}),
        ("w/o relation and direction", {1, 2}),
        ("w/o seed bridge and graph", {3, 4}),
        ("w/o TRACE prior", {5}),
        ("w/o local cross-encoder", {6}),
    ]
    rows: list[dict[str, Any]] = []
    for name, disabled in definitions:
        selected = weights if not disabled else ablated_weights(weights, disabled)
        result = frozen_products(records, scored, top20, pools, weights=selected)["reg_top20"]
        rows.append({
            "configuration": name,
            "TP": result["top20_TP"],
            "FP": result["top20_FP"],
            "recall": result["top20_recall"],
            "f1": result["top20_f1"],
            "gain": result["top20_gain"],
            "lost": result["top20_lost"],
            "actions": result["actions"],
            "action_precision": result["action_precision"],
        })
    return rows


async def evaluate_holdout30() -> dict[str, Any]:
    ensure_dirs()
    records = records_for(load_ids(HOLDOUT30_IDS))
    top20, pools, understandings, contracts = baseline_inputs(records)

    # Redirect only new captures; the frozen parameters remain loaded from v18.
    original_capture = reg.CAPTURE
    reg.CAPTURE = CAPTURE
    try:
        retrieval, retrieval_manifest = await reg._capture_holdout_retrieval(
            records, top20, pools, understandings, contracts
        )
        if not retrieval_manifest["replay_all_identical"]:
            raise RuntimeError("holdout30 retrieval replay mismatch")
        current = {"top20": top20, "candidate": pools}
        v15 = {qid: payload["reverse"] for qid, payload in retrieval.items()}
        v11 = {qid: payload["qcf"] for qid, payload in retrieval.items()}
        contract_results = {
            qid: {"contract_result": {"parsed": {"contract": contract}}}
            for qid, contract in contracts.items()
        }
        selector = reg.make_selector(reg.CROSS_CACHE)
        scored, score_manifest = reg.score_split(
            "holdout", records, contract_results, current, v15, v11, selector
        )
    finally:
        reg.CAPTURE = original_capture

    products = frozen_products(records, scored, top20, pools)
    write_csv(REPORT / "01_holdout30_metrics.csv", [
        {"configuration": "baseline_top20", **products["baseline_top20"]},
        {"configuration": "baseline_v6_final", **products["baseline_v6"]},
        {"configuration": "candidate_pool", **products["candidate"]},
        products["reg_top20"],
        products["consensus_final"],
    ])
    write_csv(REPORT / "02_holdout30_top20_actions.csv", products["top_details"])
    write_csv(REPORT / "03_holdout30_final_actions.csv", products["final_details"])
    state = {
        "schema_version": 1,
        "status": "holdout30_evaluated",
        "query_count": len(records),
        "baseline_top20": products["baseline_top20"],
        "baseline_v6": products["baseline_v6"],
        "candidate": products["candidate"],
        "reg_top20": products["reg_top20"],
        "consensus_final": products["consensus_final"],
        "retrieval_manifest": retrieval_manifest,
        "score_manifest": score_manifest,
    }
    (REPORT / "holdout30_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return state


def combine_blind50() -> dict[str, Any]:
    ensure_dirs()
    ids20, ids30 = load_ids(HOLDOUT20_IDS), load_ids(HOLDOUT30_IDS)
    records = records_for([*ids20, *ids30])
    top20, pools, _, _ = baseline_inputs(records)
    scored = {
        **scored_from_capture(ids20, OLD_CAPTURE),
        **scored_from_capture(ids30, CAPTURE),
    }
    products = frozen_products(records, scored, top20, pools)
    bootstrap = paired_bootstrap(records, top20, products["reg_top20_map"])
    ablations = fixed_ablation_rows(records, scored, top20, pools)

    write_csv(REPORT / "04_blind50_metrics.csv", [
        {"configuration": "baseline_top20", **products["baseline_top20"]},
        {"configuration": "baseline_v6_final", **products["baseline_v6"]},
        {"configuration": "candidate_pool", **products["candidate"]},
        products["reg_top20"],
        products["consensus_final"],
    ])
    write_csv(REPORT / "05_blind50_top20_actions.csv", products["top_details"])
    write_csv(REPORT / "06_blind50_final_actions.csv", products["final_details"])
    write_csv(REPORT / "07_blind50_paired_bootstrap.csv", bootstrap)
    write_csv(REPORT / "08_blind50_fixed_ablation.csv", ablations)

    old_manifest = json.loads(
        (OLD_CAPTURE / "blind_holdout20_v18_retrieval/manifest.json").read_text(encoding="utf-8")
    )
    new_manifest = json.loads(
        (CAPTURE / "blind_holdout20_v18_retrieval/manifest.json").read_text(encoding="utf-8")
    )
    calls = sum(int(row.get("api_calls") or 0) for row in [
        *old_manifest["queries"], *new_manifest["queries"]
    ])
    state = {
        "schema_version": 1,
        "status": "blind50_combined",
        "query_count": 50,
        "split_isolation": {"old20_new30_overlap": len(set(ids20) & set(ids30))},
        "baseline_top20": products["baseline_top20"],
        "baseline_v6": products["baseline_v6"],
        "candidate": products["candidate"],
        "reg_top20": products["reg_top20"],
        "consensus_final": products["consensus_final"],
        "paired_bootstrap": bootstrap,
        "fixed_ablation": ablations,
        "cost": {"openalex_calls": calls, "calls_per_query": calls / 50.0},
    }
    (REPORT / "blind50_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("holdout30", "combine50", "all"))
    args = parser.parse_args()
    if args.phase in {"holdout30", "all"}:
        result = asyncio.run(evaluate_holdout30())
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if args.phase in {"combine50", "all"}:
        result = combine_blind50()
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
