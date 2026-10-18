"""Apply already-frozen selector champions to a scored candidate file."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_v4_selector_experiments import (
    core_rescue_selector,
    feature,
    llm090,
    set_metrics,
    threshold_selector,
    top_live,
)

OUT = ROOT / "reports/joint_optimization_v4"


def evaluate(rows, champion, artifacts):
    decision = champion["decision"]
    if decision == "fixed_k2":
        return set_metrics(rows, top_live(2))
    if decision == "fixed_top20":
        return set_metrics(rows, top_live(20))
    if decision == "fixed_llm090":
        return set_metrics(rows, llm090)
    name = champion["selector"]
    names = tuple(champion["feature_names"])
    model = artifacts["models"][name]
    probabilities = model.predict_proba(np.asarray([feature(row, names) for row in rows], dtype=float))[:, 1]
    field = f"frozen_probability_{name}"
    for row, probability in zip(rows, probabilities):
        row[field] = float(probability)
    if decision == "core_rescue":
        return set_metrics(rows, core_rescue_selector(field, **json.loads(champion["threshold"])))
    max_results = 10 if decision == "recall_floor_035" else 8
    return set_metrics(rows, threshold_selector(field, float(champion["threshold"]), max_results, 2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.open(encoding="utf-8") if line.strip()]
    with (OUT / "selector_model.pkl").open("rb") as handle:
        artifacts = pickle.load(handle)
    result = {"label": args.label}
    for key, filename in (("f1_champion", "champion_f1.json"), ("balanced_champion", "champion_balanced.json"), ("competition_champion", "champion_competition.json")):
        champion = json.loads((OUT / filename).read_text(encoding="utf-8"))
        result[key] = {"config": champion, "metrics": evaluate(rows, champion, artifacts)}
    result["top20"] = set_metrics(rows, top_live(20))
    result["k2"] = set_metrics(rows, top_live(2))
    result["llm090"] = set_metrics(rows, llm090)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
