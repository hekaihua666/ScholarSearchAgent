"""Attach cached cross-encoder scores to production-distribution selector pairs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.selection.cross_encoder_selector import CPU_FALLBACK_MODEL, CrossEncoderSelector


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=CPU_FALLBACK_MODEL)
    parser.add_argument("--max-rank", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.open(encoding="utf-8") if line.strip()]
    scored = [row for row in rows if int(row["candidate_rank"]) <= args.max_rank]
    pairs = [
        (
            f"{row['query']} Required constraints: "
            + "; ".join(c["description"] for c in (row.get("constraint_graph") or {}).get("criteria") or []),
            f"{row.get('title') or ''}. {str(row.get('abstract') or '')[:1800]}",
        )
        for row in scored
    ]
    selector = CrossEncoderSelector(
        args.model,
        ROOT / "reports/joint_optimization_v4/model_cache/cross_encoder_scores.json",
        max_length=512,
    )
    values, runtime = selector.predict(pairs, batch_size=args.batch_size)
    for row, value in zip(scored, values):
        row["cross_encoder_raw"] = value
        row["cross_encoder_sigmoid_uncalibrated"] = sigmoid(value)
        row["cross_encoder_model"] = args.model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    runtime["output"] = str(args.output)
    runtime["positive_pairs"] = sum(int(row["label"]) for row in scored)
    (ROOT / "reports/joint_optimization_v4/cross_encoder_runtime.json").write_text(
        json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(runtime, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
