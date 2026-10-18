"""Evaluate saved retrieval outputs without making any API calls.

JSONL input schema per line:
{"query_id":"q1", "predicted_ids":[...], "relevant_ids":[...]}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.evaluation import macro_average, retrieval_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    details = []
    for line in args.input.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        metrics = retrieval_metrics(row.get("predicted_ids", []), row.get("relevant_ids", []), args.k)
        details.append({"query_id": row.get("query_id"), **metrics})
    report = {"k": args.k, "query_count": len(details), "macro": macro_average(details), "details": details}
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
