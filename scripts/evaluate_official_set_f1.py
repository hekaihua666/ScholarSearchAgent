"""Evaluate dynamic result sets under the explicitly candidate official protocol.

Input JSONL per line:
{"query_id":"...", "predicted":[paper objects], "gold":[paper objects]}
Paper identity uses the project's stable-ID-first matcher.  This script does
not claim to reproduce the organizer's unavailable private evaluator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from backend.paper_identity import match_identity


def row_metrics(predicted: list[dict[str, Any]], gold: list[dict[str, Any]], k: int | None = None) -> dict[str, float]:
    papers = predicted[:k] if k is not None else predicted
    matched: set[int] = set()
    for paper in papers:
        hit = next((i for i, target in enumerate(gold) if i not in matched and match_identity(paper, target)), None)
        if hit is not None: matched.add(hit)
    tp = len(matched); p = tp / len(papers) if papers else 0.0; r = tp / len(gold) if gold else 0.0
    return {"precision": p, "recall": r, "f1": 2 * tp / (len(papers) + len(gold)) if papers or gold else 0.0,
            "tp": tp, "predicted": len(papers), "gold": len(gold)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--protocol", choices=["official_set_f1_candidate", "f1_at_20_compat"], default="official_set_f1_candidate")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args(); details=[]
    for line in args.input.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        item=json.loads(line); m=row_metrics(item.get("predicted") or [], item.get("gold") or [], 20 if args.protocol == "f1_at_20_compat" else None)
        details.append({"query_id": item.get("query_id"), **m})
    macro={key:sum(row[key] for row in details)/len(details) for key in ("precision","recall","f1")} if details else {}
    tp=sum(r["tp"] for r in details); pred=sum(r["predicted"] for r in details); gold=sum(r["gold"] for r in details)
    p=tp/pred if pred else 0; r=tp/gold if gold else 0
    result={"protocol":args.protocol,"query_count":len(details),"macro":macro,
            "micro":{"precision":p,"recall":r,"f1":2*p*r/(p+r) if p+r else 0,"tp":tp,"predicted":pred,"gold":gold},"details":details}
    text=json.dumps(result,ensure_ascii=False,indent=2)
    if args.output: args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text,encoding="utf-8")
    print(text)


if __name__ == "__main__": main()
