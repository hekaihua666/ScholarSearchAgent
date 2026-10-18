"""Freeze a new stratified blind holdout before any v4 parameter selection."""

from __future__ import annotations

import hashlib,json,random,sys
from collections import Counter,defaultdict
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from backend.selection.relevance_selector import tokenize
from scripts.audit_evaluation import load_jsonl,source_cutoff

DATASET=Path(r"D:\AICompetition\references\github\SPAR\benchmark\AutoScholarQuery_test.jsonl")
OUT=ROOT/"reports/joint_optimization_v4"
PRIVATE=ROOT/"reports/frozen_eval/query_sets/blind_holdout50_v4_private.jsonl"
SEED=20260831

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def intent(q:str)->str:
 s=q.lower()
 if any(x in s for x in ("survey","review","overview")):return "survey"
 if any(x in s for x in ("latest","recent","newest")):return "frontier"
 if any(x in s for x in ("compare","comparison","versus"," vs ")):return "comparison"
 if any(x in s for x in ("first","introduced","seminal","foundation")):return "foundation"
 return "multi_constraint" if len(tokenize(q))>=12 else "broad"
def constraints(q:str)->int:
 s=q.lower();markers=("using","with","without","before","after","between","dataset","method","task","venue","author","published","application","domain")
 return sum(m in s for m in markers)
def main()->None:
 rows=load_jsonl(DATASET);v3=json.loads((ROOT/"reports/f1_selector_v3/02_selector_split_manifest.json").read_text(encoding="utf8"));sets=json.loads((ROOT/"configs/query_sets.json").read_text(encoding="utf8"))
 excluded=set(sets["dev20"])|set(sets["validation30"])|{x["query_id"] for x in v3["selector_train"]}|{x["query_id"] for x in v3["selector_calibration"]}
 eligible=[r for r in rows if r["qid"] not in excluded and source_cutoff(r) is not None]
 buckets:dict[tuple[str,str,str],list[dict[str,Any]]]=defaultdict(list)
 for r in eligible:
  n=len(r.get("answer") or []);gb="g1" if n<=1 else "g2" if n==2 else "g3plus";lb="short" if len(tokenize(r["question"]))<12 else "long"
  buckets[(gb,intent(r["question"]),lb)].append(r)
 rng=random.Random(SEED)
 for vals in buckets.values():rng.shuffle(vals)
 selected=[];keys=sorted(buckets)
 while len(selected)<50:
  progressed=False
  for k in keys:
   if buckets[k] and len(selected)<50:selected.append(buckets[k].pop());progressed=True
  if not progressed:raise RuntimeError("not enough rows")
 rng.shuffle(selected);PRIVATE.parent.mkdir(parents=True,exist_ok=True)
 with PRIVATE.open("w",encoding="utf8") as f:
  for r in selected:f.write(json.dumps(r,ensure_ascii=False)+"\n")
 public=[]
 for r in selected:
  cutoff=source_cutoff(r);public.append({"query_id":r["qid"],"query":r["question"],"cutoff":cutoff.isoformat(),"intent":intent(r["question"]),"query_length":len(tokenize(r["question"])),"hard_constraint_count_proxy":constraints(r["question"])})
 gold_dist=Counter("g1" if len(r.get("answer") or [])<=1 else "g2" if len(r.get("answer") or [])==2 else "g3plus" for r in selected)
 manifest={"query_set":"blind_holdout50_v4","count":50,"sampling_seed":SEED,"dataset":str(DATASET),"dataset_sha256":sha(DATASET),"private_subset":str(PRIVATE),"private_subset_sha256":sha(PRIVATE),
 "excluded_sets":["dev20","selector_train_500","selector_calibration_150","validation30_consumed"],"excluded_count":len(excluded),"aggregate_gold_strata":dict(gold_dist),"queries":public,
 "overlap_checks":{"dev20":len({r['qid'] for r in selected}&set(sets['dev20'])),"validation30":len({r['qid'] for r in selected}&set(sets['validation30'])),"selector_train":len({r['qid'] for r in selected}&{x['query_id'] for x in v3['selector_train']}),"selector_calibration":0}
 }
 # Correct the last overlap expression separately to keep the public manifest simple.
 manifest["overlap_checks"]["selector_calibration"]=len({r['qid'] for r in selected}&{x['query_id'] for x in v3['selector_calibration']})
 OUT.mkdir(parents=True,exist_ok=True);(OUT/"02_split_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf8")
 sets["blind_holdout50_v4"]=[r["qid"] for r in selected];(ROOT/"configs/query_sets.json").write_text(json.dumps(sets,ensure_ascii=False,indent=2),encoding="utf8")
 print(json.dumps({"count":50,"eligible":len(eligible),"overlap":manifest["overlap_checks"],"private_sha256":manifest["private_subset_sha256"]},indent=2))
if __name__=="__main__":main()
