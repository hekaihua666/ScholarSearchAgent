"""Materialize the immutable v3 fact base and bottleneck decomposition for v4."""

from __future__ import annotations

import csv, json, sys
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from backend.paper_identity import match_identity
from scripts.audit_evaluation import gold_papers,load_jsonl

DATASET=Path(r"D:\AICompetition\references\github\SPAR\benchmark\AutoScholarQuery_test.jsonl")
RUN=ROOT/"reports/frozen_eval/autoscholar20_trace_v4_official_20260824_v2"
OUT=ROOT/"reports/joint_optimization_v4"

def counts(metric:dict[str,Any])->dict[str,int]:
 m=metric["micro"];tp=int(m["true_positive"]);pred=int(m["predicted"]);gold=int(m["gold"])
 return {"tp":tp,"fp":pred-tp,"fn":gold-tp,"predicted":pred,"gold":gold}

def main()->None:
 OUT.mkdir(parents=True,exist_ok=True)
 oracle=json.loads((ROOT/"reports/f1_selector_v3/baseline_oracles.json").read_text(encoding="utf8"))
 validation=json.loads((ROOT/"reports/f1_selector_v3/validation_results.json").read_text(encoding="utf8"))
 rows=load_jsonl(DATASET)[:20];per=[]
 for row in rows:
  qdir=RUN/"queries"/row["qid"]
  response=json.loads((qdir/"live_response.json").read_text(encoding="utf8"));audit=json.loads((qdir/"live_audit.json").read_text(encoding="utf8"))
  final=response["papers"];merged=next(s["papers"] for s in audit["stages"] if s["name"]=="merged_candidates")
  gold=gold_papers(row);present=sum(any(match_identity(p,g) for p in merged) for g in gold)
  top20=sum(any(match_identity(p,g) for p in final) for g in gold)
  k2=sum(any(match_identity(p,g) for p in final[:2]) for g in gold)
  k8=sum(any(match_identity(p,g) for p in final[:8]) for g in gold)
  llm=[p for p in final if (p.get("llm_relevance_score") or 0)>=.9]
  ltp=sum(any(match_identity(p,g) for p in llm) for g in gold)
  per.append({"query_id":row["qid"],"gold":len(gold),"candidate_gold":present,"top20_tp":top20,"k2_tp":k2,"k8_tp":k8,"llm090_tp":ltp,
              "retrieval_loss":len(gold)-present,"ranking_loss":present-top20,"k2_selector_fn":top20-k2,"k8_selector_fn":top20-k8,
              "top20_fp":20-top20,"k2_fp":2-k2,"k8_fp":8-k8,"llm090_output":len(llm),"llm090_fp":len(llm)-ltp})
 with (OUT/"01_bottleneck_by_query.csv").open("w",encoding="utf-8-sig",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(per[0]));w.writeheader();w.writerows(per)
 vr=validation["results"]
 table=[]
 for eid in ["S0_fixed_top20_baseline","f1_champion_fixed_k2","balanced_champion_fixed_k8","best_low_cost_llm_threshold_090"]:
  v=vr[eid];table.append((eid,v["metrics"]["macro"],counts(v["metrics"]),v["mean_output_count"]))
 fact=f"""# v3 统一事实底稿

## dev20 oracle

| 层级 | Macro P | Macro R | Macro F1 | Micro TP |
|---|---:|---:|---:|---:|
| fixed Top20 | {oracle['baseline_fixed20']['macro']['precision']:.4f} | {oracle['baseline_fixed20']['macro']['recall']:.4f} | {oracle['baseline_fixed20']['macro']['f1']:.4f} | {oracle['baseline_fixed20']['micro']['true_positive']} |
| ranking oracle dynamic-k | {oracle['current_ranking_oracle_dynamic_k']['macro']['precision']:.4f} | {oracle['current_ranking_oracle_dynamic_k']['macro']['recall']:.4f} | {oracle['current_ranking_oracle_dynamic_k']['macro']['f1']:.4f} | {oracle['current_ranking_oracle_dynamic_k']['micro']['true_positive']} |
| candidate oracle | {oracle['candidate_oracle']['macro']['precision']:.4f} | {oracle['candidate_oracle']['macro']['recall']:.4f} | {oracle['candidate_oracle']['macro']['f1']:.4f} | {oracle['candidate_oracle']['micro']['true_positive']} |
| metadata-complete oracle | {oracle['metadata_complete_candidate_oracle']['macro']['precision']:.4f} | {oracle['metadata_complete_candidate_oracle']['macro']['recall']:.4f} | {oracle['metadata_complete_candidate_oracle']['macro']['f1']:.4f} | {oracle['metadata_complete_candidate_oracle']['micro']['true_positive']} |

dev20共有58篇gold：candidate pool命中21，Top20命中18；retrieval loss=37，ranking loss=3。

## consumed validation30

| 配置 | Macro P | Macro R | Macro F1 | TP | FP | FN | mean output |
|---|---:|---:|---:|---:|---:|---:|---:|
"""+"\n".join(f"| {eid} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {c['tp']} | {c['fp']} | {c['fn']} | {mean:.2f} |" for eid,m,c,mean in table)+"\n"
 (OUT/"00_v3_fact_base.md").write_text(fact,encoding="utf8")
 sums={k:sum(int(r[k]) for r in per) for k in ["retrieval_loss","ranking_loss","k2_selector_fn","k8_selector_fn","top20_fp","k2_fp","k8_fp","llm090_fp"]}
 decomp=f"""# v4 瓶颈分解

## dev20 总量

- retrieval loss：{sums['retrieval_loss']}篇gold未进入候选池。
- shortlist/ranking loss：{sums['ranking_loss']}篇gold在候选池但未进入Top20。
- K=2 selector FN：Top20已有但被截掉{sums['k2_selector_fn']}篇。
- K=8 selector FN：Top20已有但被截掉{sums['k8_selector_fn']}篇。
- Top20 FP：{sums['top20_fp']}。
- K=2 FP：{sums['k2_fp']}。
- K=8 FP：{sums['k8_fp']}。
- LLM-0.90 FP：{sums['llm090_fp']}。
- metadata/evidence loss：candidate oracle 21篇中有20篇摘要完整，明确缺口1篇。
- identity/date loss：v3审计未发现进入最终结果后的身份或未来日期违规。

## 资源决策

retrieval缺失37篇，是召回上限的最大缺口；Selector的当前直接问题是Top20 FP过多。v4先审计37篇是否能被现有来源取回，同时以LLM-0.90为Selector起点降低FP，不再继续扫固定K。
"""
 (OUT/"01_bottleneck_decomposition.md").write_text(decomp,encoding="utf8")
 print(json.dumps({"reports":["00_v3_fact_base.md","01_bottleneck_decomposition.md","01_bottleneck_by_query.csv"],"totals":sums},ensure_ascii=False,indent=2))

if __name__=="__main__":main()
