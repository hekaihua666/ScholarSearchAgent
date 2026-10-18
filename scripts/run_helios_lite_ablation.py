"""Offline HELIOS-Lite policy ablation on immutable dev20 channel outputs."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from backend.helios_lite import HELIOSLiteController, admit_candidates, build_state
from backend.trace_rank import canonical_key, diversify, fuse_candidates, score_candidates
from scripts.audit_evaluation import aggregate,gold_papers,load_jsonl,metrics_from_papers,source_cutoff
from scripts.run_frozen_official_benchmark import module_candidates,module_cost

CONFIGS=[
 {"experiment_id":"trace_loop_v2","policy":"all"},
 {"experiment_id":"helios_full","policy":"helios","cost":True,"gap":True,"drift":True,"admission":True},
 {"experiment_id":"helios_without_cost","policy":"helios","cost":False,"gap":True,"drift":True,"admission":True},
 {"experiment_id":"helios_without_gap","policy":"helios","cost":True,"gap":False,"drift":True,"admission":True},
 {"experiment_id":"helios_without_drift","policy":"helios","cost":True,"gap":True,"drift":False,"admission":True},
 {"experiment_id":"helios_without_admission","policy":"helios","cost":True,"gap":True,"drift":True,"admission":False},
]


def select_raw(raw:list[dict[str,Any]],understanding:dict[str,Any],enabled:bool)->tuple[list[dict[str,Any]],int]:
    scored=score_candidates(fuse_candidates(raw),understanding)
    accepted,rejected=admit_candidates(scored,enabled=enabled,limit=50)
    keys={canonical_key(p) for p in accepted}
    return [p for p in raw if canonical_key(p) in keys],len(rejected)


def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True);ap.add_argument('--dataset',type=Path,default=Path(r'D:\AICompetition\references\github\SPAR\benchmark\AutoScholarQuery_test.jsonl'));ap.add_argument('--output-dir',type=Path,default=ROOT/'reports/overnight_iteration');args=ap.parse_args()
    rows=load_jsonl(args.dataset)[:20]
    buckets={c['experiment_id']:{'m20':[],'candidate':[],'counts':[],'actions':Counter(),'seconds':0.0,'api':0.0,'trace':[]} for c in CONFIGS}
    for row in rows:
        qdir=args.run_dir/'queries'/row['qid'];snapshot=json.loads((qdir/'snapshot.json').read_text(encoding='utf8'));response=json.loads((qdir/'live_response.json').read_text(encoding='utf8'))
        cutoff=source_cutoff(row);assert cutoff is not None;mods=module_candidates(snapshot,cutoff);u=response.get('query_understanding') or {};gold=gold_papers(row)
        base=[p for name in ('lexical','semantic','arxiv','title') for p in mods[name]]
        base_fused=fuse_candidates(base);base_ranked=score_candidates(copy.deepcopy(base_fused),u);state=build_state(row['question'],u,base_ranked)
        for cfg in CONFIGS:
            chosen=[]; action='stop'; rejected=0; estimate=None
            if cfg['policy']=='all':
                action='citation+second';chosen=mods['citation']+mods['second'];cost_names=['citation','second']
            else:
                ctl=HELIOSLiteController(use_cost=cfg['cost'],use_gap=cfg['gap'],use_drift=cfg['drift']);estimate,_=ctl.choose(state)
                action=estimate.action;cost_names=[]
                if action=='citation_expand':
                    chosen,rejected=select_raw(mods['citation'],u,cfg['admission']);cost_names=['citation']
                elif action=='second_round_search':
                    chosen,rejected=select_raw(mods['second'],u,cfg['admission']);cost_names=['second']
            fused=fuse_candidates(base+chosen);ranked=score_candidates(copy.deepcopy(fused),u);top=diversify(ranked,limit=20,lambda_relevance=.88)
            b=buckets[cfg['experiment_id']];b['m20'].append(metrics_from_papers(top,gold,20));b['candidate'].append(metrics_from_papers(fused,gold,len(fused)));b['counts'].append(len(fused));b['actions'][action]+=1
            for n in cost_names:
                c=module_cost(snapshot,n);b['seconds']+=c['seconds'];b['api']+=c['api_calls']
            b['trace'].append({'query_id':row['qid'],'state_before':state.to_dict(),'action':action,'reason':estimate.reason if estimate else 'TRACE-Loop v2 fixed full expansion','predicted_utility':estimate.predicted_utility if estimate else None,'actual_new_candidates':len(chosen),'actual_accepted_candidates':len(chosen),'actual_rejected':rejected,'actual_cost_seconds':sum(module_cost(snapshot,n)['seconds'] for n in cost_names),'stop_or_continue':'stop_after_one_action'})
    rows_out=[];full={}
    for cfg in CONFIGS:
        b=buckets[cfg['experiment_id']];m=aggregate(b['m20']);c=aggregate(b['candidate']);row={'experiment_id':cfg['experiment_id'],'tp20':m['micro']['true_positive'],'macro_recall20':m['macro']['recall'],'macro_f1_20':m['macro']['f1'],'macro_ndcg20':m['macro']['ndcg'],'candidate_gold':c['micro']['true_positive'],'candidate_recall':c['macro']['recall'],'mean_candidates':round(sum(b['counts'])/20,3),'actions':json.dumps(dict(b['actions']),sort_keys=True),'expansion_seconds':round(b['seconds'],3),'expansion_api_calls':int(b['api'])};rows_out.append(row);full[cfg['experiment_id']]={'config':cfg,'summary':row,'trace':b['trace']};print(row,flush=True)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with (args.output_dir/'helios_lite_ablation.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows_out[0]));w.writeheader();w.writerows(rows_out)
    (args.output_dir/'helios_lite_ablation.json').write_text(json.dumps(full,ensure_ascii=False,indent=2),encoding='utf8')


if __name__=='__main__':main()
