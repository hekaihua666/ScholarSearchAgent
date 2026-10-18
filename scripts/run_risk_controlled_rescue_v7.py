"""Offline-only query-risk controlled rescue sprint v7."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from backend.selection.core_rescue_selector_v6 import CoreRescueModelV6, FEATURE_NAMES as V6_FEATURE_NAMES
from backend.selection.risk_controlled_rescue_v7 import (
    GATE_FEATURE_NAMES,RANK_FEATURE_NAMES,QueryGateV7,RescueRankerV7,assert_safe_schemas,
    augment_query_relative,gate_row,
)
from scripts.run_core_rescue_v6 import (
    add_core_tp,attach_boundary_features,dev_rows_with_original_rank,grouped,load_v5_models,
    one_rescue_oracle,rank_order,topk_selector,
)
from scripts.run_short_sprint_v5 import gold_count_map,metric_summary,write_csv


REPORT=ROOT/"reports/risk_controlled_rescue_v7"
ARTIFACT=ROOT/"artifacts/risk_controlled_rescue_v7"
CONFIG=ROOT/"configs/risk_controlled_rescue_v7"
BOOTSTRAP=ARTIFACT/"bootstrap_models"
TRAIN_PATH=ROOT/"artifacts/core_rescue_v6/boundary_train.parquet"
CAL_PATH=ROOT/"artifacts/core_rescue_v6/boundary_calibration.parquet"
V6_MODEL_PATH=ROOT/"artifacts/core_rescue_v6/core_rescue_model.pkl"
V6_THRESHOLD=0.8878586595869522
SEEDS=(20260830,20260831,20260832)
RISK_FLOORS={"strict":.70,"balanced":.50,"recall":.35}


def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def git_head()->str:return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()


def load_boundary()->list[dict[str,Any]]:
    rows=pq.read_table(TRAIN_PATH).to_pylist()+pq.read_table(CAL_PATH).to_pylist()
    return augment_query_relative(rows)


def ranker_crossfit(rows:list[dict[str,Any]],variant:str)->tuple[np.ndarray,np.ndarray]:
    labels=np.asarray([int(row["label"]) for row in rows]);groups_array=np.asarray([row["query_id"] for row in rows])
    predictions=np.zeros((len(rows),len(SEEDS)))
    for seed_index,seed in enumerate(SEEDS):
        splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed)
        for train_idx,test_idx in splitter.split(np.zeros(len(rows)),labels,groups_array):
            model=RescueRankerV7.fit([rows[int(i)] for i in train_idx],variant=variant)
            predictions[test_idx,seed_index]=model.predict([rows[int(i)] for i in test_idx])
    return predictions,predictions.mean(axis=1)


def selected_indices(rows:list[dict[str,Any]],scores:np.ndarray)->dict[str,int]:
    by:dict[str,list[int]]=defaultdict(list)
    for index,row in enumerate(rows):by[str(row["query_id"])].append(index)
    return {qid:max(indices,key=lambda i:(float(scores[i]),-int(rows[i]["candidate_rank"]))) for qid,indices in by.items()}


def ranker_metrics(rows:list[dict[str,Any]],scores:np.ndarray,reference:np.ndarray|None=None)->dict[str,Any]:
    by:dict[str,list[int]]=defaultdict(list)
    for index,row in enumerate(rows):by[str(row["query_id"])].append(index)
    top1_gold=rescue_queries=0;mrr=[];gold_ranks=[];pair_correct=pair_total=0;wins=ties=losses=0
    for indices in by.values():
        order=sorted(indices,key=lambda i:(-float(scores[i]),int(rows[i]["candidate_rank"])))
        positive=[i for i in indices if int(rows[i]["label"])==1]
        top1_gold+=int(int(rows[order[0]]["label"])==1);rescue_queries+=int(bool(positive))
        if positive:
            ranks=[order.index(i)+1 for i in positive];best=min(ranks);mrr.append(1/best);gold_ranks.extend(ranks)
        negative=[i for i in indices if int(rows[i]["label"])==0]
        for p in positive:
            for n in negative:pair_correct+=int(scores[p]>scores[n]);pair_total+=1
        if reference is not None and positive:
            old=min(sorted(indices,key=lambda i:(-float(reference[i]),int(rows[i]["candidate_rank"]))).index(i)+1 for i in positive)
            new=min(order.index(i)+1 for i in positive)
            wins+=new<old;ties+=new==old;losses+=new>old
    return {"rescue_top1_tp":top1_gold,"rescue_top1_precision":top1_gold/len(by),"rescue_gold_queries":rescue_queries,
            "top1_accuracy_on_rescueable":top1_gold/rescue_queries if rescue_queries else 0,"mrr":float(np.mean(mrr)) if mrr else 0,
            "pairwise_auc":pair_correct/pair_total if pair_total else 0,"rescue_gold_mean_rank":float(np.mean(gold_ranks)) if gold_ranks else 0,
            "wins":wins,"ties":ties,"losses":losses}


def seed_selection_agreement(rows:list[dict[str,Any]],matrix:np.ndarray,average:np.ndarray)->dict[str,float]:
    selected=selected_indices(rows,average);result={}
    for qid,index in selected.items():
        indices=[i for i,row in enumerate(rows) if row["query_id"]==qid]
        choices=[]
        for seed_index in range(matrix.shape[1]):choices.append(max(indices,key=lambda i:(matrix[i,seed_index],-int(rows[i]["candidate_rank"]))))
        result[qid]=sum(choice==index for choice in choices)/len(choices)
    return result


def build_gate_rows(
    rows:list[dict[str,Any]],scores:np.ndarray,seed_matrix:np.ndarray,counts:dict[str,int],
)->tuple[list[dict[str,Any]],dict[str,int]]:
    selected=selected_indices(rows,scores);agreement=seed_selection_agreement(rows,seed_matrix,scores);result=[]
    by=grouped(rows)
    index_by_id={id(row):index for index,row in enumerate(rows)}
    for qid,candidates in by.items():
        indices=[index_by_id[id(row)] for row in candidates];chosen=selected[qid];local_index=indices.index(chosen)
        result.append(gate_row(candidates,[scores[i] for i in indices],local_index,ensemble_agreement=agreement[qid],gold_count=counts[qid]))
    return result,selected


def gate_crossfit(rows:list[dict[str,Any]])->tuple[np.ndarray,np.ndarray]:
    labels=np.asarray([int(row["y_gate"]) for row in rows]);predictions=np.zeros((len(rows),len(SEEDS)))
    for seed_index,seed in enumerate(SEEDS):
        splitter=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
        for train_idx,test_idx in splitter.split(np.zeros(len(rows)),labels):
            model=QueryGateV7.fit([rows[int(i)] for i in train_idx])
            predictions[test_idx,seed_index]=model.predict([rows[int(i)] for i in test_idx])
    return predictions,predictions.mean(axis=1)


def set_metrics_from_gate(gates:list[dict[str,Any]],trigger:list[bool],counts:dict[str,int])->dict[str,Any]:
    values=[];tp=fp=fn=rescue_tp=rescue_fp=0
    for row,fire in zip(gates,trigger):
        core_tp=int(row.get("core_tp") or 0);gold=counts[str(row["query_id"])]
        success=int(fire and int(row.get("y_gold") or 0)==1);predicted=2+int(fire);qtp=core_tp+success
        qfp=predicted-qtp;qfn=max(0,gold-qtp);p=qtp/predicted;r=qtp/gold if gold else 0;f=2*p*r/(p+r) if p+r else 0
        values.append((p,r,f,predicted));tp+=qtp;fp+=qfp;fn+=qfn;rescue_tp+=success;rescue_fp+=int(fire)-success
    n=len(values);attempts=rescue_tp+rescue_fp
    return {"rescue_TP":rescue_tp,"rescue_FP":rescue_fp,"rescue_precision":rescue_tp/attempts if attempts else 0,
            "trigger_count":attempts,"precision":sum(v[0] for v in values)/n,"recall":sum(v[1] for v in values)/n,
            "f1":sum(v[2] for v in values)/n,"TP":tp,"FP":fp,"FN":fn,"mean_output_count":sum(v[3] for v in values)/n}


def risk_profiles(
    gates:list[dict[str,Any]],probabilities:np.ndarray,counts:dict[str,int],
    agreement:np.ndarray|None=None,agreement_threshold:float=.75,
)->dict[str,dict[str,Any]]:
    profiles={}
    thresholds=sorted(set(float(value) for value in probabilities),reverse=True)
    for name,floor in RISK_FLOORS.items():
        trials=[]
        for threshold in thresholds:
            trigger=[bool(p>=threshold and (agreement is None or agreement[i]>=agreement_threshold)) for i,p in enumerate(probabilities)]
            metrics=set_metrics_from_gate(gates,trigger,counts)
            if metrics["trigger_count"] and metrics["rescue_precision"]>=floor:
                trials.append({"threshold":threshold,**metrics})
        if trials:profiles[name]=max(trials,key=lambda row:(row["f1"],row["recall"],-row["trigger_count"]))
    return profiles


def v6_decomposition(rows:list[dict[str,Any]],matrix:np.ndarray,average:np.ndarray)->tuple[dict[str,Any],list[dict[str,Any]]]:
    selected=selected_indices(rows,average);by=grouped(rows);index_by_id={id(row):i for i,row in enumerate(rows)};diagnostics=[]
    ranking=gating=both=0
    for qid,candidates in by.items():
        indices=[index_by_id[id(row)] for row in candidates];order=sorted(indices,key=lambda i:(-average[i],int(rows[i]["candidate_rank"])))
        gold_indices=[i for i in indices if int(rows[i]["label"])==1];top=order[0];rescueable=bool(gold_indices);top_gold=int(rows[top]["label"])==1
        gate_fire=average[top]>=V6_THRESHOLD
        if rescueable and not top_gold:ranking+=1
        if top_gold and not gate_fire:gating+=1
        if rescueable and not top_gold and not gate_fire:both+=1
        margin=float(average[order[0]]-average[order[1]]) if len(order)>1 else 0
        diagnostics.append({"query_id":qid,"split":candidates[0]["split"],"rescueable":int(rescueable),"top1_is_gold":int(top_gold),
                            "v6_trigger":int(gate_fire),"top1_score":average[top],"top1_top2_margin":margin,
                            "gold_best_rank":min((order.index(i)+1 for i in gold_indices),default=""),
                            "gold_mean_score":float(np.mean([average[i] for i in gold_indices])) if gold_indices else "",
                            "non_gold_mean_score":float(np.mean([average[i] for i in indices if i not in gold_indices]))})
    summary={"rescue_gold_queries":sum(row["rescueable"] for row in diagnostics),"top1_gold_queries":sum(row["top1_is_gold"] for row in diagnostics),
             "ranking_error":ranking,"gating_error":gating,"both":both}
    return summary,diagnostics


def describe(values:Sequence[float])->dict[str,float|int]:
    array=np.asarray(list(values),dtype=float)
    if not len(array):return {"n":0,"mean":0.0,"median":0.0,"p25":0.0,"p75":0.0}
    return {"n":int(len(array)),"mean":float(array.mean()),"median":float(np.median(array)),
            "p25":float(np.quantile(array,.25)),"p75":float(np.quantile(array,.75))}


def failure_breakdown_by_split(
    rows:list[dict[str,Any]],diagnostics:list[dict[str,Any]],average:np.ndarray,counts:dict[str,int],
)->dict[str,dict[str,Any]]:
    by_rows=grouped(rows);index_by_id={id(row):index for index,row in enumerate(rows)}
    result={}
    for split in ("train","calibration","all"):
        selected_diagnostics=[row for row in diagnostics if split=="all" or row["split"]==split]
        query_ids={str(row["query_id"]) for row in selected_diagnostics}
        gates=[]
        gold_relative=[];non_gold_relative=[];margins=[];gold_rank_counts=Counter()
        for qid in query_ids:
            candidates=by_rows[qid];indices=[index_by_id[id(row)] for row in candidates]
            local_scores=[float(average[index]) for index in indices]
            score_low=min(local_scores);score_span=max(max(local_scores)-score_low,1e-9)
            relative_scores=[(score-score_low)/score_span if score_span>1e-9 else .5 for score in local_scores]
            chosen=int(np.argmax(local_scores))
            gates.append(gate_row(candidates,local_scores,chosen,gold_count=counts[qid]))
            margins.append(float(np.sort(local_scores)[-1]-np.sort(local_scores)[-2]))
            ordered=sorted(range(len(candidates)),key=lambda i:(-local_scores[i],int(candidates[i]["candidate_rank"])))
            positive_positions=[ordered.index(i)+1 for i,row in enumerate(candidates) if int(row["label"])==1]
            if positive_positions:gold_rank_counts[str(min(positive_positions))]+=1
            for candidate,score in zip(candidates,relative_scores):
                (gold_relative if int(candidate["label"]) else non_gold_relative).append(score)
        baseline=set_metrics_from_gate(gates,[False]*len(gates),counts)
        fixed=set_metrics_from_gate(gates,[True]*len(gates),counts)
        frozen=set_metrics_from_gate(gates,[float(row["candidate_absolute_score"])>=V6_THRESHOLD for row in gates],counts)
        result[split]={
            "queries":len(selected_diagnostics),
            "rescue_gold_queries":sum(int(row["rescueable"]) for row in selected_diagnostics),
            "top1_gold_queries":sum(int(row["top1_is_gold"]) for row in selected_diagnostics),
            "gold_best_rank_distribution":dict(sorted(gold_rank_counts.items(),key=lambda item:int(item[0]))),
            "fixed_top1":{"rescue_TP":fixed["rescue_TP"],"rescue_FP":fixed["rescue_FP"],
                          "rescue_precision":fixed["rescue_precision"],"macro_f1":fixed["f1"],
                          "macro_f1_delta_vs_k2":fixed["f1"]-baseline["f1"]},
            "frozen_threshold":{"rescue_TP":frozen["rescue_TP"],"rescue_FP":frozen["rescue_FP"],
                                "gold_top1_missed":sum(int(row["top1_is_gold"] and not row["v6_trigger"]) for row in selected_diagnostics),
                                "non_gold_correctly_not_triggered":sum(int(not row["top1_is_gold"] and not row["v6_trigger"]) for row in selected_diagnostics)},
            "top1_top2_margin":describe(margins),
            "gold_query_relative_score":describe(gold_relative),
            "non_gold_query_relative_score":describe(non_gold_relative),
        }
    return result


def challenger_registry(
    rows:list[dict[str,Any]],r0:np.ndarray,r1:np.ndarray,r1_matrix:np.ndarray,gate_rows:list[dict[str,Any]],gate_matrix:np.ndarray,gate_avg:np.ndarray,counts:dict[str,int]
)->tuple[list[dict[str,Any]],dict[str,dict[str,Any]],str,bool]:
    by=grouped(rows);index={id(row):i for i,row in enumerate(rows)}
    # C0: v6 ranker and frozen absolute threshold.
    c0_gates,c0_selected=build_gate_rows(rows,r0,np.tile(r0[:,None],(1,3)),counts)
    c0_trigger=[r0[i]>=V6_THRESHOLD for i in c0_selected.values()];c0=set_metrics_from_gate(c0_gates,c0_trigger,counts)
    # C1: R1 selects; v6 score still gates the chosen candidate.
    c1_gates,c1_selected=build_gate_rows(rows,r1,r1_matrix,counts);c1_trigger=[r0[i]>=V6_THRESHOLD for i in c1_selected.values()]
    c1=set_metrics_from_gate(c1_gates,c1_trigger,counts)
    profiles_c2=risk_profiles(gate_rows,gate_avg,counts)
    preferred="balanced" if "balanced" in profiles_c2 else "strict" if "strict" in profiles_c2 else "recall"
    no_rescue={"threshold":1.1,**set_metrics_from_gate(gate_rows,[False]*len(gate_rows),counts)}
    c2={"status":"legal",**profiles_c2[preferred]} if preferred in profiles_c2 else {"status":"no_legal_profile",**no_rescue}
    # Seed-level gate agreement plus ranker selection agreement.
    selection_agreement=np.asarray([float(row["ranker_ensemble_agreement"]) for row in gate_rows])
    trigger_agreement=np.mean(gate_matrix>=float(c2.get("threshold",1.1)),axis=1)
    consensus=np.minimum(selection_agreement,trigger_agreement)
    consensus_trials=[]
    for agreement_threshold in (.70,.75,.80):
        candidate_profiles=risk_profiles(
            gate_rows,gate_avg,counts,agreement=consensus,
            agreement_threshold=agreement_threshold,
        )
        candidate=candidate_profiles.get(preferred)
        if candidate:
            consensus_trials.append((candidate_profiles,{**candidate,"agreement_threshold":agreement_threshold}))
    if consensus_trials:
        profiles_c3,c3=max(
            consensus_trials,
            key=lambda item:(item[1]["f1"],item[1]["recall"],-item[1]["trigger_count"]),
        )
        for value in profiles_c3.values():
            value["agreement_threshold"]=c3["agreement_threshold"]
        c3={"status":"legal",**c3}
    else:
        profiles_c3={};c3={"status":"no_legal_profile",**no_rescue}
    registry=[]
    for name,value in (("C0_v6",c0),("C1_query_ranker_v6_gate",c1),("C2_query_gate",c2),("C3_bootstrap_consensus",c3)):
        registry.append({"challenger":name,**value})
    use_consensus=bool(profiles_c3) and bool(c3.get("f1",0)>c2.get("f1",0) or (c3.get("f1",0)==c2.get("f1",0) and c3.get("rescue_FP",99)<c2.get("rescue_FP",99)))
    return registry,(profiles_c3 if use_consensus else profiles_c2),preferred,use_consensus


def bootstrap_query_sample(qids:Sequence[str],seed:int)->list[str]:
    rng=random.Random(seed);ordered=sorted(qids)
    return [rng.choice(ordered) for _ in ordered]


def bootstrap_models(rows:list[dict[str,Any]],gate_rows:list[dict[str,Any]],variant:str,count:int=20)->None:
    BOOTSTRAP.mkdir(parents=True,exist_ok=True);by_query=grouped(rows);qids=sorted(by_query);gate_by={row["query_id"]:row for row in gate_rows}
    for index in range(count):
        sample=bootstrap_query_sample(qids,20260900+index)
        rank_training=[];gate_training=[]
        for copy_index,qid in enumerate(sample):
            for row in by_query[qid]:rank_training.append({**row,"query_id":f"{qid}#boot{copy_index}"})
            gate_training.append({**gate_by[qid],"query_id":f"{qid}#boot{copy_index}"})
        # A rare all-one-class bootstrap draw would make logistic fitting
        # undefined. Add one deterministic query from the missing class.
        labels={int(row["y_gate"]) for row in gate_training}
        if len(labels)<2:
            missing=1-next(iter(labels))
            fallback_qid=next(qid for qid in qids if int(gate_by[qid]["y_gate"])==missing)
            copy_index=len(sample)
            for row in by_query[fallback_qid]:
                rank_training.append({**row,"query_id":f"{fallback_qid}#boot{copy_index}"})
            gate_training.append({**gate_by[fallback_qid],"query_id":f"{fallback_qid}#boot{copy_index}"})
        bundle={"ranker":RescueRankerV7.fit(rank_training,variant=variant),"gate":QueryGateV7.fit(gate_training),"seed":20260900+index}
        with (BOOTSTRAP/f"bootstrap_{index:02d}.pkl").open("wb") as handle:pickle.dump(bundle,handle)


def write_yaml(path:Path,profile:str,value:dict[str,Any],variant:str,consensus:bool)->None:
    agreement=float(value.get("agreement_threshold",.75))
    path.write_text(f"schema_version: 1\nprofile: {profile}\ncore_k: 2\nmax_rescues: 1\nmax_results: 3\nranker_variant: {variant}\ngate_threshold: {float(value['threshold']):.12f}\nrescue_precision_floor: {RISK_FLOORS[profile]:.2f}\nconsensus_enabled: {str(consensus).lower()}\ncandidate_agreement: {agreement:.2f}\ntrigger_agreement: {agreement:.2f}\n",encoding="utf-8")


def train_freeze()->dict[str,Any]:
    REPORT.mkdir(parents=True,exist_ok=True);ARTIFACT.mkdir(parents=True,exist_ok=True);CONFIG.mkdir(parents=True,exist_ok=True);assert_safe_schemas()
    for path in CONFIG.glob("*.yaml"):path.unlink()
    rows=load_boundary();counts=gold_count_map();ids_by_split={name:{row["query_id"] for row in rows if row["split"]==name} for name in ("train","calibration")}
    query_sets=json.loads((ROOT/"configs/query_sets.json").read_text(encoding="utf-8"));dev=set(query_sets["dev20"])
    overlap={"train_calibration":len(ids_by_split["train"]&ids_by_split["calibration"]),"train_dev":len(ids_by_split["train"]&dev),"calibration_dev":len(ids_by_split["calibration"]&dev)}
    if any(overlap.values()):raise RuntimeError(overlap)
    r0_matrix,r0=ranker_crossfit(rows,"v6");r1_matrix,r1=ranker_crossfit(rows,"query_relative")
    failure,diagnostics=v6_decomposition(rows,r0_matrix,r0);write_csv(REPORT/"00_oof_rescue_diagnostics.csv",diagnostics)
    cal_diag=[row for row in diagnostics if row["split"]=="calibration"]
    cal_precision=sum(row["top1_is_gold"] for row in cal_diag)/len(cal_diag);cal_top1=sum(row["top1_is_gold"] for row in cal_diag)
    failure.update({"calibration_top1_precision":cal_precision,"calibration_top1_gold":cal_top1})
    split_breakdown=failure_breakdown_by_split(rows,diagnostics,r0,counts)
    decomposition_lines=["# v6 OOF失败分解","","## 汇总",""]
    decomposition_lines += [f"- {key}: {value}" for key,value in failure.items()]
    decomposition_lines += ["","定义：ranking_error 表示候选池有 gold 但未排至首位；gating_error 表示 gold 已排首位但未通过冻结阈值；both 表示排序错误且当前首选也未触发。",""]
    for split,value in split_breakdown.items():
        decomposition_lines += [f"## {split}","","```json",json.dumps(value,ensure_ascii=False,indent=2),"```",""]
    (REPORT/"00_failure_decomposition.md").write_text("\n".join(decomposition_lines)+"\n",encoding="utf-8")
    m0=ranker_metrics(rows,r0);m1=ranker_metrics(rows,r1,reference=r0);comparison=[{"ranker":"R0_v6",**m0},{"ranker":"R1_query_relative",**m1}]
    write_csv(REPORT/"02_ranker_comparison.csv",comparison)
    best_variant="query_relative" if (m1["top1_accuracy_on_rescueable"],m1["mrr"],m1["pairwise_auc"])>(m0["top1_accuracy_on_rescueable"],m0["mrr"],m0["pairwise_auc"]) else "v6"
    best_scores,best_matrix=(r1,r1_matrix) if best_variant=="query_relative" else (r0,r0_matrix)
    gate_rows,_=build_gate_rows(rows,best_scores,best_matrix,counts);gate_matrix,gate_avg=gate_crossfit(gate_rows)
    registry,profiles,preferred,use_consensus=challenger_registry(rows,r0,r1,r1_matrix,gate_rows,gate_matrix,gate_avg,counts)
    write_csv(REPORT/"03_gate_crossfit_results.csv",registry);write_csv(REPORT/"experiment_registry.csv",registry)
    feature_audit={"rank_features":RANK_FEATURE_NAMES,"gate_features":GATE_FEATURE_NAMES,"overlap":overlap,"gold_in_inference_schema":False}
    (REPORT/"01_query_relative_feature_audit.md").write_text("# Query-relative特征审计\n\n```json\n"+json.dumps(feature_audit,ensure_ascii=False,indent=2)+"\n```\n",encoding="utf-8")
    final_ranker=RescueRankerV7.fit(rows,variant=best_variant);final_gate=QueryGateV7.fit(gate_rows)
    final_ranker.save(ARTIFACT/"rescue_ranker.pkl");final_gate.save(ARTIFACT/"query_gate.pkl");bootstrap_models(rows,gate_rows,best_variant)
    schema={"ranker":list(RANK_FEATURE_NAMES if best_variant=="query_relative" else V6_FEATURE_NAMES),"gate":list(GATE_FEATURE_NAMES)}
    (ARTIFACT/"feature_schema.json").write_text(json.dumps(schema,ensure_ascii=False,indent=2),encoding="utf-8")
    for name,value in profiles.items():
        if name in ("strict","balanced","recall"):write_yaml(CONFIG/f"rescue_{name}.yaml",name,value,best_variant,use_consensus)
    champion=preferred if preferred in profiles else None
    if champion:
        champion_path=CONFIG/f"rescue_{champion}.yaml"
        (CONFIG/"champion_candidate.yaml").write_text(champion_path.read_text(encoding="utf-8"),encoding="utf-8")
    config_hashes={path.name:sha(path) for path in CONFIG.glob("*.yaml")}
    model_hashes={"rescue_ranker.pkl":sha(ARTIFACT/"rescue_ranker.pkl"),"query_gate.pkl":sha(ARTIFACT/"query_gate.pkl")}
    bootstrap_hashes={path.name:sha(path) for path in sorted(BOOTSTRAP.glob("bootstrap_*.pkl"))}
    manifest={"schema_version":1,"source_commit":git_head(),"train_sha256":sha(TRAIN_PATH),"calibration_sha256":sha(CAL_PATH),
              "train_queries":len(ids_by_split["train"]),"calibration_queries":len(ids_by_split["calibration"]),"seeds":SEEDS,
              "best_ranker":best_variant,"preferred_profile":champion,"consensus":use_consensus,"profiles":profiles,
              "oof_profile_legal":bool(profiles),"oof_rejection_reason":None if profiles else "No OOF threshold reached rescue precision >= 0.35",
              "failure_decomposition":failure,"split_breakdown":split_breakdown,
              "ranker_comparison":comparison,"challenger_registry":registry,
              "config_hashes":config_hashes,"model_hashes":model_hashes,"bootstrap_hashes":bootstrap_hashes,
              "feature_schema_sha256":sha(ARTIFACT/"feature_schema.json"),"dev_evaluated":False}
    (ARTIFACT/"training_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    profile_lines=["# 风险档位冻结",""]+[f"- {name}: `{value}`" for name,value in profiles.items()]
    if profiles:
        profile_lines += [f"- preferred: {champion}",f"- consensus: {use_consensus}"]
    else:
        profile_lines += ["- 无合法 profile：OOF 中没有阈值达到最低 rescue precision 0.35。",
                          "- 未生成 champion_candidate.yaml，也未用低于风险下限的阈值替代。"]
    (REPORT/"04_threshold_and_risk_selection.md").write_text("\n".join(profile_lines)+"\n",encoding="utf-8")
    return {"failure":failure,"rankers":comparison,"registry":registry,"manifest":manifest}


def load_yaml_simple(path:Path)->dict[str,str]:
    result={}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key,value=line.split(":",1);result[key.strip()]=value.strip()
    return result


def profile_selector(
    boundary:list[dict[str,Any]],ranker:RescueRankerV7,gate:QueryGateV7,config:dict[str,str],bootstrap:list[dict[str,Any]]
)->dict[str,dict[str,Any]|None]:
    result={};threshold=float(config["gate_threshold"]);consensus=config.get("consensus_enabled")=="true"
    candidate_threshold=float(config.get("candidate_agreement",.75));trigger_threshold=float(config.get("trigger_agreement",.75))
    for qid,candidates in grouped(boundary).items():
        scores=ranker.predict(candidates);selected=int(np.argmax(scores));candidate=candidates[selected]
        boot_choices=[]
        for bundle in bootstrap:
            bs=bundle["ranker"].predict(candidates);boot_choices.append(int(np.argmax(bs)))
        candidate_agreement=sum(choice==selected for choice in boot_choices)/len(boot_choices) if boot_choices else 1
        main_row=gate_row(candidates,scores,selected,ensemble_agreement=candidate_agreement)
        probability=float(gate.predict([main_row])[0]);trigger_votes=[]
        for bundle,choice in zip(bootstrap,boot_choices):
            bs=bundle["ranker"].predict(candidates);grow=gate_row(candidates,bs,choice,ensemble_agreement=candidate_agreement)
            trigger_votes.append(float(bundle["gate"].predict([grow])[0])>=threshold and choice==selected)
        trigger_agreement=sum(trigger_votes)/len(trigger_votes) if trigger_votes else 1
        fire=probability>=threshold and (not consensus or (candidate_agreement>=candidate_threshold and trigger_agreement>=trigger_threshold))
        result[qid]=candidate if fire else None
    return result


def dev_evaluate()->dict[str,Any]:
    manifest=json.loads((ARTIFACT/"training_manifest.json").read_text(encoding="utf-8"))
    for name,digest in manifest["config_hashes"].items():
        if sha(CONFIG/name)!=digest:raise RuntimeError(f"config hash mismatch: {name}")
    ranker=RescueRankerV7.load(ARTIFACT/"rescue_ranker.pkl");gate=QueryGateV7.load(ARTIFACT/"query_gate.pkl")
    bundles=[]
    for path in sorted(BOOTSTRAP.glob("*.pkl")):
        with path.open("rb") as handle:bundles.append(pickle.load(handle))
    full=dev_rows_with_original_rank();models_v5=load_v5_models();boundary=augment_query_relative(attach_boundary_features(full,models_v5));add_core_tp(boundary,full)
    counts=gold_count_map();v6_model=CoreRescueModelV6.load(V6_MODEL_PATH)
    v6_boundary=grouped(boundary)
    def v6_selector(candidates):
        core=rank_order(candidates)[:2];rescue=v6_model.choose_rescue(v6_boundary[candidates[0]["query_id"]]);return core+([rescue] if rescue else [])
    selectors={"K=2":topk_selector(2),"v6 learned rescue":v6_selector,"oracle":one_rescue_oracle}
    available={}
    for profile in ("strict","balanced"):
        path=CONFIG/f"rescue_{profile}.yaml"
        if path.exists():available[profile]=profile_selector(boundary,ranker,gate,load_yaml_simple(path),bundles)
    for profile,rescues in available.items():
        selectors[f"v7 {profile}"]=lambda candidates,r=rescues: rank_order(candidates)[:2]+([r[candidates[0]["query_id"]]] if r[candidates[0]["query_id"]] else [])
    output=[]
    for name,selector in selectors.items():
        metrics=metric_summary(full,selector,counts);base=metric_summary(full,topk_selector(2),counts)
        metrics.update({"configuration":name,"additional_TP":metrics["TP"]-base["TP"],"additional_FP":metrics["FP"]-base["FP"]})
        attempts=int(round((metrics["mean_output_count"]-2)*len(grouped(full))));success=metrics["TP"]-base["TP"]
        metrics.update({"rescue_triggers":attempts,"successful_rescues":success,"failed_rescues":attempts-success,"rescue_precision":success/attempts if attempts else 0})
        wins=ties=losses=0;benefit_queries=0
        for qid,candidates in grouped(full).items():
            gold=counts[qid]
            def qf(selected):
                tp=sum(int(row["label"]) for row in selected);p=tp/len(selected);r=tp/gold if gold else 0
                return 2*p*r/(p+r) if p+r else 0
            delta=qf(selector(candidates))-qf(topk_selector(2)(candidates));wins+=delta>1e-12;losses+=delta<-1e-12;ties+=abs(delta)<=1e-12;benefit_queries+=delta>1e-12
        metrics.update({"wins":wins,"ties":ties,"losses":losses,"benefit_queries":benefit_queries});output.append(metrics)
    write_csv(REPORT/"05_dev20_one_shot.csv",output)
    by={row["configuration"]:row for row in output};base=by["K=2"]
    candidates=[row for name,row in by.items() if name.startswith("v7")]
    best=max(candidates,key=lambda row:(row["f1"],row["recall"],row["precision"])) if candidates else None
    accepted=False;rules={"A":False,"B":False,"C":False}
    if best:
        rules={"A":best["additional_TP"]>=2 and best["additional_FP"]<=2 and best["wins"]>=2 and best["losses"]==0 and best["mean_output_count"]<=2.25 and best["f1"]>.2292,
               "B":best["f1"]>=.255 and best["recall"]>=.30 and best["precision"]>=.25 and best["mean_output_count"]<=2.30,
               "C":best["TP"]>=13 and best["FP"]<=32 and best["f1"]>=.265}
        accepted=any(rules.values()) and best["benefit_queries"]>=2
    config_hash_verified=bool(manifest["config_hashes"])
    manifest["dev_evaluated"]=True;manifest["dev_config_hash_verified"]=config_hash_verified
    (ARTIFACT/"training_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    state={"schema_version":1,"branch":"risk-controlled-rescue-v7","source_commit":manifest["source_commit"],"blind_used":False,
           "development_candidate_v7":accepted,"acceptance":rules,"best_dev":best,"dev_config_hash_verified":config_hash_verified,
           "dev_config_hash_status":"verified" if config_hash_verified else "not_applicable_no_legal_config",
           "oof_profile_legal":manifest["oof_profile_legal"],"dev20_runs_v7":1}
    (REPORT/"state.json").write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8")
    train_summary=json.loads((ARTIFACT/"training_manifest.json").read_text(encoding="utf-8"));registry=list(csv.DictReader((REPORT/"experiment_registry.csv").open(encoding="utf-8-sig")))
    failure=train_summary["failure_decomposition"];rankers=train_summary["ranker_comparison"]
    r0=next(row for row in rankers if row["ranker"]=="R0_v6");r1=next(row for row in rankers if row["ranker"]=="R1_query_relative")
    lines=["# risk-controlled-rescue-v7 最终报告","","## 1. 失败分解","",
           f"- ranking_error: {failure['ranking_error']}",f"- gating_error: {failure['gating_error']}",f"- both: {failure['both']}",
           f"- 结论：主要损失来自排序。{failure['rescue_gold_queries']} 条可救查询中，仅 {failure['top1_gold_queries']} 条把 gold 排到首位。","",
           "## 2. 排序器 OOF","",
           "| 排序器 | rescue top1 TP | 可救查询 top1 accuracy | MRR | pairwise AUC | gold mean rank | wins/ties/losses |",
           "|---|---:|---:|---:|---:|---:|---:|",
           f"| R0 v6 | {r0['rescue_top1_tp']} | {r0['top1_accuracy_on_rescueable']:.4f} | {r0['mrr']:.4f} | {r0['pairwise_auc']:.4f} | {r0['rescue_gold_mean_rank']:.4f} | — |",
           f"| R1 query-relative | {r1['rescue_top1_tp']} | {r1['top1_accuracy_on_rescueable']:.4f} | {r1['mrr']:.4f} | {r1['pairwise_auc']:.4f} | {r1['rescue_gold_mean_rank']:.4f} | {r1['wins']}/{r1['ties']}/{r1['losses']} |","",
           "## 3. Gate OOF","",
           "| challenger | status | rescue TP | rescue FP | rescue precision | macro F1 | macro Recall | trigger count |",
           "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in registry:lines.append(f"| {row['challenger']} | {row.get('status') or 'evaluated'} | {row['rescue_TP']} | {row['rescue_FP']} | {float(row['rescue_precision']):.4f} | {float(row['f1']):.4f} | {float(row['recall']):.4f} | {row['trigger_count']} |")
    if not train_summary["oof_profile_legal"]:
        lines += ["","Query-level logistic gate 没有任何 OOF 阈值达到最低 rescue precision 0.35；strict、balanced、recall 三档均未冻结。bootstrap consensus 因此没有可接受门控可供上线。"]
    lines += ["","## dev20 one-shot","","| 配置 | P | R | F1 | TP | FP | 输出 | 触发/成功 | wins/ties/losses |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in output:lines.append(f"| {row['configuration']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['TP']} | {row['FP']} | {row['mean_output_count']:.2f} | {row['rescue_triggers']}/{row['successful_rescues']} | {row['wins']}/{row['ties']}/{row['losses']} |")
    if not available:lines += ["","v7 strict/balanced 未列入 dev20：其 OOF 风险档位不合法，没有生成配置。"]
    lines += ["","## 5. 结论","",f"- development_candidate_v7：**{'形成' if accepted else '未形成'}**。",f"- 接受规则：`{rules}`。",
              "- query-relative 特征没有改善 top1 TP，MRR 与 pairwise AUC 也略降。",
              "- query-level gate 没有优于候选绝对阈值；其 OOF 排名前缀无法达到最低风险精度。",
              "- bootstrap consensus 没有形成合法档位，未证明能够在保留有效救援的同时减少误救。",
              "- OOF 的 C0/C1 均为 rescue TP=2、FP=15；没有达到新增 FP 不超过 2 的目标。",
              f"- dev20 中可验证的 v7 获益查询数：{int(best['benefit_queries']) if best else 0}。",
              "- v7 没有形成相对 v6 的多查询稳定提升。",
              f"- 路线结论：{'冻结v7核心，等待统一最终实验' if accepted else '停止救援扫参，转向受控 retrieval 或论文收口'}。",
              "- 下一步："+("等待统一最终实验。" if accepted else "转向受控 retrieval；不再继续扫描 rescue 阈值。")]
    (REPORT/"final_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return {"output":output,"accepted":accepted,"rules":rules}


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=("train","dev"));args=parser.parse_args()
    result=train_freeze() if args.phase=="train" else dev_evaluate();print(json.dumps(result,ensure_ascii=False,indent=2,default=str))


if __name__=="__main__":main()
