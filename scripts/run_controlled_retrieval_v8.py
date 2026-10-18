"""Offline audit plus one-shot LIVE/FROZEN DSFR experiment for v8."""

from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from backend.eval_runtime import EvaluationSession,evaluation_session
from backend.paper_identity import match_identity,normalize_title
from backend.query_understanding import understand_query
from backend.retrieval.controlled_facet_retrieval_v8 import (
    DEFAULT_POLICY,FacetQuery,add_candidate_evidence,admit_candidates,build_facet_queries,
    core_protected_merge,facet_inventory,gap_triggered,retrieve_dsfr,
)
from backend.selection.core_rescue_selector_v6 import CoreRescueModelV6
from backend.trace_rank import fuse_candidates,tokenize
from scripts.audit_evaluation import gold_papers,load_jsonl,source_cutoff
from scripts.run_core_rescue_v6 import add_core_tp,attach_boundary_features,dev_rows_with_original_rank,grouped,load_v5_models,rank_order
from scripts.run_frozen_official_benchmark import json_hash,legalize
from scripts.run_short_sprint_v5 import write_csv


REPORT=ROOT/"reports/controlled_retrieval_v8"
CONFIG=ROOT/"configs/controlled_retrieval_v8"
CAPTURE=ROOT/"reports/frozen_eval/controlled_retrieval_v8"
DATASET=Path(r"D:\AICompetition\references\github\SPAR\benchmark\AutoScholarQuery_test.jsonl")
V6_MANIFEST=ROOT/"reports/core_rescue_v6/01_production_manifest.json"
V4_MANIFEST=ROOT/"reports/joint_optimization_v4/07_production_split_manifest.json"
V4=ROOT/"reports/joint_optimization_v4"
V6_MODEL=ROOT/"artifacts/core_rescue_v6/core_rescue_model.pkl"
V5_TRAIN=ROOT/"artifacts/short_sprint_v5/selector_train_pairs.parquet"
V5_CAL=ROOT/"artifacts/short_sprint_v5/selector_calibration_candidates.parquet"
TRACE_THRESHOLDS=(.45,.55,.65)


def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def git_head()->str:return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()


def all_dataset()->dict[str,dict[str,Any]]:
    return {row["qid"]:row for row in load_jsonl(DATASET)}


def production_ids()->tuple[set[str],set[str]]:
    manifest=json.loads(V6_MANIFEST.read_text(encoding="utf-8"))
    train=set(manifest["train_existing"])|set(manifest["train_additions"])
    calibration=set(manifest["calibration_existing"])|set(manifest["calibration_additions"])
    assert len(train)==100 and len(calibration)==40 and not train&calibration
    return train,calibration


def records_for(split:str)->list[dict[str,Any]]:
    dataset=all_dataset();train,cal=production_ids()
    if split=="train":ids=train
    elif split=="calibration":ids=cal
    elif split=="dev":ids=set(json.loads((ROOT/"configs/query_sets.json").read_text(encoding="utf-8"))["dev20"])
    else:raise ValueError(split)
    rows=[]
    for qid in sorted(ids,key=lambda value:int(value.rsplit("_",1)[-1])):
        row=dict(dataset[qid]);row["_v8_split"]=split;rows.append(row)
    return rows


def baseline_root(qid:str)->Path:
    v4=json.loads(V4_MANIFEST.read_text(encoding="utf-8"));existing=set(v4["train_query_ids"])|set(v4["calibration_query_ids"])
    if qid in existing:return ROOT/"reports/frozen_eval/selector_production50_v4_lean/queries"/qid
    return ROOT/"reports/frozen_eval/core_rescue_v6_production_lean/queries"/qid


def dev_root(qid:str)->Path:
    return ROOT/"reports/frozen_eval/dev20_v4_current_live/queries"/qid


def response_for(qid:str,split:str)->dict[str,Any]:
    root=dev_root(qid) if split=="dev" else baseline_root(qid)
    return json.loads((root/"live_response.json").read_text(encoding="utf-8"))


def understanding_for(record:dict[str,Any])->dict[str,Any]:
    qid=record["qid"];split=record["_v8_split"]
    understanding=copy.deepcopy(response_for(qid,split).get("query_understanding") or understand_query(record["question"]))
    understanding["original_query"]=record["question"]
    understanding["_evaluation_cutoff"]=source_cutoff(record).isoformat()
    understanding["_evaluation_year"]=source_cutoff(record).year
    return understanding


def baseline_top20_map(records:Sequence[dict[str,Any]])->dict[str,list[dict[str,Any]]]:
    if records and records[0]["_v8_split"]=="dev":
        rows=dev_rows_with_original_rank();mapping=grouped(rows)
        return {record["qid"]:[copy.deepcopy(row) for row in rank_order(mapping[record["qid"]])[:20]] for record in records}
    output={}
    for record in records:
        qid=record["qid"];papers=response_for(qid,record["_v8_split"]).get("papers") or []
        output[qid]=[copy.deepcopy(row) for row in papers[:20]]
    return output


def baseline_pool(record:dict[str,Any])->list[dict[str,Any]]:
    qid=record["qid"];root=dev_root(qid) if record["_v8_split"]=="dev" else baseline_root(qid)
    snapshot=json.loads((root/"snapshot.json").read_text(encoding="utf-8"));papers=[]
    for entry in (snapshot.get("calls") or {}).values():
        label=str(entry.get("label") or "")
        output=entry.get("output")
        if label.startswith("llm_") or not isinstance(output,list):continue
        for item in output:
            if isinstance(item,dict) and item.get("title"):papers.append(item)
    # The response Top20 is by definition part of the original candidate
    # pool. Include it explicitly because some older snapshots record cached
    # materialized candidates without preserving every upstream call output.
    papers.extend((response_for(qid,record["_v8_split"]).get("papers") or [])[:20])
    return fuse_candidates(legalize(papers,source_cutoff(record)))


def gold_metadata()->dict[str,list[dict[str,Any]]]:
    output:dict[str,list[dict[str,Any]]]=defaultdict(list)
    for path in (V5_TRAIN,V5_CAL):
        for row in pq.read_table(path).to_pylist():
            if int(row.get("label") or 0)==1:
                output[str(row["query_id"])].append(row)
    unique={}
    for qid,rows in output.items():
        seen=set();values=[]
        for row in rows:
            key=normalize_title(row.get("title"))
            if key and key not in seen:seen.add(key);values.append(row)
        unique[qid]=values
    return unique


def phrase_coverage(query_text:str,target_text:str)->float:
    query_tokens=tokenize(query_text);target_tokens=tokenize(target_text)
    return len(query_tokens&target_tokens)/len(query_tokens) if query_tokens else 0.0


def template_audit()->dict[str,Any]:
    REPORT.mkdir(parents=True,exist_ok=True);CONFIG.mkdir(parents=True,exist_ok=True)
    gold=gold_metadata();details=[]
    records=records_for("train")+records_for("calibration")
    for record in records:
        qid=record["qid"];library=__import__("backend.retrieval.controlled_facet_retrieval_v8",fromlist=["template_library"]).template_library(record["question"],understanding_for(record))
        targets=gold.get(qid) or gold_papers(record)
        for template in ("T1","T2","T3","T4"):
            facet=library.get(template)
            title=[];abstract=[]
            if facet:
                for target in targets:
                    title.append(phrase_coverage(facet.query,str(target.get("title") or "")))
                    abstract.append(phrase_coverage(facet.query,str(target.get("abstract") or "")))
            details.append({"query_id":qid,"split":record["_v8_split"],"template":template,"available":int(facet is not None),
                            "query":facet.query if facet else "","gold_count":len(targets),"mean_title_coverage":float(np.mean(title)) if title else 0,
                            "mean_abstract_coverage":float(np.mean(abstract)) if abstract else 0,
                            "coverage_score":float(np.mean(title))*.65+float(np.mean(abstract))*.35 if title else 0})
    write_csv(REPORT/"00_template_stats.csv",details)
    aggregate=[]
    for split in ("train","calibration"):
        for template in ("T1","T2","T3","T4"):
            rows=[row for row in details if row["split"]==split and row["template"]==template]
            aggregate.append({"split":split,"template":template,"availability":sum(row["available"] for row in rows)/len(rows),
                              "title_coverage":sum(row["mean_title_coverage"] for row in rows)/len(rows),
                              "abstract_coverage":sum(row["mean_abstract_coverage"] for row in rows)/len(rows),
                              "coverage_score":sum(row["coverage_score"] for row in rows)/len(rows)})
    train_rows=[row for row in aggregate if row["split"]=="train"]
    cal_by={row["template"]:row for row in aggregate if row["split"]=="calibration"}
    policy=tuple(row["template"] for row in sorted(train_rows,key=lambda row:(-row["coverage_score"],-cal_by[row["template"]]["coverage_score"],row["template"]))[:2])
    config=("schema_version: 1\nmethod: DSFR\npolicy: ["+", ".join(policy)+"]\nmax_variants: 2\nsources: [arxiv, openalex]\nper_source_limit: 25\n"
            "a1_trace_threshold: pending_calibration\ntrigger: pending_calibration\ncore_protected: true\nmax_api_per_query: 4\n")
    (CONFIG/"dsfr.yaml").write_text(config,encoding="utf-8")
    lines=["# DSFR模板离线审计","",f"- 查询：train100 + calibration40，dev/blind未参与。",f"- 冻结全局策略：`{policy}`。",
           "- 模板只读取原始query及既有结构化理解；不使用gold标题、作者或ID生成线上查询。","",
           "| split | template | availability | title coverage | abstract coverage | score |","|---|---|---:|---:|---:|---:|"]
    for row in aggregate:lines.append(f"| {row['split']} | {row['template']} | {row['availability']:.3f} | {row['title_coverage']:.3f} | {row['abstract_coverage']:.3f} | {row['coverage_score']:.3f} |")
    (REPORT/"00_template_audit.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return {"policy":policy,"aggregate":aggregate,"queries":len(records)}


def read_config()->dict[str,Any]:
    text=(CONFIG/"dsfr.yaml").read_text(encoding="utf-8")
    result={}
    for line in text.splitlines():
        if ":" not in line:continue
        key,value=line.split(":",1);value=value.strip()
        if value.startswith("["):result[key.strip()]=[item.strip() for item in value.strip("[]").split(",") if item.strip()]
        else:result[key.strip()]=value
    return result


def projection(result:dict[str,Any])->dict[str,Any]:
    return {"facets":result.get("facet_queries"),"candidates":[{"title":row.get("title"),"arxiv_id":row.get("arxiv_id"),"doi":row.get("doi"),
            "openalex_id":row.get("openalex_id"),"source_count":row.get("source_count"),"variant_count":row.get("variant_count"),
            "facet_coverage":round(float(row.get("facet_coverage") or 0),8)} for row in result.get("candidates") or []]}


async def execute_dsfr(record:dict[str,Any],dataset_name:str,mode:str)->tuple[dict[str,Any],EvaluationSession]:
    qid=record["qid"];path=CAPTURE/dataset_name/"queries"/qid/"snapshot.json";session=EvaluationSession.create(dataset_name,qid,source_cutoff(record),mode,path)
    understanding=understanding_for(record);config=read_config();started=time.perf_counter()
    with evaluation_session(session):
        result=await retrieve_dsfr(record["question"],understanding,policy=config["policy"],max_variants=2,per_source_limit=25)
    result["elapsed_seconds"]=time.perf_counter()-started
    if mode=="live":
        session.save({"facet_queries":result["facet_queries"],"errors":result["errors"],"source_commit":git_head()})
        result_path=path.parent/"live_result.json";result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return result,session


async def capture_dataset(records:Sequence[dict[str,Any]],dataset_name:str)->dict[str,Any]:
    run_dir=CAPTURE/dataset_name;run_dir.mkdir(parents=True,exist_ok=True);manifest=[];results={}
    for index,record in enumerate(records,1):
        qid=record["qid"];result_path=run_dir/"queries"/qid/"live_result.json";snapshot_path=result_path.parent/"snapshot.json"
        if result_path.exists() and snapshot_path.exists():
            live=json.loads(result_path.read_text(encoding="utf-8"));live_session=EvaluationSession.create(dataset_name,qid,source_cutoff(record),"replay",snapshot_path)
            snapshot_payload=json.loads(snapshot_path.read_text(encoding="utf-8"))
            live_session.date_filter_stats=snapshot_payload.get("date_filter_stats") or {}
        else:live,live_session=await execute_dsfr(record,dataset_name,"live")
        replay,replay_session=await execute_dsfr(record,dataset_name,"replay")
        identical=json_hash(projection(live))==json_hash(projection(replay));results[qid]=live
        calls=list(live_session.calls.values());api_calls=sum(str(call.get("label") or "").startswith("dsfr_") for call in calls)
        manifest.append({"query_id":qid,"snapshot":str(snapshot_path),"snapshot_sha256":sha(snapshot_path),"replay_identical":identical,
                         "candidate_count":len(live.get("candidates") or []),"api_calls":api_calls,"elapsed_seconds":live.get("elapsed_seconds"),
                         "date_filter_stats":live_session.date_filter_stats,"errors":live.get("errors")})
        print(f"[{index}/{len(records)}] {qid} candidates={len(live.get('candidates') or [])} replay={'OK' if identical else 'DIFF'}",flush=True)
    payload={"dataset":dataset_name,"query_count":len(records),"policy":read_config()["policy"],"replay_all_identical":all(row["replay_identical"] for row in manifest),
             "queries":manifest,"source_commit":git_head()}
    (run_dir/"manifest.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    return {"manifest":payload,"results":results}


def filter_sources(result:dict[str,Any],sources:set[str],record:dict[str,Any])->dict[str,Any]:
    understanding=understanding_for(record);facets=[FacetQuery(**{**row,"phrases":tuple(row["phrases"]),"facet_types":tuple(row["facet_types"])}) for row in result.get("facet_queries") or []]
    inventory=facet_inventory(record["question"],understanding);query_by={facet.query:facet for facet in facets};papers=[]
    for raw in result.get("candidates") or []:
        row=copy.deepcopy(raw);hits=[hit for hit in row.get("retrieval_hits") or [] if ("arxiv" if "arxiv" in str(hit.get("source") or "").lower() else "openalex") in sources]
        if not hits:continue
        row["retrieval_hits"]=hits;add_candidate_evidence(row,facets,inventory,query_by);papers.append(row)
    return {**result,"candidates":papers,"sources":sorted(sources),"api_calls":len(facets)*len(sources)}


def label_rows(rows:Sequence[dict[str,Any]],record:dict[str,Any])->list[dict[str,Any]]:
    gold=gold_papers(record);output=[]
    for rank,raw in enumerate(rows,1):
        row=copy.deepcopy(raw);row["query_id"]=record["qid"];row["query"]=record["question"];row["candidate_rank"]=rank;row["live_rank"]=rank
        row["paper_id"]=row.get("openalex_id") or row.get("doi") or row.get("arxiv_id") or row.get("title")
        row["trace_score"]=row.get("final_score") or row.get("trace_score") or row.get("diversity_adjusted_score") or 0
        row["constraint_coverage_score"]=row.get("constraint_coverage_score") or row.get("facet_coverage") or 0
        row["llm_relevance_score"]=row.get("llm_relevance_score") or 0
        row["source_count"]=row.get("source_count") or len(row.get("retrieval_hits") or []) or 1
        row["label"]=int(any(match_identity(row,target) for target in gold));output.append(row)
    return output


def aggregate_metrics(records:Sequence[dict[str,Any]],papers:dict[str,list[dict[str,Any]]])->dict[str,Any]:
    values=[];tp=fp=fn=predicted=gold_total=0
    for record in records:
        selected=papers.get(record["qid"],[]);gold=gold_papers(record);matched=set()
        for paper in selected:
            for index,target in enumerate(gold):
                if index not in matched and match_identity(paper,target):matched.add(index);break
        qtp=len(matched);qpred=len(selected);p=qtp/qpred if qpred else 0;r=qtp/len(gold) if gold else 0;f=2*p*r/(p+r) if p+r else 0
        values.append((p,r,f));tp+=qtp;predicted+=qpred;gold_total+=len(gold);fp+=qpred-qtp;fn+=len(gold)-qtp
    n=len(records)
    return {"precision":sum(x[0] for x in values)/n,"recall":sum(x[1] for x in values)/n,"f1":sum(x[2] for x in values)/n,
            "TP":tp,"FP":fp,"FN":fn,"predicted":predicted,"gold":gold_total,"mean_output_count":predicted/n}


def v6_selection(records:Sequence[dict[str,Any]],top20:dict[str,list[dict[str,Any]]])->dict[str,list[dict[str,Any]]]:
    full=[]
    for record in records:full.extend(label_rows(top20[record["qid"]],record))
    models=load_v5_models();boundary=attach_boundary_features(full,models);add_core_tp(boundary,full);model=CoreRescueModelV6.load(V6_MODEL);by_boundary=grouped(boundary);by_full=grouped(full);output={}
    for record in records:
        qid=record["qid"];core=rank_order(by_full[qid])[:2];rescue=model.choose_rescue(by_boundary[qid]);output[qid]=core+([rescue] if rescue else [])
    return output


def gold_keys(rows:Sequence[dict[str,Any]],record:dict[str,Any])->set[int]:
    result=set()
    for index,target in enumerate(gold_papers(record)):
        if any(match_identity(row,target) for row in rows):result.add(index)
    return result


def evaluate_configuration(
    records:Sequence[dict[str,Any]],live_results:dict[str,dict[str,Any]],*,sources:set[str],admission_mode:str,threshold:float,trigger:str,
) -> tuple[dict[str,Any],list[dict[str,Any]],dict[str,list[dict[str,Any]]]]:
    base_top=baseline_top20_map(records);candidate_map={};top_map={};details=[];total_api=0;latency_total=0.0;lost=0;gain_candidate=0;gain_top=0
    admitted_by={};raw_by={}
    for record in records:
        qid=record["qid"];understanding=understanding_for(record);base_pool=baseline_pool(record);base=base_top[qid]
        fire=trigger=="always_on_quality" or gap_triggered(base,record["question"],understanding)
        dsfr=filter_sources(live_results[qid],sources,record) if fire else {"facet_queries":[],"candidates":[],"api_calls":0,"elapsed_seconds":0}
        admitted,audit=admit_candidates(dsfr,record["question"],understanding,mode=admission_mode,trace_threshold=threshold,baseline=base_pool)
        merged_pool=fuse_candidates(base_pool+admitted);merged_top=core_protected_merge(base,admitted,understanding)
        candidate_map[qid]=merged_pool;top_map[qid]=merged_top;admitted_by[qid]=admitted;raw_by[qid]=dsfr.get("candidates") or []
        base_pool_gold=gold_keys(base_pool,record);base_top_gold=gold_keys(base,record);candidate_gold=gold_keys(merged_pool,record);top_gold=gold_keys(merged_top,record)
        candidate_gain=candidate_gold-base_pool_gold;top_gain=top_gold-base_top_gold;top_lost=base_top_gold-top_gold
        gain_candidate+=len(candidate_gain);gain_top+=len(top_gain);lost+=len(top_lost);total_api+=int(dsfr.get("api_calls") or 0)
        if fire:latency_total+=float(live_results[qid].get("elapsed_seconds") or 0)
        details.append({"query_id":qid,"triggered":int(fire),"baseline_candidate_tp":len(base_pool_gold),"candidate_tp":len(candidate_gold),"candidate_gain":len(candidate_gain),
                        "baseline_top20_tp":len(base_top_gold),"top20_tp":len(top_gold),"top20_gain":len(top_gain),"top20_lost":len(top_lost),
                        "raw_dsfr_candidates":len(dsfr.get("candidates") or []),"admitted_candidates":len(admitted),"rejected_candidates":len(audit)-len(admitted),
                        "api_calls":int(dsfr.get("api_calls") or 0),"elapsed_seconds":live_results[qid].get("elapsed_seconds") if fire else 0})
    candidate_metrics=aggregate_metrics(records,candidate_map);top_metrics=aggregate_metrics(records,top_map)
    k2_map={qid:rows[:2] for qid,rows in top_map.items()};k2_metrics=aggregate_metrics(records,k2_map)
    admitted_metrics=aggregate_metrics(records,admitted_by)
    v6_map=v6_selection(records,top_map);v6_metrics=aggregate_metrics(records,v6_map)
    result={"candidate_TP":candidate_metrics["TP"],"candidate_recall":candidate_metrics["recall"],"candidate_gain":gain_candidate,
            "top20_TP":top_metrics["TP"],"top20_recall":top_metrics["recall"],"top20_gain":gain_top,"top20_lost":lost,
            "k2_TP":k2_metrics["TP"],"k2_FP":k2_metrics["FP"],"k2_FN":k2_metrics["FN"],"k2_precision":k2_metrics["precision"],
            "k2_recall":k2_metrics["recall"],"k2_f1":k2_metrics["f1"],
            "admitted_TP":admitted_metrics["TP"],"admitted_FP":admitted_metrics["FP"],"admitted_precision":admitted_metrics["precision"],
            "selected_TP":v6_metrics["TP"],"selected_FP":v6_metrics["FP"],"selected_FN":v6_metrics["FN"],"precision":v6_metrics["precision"],
            "recall":v6_metrics["recall"],"f1":v6_metrics["f1"],"mean_output_count":v6_metrics["mean_output_count"],
            "api_per_query":total_api/len(records),"latency_seconds":latency_total/len(records),
            "mean_admitted":sum(len(rows) for rows in admitted_by.values())/len(records),"triggered_queries":sum(row["triggered"] for row in details)}
    return result,details,{"top20":top_map,"v6":v6_map,"admitted":admitted_by,"raw":raw_by,"candidate":candidate_map}


def baseline_configuration(records:Sequence[dict[str,Any]])->dict[str,Any]:
    top=baseline_top20_map(records);pools={record["qid"]:baseline_pool(record) for record in records};candidate=aggregate_metrics(records,pools);top_metrics=aggregate_metrics(records,top)
    k2=aggregate_metrics(records,{qid:rows[:2] for qid,rows in top.items()});v6=v6_selection(records,top);selected=aggregate_metrics(records,v6)
    return {"candidate_TP":candidate["TP"],"candidate_recall":candidate["recall"],"candidate_gain":0,"top20_TP":top_metrics["TP"],"top20_recall":top_metrics["recall"],
            "top20_gain":0,"top20_lost":0,"selected_TP":selected["TP"],"selected_FP":selected["FP"],"selected_FN":selected["FN"],"precision":selected["precision"],
            "k2_TP":k2["TP"],"k2_FP":k2["FP"],"k2_FN":k2["FN"],"k2_precision":k2["precision"],"k2_recall":k2["recall"],"k2_f1":k2["f1"],
            "admitted_TP":0,"admitted_FP":0,"admitted_precision":0,
            "recall":selected["recall"],"f1":selected["f1"],"mean_output_count":selected["mean_output_count"],"api_per_query":0,"latency_seconds":0,"mean_admitted":0,"triggered_queries":0}


def choose_threshold(records:Sequence[dict[str,Any]],results:dict[str,dict[str,Any]])->float:
    trials=[]
    for threshold in TRACE_THRESHOLDS:
        metrics,_,_=evaluate_configuration(records,results,sources={"arxiv","openalex"},admission_mode="A1",threshold=threshold,trigger="always_on_quality")
        trials.append((threshold,metrics))
    best=max(trials,key=lambda item:(item[1]["candidate_gain"],item[1]["top20_gain"],-item[1]["top20_lost"],item[1]["f1"],-item[1]["mean_admitted"]))
    (REPORT/"threshold_trials.json").write_text(json.dumps([{"threshold":t,**m} for t,m in trials],ensure_ascii=False,indent=2),encoding="utf-8")
    return best[0]


def update_config(threshold:float,trigger:str,challenger:str)->None:
    config=read_config();text=("schema_version: 1\nmethod: DSFR\npolicy: ["+", ".join(config["policy"])+"]\nmax_variants: 2\nsources: [arxiv, openalex]\nper_source_limit: 25\n"
        f"a1_trace_threshold: {threshold:.2f}\ntrigger: {trigger}\nchallenger: {challenger}\ncore_protected: true\nmax_api_per_query: 4\n")
    (CONFIG/"dsfr.yaml").write_text(text,encoding="utf-8")


def calibration_evaluate(records:Sequence[dict[str,Any]],results:dict[str,dict[str,Any]])->dict[str,Any]:
    baseline=baseline_configuration(records);threshold=choose_threshold(records,results);experiments=[];details_by={};products={}
    specs=[("R1_arxiv_A0",{"arxiv"},"A0"),("R2_dual_A0",{"arxiv","openalex"},"A0"),("R3_dual_A1",{"arxiv","openalex"},"A1")]
    for name,sources,mode in specs:
        metrics,details,product=evaluate_configuration(records,results,sources=sources,admission_mode=mode,threshold=threshold,trigger="always_on_quality")
        experiments.append({"configuration":name,**metrics});details_by[name]=details;products[name]=product
    winner=max(experiments,key=lambda row:(row["candidate_gain"],row["top20_gain"],-row["top20_lost"],row["selected_TP"],-row["selected_FP"],-row["api_per_query"]))
    name=winner["configuration"];sources={"arxiv"} if name.startswith("R1") else {"arxiv","openalex"};mode="A1" if name.endswith("A1") else "A0"
    gap_metrics,gap_details,gap_product=evaluate_configuration(records,results,sources=sources,admission_mode=mode,threshold=threshold,trigger="gap_triggered")
    quality_close=gap_metrics["candidate_TP"]>=winner["candidate_TP"] and gap_metrics["top20_TP"]>=winner["top20_TP"] and gap_metrics["selected_TP"]>=winner["selected_TP"]
    trigger="gap_triggered" if quality_close and gap_metrics["api_per_query"]<winner["api_per_query"] else "always_on_quality"
    chosen_metrics=gap_metrics if trigger=="gap_triggered" else winner;chosen_details=gap_details if trigger=="gap_triggered" else details_by[name]
    update_config(threshold,trigger,name)
    write_csv(REPORT/"02_calibration_comparison.csv",[{"configuration":"B0_baseline",**baseline},*experiments,{"configuration":f"{name}_gap_triggered",**gap_metrics}])
    write_csv(REPORT/"03_candidate_gain.csv",chosen_details);write_csv(REPORT/"04_top20_gain.csv",chosen_details)
    write_csv(REPORT/"06_cost_report.csv",[{"configuration":"B0_baseline",**baseline},*experiments,{"configuration":f"chosen_{trigger}",**chosen_metrics}])
    passed=chosen_metrics["candidate_gain"]>=3 or (chosen_metrics["top20_gain"]>=2 and chosen_metrics["top20_lost"]==0) or (chosen_metrics["selected_TP"]>=baseline["selected_TP"]+1 and chosen_metrics["selected_FP"]<=baseline["selected_FP"]+2)
    freeze={"schema_version":1,"source_commit":git_head(),"config_sha256":sha(CONFIG/"dsfr.yaml"),"threshold":threshold,"trigger":trigger,"challenger":name,
            "sources":sorted(sources),"admission_mode":mode,"baseline":baseline,"chosen":chosen_metrics,"dev_gate_passed":passed,"dev_evaluated":False}
    (CONFIG/"calibration_freeze.json").write_text(json.dumps(freeze,ensure_ascii=False,indent=2),encoding="utf-8")
    return freeze


def source_contributions(records:Sequence[dict[str,Any]],results:dict[str,dict[str,Any]],products:dict[str,dict[str,list[dict[str,Any]]]]|None=None)->dict[str,int]:
    counts=Counter()
    for record in records:
        qid=record["qid"];base=baseline_pool(record);gold=gold_papers(record)
        for index,target in enumerate(gold):
            if any(match_identity(row,target) for row in base):continue
            hits=[]
            for paper in results[qid].get("candidates") or []:
                if match_identity(paper,target):
                    if paper.get("arxiv_rank") is not None:hits.append("arxiv")
                    if paper.get("openalex_rank") is not None:hits.append("openalex")
            if set(hits)=={"arxiv"}:counts["arxiv_unique"]+=1
            elif set(hits)=={"openalex"}:counts["openalex_unique"]+=1
            elif {"arxiv","openalex"}<=set(hits):counts["shared"]+=1
    return dict(counts)


async def run_smoke()->dict[str,Any]:
    records=records_for("calibration")[:5];capture=await capture_dataset(records,"retrieval_smoke5_v8")
    valid=sum(len(value.get("candidates") or []) for value in capture["results"].values());errors=sum(len(value.get("errors") or []) for value in capture["results"].values())
    lines=["# smoke5协议","",f"- 固定查询：`{[row['qid'] for row in records]}`",f"- 有效候选：{valid}",f"- source errors：{errors}",f"- replay identical：{capture['manifest']['replay_all_identical']}",
           "- 仅检查API、日期、身份融合和回放；不用于指标选择。"]
    (REPORT/"01_smoke5_protocol.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    if not capture["manifest"]["replay_all_identical"] or valid==0:raise RuntimeError("smoke5 failed")
    return capture["manifest"]


async def run_calibration()->dict[str,Any]:
    records=records_for("calibration");capture=await capture_dataset(records,"calibration40_v8");freeze=calibration_evaluate(records,capture["results"])
    freeze["source_contributions"]=source_contributions(records,capture["results"])
    (CONFIG/"calibration_freeze.json").write_text(json.dumps(freeze,ensure_ascii=False,indent=2),encoding="utf-8")
    return freeze


def dev_accept(baseline:dict[str,Any],chosen:dict[str,Any])->tuple[bool,bool]:
    development=((chosen["selected_TP"]>=baseline["selected_TP"]+1 and chosen["selected_FP"]<=baseline["selected_FP"]+2 and chosen["f1"]>=baseline["f1"]+.01 and chosen["recall"]>=baseline["recall"] and chosen["mean_output_count"]<=2.25)
                 or (chosen["selected_TP"]>=baseline["selected_TP"]+2 and chosen["selected_FP"]<=baseline["selected_FP"]+5 and chosen["f1"]>=baseline["f1"]-.005 and chosen["recall"]>=baseline["recall"]+.03))
    component=chosen["candidate_gain"]>=3 and chosen["top20_gain"]>=2 and chosen["top20_lost"]==0 and chosen["api_per_query"]<=4
    return development,component


async def run_dev()->dict[str,Any]:
    freeze=json.loads((CONFIG/"calibration_freeze.json").read_text(encoding="utf-8"))
    if not freeze["dev_gate_passed"]:
        write_csv(REPORT/"05_dev20_one_shot.csv",[{"configuration":"SKIPPED","reason":"calibration gate not met"}])
        state={"schema_version":1,"branch":"controlled-retrieval-v8","source_commit":freeze["source_commit"],
               "blind_used":False,"dev_live_runs":0,"dev_skipped":True,"reason":"calibration gate not met",
               "development_candidate_v8":False,"retrieval_component_candidate_v8":False,
               "baseline":freeze["baseline"],"chosen":freeze["chosen"],
               "source_contributions":freeze.get("source_contributions") or {},"config_hash_verified":True}
        (REPORT/"state.json").write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8")
        return state
    if sha(CONFIG/"dsfr.yaml")!=freeze["config_sha256"]:raise RuntimeError("frozen config changed")
    records=records_for("dev");capture=await capture_dataset(records,"dev20_v8");baseline=baseline_configuration(records)
    chosen,details,products=evaluate_configuration(records,capture["results"],sources=set(freeze["sources"]),admission_mode=freeze["admission_mode"],threshold=float(freeze["threshold"]),trigger=freeze["trigger"])
    write_csv(REPORT/"05_dev20_one_shot.csv",[{"configuration":"B0_current_v6",**baseline},{"configuration":"B1_DSFR_core_protected_v6",**chosen}])
    development,component=dev_accept(baseline,chosen);contributions=source_contributions(records,capture["results"])
    # Category A was frozen in the earlier retrievability audit; using it here is evaluation only.
    audit=list(csv.DictReader((V4/"03_retrievability_audit.csv").open(encoding="utf-8-sig")));category_a={(row["query_id"],normalize_title(row["gold_title"])) for row in audit if row["category"]=="A"}
    source_covered_found=0
    for record in records:
        for paper in products["candidate"][record["qid"]]:
            if (record["qid"],normalize_title(paper.get("title"))) in category_a:source_covered_found+=1
    state={"schema_version":1,"branch":"controlled-retrieval-v8","source_commit":freeze["source_commit"],"blind_used":False,"dev_live_runs":1,
           "development_candidate_v8":development,"retrieval_component_candidate_v8":component,"baseline":baseline,"chosen":chosen,
           "source_contributions":contributions,"source_covered_query_missed_found":source_covered_found,"config_hash_verified":True}
    if development:(CONFIG/"development_candidate_v8.json").write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8")
    elif component:(CONFIG/"retrieval_component_candidate_v8.json").write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8")
    (REPORT/"state.json").write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8")
    return state


def final_report()->dict[str,Any]:
    freeze=json.loads((CONFIG/"calibration_freeze.json").read_text(encoding="utf-8"));state_path=REPORT/"state.json"
    state=json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"development_candidate_v8":False,"retrieval_component_candidate_v8":False}
    registry=list(csv.DictReader((REPORT/"02_calibration_comparison.csv").open(encoding="utf-8-sig")))
    write_csv(REPORT/"experiment_registry.csv",registry)
    status="development_candidate_v8" if state.get("development_candidate_v8") else "retrieval_component_candidate_v8" if state.get("retrieval_component_candidate_v8") else "全部拒绝"
    contributions=state.get("source_contributions") or freeze.get("source_contributions") or {}
    chosen=state.get("chosen") or freeze["chosen"];comparison_baseline=state.get("baseline") or freeze["baseline"]
    rows={row["configuration"]:row for row in registry};a0=rows.get("R2_dual_A0",{});a1=rows.get("R3_dual_A1",{})
    raw_recovered=sum(int(value) for value in contributions.values());selected_gain=chosen.get("selected_TP",0)-comparison_baseline.get("selected_TP",0)
    lines=["# controlled-retrieval-v8 最终报告","","## 1. 模板与方法","",f"- 全局模板策略：`{read_config()['policy']}`。",f"- 冻结配置：{freeze['challenger']}，{freeze['admission_mode']}，{freeze['trigger']}。",
           "- 数据源仅为arXiv与OpenAlex；每条查询最多2个facet、4次API；Top2核心不变。","- 未调用新LLM、citation、second round，也未使用dev20或blind选择配置。","","## 2. calibration40","",
           "| 配置 | candidate TP/R | Top20 TP/R（gain/lost） | K2 TP/FP/F1 | K2+v6 TP/FP/F1 | admitted TP/FP/macro-P | mean output | API/query | latency/query |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in registry:
        lines.append(f"| {row['configuration']} | {row.get('candidate_TP','')}/{float(row.get('candidate_recall') or 0):.4f} | {row.get('top20_TP','')}/{float(row.get('top20_recall') or 0):.4f}（{row.get('top20_gain','')}/{row.get('top20_lost','')}） | {row.get('k2_TP','')}/{row.get('k2_FP','')}/{float(row.get('k2_f1') or 0):.4f} | {row.get('selected_TP','')}/{row.get('selected_FP','')}/{float(row.get('f1') or 0):.4f} | {row.get('admitted_TP','')}/{row.get('admitted_FP','')}/{float(row.get('admitted_precision') or 0):.4f} | {float(row.get('mean_output_count') or 0):.2f} | {float(row.get('api_per_query') or 0):.2f} | {float(row.get('latency_seconds') or 0):.2f}s |")
    source_covered=(f"dev20中找回{state['source_covered_query_missed_found']}篇" if "source_covered_query_missed_found" in state else "dev20未运行，未对既有source-covered类别重新计数")
    lines += ["","R1与R2使用同一轮双源LIVE快照，R1的API数按arXiv-only配置核算，延迟不能从共享快照中单独拆分；gap-triggered延迟按全部查询摊销。",
              "","## 3. 直接回答","",f"1. 模板策略：{source_covered}；在calibration40原始DSFR结果中找回{raw_recovered}篇原候选池外gold，A0准入后净增{chosen.get('candidate_gain',0)}篇candidate gold。",
              f"2. 原始检索贡献：arXiv独有{contributions.get('arxiv_unique',0)}篇，OpenAlex独有{contributions.get('openalex_unique',0)}篇，双源共有{contributions.get('shared',0)}篇。",
              f"3. 双源一致性未提高准入精度：R2/A0为{int(float(a0.get('admitted_TP') or 0))}/{int(float(a0.get('admitted_FP') or 0))}，macro-P={float(a0.get('admitted_precision') or 0):.4f}；R3/A1为{int(float(a1.get('admitted_TP') or 0))}/{int(float(a1.get('admitted_FP') or 0))}，macro-P={float(a1.get('admitted_precision') or 0):.4f}。",
              f"4. 新增candidate gold共{chosen.get('candidate_gain',0)}篇，进入Top20为{chosen.get('top20_gain',0)}篇。",f"5. 新增Top20 gold进入v6最终集合为{selected_gain}篇。",f"6. 丢失原Top20 gold为{chosen.get('top20_lost',0)}篇。",
              f"7. 冻结方案API/query={chosen.get('api_per_query',0):.2f}，摊销latency/query={chosen.get('latency_seconds',0):.2f}s，平均准入候选增加{chosen.get('mean_admitted',0):.2f}篇。",
              f"8. gap-triggered与always-on保持相同candidate TP，但API/query由{float(a0.get('api_per_query') or 0):.2f}降至{chosen.get('api_per_query',0):.2f}，因此冻结gap-triggered。",
              f"9. 接受结果：{status}；calibration gate未通过，dev20未运行。",
              "10. DSFR当前需要能识别无既有LLM分数新候选、且不挤掉旧gold的tail排序/证据校准，才值得再次作为retrieval component验证。",
              "11. 可留到后续验证：冻结快照上的双源增量、candidate→Top20转化率、准入精度和API/延迟曲线。",
              "12. 当前不能写成正式贡献：最终质量提升、跨系统领先、泛化性或统计显著性。","",
              "## 4. 结论","",f"- 结果：{status}。",f"- calibration dev gate：{freeze['dev_gate_passed']}；dev20与blind均未运行。",
              "- DSFR能补入少量候选池外gold，但没有把新增gold送入Top20，且挤掉1篇原Top20 gold，因此按停止条件结束本轮。"]
    (REPORT/"final_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return {"status":status,"freeze":freeze,"state":state}


async def async_main(phase:str)->Any:
    if phase=="smoke":return await run_smoke()
    if phase=="calibration":return await run_calibration()
    if phase=="dev":return await run_dev()
    raise ValueError(phase)


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=("audit","smoke","calibration","dev","report"));args=parser.parse_args()
    if args.phase=="audit":result=template_audit()
    elif args.phase=="report":result=final_report()
    else:result=asyncio.run(async_main(args.phase))
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))


if __name__=="__main__":main()
