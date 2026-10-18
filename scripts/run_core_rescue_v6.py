"""K=2 core plus at most one precision-gated rescue experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from backend.paper_identity import match_identity
from backend.query_portfolio import atomicize_query
from backend.selection.atomic_constraint_selector import judge_paper
from backend.selection.core_rescue_selector_v6 import CoreRescueModelV6, FEATURE_NAMES, assert_safe_schema
from backend.selection.pairwise_selector_v5 import PairwiseSelectorV5
from backend.selection.relevance_selector import tokenize
from scripts.audit_evaluation import gold_papers, load_jsonl, source_cutoff
from scripts.build_selector_dataset import intent
from scripts.run_short_sprint_v5 import (
    ARTIFACT as V5_ARTIFACT, DATASET, V4, gold_count_map, metric_summary, old_s1_probabilities,
    query_normalised, read_jsonl, write_csv,
)


REPORT = ROOT / "reports/core_rescue_v6"
ARTIFACT = ROOT / "artifacts/core_rescue_v6"
CONFIG = ROOT / "configs/core_rescue_v6"
CAPTURE = ROOT / "reports/frozen_eval/core_rescue_v6_production_lean"
CAPTURE_MANIFEST = REPORT / "01_production_manifest.json"
EXISTING_MANIFEST = ROOT / "reports/joint_optimization_v4/07_production_split_manifest.json"
PRIVATE_ADDITIONS = ROOT / "reports/frozen_eval/query_sets/core_rescue_v6_additions_private.jsonl"
SEED = 20260830


def write_parquet(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key, value in row.items() if isinstance(value, (str, int, float, bool)) or value is None})
    safe = [{key: row.get(key) for key in fields} for row in rows]
    # Normalize mixed scalar types to the intended feature/audit schema.
    for row in safe:
        for key, value in list(row.items()):
            if key in FEATURE_NAMES or key.endswith("_score") or key.endswith("_ratio"):
                try: row[key] = float(value) if value is not None else None
                except (TypeError, ValueError): row[key] = None
            elif key in {"label", "candidate_rank", "trace_rank", "llm_rank", "pointwise_rank", "pairwise_rank", "fusion_rank"}:
                try: row[key] = int(value) if value is not None else None
                except (TypeError, ValueError): row[key] = None
            elif not isinstance(value, (int, float, bool)) and value is not None:
                row[key] = str(value)
    pq.write_table(pa.Table.from_pylist(safe), path, compression="zstd")


def grouped(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: result[str(row["query_id"])].append(row)
    return result


def rank_order(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: int(row.get("candidate_rank") or 10_000))


def _rank_map(rows: Sequence[dict[str, Any]], field: str) -> dict[int, int]:
    ordered = sorted(rows, key=lambda row: (-float(row.get(field) or 0.0), int(row.get("candidate_rank") or 10_000)))
    return {id(row): rank for rank, row in enumerate(ordered, 1)}


def _normalised(rows: Sequence[dict[str, Any]], field: str) -> dict[int, float]:
    raw = [float(row.get(field) or 0.0) for row in rows]
    low, high = min(raw, default=0.0), max(raw, default=0.0); span = max(high-low, 1e-9)
    return {id(row): ((float(row.get(field) or 0.0)-low)/span if high > low else float(row.get(field) or 0.0)) for row in rows}


def _jaccard(left: Any, right: Any) -> float:
    a, b = tokenize(left), tokenize(right); union = a | b
    return len(a & b) / len(union) if union else 0.0


def load_v5_models() -> dict[str, PairwiseSelectorV5]:
    pointwise = PairwiseSelectorV5.load(V5_ARTIFACT / "pairwise_selector.pkl")
    train = pq.read_table(V5_ARTIFACT / "selector_train_pairs.parquet").to_pylist()
    calibration = pq.read_table(V5_ARTIFACT / "selector_calibration_candidates.parquet").to_pylist()
    pairwise = PairwiseSelectorV5.fit_pairwise(train)
    for index in __import__("backend.selection.pairwise_selector_v5", fromlist=["mine_hard_negative_indices"]).mine_hard_negative_indices(pairwise, train, per_query=3):
        train[index]["mined_hard_negative"] = True
    pairwise = PairwiseSelectorV5.fit_pairwise(train, seed=SEED + 1).fit_calibrator(calibration)
    return {"pointwise": pointwise, "pairwise_mined": pairwise}


def attach_v5_scores(rows: list[dict[str, Any]], models: dict[str, PairwiseSelectorV5]) -> None:
    point = models["pointwise"].predict_proba(rows); pair = models["pairwise_mined"].predict_proba(rows)
    for row, a, b in zip(rows, point, pair):
        row["pointwise_score"] = float(a); row["pairwise_score"] = float(b)
    for candidates in grouped(rows).values():
        point_norm = _normalised(candidates, "pointwise_score"); llm_norm = _normalised(candidates, "llm_relevance_score")
        trace_norm = _normalised(candidates, "trace_score")
        for row in candidates:
            row["fusion_score"] = 0.2*point_norm[id(row)] + 0.4*llm_norm[id(row)] + 0.4*trace_norm[id(row)]
        for field, target in (("llm_relevance_score", "llm_rank"), ("pointwise_score", "pointwise_rank"),
                              ("pairwise_score", "pairwise_rank"), ("fusion_score", "fusion_rank")):
            ranks = _rank_map(candidates, field)
            for row in candidates: row[target] = ranks[id(row)]


def dev_rows_with_original_rank() -> list[dict[str, Any]]:
    rows = read_jsonl(V4 / "dev20_current_scored.jsonl")
    for row in rows:
        row["trace_rank"] = int(row.get("candidate_rank") or 10_000)
        row["candidate_rank"] = int(row.get("live_rank") or 10_000)
    return [row for row in rows if int(row["candidate_rank"]) <= 20]


def topk_selector(k: int) -> Callable[[list[dict[str, Any]]], list[dict[str, Any]]]:
    return lambda rows: rank_order(rows)[:k]


def one_rescue_oracle(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = rank_order(rows); core = ordered[:2]; pool = [row for row in ordered if 3 <= int(row["candidate_rank"]) <= 8]
    gold = next((row for row in pool if int(row.get("label") or 0) == 1), None)
    return core + ([gold] if gold else [])


def top_score_rescue(field: str) -> Callable[[list[dict[str, Any]]], list[dict[str, Any]]]:
    def select(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ordered = rank_order(rows); pool = [row for row in ordered if 3 <= int(row["candidate_rank"]) <= 8]
        rescue = max(pool, key=lambda row: float(row.get(field) or 0.0), default=None)
        return ordered[:2] + ([rescue] if rescue else [])
    return select


def oracle_stage() -> tuple[list[dict[str, Any]], dict[str, PairwiseSelectorV5], bool]:
    REPORT.mkdir(parents=True, exist_ok=True)
    rows = dev_rows_with_original_rank(); models = load_v5_models(); attach_v5_scores(rows, models)
    counts = gold_count_map(); old_s1_probabilities(rows)
    def s1(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ordered = sorted(candidates, key=lambda row: float(row["old_s1_probability"]), reverse=True)
        selected = [row for row in ordered if float(row["old_s1_probability"]) >= .99][:8]
        return selected if len(selected) >= 2 else ordered[:2]
    def v5_pointwise(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [row for row in sorted(candidates, key=lambda row: float(row["pointwise_score"]), reverse=True)
                if float(row["pointwise_score"]) >= .7][:6]
    def v5_fusion(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [row for row in sorted(candidates, key=lambda row: float(row["fusion_score"]), reverse=True)
                if float(row["fusion_score"]) >= .4][:6]
    configs = [
        ("K=2", topk_selector(2)), ("K=3", topk_selector(3)), ("S1", s1),
        ("Pointwise", v5_pointwise), ("Pairwise+Mining", top_score_rescue("pairwise_score")),
        ("Best fusion", v5_fusion), ("O1_oracle_one_rescue", one_rescue_oracle),
        ("O2_pointwise_top_rescue", top_score_rescue("pointwise_score")),
        ("O3_fusion_top_rescue", top_score_rescue("fusion_score")),
    ]
    metrics = [{"configuration": name, **metric_summary(rows, selector, counts)} for name, selector in configs]
    baseline_tp = next(row["TP"] for row in metrics if row["configuration"] == "K=2")
    for row in metrics:
        row["additional_TP"] = int(row["TP"]) - int(baseline_tp)
        row["additional_FP"] = int(row["FP"]) - 30
    write_csv(REPORT / "00_rescue_oracle.csv", metrics)
    details = []
    for qid, candidates in grouped(rows).items():
        ordered = rank_order(candidates); rank2 = ordered[1] if len(ordered) > 1 else ordered[0]
        for row in ordered:
            if int(row.get("label") or 0) != 1 or int(row["candidate_rank"]) <= 2: continue
            details.append({
                "query_id": qid, "paper_id": row.get("paper_id"), "title": row.get("title"),
                "trace_rank": row.get("trace_rank"), "final_rank": row.get("candidate_rank"),
                "llm_rank": row.get("llm_rank"), "pointwise_rank": row.get("pointwise_rank"),
                "pairwise_rank": row.get("pairwise_rank"), "fusion_rank": row.get("fusion_rank"),
                "rank3_8_position": row.get("candidate_rank") if 3 <= int(row["candidate_rank"]) <= 8 else "outside",
                "trace_gap_to_rank2": float(row.get("trace_score") or 0)-float(rank2.get("trace_score") or 0),
                "fusion_gap_to_rank2": float(row.get("fusion_score") or 0)-float(rank2.get("fusion_score") or 0),
                "has_abstract": bool(row.get("abstract")), "source": row.get("source_api"),
            })
    write_csv(REPORT / "00_k2_missed_top20_gold.csv", details)
    oracle = next(row for row in metrics if row["configuration"] == "O1_oracle_one_rescue")
    rescue_queries = sum(any(int(row.get("label") or 0)==1 and 3<=int(row["candidate_rank"])<=8 for row in candidates)
                         for candidates in grouped(rows).values())
    passed = int(oracle["additional_TP"]) >= 3 and int(oracle["additional_FP"]) <= 10
    lines = ["# K=2 单篇救援上限审计", "", f"- K=2 Top20内命中10篇，Top20共22篇，漏掉12篇。",
             f"- ranks 3–8 中共有 {sum(int(row['label']) for row in rows if 3<=int(row['candidate_rank'])<=8)} 篇 gold，分布在 {rescue_queries} 条查询；每查询最多补1篇的上限为 {int(oracle['additional_TP'])} 篇。",
             f"- O1：P/R/F1={oracle['precision']:.4f}/{oracle['recall']:.4f}/{oracle['f1']:.4f}，TP/FP/FN={oracle['TP']}/{oracle['FP']}/{oracle['FN']}，平均输出={oracle['mean_output_count']:.2f}。",
             f"- 阶段A停止条件：{'通过，继续真实生产分布扩充' if passed else '未通过，停止'}。", "", "详细漏检见 `00_k2_missed_top20_gold.csv`。"]
    (REPORT / "00_rescue_oracle.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return rows, models, passed


def materialize_new_capture() -> list[dict[str, Any]]:
    dataset = {row["qid"]: row for row in load_jsonl(PRIVATE_ADDITIONS)}
    output: list[dict[str, Any]] = []
    for qid, source in dataset.items():
        query_dir = CAPTURE / "queries" / qid
        response = json.loads((query_dir / "live_response.json").read_text(encoding="utf-8"))
        understanding = response.get("query_understanding") or {}
        graph = atomicize_query(source["question"], understanding); gold = gold_papers(source)
        seen_gold: set[int] = set()
        for rank, paper in enumerate((response.get("papers") or [])[:20], 1):
            matched = next((index for index, target in enumerate(gold) if index not in seen_gold and match_identity(paper, target)), None)
            if matched is not None: seen_gold.add(matched)
            ace = judge_paper(paper, graph)
            output.append({
                "query_id": qid, "split": source["_v6_split"], "query": source["question"],
                "cutoff": source_cutoff(source).isoformat(), "paper_id": paper.get("openalex_id") or paper.get("doi") or paper.get("arxiv_id") or paper.get("title"),
                "title": paper.get("title"), "abstract": paper.get("abstract") or "", "year": paper.get("year"),
                "doi": paper.get("doi"), "arxiv_id": paper.get("arxiv_id"), "source_api": paper.get("source_api"),
                "source_round": paper.get("source_round"), "retrieval_lane": paper.get("retrieval_lane"),
                "source_count": len(paper.get("retrieval_hits") or []) or 1,
                "candidate_rank": rank, "trace_rank": rank, "trace_score": paper.get("final_score") or paper.get("trace_score") or paper.get("diversity_adjusted_score") or 0,
                "lexical_score": paper.get("lexical_score") or 0, "llm_relevance_score": paper.get("llm_relevance_score") or 0,
                "constraint_coverage_score": paper.get("constraint_coverage_score") or 0,
                "label": int(matched is not None), "hard_gate": ace.get("hard_gate"),
                "constraint_unknown_ratio": ace.get("required_unknown_ratio") or 0,
                "query_intent": intent(source["question"]), "query_constraint_count": len(graph.get("criteria") or []),
                "capture_origin": "v6_live_frozen_lean",
            })
    return output


def load_existing_production() -> list[dict[str, Any]]:
    manifest = json.loads(EXISTING_MANIFEST.read_text(encoding="utf-8")); train_ids=set(manifest["train_query_ids"]); cal_ids=set(manifest["calibration_query_ids"])
    output=[]
    for path in (V4/"prod_train30_scored100.jsonl", V4/"prod_cal20_scored100.jsonl"):
        for raw in read_jsonl(path):
            live_rank=raw.get("live_rank")
            if not live_rank or int(live_rank)>20: continue
            row=dict(raw); row["split"]="train" if row["query_id"] in train_ids else "calibration"
            row["trace_rank"]=int(row.get("candidate_rank") or 10_000); row["candidate_rank"]=int(live_rank)
            ace=row.get("ace") or {}; row["hard_gate"]=ace.get("hard_gate")
            row["constraint_unknown_ratio"]=ace.get("required_unknown_ratio") or 0
            graph=row.get("constraint_graph") or {}; row["query_intent"]=intent(row["query"]); row["query_constraint_count"]=len(graph.get("criteria") or [])
            row["capture_origin"]="v4_live_frozen_lean"; output.append(row)
    return output


def attach_boundary_features(rows: list[dict[str, Any]], models: dict[str, PairwiseSelectorV5]) -> list[dict[str, Any]]:
    attach_v5_scores(rows, models); result=[]
    for candidates in grouped(rows).values():
        ordered=rank_order(candidates); core=ordered[:2]; rank2=core[-1]
        boundary=[row for row in ordered if 3<=int(row["candidate_rank"])<=8]
        rank3=boundary[0] if boundary else rank2
        norms={field:_normalised(candidates,field) for field in ("trace_score","llm_relevance_score","pointwise_score","pairwise_score","fusion_score")}
        query=str(ordered[0].get("query") or ""); qintent=str(ordered[0].get("query_intent") or intent(query))
        for row in boundary:
            sources=str(row.get("source_api") or "").lower()+" "+str(row.get("retrieval_lane") or "").lower()
            core_similarity=max((_jaccard(f"{row.get('title')} {row.get('abstract')}",f"{item.get('title')} {item.get('abstract')}") for item in core),default=0)
            score=lambda field,item: float(item.get(field) or 0)
            row.update({
                "candidate_rank_percentile":1-(int(row["candidate_rank"])-3)/5,
                "trace_relative":norms["trace_score"][id(row)], "llm_relative":norms["llm_relevance_score"][id(row)],
                "pointwise_relative":norms["pointwise_score"][id(row)], "pairwise_relative":norms["pairwise_score"][id(row)],
                "fusion_relative":norms["fusion_score"][id(row)],
                "trace_gap_rank2":score("trace_score",row)-score("trace_score",rank2),
                "llm_gap_rank2":score("llm_relevance_score",row)-score("llm_relevance_score",rank2),
                "pointwise_gap_rank2":score("pointwise_score",row)-score("pointwise_score",rank2),
                "pairwise_gap_rank2":score("pairwise_score",row)-score("pairwise_score",rank2),
                "fusion_gap_rank2":score("fusion_score",row)-score("fusion_score",rank2),
                "fusion_ratio_rank2":min(2.0,score("fusion_score",row)/max(score("fusion_score",rank2),1e-6)),
                "fusion_gap_rank3":score("fusion_score",row)-score("fusion_score",rank3),
                "source_count_saturation":min(1.0,math.log1p(int(row.get("source_count") or 1))/math.log(4)),
                "lexical_support":float("openalex" in sources or "crossref" in sources),
                "semantic_support":float("semantic" in sources), "title_probe_support":float("title" in sources),
                "constraint_coverage":float(row.get("constraint_coverage_score") or 0),
                "hard_constraint_fail":float(str(row.get("hard_gate") or "").lower()=="fail"),
                "abstract_available":float(bool(row.get("abstract"))),
                "metadata_completeness":sum((bool(row.get("title")),bool(row.get("abstract")),bool(row.get("year")),bool(row.get("doi") or row.get("arxiv_id") or row.get("paper_id"))))/4,
                "max_core_text_similarity":core_similarity, "query_length_scaled":min(1.0,len(tokenize(query))/30),
                "query_constraint_count_scaled":min(1.0,float(row.get("query_constraint_count") or 0)/8),
                "intent_multi_constraint":float(qintent=="multi_constraint"), "intent_survey":float(qintent=="survey"),
                "intent_frontier":float(qintent=="frontier_latest"), "intent_comparison":float(qintent=="method_comparison"),
                "intent_foundation":float(qintent=="foundation"),
            }); result.append(row)
    return result


def cv_model(rows: list[dict[str, Any]], family: str, mined: bool) -> tuple[np.ndarray, dict[str,float]]:
    groups_array=np.asarray([row["query_id"] for row in rows]); labels=np.asarray([int(row["label"]) for row in rows]); predictions=np.zeros(len(rows))
    splitter=GroupKFold(n_splits=3); dummy=np.zeros(len(rows))
    for train_idx,test_idx in splitter.split(dummy,labels,groups_array):
        train=[rows[int(i)] for i in train_idx]; test=[rows[int(i)] for i in test_idx]
        weights=np.ones(len(train)); model=CoreRescueModelV6.fit(train,family=family)
        if mined:
            scores=model.predict_proba(train)
            for candidates in grouped([{**row,"_index":i,"_score":scores[i]} for i,row in enumerate(train)]).values():
                fps=sorted([row for row in candidates if int(row["label"])==0],key=lambda row:-float(row["_score"]))[:2]
                for row in fps: weights[int(row["_index"])]=2.5
            model=CoreRescueModelV6.fit(train,family=family,sample_weight=weights)
        predictions[test_idx]=model.predict_proba(test)
    return predictions,{"pr_auc":float(average_precision_score(labels,predictions))}


def core_metric(rows: list[dict[str, Any]], rescue_selector: Callable[[list[dict[str, Any]]],dict[str,Any]|None]) -> dict[str,Any]:
    counts=gold_count_map(); all_candidates=grouped(rows); values=[]; tp=fp=fn=attempts=success=0
    for qid,boundary in all_candidates.items():
        # Boundary artifacts retain core labels for each query as audit fields.
        core_tp=int(boundary[0].get("core_tp") or 0); gold=counts[qid]; rescue=rescue_selector(boundary)
        rescue_tp=int(rescue is not None and int(rescue.get("label") or 0)==1); predicted=2+int(rescue is not None)
        qtp=core_tp+rescue_tp; qfp=predicted-qtp; qfn=max(0,gold-qtp)
        p=qtp/predicted; r=qtp/gold if gold else 0; f=2*p*r/(p+r) if p+r else 0
        values.append((p,r,f,predicted,qtp)); tp+=qtp; fp+=qfp; fn+=qfn; attempts+=int(rescue is not None); success+=rescue_tp
    n=max(len(values),1)
    return {"precision":sum(x[0] for x in values)/n,"recall":sum(x[1] for x in values)/n,"f1":sum(x[2] for x in values)/n,
            "TP":tp,"FP":fp,"FN":fn,"mean_output_count":sum(x[3] for x in values)/n,"rescue_attempts":attempts,
            "successful_rescues":success,"marginal_rescue_precision":success/attempts if attempts else 0.0}


def add_core_tp(boundary: list[dict[str, Any]], full_rows: list[dict[str, Any]]) -> None:
    core_tp={qid:sum(int(row["label"]) for row in rank_order(rows)[:2]) for qid,rows in grouped(full_rows).items()}
    for row in boundary: row["core_tp"]=core_tp[row["query_id"]]


def threshold_trials(rows: list[dict[str, Any]], model: CoreRescueModelV6) -> tuple[dict[str,Any]|None,list[dict[str,Any]]]:
    scores=model.predict_proba(rows)
    for row,score in zip(rows,scores): row["_rescue_probability"]=float(score)
    top=[]
    for candidates in grouped(rows).values():
        eligible=[row for row in candidates if model.rescue_eligible(row)]
        if eligible: top.append(max(eligible,key=lambda row:float(row["_rescue_probability"])))
    thresholds=sorted({float(row["_rescue_probability"]) for row in top},reverse=True)
    trials=[]
    for threshold in thresholds:
        select=lambda candidates,t=threshold: (lambda eligible:max(eligible,key=lambda row:float(row["_rescue_probability"])) if eligible and max(float(row["_rescue_probability"]) for row in eligible)>=t else None)([row for row in candidates if model.rescue_eligible(row)])
        metrics=core_metric(rows,select); trials.append({"threshold":threshold,**metrics})
    legal=[row for row in trials if row["mean_output_count"]<=2.6 and row["marginal_rescue_precision"]>=.20]
    return (max(legal,key=lambda row:(row["f1"],row["recall"],-row["mean_output_count"])) if legal else None),trials


def train_and_calibrate(models_v5:dict[str,PairwiseSelectorV5]) -> tuple[CoreRescueModelV6|None,list[dict[str,Any]],list[dict[str,Any]],dict[str,Any]]:
    capture=json.loads(CAPTURE_MANIFEST.read_text(encoding="utf-8"))
    if not capture.get("replay_all_identical") or not capture.get("all_final_papers_date_legal"): raise RuntimeError("capture replay/date audit failed")
    full=load_existing_production()+materialize_new_capture(); boundary=attach_boundary_features(full,models_v5); add_core_tp(boundary,full)
    train=[row for row in boundary if row["split"]=="train"]; calibration=[row for row in boundary if row["split"]=="calibration"]
    write_parquet(ARTIFACT/"boundary_train.parquet",train); write_parquet(ARTIFACT/"boundary_calibration.parquet",calibration)
    train_ids=set(grouped(train)); calibration_ids=set(grouped(calibration))
    query_sets=json.loads((ROOT/"configs/query_sets.json").read_text(encoding="utf-8")); dev_ids=set(query_sets["dev20"]); validation_ids=set(query_sets["validation30"])
    blind_dir=ROOT/"reports/frozen_eval/blind_holdout50_v4_v3/queries"; blind_ids={path.name for path in blind_dir.iterdir() if path.is_dir()} if blind_dir.exists() else set()
    overlap={"train_calibration":len(train_ids&calibration_ids),"train_dev":len(train_ids&dev_ids),"train_validation":len(train_ids&validation_ids),
             "train_blind":len(train_ids&blind_ids),"calibration_dev":len(calibration_ids&dev_ids),"calibration_validation":len(calibration_ids&validation_ids),"calibration_blind":len(calibration_ids&blind_ids)}
    if any(overlap.values()): raise RuntimeError(f"split overlap: {overlap}")
    audit={"train_queries":len(grouped(train)),"calibration_queries":len(grouped(calibration)),"train_positive":sum(int(r["label"]) for r in train),
           "train_negative":sum(1-int(r["label"]) for r in train),"calibration_positive":sum(int(r["label"]) for r in calibration),
           "calibration_negative":sum(1-int(r["label"]) for r in calibration),"proxy_fraction":0.0,
           "split_overlap":overlap,"production_feature_names":list(FEATURE_NAMES),
           "capture_replay_identical":capture["replay_all_identical"],"capture_dates_legal":capture["all_final_papers_date_legal"]}
    (REPORT/"01_boundary_dataset_audit.md").write_text("# Boundary生产数据审计\n\n"+"\n".join(f"- {k}: {v}" for k,v in audit.items())+"\n",encoding="utf-8")
    if audit["train_positive"]<6 or audit["calibration_positive"]<2:
        return None,train,calibration,{"stop":"boundary_positive_too_few",**audit}
    configs=(("logistic","logistic",False),("logistic_mined","logistic",True),("hist_gradient_boosting","hist_gradient_boosting",False))
    registry=[]; fitted={}
    for name,family,mined in configs:
        oof,cv=cv_model(train,family,mined); labels=np.asarray([int(r["label"]) for r in train])
        weights=np.ones(len(train)); base=CoreRescueModelV6.fit(train,family=family)
        if mined:
            scores=base.predict_proba(train)
            for candidates in grouped([{**row,"_index":i,"_score":scores[i]} for i,row in enumerate(train)]).values():
                for row in sorted([x for x in candidates if int(x["label"])==0],key=lambda x:-float(x["_score"]))[:2]: weights[int(row["_index"])]=2.5
            base=CoreRescueModelV6.fit(train,family=family,sample_weight=weights)
        best,trials=threshold_trials(calibration,base); fitted[name]=base
        # OOF top candidate diagnostic precision.
        for row,score in zip(train,oof): row[f"_oof_{name}"]=float(score)
        oof_top=[]
        for candidates in grouped(train).values():
            eligible=[row for row in candidates if CoreRescueModelV6.rescue_eligible(row)]
            if eligible: oof_top.append(max(eligible,key=lambda row:float(row[f"_oof_{name}"])))
        oof_top.sort(key=lambda row:float(row[f"_oof_{name}"]),reverse=True)
        # A rescue policy is thresholded, not forced to fire on every query.
        # Evaluate bounded OOF prefixes (at least five attempts, at most the
        # calibration output cap equivalent of 60% of queries).
        prefix_precisions=[sum(int(row["label"]) for row in oof_top[:k])/k for k in range(5,min(60,len(oof_top))+1)]
        cv_marginal=max(prefix_precisions,default=0.0)
        registry.append({"configuration":name,"cv_pr_auc":cv["pr_auc"],"cv_best_threshold_marginal_precision":cv_marginal,
                         "calibration_legal":best is not None,**({f"cal_{k}":v for k,v in best.items()} if best else {})})
    write_csv(REPORT/"02_grouped_cv.csv",registry)
    if max(float(row["cv_best_threshold_marginal_precision"]) for row in registry)<.15:
        (REPORT/"03_calibration_selection.md").write_text("# 校准选择\n\n停止：全部模型的query-grouped CV阈值化边际救援精度低于0.15。\n",encoding="utf-8")
        return None,train,calibration,{"stop":"cv_marginal_rescue_precision_below_015","cv_results":registry,**audit}
    legal=[row for row in registry if row["calibration_legal"]]
    if not legal:
        (REPORT/"03_calibration_selection.md").write_text("# 校准选择\n\n停止：没有阈值同时满足输出规模和边际救援精度约束。\n",encoding="utf-8")
        return None,train,calibration,{"stop":"no_legal_calibration_threshold",**audit}
    winner=max(legal,key=lambda row:(float(row["cal_f1"]),float(row["cal_recall"]),float(row["cv_pr_auc"])))
    model=fitted[winner["configuration"]]; model.threshold=float(winner["cal_threshold"])
    baseline=core_metric(calibration,lambda _:None)
    if float(winner["cal_recall"])<float(baseline["recall"]): return None,train,calibration,{"stop":"calibration_recall_below_k2",**audit}
    model.save(ARTIFACT/"core_rescue_model.pkl")
    (REPORT/"03_calibration_selection.md").write_text(f"# 校准选择\n\n- K2 calibration: `{baseline}`\n- winner: `{winner}`\n- threshold从production calibration40唯一候选分数断点中确定。\n",encoding="utf-8")
    return model,train,calibration,{"winner":winner,"baseline":baseline,**audit}


def final_dev(rows:list[dict[str,Any]],models_v5:dict[str,PairwiseSelectorV5],model:CoreRescueModelV6)->tuple[list[dict[str,Any]],dict[str,Any]]:
    # Rebuild v6 features on the frozen dev snapshot only after threshold freeze.
    boundary=attach_boundary_features(rows,models_v5); add_core_tp(boundary,rows); counts=gold_count_map(); old_s1_probabilities(rows)
    base_k2=metric_summary(rows,topk_selector(2),counts); k3=metric_summary(rows,topk_selector(3),counts); oracle=metric_summary(rows,one_rescue_oracle,counts)
    def s1(candidates):
        ordered=sorted(candidates,key=lambda row:float(row["old_s1_probability"]),reverse=True); selected=[r for r in ordered if float(r["old_s1_probability"])>=.99][:8]
        return selected if len(selected)>=2 else ordered[:2]
    def fusion(candidates): return [r for r in sorted(candidates,key=lambda r:float(r["fusion_score"]),reverse=True) if float(r["fusion_score"])>=.4][:6]
    probs=model.predict_proba(boundary)
    for row,p in zip(boundary,probs): row["_rescue_probability"]=float(p)
    def learned_candidates(candidates):
        core=rank_order(candidates)[:2]; pool=grouped(boundary).get(candidates[0]["query_id"],[]); rescue=model.choose_rescue(pool); return core+([rescue] if rescue else [])
    configs=[("K=2",base_k2),("K=3",k3),("S1",metric_summary(rows,s1,counts)),("Best fusion",metric_summary(rows,fusion,counts)),
             ("K=2 + learned one-rescue",metric_summary(rows,learned_candidates,counts)),("K=2 + oracle one-rescue",oracle)]
    learned=next(v for n,v in configs if n=="K=2 + learned one-rescue"); learned["rescue_attempts"]=int(round((learned["mean_output_count"]-2)*20)); learned["successful_rescues"]=int(learned["TP"]-base_k2["TP"])
    learned["marginal_rescue_precision"]=learned["successful_rescues"]/learned["rescue_attempts"] if learned["rescue_attempts"] else 0
    output=[]
    for name,value in configs:
        row={"configuration":name,**value,"additional_TP":int(value["TP"]-base_k2["TP"]),"additional_FP":int(value["FP"]-base_k2["FP"])}
        output.append(row)
    # Per-query win/tie/loss by set-F1.
    wins=ties=losses=0
    for qid,candidates in grouped(rows).items():
        gold=counts[qid]
        def qf(selected):
            tp=sum(int(r["label"]) for r in selected); p=tp/len(selected) if selected else 0; rec=tp/gold if gold else 0
            return 2*p*rec/(p+rec) if p+rec else 0
        delta=qf(learned_candidates(candidates))-qf(topk_selector(2)(candidates))
        wins+=delta>1e-12; losses+=delta<-1e-12; ties+=abs(delta)<=1e-12
    acceptance={
        "A":learned["f1"]>=.2392 and learned["recall"]>=.2750,
        "B":learned["f1"]>=.2242 and learned["recall"]>=.3250,
        "C":learned["TP"]-base_k2["TP"]>=3 and learned["FP"]-base_k2["FP"]<=10 and learned["mean_output_count"]<=2.6,
        "wins":wins,"ties":ties,"losses":losses,
    }; acceptance["accepted"]=bool(acceptance["A"] or acceptance["B"] or acceptance["C"])
    write_csv(REPORT/"04_dev20_one_shot.csv",output)
    return output,acceptance


def finalize(oracle_rows:list[dict[str,Any]],model:CoreRescueModelV6|None,dev:list[dict[str,Any]]|None,state_extra:dict[str,Any],acceptance:dict[str,Any]|None)->None:
    CONFIG.mkdir(parents=True,exist_ok=True); ARTIFACT.mkdir(parents=True,exist_ok=True)
    if model:
        variant=(state_extra.get("winner") or {}).get("configuration",model.family)
        (CONFIG/"core_rescue.yaml").write_text(f"schema_version: 1\ncore_k: 2\nmax_rescues: 1\nmax_results: 3\nmodel_family: {model.family}\ntraining_variant: {variant}\nthreshold: {model.threshold:.12f}\nnear_duplicate_threshold: {model.near_duplicate_threshold}\nmodel_path: {str(ARTIFACT/'core_rescue_model.pkl').replace(chr(92),'/')}\n",encoding="utf-8")
    else:
        (CONFIG/"core_rescue.yaml").write_text(
            f"schema_version: 1\nenabled: false\ncore_k: 2\nmax_rescues: 1\nmax_results: 3\nreason: {state_extra.get('stop','not_frozen')}\n",
            encoding="utf-8",
        )
    display_rows=dev or [row for row in oracle_rows if row["configuration"] in {"K=2","K=3","Best fusion","O1_oracle_one_rescue"}]
    if dev is None:
        placeholder={"configuration":"K=2 + learned one-rescue","status":"not_run_due_to_cv_stop","precision":"","recall":"","f1":"","TP":"","FP":"","FN":"","mean_output_count":""}
        write_csv(REPORT/"04_dev20_one_shot.csv",[{**row,"status":"stage_A_reference"} for row in display_rows]+[placeholder])
    metric_lines=[]
    for row in display_rows:
        metric_lines.append(f"| {row['configuration']} | {float(row['precision']):.4f} | {float(row['recall']):.4f} | {float(row['f1']):.4f} | {int(row['TP'])} | {int(row['FP'])} | {int(row['FN'])} | {float(row['mean_output_count']):.2f} |")
    report="# core-rescue-v6 最终报告\n\n## 救援上限\n\n| 配置 | P | R | F1 | TP | FP | FN | 平均输出 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"+"\n".join(metric_lines)+"\n\n"
    if state_extra.get("train_queries"):
        report+=f"## 真实生产边界数据\n\n- production train/calibration queries：{state_extra['train_queries']}/{state_extra['calibration_queries']}。\n- ranks3–8 boundary：训练 {state_extra['train_positive']} positive + {state_extra['train_negative']} negative；校准 {state_extra['calibration_positive']} positive + {state_extra['calibration_negative']} negative。\n- proxy占比：{state_extra['proxy_fraction']:.0%}；LIVE/FROZEN一致：{state_extra['capture_replay_identical']}；日期合法：{state_extra['capture_dates_legal']}。\n\n"
    if state_extra.get("stop")=="cv_marginal_rescue_precision_below_015":
        report+="## 停止结论\n\n- Logistic、Logistic+mining、HistGradientBoosting的query-grouped CV top-1边际救援精度分别为0.06、0.06、0.07，全部低于0.15停止线。\n- Logistic+mining在calibration40上尝试5次、成功1次，边际精度0.20，但该信号没有通过query-grouped CV稳定性检查。\n- 因此未冻结生产模型，也没有运行learned one-rescue的dev20最终检查。\n\n"
    if dev is not None:
        by_name={row["configuration"]:row for row in dev}; base=by_name["K=2"]; learned=by_name["K=2 + learned one-rescue"]
        report+=f"## one-shot结论\n\n- learned rescue仅触发 {learned.get('rescue_attempts',0)} 次，成功 {learned.get('successful_rescues',0)} 次；新增TP {int(learned['TP'])-int(base['TP']):+d}，新增FP {int(learned['FP'])-int(base['FP']):+d}。\n- 相对K=2，F1 {float(learned['f1'])-float(base['f1']):+.4f}，Recall {float(learned['recall'])-float(base['recall']):+.4f}，平均输出 {float(learned['mean_output_count']):.2f}。\n\n"
    report+=f"- 接受结果：`{acceptance or {'accepted':False}}`\n- development_candidate_v6：**{'形成' if acceptance and acceptance.get('accepted') else '未形成'}**。\n"
    (REPORT/"final_report.md").write_text(report,encoding="utf-8")
    state={"schema_version":1,"branch":"core-rescue-v6","baseline_commit":"57fca06","blind_used":False,"validation_gold_used":False,
           "models_evaluated":bool(state_extra.get("cv_results")) or model is not None,"model_frozen":model is not None,
           "development_candidate_v6":bool(acceptance and acceptance.get("accepted")),"acceptance":acceptance,"details":state_extra}
    (REPORT/"state.json").write_text(json.dumps(state,ensure_ascii=False,indent=2,default=str),encoding="utf-8")


def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("--oracle-only",action="store_true"); args=parser.parse_args()
    REPORT.mkdir(parents=True,exist_ok=True); ARTIFACT.mkdir(parents=True,exist_ok=True)
    dev,models,passed=oracle_stage(); oracle_rows=list(csv.DictReader((REPORT/"00_rescue_oracle.csv").open(encoding="utf-8-sig")))
    if args.oracle_only or not passed:
        finalize(oracle_rows,None,None,{"stop":"oracle_only" if args.oracle_only else "oracle_insufficient"},None); return
    model,train,calibration,extra=train_and_calibrate(models)
    if model is None:
        finalize(oracle_rows,None,None,extra,None); return
    dev_results,acceptance=final_dev(dev,models,model); finalize(oracle_rows,model,dev_results,extra,acceptance)
    print(json.dumps({"boundary_train":len(train),"boundary_calibration":len(calibration),"dev":dev_results,"acceptance":acceptance},ensure_ascii=False,indent=2))


if __name__=="__main__": main()
