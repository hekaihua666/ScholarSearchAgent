"""Frozen evaluation runner for Evidence Contract Rank (ECRank) v9."""

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
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.eval_runtime import EvaluationSession, evaluation_session, frozen_call
from backend.paper_identity import match_identity
from backend.retrieval.controlled_facet_retrieval_v8 import gap_triggered
from backend.selection.evidence_contract_ranker_v9 import (
    ContractCard, build_contract_cards, request_contract_ranking, route_contract_core,
    route_top20, score_cards,
)
from backend.trace_rank import fuse_candidates
from scripts.audit_evaluation import source_cutoff
from scripts.run_controlled_retrieval_v8 import (
    aggregate_metrics, baseline_configuration, baseline_pool, baseline_top20_map,
    capture_dataset as capture_dsfr_dataset, filter_sources, gold_keys, gold_papers,
    records_for, understanding_for, v6_selection,
)
from scripts.run_frozen_official_benchmark import json_hash, legalize
from scripts.run_short_sprint_v5 import write_csv


REPORT = ROOT / "reports/evidence_contract_rank_v9"
CONFIG = ROOT / "configs/evidence_contract_rank_v9"
CAPTURE = ROOT / "reports/frozen_eval/evidence_contract_rank_v9"
V8_CAPTURE = ROOT / "reports/frozen_eval/controlled_retrieval_v8/calibration40_v8/queries"
THRESHOLDS = (70.0, 75.0, 80.0, 85.0, 90.0, 95.0)
MARGINS = (0.0, 5.0, 10.0, 15.0)
MAX_REPLACEMENTS = (1, 2)


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_dirs() -> None:
    REPORT.mkdir(parents=True, exist_ok=True);CONFIG.mkdir(parents=True, exist_ok=True);CAPTURE.mkdir(parents=True, exist_ok=True)


def load_v8_calibration_results(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output = {}
    for record in records:
        path = V8_CAPTURE / record["qid"] / "live_result.json"
        output[record["qid"]] = json.loads(path.read_text(encoding="utf-8"))
    return output


def candidate_pool(record: dict[str, Any], dsfr: dict[str, Any]) -> list[dict[str, Any]]:
    base = baseline_pool(record)
    raw = filter_sources(dsfr, {"arxiv", "openalex"}, record).get("candidates") or []
    return fuse_candidates(legalize([*base, *raw], source_cutoff(record)))


def cards_for(record: dict[str, Any], dsfr: dict[str, Any]) -> list[ContractCard]:
    top20 = baseline_top20_map([record])[record["qid"]]
    raw = filter_sources(dsfr, {"arxiv", "openalex"}, record).get("candidates") or []
    return build_contract_cards(record["question"], top20, raw)


async def _safe_contract_call(query: str, cards: Sequence[ContractCard]) -> dict[str, Any]:
    try:
        return await request_contract_ranking(query, cards)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {str(exc)[:300]}", "parsed": {}, "tokens": 0}


async def execute_contract(
    record: dict[str, Any], dsfr: dict[str, Any], dataset_name: str, mode: str,
) -> tuple[dict[str, Any], EvaluationSession]:
    qid = record["qid"];path = CAPTURE / dataset_name / "queries" / qid / "snapshot.json"
    session = EvaluationSession.create(dataset_name, qid, source_cutoff(record), mode, path)
    top20 = baseline_top20_map([record])[qid];understanding = understanding_for(record)
    fire = gap_triggered(top20, record["question"], understanding)
    cards = cards_for(record, dsfr) if fire else []
    newcomers = [card for card in cards if card.origin == "newcomer"]
    started = time.perf_counter();payload: dict[str, Any]
    if not newcomers:
        payload = {"skipped": True, "reason": "trigger_off_or_no_newcomers", "parsed": {}, "tokens": 0}
    else:
        signature = hashlib.sha256(json.dumps([
            {"id": card.card_id, "title": card.paper.get("title"), "doi": card.paper.get("doi"), "arxiv": card.paper.get("arxiv_id")}
            for card in cards
        ], ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        arguments = {"schema": "ecrank-v9-contract-1", "query": record["question"], "card_signature": signature, "card_count": len(cards)}
        with evaluation_session(session):
            payload = await frozen_call("ecrank_contract_listwise_v9", arguments, lambda: _safe_contract_call(record["question"], cards))
    result = {
        "query_id": qid, "triggered": bool(newcomers), "elapsed_seconds": time.perf_counter() - started,
        "cards": [{"card_id": card.card_id, "origin": card.origin, "original_rank": card.original_rank, "paper": card.paper} for card in cards],
        "contract_result": payload,
    }
    if mode == "live":
        session.save({"source_commit": git_head(), "schema": "ecrank-v9-contract-1"})
        path.parent.mkdir(parents=True, exist_ok=True)
        (path.parent / "live_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result, session


def result_projection(result: dict[str, Any]) -> dict[str, Any]:
    payload = result.get("contract_result") or {}
    return {"triggered": result.get("triggered"), "cards": [(row["card_id"], row["origin"], row["paper"].get("title")) for row in result.get("cards") or []],
            "parsed": payload.get("parsed"), "error": payload.get("error")}


async def capture_contract_dataset(
    records: Sequence[dict[str, Any]], dsfr_results: dict[str, dict[str, Any]], dataset_name: str,
) -> dict[str, Any]:
    ensure_dirs();results={};manifest=[]
    for index, record in enumerate(records, 1):
        qid=record["qid"];root=CAPTURE/dataset_name/"queries"/qid;live_path=root/"live_result.json";snapshot_path=root/"snapshot.json"
        if live_path.exists() and snapshot_path.exists():
            live=json.loads(live_path.read_text(encoding="utf-8"));live_session=EvaluationSession.create(dataset_name,qid,source_cutoff(record),"replay",snapshot_path)
        else:
            live,live_session=await execute_contract(record,dsfr_results[qid],dataset_name,"live")
        replay,_=await execute_contract(record,dsfr_results[qid],dataset_name,"replay")
        identical=json_hash(result_projection(live))==json_hash(result_projection(replay));results[qid]=live
        payload=live.get("contract_result") or {};calls=list(live_session.calls.values())
        manifest.append({"query_id":qid,"triggered":live.get("triggered"),"card_count":len(live.get("cards") or []),
                         "tokens":int(payload.get("tokens") or 0),"elapsed_seconds":live.get("elapsed_seconds"),
                         "llm_calls":sum(call.get("label")=="ecrank_contract_listwise_v9" for call in calls),
                         "error":payload.get("error"),"replay_identical":identical,
                         "snapshot_sha256":sha(snapshot_path) if snapshot_path.exists() else None})
        print(f"[{index}/{len(records)}] {qid} cards={len(live.get('cards') or [])} replay={'OK' if identical else 'DIFF'}",flush=True)
    payload={"dataset":dataset_name,"query_count":len(records),"source_commit":git_head(),
             "replay_all_identical":all(row["replay_identical"] for row in manifest),"queries":manifest}
    (CAPTURE/dataset_name/"manifest.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    return {"manifest":payload,"results":results}


def restore_cards(result: dict[str, Any]) -> list[ContractCard]:
    return [ContractCard(row["card_id"],row["origin"],row.get("original_rank"),row["paper"]) for row in result.get("cards") or []]


def evaluate_profile(
    records: Sequence[dict[str, Any]], dsfr_results: dict[str, dict[str, Any]], contract_results: dict[str, dict[str, Any]],
    *, threshold: float, margin: float, max_replacements: int, score_mode: str = "full",
) -> tuple[dict[str, Any], dict[str, dict[str, list[dict[str, Any]]]], list[dict[str, Any]]]:
    baseline_top = baseline_top20_map(records);core_map={};top20_map={};candidate_map={};details=[];actions_total=0;llm_calls=tokens=0
    core_gain=core_lost=top_gain=top_lost=0
    for record in records:
        qid=record["qid"];base=baseline_top[qid];candidate_map[qid]=candidate_pool(record,dsfr_results[qid])
        result=contract_results[qid];cards=restore_cards(result);parsed=(result.get("contract_result") or {}).get("parsed") or {}
        scored=score_cards(cards,parsed) if cards and parsed else []
        if score_mode!="full":
            for row in scored:
                assessment=row["assessment"]
                if score_mode=="point":row["contract_score"]=float(assessment.get("point_score") or 0)
                elif score_mode=="list":row["contract_score"]=float(assessment.get("list_score") or 0)
                elif score_mode=="no_bottleneck":row["contract_score"]=.5*float(assessment.get("point_score") or 0)+.5*float(assessment.get("list_score") or 0)
        if scored:selected,actions,contract_mode=route_contract_core(record["question"],base,scored,parsed,threshold=threshold,margin=margin,max_replacements=max_replacements)
        else:selected,actions,contract_mode=[copy.deepcopy(row) for row in base[:2]],[],"not_evaluated"
        routed=route_top20(base,selected,scored,threshold=threshold) if scored else [copy.deepcopy(row) for row in base]
        core_map[qid]=selected;top20_map[qid]=routed;actions_total+=len(actions)
        base_core_gold=gold_keys(base[:2],record);new_core_gold=gold_keys(selected,record);base_top_gold=gold_keys(base,record);new_top_gold=gold_keys(routed,record)
        core_gain+=len(new_core_gold-base_core_gold);core_lost+=len(base_core_gold-new_core_gold)
        top_gain+=len(new_top_gold-base_top_gold);top_lost+=len(base_top_gold-new_top_gold)
        payload=result.get("contract_result") or {};llm_calls+=int(bool(result.get("triggered")));tokens+=int(payload.get("tokens") or 0)
        details.append({"query_id":qid,"contract_mode":contract_mode,"actions":len(actions),"core_gain":len(new_core_gold-base_core_gold),"core_lost":len(base_core_gold-new_core_gold),
                        "top20_gain":len(new_top_gold-base_top_gold),"top20_lost":len(base_top_gold-new_top_gold),
                        "candidate_gain":len(gold_keys(candidate_map[qid],record)-gold_keys(baseline_pool(record),record)),
                        "incoming_titles":" | ".join(action["incoming_id"] for action in actions),
                        "min_action_margin":min((action["margin"] for action in actions),default=0)})
    candidate=aggregate_metrics(records,candidate_map);top=aggregate_metrics(records,top20_map);core=aggregate_metrics(records,core_map)
    metrics={"threshold":threshold,"margin":margin,"max_replacements":max_replacements,"score_mode":score_mode,
             "candidate_TP":candidate["TP"],"candidate_recall":candidate["recall"],"top20_TP":top["TP"],"top20_recall":top["recall"],
             "top20_gain":top_gain,"top20_lost":top_lost,"selected_TP":core["TP"],"selected_FP":core["FP"],"selected_FN":core["FN"],
             "precision":core["precision"],"recall":core["recall"],"f1":core["f1"],"mean_output_count":core["mean_output_count"],
             "core_gain":core_gain,"core_lost":core_lost,"replacement_actions":actions_total,
             "llm_calls":llm_calls,"tokens":tokens,"tokens_per_query":tokens/len(records)}
    return metrics,{"candidate":candidate_map,"top20":top20_map,"core":core_map},details


def baseline_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    top=baseline_top20_map(records);core={qid:rows[:2] for qid,rows in top.items()};candidate={record["qid"]:baseline_pool(record) for record in records}
    cm=aggregate_metrics(records,candidate);tm=aggregate_metrics(records,top);km=aggregate_metrics(records,core)
    return {"configuration":"B0_K2","candidate_TP":cm["TP"],"candidate_recall":cm["recall"],"top20_TP":tm["TP"],"top20_recall":tm["recall"],
            "top20_gain":0,"top20_lost":0,"selected_TP":km["TP"],"selected_FP":km["FP"],"selected_FN":km["FN"],
            "precision":km["precision"],"recall":km["recall"],"f1":km["f1"],"mean_output_count":km["mean_output_count"],
            "core_gain":0,"core_lost":0,"replacement_actions":0,"llm_calls":0,"tokens":0,"tokens_per_query":0}


def select_profile(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    # F1 first, then net TP preservation, then cost/risk.  All profiles share
    # one frozen LLM capture, so threshold sweeps add no API or tokens.
    return max(rows,key=lambda row:(row["f1"],row["selected_TP"],-row["selected_FP"],row["top20_gain"]-row["top20_lost"],-row["replacement_actions"],row["threshold"],row["margin"],-row["max_replacements"]))


def calibration_evaluate(records: Sequence[dict[str, Any]], dsfr: dict[str, dict[str, Any]], contract: dict[str, dict[str, Any]]) -> dict[str, Any]:
    base=baseline_metrics(records);profiles=[];products_by={};details_by={}
    for threshold in THRESHOLDS:
        for margin in MARGINS:
            for max_replacements in MAX_REPLACEMENTS:
                metrics,products,details=evaluate_profile(records,dsfr,contract,threshold=threshold,margin=margin,max_replacements=max_replacements)
                profiles.append(metrics);products_by[(threshold,margin,max_replacements)]=products;details_by[(threshold,margin,max_replacements)]=details
    chosen=select_profile(profiles);key=(chosen["threshold"],chosen["margin"],chosen["max_replacements"])
    write_csv(REPORT/"01_calibration_profiles.csv",profiles)
    write_csv(REPORT/"02_query_actions.csv",details_by[key])
    ablations=[]
    for mode in ("point","list","no_bottleneck","full"):
        metrics,_,_=evaluate_profile(records,dsfr,contract,threshold=chosen["threshold"],margin=chosen["margin"],max_replacements=chosen["max_replacements"],score_mode=mode)
        ablations.append({"configuration":mode,**metrics})
    write_csv(REPORT/"03_ablation.csv",ablations)
    # Existing v6 selection is evaluated only once after profile freeze.
    chosen_products=products_by[key];v6=aggregate_metrics(records,v6_selection(records,chosen_products["top20"]))
    chosen["v6_TP"]=v6["TP"];chosen["v6_FP"]=v6["FP"];chosen["v6_FN"]=v6["FN"];chosen["v6_f1"]=v6["f1"];chosen["v6_recall"]=v6["recall"]
    gate=((chosen["selected_TP"]>=base["selected_TP"]+2 and chosen["selected_FP"]<=base["selected_FP"] and chosen["f1"]>=base["f1"]+.02)
          or (chosen["top20_gain"]>=2 and chosen["top20_lost"]<=1))
    freeze={"schema_version":1,"source_commit":git_head(),"prompt_schema":"ecrank-v9-contract-1","baseline":base,"chosen":chosen,
            "dev_gate_passed":gate,"dev_evaluated":False,"threshold_grid":list(THRESHOLDS),"margin_grid":list(MARGINS),"max_replacements_grid":list(MAX_REPLACEMENTS)}
    (CONFIG/"calibration_freeze.json").write_text(json.dumps(freeze,ensure_ascii=False,indent=2),encoding="utf-8")
    return freeze


async def run_calibration() -> dict[str, Any]:
    ensure_dirs();records=records_for("calibration");dsfr=load_v8_calibration_results(records)
    capture=await capture_contract_dataset(records,dsfr,"calibration40_v9")
    if not capture["manifest"]["replay_all_identical"]:raise RuntimeError("calibration replay mismatch")
    return calibration_evaluate(records,dsfr,capture["results"])


async def run_dev() -> dict[str, Any]:
    ensure_dirs();freeze=json.loads((CONFIG/"calibration_freeze.json").read_text(encoding="utf-8"))
    if not freeze["dev_gate_passed"]:
        state={"status":"rejected_on_calibration","dev_live_runs":0,"blind_used":False,"freeze":freeze}
        write_csv(REPORT/"04_dev20_one_shot.csv",[{"configuration":"SKIPPED","reason":"calibration gate not met"}])
        (REPORT/"state.json").write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8");return state
    records=records_for("dev")
    # This is the only v9 dev retrieval LIVE run; capture_dataset resumes only
    # complete snapshots and immediately verifies replay.
    dsfr_capture=await capture_dsfr_dataset(records,"dev20_v9_dsfr")
    contract_capture=await capture_contract_dataset(records,dsfr_capture["results"],"dev20_v9_contract")
    chosen=freeze["chosen"];metrics,products,details=evaluate_profile(records,dsfr_capture["results"],contract_capture["results"],
        threshold=float(chosen["threshold"]),margin=float(chosen["margin"]),max_replacements=int(chosen["max_replacements"]))
    base_k2=baseline_metrics(records);base_v6=baseline_configuration(records)
    routed_v6=aggregate_metrics(records,v6_selection(records,products["top20"]))
    rows=[base_k2,{"configuration":"B0_v6",**base_v6},{"configuration":"ECRank_K2",**metrics},
          {"configuration":"ECRank_plus_v6","selected_TP":routed_v6["TP"],"selected_FP":routed_v6["FP"],"selected_FN":routed_v6["FN"],
           "precision":routed_v6["precision"],"recall":routed_v6["recall"],"f1":routed_v6["f1"],"mean_output_count":routed_v6["mean_output_count"]}]
    write_csv(REPORT/"04_dev20_one_shot.csv",rows);write_csv(REPORT/"05_dev_query_actions.csv",details)
    accepted=(metrics["selected_TP"]>=base_v6["selected_TP"]+1 and metrics["selected_FP"]<=base_v6["selected_FP"] and metrics["f1"]>=base_v6["f1"]+.01 and metrics["recall"]>=base_v6["recall"])
    accepted_v6=(routed_v6["TP"]>=base_v6["selected_TP"]+1 and routed_v6["FP"]<=base_v6["selected_FP"]+2 and routed_v6["f1"]>=base_v6["f1"]+.01)
    state={"schema_version":1,"status":"development_candidate_v9" if accepted or accepted_v6 else "rejected_on_dev",
           "branch":"evidence-calibrated-ranking-v9","source_commit":freeze["source_commit"],"blind_used":False,"dev_live_runs":1,
           "retrieval_replay_identical":dsfr_capture["manifest"]["replay_all_identical"],"contract_replay_identical":contract_capture["manifest"]["replay_all_identical"],
           "baseline_k2":base_k2,"baseline_v6":base_v6,"ecrank_k2":metrics,
           "ecrank_plus_v6":{"TP":routed_v6["TP"],"FP":routed_v6["FP"],"FN":routed_v6["FN"],"precision":routed_v6["precision"],"recall":routed_v6["recall"],"f1":routed_v6["f1"],"mean_output_count":routed_v6["mean_output_count"]},
           "accepted_ecrank_k2":accepted,"accepted_ecrank_plus_v6":accepted_v6}
    (REPORT/"state.json").write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8")
    if accepted or accepted_v6:(CONFIG/"development_candidate_v9.json").write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8")
    return state


def write_method_report() -> None:
    ensure_dirs()
    text="""# ECRank v9：方法与文献定位

## 真实问题

v8在calibration40找回4篇原候选池外gold，但只有2篇通过A0，0篇进入Top20，并挤掉1篇原Top20 gold。问题已从“检索不到”收缩为“多条件证据无法让低位新候选安全击败已有候选”。

## 方法

Evidence Contract Rank（ECRank）由四步组成：

1. **Complementary compact shortlist**：融合TRACE分数、facet bottleneck和source rank三个粗排视图，从大量DSFR候选中选12篇；与4篇incumbent组成16篇紧凑列表。
2. **Executable query contract**：LLM把复杂查询编译为required conditions、accepted aliases/subtypes、requested relation与exclusions，解决术语位置和隐式taxonomy问题。
3. **Point-list self-calibration**：同一次listwise调用同时给出独立point score、比较式list score、逐条件0-3证据和relation support；最终分数包含minimum-support bottleneck，未满足任一必要条件的候选不能只靠主题相似度胜出。
4. **Risk-bounded routing**：只有evidence-complete newcomer同时超过绝对阈值和incumbent margin时，才允许替换Top2；阈值在calibration40冻结，dev不调参。

## 与现有工作的差异

- CoRank使用紧凑语义特征扩大科学文献reranking覆盖；ECRank进一步把查询条件也编译为可核验契约，并用于安全替换。
- MultiConIR指出多条件查询存在复杂度退化、单调性和格式敏感；ECRank直接用minimum-support bottleneck约束相关性单调性。
- REARANK采用reasoning listwise reranking；ECRank增加point/list双视图、逐条件证据和incumbent风险边界。
- SCaLR通过point-view校准list-view；ECRank在无需训练大模型的情况下把这一思想落到一次结构化推理调用。
- SetR强调集合层面的需求覆盖；ECRank面向论文检索的Top2/Top20双层集合，限制新候选对已有高价值结果的破坏。

## 参考

- https://arxiv.org/abs/2505.13757
- https://aclanthology.org/2025.findings-emnlp.726/
- https://aclanthology.org/2025.emnlp-main.125/
- https://arxiv.org/abs/2411.04602
- https://aclanthology.org/2025.acl-long.861/
"""
    (REPORT/"00_method_and_literature.md").write_text(text,encoding="utf-8")


def final_report() -> dict[str, Any]:
    freeze=json.loads((CONFIG/"calibration_freeze.json").read_text(encoding="utf-8"));state_path=REPORT/"state.json"
    state=json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"status":"calibration_only"}
    base=freeze["baseline"];chosen=freeze["chosen"]
    lines=["# Evidence Contract Rank v9 最终报告","","## calibration40","",
           "| 配置 | TP/FP/FN | P | R | F1 | Top20 gain/lost | actions | LLM calls | tokens/query |","|---|---:|---:|---:|---:|---:|---:|---:|---:|",
           f"| K2 baseline | {base['selected_TP']}/{base['selected_FP']}/{base['selected_FN']} | {base['precision']:.4f} | {base['recall']:.4f} | {base['f1']:.4f} | 0/0 | 0 | 0 | 0 |",
           f"| ECRank | {chosen['selected_TP']}/{chosen['selected_FP']}/{chosen['selected_FN']} | {chosen['precision']:.4f} | {chosen['recall']:.4f} | {chosen['f1']:.4f} | {chosen['top20_gain']}/{chosen['top20_lost']} | {chosen['replacement_actions']} | {chosen['llm_calls']} | {chosen['tokens_per_query']:.1f} |","",
           f"- 冻结阈值/margin/max replacements：{chosen['threshold']}/{chosen['margin']}/{chosen['max_replacements']}。",
           f"- candidate TP：{base['candidate_TP']} → {chosen['candidate_TP']}；Top20 TP：{base['top20_TP']} → {chosen['top20_TP']}。",
           f"- calibration gate：{freeze['dev_gate_passed']}。","","## dev20","",f"- 状态：{state.get('status')}。"]
    if state.get("ecrank_k2"):
        d0=state["baseline_v6"];d1=state["ecrank_k2"];d2=state["ecrank_plus_v6"]
        lines += [f"- v6 baseline P/R/F1：{d0['precision']:.4f}/{d0['recall']:.4f}/{d0['f1']:.4f}，TP/FP={d0['selected_TP']}/{d0['selected_FP']}。",
                  f"- ECRank K2 P/R/F1：{d1['precision']:.4f}/{d1['recall']:.4f}/{d1['f1']:.4f}，TP/FP={d1['selected_TP']}/{d1['selected_FP']}。",
                  f"- ECRank+v6 P/R/F1：{d2['precision']:.4f}/{d2['recall']:.4f}/{d2['f1']:.4f}，TP/FP={d2['TP']}/{d2['FP']}。"]
    lines += ["","## 论文证据链","","- 已实现：契约结构、compact shortlist、point/list双视图、minimum-support bottleneck、风险受控替换、LIVE/FROZEN回放。",
              "- 消融：point-only、list-only、去bottleneck、full ECRank。","- 结果是否可作为主方法，以dev一次性结果为准；blind未运行。"]
    (REPORT/"final_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return {"freeze":freeze,"state":state}


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=("method","calibration","dev","report"));args=parser.parse_args()
    if args.phase=="method":result=write_method_report() or {"written":True}
    elif args.phase=="calibration":result=asyncio.run(run_calibration())
    elif args.phase=="dev":result=asyncio.run(run_dev())
    else:result=final_report()
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))


if __name__=="__main__":main()
