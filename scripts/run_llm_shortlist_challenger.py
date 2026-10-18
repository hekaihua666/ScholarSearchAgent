"""Evaluate a changed frozen shortlist with newly captured/replayable LLM rerank only."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.eval_runtime import EvaluationSession, evaluation_session, frozen_call
from backend.llm_client import llm_build_summary
from backend.trace_rank import diversify, fuse_candidates, score_candidates
from scripts.audit_evaluation import aggregate, gold_papers, load_jsonl, metrics_from_papers, source_cutoff
from scripts.run_frozen_official_benchmark import module_candidates


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def apply_llm(ranked: list[dict[str, Any]], llm_data: dict[str, Any]) -> list[dict[str, Any]]:
    papers = copy.deepcopy(ranked)
    scores = llm_data.get("relevance_scores") or {}
    flags = llm_data.get("relevance_flags") or {}
    for i, paper in enumerate(papers):
        try:
            score = max(0.0, min(float(scores.get(str(i + 1))) / 100.0, 1.0))
        except (TypeError, ValueError):
            score = None
        paper["llm_relevance_score"] = score
        if score is not None:
            paper["final_score"] = round(0.55 * float(paper.get("final_score") or 0) + 0.45 * score, 4)
        paper["llm_relevant"] = bool(flags.get(str(i + 1), score is None or score >= 0.45))
    return sorted(papers, key=lambda p: float(p.get("final_score") or 0), reverse=True)[:20]


async def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_jsonl(args.dataset)[:20]
    live_metrics=[]; replay_metrics=[]; details=[]; total_seconds=0.0; total_tokens=0
    for idx,row in enumerate(rows,1):
        source_dir=args.source_run/'queries'/row['qid']
        source_snapshot=json.loads((source_dir/'snapshot.json').read_text(encoding='utf8'))
        response=json.loads((source_dir/'live_response.json').read_text(encoding='utf8'))
        understanding=response.get('query_understanding') or {}
        cutoff=source_cutoff(row); assert cutoff is not None
        modules=module_candidates(source_snapshot,cutoff)
        included=args.modules.split(',')
        candidates=[p for name in included for p in modules[name]]
        ranked=diversify(score_candidates(fuse_candidates(candidates),understanding),limit=32,lambda_relevance=args.mmr_lambda)
        call_args={'query':row['question'],'paper_titles':[p.get('title') for p in ranked]}
        qdir=args.output_dir/'queries'/row['qid']; qdir.mkdir(parents=True,exist_ok=True)
        sp=qdir/'snapshot.json'
        if sp.exists():
            live_session=EvaluationSession.create(args.experiment_id,row['qid'],cutoff,'replay',sp)
            with evaluation_session(live_session):
                live_output=await frozen_call('llm_rerank_summary',call_args,lambda: (_ for _ in ()).throw(RuntimeError('network forbidden')))
        else:
            live_session=EvaluationSession.create(args.experiment_id,row['qid'],cutoff,'live',sp)
            with evaluation_session(live_session):
                live_output=await frozen_call('llm_rerank_summary',call_args,lambda:llm_build_summary(row['question'],ranked))
            live_session.save({'experiment_id':args.experiment_id,'modules':included,'shortlist_titles':call_args['paper_titles']})
        replay_session=EvaluationSession.create(args.experiment_id,row['qid'],cutoff,'replay',sp)
        with evaluation_session(replay_session):
            replay_output=await frozen_call('llm_rerank_summary',call_args,lambda: (_ for _ in ()).throw(RuntimeError('network forbidden')))
        if digest(live_output)!=digest(replay_output): raise RuntimeError(f'replay mismatch {row["qid"]}')
        live_data,live_tok=live_output; replay_data,_=replay_output
        live_top=apply_llm(ranked,live_data); replay_top=apply_llm(ranked,replay_data)
        gold=gold_papers(row)
        lm=metrics_from_papers(live_top,gold,20); rm=metrics_from_papers(replay_top,gold,20)
        live_metrics.append(lm);replay_metrics.append(rm)
        call=next(iter(live_session.calls.values()));total_seconds+=float(call.get('elapsed_seconds') or 0);total_tokens+=int(live_tok or 0)
        details.append({'query_id':row['qid'],'shortlist_count':len(ranked),'live_replay_identical':digest(live_top)==digest(replay_top),
                        'tp20':lm['tp'],'metrics':lm,'final_titles':[p.get('title') for p in live_top],
                        'tokens':live_tok,'seconds':call.get('elapsed_seconds')})
        print(f'[{idx}/20] {row["qid"]} tp={lm["tp"]} replay=OK',flush=True)
    result={'experiment_id':args.experiment_id,'query_set':'dev20','modules':args.modules.split(','),'query_count':20,
            'live_metrics':aggregate(live_metrics),'replay_metrics':aggregate(replay_metrics),
            'replay_all_identical':all(d['live_replay_identical'] for d in details),'llm_seconds':round(total_seconds,3),
            'llm_tokens':total_tokens,'details':details}
    (args.output_dir/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    return result


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,default=Path(r'D:\AICompetition\references\github\SPAR\benchmark\AutoScholarQuery_test.jsonl'))
    parser.add_argument('--experiment-id',default='llm_baseD_no_expansion')
    parser.add_argument('--modules',default='lexical,semantic,arxiv,title')
    parser.add_argument('--mmr-lambda',type=float,default=0.88)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'reports/frozen_eval/llm_challengers/llm_baseD_no_expansion')
    args=parser.parse_args(); result=asyncio.run(run(args)); print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
