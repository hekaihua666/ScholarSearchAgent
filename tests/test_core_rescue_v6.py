from __future__ import annotations

import json

import numpy as np

from backend.selection.core_rescue_selector_v6 import (
    FEATURE_NAMES, CoreRescueModelV6, assert_safe_schema,
)


def sample_rows():
    rows=[]
    for query_id in ("q1","q2","q3"):
        for rank in range(3,9):
            row={name:0.1 for name in FEATURE_NAMES}
            row.update({"query_id":query_id,"candidate_rank":rank,"paper_id":f"{query_id}-{rank}",
                        "label":int(rank==3),"hard_constraint_fail":0.0,"max_core_text_similarity":0.1,
                        "fusion_relative":1-(rank-3)/6})
            rows.append(row)
    return rows


def fitted_model():
    rows=sample_rows(); model=CoreRescueModelV6.fit(rows,family="logistic"); model.threshold=0.0
    return model


def test_feature_schema_has_no_gold_or_identity_memory():
    assert_safe_schema()
    assert not any(token in name for name in FEATURE_NAMES for token in ("gold","query_id","paper_id"))


def test_core_top2_is_never_modified():
    model=fitted_model(); core=[{"paper_id":"a"},{"paper_id":"b"}]
    selected=model.select(core,sample_rows()[:6])
    assert selected[:2]==core


def test_at_most_one_rescue_and_three_total_results():
    model=fitted_model(); selected=model.select([{"paper_id":"a"},{"paper_id":"b"}],sample_rows()[:6])
    assert len(selected)<=3 and len(selected[2:])<=1


def test_hard_constraint_failure_cannot_be_rescued():
    model=fitted_model(); rows=sample_rows()[:6]
    for row in rows: row["hard_constraint_fail"]=1.0
    assert model.choose_rescue(rows) is None


def test_near_duplicate_cannot_be_rescued():
    model=fitted_model(); rows=sample_rows()[:6]
    for row in rows: row["max_core_text_similarity"]=0.95
    assert model.choose_rescue(rows) is None


def test_threshold_replay_is_deterministic(tmp_path):
    model=fitted_model(); model.threshold=0.37
    path=tmp_path/"rescue.pkl"; model.save(path); replay=CoreRescueModelV6.load(path)
    rows=sample_rows()[:6]
    assert replay.threshold==0.37
    assert (model.choose_rescue(rows) or {}).get("paper_id")== (replay.choose_rescue(rows) or {}).get("paper_id")


def test_query_groups_are_disjoint_by_construction():
    rows=sample_rows(); by_query={qid:[row for row in rows if row["query_id"]==qid] for qid in {row["query_id"] for row in rows}}
    assert sum(len(group) for group in by_query.values())==len(rows)
    assert all(len({row["query_id"] for row in group})==1 for group in by_query.values())


def test_live_frozen_capture_manifest_when_available():
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/"reports/core_rescue_v6/01_production_manifest.json"
    if not path.exists(): return
    manifest=json.loads(path.read_text(encoding="utf-8"))
    assert manifest["replay_all_identical"] is True
    assert manifest["all_final_papers_date_legal"] is True
    assert all(row["live_result_hash"]==row["replay_result_hash"] for row in manifest["queries"])
