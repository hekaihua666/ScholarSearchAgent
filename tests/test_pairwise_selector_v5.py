from __future__ import annotations

import pickle

import numpy as np

from backend.selection.pairwise_selector_v5 import (
    FEATURE_NAMES,
    PairwiseSelectorV5,
    assert_safe_feature_schema,
    build_pairwise_examples,
    feature_rows,
    mine_hard_negative_indices,
)
from backend.selection.set_decision import select_final_set


def rows():
    return [
        {"query_id": "q1", "query": "graph retrieval method", "paper_id": "p1", "title": "Graph retrieval method", "abstract": "graph retrieval", "year": 2020, "cutoff": "2022-01-01", "candidate_rank": 1, "candidate_pool_size": 3, "trace_score": .9, "lexical_score": .8, "label": 1},
        {"query_id": "q1", "query": "graph retrieval method", "paper_id": "p2", "title": "Unrelated vision", "abstract": "image classification", "year": 2020, "cutoff": "2022-01-01", "candidate_rank": 2, "candidate_pool_size": 3, "trace_score": .3, "lexical_score": .2, "label": 0},
        {"query_id": "q1", "query": "graph retrieval method", "paper_id": "p3", "title": "Graph survey", "abstract": "overview", "year": 2020, "cutoff": "2022-01-01", "candidate_rank": 3, "candidate_pool_size": 3, "trace_score": .2, "lexical_score": .1, "label": 0},
        {"query_id": "q2", "query": "language agent planning", "paper_id": "p4", "title": "Language agent planning", "abstract": "planning agents", "year": 2021, "cutoff": "2022-01-01", "candidate_rank": 1, "candidate_pool_size": 2, "trace_score": .8, "lexical_score": .7, "label": 1},
        {"query_id": "q2", "query": "language agent planning", "paper_id": "p5", "title": "Database systems", "abstract": "sql database", "year": 2021, "cutoff": "2022-01-01", "candidate_rank": 2, "candidate_pool_size": 2, "trace_score": .1, "lexical_score": .1, "label": 0},
    ]


def test_feature_schema_has_no_gold_or_label_signal():
    assert_safe_feature_schema()
    assert not any("gold" in name or "label" in name for name in FEATURE_NAMES)


def test_feature_rows_are_finite_and_query_relative():
    matrix = feature_rows(rows()[:3])
    assert matrix.shape == (3, len(FEATURE_NAMES))
    assert np.isfinite(matrix).all()


def test_future_paper_fails_time_validity():
    sample = dict(rows()[0], year=2025)
    vector = feature_rows([sample])[0]
    assert vector[FEATURE_NAMES.index("time_validity")] == 0


def test_gold_materialized_flag_never_changes_features():
    left = dict(rows()[0], gold_materialized=False)
    right = dict(rows()[0], gold_materialized=True)
    assert np.allclose(feature_rows([left]), feature_rows([right]))


def test_pairwise_construction_is_symmetric_and_deterministic():
    x1, y1, pairs1 = build_pairwise_examples(rows(), seed=7)
    x2, y2, pairs2 = build_pairwise_examples(rows(), seed=7)
    assert np.allclose(x1, x2) and np.array_equal(y1, y2) and pairs1 == pairs2
    assert np.allclose(x1[0], -x1[1])


def test_pairwise_model_serialization_replays(tmp_path):
    model = PairwiseSelectorV5.fit_pairwise(rows()).fit_calibrator(rows())
    path = tmp_path / "model.pkl"; model.save(path)
    loaded = PairwiseSelectorV5.load(path)
    assert np.allclose(model.predict_proba(rows()), loaded.predict_proba(rows()))


def test_pointwise_and_pairwise_are_trainable():
    assert PairwiseSelectorV5.fit_pointwise(rows()).predict_proba(rows()).shape == (5,)
    assert PairwiseSelectorV5.fit_pairwise(rows()).predict_proba(rows()).shape == (5,)


def test_hard_negative_mining_is_deterministic_and_negative_only():
    model = PairwiseSelectorV5.fit_pairwise(rows())
    first = mine_hard_negative_indices(model, rows(), per_query=1)
    second = mine_hard_negative_indices(model, rows(), per_query=1)
    assert first == second and all(rows()[index]["label"] == 0 for index in first)


def test_dynamic_k_respects_max_bound():
    decision = select_final_set([.9] * 20, strategy="expected_f1", max_results=6)
    assert len(decision.selected_indices) <= 6


def test_query_groups_do_not_cross_pairwise_samples():
    _, _, pairs = build_pairwise_examples(rows())
    source = rows()
    assert all(source[left]["query_id"] == source[right]["query_id"] for left, right in pairs)
