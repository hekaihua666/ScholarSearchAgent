from backend.evaluation import set_f1_metrics
from backend.selection.calibration import IsotonicCalibrator, PlattCalibrator, calibration_metrics
from backend.selection.relevance_selector import RelevanceSelector
from backend.selection.set_decision import select_final_set
from backend.selection.payload import evaluation_payload
from backend.main import _selection_view


def test_set_f1_formula_and_duplicates():
    value = set_f1_metrics(["a", "a", "b"], ["a", "c"])
    assert value["true_positive"] == 1
    assert value["predicted_count"] == 2
    assert value["f1"] == 0.5


def test_dynamic_expected_f1_is_not_fixed_top20():
    decision = select_final_set([0.95, 0.8, 0.08, 0.04], strategy="expected_f1")
    assert decision.selected_indices == [0, 1]


def test_threshold_empty_and_minimum():
    assert select_final_set([0.2], strategy="global_threshold", threshold=0.5).selected_indices == []
    assert select_final_set([0.2], strategy="global_threshold", threshold=0.5, min_results=1).selected_indices == [0]


def test_dual_threshold_rescue():
    decision = select_final_set(
        [0.9, 0.45, 0.4], strategy="dual_threshold", threshold=0.6,
        low_threshold=0.35, rescue_mask=[False, True, False],
    )
    assert decision.selected_indices == [0, 1]


def test_max_result_cap_and_duplicate_indices():
    decision = select_final_set([0.9] * 30, strategy="global_threshold", threshold=0.1, max_results=5)
    assert decision.selected_indices == [0, 1, 2, 3, 4]


def test_hard_constraint_gate_and_insufficient_evidence():
    selector = RelevanceSelector(hard_constraint_min=0.5)
    query = "graph neural networks for anomaly detection"
    understanding = {"research_topic": "graph neural networks", "task_constraints": "anomaly detection"}
    good = selector.evaluate(query, understanding, {"title": "Graph neural networks for anomaly detection", "abstract": "We detect anomalies."})
    bad = selector.evaluate(query, understanding, {"title": "Graph classification", "abstract": "A benchmark."})
    sparse = selector.evaluate(query, understanding, {"title": "GNN"})
    assert good.hard_constraint_pass
    assert not bad.hard_constraint_pass
    assert sparse.label == "insufficient_evidence"


def test_calibrators_are_serializable_and_bounded():
    xs, ys = [0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]
    platt = PlattCalibrator().fit(xs, ys, steps=300)
    iso = IsotonicCalibrator().fit(xs, ys)
    assert 0 <= platt.predict_one(0.5) <= 1
    assert iso.predict_one(0.15) <= iso.predict_one(0.85)
    assert calibration_metrics([platt.predict_one(x) for x in xs], ys)["brier"] < 0.25


def test_evaluation_payload_excludes_supplementary_candidates():
    response = {
        "query": "q", "selected_results": [{"title": "selected"}],
        "supplementary_candidates": [{"title": "not submitted"}], "selection": {"profile": "f1"},
    }
    payload = evaluation_payload(response)
    assert [paper["title"] for paper in payload["papers"]] == ["selected"]


def test_search_selection_profiles_preserve_ranked_pool():
    papers = [{"title": str(i)} for i in range(20)]
    f1 = _selection_view(papers, "f1")
    balanced = _selection_view(papers, "balanced_set")
    assert len(f1["selected_results"]) == 1 and len(f1["supplementary_candidates"]) == 19
    assert len(balanced["selected_results"]) == 8 and len(balanced["ranked_candidates"]) == 20
