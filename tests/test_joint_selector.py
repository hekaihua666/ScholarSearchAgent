import asyncio

import numpy as np

from backend.selection import joint_selector
from scripts.run_v4_selector_experiments import feature, grouped_oof


def test_joint_selector_has_deterministic_fallback(monkeypatch) -> None:
    monkeypatch.setattr(joint_selector, "_champion", lambda profile: None)
    papers = [{"title": f"Paper {index}"} for index in range(12)]
    selected, metadata = asyncio.run(
        joint_selector.select_with_champion("query", {"criteria": []}, papers, "balanced_set")
    )
    assert len(selected) == 8
    assert metadata["strategy"] == "fixed_cardinality_balanced_set"


def test_balanced_set_is_always_eight_results(monkeypatch) -> None:
    monkeypatch.setattr(
        joint_selector,
        "_champion",
        lambda profile: {"decision": "fixed_top20"},
    )
    papers = [{"title": f"Paper {index}"} for index in range(20)]
    selected, metadata = asyncio.run(
        joint_selector.select_with_champion("query", {"criteria": []}, papers, "balanced_set")
    )
    assert len(selected) == 8
    assert metadata["selector"] == "K8_balanced_set"


def test_llm_only_champion_skips_cross_encoder(monkeypatch) -> None:
    class Model:
        def predict_proba(self, values):
            scores = np.asarray(values, dtype=float)[:, 0]
            return np.column_stack([1.0 - scores, scores])

    champion = {
        "selector": "S1_llm_calibrated", "decision": "f1_threshold",
        "threshold": 0.9, "feature_names": ["llm"],
    }
    monkeypatch.setattr(joint_selector, "_champion", lambda profile: champion)
    monkeypatch.setattr(
        joint_selector, "_artifacts",
        lambda: {"models": {"S1_llm_calibrated": Model()}, "variants": {}},
    )
    monkeypatch.setattr(
        joint_selector, "_cross_encoder",
        lambda: (_ for _ in ()).throw(AssertionError("cross encoder must not load")),
    )
    papers = [
        {"title": f"Paper {index}", "llm_relevance_score": score, "final_score": score}
        for index, score in enumerate((0.95, 0.91, 0.6, 0.4))
    ]
    selected, metadata = asyncio.run(
        joint_selector.select_with_champion("query", {"criteria": []}, papers, "f1")
    )
    assert len(selected) == 2
    assert metadata["cross_encoder"]["runtime"] == "not_required_by_frozen_champion"


def test_gold_fields_never_change_production_features() -> None:
    row = {"llm_relevance_score": 0.8, "candidate_rank": 2, "label": 0}
    leaked = {**row, "gold_title": "secret target", "matched_gold_index": 7, "label": 1}
    names = ("llm", "rank", "trace", "coverage")
    assert feature(row, names) == feature(leaked, names)


def test_grouped_calibration_replays_deterministically() -> None:
    rows = []
    for query_index in range(10):
        rows.extend([
            {"query_id": f"q{query_index}", "label": 1, "llm_relevance_score": 0.9, "candidate_rank": 1},
            {"query_id": f"q{query_index}", "label": 0, "llm_relevance_score": 0.1, "candidate_rank": 2},
        ])
    first, metrics_first = grouped_oof(rows, ("llm",), folds=5)
    second, metrics_second = grouped_oof(rows, ("llm",), folds=5)
    assert np.array_equal(first, second)
    assert metrics_first == metrics_second
