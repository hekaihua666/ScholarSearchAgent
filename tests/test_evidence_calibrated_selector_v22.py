from __future__ import annotations

import math

from backend.selection.evidence_calibrated_selector_v22 import (
    FEATURE_NAMES,
    EvidenceCalibratedSelectorV22,
    selector_feature_vector,
)


def test_feature_vector_is_fixed_and_finite():
    vector = selector_feature_vector(
        "papers about robust graph learning",
        {
            "title": "Robust Graph Learning",
            "abstract": "A graph method for robust learning.",
            "year": 2024,
            "cited_by_count": 100,
            "v17_cross_pct": 0.8,
            "v17_graph_pct": 0.7,
        },
        rank=2,
    )
    assert len(vector) == len(FEATURE_NAMES)
    assert all(math.isfinite(value) for value in vector)
    assert vector[FEATURE_NAMES.index("title_query_overlap")] > 0


def test_manual_projection_selects_highest_probability_stably():
    dimension = len(FEATURE_NAMES)
    rank_index = FEATURE_NAMES.index("rank_prior")
    coefficients = [0.0] * dimension
    coefficients[rank_index] = 1.0
    selector = EvidenceCalibratedSelectorV22.from_dict({
        "feature_names": list(FEATURE_NAMES),
        "mean": [0.0] * dimension,
        "scale": [1.0] * dimension,
        "coefficients": coefficients,
        "intercept": 0.0,
        "core_size": 1,
    })
    selected, audit = selector.select(
        "graph learning",
        [{"title": "first"}, {"title": "second"}],
    )
    assert selected[0]["title"] == "first"
    assert audit["selector"] == "ECS_v22"


def test_feature_schema_contains_no_identity_or_label_fields():
    lowered = " ".join(FEATURE_NAMES).lower()
    for forbidden in ("gold", "label", "query_id", "paper_id", "title_hash"):
        assert forbidden not in lowered
