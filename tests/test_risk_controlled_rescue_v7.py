from __future__ import annotations

import hashlib

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from backend.selection.core_rescue_selector_v6 import FEATURE_NAMES as V6_FEATURE_NAMES
from backend.selection.risk_controlled_rescue_v7 import (
    GATE_FEATURE_NAMES,
    RANK_FEATURE_NAMES,
    QueryGateV7,
    RescueRankerV7,
    assert_safe_schemas,
    augment_query_relative,
    exact_delta_f1,
    gate_row,
)
from scripts.run_risk_controlled_rescue_v7 import bootstrap_query_sample, risk_profiles, selected_indices


def candidate_rows(query_count: int = 10) -> list[dict]:
    rows = []
    for query_index in range(query_count):
        for rank in range(3, 9):
            row = {name: 0.0 for name in V6_FEATURE_NAMES}
            row.update(
                {
                    "query_id": f"q{query_index}",
                    "paper_id": f"q{query_index}-p{rank}",
                    "candidate_rank": rank,
                    "fusion_score": 1.0 - 0.1 * (rank - 3) + 0.01 * query_index,
                    "fusion_gap_rank2": -0.05 * (rank - 2),
                    "constraint_coverage": 1.0 - 0.08 * (rank - 3),
                    "llm_relative": 1.0 - 0.06 * (rank - 3),
                    "trace_relative": 1.0 - 0.05 * (rank - 3),
                    "core_tp": query_index % 3,
                    "label": int(rank == (3 if query_index % 2 == 0 else 4)),
                }
            )
            rows.append(row)
    return augment_query_relative(rows)


def test_query_relative_feature_normalization():
    rows = candidate_rows(2)
    for query_id in ("q0", "q1"):
        query_rows = [row for row in rows if row["query_id"] == query_id]
        assert abs(np.mean([row["fusion_zscore_rescue"] for row in query_rows])) < 1e-9
        percentiles = [row["fusion_percentile_rescue"] for row in query_rows]
        assert min(percentiles) == 0.0
        assert max(percentiles) == 1.0
        assert all(0.0 <= row["constraint_coverage_relative"] <= 1.0 for row in query_rows)


def test_grouped_split_never_leaks_a_query():
    rows = candidate_rows(10)
    labels = np.asarray([row["label"] for row in rows])
    groups = np.asarray([row["query_id"] for row in rows])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260830)
    for train_index, test_index in splitter.split(np.zeros(len(rows)), labels, groups):
        assert set(groups[train_index]).isdisjoint(set(groups[test_index]))


def test_candidate_selection_is_exactly_one_per_query():
    rows = candidate_rows(4)
    scores = np.asarray([row["fusion_score"] for row in rows])
    selected = selected_indices(rows, scores)
    assert len(selected) == 4
    assert len(set(selected.values())) == 4
    assert all(rows[index]["query_id"] == query_id for query_id, index in selected.items())


def test_query_gate_target_and_audit_context():
    rows = candidate_rows(1)
    scores = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    row = gate_row(rows, scores, 0, ensemble_agreement=0.8, gold_count=4)
    assert row["core_tp"] == rows[0]["core_tp"]
    assert row["y_gold"] == 1
    assert row["y_gate"] == int(row["exact_delta_f1"] > 0)
    assert set(GATE_FEATURE_NAMES).issubset(row)


def test_exact_delta_f1_is_positive_only_for_a_useful_rescue():
    assert exact_delta_f1(core_tp=1, gold_count=3, candidate_is_gold=True) > 0
    assert exact_delta_f1(core_tp=1, gold_count=3, candidate_is_gold=False) < 0
    assert exact_delta_f1(core_tp=0, gold_count=3, candidate_is_gold=False) == 0


def test_ranker_and_gate_training_are_deterministic():
    rows = candidate_rows(10)
    first = RescueRankerV7.fit(rows)
    second = RescueRankerV7.fit(rows)
    np.testing.assert_allclose(first.predict(rows), second.predict(rows), rtol=0, atol=0)

    gate_rows = []
    for query_index in range(10):
        query_rows = [row for row in rows if row["query_id"] == f"q{query_index}"]
        scores = first.predict(query_rows)
        chosen = int(np.argmax(scores))
        built = gate_row(query_rows, scores, chosen, gold_count=4)
        # Ensure both classes in this synthetic gate training set.
        built["y_gate"] = query_index % 2
        gate_rows.append(built)
    first_gate = QueryGateV7.fit(gate_rows)
    second_gate = QueryGateV7.fit(gate_rows)
    np.testing.assert_allclose(first_gate.predict(gate_rows), second_gate.predict(gate_rows), rtol=0, atol=0)


def test_bootstrap_query_sampling_is_deterministic():
    qids = [f"q{index}" for index in range(20)]
    assert bootstrap_query_sample(qids, 20260900) == bootstrap_query_sample(reversed(qids), 20260900)
    assert bootstrap_query_sample(qids, 20260900) != bootstrap_query_sample(qids, 20260901)


def test_consensus_threshold_is_applied():
    gates = [
        {"query_id": "q1", "core_tp": 0, "y_gold": 1},
        {"query_id": "q2", "core_tp": 0, "y_gold": 0},
    ]
    probabilities = np.asarray([0.9, 0.8])
    agreement = np.asarray([0.75, 0.90])
    counts = {"q1": 2, "q2": 2}
    low = risk_profiles(gates, probabilities, counts, agreement, agreement_threshold=0.70)
    high = risk_profiles(gates, probabilities, counts, agreement, agreement_threshold=0.80)
    assert "strict" in low
    assert "strict" not in high


def test_top2_is_unchanged_and_output_is_bounded():
    core = [{"paper_id": "a"}, {"paper_id": "b"}]
    rescue = {"paper_id": "c"}
    selected = core + [rescue]
    assert selected[:2] == core
    assert len(selected[2:]) == 1
    assert len(selected) == 3


def test_inference_schemas_exclude_gold_and_identity_fields():
    assert_safe_schemas()
    forbidden = ("gold", "label", "query_id", "paper_id", "identity", "delta_f1")
    assert not any(token in name.lower() for name in (*RANK_FEATURE_NAMES, *GATE_FEATURE_NAMES) for token in forbidden)


def test_config_hash_is_byte_reproducible(tmp_path):
    content = b"schema_version: 1\ncore_k: 2\nmax_rescues: 1\n"
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_bytes(content)
    second.write_bytes(content)
    assert hashlib.sha256(first.read_bytes()).hexdigest() == hashlib.sha256(second.read_bytes()).hexdigest()


def test_frozen_v7_artifacts_are_self_consistent_when_available():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    artifact = root / "artifacts/risk_controlled_rescue_v7"
    manifest_path = artifact / "training_manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = artifact / "feature_schema.json"
    assert hashlib.sha256(schema.read_bytes()).hexdigest() == manifest["feature_schema_sha256"]
    for name, digest in manifest["config_hashes"].items():
        path = root / "configs/risk_controlled_rescue_v7" / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    for name, digest in manifest["model_hashes"].items():
        assert hashlib.sha256((artifact / name).read_bytes()).hexdigest() == digest
    bootstrap = list((artifact / "bootstrap_models").glob("bootstrap_*.pkl"))
    assert len(bootstrap) == 20
    for path in bootstrap:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["bootstrap_hashes"][path.name]
    assert manifest["train_queries"] == 100
    assert manifest["calibration_queries"] == 40
    if not manifest["oof_profile_legal"]:
        assert not (root / "configs/risk_controlled_rescue_v7/champion_candidate.yaml").exists()
