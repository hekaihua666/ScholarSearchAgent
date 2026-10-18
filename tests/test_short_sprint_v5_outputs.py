from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/short_sprint_v5"
ARTIFACT = ROOT / "artifacts/short_sprint_v5"


def test_short_sprint_split_isolation_and_holdout_flags():
    state = json.loads((REPORT / "state.json").read_text(encoding="utf-8"))
    assert state["blind50_used"] is False
    assert state["validation30_gold_used"] is False
    audit = (REPORT / "01_selector_data_audit.md").read_text(encoding="utf-8")
    for name in ("train_calibration", "train_dev", "train_validation", "train_blind", "calibration_dev", "calibration_blind"):
        assert f"'{name}': 0" in audit


def test_all_legal_train_gold_was_materialized_before_identity_deduplication():
    with (REPORT / "01_positive_coverage.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "train"]
    assert len(rows) == 500
    assert sum(int(row["materialized_positives"]) for row in rows) == 1212
    assert sum(int(row["missing_metadata"]) for row in rows) == 0
    assert sum(int(row["identity_failures"]) for row in rows) == 0
    assert sum(int(row["date_filtered"]) for row in rows) == 10


def test_materialized_positive_cutoffs_are_legal():
    rows = pq.read_table(ARTIFACT / "selector_train_pairs.parquet").to_pylist()
    materialized = [row for row in rows if row.get("gold_materialized")]
    assert materialized
    for row in materialized:
        published = datetime.fromisoformat(str(row["published"]).replace("Z", "+00:00")).replace(tzinfo=None)
        cutoff = datetime.fromisoformat(str(row["cutoff"])).replace(tzinfo=None)
        assert published <= cutoff


def test_same_snapshot_k2_baseline_is_replay_stable():
    with (REPORT / "00_same_snapshot_baselines.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = {row["configuration"]: row for row in csv.DictReader(handle)}
    k2 = rows["K=2"]
    assert int(k2["TP"]) == 10 and int(k2["FP"]) == 30 and int(k2["FN"]) == 48
    assert abs(float(k2["f1"]) - 0.22916666666666666) < 1e-12
