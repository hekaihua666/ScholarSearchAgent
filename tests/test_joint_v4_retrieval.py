from __future__ import annotations

import importlib
import json
from pathlib import Path

from backend.query_portfolio import atomicize_query, build_query_portfolio
from backend.retrieval_frontier import choose_dual_frontier


def test_query_atomicization_schema_and_relation() -> None:
    understanding = {
        "research_topic": "target networks deep Q learning",
        "method_constraints": "target network",
        "task_constraints": "reinforcement learning",
        "domain_constraints": "not specified",
        "time_range": "not specified",
        "search_intent": "method_lineage",
    }
    graph = atomicize_query(
        "Studies that analyse using target networks for Deep Q-learning",
        understanding,
    )
    assert 2 <= len(graph["criteria"]) <= 8
    assert all(
        {"criterion_id", "description", "type", "importance", "verification_target", "unknown_allowed"}
        <= set(item)
        for item in graph["criteria"]
    )
    assert graph["relation_constraints"] == ["uses"]
    assert any(item["importance"] == "required" for item in graph["criteria"])


def test_query_portfolio_is_small_and_differentiated() -> None:
    query = "Papers comparing graph attention and transformers for multivariate anomaly detection"
    understanding = {
        "research_topic": "multivariate anomaly detection",
        "method_constraints": "graph attention, transformer",
        "task_constraints": "anomaly detection",
        "domain_constraints": "time series",
        "time_range": "not specified",
        "search_intent": "comparison",
    }
    graph = atomicize_query(query, understanding)
    portfolio = build_query_portfolio(query, understanding, graph)
    assert 2 <= len(portfolio) <= 6
    assert len({item["query"].lower() for item in portfolio}) == len(portfolio)
    assert all("serves" in item and "drift_score" in item for item in portfolio)
    assert portfolio[0]["query"] == query


def test_bridge_frontier_is_exploration_only() -> None:
    papers = []
    for index in range(10):
        papers.append(
            {
                "title": f"Paper {index}",
                "openalex_id": f"W{index}",
                "final_score": 1 - index / 20,
                "lexical_score": 0.5 if index < 4 else 0.1,
                "constraint_coverage_score": 0.5 if index < 4 else 0.1,
                "semantic_provenance_score": 0.2,
                "referenced_works": [f"R{index}-{j}" for j in range(index + 1)],
                "related_works": [f"X{index}-{j}" for j in range(3)],
            }
        )
    frontiers = choose_dual_frontier(papers, relevant_limit=3, bridge_limit=3)
    assert len(frontiers["relevant"]) == 3
    assert len(frontiers["bridge"]) == 3
    assert not ({p["openalex_id"] for p in frontiers["relevant"]} & {p["openalex_id"] for p in frontiers["bridge"]})
    assert all(p["frontier_role"] == "bridge" for p in frontiers["bridge"])
    # The frontier controller returns no selected set and makes no acceptance decision.
    assert "selected" not in frontiers


def test_oracle_audit_is_isolated_from_backend() -> None:
    audit = importlib.import_module("scripts.audit_missing_gold_retrievability")
    assert hasattr(audit, "load_missing")
    backend_root = Path(__file__).resolve().parents[1] / "backend"
    for source in backend_root.glob("*.py"):
        text = source.read_text(encoding="utf-8")
        assert "audit_missing_gold_retrievability" not in text
        assert "gold_title" not in text


def test_blind_holdout_is_disjoint_and_frozen() -> None:
    root = Path(__file__).resolve().parents[1]
    sets = json.loads((root / "configs/query_sets.json").read_text(encoding="utf-8"))
    blind = set(sets["blind_holdout50_v4"])
    assert len(blind) == 50
    assert not blind & set(sets["dev20"])
    assert not blind & set(sets["validation30"])
    manifest = json.loads((root / "reports/joint_optimization_v4/02_split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["private_subset_sha256"] == "5057eb469f962a6e471c09ab430d9bad09ff855019570167b4c2df149fd43d59"
