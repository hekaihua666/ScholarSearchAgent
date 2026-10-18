"""Runtime materialization of frozen v4 selector champions."""

from __future__ import annotations

import asyncio
import json
import math
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from backend.selection.atomic_constraint_selector import judge_paper
from backend.selection.cross_encoder_selector import CPU_FALLBACK_MODEL, CrossEncoderSelector
from backend.selection.evidence_calibrated_selector_v22 import EvidenceCalibratedSelectorV22

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/joint_optimization_v4"
EVIDENCE_V22_MODEL = ROOT / "configs/evidence_calibrated_selector_v22/model.json"


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


@lru_cache(maxsize=1)
def _artifacts() -> dict[str, Any] | None:
    model_path = REPORT / "selector_model.pkl"
    if not model_path.exists():
        return None
    with model_path.open("rb") as handle:
        return pickle.load(handle)


@lru_cache(maxsize=1)
def _cross_encoder() -> CrossEncoderSelector:
    return CrossEncoderSelector(
        CPU_FALLBACK_MODEL,
        ROOT / "cache/v4_cross_encoder_scores.json",
        max_length=512,
    )


def _champion(profile: str) -> dict[str, Any] | None:
    filename = "champion_f1.json" if profile == "f1" else "champion_competition.json"
    path = REPORT / filename
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@lru_cache(maxsize=1)
def _evidence_v22() -> EvidenceCalibratedSelectorV22 | None:
    return EvidenceCalibratedSelectorV22.load(EVIDENCE_V22_MODEL) if EVIDENCE_V22_MODEL.exists() else None


def _feature(paper: dict[str, Any], names: list[str] | tuple[str, ...]) -> list[float]:
    values = {
        "llm": float(paper.get("llm_relevance_score") or 0),
        "cross": float(paper.get("cross_encoder_probability") or 0),
        "ace": float(paper.get("ace_probability") or 0),
        "trace": float(paper.get("trace_score_before_joint") or paper.get("final_score") or 0),
        "rank": 1.0 / max(int(paper.get("live_rank") or 100), 1),
        "abstract": float(bool(paper.get("abstract"))),
        "coverage": float(paper.get("constraint_coverage_score") or 0),
    }
    return [values[name] for name in names]


async def select_with_champion(
    query: str,
    constraint_graph: dict[str, Any],
    papers: list[dict[str, Any]],
    profile: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # The product-facing balanced profile is a stable eight-result view.  The
    # frozen competition champion remains available to the F1 profile only;
    # allowing its Top20 decision to leak into this branch made the API output
    # disagree with the interface label and export semantics.
    if profile == "balanced_set":
        selected = list(papers[:8])
        return selected, {
            "strategy": "fixed_cardinality_balanced_set",
            "selector": "K8_balanced_set",
            "profile": profile,
            "selected_count": len(selected),
        }
    champion = _champion(profile)
    if champion and champion.get("decision") == "evidence_calibrated_k1":
        selector = _evidence_v22()
        if selector is not None:
            selected, metadata = selector.select(query, papers, incumbent_count=min(20, len(papers)))
            metadata["profile"] = profile
            return selected, metadata
        return list(papers[:1]), {
            "strategy": "fixed_cardinality_fallback",
            "reason": "evidence_v22_model_missing",
            "profile": profile,
            "selected_count": min(1, len(papers)),
        }
    artifacts = _artifacts()
    if not champion or not artifacts:
        k = 2 if profile == "f1" else 8
        return list(papers[:k]), {"strategy": "fixed_cardinality_fallback", "profile": profile, "selected_count": min(k, len(papers))}
    if champion.get("decision") == "fixed_k2":
        return list(papers[:2]), {"strategy": "frozen_v4_champion", "selector": "S0_current_K2", "profile": profile, "selected_count": min(2, len(papers))}
    if champion.get("decision") == "fixed_top20":
        return list(papers[:20]), {"strategy": "frozen_v4_champion", "selector": "S0_current_Top20", "profile": profile, "selected_count": min(20, len(papers))}
    if champion.get("decision") == "fixed_llm090":
        selected = [paper for paper in papers if float(paper.get("llm_relevance_score") or 0) >= 0.9]
        return selected, {"strategy": "frozen_v4_champion", "selector": "S1_current_LLM090", "profile": profile, "selected_count": len(selected)}
    selector_name = str(champion["selector"])
    model = (artifacts.get("models") or {}).get(selector_name)
    names = tuple(champion.get("feature_names") or (artifacts.get("variants") or {}).get(selector_name) or ())
    if model is None or not names:
        k = 2 if profile == "f1" else 8
        return list(papers[:k]), {"strategy": "fixed_cardinality_fallback", "reason": "champion_model_missing", "profile": profile, "selected_count": min(k, len(papers))}

    for rank, paper in enumerate(papers, 1):
        paper["live_rank"] = rank
        paper["trace_score_before_joint"] = float(paper.get("final_score") or 0)
    runtime: dict[str, Any] = {"runtime": "not_required_by_frozen_champion"}
    if "cross" in names:
        documents = [f"{paper.get('title', '')}. {str(paper.get('abstract') or '')[:1800]}" for paper in papers]
        criteria = "; ".join(item.get("description", "") for item in constraint_graph.get("criteria") or [])
        pairs = [(f"{query} Required constraints: {criteria}", document) for document in documents]
        raw, runtime = await asyncio.to_thread(_cross_encoder().predict, pairs, 32)
        for paper, cross_raw in zip(papers, raw):
            paper["cross_encoder_raw"] = cross_raw
            paper["cross_encoder_probability"] = _sigmoid(cross_raw)
    if "ace" in names:
        for paper in papers:
            ace = judge_paper(paper, constraint_graph)
            paper["ace_probability"] = ace["raw_probability"]
            paper["ace_hard_gate"] = ace["hard_gate"]
            paper["ace_evidence_coverage"] = ace["evidence_coverage"]
    probabilities = model.predict_proba(np.asarray([_feature(paper, names) for paper in papers], dtype=float))[:, 1]
    for paper, probability in zip(papers, probabilities):
        paper["joint_selector_probability"] = float(probability)
    threshold = float(champion.get("threshold") or 0.5)
    max_results = 8 if profile == "f1" else 10
    ordered = sorted(papers, key=lambda paper: float(paper.get("joint_selector_probability") or 0), reverse=True)
    selected = [paper for paper in ordered if paper.get("ace_hard_gate") != "fail" and float(paper["joint_selector_probability"]) >= threshold][:max_results]
    if len(selected) < 2:
        selected = [paper for paper in ordered if paper.get("ace_hard_gate") != "fail"][:2]
    return selected, {
        "strategy": "frozen_v4_champion",
        "selector": selector_name,
        "profile": profile,
        "threshold": threshold,
        "selected_count": len(selected),
        "cross_encoder": runtime,
    }
