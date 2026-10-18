"""Atomic-Constraint Entailment (ACE) paper selector."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Literal


State = Literal["ENTAILED", "CONTRADICTED", "UNKNOWN"]
TOKEN_RE = re.compile(r"[a-z][a-z0-9+_.-]*", re.I)
RELATION_CUES = {
    "introduces": ("we propose", "we introduce", "we present", "novel ", "first "),
    "uses": ("using ", "based on", "employ", "utilize", "with "),
    "compares": ("compare", "comparison", "versus", "baseline", "outperform"),
    "evaluates": ("evaluate", "experiment", "benchmark", "results on"),
    "surveys": ("survey", "review", "overview", "systematic study"),
    "extends": ("extend", "extension", "build upon", "follow-up"),
    "replicates": ("replicate", "reproduce", "reproduction"),
    "applies-to": ("apply", "application", "for "),
}


def _tokens(value: str) -> set[str]:
    stop = {"the", "and", "for", "with", "using", "from", "paper", "study", "method", "model"}
    return {token.lower() for token in TOKEN_RE.findall(value or "") if len(token) > 1 and token.lower() not in stop}


def _span(text: str, needles: set[str], limit: int = 220) -> str:
    if not text or not needles:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    best = max(sentences, key=lambda sentence: len(_tokens(sentence) & needles), default="")
    return best.strip()[:limit] if _tokens(best) & needles else ""


def _year_range(description: str) -> tuple[int | None, int | None]:
    years = [int(value) for value in re.findall(r"(?:19|20)\d{2}", description or "")]
    if len(years) >= 2:
        return min(years), max(years)
    if years and "since" in description.lower():
        return years[0], None
    return None, None


def _judge(criterion: dict[str, Any], paper: dict[str, Any]) -> dict[str, Any]:
    description = str(criterion.get("description") or "")
    kind = str(criterion.get("type") or "topic")
    title = str(paper.get("title") or "")
    abstract = str(paper.get("abstract") or "")
    text = f"{title}. {abstract}".strip()
    lowered = f" {text.lower()} "
    target = _tokens(description)
    overlap = len(target & _tokens(text)) / max(len(target), 1)
    state: State = "UNKNOWN"
    evidence = _span(text, target)
    confidence = 0.45

    if kind == "time":
        start, end = _year_range(description)
        try:
            year = int(paper.get("year"))
        except (TypeError, ValueError):
            year = None
        if year is not None and (start is not None or end is not None):
            valid = (start is None or year >= start) and (end is None or year <= end)
            state, evidence, confidence = ("ENTAILED" if valid else "CONTRADICTED"), str(year), 1.0
    elif kind == "negative":
        if target & _tokens(text):
            state, evidence, confidence = "CONTRADICTED", _span(text, target), 0.9
    elif kind == "relation":
        relation = description.lower()
        cues = RELATION_CUES.get(relation, ())
        matched = next((cue.strip() for cue in cues if cue in lowered), "")
        if matched:
            state, evidence, confidence = "ENTAILED", _span(text, _tokens(matched)) or matched, 0.78
        elif relation == "surveys" and any(cue in lowered for cue in ("we propose", "novel method", "we introduce")):
            state, evidence, confidence = "CONTRADICTED", _span(text, {"propose", "novel", "introduce"}), 0.72
    elif kind == "paper_type":
        wanted_survey = bool({"survey", "review", "overview"} & target)
        actual_survey = any(word in title.lower() for word in ("survey", "review", "overview"))
        if wanted_survey:
            state = "ENTAILED" if actual_survey else "UNKNOWN"
            confidence = 0.92 if actual_survey else 0.4
    else:
        if overlap >= 0.60 or (len(target) == 1 and overlap == 1.0):
            state, confidence = "ENTAILED", min(0.62 + 0.35 * overlap, 0.97)
        elif overlap > 0:
            state, confidence = "UNKNOWN", 0.45 + 0.25 * overlap

    return {
        **criterion,
        "state": state,
        "evidence_span": evidence,
        "confidence": round(confidence, 4),
        "lexical_coverage": round(overlap, 4),
    }


def _state_score(state: State, unknown_score: float) -> float:
    return 1.0 if state == "ENTAILED" else 0.0 if state == "CONTRADICTED" else unknown_score


@dataclass(frozen=True)
class ACEConfig:
    unknown_score: float = 0.4
    aggregate: Literal["minimum", "geometric"] = "geometric"
    relation_weight: float = 1.8
    required_weight: float = 1.4


def judge_paper(
    paper: dict[str, Any], constraint_graph: dict[str, Any], config: ACEConfig = ACEConfig()
) -> dict[str, Any]:
    judged = [_judge(criterion, paper) for criterion in constraint_graph.get("criteria") or []]
    required = [item for item in judged if item.get("importance") == "required"]
    contradicted = [item for item in required if item["state"] == "CONTRADICTED"]
    unknown = [item for item in required if item["state"] == "UNKNOWN"]
    hard_gate = "fail" if contradicted else "unknown" if unknown else "pass"
    weighted: list[tuple[float, float]] = []
    for item in judged:
        weight = config.required_weight if item.get("importance") == "required" else 1.0
        if item.get("type") == "relation":
            weight *= config.relation_weight
        weighted.append((_state_score(item["state"], config.unknown_score), weight))
    if not weighted:
        criterion_score = config.unknown_score
    elif config.aggregate == "minimum":
        criterion_score = min(score for score, _ in weighted)
    else:
        eps = 1e-5
        criterion_score = math.exp(
            sum(weight * math.log(eps + score) for score, weight in weighted)
            / sum(weight for _, weight in weighted)
        )
    entailed_required = sum(item["state"] == "ENTAILED" for item in required)
    evidence_coverage = sum(bool(item.get("evidence_span")) for item in judged) / max(len(judged), 1)
    raw_probability = 0.0 if hard_gate == "fail" else min(
        1.0,
        0.72 * criterion_score
        + 0.18 * (entailed_required / max(len(required), 1))
        + 0.10 * evidence_coverage,
    )
    return {
        "paper_id": paper.get("openalex_id") or paper.get("doi") or paper.get("title"),
        "criteria": judged,
        "hard_gate": hard_gate,
        "required_entailed_ratio": round(entailed_required / max(len(required), 1), 4),
        "required_unknown_ratio": round(len(unknown) / max(len(required), 1), 4),
        "evidence_coverage": round(evidence_coverage, 4),
        "criterion_score": round(criterion_score, 6),
        "cross_encoder_score": None,
        "pasa_selector_score": None,
        "trace_score": float(paper.get("final_score") or 0),
        "disagreement": 0.0,
        "raw_probability": round(raw_probability, 6),
        "calibrated_probability": round(raw_probability, 6),
        "decision": "reject" if hard_gate == "fail" else "recover_evidence" if hard_gate == "unknown" else "accept",
    }


def score_candidates(
    papers: list[dict[str, Any]], constraint_graph: dict[str, Any], config: ACEConfig = ACEConfig()
) -> list[dict[str, Any]]:
    return [judge_paper(paper, constraint_graph, config) for paper in papers]
