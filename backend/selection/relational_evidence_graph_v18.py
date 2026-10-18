"""Relational Evidence Graph scoring for citation-frontier candidates.

REG treats a complex academic query as a conjunction of condition nodes plus
one relational predicate.  A candidate is scored by condition coverage,
predicate evidence, citation-direction compatibility, graph support and the
existing relevance priors.  The explicit minimum-condition term prevents a
paper that matches only one half of a compositional query from receiving a
high score solely because it is topically similar.
"""

from __future__ import annotations

import copy
import math
import re
from typing import Any, Mapping, Sequence

from backend.paper_identity import match_identity, normalize_title
from backend.selection.evidence_contract_ranker_v9 import _float
from backend.selection.local_graph_reranker_v17 import (
    _deduplicate,
    _identity_keys,
    _source_rank,
)


TOKEN_RE = re.compile(r"[a-z][a-z0-9-]*", re.I)
STOPWORDS = {
    "a", "an", "and", "are", "as", "be", "both", "by", "during", "for",
    "from", "in", "into", "is", "it", "method", "model", "must", "of",
    "on", "only", "or", "paper", "related", "specific", "study", "such",
    "that", "the", "this", "to", "using", "which", "with", "work",
}

RELATION_FAMILIES = {
    "proposal": ("propos", "introduc", "present", "novel", "first", "pioneer", "origina"),
    "analysis": ("analy", "theor", "investigat", "understand", "evaluat", "assess", "study"),
    "extension": ("extend", "improv", "generaliz", "enhanc", "advance", "build", "follow"),
    "taxonomy": ("classif", "categor", "taxonomy", "cluster", "group", "type"),
    "usage": ("use", "employ", "apply", "combin", "based", "utiliz"),
    "survey": ("survey", "review", "overview", "systematic", "discuss"),
}


def select_quota_frontier_pool(
    frontier: Sequence[dict[str, Any]],
    incumbents: Sequence[dict[str, Any]],
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Preserve deep candidates from each citation lane with explicit quotas."""

    incumbent_keys = set().union(*(_identity_keys(paper) for paper in incumbents)) if incumbents else set()
    legal = [paper for paper in _deduplicate(frontier) if not (_identity_keys(paper) & incumbent_keys)]
    score_view = sorted(legal, key=lambda paper: (
        -_float(paper.get("final_score")), _source_rank(paper), normalize_title(paper.get("title")),
    ))
    graph_view = sorted(legal, key=lambda paper: (
        -int(paper.get("citation_seed_count") or 0),
        -_float(paper.get("constraint_coverage_score")),
        _source_rank(paper),
    ))
    source_view = sorted(legal, key=lambda paper: (
        _source_rank(paper), -_float(paper.get("final_score")),
    ))
    authority_view = sorted(legal, key=lambda paper: (
        -int(paper.get("cited_by_count") or 0), -_float(paper.get("final_score")),
    ))
    lane_views = []
    for lane in sorted({str(paper.get("citation_lane") or "") for paper in legal}):
        if lane:
            lane_views.append(sorted(
                [paper for paper in legal if str(paper.get("citation_lane") or "") == lane],
                key=lambda paper: (
                    -int(paper.get("citation_seed_count") or 0),
                    _source_rank(paper),
                    -_float(paper.get("final_score")),
                ),
            ))
    ordered = [
        *score_view[:60],
        *graph_view[:50],
        *(paper for view in lane_views for paper in view[:45]),
        *source_view[:25],
        *authority_view[:25],
        *score_view,
    ]
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for paper in ordered:
        keys = _identity_keys(paper)
        if not keys or keys & seen:
            continue
        output.append(copy.deepcopy(paper)); seen.update(keys)
        if len(output) >= limit:
            break
    return output


def _stem(token: str) -> str:
    value = token.lower().replace("-", "")
    for suffix in ("ization", "isation", "ation", "ingly", "edly", "ments", "ment", "ness", "ing", "ied", "ies", "ed", "es", "s"):
        if value.endswith(suffix) and len(value) - len(suffix) >= 4:
            value = value[:-len(suffix)]
            break
    return value


def _tokens(text: Any) -> set[str]:
    raw = str(text or "").lower()
    raw = raw.replace("source-free", "sourcefree").replace("source free", "sourcefree")
    raw = raw.replace("pre-trained", "pretrained").replace("pre trained", "pretrained")
    return {
        _stem(token) for token in TOKEN_RE.findall(raw)
        if len(token) > 1 and token not in STOPWORDS
    }


def _normalized_text(text: Any) -> str:
    return normalize_title(str(text or "")).replace("sourcefree", "sourcefree")


def _semantic_flags(text: str) -> set[str]:
    lowered = text.lower()
    flags: set[str] = set()
    if any(phrase in lowered for phrase in (
        "source-free", "source free", "without access to source data",
        "without source data", "no access to source data", "source data is not available",
    )):
        flags.add("sourcefree")
    if any(phrase in lowered for phrase in ("pre-trained", "pretrained", "pre-training", "pretraining")):
        flags.add("pretrained")
    if any(phrase in lowered for phrase in ("category shift", "label space", "open-set", "open set", "partial-set", "partial set")):
        flags.add("categoryshift")
    if "global descent" in lowered or "gradient descent" in lowered:
        flags.add("gradientdescent")
    return flags


def phrase_coverage(text: str, phrase: str) -> float:
    target = _tokens(phrase) | _semantic_flags(phrase)
    observed = _tokens(text) | _semantic_flags(text)
    if not target:
        return 0.0
    lexical = len(target & observed) / len(target)
    exact = float(bool(_normalized_text(phrase)) and _normalized_text(phrase) in _normalized_text(text))
    return min(1.0, 0.82 * lexical + 0.18 * exact)


def condition_score(paper: dict[str, Any], condition: dict[str, Any]) -> float:
    text = f"{paper.get('title', '')}. {paper.get('abstract', '')}"
    phrases = [str(condition.get("description") or ""), *[str(value) for value in condition.get("aliases") or []]]
    values = [phrase_coverage(text, phrase) for phrase in phrases if phrase]
    return max(values, default=0.0)


def relation_family(relation: str, question: str = "") -> str:
    value = f"{relation} {question}".lower()
    if re.search(r"propos|introduc|first|pioneer|who", value):
        return "proposal"
    if re.search(r"extend|improv|generaliz|enhanc|build upon|follow", value):
        return "extension"
    if re.search(r"analy|theor|investigat|evaluat|assess", value):
        return "analysis"
    if re.search(r"classif|categor|taxonomy|cluster|types", value):
        return "taxonomy"
    if re.search(r"survey|review|overview|discuss", value):
        return "survey"
    if re.search(r"\buse|employ|apply|combin|based", value):
        return "usage"
    return "analysis"


def relation_evidence(paper: dict[str, Any], family: str) -> float:
    title = str(paper.get("title") or "").lower()
    abstract = str(paper.get("abstract") or "").lower()
    cues = RELATION_FAMILIES.get(family, ())
    title_hits = sum(cue in title for cue in cues)
    abstract_hits = sum(cue in abstract for cue in cues)
    return min(1.0, (2.0 * title_hits + abstract_hits) / 3.0)


def direction_compatibility(paper: dict[str, Any], family: str) -> float:
    lane = str(paper.get("citation_lane") or "").lower()
    direction = str(paper.get("citation_direction") or "").lower()
    forward = "forward" in lane or "seed_cites_candidate" in direction
    reverse = "reverse" in lane or "candidate_cites_seed" in direction
    if family == "proposal":
        return 1.0 if forward else 0.30 if reverse else 0.45
    if family in {"extension", "analysis"}:
        return 1.0 if reverse else 0.55 if forward else 0.45
    return 0.70 if (forward or reverse) else 0.45


def seed_bridge_score(paper: dict[str, Any], question: str, contract: dict[str, Any]) -> float:
    query_tokens = _tokens(question)
    for condition in contract.get("conditions") or []:
        query_tokens |= _tokens(condition.get("description"))
        for alias in condition.get("aliases") or []:
            query_tokens |= _tokens(alias)
    if not query_tokens:
        return 0.0
    best = 0.0
    for title in paper.get("citation_seed_titles") or []:
        tokens = _tokens(title)
        if tokens:
            best = max(best, len(tokens & query_tokens) / math.sqrt(len(tokens) * len(query_tokens)))
    return min(1.0, best)


def attach_relational_features(
    papers: Sequence[dict[str, Any]],
    question: str,
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    family = relation_family(str(contract.get("relation") or ""), question)
    output: list[dict[str, Any]] = []
    for paper in papers:
        row = copy.deepcopy(paper)
        conditions = [condition_score(row, condition) for condition in contract.get("conditions") or []]
        if conditions:
            average = sum(conditions) / len(conditions)
            minimum = min(conditions)
            conjunction = math.sqrt(max(0.0, average * minimum))
        else:
            average = minimum = conjunction = _float(row.get("constraint_coverage_score"))
        row["v18_condition_average"] = average
        row["v18_condition_minimum"] = minimum
        row["v18_conjunction"] = conjunction
        row["v18_relation"] = relation_evidence(row, family)
        row["v18_direction"] = direction_compatibility(row, family)
        row["v18_seed_bridge"] = seed_bridge_score(row, question, contract)
        row["v18_relation_family"] = family
        output.append(row)
    return output


def relational_score(
    paper: dict[str, Any],
    weights: Sequence[float] | Mapping[str, Sequence[float]],
) -> float:
    if isinstance(weights, Mapping):
        family = str(paper.get("v18_relation_family") or "default")
        vector = weights.get(family) or weights.get("default")
        if vector is None:
            raise ValueError("adaptive weights require a default vector")
    else:
        vector = weights
    condition, relation, direction, bridge, graph, trace, cross = (float(value) for value in vector)
    return (
        condition * _float(paper.get("v18_conjunction"))
        + relation * _float(paper.get("v18_relation"))
        + direction * _float(paper.get("v18_direction"))
        + bridge * _float(paper.get("v18_seed_bridge"))
        + graph * _float(paper.get("v17_graph_pct"))
        + trace * _float(paper.get("v17_trace_pct"))
        + cross * _float(paper.get("v17_cross_pct"))
    )


def route_relational_top20(
    incumbents: Sequence[dict[str, Any]],
    frontier: Sequence[dict[str, Any]],
    *,
    weights: Sequence[float] | Mapping[str, Sequence[float]],
    margin: float,
    score_floor: float,
    max_promotions: int,
    protect_head: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    current = [copy.deepcopy(paper) for paper in incumbents]
    incoming = sorted(
        (copy.deepcopy(paper) for paper in frontier),
        key=lambda paper: (
            -relational_score(paper, weights),
            -_float(paper.get("v18_condition_minimum")),
            normalize_title(paper.get("title")),
        ),
    )
    actions: list[dict[str, Any]] = []
    for candidate in incoming:
        if len(actions) >= max_promotions:
            break
        incoming_score = relational_score(candidate, weights)
        if incoming_score < score_floor:
            break
        removable = list(enumerate(current[protect_head:], start=protect_head))
        if not removable:
            break
        index, incumbent = min(
            removable,
            key=lambda item: (
                relational_score(item[1], weights),
                _float(item[1].get("v18_condition_minimum")),
                -item[0],
            ),
        )
        incumbent_score = relational_score(incumbent, weights)
        if incoming_score < incumbent_score + margin:
            break
        current[index] = candidate
        actions.append({
            "incoming_title": candidate.get("title"),
            "outgoing_title": incumbent.get("title"),
            "incoming_score": incoming_score,
            "outgoing_score": incumbent_score,
            "margin": incoming_score - incumbent_score,
            "condition_minimum": candidate.get("v18_condition_minimum"),
            "relation": candidate.get("v18_relation"),
            "direction": candidate.get("v18_direction"),
            "family": candidate.get("v18_relation_family"),
        })
    return current, actions


def dual_view_consensus(
    frontier: Sequence[dict[str, Any]],
    strict_weights: Sequence[float],
    graph_weights: Sequence[float],
    *,
    strict_floor: float,
    graph_floor: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Return one candidate only when contract and graph views agree at rank 1."""

    if not frontier:
        return None, {"reason": "empty_frontier"}
    strict = max(frontier, key=lambda paper: relational_score(paper, strict_weights))
    graph = max(frontier, key=lambda paper: relational_score(paper, graph_weights))
    strict_score = relational_score(strict, strict_weights)
    graph_score = relational_score(graph, graph_weights)
    same = bool(match_identity(strict, graph) or match_identity(graph, strict))
    accepted = same and strict_score >= strict_floor and graph_score >= graph_floor
    return (copy.deepcopy(strict) if accepted else None), {
        "reason": "accepted" if accepted else "view_disagreement_or_floor",
        "same_candidate": same,
        "strict_title": strict.get("title"),
        "graph_title": graph.get("title"),
        "strict_score": strict_score,
        "graph_score": graph_score,
    }


def add_consensus_to_final(
    selected: Sequence[dict[str, Any]],
    candidate: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], bool]:
    output = [copy.deepcopy(paper) for paper in selected]
    if candidate is None or any(match_identity(candidate, paper) or match_identity(paper, candidate) for paper in output):
        return output, False
    output.append(copy.deepcopy(candidate))
    return output, True
