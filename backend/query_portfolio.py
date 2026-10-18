"""Constraint graph and compact, differentiated query portfolio for TRACE v4."""

from __future__ import annotations

import re
from typing import Any


_TOKENS = re.compile(r"[A-Za-z][A-Za-z0-9+_.-]*")
_BOILERPLATE = re.compile(
    r"\b(?:can you|could you|please|tell me|find me|show me|are there|is there|"
    r"any|some|papers?|studies|resources|work|works|about|providing information on)\b",
    re.I,
)
_RELATIONS = {
    "introduces": ("introduce", "introduced", "propose", "proposed", "first"),
    "uses": (" use ", " uses ", " using ", "based on", "employ", "utilize"),
    "compares": ("compare", "comparison", "versus", " vs "),
    "evaluates": ("evaluate", "evaluation", "benchmark", "experiment"),
    "surveys": ("survey", "review", "overview"),
    "extends": ("extend", "extension", "follow-up"),
    "replicates": ("replicate", "reproduce", "reproduction"),
    "applies-to": ("applied to", "applies to", "application of"),
}
_TYPE_PATTERNS = {
    "dataset": ("dataset", "benchmark", "corpus"),
    "method": ("method", "model", "architecture", "algorithm", "using", "via", "based on"),
    "task": ("task", "detection", "classification", "generation", "prediction", "retrieval", "rendering"),
    "domain": ("domain", "scene", "medical", "language", "vision", "robot", "reinforcement"),
    "venue": ("conference", "journal", "venue", "published in"),
    "paper_type": ("survey", "review", "benchmark", "theoretical", "empirical"),
}


def _clean(text: str) -> str:
    text = _BOILERPLATE.sub(" ", text or "")
    text = re.sub(r"[^A-Za-z0-9+_.-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip(" .,-")


def _terms(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text or text.lower() in {"not specified", "unknown", "n/a"}:
        return []
    return [part.strip() for part in re.split(r"[,;/]", text) if part.strip()]


def _token_set(text: str) -> set[str]:
    return {token.lower() for token in _TOKENS.findall(text or "") if len(token) > 1}


def _dedupe_drift(candidates: list[dict[str, Any]], limit: int = 6) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    for item in candidates:
        raw_query = str(item.get("query") or "")
        query = re.sub(r"\s+", " ", raw_query).strip() if item.get("form") == "original" else _clean(raw_query)
        if not query:
            continue
        current = _token_set(query)
        max_similarity = 0.0
        for previous in accepted:
            prior = _token_set(previous["query"])
            union = current | prior
            max_similarity = max(max_similarity, len(current & prior) / len(union) if union else 1.0)
        if max_similarity >= 0.84:
            continue
        item = dict(item)
        item["query"] = query
        item["drift_score"] = round(1.0 - max_similarity, 4) if accepted else 0.0
        accepted.append(item)
        if len(accepted) >= limit:
            break
    return accepted


def atomicize_query(query: str, understanding: dict[str, Any]) -> dict[str, Any]:
    """Build a compact constraint graph without using benchmark labels."""
    lowered = f" {query.lower()} "
    criteria: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(description: str, kind: str, required: bool = True, target: str = "abstract") -> None:
        description = _clean(description)
        key = (kind, description.lower())
        if not description or key in seen or len(criteria) >= 8:
            return
        seen.add(key)
        criteria.append(
            {
                "criterion_id": f"c{len(criteria) + 1}",
                "description": description,
                "type": kind,
                "importance": "required" if required else "preferred",
                "verification_target": target,
                "unknown_allowed": kind not in {"time", "venue", "negative"},
            }
        )

    topic = str(understanding.get("research_topic") or "")
    if topic and topic.lower() != "not specified":
        add(topic, "topic", True, "title|abstract")
    for field, kind in (
        ("method_constraints", "method"),
        ("task_constraints", "task"),
        ("domain_constraints", "domain"),
    ):
        for term in _terms(understanding.get(field)):
            add(term, kind, True, "title|abstract")

    relation_constraints: list[str] = []
    for relation, markers in _RELATIONS.items():
        if any(marker in lowered for marker in markers):
            relation_constraints.append(relation)
            add(relation, "relation", True, "title|abstract|snippet")
            break

    time_range = str(understanding.get("time_range") or "")
    metadata_constraints: list[str] = []
    if time_range and time_range != "not specified":
        metadata_constraints.append(time_range)
        add(time_range, "time", True, "metadata")

    negative_constraints = []
    for match in re.finditer(r"\b(?:without|excluding|except|not)\s+([A-Za-z0-9+_.-]+(?:\s+[A-Za-z0-9+_.-]+){0,3})", query, re.I):
        value = _clean(match.group(1))
        if value:
            negative_constraints.append(value)
            add(value, "negative", True, "title|abstract")

    # If the old analyzer collapsed the query to one broad topic, retain one
    # distinctive clause as a preferred criterion rather than inventing labels.
    concise = _clean(query)
    if len(criteria) < 2 and concise and concise.lower() != _clean(topic).lower():
        add(concise, "topic", False, "title|abstract")

    key_entities = sorted(
        {token for token in _TOKENS.findall(query) if token.isupper() or any(char.isdigit() for char in token)},
        key=lambda value: (-len(value), value),
    )[:8]
    return {
        "intent": understanding.get("search_intent") or "broad",
        "criteria": criteria,
        "required_semantic_criteria": [c["description"] for c in criteria if c["importance"] == "required" and c["type"] not in {"time", "negative"}],
        "optional_semantic_criteria": [c["description"] for c in criteria if c["importance"] == "preferred"],
        "metadata_constraints": metadata_constraints,
        "relation_constraints": relation_constraints,
        "negative_constraints": negative_constraints,
        "key_entities": key_entities,
        "synonym_groups": [],
        "uncertain_terms": [],
    }


def build_query_portfolio(query: str, understanding: dict[str, Any], graph: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate at most six source-ready variants, each serving a distinct facet."""
    required = list(graph.get("required_semantic_criteria") or [])
    optional = list(graph.get("optional_semantic_criteria") or [])
    canonical_parts = [
        str(understanding.get("research_topic") or ""),
        *_terms(understanding.get("method_constraints")),
        *_terms(understanding.get("task_constraints")),
        *_terms(understanding.get("domain_constraints")),
    ]
    canonical = " ".join(part for part in canonical_parts if part and part.lower() != "not specified")
    distinctive = sorted(required + optional, key=lambda value: (-len(_token_set(value)), -len(value)))
    rare_pair = " ".join(distinctive[:2])
    title_like = _clean(query)
    pseudo = f"{canonical}. Study satisfying {'; '.join(required[:4])}." if canonical else title_like
    candidates = [
        {"variant_id": "Q0", "query": query, "serves": ["original_intent"], "form": "original"},
        {"variant_id": "Q1", "query": canonical, "serves": ["canonical_entities"], "form": "canonical"},
        {"variant_id": "Q2", "query": " ".join(required[:4]), "serves": ["required_facets"], "form": "conjunction"},
        {"variant_id": "Q3", "query": rare_pair, "serves": ["scarce_facets"], "form": "rare_pair"},
        {"variant_id": "Q4", "query": title_like, "serves": ["title_terms"], "form": "title_like"},
        {"variant_id": "Q5", "query": pseudo, "serves": ["semantic_description"], "form": "pseudo_document"},
    ]
    portfolio = _dedupe_drift(candidates, limit=6)
    # Preserve stable IDs after near-duplicate removal.
    for index, item in enumerate(portfolio):
        item["order"] = index
    return portfolio
