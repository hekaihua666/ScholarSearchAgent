"""Coverage-Adaptive Contract Verification (COVR) for ECRank.

The first ECRank pass is deliberately compact. COVR audits the residual
source-ranked retrieval mass only when that pass cannot safely promote a new
paper. Two incumbent anchors share the verification call with at most eight
overflow papers, allowing point-view scores to be aligned to the frozen
listwise score scale without benchmark labels.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from typing import Any, Iterable, Sequence

from backend.llm_client import _chat
from backend.paper_identity import match_identity, normalize_title
from backend.selection.evidence_contract_ranker_v9 import (
    ContractCard,
    _bounded,
    _float,
    source_rank_prior,
)


MAX_ANCHORS = 2
MAX_OVERFLOW = 8
MAX_VERIFY_CARDS = MAX_ANCHORS + MAX_OVERFLOW
FORBIDDEN_MARKERS = ("gold", "label", "query_id", "expected_cardinality", "matched_gold")


def identity_key(paper: dict[str, Any]) -> str:
    doi = str(paper.get("doi") or "").lower().removeprefix("https://doi.org/").strip()
    if doi:
        return f"doi:{doi}"
    arxiv = str(paper.get("arxiv_id") or "").lower().strip()
    if arxiv:
        return f"arxiv:{arxiv}"
    openalex = str(paper.get("openalex_id") or "").lower().strip()
    if openalex:
        return f"openalex:{openalex}"
    return f"title:{normalize_title(paper.get('title'))}"


def _dedupe(papers: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for paper in papers:
        if not paper.get("title"):
            continue
        if any(match_identity(paper, old) or match_identity(old, paper) for old in output):
            continue
        output.append(paper)
    return output


def select_source_overflow(
    newcomers: Sequence[dict[str, Any]],
    baseline_top20: Sequence[dict[str, Any]],
    initial_cards: Sequence[ContractCard],
    limit: int = MAX_OVERFLOW,
) -> list[dict[str, Any]]:
    """Select the unobserved head of the source view, not another TRACE rerank.

    The initial round-robin shortlist consumes only the first few entries of
    each coarse view. Selecting source-ranked residuals therefore increases
    coverage rather than repeatedly judging the same TRACE-heavy candidates.
    """
    excluded = [*baseline_top20, *(card.paper for card in initial_cards)]
    legal = [
        paper for paper in _dedupe(newcomers)
        if not any(match_identity(paper, old) or match_identity(old, paper) for old in excluded)
    ]
    legal.sort(key=lambda paper: (
        -source_rank_prior(paper),
        -_float(paper.get("final_score")),
        normalize_title(paper.get("title")),
    ))
    return legal[:limit]


def build_verification_cards(
    query: str,
    baseline_top20: Sequence[dict[str, Any]],
    newcomers: Sequence[dict[str, Any]],
    initial_cards: Sequence[ContractCard],
) -> list[ContractCard]:
    by_rank = sorted(
        (card for card in initial_cards if card.origin == "baseline" and card.original_rank),
        key=lambda card: int(card.original_rank or 99),
    )
    anchors = by_rank[:MAX_ANCHORS]
    if len(anchors) < MAX_ANCHORS:
        anchors = [
            ContractCard("", "anchor", rank, paper)
            for rank, paper in enumerate(baseline_top20[:MAX_ANCHORS], 1)
        ]
    else:
        anchors = [ContractCard("", "anchor", card.original_rank, card.paper) for card in anchors]
    overflow = [
        ContractCard("", "overflow", None, paper)
        for paper in select_source_overflow(newcomers, baseline_top20, initial_cards)
    ]
    cards = [*anchors, *overflow]
    ordered = sorted(cards, key=lambda card: hashlib.sha256(
        f"{query}\0covr\0{identity_key(card.paper)}".encode("utf-8")
    ).hexdigest())
    return [
        ContractCard(f"V{index:02d}", card.origin, card.original_rank, card.paper)
        for index, card in enumerate(ordered, 1)
    ]


def should_verify_overflow(
    initial_scored: Sequence[dict[str, Any]],
    initial_actions: Sequence[dict[str, Any]],
    cards: Sequence[ContractCard],
    *,
    threshold: float,
) -> bool:
    if initial_actions or not any(card.origin == "overflow" for card in cards):
        return False
    already_resolved = any(
        row.get("origin") == "newcomer"
        and row.get("evidence_complete")
        and _float(row.get("contract_score")) >= threshold
        for row in initial_scored
    )
    return not already_resolved


def build_verification_prompt(
    query: str,
    contract: dict[str, Any],
    cards: Sequence[ContractCard],
) -> str:
    payload = [card.prompt_payload() for card in cards]
    safe_contract = {
        "conditions": (contract.get("conditions") or [])[:5],
        "relation": str(contract.get("relation") or "")[:300],
        "exclusions": (contract.get("exclusions") or [])[:8],
    }
    return (
        "You are the overflow verifier of a scientific-paper retrieval system.\n"
        "The query contract below is frozen. Do not rewrite, merge, weaken, or add conditions.\n"
        "Judge every paper independently from title and abstract. Recognize established scientific aliases, "
        "but do not infer unsupported contributions. General topical similarity is not enough.\n"
        "For every contract condition and the requested relation use support 0=absent/contradicted, "
        "1=weak, 2=credible implicit support, 3=explicit support.\n"
        "Return valid JSON only with exactly this shape:\n"
        '{"assessments":{"V01":{"point_score":0,"verdict":"relevant|partial|irrelevant",'
        '"condition_support":{"C1":0},"relation_support":0,"contradiction":false,"evidence":"..."}}}\n'
        "Every candidate id must occur exactly once. point_score is an integer 0-100.\n"
        f"QUERY: {query}\n"
        f"FROZEN_CONTRACT: {json.dumps(safe_contract, ensure_ascii=False, separators=(',', ':'))}\n"
        f"CANDIDATES: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )


def assert_prompt_safe(prompt: str) -> None:
    prefix = prompt.split("QUERY:", 1)[0].lower()
    if any(marker in prefix for marker in FORBIDDEN_MARKERS):
        raise AssertionError("benchmark-derived marker in COVR prompt")


def _strip_json(text: str) -> str:
    value = str(text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.I)
        value = re.sub(r"\s*```$", "", value)
    start, end = value.find("{"), value.rfind("}")
    return value[start:end + 1] if start >= 0 and end > start else value


def parse_verification_response(text: str, card_ids: Sequence[str]) -> dict[str, Any]:
    raw = json.loads(_strip_json(text))
    rows = raw.get("assessments") or {}
    if isinstance(rows, list):
        rows = {str(row.get("id")): row for row in rows if isinstance(row, dict)}
    output: dict[str, dict[str, Any]] = {}
    for card_id in card_ids:
        row = rows.get(card_id) or {}
        support = row.get("condition_support") or {}
        if isinstance(support, list):
            support = {f"C{index + 1}": value for index, value in enumerate(support)}
        output[card_id] = {
            "point_score": _bounded(row.get("point_score", row.get("score"))),
            "verdict": str(row.get("verdict") or "irrelevant").lower(),
            "condition_support": {
                str(key): _bounded(value, 0, 3) for key, value in support.items()
            },
            "relation_support": _bounded(row.get("relation_support"), 0, 3),
            "contradiction": bool(row.get("contradiction")),
            "evidence": str(row.get("evidence") or "")[:500],
        }
    return {"assessments": output}


async def request_overflow_verification(
    query: str,
    contract: dict[str, Any],
    cards: Sequence[ContractCard],
) -> dict[str, Any]:
    prompt = build_verification_prompt(query, contract, cards)
    assert_prompt_safe(prompt)
    content, tokens = await _chat(
        [{"role": "user", "content": prompt}], max_tokens=2200, temperature=0.0
    )
    return {
        "parsed": parse_verification_response(content, [card.card_id for card in cards]),
        "raw_content": content,
        "tokens": tokens,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    }


def verification_score(assessment: dict[str, Any]) -> tuple[float, bool]:
    supports = list((assessment.get("condition_support") or {}).values())
    mean_support = sum(map(_float, supports)) / len(supports) / 3.0 if supports else 0.0
    minimum_support = min(map(_float, supports)) / 3.0 if supports else 0.0
    relation = _float(assessment.get("relation_support")) / 3.0
    point = _bounded(assessment.get("point_score")) / 100.0
    score = 100.0 * (0.45 * point + 0.25 * mean_support + 0.20 * minimum_support + 0.10 * relation)
    if assessment.get("contradiction"):
        score *= 0.25
    complete = (
        str(assessment.get("verdict")) == "relevant"
        and bool(supports)
        and minimum_support >= 2 / 3
        and relation >= 2 / 3
        and not assessment.get("contradiction")
    )
    return score, complete


def score_verification_cards(
    cards: Sequence[ContractCard],
    parsed: dict[str, Any],
    initial_scored: Sequence[dict[str, Any]],
    *,
    calibrate_with_anchors: bool = True,
) -> tuple[list[dict[str, Any]], float]:
    assessments = parsed.get("assessments") or {}
    initial_by_key = {identity_key(row["paper"]): row for row in initial_scored}
    raw_by_id: dict[str, tuple[float, bool]] = {
        card.card_id: verification_score(assessments.get(card.card_id) or {}) for card in cards
    }
    offsets: list[float] = []
    for card in cards:
        if card.origin != "anchor":
            continue
        initial = initial_by_key.get(identity_key(card.paper))
        if initial:
            offsets.append(_float(initial.get("contract_score")) - raw_by_id[card.card_id][0])
    # A small clip corrects prompt-scale drift without allowing two anchors to
    # dominate a genuinely explicit overflow match.
    offset = max(-5.0, min(5.0, statistics.median(offsets))) if offsets and calibrate_with_anchors else 0.0
    output: list[dict[str, Any]] = []
    for card in cards:
        if card.origin != "overflow":
            continue
        raw_score, complete = raw_by_id[card.card_id]
        adjusted = round(_bounded(raw_score + offset), 6)
        assessment = dict(assessments.get(card.card_id) or {})
        # The routing interface expects point/list fields. Repeating the
        # anchor-aligned point value does not fabricate a second observation;
        # it merely prevents the legacy distributive router from discarding it.
        assessment["point_score"] = adjusted
        assessment["list_score"] = adjusted
        output.append({
            "card_id": card.card_id,
            "origin": "newcomer",
            "original_rank": None,
            "paper": card.paper,
            "contract_rank": len(initial_scored) + len(output) + 1,
            "contract_score": adjusted,
            "raw_verification_score": raw_score,
            "anchor_offset": offset,
            "evidence_complete": complete,
            "assessment": assessment,
            "verification_stage": "source_overflow",
        })
    return sorted(output, key=lambda row: (-row["contract_score"], row["card_id"])), offset


def assert_no_forbidden_state(cards: Sequence[ContractCard]) -> None:
    for card in cards:
        for key in card.paper:
            if any(marker in str(key).lower() for marker in FORBIDDEN_MARKERS):
                raise AssertionError(f"forbidden field carried into COVR card: {key}")
