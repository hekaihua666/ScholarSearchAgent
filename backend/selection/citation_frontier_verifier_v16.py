"""Contract verifier for the low-precision SCGQ citation frontier."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Sequence

from backend.llm_client import _chat
from backend.paper_identity import match_identity, normalize_title
from backend.selection.coverage_adaptive_contract_v10 import identity_key
from backend.selection.evidence_contract_ranker_v9 import ContractCard, _float


MAX_ANCHORS = 2
MAX_FRONTIER_CARDS = 16
MAX_COARSE_TITLES = 100


def select_frontier_candidates(
    frontier: Sequence[dict[str, Any]],
    existing_pool: Sequence[dict[str, Any]],
    limit: int = MAX_FRONTIER_CARDS,
) -> list[dict[str, Any]]:
    legal: list[dict[str, Any]] = []
    for paper in frontier:
        if not paper.get("title"):
            continue
        if any(match_identity(paper, old) or match_identity(old, paper) for old in existing_pool):
            continue
        if any(match_identity(paper, old) or match_identity(old, paper) for old in legal):
            continue
        legal.append(paper)

    def source_rank(paper: dict[str, Any]) -> int:
        try:
            return int(paper.get("_openalex_rank") or 999)
        except (TypeError, ValueError):
            return 999

    views = (
        sorted(legal, key=lambda paper: (-_float(paper.get("final_score")), source_rank(paper), normalize_title(paper.get("title")))),
        sorted(legal, key=lambda paper: (-int(paper.get("citation_seed_count") or 0), -_float(paper.get("constraint_coverage_score")), source_rank(paper))),
        sorted(legal, key=lambda paper: (source_rank(paper), -_float(paper.get("final_score")), normalize_title(paper.get("title")))),
        sorted(legal, key=lambda paper: (-int(paper.get("cited_by_count") or 0), -_float(paper.get("final_score")), normalize_title(paper.get("title")))),
    )
    selected: list[dict[str, Any]] = []
    for depth in range(max((len(view) for view in views), default=0)):
        for view in views:
            if depth >= len(view):
                continue
            paper = view[depth]
            if paper not in selected:
                selected.append(paper)
            if len(selected) >= limit:
                return selected
    return selected


def select_coarse_title_pool(
    frontier: Sequence[dict[str, Any]], existing_pool: Sequence[dict[str, Any]],
    limit: int = MAX_COARSE_TITLES,
) -> list[dict[str, Any]]:
    legal: list[dict[str, Any]] = []
    for paper in frontier:
        if not paper.get("title"):
            continue
        if any(match_identity(paper, old) or match_identity(old, paper) for old in existing_pool):
            continue
        if any(match_identity(paper, old) or match_identity(old, paper) for old in legal):
            continue
        legal.append(paper)

    def source_rank(paper: dict[str, Any]) -> int:
        try:
            return int(paper.get("_openalex_rank") or 999)
        except (TypeError, ValueError):
            return 999

    score_view = sorted(legal, key=lambda paper: (
        -_float(paper.get("final_score")), source_rank(paper), normalize_title(paper.get("title"))
    ))[:60]
    citation_view = sorted(legal, key=lambda paper: (
        -int(paper.get("citation_seed_count") or 0),
        -_float(paper.get("constraint_coverage_score")),
        source_rank(paper),
    ))[:40]
    source_view = sorted(legal, key=lambda paper: (
        source_rank(paper), -_float(paper.get("final_score"))
    ))[:20]
    output: list[dict[str, Any]] = []
    for paper in [*score_view, *citation_view, *source_view, *legal]:
        if paper not in output:
            output.append(paper)
        if len(output) >= limit:
            break
    return output


def build_title_cards(
    query: str, frontier: Sequence[dict[str, Any]], existing_pool: Sequence[dict[str, Any]],
) -> list[ContractCard]:
    papers = select_coarse_title_pool(frontier, existing_pool)
    ordered = sorted(papers, key=lambda paper: hashlib.sha256(
        f"{query}\0title-funnel\0{identity_key(paper)}".encode("utf-8")
    ).hexdigest())
    return [ContractCard(f"T{index:03d}", "overflow", None, paper) for index, paper in enumerate(ordered, 1)]


def build_title_screen_prompt(
    query: str, contract: dict[str, Any], cards: Sequence[ContractCard], limit: int = MAX_FRONTIER_CARDS,
) -> str:
    payload = [{
        "id": card.card_id,
        "title": str(card.paper.get("title") or "")[:300],
        "year": card.paper.get("year"),
        "citation_seed_count": int(card.paper.get("citation_seed_count") or 0),
        "citation_direction": card.paper.get("citation_direction"),
    } for card in cards]
    safe_contract = {
        "conditions": (contract.get("conditions") or [])[:5],
        "relation": str(contract.get("relation") or "")[:300],
        "exclusions": (contract.get("exclusions") or [])[:8],
    }
    return (
        "You are the title funnel of a scientific citation-frontier search system.\n"
        "Select at most %d papers whose titles are most likely to satisfy the query contract. "
        "A title may use an established alias or be a representative method inside a requested taxonomy. "
        "Prefer plausible recall at this stage; abstracts will be verified later. Do not select merely topical titles.\n"
        "Return valid JSON only: {\"selected\":[{\"id\":\"T001\",\"score\":0,\"reason\":\"...\"}]}\n"
        "Scores are integers 0-100. Use only candidate IDs and do not return more than %d.\n"
        "QUERY: %s\nCONTRACT: %s\nCANDIDATE_TITLES: %s"
    ) % (
        limit, limit, query,
        json.dumps(safe_contract, ensure_ascii=False, separators=(",", ":")),
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    )


def _strip_json(text: str) -> str:
    value = str(text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.I)
        value = re.sub(r"\s*```$", "", value)
    start, end = value.find("{"), value.rfind("}")
    return value[start:end + 1] if start >= 0 and end > start else value


def parse_title_screen(text: str, card_ids: Sequence[str], limit: int = MAX_FRONTIER_CARDS) -> dict[str, Any]:
    raw = json.loads(_strip_json(text))
    valid = set(card_ids); selected = []
    for row in raw.get("selected") or []:
        if isinstance(row, str):
            row = {"id": row}
        if not isinstance(row, dict):
            continue
        card_id = str(row.get("id") or "")
        if card_id not in valid or any(old["id"] == card_id for old in selected):
            continue
        selected.append({
            "id": card_id,
            "score": min(100.0, max(0.0, _float(row.get("score")))),
            "reason": str(row.get("reason") or "")[:300],
        })
        if len(selected) >= limit:
            break
    return {"selected": selected}


async def request_title_screen(
    query: str, contract: dict[str, Any], cards: Sequence[ContractCard],
) -> dict[str, Any]:
    prompt = build_title_screen_prompt(query, contract, cards)
    content, tokens = await _chat([{"role": "user", "content": prompt}], max_tokens=1800, temperature=0.0)
    return {
        "parsed": parse_title_screen(content, [card.card_id for card in cards]),
        "raw_content": content,
        "tokens": tokens,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    }


def cards_from_title_screen(
    query: str,
    selected_core: Sequence[dict[str, Any]],
    title_cards: Sequence[ContractCard],
    parsed: dict[str, Any],
) -> list[ContractCard]:
    by_id = {card.card_id: card.paper for card in title_cards}
    candidates = [
        by_id[row["id"]] for row in (parsed.get("selected") or []) if row.get("id") in by_id
    ][:MAX_FRONTIER_CARDS]
    anchors = [ContractCard("", "anchor", rank, paper) for rank, paper in enumerate(selected_core[:2], 1)]
    overflow = [ContractCard("", "overflow", None, paper) for paper in candidates]
    cards = [*anchors, *overflow]
    ordered = sorted(cards, key=lambda card: hashlib.sha256(
        f"{query}\0evidence-funnel\0{identity_key(card.paper)}".encode("utf-8")
    ).hexdigest())
    return [ContractCard(f"E{index:02d}", card.origin, card.original_rank, card.paper) for index, card in enumerate(ordered, 1)]


def build_frontier_verification_cards(
    query: str,
    selected_core: Sequence[dict[str, Any]],
    frontier: Sequence[dict[str, Any]],
    existing_pool: Sequence[dict[str, Any]],
) -> list[ContractCard]:
    anchors = [
        ContractCard("", "anchor", rank, paper)
        for rank, paper in enumerate(selected_core[:MAX_ANCHORS], 1)
    ]
    candidates = [
        ContractCard("", "overflow", None, paper)
        for paper in select_frontier_candidates(frontier, existing_pool)
    ]
    cards = [*anchors, *candidates]
    ordered = sorted(cards, key=lambda card: hashlib.sha256(
        f"{query}\0citation-frontier\0{identity_key(card.paper)}".encode("utf-8")
    ).hexdigest())
    return [
        ContractCard(f"C{index:02d}", card.origin, card.original_rank, card.paper)
        for index, card in enumerate(ordered, 1)
    ]


def should_verify_frontier(
    cards: Sequence[ContractCard], prior_actions: int, contract: dict[str, Any],
) -> bool:
    return (
        prior_actions == 0
        and bool(contract.get("conditions"))
        and any(card.origin == "overflow" for card in cards)
        and sum(card.origin == "anchor" for card in cards) == MAX_ANCHORS
    )
