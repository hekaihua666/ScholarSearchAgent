from __future__ import annotations

import json

from backend.selection.coverage_adaptive_contract_v10 import (
    MAX_VERIFY_CARDS,
    build_verification_cards,
    build_verification_prompt,
    parse_verification_response,
    score_verification_cards,
    select_source_overflow,
    should_verify_overflow,
)
from backend.selection.evidence_contract_ranker_v9 import build_contract_cards


def paper(title: str, rank: int, score: float = .3):
    return {
        "title": title,
        "abstract": f"Evidence for {title}",
        "openalex_rank": rank,
        "openalex_id": f"https://openalex.org/{title.replace(' ', '-')}",
        "doi": f"10.9999/{title.replace(' ', '-').lower()}",
        "final_score": score,
        "facet_coverage": .4,
        "topic_coverage": .4,
        "abstract_coverage": .4,
        "constraint_coverage_score": .4,
    }


def alpha_name(prefix: str, index: int) -> str:
    return f"{prefix} {chr(97 + index // 26)}{chr(97 + index % 26)}"


def test_source_overflow_excludes_baseline_and_first_shortlist():
    baseline = [paper(alpha_name("base", i), i) for i in range(1, 21)]
    newcomers = [paper(alpha_name("new", i), i, i / 100) for i in range(1, 31)]
    initial = build_contract_cards("query", baseline, newcomers)
    overflow = select_source_overflow(newcomers, baseline, initial)
    excluded = {card.paper["title"] for card in initial}
    assert len(overflow) == 8
    assert not ({row["title"] for row in overflow} & excluded)
    assert [row["openalex_rank"] for row in overflow] == sorted(row["openalex_rank"] for row in overflow)


def test_verification_cards_are_bounded_and_opaque():
    baseline = [paper(alpha_name("base", i), i) for i in range(1, 21)]
    newcomers = [paper(alpha_name("new", i), i) for i in range(1, 31)]
    initial = build_contract_cards("query", baseline, newcomers)
    cards = build_verification_cards("query", baseline, newcomers, initial)
    assert len(cards) <= MAX_VERIFY_CARDS
    assert sum(card.origin == "anchor" for card in cards) == 2
    assert sum(card.origin == "overflow" for card in cards) == 8
    assert all(card.card_id.startswith("V") for card in cards)
    assert all("origin" not in card.prompt_payload() for card in cards)


def test_trigger_only_when_first_pass_unresolved():
    cards = [type("Card", (), {"origin": "overflow"})()]
    unresolved = [{"origin": "newcomer", "evidence_complete": False, "contract_score": 99}]
    resolved = [{"origin": "newcomer", "evidence_complete": True, "contract_score": 96}]
    assert should_verify_overflow(unresolved, [], cards, threshold=95)
    assert not should_verify_overflow(resolved, [], cards, threshold=95)
    assert not should_verify_overflow(unresolved, [{"incoming_id": "P01"}], cards, threshold=95)


def test_prompt_reuses_frozen_contract_and_has_no_benchmark_fields():
    cards = build_verification_cards(
        "query", [paper("base 1", 1), paper("base 2", 2)], [paper("new 1", 3)], []
    )
    prompt = build_verification_prompt(
        "query", {"conditions": [{"id": "C1", "description": "method"}], "relation": "introduced"}, cards
    )
    lowered = prompt.lower()
    assert "frozen_contract" in lowered and "do not rewrite" in lowered
    assert "query_id" not in lowered and "matched_gold" not in lowered


def test_parser_and_anchor_calibration_are_deterministic():
    anchor = paper("anchor", 1)
    candidate = paper("candidate", 11)
    cards = [
        type("Card", (), {"card_id": "V01", "origin": "anchor", "paper": anchor})(),
        type("Card", (), {"card_id": "V02", "origin": "overflow", "paper": candidate})(),
    ]
    response = json.dumps({"assessments": {
        "V01": {"point_score": 60, "verdict": "partial", "condition_support": {"C1": 1}, "relation_support": 1},
        "V02": {"point_score": 100, "verdict": "relevant", "condition_support": {"C1": 3}, "relation_support": 3},
    }})
    parsed = parse_verification_response(response, ["V01", "V02"])
    initial = [{"paper": anchor, "contract_score": 50}]
    scored, offset = score_verification_cards(cards, parsed, initial)
    assert round(offset, 4) == 4.6667
    assert scored[0]["contract_score"] == 100
    assert scored[0]["evidence_complete"]
