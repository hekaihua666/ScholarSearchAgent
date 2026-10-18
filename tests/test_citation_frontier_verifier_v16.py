from backend.selection.citation_frontier_verifier_v16 import (
    MAX_FRONTIER_CARDS,
    build_title_cards,
    build_title_screen_prompt,
    cards_from_title_screen,
    build_frontier_verification_cards,
    parse_title_screen,
    select_coarse_title_pool,
    select_frontier_candidates,
    should_verify_frontier,
)


def paper(title: str, wid: int, score: float, source_rank: int, seed_count: int = 1):
    return {
        "title": title,
        "openalex_id": f"https://openalex.org/W{wid}",
        "final_score": score,
        "constraint_coverage_score": score,
        "_openalex_rank": source_rank,
        "citation_seed_count": seed_count,
        "cited_by_count": wid,
        "abstract": f"Evidence for {title}",
    }


def test_frontier_shortlist_is_bounded_multiview_and_excludes_existing():
    existing = [paper("existing paper", 1, .9, 1)]
    frontier = [existing[0], *[
        paper(f"candidate {chr(97+i//26)}{chr(97+i%26)}", i + 10, i / 50, 40 - i, 2 if i == 20 else 1)
        for i in range(30)
    ]]
    selected = select_frontier_candidates(frontier, existing)
    assert len(selected) == MAX_FRONTIER_CARDS
    assert all(row["title"] != "existing paper" for row in selected)
    assert any(row["citation_seed_count"] == 2 for row in selected)


def test_cards_are_opaque_and_keep_two_anchors():
    core = [paper("anchor alpha", 1, .8, 1), paper("anchor beta", 2, .7, 2)]
    cards = build_frontier_verification_cards(
        "query", core, [paper("new candidate", 3, .9, 3)], core
    )
    assert sum(card.origin == "anchor" for card in cards) == 2
    assert sum(card.origin == "overflow" for card in cards) == 1
    assert all(card.card_id.startswith("C") for card in cards)
    assert all("origin" not in card.prompt_payload() for card in cards)


def test_trigger_requires_contract_and_no_prior_action():
    core = [paper("a", 1, .8, 1), paper("b", 2, .7, 2)]
    cards = build_frontier_verification_cards("query", core, [paper("c", 3, .9, 3)], core)
    contract = {"conditions": [{"id": "C1"}]}
    assert should_verify_frontier(cards, 0, contract)
    assert not should_verify_frontier(cards, 1, contract)
    assert not should_verify_frontier(cards, 0, {})


def test_coarse_title_funnel_is_bounded_and_parse_selects_valid_ids():
    existing = [paper("existing", 1, .9, 1)]
    frontier = [paper(f"frontier {chr(97+i//26)}{chr(97+i%26)}", i + 10, i / 120, 100 - i, 2) for i in range(120)]
    pool = select_coarse_title_pool(frontier, existing)
    assert len(pool) == 100
    cards = build_title_cards("query", frontier, existing)
    prompt = build_title_screen_prompt("query", {"conditions": [{"id": "C1"}]}, cards)
    assert "CANDIDATE_TITLES" in prompt and '"abstract":' not in prompt.lower()
    parsed = parse_title_screen('{"selected":[{"id":"T001","score":99},{"id":"BAD","score":100}]}', [c.card_id for c in cards])
    assert parsed["selected"] == [{"id": "T001", "score": 99.0, "reason": ""}]


def test_second_stage_cards_follow_title_selection():
    core = [paper("anchor one", 1, .9, 1), paper("anchor two", 2, .8, 2)]
    title_cards = build_title_cards("query", [paper("candidate alpha", 3, .7, 3), paper("candidate beta", 4, .6, 4)], core)
    selected = {"selected": [{"id": title_cards[0].card_id, "score": 90, "reason": "likely"}]}
    cards = cards_from_title_screen("query", core, title_cards, selected)
    assert sum(card.origin == "anchor" for card in cards) == 2
    assert sum(card.origin == "overflow" for card in cards) == 1
