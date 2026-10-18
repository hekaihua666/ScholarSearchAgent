from backend.selection.recall_constrained import select_core_rescue


def paper(title, probability, llm=0.0, cross=0.0, ace=0.0, gate="pass"):
    return {
        "title": title,
        "probability": probability,
        "llm_relevance_score": llm,
        "cross_encoder_probability": cross,
        "ace_probability": ace,
        "hard_gate": gate,
    }


def test_core_rescue_deduplicates_and_obeys_hard_gate() -> None:
    papers = [
        paper("A", 0.95, 0.9, 0.9, 0.9),
        paper("A", 0.92, 0.9, 0.9, 0.9),
        paper("B", 0.85, 0.9, 0.9, 0.9, gate="fail"),
        paper("C", 0.62, 0.8, 0.8, 0.3),
        paper("D", 0.3),
    ]
    selected = select_core_rescue(
        papers, lambda item: item["probability"], 0.8, 0.55, min_results=2, max_results=4
    )
    assert [item["title"] for item in selected] == ["A", "C"]
    assert selected[1]["set_decision"] == "rescue"


def test_minimum_cardinality_rescue_never_adds_failed_gate() -> None:
    papers = [paper("A", 0.2, gate="fail"), paper("B", 0.1), paper("C", 0.05)]
    selected = select_core_rescue(papers, lambda item: item["probability"], 0.9, 0.8, 2, 3)
    assert [item["title"] for item in selected] == ["B", "C"]
