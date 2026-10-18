from backend.adaptive_controller import assess_retrieval


def test_empty_results_trigger_query_evolution():
    assert assess_retrieval("graph retrieval", []).next_action == "query_evolution"


def test_citation_seed_triggers_graph_expansion_when_evidence_is_weak():
    papers = [{"title": "Graph retrieval", "final_score": .35, "constraint_coverage_score": .2,
               "source_api": "OpenAlex", "metadata_score": .7, "retrieval_hits": [{}],
               "referenced_works": ["https://openalex.org/W1"]}]
    assert assess_retrieval("graph retrieval", papers).next_action == "citation_expansion"


def test_budget_is_hard_stop():
    papers = [{"title": "Relevant", "final_score": .2, "constraint_coverage_score": .1,
               "source_api": "x", "metadata_score": .5, "retrieval_hits": []}]
    assert assess_retrieval("relevant", papers, api_calls=12).next_action == "stop_budget"
