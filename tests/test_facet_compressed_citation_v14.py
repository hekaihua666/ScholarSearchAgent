from backend.retrieval.facet_compressed_citation_v14 import build_facet_compressed_query


def test_compression_uses_structured_topic_method_and_task():
    understanding = {
        "research_topic": "unsupervised segmentation classification",
        "method_constraints": "clustering, invariance, pre-trained models",
        "task_constraints": "unsupervised segmentation",
    }
    query = build_facet_compressed_query("Could you mention some works that classify ...?", understanding, {})
    assert query == "unsupervised segmentation classification clustering invariance pre-trained models unsupervised segmentation"
    assert "Could you" not in query


def test_compression_falls_back_to_contract_aliases():
    contract = {"conditions": [
        {"id": "C1", "description": "target networks", "aliases": ["DQN target network"]},
        {"id": "C2", "description": "deep Q-learning", "aliases": []},
    ]}
    query = build_facet_compressed_query("Which papers?", {}, contract)
    assert query == "DQN target network deep Q-learning"


def test_compression_has_bounded_length_and_no_request_boilerplate():
    query = build_facet_compressed_query(
        "What papers discuss graph learning?", {"research_topic": "graph learning " * 80}, {}, limit=80
    )
    assert len(query) <= 80
