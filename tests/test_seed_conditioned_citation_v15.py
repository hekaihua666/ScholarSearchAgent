from backend.retrieval.seed_conditioned_citation_v15 import build_seed_conditioned_query


def test_seed_consensus_inserts_semantic_between_query_terms():
    understanding = {"research_topic": "unsupervised segmentation classification"}
    seeds = [
        {"title": "Deep Clustering for Unsupervised Learning of Visual Features"},
        {"title": "Unsupervised Semantic Segmentation by Distilling Feature Correspondences"},
        {"title": "Recent Progress in Semantic Image Segmentation"},
        {"title": "Unsupervised Semantic Segmentation with Transformers"},
    ]
    assert build_seed_conditioned_query("long request", understanding, seeds) == "unsupervised semantic segmentation"


def test_seed_query_keeps_target_network_topic_compact():
    understanding = {"research_topic": "Target networks in Deep Q-learning"}
    seeds = [
        {"title": "DeepMellow Removing the Need for a Target Network in Deep Q-Learning"},
        {"title": "Beyond Target Networks Improving Deep Q-learning"},
    ]
    value = build_seed_conditioned_query("request", understanding, seeds)
    assert "target" in value and "q-learning" in value and len(value.split()) <= 6


def test_seed_query_falls_back_without_structured_topic():
    value = build_seed_conditioned_query(
        "Which papers discuss face reenactment expression transfer?", {}, []
    )
    assert value == "face reenactment expression transfer"
