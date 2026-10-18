from datetime import date

from backend.retrieval.local_hybrid import LocalHybridIndex


def test_local_index_cutoff_and_identity_dedup(tmp_path) -> None:
    index = LocalHybridIndex(tmp_path / "papers.db")
    try:
        inserted = index.add([
            {"title": "Graph Attention for Anomaly Detection", "abstract": "time series anomaly", "arxiv_id": "2009.02040", "year": 2020, "publication_date": "2020-09-04"},
            {"title": "Graph Attention for Anomaly Detection", "abstract": "duplicate", "arxiv_id": "2009.02040v2", "year": 2021, "publication_date": "2021-01-01"},
            {"title": "Future Graph Anomaly Method", "abstract": "graph anomaly", "arxiv_id": "2501.00001", "year": 2025, "publication_date": "2025-01-01"},
        ])
        assert inserted == 2
        results = index.search("graph anomaly", date(2023, 12, 31))
        assert [paper["arxiv_id"] for paper in results] == ["2009.02040"]
    finally:
        index.close()
