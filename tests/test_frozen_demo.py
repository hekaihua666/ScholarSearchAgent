import asyncio

from backend.main import SearchRequest, search


def test_frozen_demo_requires_no_network(monkeypatch):
    monkeypatch.setenv("SCHOLAR_FROZEN_DEMO", "1")
    response = asyncio.run(search(SearchRequest(query="离线演示查询", mode="balanced")))
    assert response.query == "离线演示查询"
    assert response.metrics["frozen_demo"] is True
    assert response.search_process["frozen_demo"] is True
    assert response.papers
