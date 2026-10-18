from __future__ import annotations

import asyncio

import backend.retrieval.evidence_backprojection_v21 as ebp
from backend.retrieval.evidence_backprojection_v21 import (
    annotate_backprojection_candidates,
    attach_backprojection_views,
    rank_backprojection_targets,
    retrieve_evidence_backprojection,
    source_evidence_score,
)


def paper(title: str, work_id: str, score: float, refs=()):
    return {
        "title": title,
        "openalex_id": work_id,
        "referenced_works": list(refs),
        "lexical_score": score,
        "final_score": score,
        "v17_cross_pct": score,
        "v17_trace_pct": score,
        "v17_graph_pct": score,
    }


def test_tied_raw_views_receive_equal_percentiles():
    rows = attach_backprojection_views(
        [paper("A", "W1", .5)],
        [paper("B", "W2", .5), paper("C", "W3", .9)],
    )
    assert rows[0]["ebp_final_pct"] == rows[1]["ebp_final_pct"]
    assert rows[2]["ebp_final_pct"] == 1.0


def test_source_score_is_controlled_by_third_strongest_view():
    row = {
        "v17_cross_pct": 1.0,
        "v17_trace_pct": .9,
        "v17_graph_pct": .8,
        "ebp_lexical_pct": .2,
        "ebp_final_pct": .1,
    }
    assert source_evidence_score(row) == .8 + .05 * (1.0 + .9 + .8) / 3


def test_backprojection_aggregates_support_and_excludes_existing_ids():
    rows = [
        paper("seed A", "W1", .9, ["W9", "W10", "W2"]),
        paper("seed B", "W2", .8, ["https://openalex.org/W9"]),
    ]
    targets = rank_backprojection_targets(rows, [], limit=10)
    assert [target["openalex_id"] for target in targets] == ["W9", "W10"]
    assert targets[0]["support_count"] == 2
    assert targets[0]["support_seed_titles"] == ["seed A", "seed B"]


def test_annotation_retains_explainable_path():
    result = annotate_backprojection_candidates(
        [{"title": "target", "openalex_id": "https://openalex.org/W9"}],
        [{"openalex_id": "W9", "score": .7, "rank": 2,
          "support_count": 3, "support_seed_titles": ["A", "B", "C"]}],
    )[0]
    assert result["citation_lane"] == "evidence_backprojection"
    assert result["citation_backprojection_score"] == .7
    assert result["citation_seed_count"] == 3


def test_async_retrieval_uses_one_batched_call(monkeypatch):
    async def fake_fetch(ids, source_query, limit):
        assert ids == ["W9"]
        assert source_query.startswith("ebp_v21")
        assert limit == 100
        return [{"title": "target", "openalex_id": "W9"}]

    async def direct_call(name, arguments, factory):
        assert name == "ebp_v21_forward_references"
        assert arguments["work_ids"] == ["W9"]
        return await factory()

    monkeypatch.setattr(ebp, "fetch_openalex_works", fake_fetch)
    monkeypatch.setattr(ebp, "frozen_call", direct_call)
    monkeypatch.setattr(ebp, "filter_for_cutoff", lambda rows, source: rows)
    result = asyncio.run(retrieve_evidence_backprojection(
        [paper("seed", "W1", 1.0, ["W9"])], [],
    ))
    assert result["api_calls"] == 1
    assert result["candidates"][0]["title"] == "target"
    assert result["candidates"][0]["citation_seed_titles"] == ["seed"]
