"""Opt-in, context-local retrieval snapshots for evaluation audits.

The collector is inert during normal API operation and does not alter ranking.
"""

from __future__ import annotations

import time
from contextvars import ContextVar
from typing import Any


_AUDIT: ContextVar[dict[str, Any] | None] = ContextVar("retrieval_audit", default=None)


def start_audit(query: str) -> object:
    return _AUDIT.set({"query": query, "started": time.perf_counter(), "stages": [], "events": []})


def finish_audit(token: object) -> dict[str, Any]:
    data = _AUDIT.get() or {}
    if data:
        data["total_seconds"] = round(time.perf_counter() - data["started"], 6)
    _AUDIT.reset(token)  # type: ignore[arg-type]
    return data


def _compact(paper: dict[str, Any]) -> dict[str, Any]:
    return {
        key: paper.get(key)
        for key in (
            "title", "arxiv_id", "doi", "openalex_id", "paper_id", "url", "year",
            "source_api", "source_query", "source_round", "retrieval_lane", "retrieval_hits",
            "final_score", "diversity_adjusted_score", "llm_relevance_score", "llm_relevant",
        )
        if paper.get(key) not in (None, "", [], {})
    }


def record_stage(name: str, papers: list[dict[str, Any]], **metadata: Any) -> None:
    data = _AUDIT.get()
    if data is None:
        return
    data["stages"].append({
        "name": name,
        "elapsed_seconds": round(time.perf_counter() - data["started"], 6),
        "count": len(papers),
        "metadata": metadata,
        "papers": [_compact(paper) for paper in papers],
    })


def record_event(name: str, **metadata: Any) -> None:
    data = _AUDIT.get()
    if data is None:
        return
    data["events"].append({
        "name": name,
        "elapsed_seconds": round(time.perf_counter() - data["started"], 6),
        **metadata,
    })

