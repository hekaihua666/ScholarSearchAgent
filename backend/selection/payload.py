"""Competition-facing payload boundary: supplementary papers never leak out."""

from __future__ import annotations

from typing import Any


def evaluation_payload(response: dict[str, Any]) -> dict[str, Any]:
    selected = response.get("selected_results")
    if selected is None:
        selected = response.get("papers") or []
    return {
        "query": response.get("query"),
        "papers": [
            {key: paper.get(key) for key in ("arxiv_id", "doi", "openalex_id", "title", "url") if paper.get(key)}
            for paper in selected
        ],
        "selection_profile": (response.get("selection") or {}).get("profile"),
    }
