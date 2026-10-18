"""Evidence recovery and second-pass verification for borderline ACE cases."""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable


def validate_evidence_span(span: str, paper: dict[str, Any]) -> bool:
    normalized = " ".join((span or "").lower().split())
    source = " ".join(f"{paper.get('title', '')} {paper.get('abstract', '')}".lower().split())
    return bool(normalized) and normalized in source


def needs_verification(result: dict[str, Any], threshold: float = 0.65) -> bool:
    probability = float(result.get("calibrated_probability") or 0)
    return (
        result.get("hard_gate") == "unknown"
        or abs(probability - threshold) <= 0.10
        or float(result.get("disagreement") or 0) >= 0.35
    )


async def verify_with_llm(
    query: str,
    graph: dict[str, Any],
    papers: list[dict[str, Any]],
    chat: Callable[[list[dict[str, str]], int, float], Awaitable[tuple[str, int]]],
) -> tuple[list[dict[str, Any]], int]:
    """Ask one batched verifier call to challenge first-pass criterion judgments."""
    compact = [
        {"paper_index": index, "title": p.get("title"), "abstract": str(p.get("abstract") or "")[:1600]}
        for index, p in enumerate(papers)
    ]
    prompt = (
        "Verify atomic academic-query constraints. Judge every criterion before any overall decision. "
        "Use only the supplied title/abstract. Return JSON array; each item has paper_index and judgments; "
        "each judgment has criterion_id, state ENTAILED|CONTRADICTED|UNKNOWN, and an exact shortest evidence_span. "
        "UNKNOWN is allowed and is not negative. Look specifically for mistakes in an earlier selector.\n"
        f"Query: {query}\nCriteria: {json.dumps(graph.get('criteria') or [], ensure_ascii=False)}\n"
        f"Papers: {json.dumps(compact, ensure_ascii=False)}"
    )
    content, tokens = await chat([{"role": "user", "content": prompt}], 4000, 0.0)
    try:
        parsed = json.loads(content.strip().removeprefix("```json").removesuffix("```").strip())
    except Exception:
        return [], tokens
    return parsed if isinstance(parsed, list) else [], tokens
