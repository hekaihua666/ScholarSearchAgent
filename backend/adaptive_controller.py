"""TRACE-Loop process-aware retrieval controller.

Unlike a fixed two-round workflow, this controller measures evidence sufficiency,
search drift and source agreement before choosing the next action.  It is a
deterministic, auditable alternative to training a policy model with RL.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from backend.trace_rank import tokenize


@dataclass
class LoopAssessment:
    sufficiency: float
    relevance: float
    constraint_coverage: float
    source_diversity: float
    agreement: float
    metadata_quality: float
    drift: float
    next_action: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_retrieval(
    original_query: str,
    papers: list[dict[str, Any]],
    api_calls: int = 0,
    max_api_calls: int = 12,
) -> LoopAssessment:
    top = papers[:20]
    if not top:
        return LoopAssessment(0, 0, 0, 0, 0, 0, 1, "query_evolution", "候选为空，需要改写查询并切换数据源")
    relevance = sum(float(p.get("final_score") or 0) for p in top[:10]) / min(len(top), 10)
    coverage = sum(float(p.get("constraint_coverage_score") or 0) for p in top[:10]) / min(len(top), 10)
    sources = {str(p.get("source_api") or "unknown") for p in top}
    source_diversity = min(len(sources) / 3.0, 1.0)
    agreement = sum(len(p.get("retrieval_hits") or []) >= 2 for p in top) / len(top)
    metadata = sum(float(p.get("metadata_score") or 0) for p in top) / len(top)
    query_tokens = tokenize(original_query)
    title_tokens = set().union(*(tokenize(str(p.get("title") or "")) for p in top[:10]))
    drift = 1.0 - (len(query_tokens & title_tokens) / len(query_tokens) if query_tokens else relevance)
    drift = max(0.0, min(drift, 1.0))
    sufficiency = 100 * (
        0.32 * relevance + 0.25 * coverage + 0.14 * source_diversity
        + 0.14 * agreement + 0.10 * metadata + 0.05 * (1 - drift)
    )
    has_citation_seeds = any(p.get("referenced_works") for p in top[:8])
    if api_calls >= max_api_calls:
        action, reason = "stop_budget", "已达到API预算，停止扩展并返回当前最优结果"
    elif drift > 0.78:
        action, reason = "query_evolution", "检索结果出现明显主题漂移，优先重新规划查询"
    elif has_citation_seeds and sufficiency < 72:
        action, reason = "citation_expansion", "当前证据不足且存在高相关种子论文，沿引文网络扩展"
    elif sufficiency < 78:
        action, reason = "query_evolution", "约束覆盖或多源一致性不足，执行补充检索"
    else:
        action, reason = "stop_sufficient", "证据覆盖和多源一致性已达到停止阈值"
    return LoopAssessment(
        round(sufficiency, 2), round(relevance, 4), round(coverage, 4),
        round(source_diversity, 4), round(agreement, 4), round(metadata, 4),
        round(drift, 4), action, reason,
    )
