"""Production adapter for frozen REG plus multi-view evidence quorum."""

from __future__ import annotations

import asyncio
import copy
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from backend.retrieval.query_conditioned_citation_v11 import retrieve_query_conditioned_frontier
from backend.retrieval.seed_conditioned_citation_v15 import retrieve_seed_conditioned_reverse_frontier
from backend.retrieval.evidence_backprojection_v21 import retrieve_evidence_backprojection
from backend.selection.local_graph_reranker_v17 import make_selector, score_query_pool
from backend.selection.evidence_quorum_router_v20 import (
    evidence_quorum_score,
    route_evidence_quorum_top20,
)
from backend.selection.relational_evidence_graph_v18 import (
    attach_relational_features,
    relational_score,
    select_quota_frontier_pool,
)
from backend.trace_rank import fuse_candidates, score_candidates


ROOT = Path(__file__).resolve().parents[2]
FREEZE_PATH = ROOT / "configs/relational_evidence_graph_v18/calibration_freeze.json"
CROSS_CACHE = ROOT / "cache/v18_relational_runtime_cross_scores.json"
DEFAULT_WEIGHTS = {
    "default": (0.45, 0.15, 0.10, 0.05, 0.10, 0.05, 0.10),
    "taxonomy": (0.35, 0.15, 0.10, 0.10, 0.15, 0.10, 0.05),
    "survey": (0.35, 0.15, 0.10, 0.10, 0.15, 0.10, 0.05),
    "usage": (0.35, 0.15, 0.10, 0.10, 0.15, 0.10, 0.05),
}


def _infer_relation(text: str) -> str:
    value = str(text or "").casefold()
    families = (
        (r"propos|introduc|pioneer|first\b|提出|首创|开创", "proposes or introduces"),
        (r"extend|improv|generaliz|enhanc|build upon|扩展|改进|增强|推广", "extends or improves"),
        (r"analy|theor|investigat|evaluat|分析|理论|评估|研究机理", "theoretically analyzes"),
        (r"classif|categor|taxonomy|types?\b|分类|类别|类型", "classifies or categorizes"),
        (r"survey|review|overview|综述|回顾", "surveys or reviews"),
        (r"\buse|employ|apply|utiliz|使用|采用|应用", "uses or applies"),
    )
    return next((label for pattern, label in families if re.search(pattern, value)), "")


@lru_cache(maxsize=1)
def frozen_config() -> dict[str, Any]:
    if FREEZE_PATH.exists():
        payload = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        chosen = payload.get("chosen") or {}
        return {
            "weights": payload.get("weights") or DEFAULT_WEIGHTS,
            "score_floor": float(chosen.get("score_floor") or 0.62),
            "margin": float(chosen.get("margin") or 0.10),
            "max_promotions": int(chosen.get("max_promotions") or 1),
            "protect_head": int(chosen.get("protect_head") or 15),
            "source_commit": payload.get("source_commit"),
            "quorum_floor": 0.97,
            "quorum_margin": 0.20,
            "quorum_protect_head": 18,
            "required_views": 3,
            "agreement_bonus": 0.05,
        }
    return {
        "weights": DEFAULT_WEIGHTS, "score_floor": 0.62, "margin": 0.10,
        "max_promotions": 1, "protect_head": 15, "source_commit": None,
        "quorum_floor": 0.97, "quorum_margin": 0.20,
        "quorum_protect_head": 18, "required_views": 3, "agreement_bonus": 0.05,
    }


@lru_cache(maxsize=1)
def runtime_selector() -> Any:
    return make_selector(CROSS_CACHE)


def contract_from_understanding(understanding: dict[str, Any]) -> dict[str, Any]:
    graph = understanding.get("constraint_graph") or {}
    conditions: list[dict[str, Any]] = []
    seen_conditions: set[str] = set()
    relation = ""
    exclusions = list(graph.get("negative_constraints") or [])

    def add_condition(identifier: str, description: str, aliases: Sequence[Any] = ()) -> None:
        key = re.sub(r"[\W_]+", "", description.casefold(), flags=re.UNICODE)
        if not key or key in seen_conditions:
            return
        seen_conditions.add(key)
        conditions.append({
            "id": identifier or f"c{len(conditions)+1}",
            "description": description,
            "aliases": list(dict.fromkeys(str(value).strip() for value in aliases if str(value).strip())),
        })

    for criterion in graph.get("criteria") or []:
        kind = str(criterion.get("type") or "topic")
        description = str(criterion.get("description") or "").strip()
        if not description:
            continue
        if kind == "relation":
            relation = relation or description
        elif kind == "negative":
            exclusions.append(description)
        elif kind not in {"time", "paper_type"}:
            add_condition(
                str(criterion.get("criterion_id") or ""), description,
                criterion.get("aliases") or [],
            )
    relation = relation or str((graph.get("relation_constraints") or [""])[0]).strip()
    relation = relation or _infer_relation(str(understanding.get("original_query") or ""))
    for field in ("research_topic", "method_constraints", "task_constraints", "domain_constraints"):
        value = understanding.get(field)
        values = value if isinstance(value, list) else re.split(r"[,;|]", str(value or ""))
        for item in values:
            text = str(item or "").strip()
            if text and text.lower() not in {"not specified", "unknown", "n/a"}:
                add_condition(f"c{len(conditions)+1}", text)
    clean_exclusions = list(dict.fromkeys(
        str(value).strip() for value in exclusions if str(value).strip()
    ))
    return {"conditions": conditions[:6], "relation": relation, "exclusions": clean_exclusions[:8]}


def route_runtime_frontier(
    query: str,
    contract: dict[str, Any],
    incumbents: Sequence[dict[str, Any]],
    frontier: Sequence[dict[str, Any]],
    selector: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = frozen_config()
    quota = select_quota_frontier_pool(frontier, incumbents, limit=200)
    cross_rows, runtime = score_query_pool(
        query, contract, [*incumbents, *quota], selector, batch_size=64,
    )
    rows = attach_relational_features(cross_rows, query, contract)
    n = len(incumbents)
    routed, actions = route_evidence_quorum_top20(
        rows[:n], rows[n:], weights=config["weights"],
        reg_score_floor=config["score_floor"], reg_margin=config["margin"],
        reg_protect_head=config["protect_head"],
        quorum_floor=config["quorum_floor"], quorum_margin=config["quorum_margin"],
        quorum_protect_head=config["quorum_protect_head"],
        required_views=config["required_views"], agreement_bonus=config["agreement_bonus"],
    )
    action_titles = {str(action.get("incoming_title") or "") for action in actions}
    for paper in routed:
        score = relational_score(paper, config["weights"])
        paper["relational_evidence"] = {
            "score": round(score, 4),
            "condition_conjunction": round(float(paper.get("v18_conjunction") or 0), 4),
            "weakest_condition": round(float(paper.get("v18_condition_minimum") or 0), 4),
            "relation_evidence": round(float(paper.get("v18_relation") or 0), 4),
            "citation_direction": round(float(paper.get("v18_direction") or 0), 4),
            "seed_bridge": round(float(paper.get("v18_seed_bridge") or 0), 4),
            "evidence_quorum": round(evidence_quorum_score(
                paper,
                required_views=config["required_views"],
                agreement_bonus=config["agreement_bonus"],
            ), 4),
            "relation_family": paper.get("v18_relation_family"),
            "promoted_from_frontier": str(paper.get("title") or "") in action_titles,
            "promotion_route": next((
                action.get("route") for action in actions
                if str(action.get("incoming_title") or "") == str(paper.get("title") or "")
            ), None),
        }
    return routed, {
        "strategy": "REG_MEQ_v20_relational_evidence_quorum",
        "frontier_count": len(frontier), "quota_count": len(quota),
        "actions": actions, "local_reranker": runtime,
        "config": {key: value for key, value in config.items() if key != "weights"},
    }


async def expand_and_route_relational(
    query: str,
    understanding: dict[str, Any],
    incumbents: Sequence[dict[str, Any]],
    candidate_pool: Sequence[dict[str, Any]],
    *,
    selector: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run graph retrieval, EBP candidate expansion and frozen Top20 routing."""

    contract = contract_from_understanding(understanding)
    reverse = await retrieve_seed_conditioned_reverse_frontier(
        query, understanding, contract, incumbents, candidate_pool,
    )
    qcf = await retrieve_query_conditioned_frontier(
        query, understanding, contract, incumbents, candidate_pool,
    )
    frontier = fuse_candidates([*(reverse.get("candidates") or []), *(qcf.get("candidates") or [])])
    chosen_selector = selector or runtime_selector()

    # EBP needs the same query-relative evidence views as REG-MEQ.  This first
    # pass only establishes the evidence field; the frozen Top20 router remains
    # unchanged and therefore cannot be perturbed by the new candidates.
    quota = select_quota_frontier_pool(frontier, incumbents, limit=200)
    evidence_rows, evidence_runtime = await asyncio.to_thread(
        score_query_pool, query, contract, [*incumbents, *quota], chosen_selector,
        batch_size=64,
    )
    evidence_rows = await asyncio.to_thread(
        attach_relational_features, evidence_rows, query, contract,
    )
    n = len(incumbents)
    ebp = await retrieve_evidence_backprojection(evidence_rows[:n], evidence_rows[n:])
    ebp_candidates = score_candidates(
        list(ebp.get("candidates") or []), understanding,
    )
    ebp_candidates.sort(key=lambda paper: (
        int(paper.get("citation_backprojection_rank") or 10**9),
        -float(paper.get("final_score") or 0),
        str(paper.get("title") or "").casefold(),
    ))

    routed, audit = await asyncio.to_thread(
        route_runtime_frontier, query, contract, copy.deepcopy(incumbents), frontier, chosen_selector,
    )
    audit.update({
        "enabled": True,
        "api_calls": (
            int(reverse.get("api_calls") or 0)
            + int(qcf.get("api_calls") or 0)
            + int(ebp.get("api_calls") or 0)
        ),
        "reverse_candidates": len(reverse.get("candidates") or []),
        "qcf_candidates": len(qcf.get("candidates") or []),
        "ebp_candidates": len(ebp_candidates),
        "ebp_targets": len(ebp.get("targets") or []),
        "ebp_local_evidence_runtime": evidence_runtime,
        "supplementary_candidates": ebp_candidates,
        "errors": [
            *(reverse.get("errors") or []),
            *(qcf.get("errors") or []),
            *(ebp.get("errors") or []),
        ],
        "contract": contract,
    })
    return routed, audit
