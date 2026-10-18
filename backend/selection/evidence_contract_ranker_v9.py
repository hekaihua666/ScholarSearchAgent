"""Evidence Contract Rank (ECRank) for complex academic-paper queries.

ECRank compiles a natural-language query into a multi-condition contract and
asks one listwise LLM call to score a compact, source-diverse shortlist from
both point and list views.  A risk-bounded router may replace weak incumbents,
but only with evidence-complete newcomers whose calibrated score clears an
absolute threshold and an incumbent margin.

The module contains no benchmark labels, query ids, title patches, or expected
answer cardinality.  All production decisions depend only on the query,
candidate metadata, retrieval provenance, and the frozen LLM response.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from backend.llm_client import _chat
from backend.paper_identity import match_identity, normalize_title


MAX_BASELINE_CARDS = 4
MAX_NEWCOMER_CARDS = 12
MAX_CARDS = MAX_BASELINE_CARDS + MAX_NEWCOMER_CARDS
FORBIDDEN_MARKERS = ("gold", "label", "query_id", "expected_cardinality", "matched_gold")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bounded(value: Any, low: float = 0.0, high: float = 100.0) -> float:
    return min(high, max(low, _float(value)))


def _identity_key(paper: dict[str, Any]) -> str:
    doi = str(paper.get("doi") or "").lower().removeprefix("https://doi.org/").strip()
    if doi:
        return f"doi:{doi}"
    arxiv = str(paper.get("arxiv_id") or "").lower().strip()
    if arxiv:
        return f"arxiv:{arxiv}"
    openalex = str(paper.get("openalex_id") or "").lower().strip()
    if openalex:
        return f"openalex:{openalex}"
    return f"title:{normalize_title(paper.get('title'))}"


def evidence_prior(paper: dict[str, Any]) -> float:
    """Training-free compact score used only to form the LLM shortlist."""
    topic = _bounded(paper.get("topic_coverage"), 0, 1)
    facet = _bounded(paper.get("facet_coverage"), 0, 1)
    text = max(
        _bounded(paper.get("title_coverage"), 0, 1),
        _bounded(paper.get("abstract_coverage"), 0, 1),
    )
    coverage = _bounded(paper.get("constraint_coverage_score"), 0, 1)
    rrf = _bounded(paper.get("rrf_score"), 0, 1)
    entity = float(bool(paper.get("exact_entity_match")))
    conditions = (topic, facet, text)
    return 0.35 * min(conditions) + 0.25 * sum(conditions) / 3 + 0.15 * coverage + 0.15 * rrf + 0.10 * entity


def source_rank_prior(paper: dict[str, Any]) -> float:
    ranks = [
        int(value) for value in (paper.get("arxiv_rank"), paper.get("openalex_rank"))
        if value is not None and str(value).isdigit()
    ]
    return 1.0 / (60.0 + min(ranks, default=100))


def _dedupe(papers: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for paper in papers:
        if not paper.get("title"):
            continue
        if any(match_identity(paper, old) or match_identity(old, paper) for old in result):
            continue
        result.append(paper)
    return result


def select_newcomers(
    newcomers: Sequence[dict[str, Any]], baseline: Sequence[dict[str, Any]], limit: int = MAX_NEWCOMER_CARDS,
) -> list[dict[str, Any]]:
    """Union complementary coarse views so low-ranked compositional hits survive."""
    legal = [
        paper for paper in _dedupe(newcomers)
        if not any(match_identity(paper, old) or match_identity(old, paper) for old in baseline)
    ]
    views = (
        sorted(legal, key=lambda paper: _float(paper.get("final_score")), reverse=True),
        sorted(legal, key=evidence_prior, reverse=True),
        sorted(legal, key=source_rank_prior, reverse=True),
    )
    selected: list[dict[str, Any]] = []
    for depth in range(max((len(view) for view in views), default=0)):
        for view in views:
            if depth >= len(view):
                continue
            paper = view[depth]
            if paper not in selected:
                selected.append(paper)
            if len(selected) >= limit:
                return selected
    return selected


@dataclass(frozen=True)
class ContractCard:
    card_id: str
    origin: str
    original_rank: int | None
    paper: dict[str, Any]

    def prompt_payload(self) -> dict[str, Any]:
        paper = self.paper
        return {
            "id": self.card_id,
            "title": str(paper.get("title") or "")[:300],
            "abstract": str(paper.get("abstract") or "")[:1100],
            "year": paper.get("year"),
            "retrieval_evidence": {
                "facet_coverage": round(_bounded(paper.get("facet_coverage"), 0, 1), 3),
                "constraint_coverage": round(_bounded(paper.get("constraint_coverage_score"), 0, 1), 3),
                "source_count": int(paper.get("source_count") or 1),
                "variant_count": int(paper.get("variant_count") or 1),
                "best_source_rank": min(
                    [int(value) for value in (paper.get("arxiv_rank"), paper.get("openalex_rank")) if value is not None]
                    or [int(paper.get("candidate_rank") or 100)]
                ),
            },
        }


def build_contract_cards(
    query: str, baseline_top20: Sequence[dict[str, Any]], newcomers: Sequence[dict[str, Any]],
) -> list[ContractCard]:
    baseline = list(baseline_top20[:MAX_BASELINE_CARDS])
    new = select_newcomers(newcomers, baseline_top20, MAX_NEWCOMER_CARDS)
    cards = [
        ContractCard("", "baseline", rank, paper) for rank, paper in enumerate(baseline, 1)
    ] + [ContractCard("", "newcomer", None, paper) for paper in new]
    # Opaque ids and deterministic hash ordering avoid exposing incumbent/newcomer
    # status to the LLM and reduce rank-position anchoring.
    ordered = sorted(cards, key=lambda card: hashlib.sha256(
        f"{query}\0{_identity_key(card.paper)}".encode("utf-8")
    ).hexdigest())
    return [ContractCard(f"P{index:02d}", card.origin, card.original_rank, card.paper) for index, card in enumerate(ordered, 1)]


def build_contract_prompt(query: str, cards: Sequence[ContractCard]) -> str:
    payload = [card.prompt_payload() for card in cards]
    return (
        "You are Evidence Contract Rank, a strict scientific-paper reranker for complex academic queries.\n"
        "The query may identify papers through contributions, accepted aliases, survey taxonomies, or implicit paradigm names rather than literal title words.\n"
        "Step 1: compile a QUERY CONTRACT with 1-5 required conditions, accepted research aliases/subtypes, the requested relation (introduced/proposed/used/evaluated/discussed), and explicit exclusions.\n"
        "Step 2: assess every candidate twice: point_score judges the candidate independently; list_score judges it comparatively against this list.\n"
        "For each contract condition and the requested relation, use support 0=absent/contradicted, 1=weak, 2=credible implicit support, 3=explicit support. General topical similarity is insufficient.\n"
        "Use established scientific terminology to recognize aliases, but do not invent paper content. Penalize another field, wrong relation, or missing required conditions.\n"
        "Return valid JSON only, no markdown. Use exactly this shape:\n"
        '{"contract":{"conditions":[{"id":"C1","description":"...","aliases":["..."]}],"relation":"...","exclusions":["..."]},'
        '"ranking":["P01","P02"],"assessments":{"P01":{"point_score":0,"list_score":0,"verdict":"relevant|partial|irrelevant",'
        '"condition_support":{"C1":0},"relation_support":0,"contradiction":false,"evidence":"..."}}}\n'
        "Every candidate id must occur exactly once in ranking and assessments. Scores are integers 0-100.\n"
        f"QUERY: {query}\nCANDIDATES: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )


def assert_prompt_safe(prompt: str) -> None:
    # Natural scientific text may legitimately contain words such as
    # "label" or "gold".  Audit only the instruction/schema prefix; candidate
    # content is already projected through ContractCard.prompt_payload().
    lowered = prompt.split("QUERY:", 1)[0].lower()
    if any(marker in lowered for marker in FORBIDDEN_MARKERS):
        raise AssertionError("benchmark-derived marker in ECRank prompt")


def _strip_json(text: str) -> str:
    value = str(text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.I)
        value = re.sub(r"\s*```$", "", value)
    start, end = value.find("{"), value.rfind("}")
    return value[start:end + 1] if start >= 0 and end > start else value


def parse_contract_response(text: str, card_ids: Sequence[str]) -> dict[str, Any]:
    raw = json.loads(_strip_json(text))
    valid = set(card_ids);assessments_raw = raw.get("assessments") or {}
    if isinstance(assessments_raw, list):
        assessments_raw = {str(row.get("id")): row for row in assessments_raw if isinstance(row, dict)}
    ranking_raw = raw.get("ranking") or []
    ranking: list[str] = []
    for item in ranking_raw:
        card_id = str(item.get("id")) if isinstance(item, dict) else str(item)
        if card_id in valid and card_id not in ranking:
            ranking.append(card_id)
    assessments: dict[str, dict[str, Any]] = {}
    for card_id in card_ids:
        row = assessments_raw.get(card_id) or {}
        support = row.get("condition_support") or {}
        if isinstance(support, list):
            support = {f"C{index + 1}": value for index, value in enumerate(support)}
        assessments[card_id] = {
            "point_score": _bounded(row.get("point_score", row.get("score"))),
            "list_score": _bounded(row.get("list_score", row.get("score"))),
            "verdict": str(row.get("verdict") or "irrelevant").lower(),
            "condition_support": {str(key): _bounded(value, 0, 3) for key, value in support.items()},
            "relation_support": _bounded(row.get("relation_support"), 0, 3),
            "contradiction": bool(row.get("contradiction")),
            "evidence": str(row.get("evidence") or "")[:500],
        }
    ranking.extend(card_id for card_id in card_ids if card_id not in ranking)
    return {"contract": raw.get("contract") or {}, "ranking": ranking, "assessments": assessments}


async def request_contract_ranking(query: str, cards: Sequence[ContractCard]) -> dict[str, Any]:
    prompt = build_contract_prompt(query, cards);assert_prompt_safe(prompt)
    content, tokens = await _chat([{"role": "user", "content": prompt}], max_tokens=3200, temperature=0.0)
    parsed = parse_contract_response(content, [card.card_id for card in cards])
    return {"parsed": parsed, "raw_content": content, "tokens": tokens, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()}


def contract_score(assessment: dict[str, Any], rank: int, size: int) -> tuple[float, bool]:
    supports = list((assessment.get("condition_support") or {}).values())
    mean_support = sum(map(_float, supports)) / len(supports) / 3.0 if supports else 0.0
    minimum_support = min(map(_float, supports)) / 3.0 if supports else 0.0
    relation = _float(assessment.get("relation_support")) / 3.0
    point = _bounded(assessment.get("point_score")) / 100.0
    list_view = _bounded(assessment.get("list_score")) / 100.0
    rank_view = 1.0 - (rank - 1) / max(size - 1, 1)
    # Point/list self-calibration plus an explicit bottleneck term.  Ranking
    # position is only a weak tie-breaker, limiting list-order instability.
    score = 100.0 * (0.30 * point + 0.25 * list_view + 0.20 * mean_support + 0.15 * minimum_support + 0.05 * relation + 0.05 * rank_view)
    if assessment.get("contradiction"):
        score *= 0.25
    complete = (
        str(assessment.get("verdict")) == "relevant"
        and bool(supports) and minimum_support >= 2 / 3 and relation >= 2 / 3
        and not assessment.get("contradiction")
    )
    return score, complete


def score_cards(cards: Sequence[ContractCard], parsed: dict[str, Any]) -> list[dict[str, Any]]:
    ranking = parsed.get("ranking") or [];rank_by = {card_id: index for index, card_id in enumerate(ranking, 1)}
    assessments = parsed.get("assessments") or {};output=[]
    for card in cards:
        rank = rank_by.get(card.card_id, len(cards));assessment=assessments.get(card.card_id) or {}
        score,complete=contract_score(assessment,rank,len(cards))
        output.append({"card_id":card.card_id,"origin":card.origin,"original_rank":card.original_rank,
                       "paper":card.paper,"contract_rank":rank,"contract_score":score,
                       "evidence_complete":complete,"assessment":assessment})
    return sorted(output,key=lambda row:(-row["contract_score"],row["contract_rank"],row["card_id"]))


def infer_contract_mode(query: str, parsed: dict[str, Any]) -> str:
    """Detect the narrow case where different papers satisfy parallel asks.

    Example: "What papers introduced FGSM and BIM?" asks for a *set* of
    origin papers; requiring every paper to introduce both methods violates
    relevance monotonicity.  We deliberately use a conservative pattern so
    ordinary multi-condition queries remain per-paper conjunctive.
    """
    contract=parsed.get("contract") or {};conditions=contract.get("conditions") or []
    descriptions=[str(row.get("description") or "").lower() for row in conditions if isinstance(row,dict)]
    relation=str(contract.get("relation") or "").lower().split("/",1)[0].strip()
    relation_root=re.sub(r"(?:ed|d|s)$","",relation)
    plural=bool(re.search(r"\b(papers|works|studies)\b",query,re.I))
    coordinated=bool(re.search(r"\band\b",query,re.I))
    parallel=(len(descriptions)>=2 and bool(relation_root) and all(
        re.search(rf"\bpaper\b.*\b{re.escape(relation_root)}\w*\b",description) for description in descriptions
    ))
    return "set_distributive" if plural and coordinated and parallel else "per_paper_conjunctive"


def specialist_score(row: dict[str, Any]) -> tuple[float,bool]:
    assessment=row.get("assessment") or {};supports=list((assessment.get("condition_support") or {}).values())
    maximum=max(map(_float,supports),default=0.0)/3.0;mean=sum(map(_float,supports))/len(supports)/3.0 if supports else 0.0
    relation=_float(assessment.get("relation_support"))/3.0;point=_bounded(assessment.get("point_score"))/100.0;list_view=_bounded(assessment.get("list_score"))/100.0
    score=100.0*(.35*point+.30*list_view+.20*maximum+.10*mean+.05*relation)
    valid=(str(assessment.get("verdict"))=="relevant" and maximum>=2/3 and relation>=2/3 and not assessment.get("contradiction"))
    return score,valid


def route_distributive_core(
    baseline_top20:Sequence[dict[str,Any]],scored_cards:Sequence[dict[str,Any]],*,threshold:float,margin:float,max_replacements:int,
) -> tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    incumbents=[row for row in scored_cards if row["origin"]=="baseline" and int(row.get("original_rank") or 99)<=2]
    incumbents.sort(key=lambda row:int(row["original_rank"]));current={int(row["original_rank"]):row for row in incumbents};actions=[]
    condition_ids=sorted({str(key) for row in scored_cards for key in (row.get("assessment",{}).get("condition_support") or {})})
    def coverage(rows:Iterable[dict[str,Any]])->dict[str,float]:
        values={key:0.0 for key in condition_ids}
        for row in rows:
            for key,value in (row.get("assessment",{}).get("condition_support") or {}).items():values[str(key)]=max(values.get(str(key),0.0),_float(value))
        return values
    newcomers=[]
    for row in scored_cards:
        if row["origin"]!="newcomer":continue
        score,valid=specialist_score(row)
        if valid and score>=threshold:newcomers.append({**row,"specialist_score":score})
    newcomers.sort(key=lambda row:(-row["specialist_score"],row["contract_rank"]))
    for candidate in newcomers:
        if len(actions)>=max_replacements or not current:break
        before=coverage(current.values());support=candidate["assessment"].get("condition_support") or {}
        gain=sum(max(0.0,_float(support.get(key))-before.get(key,0.0)) for key in condition_ids)
        if gain<1.0:continue
        slot,worst=min(current.items(),key=lambda item:(item[1]["contract_score"],-item[0]))
        if candidate["specialist_score"]<worst["contract_score"]+margin:continue
        current[slot]=candidate
        actions.append({"incoming_id":candidate["card_id"],"outgoing_id":worst["card_id"],"incoming_score":candidate["specialist_score"],
                        "outgoing_score":worst["contract_score"],"margin":candidate["specialist_score"]-worst["contract_score"],
                        "coverage_gain":gain,"contract_mode":"set_distributive"})
    selected=[dict(current[slot]["paper"]) for slot in sorted(current)[:2]]
    for rank,paper in enumerate(selected,1):paper["candidate_rank"]=rank;paper["live_rank"]=rank
    if len(selected)<2:selected=[dict(row) for row in baseline_top20[:2]];actions=[]
    return selected,actions


def route_contract_core(
    query:str,baseline_top20:Sequence[dict[str,Any]],scored_cards:Sequence[dict[str,Any]],parsed:dict[str,Any],*,threshold:float,margin:float,max_replacements:int,
) -> tuple[list[dict[str,Any]],list[dict[str,Any]],str]:
    mode=infer_contract_mode(query,parsed)
    if mode=="set_distributive":selected,actions=route_distributive_core(baseline_top20,scored_cards,threshold=threshold,margin=margin,max_replacements=max_replacements)
    else:selected,actions=route_core(baseline_top20,scored_cards,threshold=threshold,margin=margin,max_replacements=max_replacements)
    return selected,actions,mode


def route_core(
    baseline_top20: Sequence[dict[str, Any]], scored_cards: Sequence[dict[str, Any]], *,
    threshold: float, margin: float, max_replacements: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Risk-bounded replacement of at most two current Top-2 incumbents."""
    incumbents = [
        row for row in scored_cards if row["origin"] == "baseline" and int(row.get("original_rank") or 99) <= 2
    ]
    incumbents.sort(key=lambda row:int(row["original_rank"]));current={int(row["original_rank"]):row for row in incumbents};actions=[]
    newcomers = [row for row in scored_cards if row["origin"] == "newcomer" and row["evidence_complete"] and row["contract_score"] >= threshold]
    for candidate in newcomers:
        if len(actions) >= max_replacements or not current:
            break
        slot,worst = min(current.items(),key=lambda item:(item[1]["contract_score"],-item[0]))
        if candidate["contract_score"] < worst["contract_score"] + margin:
            continue
        current[slot]=candidate
        actions.append({"incoming_id":candidate["card_id"],"outgoing_id":worst["card_id"],
                        "incoming_score":candidate["contract_score"],"outgoing_score":worst["contract_score"],
                        "margin":candidate["contract_score"]-worst["contract_score"]})
    selected=[dict(current[slot]["paper"]) for slot in sorted(current)[:2]]
    for rank,paper in enumerate(selected,1):paper["candidate_rank"]=rank;paper["live_rank"]=rank
    # If an LLM response omitted an incumbent, fail closed to the original core.
    if len(selected)<2:
        selected=[dict(row) for row in baseline_top20[:2]];actions=[]
    return selected,actions


def route_top20(
    baseline_top20: Sequence[dict[str, Any]], selected_core: Sequence[dict[str, Any]],
    scored_cards: Sequence[dict[str, Any]], *, threshold: float, limit: int = 20,
) -> list[dict[str, Any]]:
    promoted = [
        dict(row["paper"]) for row in scored_cards
        if row["origin"]=="newcomer" and row["evidence_complete"] and row["contract_score"]>=threshold
    ]
    ordered = _dedupe([*selected_core,*promoted,*baseline_top20])[:limit]
    for rank,paper in enumerate(ordered,1):paper["candidate_rank"]=rank;paper["live_rank"]=rank
    return ordered


def assert_no_forbidden_state(cards: Sequence[ContractCard]) -> None:
    for card in cards:
        for key in card.paper:
            if any(marker in str(key).lower() for marker in FORBIDDEN_MARKERS):
                raise AssertionError(f"forbidden field carried into ECRank card: {key}")
