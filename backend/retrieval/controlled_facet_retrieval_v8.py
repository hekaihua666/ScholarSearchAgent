"""Controlled dual-source facet retrieval (DSFR) for the v8 experiment.

The component is intentionally deterministic and small: it derives at most two
short queries from the existing structured query object, calls only arXiv and
OpenAlex, and never changes the baseline Top-2 core.
"""

from __future__ import annotations

import asyncio
import copy
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

from backend.arxiv_client import search_arxiv
from backend.eval_runtime import current_session, filter_for_cutoff, frozen_call
from backend.openalex_client import OpenAlexRateLimitError, search_openalex
from backend.paper_identity import match_identity, normalize_title, paper_identity
from backend.trace_rank import diversify, fuse_candidates, score_candidates, tokenize


TEMPLATE_NAMES = ("T1", "T2", "T3", "T4")
DEFAULT_POLICY = ("T2", "T3")
MAX_VARIANTS = 2
MAX_PHRASES = 5
PER_SOURCE_LIMIT = 25
_ARXIV_LOCK = asyncio.Lock()
_ARXIV_LAST_CALL = 0.0

_EMPTY = {"", "not specified", "none", "n/a", "unknown"}
_GENERIC = {
    "paper", "papers", "work", "works", "study", "studies", "research",
    "method", "methods", "task", "tasks", "dataset", "datasets", "model",
    "models", "algorithm", "algorithms", "application", "applications",
    "natural language processing", "machine learning", "computer science",
}


@dataclass(frozen=True)
class FacetQuery:
    facet_query_id: str
    template: str
    query: str
    phrases: tuple[str, ...]
    facet_types: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = re.sub(r"\s+", " ", str(raw or "")).strip(" ,.;:()[]{}\"'")
        key = value.casefold()
        if key in _EMPTY or key in seen or len(value) < 2:
            continue
        seen.add(key);result.append(value)
    return result


def _split(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return _dedupe(str(item) for item in value)
    return _dedupe(re.split(r"[,;/|]", str(value or "")))


def _in_original(phrase: str, original_query: str) -> bool:
    compact = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", phrase.casefold())
    source = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", original_query.casefold())
    return bool(compact and compact in source)


def facet_inventory(original_query: str, understanding: dict[str, Any]) -> dict[str, list[str]]:
    """Read the project's actual query-understanding fields without inventing facets."""
    graph = understanding.get("constraint_graph") or {}
    criteria: dict[str, list[str]] = {name: [] for name in ("topic", "method", "task", "dataset", "domain")}
    for item in graph.get("criteria") or []:
        kind = str(item.get("type") or "").lower()
        description = str(item.get("description") or "")
        if kind in criteria and description.casefold() not in _EMPTY:
            criteria[kind].append(description)

    topic = _dedupe([
        str(understanding.get("research_topic") or ""),
        *criteria["topic"],
    ])
    methods = _dedupe([*_split(understanding.get("method_constraints")), *criteria["method"]])
    tasks = _dedupe([*_split(understanding.get("task_constraints")), *criteria["task"]])
    datasets = _dedupe(criteria["dataset"])
    domains = _dedupe([*_split(understanding.get("domain_constraints")), *criteria["domain"]])

    graph_entities = _split(graph.get("key_entities"))
    quoted = re.findall(r'["“]([^"”]{2,60})["”]', original_query)
    abbreviations = re.findall(r"\b[A-Z][A-Z0-9-]{1,12}\b", original_query)
    entities = [value for value in _dedupe([*graph_entities, *quoted, *abbreviations]) if _in_original(value, original_query)]

    def informative(values: list[str]) -> list[str]:
        kept = [value for value in values if value.casefold() not in _GENERIC]
        return kept or values[:1]

    return {
        "topic": informative(topic)[:3],
        "method": informative(methods)[:3],
        "task": informative(tasks)[:3],
        "dataset": informative(datasets)[:2],
        "domain": informative(domains)[:2],
        "entity": entities[:3],
    }


def _topic_term(topic: str) -> str:
    terms = sorted(tokenize(topic), key=lambda value: (-len(value), value))
    return terms[0] if terms else topic


def template_library(original_query: str, understanding: dict[str, Any]) -> dict[str, FacetQuery]:
    inventory = facet_inventory(original_query, understanding)
    topic = inventory["topic"][:1]
    method = inventory["method"][:1]
    task = (inventory["task"] or inventory["dataset"] or inventory["domain"])[:1]
    entity = inventory["entity"][:1]
    topic_term = [_topic_term(topic[0])] if topic else []
    definitions = {
        "T1": (entity + topic, ("entity", "topic")),
        "T2": (topic + method, ("topic", "method")),
        "T3": (topic + task, ("topic", "task" if inventory["task"] else "dataset" if inventory["dataset"] else "domain")),
        "T4": (method + task + topic_term, ("method", "task", "topic")),
    }
    result: dict[str, FacetQuery] = {}
    for template, (raw_phrases, raw_types) in definitions.items():
        phrases = _dedupe(raw_phrases)[:MAX_PHRASES]
        if len(phrases) < 2:
            continue
        query = " ".join(phrases)
        # A facet query must not silently become the original long question.
        if normalize_title(query) == normalize_title(original_query):
            continue
        result[template] = FacetQuery(
            facet_query_id=template,
            template=template,
            query=query,
            phrases=tuple(phrases),
            facet_types=tuple(raw_types[:len(phrases)]),
        )
    return result


def build_facet_queries(
    original_query: str,
    understanding: dict[str, Any],
    policy: Sequence[str] = DEFAULT_POLICY,
    max_variants: int = MAX_VARIANTS,
) -> list[FacetQuery]:
    library = template_library(original_query, understanding)
    selected: list[FacetQuery] = []
    for template in policy:
        candidate = library.get(str(template))
        if candidate and normalize_title(candidate.query) not in {normalize_title(item.query) for item in selected}:
            selected.append(candidate)
        if len(selected) >= min(max_variants, MAX_VARIANTS):
            break
    if len(selected) < min(max_variants, MAX_VARIANTS):
        for template in TEMPLATE_NAMES:
            candidate = library.get(template)
            if candidate and normalize_title(candidate.query) not in {normalize_title(item.query) for item in selected}:
                selected.append(candidate)
            if len(selected) >= min(max_variants, MAX_VARIANTS):
                break
    return selected


async def _source_call(facet: FacetQuery, source: str, per_source_limit: int) -> tuple[list[dict[str, Any]], str | None]:
    global _ARXIV_LAST_CALL
    arguments = {"facet_query_id":facet.facet_query_id,"query":facet.query,"per_page":per_source_limit,"source":source}
    try:
        if source == "arxiv":
            # arXiv asks automated clients to leave roughly three seconds
            # between requests. OpenAlex calls may continue concurrently.
            async with _ARXIV_LOCK:
                if current_session() is None or current_session().mode=="live":
                    wait_for=3.05-(asyncio.get_running_loop().time()-_ARXIV_LAST_CALL)
                    if wait_for>0:await asyncio.sleep(wait_for)
                papers = await frozen_call(
                    f"dsfr_arxiv_{facet.facet_query_id}", arguments,
                    lambda: search_arxiv(facet.query, per_page=per_source_limit),
                )
                if current_session() is None or current_session().mode=="live":
                    _ARXIV_LAST_CALL=asyncio.get_running_loop().time()
        else:
            papers = await frozen_call(
                f"dsfr_openalex_{facet.facet_query_id}", arguments,
                lambda: search_openalex(facet.query, per_page=per_source_limit),
            )
    except (OpenAlexRateLimitError, asyncio.TimeoutError, Exception) as exc:
        # The caller records the bounded source failure; no fallback source is allowed in v8.
        return [], f"{type(exc).__name__}: {str(exc)[:160]}"
    legal = filter_for_cutoff(papers, f"dsfr_{source}_{facet.facet_query_id}")
    output=[]
    for rank,paper in enumerate(legal,1):
        item=dict(paper);item["_source_rank"]=rank;item["facet_query_id"]=facet.facet_query_id
        item["facet_template"]=facet.template;item["source_query"]=facet.query
        if source=="arxiv":item["arxiv_id"]=item.get("url") or item.get("openalex_id")
        output.append(item)
    return output,None


async def retrieve_dsfr(
    original_query: str,
    understanding: dict[str, Any],
    *,
    policy: Sequence[str] = DEFAULT_POLICY,
    sources: Sequence[str] = ("arxiv", "openalex"),
    max_variants: int = MAX_VARIANTS,
    per_source_limit: int = PER_SOURCE_LIMIT,
) -> dict[str, Any]:
    facets=build_facet_queries(original_query,understanding,policy,max_variants)
    tasks=[_source_call(facet,source,per_source_limit) for facet in facets for source in sources]
    results=await asyncio.gather(*tasks) if tasks else []
    papers=[];errors=[];cursor=0
    for facet in facets:
        for source in sources:
            source_papers,error=results[cursor];cursor+=1;papers.extend(source_papers)
            if error:errors.append({"facet_query_id":facet.facet_query_id,"source":source,"error":error})
    fused=fuse_candidates(papers)
    query_by_text={facet.query:facet for facet in facets}
    scored=score_candidates(fused,understanding)
    inventory=facet_inventory(original_query,understanding)
    for paper in scored:
        add_candidate_evidence(paper,facets,inventory,query_by_text)
    return {
        "facet_queries":[facet.to_dict() for facet in facets],
        "candidates":scored,
        "errors":errors,
        "api_calls":len(tasks),
        "sources":list(sources),
    }


def _coverage(tokens: set[str], phrase: str) -> float:
    required=tokenize(phrase)
    return len(tokens & required)/len(required) if required else 0.0


def add_candidate_evidence(
    paper: dict[str, Any],
    facets: Sequence[FacetQuery],
    inventory: dict[str, list[str]],
    query_by_text: dict[str, FacetQuery] | None = None,
) -> None:
    title_tokens=tokenize(str(paper.get("title") or ""));abstract_tokens=tokenize(str(paper.get("abstract") or ""));all_tokens=title_tokens|abstract_tokens
    phrases=_dedupe(value for values in inventory.values() for value in values)
    matched=[phrase for phrase in phrases if _coverage(all_tokens,phrase)>=.5]
    hits=paper.get("retrieval_hits") or []
    query_by_text=query_by_text or {facet.query:facet for facet in facets}
    facet_ids=_dedupe((query_by_text.get(str(hit.get("query") or "")) or FacetQuery("","","",(),())).facet_query_id for hit in hits)
    sources={"arxiv" if "arxiv" in str(hit.get("source") or "").lower() else "openalex" for hit in hits}
    source_ranks={source:[int(hit.get("rank") or 100) for hit in hits if source in str(hit.get("source") or "").lower()] for source in ("arxiv","openalex")}
    entities=inventory.get("entity") or []
    text=(str(paper.get("title") or "")+" "+str(paper.get("abstract") or "")).casefold()
    union_tokens=set().union(*(tokenize(phrase) for phrase in phrases)) if phrases else set()
    paper.update({
        "facet_query_ids":facet_ids,
        "matched_facets":matched,
        "facet_coverage":len(matched)/len(phrases) if phrases else 0.0,
        "arxiv_rank":min(source_ranks["arxiv"],default=None),
        "openalex_rank":min(source_ranks["openalex"],default=None),
        "source_count":len(sources),
        "variant_count":len(facet_ids),
        "exact_entity_match":bool(entities and any(entity.casefold() in text for entity in entities)),
        "title_coverage":len(title_tokens&union_tokens)/len(union_tokens) if union_tokens else 0.0,
        "abstract_coverage":len(abstract_tokens&union_tokens)/len(union_tokens) if union_tokens else 0.0,
        "topic_coverage":max((_coverage(all_tokens,phrase) for phrase in inventory.get("topic") or []),default=0.0),
        "existing_trace_score":float(paper.get("final_score") or 0.0),
        "existing_llm_score":paper.get("llm_relevance_score"),
    })


def same_paper(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return bool(match_identity(left,right) or match_identity(right,left))


def admission_decision(
    paper: dict[str, Any],
    inventory: dict[str, list[str]],
    *,
    mode: str,
    trace_threshold: float,
    baseline: Sequence[dict[str, Any]] = (),
) -> tuple[bool,str]:
    identity=paper_identity(paper)
    if not paper.get("title") or not (identity.title or identity.arxiv_ids or identity.dois or identity.openalex_ids):return False,"metadata_failure"
    if any(same_paper(paper,item) for item in baseline):return False,"duplicate"
    if not paper.get("abstract") and float(paper.get("title_coverage") or 0)<.25:return False,"missing_evidence"
    if float(paper.get("topic_coverage") or 0)<.25:return False,"query_drift"
    required=[*inventory.get("topic",[]),*inventory.get("method",[]),*inventory.get("task",[]),*inventory.get("dataset",[])]
    text_tokens=tokenize(f"{paper.get('title','')} {paper.get('abstract','')}")
    supported=sum(_coverage(text_tokens,phrase)>=.5 for phrase in required)
    entity_topic=bool(paper.get("exact_entity_match") and float(paper.get("topic_coverage") or 0)>=.5)
    a0=entity_topic or supported>=2
    if mode=="A0":return (True,"admitted_a0") if a0 else (False,"low_facet_coverage")
    dual=bool(int(paper.get("source_count") or 0)>=2 or int(paper.get("variant_count") or 0)>=2)
    trace=bool(a0 and float(paper.get("existing_trace_score") or 0)>=trace_threshold)
    if dual or trace:return True,"admitted_a1"
    return False,"single_weak_source" if a0 else "low_facet_coverage"


def admit_candidates(
    dsfr: dict[str, Any],original_query: str,understanding: dict[str,Any],*,mode:str,trace_threshold:float,baseline:Sequence[dict[str,Any]],
) -> tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    inventory=facet_inventory(original_query,understanding);admitted=[];audit=[]
    for candidate in dsfr.get("candidates") or []:
        accepted,reason=admission_decision(candidate,inventory,mode=mode,trace_threshold=trace_threshold,baseline=baseline)
        audit.append({"title":candidate.get("title"),"paper_id":candidate.get("openalex_id") or candidate.get("doi") or candidate.get("arxiv_id"),
                      "accepted":accepted,"reason":reason,"source_count":candidate.get("source_count"),"variant_count":candidate.get("variant_count"),
                      "facet_coverage":candidate.get("facet_coverage"),"trace_score":candidate.get("existing_trace_score")})
        if accepted:admitted.append(copy.deepcopy(candidate))
    return admitted,audit


def core_protected_merge(
    baseline_top20: Sequence[dict[str,Any]],admitted:Sequence[dict[str,Any]],understanding:dict[str,Any],*,limit:int=20,
) -> list[dict[str,Any]]:
    baseline=[copy.deepcopy(row) for row in baseline_top20];core=baseline[:2];tail=baseline[2:]
    new=[copy.deepcopy(row) for row in admitted if not any(same_paper(row,old) for old in baseline)]
    scored=score_candidates(tail+new,understanding)
    selected_tail=diversify(scored,limit=max(0,limit-len(core)),lambda_relevance=.88)
    result=core+selected_tail
    for rank,paper in enumerate(result,1):paper["candidate_rank"]=rank;paper["live_rank"]=rank
    return result


def gap_triggered(baseline_top20:Sequence[dict[str,Any]],original_query:str,understanding:dict[str,Any])->bool:
    inventory=facet_inventory(original_query,understanding)
    structured=sum(bool(inventory[name]) for name in ("topic","method","task","dataset","domain","entity"))
    if structured<3:return False
    high=sum(float(row.get("final_score") or row.get("trace_score") or 0)>=.55 and float(row.get("constraint_coverage_score") or 0)>=.45 for row in baseline_top20[:8])
    required=[*inventory.get("topic",[]),*inventory.get("method",[]),*inventory.get("task",[]),*inventory.get("dataset",[])]
    incomplete=any(max((_coverage(tokenize(f"{row.get('title','')} {row.get('abstract','')}"),phrase) for row in baseline_top20[:8]),default=0)<.5 for phrase in required)
    sources=" ".join(str(row.get("source_api") or "").lower() for row in baseline_top20)
    missing_base_source="arxiv" not in sources or "openalex" not in sources
    return high<2 or incomplete or missing_base_source


def api_budget(facet_queries:Sequence[FacetQuery],sources:Sequence[str])->int:
    return min(len(facet_queries),MAX_VARIANTS)*len(tuple(sources))
