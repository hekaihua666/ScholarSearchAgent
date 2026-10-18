"""论文检索工具白名单：模型选动作，后端验证参数并执行已有检索函数。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from backend.eval_runtime import filter_for_cutoff
from backend.openalex_client import (
    _parse_year_filter, fetch_openalex_citing, fetch_openalex_works, search_openalex_semantic,
)

SearchFunction = Callable[[str, int, str | None], Awaitable[list[dict[str, Any]]]]
ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=600)]


class _Arguments(BaseModel):
    """拒绝未知字段和隐式类型转换，避免模型参数直接进入实际函数。"""

    model_config = ConfigDict(extra="forbid", strict=True)


class SearchArguments(_Arguments):
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1500)]
    source: Literal["auto", "arxiv"] = "auto"
    limit: int = Field(default=20, ge=1, le=30)


class SemanticArguments(_Arguments):
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1500)]
    limit: int = Field(default=25, ge=1, le=30)


class CitationArguments(_Arguments):
    paper_id: str = Field(pattern=r"^P[1-9][0-9]*$")
    direction: Literal["references", "citing"]
    limit: int = Field(default=20, ge=1, le=30)


class FinishArguments(_Arguments):
    reason: ShortText = Field(description="停止依据；工具失败、扩展为空等执行限制可在此说明，不等同于用户条件未满足。")
    evidence_ids: list[ShortText] = Field(max_length=20)
    uncovered_constraints: list[ShortText] = Field(
        max_length=12,
        description="仅列用户实际提出但缺少论文证据的要求；未指定的年份、数据集等不能算缺口。没有未覆盖要求时返回空数组。",
    )


ARGUMENT_MODELS = {
    "search_papers": SearchArguments,
    "semantic_search": SemanticArguments,
    "expand_citations": CitationArguments,
    "finish_search": FinishArguments,
}

_DESCRIPTIONS = {
    "search_papers": "按检索式搜索真实论文。auto 使用主检索及备用来源，arxiv 搜索预印本。根据缺口生成查询，不重复已经执行的查询。",
    "semantic_search": "根据自然语言描述搜索语义相关论文，适合关键词匹配不足时使用。",
    "expand_citations": "只围绕已经返回的 P 编号论文扩展参考文献或引用它的后续论文；模型不能自行编造论文编号。",
    "finish_search": "结束检索。说明停止依据、已有论文编号和仍未覆盖的条件。evidence_ids 只能引用已返回的 P 编号；不能将预算耗尽说成已找全。",
}

# 使用普通原生 tools 协议；不依赖 beta strict 模式，参数由本地 Pydantic 再校验。
TOOL_DEFINITIONS = [
    {"type": "function", "function": {
        "name": name, "description": _DESCRIPTIONS[name], "parameters": model.model_json_schema(),
    }}
    for name, model in ARGUMENT_MODELS.items()
]


def validate_arguments(name: str, arguments: Any) -> dict[str, Any]:
    """只允许工具白名单中的动作；绝不通过 eval 或任意属性访问执行模型文本。"""
    if name not in ARGUMENT_MODELS:
        raise ValueError("unknown_tool")
    return ARGUMENT_MODELS[name].model_validate(arguments).model_dump()


class AgentToolbox:
    """适配已有检索接口；查询时间限制始终来自原始需求，不由工具参数放宽。"""

    def __init__(self, lexical_search: SearchFunction, arxiv_search: SearchFunction):
        self.lexical_search = lexical_search
        self.arxiv_search = arxiv_search

    async def execute(
        self, name: str, arguments: dict[str, Any], papers_by_id: dict[str, dict[str, Any]],
        time_range: str | None,
    ) -> list[dict[str, Any]]:
        """每次执行一个有界检索动作；finish_search 由循环调度器处理。"""
        if name == "search_papers":
            search = self.arxiv_search if arguments["source"] == "arxiv" else self.lexical_search
            papers = await search(arguments["query"], arguments["limit"], time_range)
        elif name == "semantic_search":
            papers = await search_openalex_semantic(arguments["query"], arguments["limit"])
        elif name == "expand_citations":
            # 只能扩展后端实际收集的论文，防止从模型自造 ID 开始引文爬取。
            seed = papers_by_id.get(arguments["paper_id"])
            if seed is None:
                raise ValueError("unknown_paper_id")
            source_query = f"citation:{arguments['direction']}:{arguments['paper_id']}"
            if arguments["direction"] == "references":
                reference_ids = list(seed.get("referenced_works") or [])[:arguments["limit"]]
                papers = await fetch_openalex_works(reference_ids, source_query, arguments["limit"])
            elif seed.get("openalex_id"):
                papers = await fetch_openalex_citing(seed["openalex_id"], source_query, arguments["limit"])
            else:
                raise ValueError("citation_identifier_unavailable")
        else:
            raise ValueError("unknown_tool")

        # 语义/引文来源不一定在远端支持年份过滤，统一在返回后再施加原始时间限制。
        papers = filter_for_cutoff(papers, f"agent:{name}")
        year_filter = _parse_year_filter(time_range)
        if year_filter:
            first_year, last_year = (int(value) for value in year_filter.split("-"))
            papers = [paper for paper in papers
                      if paper.get("year") is not None and first_year <= int(paper["year"]) <= last_year]
        return [dict(paper) for paper in papers[:arguments["limit"]] if str(paper.get("title") or "").strip()]
