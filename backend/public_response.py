"""网页响应白名单：内部评分、证据图和检索过程留在后台。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PublicPaper(BaseModel):
    """卡片、排序和引用导出所需的论文信息，不含算法调试字段。"""

    rank: int
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    abstract: str = ""
    cited_by_count: int = 0
    doi: str | None = None
    url: str | None = None
    source_api: str = "OpenAlex"


class PublicSearchResponse(BaseModel):
    query: str
    papers: list[PublicPaper]
    notices: list[str] = Field(default_factory=list)


def build_public_response(payload: dict[str, Any]) -> PublicSearchResponse:
    """先用后台评分确定顺序，再返回去重后的全量论文，不发送分数。

    保持原网页的标题+年份去重规则及稳定排序，避免精简响应影响论文
    数量、顺序和全量导出。白名单也适用于旧缓存，新增内部字段不会泄漏。
    """
    candidates = payload.get("ranked_candidates") or payload.get("papers") or []
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for paper in candidates:
        key = (str(paper.get("title") or ""), str(paper.get("year") or ""))
        unique.setdefault(key, paper)
    ordered = sorted(unique.values(), key=lambda paper: float(paper.get("final_score") or 0), reverse=True)
    papers = [
        PublicPaper.model_validate({
            **{key: paper[key] for key in PublicPaper.model_fields if paper.get(key) is not None},
            "rank": rank,
        })
        for rank, paper in enumerate(ordered, 1)
    ]

    # 只传固定的用户提示，内部异常文本、调用记录和查询式不离开后台。
    process = payload.get("search_process") or {}
    notices = []
    if process.get("fallback_queries"):
        notices.append("部分查询已切换到备用检索来源。")
    if (process.get("relational_evidence_graph") or {}).get("error"):
        notices.append("高级关系排序未完成，当前展示基础检索结果。")
    if process.get("frozen_demo"):
        notices.append("当前为离线演示，展示预先保存的论文结果，未执行实时检索。")
    elif process.get("search_strategy") == "agent":
        if (payload.get("metrics") or {}).get("from_cache"):
            notices.append("本次展示相同问题的历史检索结果。")
        # 仅将固定提示传到网页；模型结束文本、原始异常和执行轨迹留在后台。
        stop_reason = (process.get("agent") or {}).get("stop_reason")
        stop_notices = {
            "time_budget": "已达到本次检索时间上限，展示已获取的论文。",
            "tool_budget": "已达到本次检索调用上限，展示已获取的论文。",
            "model_budget": "已达到本次检索决策次数上限，展示已获取的论文。",
            "no_gain": "连续检索未增加新论文，本次检索已结束。",
            "model_error": "模型连接中断，展示中断前已获取的论文。",
            "model_timeout": "模型响应超时，展示已获取的论文。",
            "model_protocol_error": "模型未能继续检索，展示已获取的论文。",
            "tool_errors": "后续检索未能完成，展示已获取的论文。",
            "model_stopped_partial": "本次检索已结束，部分研究条件仍缺少充分证据。",
            "model_finished": "本次检索已结束，研究条件的完整覆盖仍需核对。",
            "no_results": "本次未检索到论文，可尝试更具体的研究条件。",
        }
        if stop_reason in stop_notices:
            notices.append(stop_notices[stop_reason])
    return PublicSearchResponse(query=payload["query"], papers=papers, notices=notices)
