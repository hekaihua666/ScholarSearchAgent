"""DeepSeek LLM client — query understanding, autonomous expansion, summarization.

Three LLM calls per search (cost-controlled):
  1. llm_understand_query  → structured sub-queries (replaces rule-based parser)
  2. llm_expand_queries    → autonomous 2nd-round queries + gap analysis (the Agent core)
  3. llm_build_summary     → summary + recommendation reasons + relevance flags (one batch)

Every function returns its token usage so the caller can report total cost.
Falls back to rule-based / mock on any error so the system stays functional
even when the API key is missing or the service is down.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

_API_KEY: str = os.environ.get("DEEPSEEK_API_KEY", "")
_BASE_URL = "https://api.deepseek.com/v1/chat/completions"
_MODEL = "deepseek-chat"


def agent_model_name() -> str:
    """Agent 模型单独配置，保留旧工作流的模型和冻结评测口径。"""
    return os.getenv("DEEPSEEK_AGENT_MODEL", "deepseek-flash").strip() or "deepseek-flash"


async def llm_tool_turn(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]], *, max_tokens: int = 1200,
) -> tuple[dict[str, Any], int]:
    """执行一次原生工具调用决策，保留 assistant 消息供应用配对回传。

    与旧 _chat 的文本返回不同，工具调用时 content 可以为空，动作位于
    tool_calls 中。函数只请求模型，实际论文搜索由后端白名单执行。
    """
    if not _available():
        raise RuntimeError("Agent 模型未配置")
    body = {
        "model": agent_model_name(), "messages": messages, "tools": tools,
        "tool_choice": "auto", "max_tokens": max_tokens, "temperature": 0.15,
        # 第一版使用非思考模式，控制延迟；仍保留响应中的 reasoning_content 以兼容协议。
        "thinking": {"type": "disabled"},
    }
    headers = {"Authorization": f"Bearer {_API_KEY}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=8.0)) as client:
        response = await client.post(_BASE_URL, json=body, headers=headers)
        response.raise_for_status()
        payload = response.json()
    choice = payload["choices"][0]
    if choice.get("finish_reason") in {"length", "content_filter", "insufficient_system_resource"}:
        raise ValueError("模型本次响应未完整完成")
    raw_message = choice["message"]
    if raw_message.get("role") != "assistant":
        raise ValueError("模型响应角色无效")
    # 只回传协议需要的字段，不把供应商附加调试数据带进后续请求。
    message = {key: raw_message[key] for key in ("role", "content", "tool_calls", "reasoning_content")
               if key in raw_message}
    if not message.get("tool_calls") and not message.get("content"):
        raise ValueError("模型返回空响应")
    return message, int((payload.get("usage") or {}).get("total_tokens", 0))


def _available() -> bool:
    return bool(_API_KEY)


def _strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) >= 2 else text
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


async def _chat(
    messages: list[dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> tuple[str, int]:
    """Return (content, total_tokens). Raises on HTTP error."""
    headers = {
        "Authorization": f"Bearer {_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": _MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=8.0)) as client:
        resp = await client.post(_BASE_URL, json=body, headers=headers)
        resp.raise_for_status()
        payload = resp.json()
        content = payload["choices"][0]["message"]["content"]
        tokens = int((payload.get("usage") or {}).get("total_tokens", 0))
        return content, tokens


# ── 1. Query understanding ─────────────────────────────────────────────────────

async def llm_understand_query(query: str) -> tuple[dict[str, Any] | None, int]:
    """Parse a complex academic query into structured fields + 5 search sub-queries.

    Returns (result_or_None, tokens_used). None on failure → caller falls back.
    """
    if not _available():
        return None, 0

    prompt = (
        "You are an academic paper search assistant. Parse the research query below "
        "and return **valid JSON only** (no markdown fences, no extra text).\n\n"
        f"Query: {query}\n\n"
        "Return a JSON object with exactly these fields:\n"
        '- "research_topic": concise English phrase (2–6 words) capturing the core topic\n'
        '- "method_constraints": methods/algorithms/models mentioned, comma-separated, or "not specified"\n'
        '- "task_constraints": tasks/applications mentioned, comma-separated, or "not specified"\n'
        '- "domain_constraints": research domains/fields, comma-separated, or "not specified"\n'
        '- "time_range": one of "last 3 years" | "last 5 years" | "last 10 years" | "not specified" | "YYYY-YYYY"\n'
        '- "sub_queries": array of exactly 5 **distinct English** search queries for OpenAlex/Semantic Scholar\n'
        '- "candidate_title_queries": array of 0–3 likely existing paper titles or distinctive title fragments. '
        "They are search hypotheses only and must be verified against an academic API before display.\n\n"
        "Sub-queries must be diverse and complementary:\n"
        "  [0] core topic keywords\n"
        "  [1] topic + specific method/model\n"
        "  [2] topic + application/task\n"
        "  [3] related concept or variant\n"
        "  [4] survey or benchmark"
    )

    try:
        content, tokens = await _chat([{"role": "user", "content": prompt}], max_tokens=650, temperature=0.15)
        data = json.loads(_strip_fence(content))
        if isinstance(data.get("sub_queries"), list) and len(data["sub_queries"]) >= 3:
            return data, tokens
        return None, tokens
    except Exception:
        return None, 0


# ── 2. Autonomous second-round expansion (the Agent core) ──────────────────────

async def llm_expand_queries(
    query: str,
    top_papers: list[dict[str, Any]],
) -> tuple[str, list[str], int]:
    """Analyze first-round results, identify coverage gaps, and autonomously
    generate targeted second-round search queries.

    This is the iterative-retrieval Agent step required by the competition:
    the LLM *reasons about what is missing* rather than mechanically expanding
    keywords.

    Returns (gap_analysis_zh, expanded_queries, tokens_used).
    On failure returns ("", [], 0) → caller falls back to rule-based expansion.
    """
    if not _available() or not top_papers:
        return "", [], 0

    papers_lines = "\n".join(
        f"{i + 1}. {p.get('title', 'Untitled')} ({p.get('year', '?')})"
        for i, p in enumerate(top_papers[:5])
    )

    prompt = (
        "You are an autonomous academic search agent running an iterative retrieval loop.\n"
        f"Original user query: {query}\n\n"
        f"First-round retrieved papers (top 5):\n{papers_lines}\n\n"
        "Analyze what aspects of the user's query are NOT yet well covered by these "
        "results (e.g. a specific method, sub-task, dataset, application, or recent work "
        "that is missing). Then design targeted follow-up searches to fill those gaps.\n\n"
        "Return **valid JSON only** (no markdown fences) with:\n"
        '- "gap_analysis": one or two **Chinese** sentences explaining what is missing '
        "and what the follow-up searches target\n"
        '- "expanded_queries": array of exactly 2 **distinct English** search queries '
        "that address the identified gaps (must differ from obvious repeats of the topic)"
    )

    try:
        content, tokens = await _chat([{"role": "user", "content": prompt}], max_tokens=320, temperature=0.4)
        data = json.loads(_strip_fence(content))
        queries = data.get("expanded_queries") or []
        gap = data.get("gap_analysis", "")
        queries = [q for q in queries if isinstance(q, str) and q.strip()][:2]
        if queries:
            return gap, queries, tokens
        return "", [], tokens
    except Exception:
        return "", [], 0


# ── 3. Summary + recommendation reasons + relevance flags (one batched call) ───

async def llm_build_summary(
    query: str,
    papers: list[dict[str, Any]],
) -> tuple[dict[str, Any], int]:
    """Generate a structured summary, per-paper recommendation reasons, AND
    per-paper relevance flags in a single call (cost control).

    Returns (result_or_empty, tokens_used). {} on failure → caller falls back.
    """
    if not _available() or not papers:
        return {}, 0

    papers_lines = "\n".join(
        f"{i + 1}. {p.get('title', 'Untitled')} ({p.get('year', '?')}) "
        f"— {(p.get('abstract') or 'No abstract available')[:650].strip()}"
        for i, p in enumerate(papers[:32])
    )

    prompt = (
        "You are an academic search assistant. Given the user query and retrieved papers, "
        "return **valid JSON only** (no markdown fences, no extra text).\n\n"
        f"User query: {query}\n\n"
        f"Retrieved papers:\n{papers_lines}\n\n"
        "Return a JSON object with:\n"
        '- "question_understanding": one Chinese sentence summarizing what the user wants\n'
        '- "retrieval_strategy": one Chinese sentence describing how the search was performed\n'
        '- "overall_summary": 2–3 Chinese sentences summarizing key findings and coverage\n'
        '- "recommendation_reasons": object mapping each supplied paper number '
        "to one concise Chinese sentence (35–65 Chinese characters). For EACH paper, name its "
        "specific method, task or application from the supplied title/abstract and explain which "
        "part of the user query it addresses. Do not repeat the query or reuse generic relevance "
        "and ranking claims across papers. Do not invent contributions, results, full-text evidence "
        "or representative status. If the abstract is missing or evidence is weak, state that "
        "limitation explicitly instead of claiming relevance.\n"
        '- "relevance_scores": object mapping every paper number to an integer 0–100. '
        "Judge strict satisfaction of the original query constraints, not general topical similarity.\n"
        '- "relevance_flags": object mapping every paper number to a boolean '
        "— true if the paper is genuinely relevant to the query, false if it is off-topic "
        "or low quality and should be filtered out"
    )

    try:
        content, tokens = await _chat([{"role": "user", "content": prompt}], max_tokens=4096, temperature=0.15)
        data = json.loads(_strip_fence(content))
        if "overall_summary" in data:
            return data, tokens
        return {}, tokens
    except Exception:
        return {}, 0
