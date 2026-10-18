"""验证原生工具协议、Agent API 分流，以及网页只接收真实论文与固定提示。"""

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from backend import agent_retrieval, llm_client, main
from backend.agent_tools import TOOL_DEFINITIONS
from backend.public_response import build_public_response


@pytest.fixture
def isolated_agent(monkeypatch):
    """隔离本机离线开关和历史缓存，确保测试实际执行新闭环。"""
    monkeypatch.delenv("SCHOLAR_FROZEN_DEMO", raising=False)
    monkeypatch.setattr(main.response_cache, "load", lambda key: None)
    monkeypatch.setattr(main.response_cache, "save", lambda *args: None)


def test_transport_sends_tools_and_preserves_native_assistant_message(monkeypatch):
    requests = []
    assistant = {"role": "assistant", "content": None, "tool_calls": [{
        "id": "native_call", "type": "function",
        "function": {"name": "search_papers", "arguments": '{"query":"RAG research"}'},
    }], "reasoning_content": ""}

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": assistant, "finish_reason": "tool_calls"}],
                                         "usage": {"total_tokens": 120}})

    original_client = httpx.AsyncClient
    monkeypatch.setattr(llm_client.httpx, "AsyncClient", lambda **kwargs: original_client(
        transport=httpx.MockTransport(handler), **kwargs))
    monkeypatch.setattr(llm_client, "_API_KEY", "test-placeholder")
    monkeypatch.setenv("DEEPSEEK_AGENT_MODEL", "configured-agent-model")
    message, tokens = asyncio.run(llm_client.llm_tool_turn(
        [{"role": "user", "content": "RAG research"}], TOOL_DEFINITIONS))
    assert message == assistant and tokens == 120
    assert requests[0]["model"] == "configured-agent-model"
    assert requests[0]["tools"] == TOOL_DEFINITIONS
    assert requests[0]["tool_choice"] == "auto"


def test_transport_rejects_truncated_response(monkeypatch):
    original_client = httpx.AsyncClient
    monkeypatch.setattr(llm_client.httpx, "AsyncClient", lambda **kwargs: original_client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"choices": [{
            "message": {"role": "assistant", "content": "unfinished"}, "finish_reason": "length",
        }]})), **kwargs))
    monkeypatch.setattr(llm_client, "_API_KEY", "test-placeholder")
    with pytest.raises(ValueError, match="未完整完成"):
        asyncio.run(llm_client.llm_tool_turn([], TOOL_DEFINITIONS))


def test_agent_endpoint_executes_native_loop_and_keeps_private_state_out_of_page(isolated_agent, monkeypatch):
    replies = iter([
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "search", "type": "function", "function": {
                "name": "search_papers", "arguments": '{"query":"RAG research"}',
            },
        }]},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "finish", "type": "function", "function": {
                "name": "finish_search", "arguments": json.dumps({
                    "reason": "internal explanation", "evidence_ids": ["P1"], "uncovered_constraints": [],
                }),
            },
        }]},
    ])

    async def model(*args, **kwargs):
        return next(replies), 50

    async def lexical(*args):
        return [{"title": "Retrieval Augmented Generation", "year": 2024,
                 "abstract": "RAG research for question answering.", "source_api": "OpenAlex"}], "OpenAlex"

    monkeypatch.setattr(agent_retrieval, "llm_tool_turn", model)
    monkeypatch.setattr(main, "_search_one", lexical)
    with TestClient(main.app) as client:
        response = client.post("/api/search", json={"query": "RAG 研究论文", "search_strategy": "agent"})
    assert response.status_code == 200
    assert response.json()["papers"][0]["title"] == "Retrieval Augmented Generation"
    assert response.json()["notices"] == []
    assert "internal explanation" not in response.text and "agent_paper_id" not in response.text
    assert set(response.json()) == {"query", "papers", "notices"}


def test_agent_missing_model_is_clear_failure_without_silent_workflow_fallback(isolated_agent, monkeypatch):
    monkeypatch.setattr(llm_client, "_API_KEY", "")
    with TestClient(main.app) as client:
        response = client.post("/api/search", json={"query": "RAG 研究论文", "search_strategy": "agent"})
    assert response.status_code == 503
    assert "Agent" in response.json()["detail"]


def test_agent_cache_namespace_is_separate_from_legacy_workflow(isolated_agent, monkeypatch):
    cache_keys = []

    def load(key):
        cache_keys.append(key)
        return None

    async def failed_model(*args, **kwargs):
        raise RuntimeError("private provider text")

    monkeypatch.setattr(main.response_cache, "load", load)
    monkeypatch.setattr(agent_retrieval, "llm_tool_turn", failed_model)
    with TestClient(main.app) as client:
        response = client.post("/api/search", json={"query": "RAG 研究论文", "search_strategy": "agent"})
    assert response.status_code == 503
    assert cache_keys[0].startswith("[agent-v2|")
    assert "private provider text" not in response.text


@pytest.mark.parametrize("reason", ["time_budget", "model_error", "tool_budget", "model_stopped_partial"])
def test_partial_results_use_fixed_user_notice_without_internal_error(reason):
    public = build_public_response({"query": "RAG", "papers": [{"title": "Paper"}], "search_process": {
        "search_strategy": "agent", "agent": {"stop_reason": reason, "finish_reason": "private finish text"},
    }})
    assert public.papers and public.notices
    assert "private finish text" not in str(public.model_dump())


def test_frozen_demo_remains_offline_and_identifies_itself_for_agent_requests(monkeypatch):
    monkeypatch.setenv("SCHOLAR_FROZEN_DEMO", "1")
    with TestClient(main.app) as client:
        response = client.post("/api/search", json={"query": "RAG 论文", "search_strategy": "agent"})
    assert response.status_code == 200
    assert any("离线演示" in notice for notice in response.json()["notices"])


def test_legacy_request_defaults_remain_compatible():
    assert main.SearchRequest(query="RAG").search_strategy == "workflow"
    with pytest.raises(ValueError):
        main.SearchRequest(query="RAG", search_strategy="unknown")
