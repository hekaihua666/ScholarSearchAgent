"""验证真实闭环的控制行为；模型和论文接口使用可控替身，不消耗外部额度。"""

import asyncio
import copy
import json
from datetime import datetime

import pytest

from backend.agent_retrieval import AgentBudget, retrieve_with_agent
from backend.agent_tools import AgentToolbox, validate_arguments
from backend.eval_runtime import EvaluationSession, evaluation_session
from backend.query_understanding import understand_query


def paper(title="Graph traffic prediction", **updates):
    """提供有真实字段形状的测试论文，不将其作为线上质量证据。"""
    return {"title": title, "abstract": "We propose graph neural networks for traffic prediction.",
            "year": 2024, "authors": ["Test Author"], "source_api": "OpenAlex", **updates}


def call(name, arguments, call_id="call_1"):
    return {"role": "assistant", "content": None, "tool_calls": [{
        "id": call_id, "type": "function", "function": {
            "name": name, "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
        },
    }]}


def finish(ids=None, uncovered=None):
    return call("finish_search", {"reason": "根据已取得的论文结束本次检索。",
                                  "evidence_ids": ids or ["P1"], "uncovered_constraints": uncovered or []}, "finish")


class ScriptedModel:
    """保存每次模型实际收到的消息，用来检查工具反馈是否进入下一次决策。"""

    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    async def __call__(self, messages, tools, **kwargs):
        self.requests.append(copy.deepcopy(messages))
        return next(self.replies), 40


class FakeToolbox:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    async def execute(self, name, arguments, papers_by_id, time_range):
        self.calls.append((name, arguments))
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        return copy.deepcopy(output)


def run(model, toolbox, **kwargs):
    query = "graph neural network traffic prediction research"
    understanding = understand_query(query)
    understanding["original_query"] = query
    return asyncio.run(retrieve_with_agent(query, understanding, toolbox, model_call=model, **kwargs))


def test_model_can_search_more_than_two_rounds_switch_tools_and_finish():
    model = ScriptedModel([
        call("search_papers", {"query": "graph traffic"}, "c1"),
        call("semantic_search", {"query": "travel time prediction"}, "c2"),
        call("expand_citations", {"paper_id": "P1", "direction": "references"}, "c3"),
        call("search_papers", {"query": "traffic benchmarks", "source": "arxiv"}, "c4"),
        finish(["P1", "P2"]),
    ])
    toolbox = FakeToolbox([
        [paper()], [paper("Travel time", doi="10.1000/travel")],
        [paper(doi="10.1000/traffic"), paper("Traffic foundation")], [paper("Traffic benchmarks")],
    ])
    result = run(model, toolbox)
    assert result.stop_reason == "completed"
    assert result.model_calls == 5 and result.executed_searches == 4
    assert len(result.papers) == 4 and result.raw_candidate_count == 5
    assert len({p["agent_paper_id"] for p in result.papers}) == 4
    assert next(p for p in result.papers if p["title"] == "Graph traffic prediction")["agent_paper_id"] == "P1"
    feedback = [message for message in model.requests[1] if message["role"] == "tool"]
    assert feedback[0]["tool_call_id"] == "c1"
    assert json.loads(feedback[0]["content"])["papers"][0]["paper_id"] == "P1"
    assert "We propose" in feedback[0]["content"]
    # 第五次决策仍知道第一次执行过的查询，裁剪对话不会丢失操作历史。
    state = json.loads(model.requests[-1][2]["content"].split("：", 1)[1])
    assert state["past_actions"][0]["arguments"]["query"] == "graph traffic"


def test_model_can_finish_after_one_search_and_report_uncovered_constraints():
    model = ScriptedModel([call("search_papers", {"query": "graph traffic"}),
                           finish(uncovered=["尚未取得行程时间预测的充分证据"])])
    result = run(model, FakeToolbox([[paper()]]))
    assert result.stop_reason == "model_stopped_partial"
    assert result.executed_searches == 1
    assert result.uncovered_constraints


@pytest.mark.parametrize("name,arguments", [
    ("delete_file", {"path": "anything"}),
    ("search_papers", {"query": "traffic", "limit": 1000}),
    ("search_papers", {"query": "traffic", "limit": "20"}),
    ("search_papers", {"query": "   "}),
    ("search_papers", {"query": "traffic", "unexpected": True}),
    ("search_papers", "{broken-json"),
])
def test_invalid_tools_and_arguments_do_not_execute(name, arguments):
    model = ScriptedModel([call(name, arguments), call("search_papers", {"query": "traffic"}, "valid"), finish()])
    toolbox = FakeToolbox([[paper()]])
    result = run(model, toolbox)
    assert result.stop_reason == "completed"
    assert len(toolbox.calls) == 1
    feedback = [m for m in model.requests[1] if m["role"] == "tool"]
    assert json.loads(feedback[0]["content"])["error"] == "tool_failed"


def test_tool_results_must_precede_completion_and_fabricated_evidence_is_rejected():
    model = ScriptedModel([
        {"role": "assistant", "content": "已经找到论文。"},
        call("search_papers", {"query": "traffic"}), finish(["P999"]), finish(),
    ])
    result = run(model, FakeToolbox([[paper()]]))
    assert result.stop_reason == "completed"
    assert result.model_calls == 4
    assert result.evidence_ids == ["P1"]
    assert result.trace[1]["error"] == "tool_failed"


def test_duplicate_query_with_different_limit_does_not_repeat_network_work():
    model = ScriptedModel([
        call("search_papers", {"query": "Graph traffic", "limit": 10}, "a"),
        call("search_papers", {"query": " graph   TRAFFIC ", "limit": 20}, "b"),
        finish(),
    ])
    toolbox = FakeToolbox([[paper()]])
    result = run(model, toolbox)
    assert result.stop_reason == "completed" and len(toolbox.calls) == 1
    assert result.trace[1]["error"] == "duplicate_action"


def test_tool_budget_is_enforced_inside_a_batch():
    batch = call("search_papers", {"query": "traffic"}, "a")
    batch["tool_calls"] += call("semantic_search", {"query": "travel"}, "b")["tool_calls"]
    toolbox = FakeToolbox([[paper()]])
    result = run(ScriptedModel([batch]), toolbox, budget=AgentBudget(max_tool_calls=1))
    assert result.stop_reason == "tool_budget" and len(toolbox.calls) == 1
    assert result.trace[-1]["error"] == "skipped"
    assert result.papers


def test_no_new_candidates_stops_without_claiming_full_coverage():
    model = ScriptedModel([call("search_papers", {"query": f"traffic {i}"}, str(i)) for i in range(4)])
    result = run(model, FakeToolbox([[paper()]] * 4))
    assert result.stop_reason == "no_gain"
    assert len(result.papers) == 1 and result.evidence_ids == []


def test_model_failure_preserves_candidates_and_does_not_leak_error_text():
    model = ScriptedModel([call("search_papers", {"query": "traffic"})])
    result = run(model, FakeToolbox([[paper()]]))
    assert result.stop_reason == "model_error" and result.papers
    assert result.model_calls == 2
    assert result.trace[-1] == {"action": "model", "status": "error", "error_type": "RuntimeError"}


def test_deadline_cancels_slow_tool_and_preserves_previous_results():
    class SlowToolbox(FakeToolbox):
        async def execute(self, *args):
            if self.calls:
                await asyncio.sleep(1)
            return await super().execute(*args)

    model = ScriptedModel([call("search_papers", {"query": "first"}),
                           call("semantic_search", {"query": "second"}, "slow")])
    result = run(model, SlowToolbox([[paper()], [paper("late")]]),
                 budget=AgentBudget(max_seconds=0.05, tool_timeout=1))
    assert result.stop_reason == "time_budget" and len(result.papers) == 1
    assert result.elapsed_seconds < 0.5


def test_native_final_response_is_supported_after_real_tool_use():
    model = ScriptedModel([call("search_papers", {"query": "traffic"}),
                           {"role": "assistant", "content": "本次检索结束。"}])
    result = run(model, FakeToolbox([[paper()]]))
    assert result.stop_reason == "model_finished" and result.papers
    assert result.uncovered_constraints


def test_failed_source_is_not_reported_as_a_successful_empty_search():
    # 失败请求不能算成成功的空结果，避免页面将来源故障解释为没有相关论文。
    model = ScriptedModel([call("search_papers", {"query": "traffic"}),
                           {"role": "assistant", "content": "数据源暂不可用。"}])
    result = run(model, FakeToolbox([RuntimeError("source failed")]))
    assert result.stop_reason == "tool_errors"
    assert result.successful_searches == 0 and not result.papers


def test_successful_empty_search_can_end_without_fabricating_papers():
    model = ScriptedModel([call("search_papers", {"query": "traffic"}),
                           call("finish_search", {"reason": "没有取得论文", "evidence_ids": [],
                                                  "uncovered_constraints": ["原始研究条件"]}, "empty")])
    result = run(model, FakeToolbox([[]]))
    assert result.stop_reason == "no_results"
    assert result.successful_searches == 1 and not result.papers


@pytest.mark.parametrize("metadata", [
    {"reason": "结束", "evidence_ids": ["P1"]},
    {"reason": "说明过长" * 200, "evidence_ids": ["P1"], "uncovered_constraints": []},
])
def test_invalid_finish_metadata_keeps_results_without_claiming_complete_coverage(metadata):
    model = ScriptedModel([call("search_papers", {"query": "traffic"}), call("finish_search", metadata, "end")])
    result = run(model, FakeToolbox([[paper()]]))
    assert result.stop_reason == "model_finished" and result.papers
    assert result.model_calls == 2
    assert result.evidence_ids == [] and result.uncovered_constraints
    assert result.trace[-1]["warning"] == "invalid_finish_metadata"


def test_multiple_native_calls_all_have_matching_tool_responses_before_next_decision():
    batch = call("search_papers", {"query": "traffic"}, "first")
    batch["tool_calls"] += call("semantic_search", {"query": "travel time"}, "second")["tool_calls"]
    model = ScriptedModel([batch, finish(["P1", "P2"])])
    result = run(model, FakeToolbox([[paper()], [paper("Travel time")]]))
    assert result.stop_reason == "completed"
    feedback = [message for message in model.requests[1] if message["role"] == "tool"]
    assert [message["tool_call_id"] for message in feedback] == ["first", "second"]
    assert all(json.loads(message["content"])["new_papers"] == 1 for message in feedback)


def test_tools_enforce_original_time_range_and_known_citation_ids(monkeypatch):
    async def search_stub(*args):
        return [paper("old", year=2019), paper("recent", year=2024), paper("unknown", year=None)]

    from backend import agent_tools
    monkeypatch.setattr(agent_tools, "search_openalex_semantic", search_stub)
    toolbox = AgentToolbox(search_stub, search_stub)
    papers = asyncio.run(toolbox.execute("semantic_search", {"query": "traffic", "limit": 20}, {}, "2023-2025"))
    assert [p["title"] for p in papers] == ["recent"]
    with pytest.raises(ValueError, match="unknown_paper_id"):
        asyncio.run(toolbox.execute("expand_citations", {
            "paper_id": "P999", "direction": "references", "limit": 20,
        }, {}, None))


def test_agent_model_and_tools_can_be_frozen_and_replayed_without_external_calls(tmp_path):
    path = tmp_path / "agent.json"
    cutoff = datetime(2025, 6, 1)
    session = EvaluationSession.create("agent", "q1", cutoff, "live", path)
    with evaluation_session(session):
        first = run(ScriptedModel([call("search_papers", {"query": "traffic"}), finish()]), FakeToolbox([[paper()]]))
    session.save()
    # 回放键不能包含实时倒计时；替身没有任何输出，若发起外部调用就会失败。
    replay = EvaluationSession.create("agent", "q1", cutoff, "replay", path)
    model, toolbox = ScriptedModel([]), FakeToolbox([])
    with evaluation_session(replay):
        second = run(model, toolbox)
    assert second.stop_reason == first.stop_reason == "completed"
    assert second.papers == first.papers
    assert model.requests == [] and toolbox.calls == []


def test_tool_schema_does_not_allow_unknown_fields():
    with pytest.raises(ValueError):
        validate_arguments("expand_citations", {"paper_id": "https://unknown.example/W1", "direction": "references"})
