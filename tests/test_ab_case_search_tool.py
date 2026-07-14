from __future__ import annotations

import pytest

from app.prompts.supervisor import build_supervisor_prompt
from app.prompts.task_planner import ToolDefinition, build_task_planner_prompt
from app.graph.runtime_graph import _looks_like_ab_case_query
from app.tools.ab_case_search import ABCaseSearchTool


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


class _RecordingAsyncClient:
    last_url: str | None = None
    last_request_json: dict[str, object] | None = None

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self) -> "_RecordingAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, url: str, json: dict[str, object]) -> _FakeResponse:
        type(self).last_url = url
        type(self).last_request_json = dict(json)
        query_text = str(json.get("query") or "")
        if json.get("event_text") or json.get("reason_text"):
            query_text = f"事件：\n{json.get('event_text') or ''}\n缘由：\n{json.get('reason_text') or ''}"
        return _FakeResponse(
            {
                "collection": "ab_case_score_examples_bge_m31",
                "query_text": query_text,
                "total": 1,
                "returned": 1,
                "top_score": 0.86,
                "results": [
                    {
                        "example_id": 123,
                        "similarity": 0.86,
                        "doc_id": "case_doc_001",
                        "source_file": "B类案例.docx",
                        "case_class": str(json.get("case_class") or "B"),
                        "case_no": "B-001",
                        "event_text": "客户反馈设备运行过程中偶发停机。",
                        "reason_text": "传感器接线松动。",
                        "score_reason": "异常影响客户使用且原因可预防。",
                        "score_delta": -1,
                        "score_text": "扣1分",
                        "evidence_text": "原文证据片段...",
                        "needs_review": False,
                        "vector_text": None,
                    }
                ],
            }
        )


def _build_ab_case_planner_tools() -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="ab_case_search_tool",
            description="检索 A/B 案例评分样例；A案例表示好事奖励，B案例表示坏事惩罚。",
            input_schema=ABCaseSearchTool().get_input_schema(),
            output_schema=ABCaseSearchTool().get_output_schema(),
            supported_task_types=["ab_case_search"],
            default_task_type="ab_case_search",
            supported_tags=["ab_case", "case_search", "rag", "business"],
        ),
        ToolDefinition(
            name="text_generate_tool",
            description="Generate a final answer from prompt and optional context.",
            input_schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "context": {"type": "string"},
                    "style": {"type": "string"},
                    "audience": {"type": "string"},
                },
            },
            output_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            supported_task_types=["text_generation"],
            default_task_type="text_generation",
            supported_tags=["llm", "generation", "text"],
        ),
    ]


@pytest.mark.asyncio
async def test_ab_case_search_tool_posts_to_monthly_cases_search(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.tools.ab_case_search.httpx.AsyncClient", _RecordingAsyncClient)
    tool = ABCaseSearchTool()

    result = await tool.arun(
        {
            "query": "这个问题有没有类似 A/B 案例，可以怎么评分？",
            "top_k": "3",
            "threshold": "0.6",
            "rag_base_url": "http://example.com",
        }
    )

    assert result.success is True
    assert _RecordingAsyncClient.last_url == "http://example.com/monthly/cases/search"
    assert _RecordingAsyncClient.last_request_json == {
        "query": "这个问题有没有类似 A/B 案例，可以怎么评分？",
        "top_k": 3,
        "min_score": 0.6,
    }
    assert result.output is not None
    assert result.output["query"] == "这个问题有没有类似 A/B 案例，可以怎么评分？"
    assert result.output["query_text"] == "这个问题有没有类似 A/B 案例，可以怎么评分？"
    assert result.output["collection"] == "ab_case_score_examples_bge_m31"
    assert result.output["total"] == 1
    assert result.output["returned"] == 1
    assert result.output["top_score"] == 0.86
    assert result.output["top_similarity"] == 0.86
    assert result.output["min_score"] == 0.6
    assert result.output["low_relevance"] is False
    assert "123" in result.output["case_context_text"]
    assert "event_text" in result.output["case_context_text"]


@pytest.mark.asyncio
async def test_ab_case_search_tool_sends_contract_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.tools.ab_case_search.httpx.AsyncClient", _RecordingAsyncClient)
    tool = ABCaseSearchTool()

    result = await tool.arun(
        {
            "query": "query 应该被后端忽略",
            "event_text": "项目交付延期，客户验收未按计划完成",
            "reason_text": "负责人未及时协调资源，风险预警不足",
            "case_class": "b",
            "top_k": "80",
            "include_review": "false",
            "include_formula_score": False,
            "include_evidence": True,
            "include_vector_text": "false",
            "min_score": "0.35",
            "rag_base_url": "http://example.com/",
        }
    )

    assert result.success is True
    assert _RecordingAsyncClient.last_url == "http://example.com/monthly/cases/search"
    assert _RecordingAsyncClient.last_request_json == {
        "top_k": 50,
        "query": "query 应该被后端忽略",
        "event_text": "项目交付延期，客户验收未按计划完成",
        "reason_text": "负责人未及时协调资源，风险预警不足",
        "case_class": "B",
        "min_score": 0.35,
        "include_review": False,
        "include_formula_score": False,
        "include_evidence": True,
        "include_vector_text": False,
    }
    assert result.output is not None
    assert result.output["query_text"] == "事件：\n项目交付延期，客户验收未按计划完成\n缘由：\n负责人未及时协调资源，风险预警不足"
    assert result.output["results"][0]["case_class"] == "B"


def test_supervisor_prompt_guides_ab_case_requests_to_complex_task() -> None:
    prompt = build_supervisor_prompt(
        user_input="这个异常有没有类似 A/B 案例可以参考评分？",
        context_summary="None",
    )

    assert "A/B 案例" in prompt.system_prompt
    assert "A案例表示好事奖励" in prompt.system_prompt
    assert "B案例表示坏事惩罚" in prompt.system_prompt
    assert "COMPLEX_TASK" in prompt.system_prompt
    assert "按案例打分" in prompt.user_prompt


def test_planner_prompt_includes_ab_case_guidance_and_example() -> None:
    prompt = build_task_planner_prompt(
        user_input="这个异常有没有类似 A/B 案例可以参考评分？",
        tools=_build_ab_case_planner_tools(),
        context_summary="None",
        planning_timestamp="2026-07-09T00:00:00+00:00",
    )

    assert "ab_case_search_tool" in prompt.user_prompt
    assert "POST /monthly/cases/search" in prompt.user_prompt
    assert "A案例是好事奖励" in prompt.user_prompt
    assert "B案例是坏事惩罚" in prompt.user_prompt
    assert "{{ab_case_results.case_context_text}}" in prompt.user_prompt
    assert '"tool": "ab_case_search_tool"' in prompt.user_prompt
    assert '"output_key": "ab_case_results"' in prompt.user_prompt
    assert "ab_case_search_tool -> text_generate_tool" in prompt.system_prompt
    assert "A案例表示好事奖励" in prompt.system_prompt


def test_runtime_ab_case_heuristic_recognizes_reward_and_punishment_cases() -> None:
    assert _looks_like_ab_case_query("这个好事应该算A案例奖励吗？") is True
    assert _looks_like_ab_case_query("这个问题算B案例惩罚还是普通问题？") is True
    assert _looks_like_ab_case_query("查一下有没有类似奖励案例可以参考") is True
