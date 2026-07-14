from __future__ import annotations

import json

import httpx
import pytest

from app.schemas.llm import LLMFunctionSchema, LLMMessage, LLMRequest
from app.schemas.qwen import QwenConfig
from app.tools.qwen_client import QwenClientError, QwenLLMClient


def _build_client(handler) -> QwenLLMClient:
    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)
    return QwenLLMClient(
        config=QwenConfig(api_key="test-key", model_name="qwen-plus"),
        http_client=http_client,
    )


@pytest.mark.asyncio
async def test_qwen_client_parses_function_call_response():
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["model"] == "qwen-plus"
        assert payload["tools"][0]["function"]["name"] == "emit_task_plan"
        assert payload["tool_choice"]["function"]["name"] == "emit_task_plan"
        return httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-test",
                "model": "qwen-plus",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "emit_task_plan",
                                        "arguments": json.dumps({"goal": "test goal", "tasks": []}),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )

    client = _build_client(handler)
    response = await client.generate(
        LLMRequest(
            prompt="Plan the work",
            system_prompt="Return tool calls only",
            messages=[
                LLMMessage(role="system", content="Return tool calls only"),
                LLMMessage(role="user", content="Plan the work"),
            ],
            model_name="qwen-plus",
            function_schemas=[
                LLMFunctionSchema(
                    name="emit_task_plan",
                    description="Return a task plan",
                    parameters_schema={"type": "object"},
                    schema_name="TaskPlan",
                    schema_version="v1",
                )
            ],
            tool_choice="emit_task_plan",
            request_id="req_test",
            session_id="sess_test",
            trace_id="trace_test",
        )
    )

    assert response.function_call is not None
    assert response.function_call.tool_name == "emit_task_plan"
    assert response.function_call.arguments["goal"] == "test goal"
    assert response.usage is not None
    assert response.usage.total_tokens == 15


@pytest.mark.asyncio
async def test_qwen_client_rejects_streaming_with_function_calling():
    client = _build_client(lambda request: httpx.Response(status_code=200, json={}))

    with pytest.raises(QwenClientError):
        async for _ in client.stream(
            LLMRequest(
                prompt="Plan the work",
                system_prompt="Return tool calls only",
                messages=[
                    LLMMessage(role="system", content="Return tool calls only"),
                    LLMMessage(role="user", content="Plan the work"),
                ],
                function_schemas=[
                    LLMFunctionSchema(
                        name="emit_task_plan",
                        description="Return a task plan",
                        parameters_schema={"type": "object"},
                        schema_name="TaskPlan",
                        schema_version="v1",
                    )
                ],
                stream=True,
            )
        ):
            pass
