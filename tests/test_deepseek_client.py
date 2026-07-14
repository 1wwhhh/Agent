from __future__ import annotations

import json

import httpx
import pytest

from app.schemas.deepseek import DeepSeekConfig
from app.schemas.llm import LLMFunctionSchema, LLMMessage, LLMRequest
from app.tools.deepseek_client import DeepSeekLLMClient


def _build_client(handler) -> DeepSeekLLMClient:
    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)
    return DeepSeekLLMClient(
        config=DeepSeekConfig(api_key="test-key", model_name="deepseek-v4-flash"),
        http_client=http_client,
    )


@pytest.mark.asyncio
async def test_deepseek_client_parses_function_call_response():
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["model"] == "deepseek-v4-flash"
        assert payload["tools"][0]["function"]["name"] == "emit_task_plan"
        assert payload["tool_choice"]["function"]["name"] == "emit_task_plan"
        return httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-test",
                "model": "deepseek-v4-flash",
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
            model_name="deepseek-v4-flash",
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
async def test_deepseek_client_uses_async_response_close(monkeypatch):
    close_called = False
    aclose_called = False

    async def handler(request: httpx.Request) -> httpx.Response:
        response = httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-test",
                "model": "deepseek-v4-flash",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "hello",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )
        original_aclose = response.aclose

        def fail_close():
            nonlocal close_called
            close_called = True
            raise AssertionError("sync close should not be called for async responses")

        async def track_aclose():
            nonlocal aclose_called
            aclose_called = True
            return await original_aclose()

        monkeypatch.setattr(response, "close", fail_close)
        monkeypatch.setattr(response, "aclose", track_aclose)
        return response

    client = _build_client(handler)
    response = await client.generate(
        LLMRequest(
            prompt="Say hello",
            system_prompt="Return plain text",
            messages=[
                LLMMessage(role="system", content="Return plain text"),
                LLMMessage(role="user", content="Say hello"),
            ],
            model_name="deepseek-v4-flash",
        )
    )

    assert response.text == "hello"
    assert close_called is False
    assert aclose_called is True


def test_deepseek_adapter_normalizes_tool_choice_values():
    client = DeepSeekLLMClient(
        config=DeepSeekConfig(api_key="test-key", model_name="deepseek-v4-flash"),
    )
    adapter = client._provider_states[0].adapter

    dict_payload = adapter.build_payload(
        request=LLMRequest(
            prompt="Plan the work",
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
        ),
        stream=False,
    )
    none_payload = adapter.build_payload(
        request=LLMRequest(
            prompt="Plan the work",
            function_schemas=[
                LLMFunctionSchema(
                    name="emit_task_plan",
                    description="Return a task plan",
                    parameters_schema={"type": "object"},
                    schema_name="TaskPlan",
                    schema_version="v1",
                )
            ],
        ),
        stream=False,
    )
    required_payload = adapter.build_payload(
        request=LLMRequest(
            prompt="Plan the work",
            function_schemas=[
                LLMFunctionSchema(
                    name="emit_task_plan",
                    description="Return a task plan",
                    parameters_schema={"type": "object"},
                    schema_name="TaskPlan",
                    schema_version="v1",
                )
            ],
        ).model_copy(update={"tool_choice": "required"}),
        stream=False,
    )

    assert dict_payload["tool_choice"]["function"]["name"] == "emit_task_plan"
    assert none_payload["tool_choice"] == "auto"
    assert required_payload["tool_choice"] == "required"
    assert dict_payload["thinking"] == {"type": "disabled"}
    assert none_payload["thinking"] == {"type": "disabled"}
    assert required_payload["thinking"] == {"type": "disabled"}
    assert "tools" in dict_payload
    assert "tools" in none_payload
    assert "tools" in required_payload


def test_deepseek_adapter_does_not_disable_thinking_for_plain_text_requests():
    client = DeepSeekLLMClient(
        config=DeepSeekConfig(api_key="test-key", model_name="deepseek-v4-flash"),
    )
    adapter = client._provider_states[0].adapter

    payload = adapter.build_payload(request=LLMRequest(prompt="Say hello"), stream=False)

    assert "thinking" not in payload


@pytest.mark.asyncio
async def test_deepseek_client_streams_text_chunks():
    async def handler(request: httpx.Request) -> httpx.Response:
        stream = "\n".join(
            [
                "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}",
                "data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}",
                "data: [DONE]",
            ]
        )
        return httpx.Response(status_code=200, content=stream.encode("utf-8"))

    client = _build_client(handler)
    chunks = []
    async for chunk in client.stream(
        LLMRequest(
            prompt="Say hello",
            system_prompt="Stream plain text",
            messages=[
                LLMMessage(role="system", content="Stream plain text"),
                LLMMessage(role="user", content="Say hello"),
            ],
            stream=True,
        )
    ):
        chunks.append(chunk.delta_text)

    assert "".join(chunks).replace(" ", "") == "Helloworld"
