from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import json
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel

from app.adapters.base import ModelAdapter
from app.agents import SupervisorAgent
from app.graph import GraphRuntimeDependencies, LangGraphState, build_langgraph_runtime, coerce_langgraph_state
from app.llm.client import LLMClient as NewLLMClient
from app.llm.function_calling import FunctionCallingAdapter, FunctionCallingAdapterError
from app.planner import LLMTaskPlanner
from app.llm.structured import invoke_structured_output, parse_json_output
from app.router import TaskRouter
from app.router.capability import capability_from_tool
from app.schemas.llm import LLMFunctionCall, LLMFunctionSchema, LLMRequest, LLMResponse, LLMResponseChunk
from app.schemas.model import ModelConfig, ModelProvider
from app.tools.llm_reason import LLMReasonTool
from app.tools.text_generate import TextGenerateTool
from app.tools.llm_client import LLMClient as CompatLLMClient
from app.tools.llm_client import (
    CircuitBreakerOpenError,
    CircuitBreakerConfig,
    LLMAuthenticationError,
    LLMInvalidResponseError,
    LLMProviderRetryableError,
    LLMProviderUnavailableError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from tests.support.models import PlannerScenario, RuntimeTestInput, TaskSpec
from tests.support.runtime_runner import run_runtime


class StructuredPayload(BaseModel):
    text: str


class FunctionPayload(BaseModel):
    text: str


class StubStandaloneClient(CompatLLMClient):
    def __init__(self, *, outcomes: list[object], **kwargs) -> None:
        super().__init__(**kwargs)
        self.outcomes = list(outcomes)
        self.calls = 0

    async def _generate(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, LLMResponse):
            return outcome
        return LLMResponse(text=str(outcome), model_name=self.model_name, raw_response={"source": "stub"})

    async def _stream(self, request: LLMRequest) -> AsyncIterator[LLMResponseChunk]:
        raise NotImplementedError


class StubAdapter(ModelAdapter):
    def __init__(self, *, provider_name: str, outcomes: list[object], **kwargs) -> None:
        super().__init__(**kwargs)
        self._provider_name = provider_name
        self.outcomes = list(outcomes)
        self.calls = 0

    @property
    def provider_name(self) -> str:
        return self._provider_name

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class RuntimeStructuredClient(CompatLLMClient):
    async def _generate(self, request: LLMRequest) -> LLMResponse:
        function_name = request.tool_choice or (request.function_schemas[0].name if request.function_schemas else None)
        if function_name == "route_user_request":
            payload = {
                "route": "SIMPLE_TASK",
                "complexity": "simple",
                "needs_planning": False,
                "reason": "test route",
            }
        elif function_name == "emit_text_generation_output":
            payload = {
                "text": "generated text",
                "audience": "general",
                "style": "plain",
            }
        elif function_name == "emit_reasoning_output":
            payload = {
                "text": "reasoned text",
                "summary": "reasoned summary",
                "key_points": ["point"],
            }
        elif function_name == "emit_task_plan":
            payload = {
                "goal": request.prompt or "goal",
                "tasks": [],
            }
        else:
            payload = {"text": "fallback"}

        function_schema = request.function_schemas[0] if request.function_schemas else None
        return LLMResponse(
            text=json.dumps({"tool_name": function_name, "arguments": payload}),
            model_name=self.model_name or "runtime-structured-client",
            model_version=self.model_version,
            request_id=request.request_id,
            session_id=request.session_id,
            trace_id=request.trace_id,
            prompt_name=request.prompt_name,
            prompt_version=request.prompt_version,
            function_call=LLMFunctionCall(
                tool_name=function_name or "unknown",
                arguments=payload,
                schema_name=function_schema.schema_name if function_schema is not None else None,
                schema_version=function_schema.schema_version if function_schema is not None else None,
            ),
            raw_response={"provider": "runtime_structured_client"},
        )


def _build_config(*, provider: ModelProvider = ModelProvider.DEEPSEEK, max_retries: int = 0) -> ModelConfig:
    return ModelConfig(
        provider=provider,
        api_key="test-key",
        base_url="https://example.com",
        model_name=f"{provider.value}-test-model",
        timeout_seconds=30,
        max_retries=max_retries,
    )


def _build_rate_limit_error() -> httpx.HTTPStatusError:
    response = httpx.Response(status_code=429, request=httpx.Request("POST", "https://example.com"))
    return httpx.HTTPStatusError("rate limited", request=response.request, response=response)


def _build_auth_error() -> httpx.HTTPStatusError:
    response = httpx.Response(status_code=401, request=httpx.Request("POST", "https://example.com"))
    return httpx.HTTPStatusError("unauthorized", request=response.request, response=response)


def test_model_adapter_headers_are_ascii_safe_for_non_ascii_trace_values() -> None:
    adapter = StubAdapter(provider_name="deepseek", outcomes=[], config=_build_config())

    headers = adapter.build_headers(
        request=LLMRequest(
            prompt="hello",
            request_id="req_中文",
            session_id="sess_中文",
            trace_id="trace:dept_plan_owner_group_evidence:刘宝莹\nbad",
        )
    )

    for header_name in ("X-Trace-Id", "X-Request-Id", "X-Session-Id"):
        headers[header_name].encode("ascii")
        assert "\n" not in headers[header_name]
        assert "\r" not in headers[header_name]
    assert "刘宝莹" not in headers["X-Trace-Id"]
    assert headers["X-Trace-Id"].startswith("trace:dept_plan_owner_group_evidence:")
    assert len(headers["X-Trace-Id"]) <= 512


@pytest.mark.asyncio
async def test_llm_client_normal_call_finalizes_request_metadata() -> None:
    records: list[dict[str, object]] = []
    client = StubStandaloneClient(
        outcomes=[LLMResponse(text="hello", raw_response={"source": "standalone"})],
        model_name="stub-model",
        llm_call_recorder=records.append,
    )

    response = await client.generate(
        LLMRequest(
            prompt="hello",
            request_id="req_1",
            session_id="sess_1",
            trace_id="trace_1",
            prompt_name="prompt",
            prompt_version="v1",
            metadata={"operation": "tool"},
        )
    )

    assert response.text == "hello"
    assert response.request_id == "req_1"
    assert response.session_id == "sess_1"
    assert response.trace_id == "trace_1"
    assert records[0]["operation"] == "tool"
    assert records[0]["success"] is True


@pytest.mark.asyncio
async def test_llm_client_classifies_timeout() -> None:
    client = StubStandaloneClient(outcomes=[asyncio.TimeoutError()], model_name="stub-model")

    with pytest.raises(LLMTimeoutError):
        await client.generate(LLMRequest(prompt="hello"))

    assert client.calls == 1


@pytest.mark.asyncio
async def test_llm_client_classifies_rate_limit() -> None:
    client = StubStandaloneClient(outcomes=[_build_rate_limit_error()], model_name="stub-model")

    with pytest.raises(LLMRateLimitError):
        await client.generate(LLMRequest(prompt="hello"))


@pytest.mark.asyncio
async def test_llm_client_classifies_provider_unavailable() -> None:
    client = StubStandaloneClient(
        outcomes=[httpx.ConnectError("connection failed", request=httpx.Request("POST", "https://example.com"))],
        model_name="stub-model",
    )

    with pytest.raises(LLMProviderUnavailableError):
        await client.generate(LLMRequest(prompt="hello"))


@pytest.mark.asyncio
async def test_llm_client_classifies_invalid_response() -> None:
    client = StubStandaloneClient(outcomes=[ValueError("bad response")], model_name="stub-model")

    with pytest.raises(LLMInvalidResponseError):
        await client.generate(LLMRequest(prompt="hello"))


@pytest.mark.asyncio
async def test_llm_client_retries_retryable_error() -> None:
    client = StubStandaloneClient(
        outcomes=[asyncio.TimeoutError(), LLMResponse(text="ok")],
        model_name="stub-model",
        max_retries=1,
    )

    response = await client.generate(LLMRequest(prompt="hello"))

    assert response.text == "ok"
    assert client.calls == 2


@pytest.mark.asyncio
async def test_llm_client_does_not_retry_non_retryable_error() -> None:
    client = StubStandaloneClient(
        outcomes=[_build_auth_error()],
        model_name="stub-model",
        max_retries=3,
    )

    with pytest.raises(LLMAuthenticationError):
        await client.generate(LLMRequest(prompt="hello"))

    assert client.calls == 1


@pytest.mark.asyncio
async def test_llm_client_fallbacks_on_rate_limit_error() -> None:
    primary = StubAdapter(
        provider_name="primary",
        outcomes=[_build_rate_limit_error()],
        config=_build_config(provider=ModelProvider.DEEPSEEK, max_retries=0),
    )
    secondary = StubAdapter(
        provider_name="secondary",
        outcomes=[LLMResponse(text="secondary ok", raw_response={"provider": "secondary"})],
        config=_build_config(provider=ModelProvider.QWEN, max_retries=0),
    )
    client = CompatLLMClient(adapters=[primary, secondary], max_retries=0)

    response = await client.generate(LLMRequest(prompt="hello", request_id="req_fallback"))

    assert response.text == "secondary ok"
    assert response.request_id == "req_fallback"
    assert response.raw_response["selected_provider"] == "secondary"
    assert response.raw_response["attempted_providers"] == ["primary", "secondary"]
    assert primary.calls == 1
    assert secondary.calls == 1
    assert response.raw_response["selected_provider"] == "secondary"


@pytest.mark.asyncio
async def test_llm_client_fallbacks_on_timeout_error() -> None:
    primary = StubAdapter(
        provider_name="primary",
        outcomes=[LLMTimeoutError("timed out", provider="primary", model="p1", operation="tool")],
        config=_build_config(provider=ModelProvider.DEEPSEEK, max_retries=0),
    )
    secondary = StubAdapter(
        provider_name="secondary",
        outcomes=[LLMResponse(text="secondary ok", raw_response={"provider": "secondary"})],
        config=_build_config(provider=ModelProvider.QWEN, max_retries=0),
    )
    client = CompatLLMClient(adapters=[primary, secondary], max_retries=0)

    response = await client.generate(LLMRequest(prompt="hello"))

    assert response.text == "secondary ok"
    assert response.raw_response["selected_provider"] == "secondary"
    assert primary.calls == 1
    assert secondary.calls == 1


@pytest.mark.asyncio
async def test_llm_client_fallbacks_on_provider_unavailable_error() -> None:
    primary = StubAdapter(
        provider_name="primary",
        outcomes=[LLMProviderUnavailableError("down", provider="primary", model="p1", operation="tool")],
        config=_build_config(provider=ModelProvider.DEEPSEEK, max_retries=0),
    )
    secondary = StubAdapter(
        provider_name="secondary",
        outcomes=[LLMResponse(text="secondary ok", raw_response={"provider": "secondary"})],
        config=_build_config(provider=ModelProvider.QWEN, max_retries=0),
    )
    client = CompatLLMClient(adapters=[primary, secondary], max_retries=0)

    response = await client.generate(LLMRequest(prompt="hello"))

    assert response.text == "secondary ok"
    assert response.raw_response["selected_provider"] == "secondary"
    assert primary.calls == 1
    assert secondary.calls == 1


@pytest.mark.asyncio
async def test_llm_client_fallbacks_on_circuit_open_error() -> None:
    primary = StubAdapter(
        provider_name="primary",
        outcomes=[CircuitBreakerOpenError("circuit open", provider="primary", model="p1", operation="tool")],
        config=_build_config(provider=ModelProvider.DEEPSEEK, max_retries=0),
    )
    secondary = StubAdapter(
        provider_name="secondary",
        outcomes=[LLMResponse(text="secondary ok", raw_response={"provider": "secondary"})],
        config=_build_config(provider=ModelProvider.QWEN, max_retries=0),
    )
    client = CompatLLMClient(adapters=[primary, secondary], max_retries=0)

    response = await client.generate(LLMRequest(prompt="hello"))

    assert response.text == "secondary ok"
    assert response.raw_response["selected_provider"] == "secondary"
    assert primary.calls == 1
    assert secondary.calls == 1


@pytest.mark.asyncio
async def test_llm_client_fallbacks_on_provider_retryable_error() -> None:
    primary = StubAdapter(
        provider_name="primary",
        outcomes=[LLMProviderRetryableError("retryable", provider="primary", model="p1", operation="tool")],
        config=_build_config(provider=ModelProvider.DEEPSEEK, max_retries=0),
    )
    secondary = StubAdapter(
        provider_name="secondary",
        outcomes=[LLMResponse(text="secondary ok", raw_response={"provider": "secondary"})],
        config=_build_config(provider=ModelProvider.QWEN, max_retries=0),
    )
    client = CompatLLMClient(adapters=[primary, secondary], max_retries=0)

    response = await client.generate(LLMRequest(prompt="hello"))

    assert response.text == "secondary ok"
    assert response.raw_response["selected_provider"] == "secondary"
    assert primary.calls == 1
    assert secondary.calls == 1


@pytest.mark.asyncio
async def test_llm_client_raises_last_structured_error_when_all_providers_fail() -> None:
    primary = StubAdapter(
        provider_name="primary",
        outcomes=[_build_rate_limit_error()],
        config=_build_config(provider=ModelProvider.DEEPSEEK, max_retries=0),
    )
    secondary = StubAdapter(
        provider_name="secondary",
        outcomes=[httpx.ConnectError("down", request=httpx.Request("POST", "https://example.com"))],
        config=_build_config(provider=ModelProvider.QWEN, max_retries=0),
    )
    client = CompatLLMClient(adapters=[primary, secondary], max_retries=0)

    with pytest.raises(LLMProviderUnavailableError) as exc_info:
        await client.generate(LLMRequest(prompt="hello"))

    assert exc_info.value.provider == "secondary"


@pytest.mark.asyncio
async def test_llm_client_does_not_fallback_on_authentication_error() -> None:
    primary = StubAdapter(
        provider_name="primary",
        outcomes=[LLMAuthenticationError("bad key", provider="primary", model="p1", operation="tool")],
        config=_build_config(provider=ModelProvider.DEEPSEEK, max_retries=0),
    )
    secondary = StubAdapter(
        provider_name="secondary",
        outcomes=[LLMResponse(text="secondary ok", raw_response={"provider": "secondary"})],
        config=_build_config(provider=ModelProvider.QWEN, max_retries=0),
    )
    client = CompatLLMClient(adapters=[primary, secondary], max_retries=0)

    with pytest.raises(LLMAuthenticationError) as exc_info:
        await client.generate(LLMRequest(prompt="hello"))

    assert exc_info.value.provider == "primary"
    assert primary.calls == 1
    assert secondary.calls == 0


@pytest.mark.asyncio
async def test_llm_client_does_not_fallback_on_invalid_response_error() -> None:
    primary = StubAdapter(
        provider_name="primary",
        outcomes=[LLMInvalidResponseError("bad response", provider="primary", model="p1", operation="tool")],
        config=_build_config(provider=ModelProvider.DEEPSEEK, max_retries=0),
    )
    secondary = StubAdapter(
        provider_name="secondary",
        outcomes=[LLMResponse(text="secondary ok", raw_response={"provider": "secondary"})],
        config=_build_config(provider=ModelProvider.QWEN, max_retries=0),
    )
    client = CompatLLMClient(adapters=[primary, secondary], max_retries=0)

    with pytest.raises(LLMInvalidResponseError) as exc_info:
        await client.generate(LLMRequest(prompt="hello"))

    assert exc_info.value.provider == "primary"
    assert primary.calls == 1
    assert secondary.calls == 0


@pytest.mark.asyncio
async def test_llm_client_opens_circuit_breaker_after_threshold() -> None:
    adapter = StubAdapter(
        provider_name="primary",
        outcomes=[ValueError("bad response")],
        config=_build_config(max_retries=0),
    )
    client = CompatLLMClient(
        adapters=[adapter],
        max_retries=0,
        circuit_breaker_config=CircuitBreakerConfig(failure_threshold=1, reset_timeout_seconds=60),
    )

    with pytest.raises(LLMInvalidResponseError):
        await client.generate(LLMRequest(prompt="hello"))
    with pytest.raises(CircuitBreakerOpenError):
        await client.generate(LLMRequest(prompt="hello again"))

    assert adapter.calls == 1


@pytest.mark.asyncio
async def test_llm_client_half_open_recovery_closes_circuit() -> None:
    adapter = StubAdapter(
        provider_name="primary",
        outcomes=[ValueError("bad response"), LLMResponse(text="recovered"), LLMResponse(text="stable")],
        config=_build_config(max_retries=0),
    )
    client = CompatLLMClient(
        adapters=[adapter],
        max_retries=0,
        circuit_breaker_config=CircuitBreakerConfig(failure_threshold=1, reset_timeout_seconds=0),
    )

    with pytest.raises(LLMInvalidResponseError):
        await client.generate(LLMRequest(prompt="hello"))

    recovered = await client.generate(LLMRequest(prompt="hello again"))
    stable = await client.generate(LLMRequest(prompt="hello third"))

    assert recovered.text == "recovered"
    assert stable.text == "stable"
    assert client._provider_states[0].circuit_breaker.state.value == "closed"


def test_structured_output_validate_success() -> None:
    result = parse_json_output(raw_output='{"text":"ok"}', output_model=StructuredPayload)

    assert result.output.text == "ok"
    assert result.parsed_payload["text"] == "ok"


@pytest.mark.asyncio
async def test_structured_output_validate_failure_retries_and_raises() -> None:
    calls = 0

    async def _generate(request: LLMRequest) -> str:
        nonlocal calls
        calls += 1
        return '{"missing":"text"}'

    with pytest.raises(LLMInvalidResponseError) as exc_info:
        await invoke_structured_output(
            generate=_generate,
            request=LLMRequest(prompt="hello", max_validation_retries=1),
            output_model=StructuredPayload,
        )

    assert exc_info.value.error_type == "SCHEMA_VALIDATION_FAILED"
    assert calls == 2


def test_structured_output_invalid_json_is_classified() -> None:
    with pytest.raises(LLMInvalidResponseError) as exc_info:
        parse_json_output(raw_output='{"text":', output_model=StructuredPayload)

    assert exc_info.value.error_type == "INVALID_JSON"


def test_model_adapter_repairs_function_argument_missing_comma() -> None:
    adapter = StubAdapter(
        provider_name="primary",
        outcomes=[],
        config=_build_config(max_retries=0),
    )

    parsed = adapter.parse_function_arguments(
        '{"plan_id":"13" "status":"已完成" "reason":"已完成测试" "evidence":"周报明确完成"}'
    )

    assert parsed == {
        "plan_id": "13",
        "status": "已完成",
        "reason": "已完成测试",
        "evidence": "周报明确完成",
    }


def test_model_adapter_extracts_loose_single_judgement_arguments() -> None:
    adapter = StubAdapter(
        provider_name="primary",
        outcomes=[],
        config=_build_config(max_retries=0),
    )

    parsed = adapter.parse_function_arguments(
        "plan_id: 13\nstatus: 已完成\nreason: 周报明确说明已完成测试。\nevidence: 2026-05-09 周报写明完成冒烟测试。"
    )

    assert parsed == {
        "plan_id": "13",
        "status": "已完成",
        "reason": "周报明确说明已完成测试。",
        "evidence": "2026-05-09 周报写明完成冒烟测试。",
    }


@pytest.mark.asyncio
async def test_function_calling_schema_validate_success() -> None:
    client = StubStandaloneClient(
        outcomes=[
            LLMResponse(
                text='{"tool_name":"emit_payload","arguments":{"text":"ok"}}',
                function_call=LLMFunctionCall(tool_name="emit_payload", arguments={"text": "ok"}),
            )
        ],
        model_name="stub-model",
    )

    result = await FunctionCallingAdapter().invoke_structured(
        client=client,
        request=LLMRequest(prompt="hello"),
        function_schema=LLMFunctionSchema(
            name="emit_payload",
            description="Return the payload",
            parameters_schema=FunctionPayload.model_json_schema(),
            schema_name="FunctionPayload",
            schema_version="v1",
        ),
        output_model=FunctionPayload,
    )

    assert result.output.text == "ok"
    assert result.function_call.tool_name == "emit_payload"


@pytest.mark.asyncio
async def test_function_calling_missing_arguments_is_classified() -> None:
    client = StubStandaloneClient(
        outcomes=[LLMResponse(text="{}", function_call=LLMFunctionCall(tool_name="emit_payload", arguments={}))],
        model_name="stub-model",
    )

    with pytest.raises(FunctionCallingAdapterError) as exc_info:
        await FunctionCallingAdapter().invoke_structured(
            client=client,
            request=LLMRequest(prompt="hello", max_validation_retries=0),
            function_schema=LLMFunctionSchema(
                name="emit_payload",
                description="Return the payload",
                parameters_schema=FunctionPayload.model_json_schema(),
                schema_name="FunctionPayload",
                schema_version="v1",
            ),
            output_model=FunctionPayload,
        )

    assert exc_info.value.error_type == "MISSING_ARGUMENTS"


@pytest.mark.asyncio
async def test_function_calling_unsupported_function_is_classified() -> None:
    client = StubStandaloneClient(
        outcomes=[
            LLMResponse(
                text='{"tool_name":"wrong_tool","arguments":{"text":"ok"}}',
                function_call=LLMFunctionCall(tool_name="wrong_tool", arguments={"text": "ok"}),
            )
        ],
        model_name="stub-model",
    )

    with pytest.raises(FunctionCallingAdapterError) as exc_info:
        await FunctionCallingAdapter().invoke_structured(
            client=client,
            request=LLMRequest(prompt="hello", max_validation_retries=0),
            function_schema=LLMFunctionSchema(
                name="emit_payload",
                description="Return the payload",
                parameters_schema=FunctionPayload.model_json_schema(),
                schema_name="FunctionPayload",
                schema_version="v1",
            ),
            output_model=FunctionPayload,
        )

    assert exc_info.value.error_type == "UNSUPPORTED_FUNCTION"


@pytest.mark.asyncio
async def test_llm_call_recorder_is_called() -> None:
    records: list[dict[str, object]] = []
    client = StubStandaloneClient(
        outcomes=[LLMResponse(text="ok")],
        model_name="stub-model",
        llm_call_recorder=records.append,
    )

    await client.generate(LLMRequest(prompt="hello", metadata={"operation": "planner"}))

    assert len(records) == 1
    assert records[0]["operation"] == "planner"


@pytest.mark.asyncio
async def test_llm_call_recorder_exception_is_isolated() -> None:
    client = StubStandaloneClient(
        outcomes=[LLMResponse(text="ok")],
        model_name="stub-model",
        llm_call_recorder=lambda record: (_ for _ in ()).throw(RuntimeError("recorder failed")),
    )

    response = await client.generate(LLMRequest(prompt="hello"))

    assert response.text == "ok"
    assert client.calls == 1


def test_compatibility_wrapper_reexports_new_llm_client() -> None:
    assert CompatLLMClient is NewLLMClient


def test_new_llm_client_does_not_depend_on_legacy_wrapper() -> None:
    source = Path("app/llm/client.py").read_text(encoding="utf-8")

    assert "app.tools.llm_client" not in source


def test_legacy_wrapper_stays_thin_and_delegates_to_new_layer() -> None:
    source = Path("app/tools/llm_client.py").read_text(encoding="utf-8")

    assert "from app.llm import (" in source
    assert "class LLMClient" not in source
    assert "async def generate" not in source
    assert "class LLMRetryPolicy" not in source
    assert "CircuitBreaker(" not in source


@pytest.mark.asyncio
async def test_runtime_graph_writes_llm_calls_for_tool_client() -> None:
    client = RuntimeStructuredClient(model_name="runtime-structured-client")
    router = TaskRouter()
    reason_tool = LLMReasonTool(client=client)
    text_tool = TextGenerateTool(client=client)
    await router.register_tools(
        [
            (reason_tool, capability_from_tool(reason_tool)),
            (text_tool, capability_from_tool(text_tool)),
        ]
    )
    graph = build_langgraph_runtime(
        GraphRuntimeDependencies(
            router=router,
            supervisor_agent=SupervisorAgent(client=client),
            planner_agent=LLMTaskPlanner(client=client),
            repair_llm_client=client,
        )
    )
    state = LangGraphState.create(
        request_id="req_tool_llm_calls",
        session_id="sess_tool_llm_calls",
        user_input="write a short paragraph",
    )

    try:
        final_state = coerce_langgraph_state(await graph.ainvoke(state))
    finally:
        await client.aclose()

    llm_calls = final_state.metadata["llm_calls"]
    assert any(item["operation"] == "tool" for item in llm_calls)
    assert all(item["success"] is True for item in llm_calls)


@pytest.mark.asyncio
async def test_runtime_graph_writes_llm_calls_for_planner_and_repair_clients() -> None:
    planner_result = await run_runtime(
        RuntimeTestInput(
            name="planner_llm_calls",
            user_input="build a plan",
            force_route="COMPLEX_TASK",
            use_planner_agent=True,
            task_specs=[
                TaskSpec(
                    task_id="task_1",
                    task_name="task_1",
                    description="task",
                    tool="llm_reason_tool",
                    prompt="hello",
                    output_key="result",
                )
            ],
        )
    )
    repair_result = await run_runtime(
        RuntimeTestInput(
            name="repair_llm_calls",
            user_input="repair this plan",
            force_route="COMPLEX_TASK",
            planner_scenario=PlannerScenario.INVALID_JSON,
            parser_repair_outputs=[
                json.dumps(
                    {
                        "goal": "repair this plan",
                        "tasks": [
                            {
                                "task_id": "task_1",
                                "task_name": "task_1",
                                "description": "task",
                                "tool": "llm_reason_tool",
                                "input": {"prompt": "hello"},
                                "output_key": "result",
                                "depends_on": [],
                                "priority": 1,
                                "status": "PENDING",
                                "retry_count": 0,
                                "max_retry": 1,
                                "timeout": 60,
                                "created_at": "2026-01-01T00:00:00+00:00",
                            }
                        ],
                    }
                )
            ],
            task_specs=[
                TaskSpec(
                    task_id="task_1",
                    task_name="task_1",
                    description="task",
                    tool="llm_reason_tool",
                    prompt="hello",
                    output_key="result",
                )
            ],
        )
    )

    planner_calls = planner_result.final_output["metadata"]["llm_calls"]
    repair_calls = repair_result.final_output["metadata"]["llm_calls"]

    assert any(item["operation"] == "planner" for item in planner_calls)
    assert any(item["operation"] == "repair" for item in repair_calls)
