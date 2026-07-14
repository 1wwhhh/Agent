from __future__ import annotations

import httpx
import pytest

from app.adapters.base import ModelAdapter
from app.executor.task_executor import TaskExecutor
from app.queue.task_queue import TaskQueue
from app.router import TaskRouter
from app.router.capability import capability_from_tool
from app.schemas.context import ContextStore, IdempotencyRecord, RuntimeContext
from app.schemas.llm import LLMRequest
from app.schemas.model import ModelConfig, ModelProvider
from app.schemas.task import TaskModel, TaskStatus
from app.tools.base import BaseTool
from app.tools.llm_client import (
    FAIL_FAST_TIMEOUT_MARKER,
    RETRYABLE_ERROR_MARKER,
    LLMClient,
    LLMProviderRetryableError,
    LLMProviderTimeoutError,
)
from app.tools.llm_reason import LLMReasonTool
from app.tools.text_generate import TextGenerateTool
from tests.support.models import RuntimeTestInput, TaskSpec
from tests.support.runtime_runner import run_runtime
from tests.support.test_case_generator import TestCaseGenerator
from tests.support.tools import StaticStructuredLLMClient


class FailFastTextGenerateTool(BaseTool):
    name: str = "text_generate_tool"
    description: str = "Simulated text generation timeout"
    timeout: int = 75

    async def _arun(self, payload: dict[str, object], context: ContextStore | None = None):
        return self.build_result(
            success=False,
            error=f"{FAIL_FAST_TIMEOUT_MARKER}: simulated provider timeout",
            metadata={"fail_fast_timeout": True},
        )


class FailFastReasonTool(BaseTool):
    name: str = "llm_reason_tool"
    description: str = "Simulated reasoning timeout"
    timeout: int = 75

    async def _arun(self, payload: dict[str, object], context: ContextStore | None = None):
        return self.build_result(
            success=False,
            error=f"{FAIL_FAST_TIMEOUT_MARKER}: simulated provider timeout",
            metadata={"timeout_fail_fast": True},
        )


class RetryableReasonTool(BaseTool):
    name: str = "llm_reason_tool"
    description: str = "Simulated recoverable reasoning error"
    timeout: int = 75

    async def _arun(self, payload: dict[str, object], context: ContextStore | None = None):
        return self.build_result(
            success=False,
            error=f"{RETRYABLE_ERROR_MARKER}: simulated 429",
            metadata={"retryable_error": True},
        )


class TimeoutAdapter(ModelAdapter):
    calls: int = 0

    @property
    def provider_name(self) -> str:
        return "timeout_provider"

    async def generate(self, request: LLMRequest):
        self.calls += 1
        raise httpx.ReadTimeout("simulated read timeout")


class RetryableAdapter(ModelAdapter):
    calls: int = 0

    @property
    def provider_name(self) -> str:
        return "retryable_provider"

    async def generate(self, request: LLMRequest):
        self.calls += 1
        response = httpx.Response(status_code=429, request=httpx.Request("POST", "https://example.com"))
        raise httpx.HTTPStatusError("simulated 429", request=response.request, response=response)


@pytest.mark.asyncio
async def test_retry_updates_latest_execution_result_without_idempotency_conflict():
    result = await run_runtime(TestCaseGenerator.fail_once_case())

    assert result.final_output["success"] is True
    assert result.task_states["task_retry"]["status"] == "SUCCESS"
    assert all("idempotency key" not in error["message"] for error in result.context_snapshot["errors"])

    idempotency_records = result.context_snapshot["idempotency_records"]
    assert len(idempotency_records) == 1
    stored_record = next(iter(idempotency_records.values()))
    assert stored_record["final_status"] == "SUCCESS"
    assert stored_record["attempt_key"].endswith(":attempt:2")

    latest_results = result.context_snapshot["shared_data"]["task_execution_results"]
    assert latest_results["task_retry"]["final_status"] == "SUCCESS"
    assert latest_results["task_retry"]["attempt_key"].endswith(":attempt:2")

    history = result.context_snapshot["shared_data"]["task_execution_results_history"]
    task_history = [item for item in history if item["task_id"] == "task_retry"]
    assert [item["final_status"] for item in task_history] == ["RETRY", "SUCCESS"]


def test_irreversible_success_keeps_strict_idempotency_protection():
    context = ContextStore(
        runtime=RuntimeContext(
            request_id="req_test",
            session_id="sess_test",
            user_input="test",
        )
    )
    context.set_idempotency_record(
        IdempotencyRecord(
            idempotency_key="req_test:task_1:output",
            task_id="task_1",
            output_key="output",
            tool_name="llm_reason_tool",
            attempt_key="req_test:task_1:output:attempt:1",
            final_status=TaskStatus.SUCCESS,
            success=True,
            irreversible=True,
            output={"text": "first"},
        )
    )

    with pytest.raises(ValueError):
        context.set_idempotency_record(
            IdempotencyRecord(
                idempotency_key="req_test:task_1:output",
                task_id="task_1",
                output_key="output",
                tool_name="llm_reason_tool",
                attempt_key="req_test:task_1:output:attempt:2",
                final_status=TaskStatus.SUCCESS,
                success=True,
                irreversible=True,
                output={"text": "second"},
            )
        )


def test_executor_timeout_resolution_prefers_tool_defaults_and_layered_text_generate_timeouts():
    runtime_input = RuntimeTestInput(
        name="timeout_resolution_case",
        user_input="test",
        task_specs=[
            TaskSpec(
                task_id="task_1",
                task_name="task_1",
                description="task",
                tool="llm_reason_tool",
                prompt="prompt",
                output_key="output",
            )
        ],
    )
    client = StaticStructuredLLMClient(test_input=runtime_input)
    llm_reason_tool = LLMReasonTool(client=client)
    text_generate_tool = TextGenerateTool(client=client)

    context = ContextStore(
        runtime=RuntimeContext(
            request_id="req_timeout",
            session_id="sess_timeout",
            user_input="timeout",
        )
    )
    queue = TaskQueue(context=context)
    executor = TaskExecutor(context=context, queue=queue, router=TaskRouter())

    reason_task = TaskModel(
        task_id="task_reason",
        task_name="reason",
        description="reason task",
        tool="llm_reason_tool",
        input={},
        output_key="reason_output",
        timeout=60,
    )
    text_task = TaskModel(
        task_id="task_text",
        task_name="text",
        description="text task",
        tool="text_generate_tool",
        input={},
        output_key="text_output",
        timeout=60,
    )

    assert llm_reason_tool.timeout == 75
    assert llm_reason_tool.default_timeout_seconds == 45
    assert text_generate_tool.timeout == 75
    assert text_generate_tool.default_timeout_seconds == 45

    reason_settings = executor._resolve_timeout_settings(
        task=reason_task,
        tool_timeout=llm_reason_tool.timeout,
        payload={},
    )
    assert reason_settings == {
        "timeout_seconds": 45,
        "tool_timeout_seconds": 75,
        "executor_timeout_seconds": 90,
    }

    text_settings = executor._resolve_timeout_settings(
        task=text_task,
        tool_timeout=text_generate_tool.timeout,
        payload={},
    )
    assert text_settings == {
        "timeout_seconds": 45,
        "tool_timeout_seconds": 75,
        "executor_timeout_seconds": 90,
    }

    explicit_text_settings = executor._resolve_timeout_settings(
        task=text_task,
        tool_timeout=text_generate_tool.timeout,
        payload={"timeout_seconds": 50, "tool_timeout_seconds": 80, "executor_timeout_seconds": 100},
    )
    assert explicit_text_settings == {
        "timeout_seconds": 50,
        "tool_timeout_seconds": 80,
        "executor_timeout_seconds": 100,
    }


@pytest.mark.asyncio
async def test_text_generate_timeout_failure_skips_retry_and_finishes_terminally():
    context = ContextStore(
        runtime=RuntimeContext(
            request_id="req_fail_fast",
            session_id="sess_fail_fast",
            user_input="fail fast",
        )
    )
    queue = TaskQueue(context=context)
    router = TaskRouter()
    tool = FailFastTextGenerateTool()
    await router.register_tool(tool, capability_from_tool(tool))

    task = TaskModel(
        task_id="task_text",
        task_name="text",
        description="text",
        tool="text_generate_tool",
        input={"prompt": "hello"},
        output_key="text_output",
        max_retry=2,
        timeout=60,
    )
    await queue.initialize([task])
    ready_tasks = await queue.get_ready_tasks()
    assert [item.task_id for item in ready_tasks] == [task.task_id]

    executor = TaskExecutor(context=context, queue=queue, router=router)
    result = await executor.execute_task(task.task_id)

    assert result.final_status == TaskStatus.TIMEOUT
    assert result.retry_scheduled is False
    assert context.tasks[task.task_id].status == TaskStatus.TIMEOUT

    latest_result = context.shared_data["task_execution_results"][task.task_id]
    assert latest_result["final_status"] == "TIMEOUT"
    assert latest_result["retry_scheduled"] is False


@pytest.mark.asyncio
async def test_llm_reason_timeout_failure_skips_retry_and_finishes_terminally():
    context = ContextStore(
        runtime=RuntimeContext(
            request_id="req_reason_timeout",
            session_id="sess_reason_timeout",
            user_input="reason timeout",
        )
    )
    queue = TaskQueue(context=context)
    router = TaskRouter()
    tool = FailFastReasonTool()
    await router.register_tool(tool, capability_from_tool(tool))

    task = TaskModel(
        task_id="task_reason",
        task_name="reason",
        description="reason",
        tool="llm_reason_tool",
        input={"prompt": "hello"},
        output_key="reason_output",
        max_retry=2,
        timeout=60,
    )
    await queue.initialize([task])
    ready_tasks = await queue.get_ready_tasks()
    assert [item.task_id for item in ready_tasks] == [task.task_id]

    executor = TaskExecutor(context=context, queue=queue, router=router)
    result = await executor.execute_task(task.task_id)

    assert result.final_status == TaskStatus.TIMEOUT
    assert result.retry_scheduled is False
    assert context.tasks[task.task_id].status == TaskStatus.TIMEOUT


@pytest.mark.asyncio
async def test_llm_reason_retryable_failure_allows_only_one_retry():
    context = ContextStore(
        runtime=RuntimeContext(
            request_id="req_reason_retry",
            session_id="sess_reason_retry",
            user_input="reason retry",
        )
    )
    queue = TaskQueue(context=context)
    router = TaskRouter()
    tool = RetryableReasonTool()
    await router.register_tool(tool, capability_from_tool(tool))

    task = TaskModel(
        task_id="task_reason",
        task_name="reason",
        description="reason",
        tool="llm_reason_tool",
        input={"prompt": "hello"},
        output_key="reason_output",
        max_retry=2,
        timeout=60,
    )
    await queue.initialize([task])

    executor = TaskExecutor(context=context, queue=queue, router=router)

    ready_tasks = await queue.get_ready_tasks()
    assert [item.task_id for item in ready_tasks] == [task.task_id]
    first_result = await executor.execute_task(task.task_id)
    assert first_result.final_status == TaskStatus.RETRY
    assert first_result.retry_scheduled is True
    assert context.tasks[task.task_id].status == TaskStatus.RETRY

    ready_tasks = await queue.get_ready_tasks()
    assert [item.task_id for item in ready_tasks] == [task.task_id]
    second_result = await executor.execute_task(task.task_id)
    assert second_result.final_status == TaskStatus.FAILED
    assert second_result.retry_scheduled is False
    assert context.tasks[task.task_id].status == TaskStatus.FAILED


@pytest.mark.asyncio
async def test_llm_client_disables_provider_retries_for_fail_fast_timeout_requests():
    adapter = TimeoutAdapter(
        config=ModelConfig(
            provider=ModelProvider.DEEPSEEK,
            api_key="test-key",
            base_url="https://example.com",
            model_name="test-model",
            timeout_seconds=60,
            max_retries=3,
        )
    )
    client = LLMClient(adapters=[adapter], max_retries=3)

    request = LLMRequest(
        prompt="hello",
        timeout_seconds=45,
        metadata={"fail_fast_timeout": True},
    )

    with pytest.raises(LLMProviderTimeoutError):
        await client.generate(request)

    assert adapter.calls == 1


@pytest.mark.asyncio
async def test_llm_client_disables_provider_retries_for_minimal_retry_policy_requests():
    adapter = RetryableAdapter(
        config=ModelConfig(
            provider=ModelProvider.DEEPSEEK,
            api_key="test-key",
            base_url="https://example.com",
            model_name="test-model",
            timeout_seconds=60,
            max_retries=3,
        )
    )
    client = LLMClient(adapters=[adapter], max_retries=3)

    request = LLMRequest(
        prompt="hello",
        timeout_seconds=45,
        metadata={"disable_provider_retries": True},
    )

    with pytest.raises(LLMProviderRetryableError):
        await client.generate(request)

    assert adapter.calls == 1
