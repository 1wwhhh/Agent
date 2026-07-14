from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.executor.exceptions import InvalidTaskStateTransitionError
from app.executor.state_guard import validate_transition
from app.executor.task_executor import TaskExecutor
from app.queue.task_queue import TaskQueue
from app.router import TaskRouter
from app.router.capability import capability_from_tool
from app.schemas.context import ContextStore, IdempotencyRecord, RuntimeContext
from app.schemas.task import TaskModel, TaskStatus
from tests.support.runtime_runner import run_runtime
from tests.support.test_case_generator import TestCaseGenerator
from tests.support.tools import RuntimeTestTool


def _build_context() -> ContextStore:
    return ContextStore(
        runtime=RuntimeContext(
            request_id="req_executor_stability",
            session_id="sess_executor_stability",
            user_input="executor stability",
        )
    )


def _build_task(
    *,
    task_id: str,
    tool: str = "text_generate_tool",
    output_key: str | None = None,
    behavior: str = "success",
    max_retry: int = 1,
    timeout: int = 60,
    depends_on: list[str] | None = None,
    metadata: dict[str, object] | None = None,
) -> TaskModel:
    payload = {"prompt": task_id, "task_id": task_id, "behavior": behavior}
    if metadata:
        payload.update(metadata)
    return TaskModel(
        task_id=task_id,
        task_name=task_id,
        description=f"task {task_id}",
        tool=tool,
        input=payload,
        output_key=output_key or f"{task_id}_output",
        depends_on=depends_on or [],
        max_retry=max_retry,
        timeout=timeout,
    )


async def _build_executor(
    *,
    context: ContextStore,
    tasks: list[TaskModel],
    failure_records: list[dict[str, object]] | None = None,
) -> tuple[TaskQueue, TaskExecutor]:
    queue = TaskQueue(context=context, max_concurrency=8)
    await queue.initialize(tasks)
    router = TaskRouter()
    reason_tool = RuntimeTestTool(name="llm_reason_tool", description="reason")
    text_tool = RuntimeTestTool(name="text_generate_tool", description="text")
    await router.register_tools(
        [
            (reason_tool, capability_from_tool(reason_tool)),
            (text_tool, capability_from_tool(text_tool)),
        ]
    )
    executor = TaskExecutor(
        context=context,
        queue=queue,
        router=router,
        failure_recorder=(failure_records.append if failure_records is not None else None),
    )
    return queue, executor


def test_state_guard_allows_legal_transitions() -> None:
    validate_transition(TaskStatus.PENDING, TaskStatus.QUEUED)
    validate_transition(TaskStatus.QUEUED, TaskStatus.RUNNING)
    validate_transition(TaskStatus.QUEUED, TaskStatus.SUCCESS)
    validate_transition(TaskStatus.RUNNING, TaskStatus.RETRY)
    validate_transition(TaskStatus.RUNNING, TaskStatus.TIMEOUT)
    validate_transition(TaskStatus.RETRY, TaskStatus.QUEUED)


def test_state_guard_rejects_illegal_transitions() -> None:
    with pytest.raises(InvalidTaskStateTransitionError):
        validate_transition(TaskStatus.SUCCESS, TaskStatus.RUNNING)

    with pytest.raises(InvalidTaskStateTransitionError):
        validate_transition(TaskStatus.TIMEOUT, TaskStatus.RUNNING)


@pytest.mark.asyncio
async def test_executor_does_not_duplicate_dependency_context_when_template_references_output() -> None:
    context = _build_context()
    dependency = _build_task(
        task_id="task_2",
        tool="llm_reason_tool",
        output_key="rag_summary",
    )
    text_task = _build_task(
        task_id="task_3",
        depends_on=["task_2"],
        metadata={"context": "{{rag_summary.text}}"},
    )
    _, executor = await _build_executor(context=context, tasks=[dependency, text_task])
    context.task_results["rag_summary"] = {"text": "large summarized evidence"}

    payload = executor._build_tool_payload(context.tasks["task_3"])

    assert payload["context"] == "{{rag_summary.text}}"
    assert "Dependency Context" not in payload["context"]


@pytest.mark.asyncio
async def test_queue_transition_entrypoints_call_state_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    transitions: list[tuple[TaskStatus, TaskStatus]] = []

    def _spy(old_status: TaskStatus, new_status: TaskStatus) -> None:
        transitions.append((old_status, new_status))

    monkeypatch.setattr("app.queue.task_queue.validate_transition", _spy)

    context = _build_context()
    queue = TaskQueue(context=context)
    task = _build_task(task_id="task_guard")

    await queue.initialize([task])
    ready = await queue.get_ready_tasks()
    assert [item.task_id for item in ready] == ["task_guard"]
    await queue.mark_task_running("task_guard")
    await queue.mark_task_success("task_guard")

    assert transitions == [
        (TaskStatus.PENDING, TaskStatus.QUEUED),
        (TaskStatus.QUEUED, TaskStatus.RUNNING),
        (TaskStatus.RUNNING, TaskStatus.SUCCESS),
    ]


@pytest.mark.asyncio
async def test_retryable_exception_transitions_through_retry_loop() -> None:
    context = _build_context()
    task = _build_task(task_id="task_retry", behavior="fail_once", max_retry=1)
    queue, executor = await _build_executor(context=context, tasks=[task])

    ready = await queue.get_ready_tasks()
    assert [item.task_id for item in ready] == ["task_retry"]
    first = await executor.execute_task("task_retry")
    assert first.final_status == TaskStatus.RETRY
    assert first.retry_scheduled is True

    ready = await queue.get_ready_tasks()
    assert [item.task_id for item in ready] == ["task_retry"]
    second = await executor.execute_task("task_retry")
    assert second.final_status == TaskStatus.SUCCESS
    assert second.retry_scheduled is False

    statuses = [item.status.value for item in context.execution_history if item.task_id == "task_retry"]
    assert statuses == ["QUEUED", "RUNNING", "RETRY", "QUEUED", "RUNNING", "SUCCESS"]


@pytest.mark.asyncio
async def test_non_retryable_exception_becomes_failed() -> None:
    context = _build_context()
    task = _build_task(task_id="task_exception", tool="llm_reason_tool", behavior="tool_exception", max_retry=1)
    queue, executor = await _build_executor(context=context, tasks=[task])

    await queue.get_ready_tasks()
    result = await executor.execute_task("task_exception")

    assert result.final_status == TaskStatus.FAILED
    assert result.retry_scheduled is False
    assert context.tasks["task_exception"].status == TaskStatus.FAILED


@pytest.mark.asyncio
async def test_timeout_becomes_timeout_and_records_failure_metadata() -> None:
    context = _build_context()
    failure_records: list[dict[str, object]] = []
    task = _build_task(
        task_id="task_timeout",
        behavior="timeout",
        max_retry=0,
        timeout=1,
        metadata={
            "timeout_seconds": 1,
            "tool_timeout_seconds": 2,
            "executor_timeout_seconds": 3,
            "timeout_sleep_seconds": 3.5,
        },
    )
    queue, executor = await _build_executor(context=context, tasks=[task], failure_records=failure_records)

    await queue.get_ready_tasks()
    result = await executor.execute_task("task_timeout")

    assert result.final_status == TaskStatus.TIMEOUT
    assert context.tasks["task_timeout"].status == TaskStatus.TIMEOUT
    assert failure_records[0]["status"] == "TIMEOUT"
    assert failure_records[0]["error_type"] == "TOOL_TIMEOUT"
    assert failure_records[0]["attempt_key"] == result.attempt_key
    assert failure_records[0]["idempotency_key"] == result.idempotency_key


@pytest.mark.asyncio
async def test_retry_exhausted_becomes_failed_without_requeue() -> None:
    context = _build_context()
    task = _build_task(task_id="task_exhausted", behavior="fail_once", max_retry=0)
    queue, executor = await _build_executor(context=context, tasks=[task])

    await queue.get_ready_tasks()
    result = await executor.execute_task("task_exhausted")

    assert result.final_status == TaskStatus.FAILED
    assert result.retry_scheduled is False
    statuses = [item.status.value for item in context.execution_history if item.task_id == "task_exhausted"]
    assert statuses == ["QUEUED", "RUNNING", "FAILED"]


@pytest.mark.asyncio
async def test_queued_to_success_idempotent_recovery_path() -> None:
    context = _build_context()
    task = _build_task(task_id="task_restore", max_retry=0)
    context.set_idempotency_record(
        IdempotencyRecord(
            idempotency_key=f"{context.runtime.request_id}:task_restore:task_restore_output",
            task_id="task_restore",
            output_key="task_restore_output",
            tool_name="text_generate_tool",
            attempt_key=f"{context.runtime.request_id}:task_restore:task_restore_output:attempt:1",
            final_status=TaskStatus.SUCCESS,
            success=True,
            irreversible=True,
            output={"text": "restored"},
            recorded_at=datetime.now(timezone.utc),
        )
    )
    queue, executor = await _build_executor(context=context, tasks=[task])

    ready = await queue.get_ready_tasks()
    assert [item.task_id for item in ready] == ["task_restore"]
    result = await executor.execute_task("task_restore")

    assert result.final_status == TaskStatus.SUCCESS
    assert context.task_results["task_restore_output"] == {"text": "restored"}
    assert context.tasks["task_restore"].status == TaskStatus.SUCCESS
    assert context.shared_data.get("runtime_tool_trace") in (None, [])


@pytest.mark.asyncio
async def test_batch_isolation_keeps_independent_ready_tasks_running() -> None:
    context = _build_context()
    failure_records: list[dict[str, object]] = []
    failing = _build_task(task_id="task_fail", tool="llm_reason_tool", behavior="tool_exception", max_retry=0)
    healthy = _build_task(task_id="task_ok", behavior="success", max_retry=0)
    queue, executor = await _build_executor(
        context=context,
        tasks=[failing, healthy],
        failure_records=failure_records,
    )

    ready = await queue.get_ready_tasks()
    results = await executor.execute_ready_tasks([item.task_id for item in ready])

    assert {item.task_id: item.final_status for item in results} == {
        "task_fail": TaskStatus.FAILED,
        "task_ok": TaskStatus.SUCCESS,
    }
    assert context.tasks["task_fail"].status == TaskStatus.FAILED
    assert context.tasks["task_ok"].status == TaskStatus.SUCCESS
    assert "task_ok_output" in context.task_results
    assert failure_records


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_status", [TaskStatus.FAILED, TaskStatus.TIMEOUT, TaskStatus.QUEUED])
async def test_invalid_retry_transition_does_not_mutate_status_or_retry_count(initial_status: TaskStatus) -> None:
    context = _build_context()
    queue = TaskQueue(context=context)
    task = _build_task(task_id=f"task_{initial_status.value.lower()}", max_retry=2)
    task.status = initial_status
    original_retry_count = task.retry_count

    await queue.hydrate([task])

    with pytest.raises(InvalidTaskStateTransitionError):
        await queue.mark_task_for_retry(task.task_id)

    stored_task = context.tasks[task.task_id]
    assert stored_task.status == initial_status
    assert stored_task.retry_count == original_retry_count


@pytest.mark.asyncio
async def test_post_success_result_callback_error_does_not_flip_success() -> None:
    context = _build_context()
    task = _build_task(task_id="task_callback_success", max_retry=0)
    queue = TaskQueue(context=context)
    await queue.initialize([task])
    router = TaskRouter()
    tool = RuntimeTestTool(name="text_generate_tool", description="text")
    await router.register_tool(tool, capability_from_tool(tool))

    async def _broken_callback(result) -> None:
        raise RuntimeError("callback exploded")

    executor = TaskExecutor(
        context=context,
        queue=queue,
        router=router,
        result_callback=_broken_callback,
    )

    await queue.get_ready_tasks()
    result = await executor.execute_task(task.task_id)

    assert result.final_status == TaskStatus.SUCCESS
    assert context.tasks[task.task_id].status == TaskStatus.SUCCESS
    assert [item.status.value for item in context.execution_history if item.task_id == task.task_id] == [
        "QUEUED",
        "RUNNING",
        "SUCCESS",
    ]
    assert any(item.details.get("error_code") == "post_success_hook_error" for item in context.errors)


@pytest.mark.asyncio
async def test_post_success_checkpoint_error_does_not_flip_success() -> None:
    context = _build_context()
    task = _build_task(task_id="task_checkpoint_success", max_retry=0)
    queue = TaskQueue(context=context)
    await queue.initialize([task])
    router = TaskRouter()
    tool = RuntimeTestTool(name="text_generate_tool", description="text")
    await router.register_tool(tool, capability_from_tool(tool))

    async def _broken_checkpoint(event: str, metadata: dict[str, object] | None) -> None:
        raise RuntimeError(f"checkpoint exploded during {event}")

    executor = TaskExecutor(
        context=context,
        queue=queue,
        router=router,
        checkpoint_saver=_broken_checkpoint,
    )

    await queue.get_ready_tasks()
    result = await executor.execute_task(task.task_id)

    assert result.final_status == TaskStatus.SUCCESS
    assert context.tasks[task.task_id].status == TaskStatus.SUCCESS
    assert [item.status.value for item in context.execution_history if item.task_id == task.task_id] == [
        "QUEUED",
        "RUNNING",
        "SUCCESS",
    ]
    assert any(item.details.get("error_code") == "checkpoint_error" for item in context.errors)


@pytest.mark.asyncio
async def test_failure_recorder_error_does_not_break_executor_result() -> None:
    context = _build_context()
    task = _build_task(task_id="task_failure_recorder", tool="llm_reason_tool", behavior="tool_exception", max_retry=0)
    queue = TaskQueue(context=context)
    await queue.initialize([task])
    router = TaskRouter()
    tool = RuntimeTestTool(name="llm_reason_tool", description="reason")
    await router.register_tool(tool, capability_from_tool(tool))

    def _broken_failure_recorder(record: dict[str, object]) -> None:
        raise RuntimeError("failure recorder exploded")

    executor = TaskExecutor(
        context=context,
        queue=queue,
        router=router,
        failure_recorder=_broken_failure_recorder,
    )

    await queue.get_ready_tasks()
    result = await executor.execute_task(task.task_id)

    assert result.final_status == TaskStatus.FAILED
    assert context.tasks[task.task_id].status == TaskStatus.FAILED
    assert any("failure recorder error" in item.message for item in context.errors)


@pytest.mark.asyncio
async def test_failed_dependency_cancels_downstream_but_independent_task_succeeds() -> None:
    result = await run_runtime(TestCaseGenerator.partial_failure_branch_case())

    assert result.task_states["task_fail_root"]["status"] == "FAILED"
    assert result.task_states["task_fail_child"]["status"] == "CANCELLED"
    assert result.task_states["task_ok"]["status"] == "SUCCESS"
    assert "healthy_output" in result.context_snapshot["task_results"]


@pytest.mark.asyncio
async def test_runtime_graph_records_task_failures_without_polluting_execution_history() -> None:
    result = await run_runtime(TestCaseGenerator.timeout_case())

    task_failures = result.final_output["metadata"]["task_failures"]
    assert len(task_failures) == 1
    failure = task_failures[0]
    assert failure["task_id"] == "task_timeout"
    assert failure["status"] == "TIMEOUT"
    assert failure["error_type"] == "TOOL_TIMEOUT"
    assert failure["attempt_key"].endswith(":attempt:1")
    assert "idempotency_key" in failure

    history = result.context_snapshot["execution_history"]
    assert history
    assert all("error_type" not in item for item in history)
    assert all(
        item["status"] in {"QUEUED", "RUNNING", "SUCCESS", "FAILED", "TIMEOUT", "RETRY", "CANCELLED"}
        for item in history
    )
