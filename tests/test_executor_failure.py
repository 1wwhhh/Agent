from __future__ import annotations

import pytest

from tests.support.assertions import (
    assert_context_state as _assert_context_state,
    assert_execution_order as _assert_execution_order,
    assert_task_states as _assert_task_states,
)
from tests.support.runtime_runner import run_runtime as _run_runtime, setup_graph as _setup_graph
from tests.support.test_case_generator import TestCaseGenerator


async def setup_graph(runtime_input=None):
    return await _setup_graph(runtime_input or TestCaseGenerator.fail_once_case())


async def run_runtime(runtime_input):
    return await _run_runtime(runtime_input)


def assert_task_states(result, *, expected_statuses):
    _assert_task_states(result, expected_statuses=expected_statuses)


def assert_context_state(result, *, expected_output_keys=None, missing_output_keys=None):
    _assert_context_state(
        result,
        expected_output_keys=expected_output_keys,
        missing_output_keys=missing_output_keys,
    )


def assert_execution_order(result, *, before_after_pairs=None, blocked_tasks=None):
    _assert_execution_order(result, before_after_pairs=before_after_pairs, blocked_tasks=blocked_tasks)


@pytest.mark.asyncio
async def test_tool_failure_enters_retry_then_recovers():
    runtime_input = TestCaseGenerator.fail_once_case()
    result = await run_runtime(runtime_input)

    assert_task_states(result, expected_statuses={"task_retry": "SUCCESS"})
    assert_context_state(result, expected_output_keys=["retry_output"])
    retry_events = [item for item in result.execution_trace["execution_history"] if item["status"] == "RETRY"]
    assert retry_events, "expected RETRY event in execution history"
    assert result.task_states["task_retry"]["retry_count"] == 1
    assert result.metrics.retry_rate > 0


@pytest.mark.asyncio
async def test_timeout_transitions_to_timeout_and_releases_task():
    runtime_input = TestCaseGenerator.timeout_case()
    result = await run_runtime(runtime_input)

    assert_task_states(result, expected_statuses={"task_timeout": "TIMEOUT"})
    assert_context_state(result, missing_output_keys=["timeout_output"])
    assert any(call["status"] == "TIMEOUT" for call in result.execution_trace["tool_invocations"])
    assert result.final_output["success"] is False


@pytest.mark.asyncio
async def test_executor_internal_crash_is_captured_as_terminal_failure():
    runtime_input = TestCaseGenerator.executor_crash_case()
    result = await run_runtime(runtime_input)

    assert_task_states(result, expected_statuses={"task_executor_crash": "FAILED"})
    assert_context_state(result, missing_output_keys=["executor_crash_output"])
    assert any("executor internal error" in error["message"] for error in result.context_snapshot["errors"])
    assert result.final_output["success"] is False
