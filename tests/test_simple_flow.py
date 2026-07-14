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
    return await _setup_graph(runtime_input or TestCaseGenerator.simple_success_case())


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
async def test_simple_task_bypasses_planner_and_executes_directly():
    runtime_input = TestCaseGenerator.simple_success_case()
    result = await run_runtime(runtime_input)

    assert result.final_output["supervisor_route"] == "SIMPLE_TASK"
    assert result.final_output["success"] is True
    assert_task_states(result, expected_statuses={"task_1": "SUCCESS"})
    assert_context_state(result, expected_output_keys=["final_result"])
    assert result.context_snapshot["task_results"]["final_result"]["text"] == "text_generate_tool::Write a short greeting."
    assert all(call["tool_name"] != "planner_test_tool" for call in result.execution_trace["tool_invocations"])
    assert "planner" not in result.execution_trace["node_timings"]
    assert "parser" not in result.execution_trace["node_timings"]
    assert result.metrics.task_success_rate == 1.0
    assert result.metrics.context_consistency_rate == 1.0
