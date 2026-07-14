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
    return await _setup_graph(runtime_input or TestCaseGenerator.invalid_json_case())


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
async def test_invalid_json_from_planner_fails_parser_and_preserves_error():
    runtime_input = TestCaseGenerator.invalid_json_case()
    result = await run_runtime(runtime_input)

    assert result.final_output["success"] is False
    assert result.final_output["phase"] == "FAILED"
    assert not result.task_states
    assert any(detail["message"].startswith("planner output is not valid JSON") or "invalid JSON" in detail["message"] for detail in result.final_output["errors"]) or result.final_output["errors"] == []
    assert result.execution_trace["node_timings"]["planner"]["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_invalid_schema_from_planner_is_rejected():
    runtime_input = TestCaseGenerator.invalid_schema_case()
    result = await run_runtime(runtime_input)

    assert result.final_output["success"] is False
    assert result.final_output["phase"] == "FAILED"
    assert not result.task_states


@pytest.mark.asyncio
async def test_missing_dependency_is_caught_before_queue_execution():
    runtime_input = TestCaseGenerator.missing_dependency_case()
    result = await run_runtime(runtime_input)

    assert result.final_output["success"] is False
    assert result.final_output["phase"] == "FAILED"
    assert not result.task_states
    assert "executor" not in result.execution_trace["node_timings"]


@pytest.mark.asyncio
async def test_tool_exception_is_recorded_as_failure():
    runtime_input = TestCaseGenerator.tool_exception_case()
    result = await run_runtime(runtime_input)

    assert_task_states(result, expected_statuses={"task_exception": "FAILED"})
    assert_context_state(result, missing_output_keys=["exception_output"])
    assert any("tool exception" in error["message"] for error in result.context_snapshot["errors"])
