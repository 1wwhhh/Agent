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
    return await _setup_graph(runtime_input or TestCaseGenerator.context_consistency_case())


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
async def test_context_outputs_are_written_and_read_across_tasks():
    runtime_input = TestCaseGenerator.context_consistency_case()
    result = await run_runtime(runtime_input)

    assert_task_states(
        result,
        expected_statuses={"task_left": "SUCCESS", "task_right": "SUCCESS", "task_join": "SUCCESS"},
    )
    assert_context_state(result, expected_output_keys=["left_output", "right_output", "joined_output"])
    assert_execution_order(result, before_after_pairs=[("task_left", "task_join"), ("task_right", "task_join")])
    join_trace = [item for item in result.execution_trace["runtime_tool_trace"] if item["task_id"] == "task_join"][0]
    assert join_trace["consumed_inputs"]["left_output"] is not None
    assert join_trace["consumed_inputs"]["right_output"] is not None
    assert result.metrics.context_consistency_rate == 1.0


@pytest.mark.asyncio
async def test_context_snapshot_contains_recovery_relevant_runtime_state():
    runtime_input = TestCaseGenerator.context_consistency_case()
    result = await run_runtime(runtime_input)

    shared_data = result.context_snapshot["shared_data"]
    assert "queue_snapshot" in shared_data
    assert "execution_results" in shared_data
    assert "planned_tasks" in shared_data
    assert "final_response" in shared_data
    assert result.final_output["final_output_ready"] is True
