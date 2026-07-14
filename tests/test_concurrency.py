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
    return await _setup_graph(runtime_input or TestCaseGenerator.parallel_dag_case())


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
async def test_parallel_tasks_run_without_state_pollution():
    runtime_input = TestCaseGenerator.parallel_dag_case()
    result = await run_runtime(runtime_input)

    assert_task_states(
        result,
        expected_statuses={"task_a": "SUCCESS", "task_b": "SUCCESS", "task_merge": "SUCCESS"},
    )
    assert_context_state(result, expected_output_keys=["branch_a", "branch_b", "merged"])
    assert_execution_order(result, before_after_pairs=[("task_a", "task_merge"), ("task_b", "task_merge")])
    assert result.context_snapshot["task_results"]["branch_a"]["task_id"] == "task_a"
    assert result.context_snapshot["task_results"]["branch_b"]["task_id"] == "task_b"
    assert result.metrics.context_consistency_rate == 1.0


@pytest.mark.asyncio
async def test_stress_fifty_tasks_complete_without_deadlock():
    runtime_input = TestCaseGenerator.stress_50_tasks_case()
    result = await run_runtime(runtime_input)

    expected_statuses = {f"task_{index + 1}": "SUCCESS" for index in range(50)}
    assert_task_states(result, expected_statuses=expected_statuses)
    assert len(result.context_snapshot["task_results"]) == 50
    assert result.metrics.task_success_rate == 1.0
    assert result.metrics.dag_correctness_rate == 1.0
    assert result.execution_trace["node_timings"]["executor"]["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_deep_dag_preserves_order_across_many_layers():
    runtime_input = TestCaseGenerator.deep_dag_case(depth=8)
    result = await run_runtime(runtime_input)

    expected_statuses = {f"task_{index + 1}": "SUCCESS" for index in range(8)}
    assert_task_states(result, expected_statuses=expected_statuses)
    expected_pairs = [(f"task_{index}", f"task_{index + 1}") for index in range(1, 8)]
    assert_execution_order(result, before_after_pairs=expected_pairs)
    assert result.metrics.dag_correctness_rate == 1.0
