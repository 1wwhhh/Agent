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
    return await _setup_graph(runtime_input or TestCaseGenerator.complex_flow_case())


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
async def test_complex_flow_runs_planner_parser_queue_executor_and_aggregator():
    runtime_input = TestCaseGenerator.complex_flow_case()
    result = await run_runtime(runtime_input)

    assert result.final_output["supervisor_route"] == "COMPLEX_TASK"
    assert result.final_output["success"] is True
    assert_task_states(result, expected_statuses={"task_1": "SUCCESS", "task_2": "SUCCESS"})
    assert_context_state(result, expected_output_keys=["analysis", "report"])
    assert_execution_order(result, before_after_pairs=[("task_1", "task_2")])
    assert "planner" in result.execution_trace["node_timings"]
    assert "parser" in result.execution_trace["node_timings"]
    assert "queue" in result.execution_trace["node_timings"]
    assert "executor" in result.execution_trace["node_timings"]
    assert len(result.task_states) == 2
    assert result.metrics.dag_correctness_rate == 1.0


@pytest.mark.asyncio
async def test_sequential_dag_obeys_depends_on_order():
    runtime_input = TestCaseGenerator.sequential_dag_case(depth=3)
    result = await run_runtime(runtime_input)

    assert_task_states(
        result,
        expected_statuses={"task_1": "SUCCESS", "task_2": "SUCCESS", "task_3": "SUCCESS"},
    )
    assert_context_state(result, expected_output_keys=["output_1", "output_2", "output_3"])
    assert_execution_order(result, before_after_pairs=[("task_1", "task_2"), ("task_2", "task_3")])
    join_trace = [item for item in result.execution_trace["runtime_tool_trace"] if item["task_id"] == "task_2"][0]
    assert "output_1" in join_trace["consumed_inputs"]
    assert result.metrics.dag_correctness_rate == 1.0


@pytest.mark.asyncio
async def test_failed_branch_does_not_break_independent_branch():
    runtime_input = TestCaseGenerator.partial_failure_branch_case()
    result = await run_runtime(runtime_input)

    assert_task_states(
        result,
        expected_statuses={
            "task_fail_root": "FAILED",
            "task_fail_child": "CANCELLED",
            "task_ok": "SUCCESS",
        },
    )
    assert_context_state(result, expected_output_keys=["healthy_output"], missing_output_keys=["failed_branch", "blocked_output"])
    assert_execution_order(result, blocked_tasks=["task_fail_child"])
    assert result.final_output["success"] is False
    assert result.metrics.task_success_rate == pytest.approx(1 / 3, rel=1e-3)
