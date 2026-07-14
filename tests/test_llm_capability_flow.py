from __future__ import annotations

import pytest

from app.prompts import SUPERVISOR_PROMPT_VERSION, TASK_PLANNER_PROMPT_VERSION
from tests.support.assertions import (
    assert_context_state as _assert_context_state,
    assert_execution_order as _assert_execution_order,
    assert_task_states as _assert_task_states,
)
from tests.support.runtime_runner import run_runtime as _run_runtime, setup_graph as _setup_graph
from tests.support.test_case_generator import TestCaseGenerator


async def setup_graph(runtime_input=None):
    return await _setup_graph(runtime_input or TestCaseGenerator.structured_llm_complex_flow_case())


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
async def test_structured_supervisor_and_planner_run_through_langgraph():
    runtime_input = TestCaseGenerator.structured_llm_complex_flow_case()
    result = await run_runtime(runtime_input)

    assert result.final_output["supervisor_route"] == "COMPLEX_TASK"
    assert result.final_output["success"] is True
    assert_task_states(result, expected_statuses={"task_1": "SUCCESS", "task_2": "SUCCESS"})
    assert_context_state(result, expected_output_keys=["analysis", "report"])
    assert_execution_order(result, before_after_pairs=[("task_1", "task_2")])
    assert result.context_snapshot["shared_data"]["supervisor_decision"]["route"] == "COMPLEX_TASK"
    assert result.context_snapshot["shared_data"]["supervisor_trace"]["prompt_version"] == SUPERVISOR_PROMPT_VERSION
    assert result.context_snapshot["shared_data"]["planner_trace"]["prompt_version"] == TASK_PLANNER_PROMPT_VERSION
