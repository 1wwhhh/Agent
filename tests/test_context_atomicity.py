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
    return await _setup_graph(runtime_input or TestCaseGenerator.shared_key_conflict_case())


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
async def test_parallel_shared_key_writes_do_not_silently_overwrite_context():
    runtime_input = TestCaseGenerator.shared_key_conflict_case()
    result = await run_runtime(runtime_input)

    statuses = {task_id: payload["status"] for task_id, payload in result.task_states.items()}
    assert set(statuses.values()) == {"SUCCESS", "FAILED"}
    assert "conflict_key" in result.context_snapshot["shared_data"]
    assert any("不允许静默覆盖" in error["message"] for error in result.context_snapshot["errors"])
    successful_outputs = [
        task_id
        for task_id, payload in result.task_states.items()
        if payload["status"] == "SUCCESS" and payload["output_key"] in result.context_snapshot["task_results"]
    ]
    assert len(successful_outputs) == 1
