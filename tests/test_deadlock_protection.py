from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from app.context import LocalCheckpointStore, RuntimeCheckpointManager
from tests.support.assertions import (
    assert_context_state as _assert_context_state,
    assert_execution_order as _assert_execution_order,
    assert_task_states as _assert_task_states,
)
from tests.support.runtime_runner import run_runtime as _run_runtime, setup_graph as _setup_graph
from tests.support.test_case_generator import TestCaseGenerator


async def setup_graph(runtime_input=None):
    return await _setup_graph(runtime_input or TestCaseGenerator.cyclic_dependency_case())


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
async def test_circular_dependency_fails_fast_with_explicit_error():
    runtime_input = TestCaseGenerator.cyclic_dependency_case()
    result = await run_runtime(runtime_input)

    assert result.final_output["success"] is False
    assert "循环依赖" in result.final_output["metadata"]["failure"]["message"]


@pytest.mark.asyncio
async def test_missing_dependency_fails_fast_as_invalid_dag():
    runtime_input = TestCaseGenerator.missing_dependency_case()
    result = await run_runtime(runtime_input)

    assert result.final_output["success"] is False
    assert "undefined dependencies" in result.final_output["metadata"]["failure"]["message"]


@pytest.mark.asyncio
async def test_permanently_blocked_dag_fails_fast_during_resume():
    checkpoint_dir = _prepare_checkpoint_dir("deadlock_resume_case")
    base = TestCaseGenerator.checkpoint_interrupt_case()
    initial_input = base.model_copy(
        update={"runtime_metadata": {"checkpoint_enabled": True, "checkpoint_dir": str(checkpoint_dir)}}
    )
    initial = await run_runtime(initial_input)
    assert initial.final_output["success"] is True

    manager = RuntimeCheckpointManager(store=LocalCheckpointStore(checkpoint_dir), enabled=True)
    checkpoint = await manager.load_latest_checkpoint(request_id="req_checkpoint_interrupt_case")
    assert checkpoint is not None

    blocked_input = base.model_copy(
        update={
            "runtime_metadata": {
                "checkpoint_enabled": True,
                "checkpoint_dir": str(checkpoint_dir),
                "restore_checkpoint_id": checkpoint.checkpoint_id,
                "task_status_overrides": {"task_stage_1": "FAILED", "task_stage_2": "PENDING"},
                "remove_task_result_keys": ["stage_2_output"],
            }
        }
    )

    blocked = await run_runtime(blocked_input)
    assert blocked.final_output["success"] is False
    assert "永久阻塞" in blocked.final_output["metadata"]["failure"]["message"]


def _prepare_checkpoint_dir(case_name: str) -> Path:
    path = Path("outputs") / "test_checkpoints" / case_name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
