from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from app.context import LocalCheckpointStore, RuntimeCheckpointManager
from tests.support.assertions import (
    assert_context_state as _assert_context_state,
    assert_execution_order as _assert_execution_order,
    assert_task_states as _assert_task_states,
)
from tests.support.runtime_runner import (
    resume_runtime_from_latest_checkpoint as _resume_runtime_from_latest_checkpoint,
    run_runtime as _run_runtime,
    setup_graph as _setup_graph,
)
from tests.support.test_case_generator import TestCaseGenerator


async def setup_graph(runtime_input=None):
    return await _setup_graph(runtime_input or TestCaseGenerator.checkpoint_interrupt_case())


async def run_runtime(runtime_input):
    return await _run_runtime(runtime_input)


async def resume_runtime_from_latest_checkpoint(runtime_input, *, request_id=None, session_id=None):
    return await _resume_runtime_from_latest_checkpoint(runtime_input, request_id=request_id, session_id=session_id)


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
async def test_checkpoint_snapshot_is_json_serializable_and_complete():
    checkpoint_dir = _prepare_checkpoint_dir("checkpoint_snapshot_case")
    base = TestCaseGenerator.context_consistency_case()
    runtime_input = base.model_copy(
        update={
            "name": "checkpoint_snapshot_case",
            "runtime_metadata": {"checkpoint_enabled": True, "checkpoint_dir": str(checkpoint_dir)},
        }
    )

    result = await run_runtime(runtime_input)
    manager = RuntimeCheckpointManager(store=LocalCheckpointStore(checkpoint_dir), enabled=True)
    checkpoint = await manager.load_latest_checkpoint(request_id="req_checkpoint_snapshot_case")

    assert checkpoint is not None
    assert "context" in checkpoint.snapshot_payload
    assert "snapshot" in checkpoint.snapshot_payload
    assert "planned_tasks" in checkpoint.snapshot_payload
    assert "execution_results" in checkpoint.snapshot_payload
    assert "task_results" in checkpoint.snapshot_payload["context"]
    assert "execution_history" in checkpoint.snapshot_payload["context"]
    json.dumps(checkpoint.model_dump(mode="json"), ensure_ascii=False)
    assert result.execution_trace["last_checkpoint"] is not None


@pytest.mark.asyncio
async def test_failed_execution_can_resume_from_latest_checkpoint():
    checkpoint_dir = _prepare_checkpoint_dir("checkpoint_interrupt_case")
    base = TestCaseGenerator.checkpoint_interrupt_case()
    interrupted_input = base.model_copy(
        update={
            "runtime_metadata": {"checkpoint_enabled": True, "checkpoint_dir": str(checkpoint_dir)},
            "graph_metadata": {"interrupt_after_task_count": 1, "executor_batch_limit": 1},
        }
    )

    interrupted = await run_runtime(interrupted_input)
    assert interrupted.final_output["success"] is False

    resumed_input = base.model_copy(
        update={
            "runtime_metadata": {"checkpoint_enabled": True, "checkpoint_dir": str(checkpoint_dir)},
        }
    )
    resumed = await resume_runtime_from_latest_checkpoint(
        resumed_input,
        request_id="req_checkpoint_interrupt_case",
    )

    assert_task_states(
        resumed,
        expected_statuses={"task_stage_1": "SUCCESS", "task_stage_2": "SUCCESS"},
    )
    assert_context_state(resumed, expected_output_keys=["stage_1_output", "stage_2_output"])
    assert_execution_order(resumed, before_after_pairs=[("task_stage_1", "task_stage_2")])
    assert resumed.final_output["success"] is True
    assert resumed.execution_trace["last_checkpoint"] is not None


def _prepare_checkpoint_dir(case_name: str) -> Path:
    path = Path("outputs") / "test_checkpoints" / case_name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
