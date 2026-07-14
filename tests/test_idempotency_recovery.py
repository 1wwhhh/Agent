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
    return await _setup_graph(runtime_input or TestCaseGenerator.idempotency_replay_case())


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
async def test_irreversible_task_is_not_reexecuted_during_checkpoint_replay():
    checkpoint_dir = _prepare_checkpoint_dir("idempotency_replay_case")
    base = TestCaseGenerator.idempotency_replay_case()
    initial_input = base.model_copy(
        update={"runtime_metadata": {"checkpoint_enabled": True, "checkpoint_dir": str(checkpoint_dir)}}
    )

    initial = await run_runtime(initial_input)
    assert initial.final_output["success"] is True

    manager = RuntimeCheckpointManager(store=LocalCheckpointStore(checkpoint_dir), enabled=True)
    checkpoint = await manager.load_latest_checkpoint(request_id="req_idempotency_replay_case")
    assert checkpoint is not None

    replay_input = base.model_copy(
        update={
            "runtime_metadata": {
                "checkpoint_enabled": True,
                "checkpoint_dir": str(checkpoint_dir),
                "restore_checkpoint_id": checkpoint.checkpoint_id,
                "task_status_overrides": {"task_irreversible": "PENDING"},
                "remove_task_result_keys": ["irreversible_output"],
            }
        }
    )

    replay = await run_runtime(replay_input)
    assert_task_states(replay, expected_statuses={"task_irreversible": "SUCCESS"})
    assert_context_state(replay, expected_output_keys=["irreversible_output"])
    trace_entries = [
        item for item in replay.execution_trace["runtime_tool_trace"] if item["task_id"] == "task_irreversible"
    ]
    assert len(trace_entries) == 1
    idempotency_records = replay.context_snapshot["idempotency_records"]
    assert len(idempotency_records) == 1
    stored_record = next(iter(idempotency_records.values()))
    assert stored_record["irreversible"] is True


def _prepare_checkpoint_dir(case_name: str) -> Path:
    path = Path("outputs") / "test_checkpoints" / case_name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
