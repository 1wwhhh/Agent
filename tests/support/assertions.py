from __future__ import annotations

from tests.support.models import TestResult


def assert_task_states(result: TestResult, *, expected_statuses: dict[str, str]) -> None:
    for task_id, expected_status in expected_statuses.items():
        assert task_id in result.task_states, f"missing task state for {task_id}"
        assert result.task_states[task_id]["status"] == expected_status, (
            f"unexpected status for {task_id}: "
            f"{result.task_states[task_id]['status']} != {expected_status}"
        )


def assert_context_state(
    result: TestResult,
    *,
    expected_output_keys: list[str] | None = None,
    missing_output_keys: list[str] | None = None,
) -> None:
    task_results = result.context_snapshot["task_results"]
    for output_key in expected_output_keys or []:
        assert output_key in task_results, f"expected output_key '{output_key}' not found in context"
    for output_key in missing_output_keys or []:
        assert output_key not in task_results, f"unexpected output_key '{output_key}' found in context"


def assert_execution_order(
    result: TestResult,
    *,
    before_after_pairs: list[tuple[str, str]] | None = None,
    blocked_tasks: list[str] | None = None,
) -> None:
    execution_history = result.execution_trace["execution_history"]
    running_indices: dict[str, int] = {}
    for index, record in enumerate(execution_history):
        if record["status"] == "RUNNING" and record["task_id"] not in running_indices:
            running_indices[record["task_id"]] = index

    for before_task, after_task in before_after_pairs or []:
        assert before_task in running_indices, f"missing RUNNING record for {before_task}"
        assert after_task in running_indices, f"missing RUNNING record for {after_task}"
        assert running_indices[before_task] < running_indices[after_task], (
            f"execution order violated: {before_task} should run before {after_task}"
        )

    task_states = result.task_states
    for task_id in blocked_tasks or []:
        assert task_states[task_id]["status"] in {"CANCELLED", "PENDING", "FAILED", "TIMEOUT"}, (
            f"blocked task {task_id} unexpectedly executed with status {task_states[task_id]['status']}"
        )
