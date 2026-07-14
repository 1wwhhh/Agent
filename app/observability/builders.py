from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from enum import Enum
from typing import Any

from app.schemas.observability import PersistedExecutionTrace, ReplayResult, RequestMetricsSnapshot
from app.schemas.task import TaskStatus
from app.state import LangGraphState

from .snapshots import DebugSnapshot, MetricsSnapshot, ReplaySnapshot, RuntimeTraceSnapshot


def build_latency_breakdown(node_timings: dict[str, Any]) -> dict[str, float]:
    latency = {name: float(data.get("duration_ms", 0.0)) for name, data in node_timings.items()}
    latency["total"] = round(sum(latency.values()), 3)
    return latency


def build_tool_calls(state: LangGraphState) -> list[dict[str, Any]]:
    tasks_by_id = state.context.tasks
    task_results = state.context.task_results
    tool_calls: list[dict[str, Any]] = []
    for record in state.context.tool_call_chain:
        task = tasks_by_id.get(record.task_id) if record.task_id else None
        output = None
        if task is not None and task.output_key in task_results:
            output = task_results.get(task.output_key)
        elif isinstance(record.metadata, dict):
            output = record.metadata.get("output")

        tool_calls.append(
            {
                "tool_name": record.tool_name,
                "task_id": record.task_id,
                "input": record.metadata.get("input") if isinstance(record.metadata, dict) else None,
                "output": deepcopy(output),
                "status": record.status,
                "timestamp": record.started_at.isoformat(),
                "latency_ms": record.latency_ms,
            }
        )
    return tool_calls


def build_task_graph_trace(state: LangGraphState) -> dict[str, Any]:
    planned_by_id = {task.task_id: task for task in state.planned_tasks}
    runtime_by_id = dict(state.context.tasks)
    ordered_task_ids = list(dict.fromkeys([*planned_by_id.keys(), *runtime_by_id.keys()]))

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    retry_tasks: list[str] = []
    cancelled_tasks: list[str] = []
    timeout_tasks: list[str] = []

    for task_id in ordered_task_ids:
        task = runtime_by_id.get(task_id) or planned_by_id.get(task_id)
        if task is None:
            continue
        status = _status_value(task.status)
        node = {
            "task_id": task.task_id,
            "name": task.task_name,
            "tool": task.tool,
            "status": status,
            "retry_count": int(task.retry_count),
            "depends_on": list(task.depends_on),
        }
        nodes.append(node)
        edges.extend({"from": dependency, "to": task.task_id} for dependency in task.depends_on)

        if int(task.retry_count) > 0 or status == TaskStatus.RETRY.value:
            retry_tasks.append(task.task_id)
        if status == TaskStatus.CANCELLED.value:
            cancelled_tasks.append(task.task_id)
        if status == TaskStatus.TIMEOUT.value:
            timeout_tasks.append(task.task_id)

    execution_sequence = [
        {
            "task_id": record.task_id,
            "status": _status_value(record.status),
            "attempt": int(record.attempt),
            "recorded_at": record.recorded_at.isoformat(),
            "idempotency_key": record.idempotency_key,
            "message": record.message,
        }
        for record in state.context.execution_history
    ]

    return {
        "nodes": nodes,
        "edges": edges,
        "execution_sequence": execution_sequence,
        "retry_tasks": list(dict.fromkeys(retry_tasks)),
        "cancelled_tasks": list(dict.fromkeys(cancelled_tasks)),
        "timeout_tasks": list(dict.fromkeys(timeout_tasks)),
    }


def build_metrics_snapshot(state: LangGraphState, *, finished_at: datetime) -> MetricsSnapshot:
    started_at = state.context.runtime.timestamp
    tasks = list(state.context.tasks.values())
    total_tasks = len(tasks)
    safe_total = total_tasks or 1

    successful_tasks = sum(1 for task in tasks if _status_value(task.status) == TaskStatus.SUCCESS.value)
    failed_tasks = sum(1 for task in tasks if _status_value(task.status) == TaskStatus.FAILED.value)
    cancelled_tasks = sum(1 for task in tasks if _status_value(task.status) == TaskStatus.CANCELLED.value)
    timeout_tasks = sum(1 for task in tasks if _status_value(task.status) == TaskStatus.TIMEOUT.value)
    retry_count = sum(int(task.retry_count) for task in tasks)
    successful_output_keys = sum(
        1
        for task in tasks
        if _status_value(task.status) == TaskStatus.SUCCESS.value and task.output_key in state.context.task_results
    )

    metadata = state.metadata
    llm_calls = list(metadata.get("llm_calls", []))
    parser_repair_history = list(metadata.get("parser_repair_history", []))
    routing_history = list(metadata.get("routing_history", []))
    node_timings = dict(metadata.get("node_timings", {}))

    latency = build_latency_breakdown(node_timings)
    runtime_latency_ms = max(0, int((finished_at - started_at).total_seconds() * 1000))

    return MetricsSnapshot(
        request_id=state.context.runtime.request_id,
        session_id=state.context.runtime.session_id,
        phase=state.phase.value,
        supervisor_route=state.supervisor_route,
        total_tasks=total_tasks,
        successful_tasks=successful_tasks,
        failed_tasks=failed_tasks,
        cancelled_tasks=cancelled_tasks,
        timeout_tasks=timeout_tasks,
        retry_count=retry_count,
        llm_call_count=len(llm_calls),
        fallback_count=sum(1 for item in llm_calls if isinstance(item, dict) and item.get("fallback_used") is True),
        parser_repair_count=len(parser_repair_history),
        routing_denied_count=sum(
            1 for item in routing_history if isinstance(item, dict) and item.get("routing_result") == "DENIED"
        ),
        runtime_latency_ms=runtime_latency_ms,
        task_success_rate=successful_tasks / safe_total,
        dag_correctness_rate=_calculate_dag_correctness(state),
        retry_rate=retry_count / safe_total,
        context_consistency_rate=successful_output_keys / max(successful_tasks, 1),
        latency=latency,
    )


def build_request_metrics_snapshot(
    metrics: MetricsSnapshot,
    *,
    recorded_at: datetime | None = None,
) -> RequestMetricsSnapshot:
    payload = {
        "request_id": metrics.request_id,
        "session_id": metrics.session_id,
        "phase": metrics.phase,
        "supervisor_route": metrics.supervisor_route,
        "task_success_rate": metrics.task_success_rate,
        "dag_correctness_rate": metrics.dag_correctness_rate,
        "retry_rate": metrics.retry_rate,
        "retry_count": metrics.retry_count,
        "context_consistency_rate": metrics.context_consistency_rate,
        "latency": deepcopy(metrics.latency),
    }
    if recorded_at is not None:
        payload["recorded_at"] = recorded_at.isoformat()
    return RequestMetricsSnapshot(**payload)


def build_runtime_trace_snapshot(state: LangGraphState, *, finished_at: datetime) -> RuntimeTraceSnapshot:
    metrics = build_metrics_snapshot(state, finished_at=finished_at)
    metadata = state.metadata
    debug_events = _build_debug_events(state)

    return RuntimeTraceSnapshot(
        request_id=state.context.runtime.request_id,
        session_id=state.context.runtime.session_id,
        started_at=state.context.runtime.timestamp.isoformat(),
        finished_at=finished_at.isoformat(),
        status=_build_runtime_status(state),
        latency_ms=metrics.runtime_latency_ms,
        task_graph=build_task_graph_trace(state),
        task_results=deepcopy(state.context.task_results),
        task_failures=deepcopy(list(metadata.get("task_failures", []))),
        parser_repair_history=deepcopy(list(metadata.get("parser_repair_history", []))),
        routing_history=deepcopy(list(metadata.get("routing_history", []))),
        llm_calls=deepcopy(list(metadata.get("llm_calls", []))),
        metrics=metrics,
        errors=[error.model_dump(mode="json") for error in state.context.errors],
        debug_events=debug_events,
    )


def build_debug_snapshot(state: LangGraphState, *, finished_at: datetime) -> DebugSnapshot:
    planner_prompt = (
        state.planner_prompt.model_dump(mode="json")
        if state.planner_prompt is not None
        else state.context.shared_data.get("planner_prompt")
    )
    parsed_plan = (
        state.parsed_plan.model_dump(mode="json")
        if state.parsed_plan is not None
        else state.context.shared_data.get("parsed_plan")
    )

    return DebugSnapshot(
        request_id=state.context.runtime.request_id,
        planner_output={
            "planner_prompt": deepcopy(planner_prompt),
            "raw_plan_text": state.raw_plan_text or state.context.shared_data.get("raw_plan_text"),
            "parsed_plan": deepcopy(parsed_plan),
        },
        repair_attempts=deepcopy(list(state.metadata.get("parser_repair_history", []))),
        routing_decisions=deepcopy(list(state.metadata.get("routing_history", []))),
        executor_events=[record.model_dump(mode="json") for record in state.context.execution_history],
        llm_calls=deepcopy(list(state.metadata.get("llm_calls", []))),
        metrics=build_metrics_snapshot(state, finished_at=finished_at),
        errors=[error.model_dump(mode="json") for error in state.context.errors],
    )


def build_replay_snapshot(
    trace: PersistedExecutionTrace,
    replay_result: ReplayResult | None = None,
) -> ReplaySnapshot:
    replay_steps = replay_result.steps if replay_result is not None else []
    last_step = _select_last_stateful_step(replay_steps)
    if last_step is None and trace.events:
        event = trace.events[-1]
        task_graph = deepcopy(event.task_graph)
        runtime_context_snapshot = deepcopy(event.context_snapshot)
    else:
        task_graph = deepcopy(last_step.task_graph) if last_step is not None else {}
        runtime_context_snapshot = deepcopy(last_step.context_snapshot) if last_step is not None else {}

    final_output = None
    if replay_result is not None and replay_result.final_output:
        final_output = deepcopy(replay_result.final_output)
    elif runtime_context_snapshot.get("final_output") is not None:
        final_output = deepcopy(runtime_context_snapshot.get("final_output"))

    metadata = final_output.get("metadata", {}) if isinstance(final_output, dict) else {}
    shared_data = runtime_context_snapshot.get("shared_data", {}) if isinstance(runtime_context_snapshot, dict) else {}
    runtime_payload = runtime_context_snapshot.get("runtime", {}) if isinstance(runtime_context_snapshot, dict) else {}
    start_event = next((event for event in trace.events if event.layer == "request" and event.event == "start"), None)

    raw_user_input = runtime_payload.get("user_input")
    if raw_user_input is None and start_event is not None:
        raw_user_input = start_event.metadata.get("input")

    planner_raw_output = shared_data.get("raw_plan_text")
    repaired_output = shared_data.get("parsed_plan")

    return ReplaySnapshot(
        request_id=trace.request_id,
        session_id=trace.session_id,
        raw_user_input=raw_user_input,
        planner_raw_output=deepcopy(planner_raw_output),
        repaired_output=deepcopy(repaired_output),
        task_graph=task_graph,
        runtime_context_snapshot=runtime_context_snapshot,
        routing_decisions=deepcopy(list(metadata.get("routing_history", []))),
        llm_calls=deepcopy(list(metadata.get("llm_calls", []))),
        final_output=deepcopy(final_output),
        errors=deepcopy(list(runtime_context_snapshot.get("errors", []))) if runtime_context_snapshot else [],
    )


def _select_last_stateful_step(steps: list[Any]) -> Any | None:
    stateful_steps = [step for step in steps if step.context_snapshot or step.task_states or step.task_graph]
    if stateful_steps:
        return stateful_steps[-1]
    if steps:
        return steps[-1]
    return None


def _build_debug_events(state: LangGraphState) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for node_name in list(state.metadata.get("completed_nodes", [])):
        events.append({"type": "completed_node", "node": node_name})
    for error in list(state.context.shared_data.get("executor_nonfatal_errors", [])):
        events.append({"type": "executor_nonfatal_error", "payload": deepcopy(error)})
    return events


def _build_runtime_status(state: LangGraphState) -> str:
    if state.phase.value == "FAILED":
        return "FAILED"

    failed_statuses = {TaskStatus.FAILED.value, TaskStatus.TIMEOUT.value, TaskStatus.CANCELLED.value}
    if state.agent_state.failed_task_ids or any(_status_value(task.status) in failed_statuses for task in state.context.tasks.values()):
        return "FAILED"

    if state.agent_state.final_output_ready or state.phase.value == "COMPLETED":
        return "SUCCESS"

    return "RUNNING"


def _calculate_dag_correctness(state: LangGraphState) -> float:
    tasks = list(state.context.tasks.values())
    edge_count = sum(len(task.depends_on) for task in tasks)
    planned_task_ids = {task.task_id for task in state.planned_tasks}
    runtime_ids = set(state.context.tasks.keys())

    if not runtime_ids:
        return 0.0
    if planned_task_ids and planned_task_ids != runtime_ids:
        return 0.0
    if edge_count == 0:
        return 1.0

    success_indices: dict[str, int] = {}
    running_indices: dict[str, int] = {}
    for index, record in enumerate(state.context.execution_history):
        record_status = _status_value(record.status)
        if record_status == TaskStatus.SUCCESS.value and record.task_id not in success_indices:
            success_indices[record.task_id] = index
        if record_status == TaskStatus.RUNNING.value and record.task_id not in running_indices:
            running_indices[record.task_id] = index

    satisfied = 0
    for task in tasks:
        task_run_index = running_indices.get(task.task_id, float("inf"))
        for dependency in task.depends_on:
            dependency_success_index = success_indices.get(dependency, float("-inf"))
            if dependency_success_index < task_run_index:
                satisfied += 1
    return satisfied / edge_count


def _status_value(status: Any) -> str:
    if isinstance(status, Enum):
        return str(status.value)
    return str(status)
