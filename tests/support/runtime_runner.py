from __future__ import annotations

from enum import Enum
from typing import Any

from app.context import LocalCheckpointStore, RuntimeCheckpointManager
from app.graph import GraphRuntimeDependencies, LangGraphState, build_langgraph_runtime, coerce_langgraph_state
from app.router import TaskRouter
from app.router.capability import capability_from_tool
from app.schemas.task import TaskStatus
from tests.support.models import RuntimeMetrics, RuntimeTestInput, TestResult
from tests.support.tools import (
    ParserRepairTestLLMClient,
    PlannerTestTool,
    RuntimeTestTool,
    build_planner_agent,
    build_supervisor_agent,
)


async def setup_graph(runtime_input: RuntimeTestInput):
    router = TaskRouter()
    reason_tool = RuntimeTestTool(
        name="llm_reason_tool",
        description="Deterministic reasoning test tool",
        tags=["reasoning", "llm"],
    )
    text_tool = RuntimeTestTool(
        name="text_generate_tool",
        description="Deterministic text test tool",
        tags=["text", "generation", "llm"],
    )
    await router.register_tools(
        [
            (reason_tool, capability_from_tool(reason_tool)),
            (text_tool, capability_from_tool(text_tool)),
        ]
    )

    checkpoint_manager = None
    if runtime_input.runtime_metadata.get("checkpoint_enabled"):
        checkpoint_dir = runtime_input.runtime_metadata.get("checkpoint_dir", "outputs/checkpoints")
        checkpoint_manager = RuntimeCheckpointManager(store=LocalCheckpointStore(checkpoint_dir), enabled=True)

    planner_tool = PlannerTestTool(
        name="planner_test_tool",
        description="Deterministic planner tool for runtime integration tests",
        scenario=runtime_input.planner_scenario,
        test_input=runtime_input,
    )
    repair_llm_client = ParserRepairTestLLMClient(test_input=runtime_input)
    graph = build_langgraph_runtime(
        GraphRuntimeDependencies(
            router=router,
            supervisor_agent=build_supervisor_agent(runtime_input) if runtime_input.use_supervisor_agent else None,
            planner_tool=planner_tool,
            planner_agent=build_planner_agent(runtime_input) if runtime_input.use_planner_agent else None,
            repair_llm_client=repair_llm_client,
            checkpoint_manager=checkpoint_manager,
            queue_max_concurrency=runtime_input.queue_max_concurrency,
            planner_model_name=runtime_input.planner_model_name,
            planner_temperature=runtime_input.planner_temperature,
        )
    )
    setattr(graph, "_repair_llm_client", repair_llm_client)
    return graph


async def run_runtime(runtime_input: RuntimeTestInput) -> TestResult:
    return await execute_test_case(runtime_input)


async def execute_test_case(runtime_input: RuntimeTestInput) -> TestResult:
    graph = await setup_graph(runtime_input)
    state = await _build_initial_state(runtime_input)
    result_state = coerce_langgraph_state(await graph.ainvoke(state))
    repair_llm_client = getattr(graph, "_repair_llm_client", None)
    repair_llm_call_count = getattr(repair_llm_client, "call_count", 0)
    return _build_test_result(result_state, repair_llm_call_count=repair_llm_call_count)


async def resume_runtime_from_latest_checkpoint(
    runtime_input: RuntimeTestInput,
    *,
    request_id: str | None = None,
    session_id: str | None = None,
) -> TestResult:
    graph = await setup_graph(runtime_input)
    checkpoint_dir = runtime_input.runtime_metadata.get("checkpoint_dir", "outputs/checkpoints")
    manager = RuntimeCheckpointManager(store=LocalCheckpointStore(checkpoint_dir), enabled=True)
    checkpoint = await manager.load_latest_checkpoint(request_id=request_id, session_id=session_id)
    if checkpoint is None:
        raise FileNotFoundError("未找到可恢复的检查点。")
    state = await manager.restore_state(checkpoint)
    result_state = coerce_langgraph_state(await graph.ainvoke(state))
    repair_llm_client = getattr(graph, "_repair_llm_client", None)
    repair_llm_call_count = getattr(repair_llm_client, "call_count", 0)
    return _build_test_result(result_state, repair_llm_call_count=repair_llm_call_count)


def _build_test_result(state: LangGraphState, *, repair_llm_call_count: int = 0) -> TestResult:
    task_states = {
        task_id: {
            "status": task.status.value if hasattr(task.status, "value") else str(task.status),
            "retry_count": task.retry_count,
            "max_retry": task.max_retry,
            "depends_on": list(task.depends_on),
            "output_key": task.output_key,
            "tool": task.tool,
            "idempotency_key": task.idempotency_key,
            "irreversible": task.irreversible,
        }
        for task_id, task in state.context.tasks.items()
    }
    context_snapshot = {
        "runtime": state.context.runtime.model_dump(mode="json"),
        "tasks": {task_id: task.model_dump(mode="json") for task_id, task in state.context.tasks.items()},
        "task_results": state.context.task_results,
        "shared_data": state.context.shared_data,
        "errors": [item.model_dump(mode="json") for item in state.context.errors],
        "token_usage": [item.model_dump(mode="json") for item in state.context.token_usage],
        "tool_call_chain": [item.model_dump(mode="json") for item in state.context.tool_call_chain],
        "idempotency_records": {
            key: item.model_dump(mode="json") for key, item in state.context.idempotency_records.items()
        },
        "execution_history": [item.model_dump(mode="json") for item in state.context.execution_history],
        "final_output": state.context.final_output,
    }
    execution_trace = {
        "phase": state.phase.value,
        "supervisor_route": state.supervisor_route,
        "node_timings": state.metadata.get("node_timings", {}),
        "tool_invocations": context_snapshot["tool_call_chain"],
        "execution_history": context_snapshot["execution_history"],
        "runtime_tool_trace": state.context.shared_data.get("runtime_tool_trace", []),
        "latest_route_decision": state.latest_route_decision.model_dump(mode="json") if state.latest_route_decision else None,
        "last_checkpoint": state.context.shared_data.get("last_checkpoint"),
        "repair_llm_call_count": repair_llm_call_count,
    }
    metrics = _compute_metrics(state)
    return TestResult(
        final_output=state.final_response,
        task_states=task_states,
        context_snapshot=context_snapshot,
        execution_trace=execution_trace,
        metrics=metrics,
    )


def _compute_metrics(state: LangGraphState) -> RuntimeMetrics:
    tasks = list(state.context.tasks.values())
    total_tasks = len(tasks) or 1
    successful_tasks = sum(1 for task in tasks if _status_value(task.status) == "SUCCESS")
    retries = sum(task.retry_count for task in tasks)
    successful_output_keys = sum(
        1 for task in tasks if _status_value(task.status) == "SUCCESS" and task.output_key in state.context.task_results
    )
    dag_correctness = _calculate_dag_correctness(state)

    node_timings = state.metadata.get("node_timings", {})
    latencies_ms = {name: float(data.get("duration_ms", 0.0)) for name, data in node_timings.items()}
    latencies_ms["total"] = round(sum(latencies_ms.values()), 3)

    return RuntimeMetrics(
        task_success_rate=successful_tasks / total_tasks,
        dag_correctness_rate=dag_correctness,
        retry_rate=retries / total_tasks,
        context_consistency_rate=successful_output_keys / max(successful_tasks, 1),
        latencies_ms=latencies_ms,
    )


def _calculate_dag_correctness(state: LangGraphState) -> float:
    tasks = list(state.context.tasks.values())
    edge_count = sum(len(task.depends_on) for task in tasks)
    planned_task_ids = {task.task_id for task in state.planned_tasks}
    runtime_ids = set(state.context.tasks.keys())
    if not runtime_ids:
        return 0.0
    if planned_task_ids and planned_task_ids != runtime_ids:
        return 0.0

    success_indices = {}
    running_indices = {}
    for index, record in enumerate(state.context.execution_history):
        if _status_value(record.status) == "SUCCESS" and record.task_id not in success_indices:
            success_indices[record.task_id] = index
        if _status_value(record.status) == "RUNNING" and record.task_id not in running_indices:
            running_indices[record.task_id] = index

    if edge_count == 0:
        return 1.0

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


async def _build_initial_state(runtime_input: RuntimeTestInput) -> LangGraphState:
    restore_checkpoint_id = runtime_input.runtime_metadata.get("restore_checkpoint_id")
    if restore_checkpoint_id:
        checkpoint_dir = runtime_input.runtime_metadata.get("checkpoint_dir", "outputs/checkpoints")
        manager = RuntimeCheckpointManager(store=LocalCheckpointStore(checkpoint_dir), enabled=True)
        checkpoint = await manager.load_checkpoint(str(restore_checkpoint_id))
        state = await manager.restore_state(checkpoint)
        _apply_state_overrides(state, runtime_input)
        return state

    state = LangGraphState.create(
        request_id=f"req_{runtime_input.name}",
        session_id=f"sess_{runtime_input.name}",
        user_input=runtime_input.user_input,
        runtime_metadata=runtime_input.runtime_metadata,
        graph_metadata={**runtime_input.graph_metadata, **({"force_route": runtime_input.force_route} if runtime_input.force_route else {})},
    )
    _apply_state_overrides(state, runtime_input)
    return state


def _apply_state_overrides(state: LangGraphState, runtime_input: RuntimeTestInput) -> None:
    status_overrides = runtime_input.runtime_metadata.get("task_status_overrides", {})
    for task_id, status in status_overrides.items():
        if task_id in state.context.tasks:
            state.context.tasks[task_id].status = TaskStatus(str(status))
        for planned_task in state.planned_tasks:
            if planned_task.task_id == task_id:
                planned_task.status = TaskStatus(str(status))

    for output_key in runtime_input.runtime_metadata.get("remove_task_result_keys", []):
        state.context.task_results.pop(str(output_key), None)

    for key in runtime_input.runtime_metadata.get("remove_idempotency_keys", []):
        state.context.idempotency_records.pop(str(key), None)

    if status_overrides:
        state.agent_state.pending_task_ids.clear()
        state.agent_state.completed_task_ids.clear()
        state.agent_state.failed_task_ids.clear()
        state.agent_state.current_task_id = None
        state.agent_state.final_output_ready = False
        for task in state.context.tasks.values():
            state.agent_state.sync_task_status(task)
