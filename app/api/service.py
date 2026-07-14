from __future__ import annotations

from collections.abc import Awaitable, Callable
from enum import Enum
import logging
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

from app.api.bootstrap import build_env_runtime_components
from app.api.schemas import AgentRequest, AgentResponse, AgentTaskState
from app.context import LocalCheckpointStore, RuntimeCheckpointManager
from app.graph import GraphRuntimeDependencies, LangGraphState, build_langgraph_runtime, coerce_langgraph_state
from app.observability import (
    LocalExecutionTraceStore,
    RuntimeMetricsCollector,
    RuntimeReplayEngine,
    build_debug_snapshot,
    build_latency_breakdown,
    build_metrics_snapshot,
    build_replay_snapshot,
    build_request_metrics_snapshot,
    build_runtime_trace_snapshot,
    build_task_graph_trace,
    build_tool_calls,
    safe_observe,
)
from app.observability.snapshots import DebugSnapshot, MetricsSnapshot, RuntimeTraceSnapshot
from app.schemas.observability import ReplayMode
from app.schemas.task import utc_now
from app.utils import bind_request_id, clear_request_id, configure_runtime_logger, runtime_log, runtime_progress

RuntimeComponentsBuilder = Callable[[AgentRequest], Awaitable[Any]]
_runtime_components_builder: RuntimeComponentsBuilder | None = None
_DEFAULT_TRACE_DIRECTORY = Path("outputs") / "runtime_traces"
_runtime_trace_store = LocalExecutionTraceStore(_DEFAULT_TRACE_DIRECTORY)
_runtime_replay_engine = RuntimeReplayEngine(trace_store=_runtime_trace_store)
_runtime_metrics_collector = RuntimeMetricsCollector()
LOGGER = configure_runtime_logger()


def _build_runtime_metadata(agent_request: AgentRequest) -> dict[str, Any] | None:
    metadata: dict[str, Any] = {}
    if agent_request.client_timezone:
        metadata["client_timezone"] = agent_request.client_timezone
    return metadata or None


async def run_runtime(
    agent_request: AgentRequest,
    *,
    debug: bool = False,
    replay: bool = False,
) -> AgentResponse:
    # 直接判断是不是replay
    if replay:
        return await _replay_runtime(agent_request=agent_request, debug=debug)

    request_id = f"req_{uuid4().hex}"
    session_id = agent_request.session_id or f"sess_{uuid4().hex}"
    request_started_at = time.perf_counter()
    request_token = bind_request_id(request_id)
    components = None
    runtime_progress(
        step="request",
        status="收到请求",
        detail=f"输入={agent_request.user_input[:80]}",
        request_id=request_id,
        session_id=session_id,
    )
    # 添加日志信息，写trace文件
    try:
        runtime_log(
            layer="request",
            event="start",
            data={
                "session_id": session_id,
                "input": agent_request.user_input,
                "debug": debug,
                "replay": False,
            },
            logger=LOGGER,
        )
        await safe_observe(
            "trace_store.record_request_event.start",
            _runtime_trace_store.record_request_event,
            request_id=request_id,
            session_id=session_id,
            layer="request",
            event="start",
            metadata={"input": agent_request.user_input, "debug": debug},
        )
        # 提前准备配置
        runtime_progress(step="components", status="初始化", detail="正在构建运行时组件", session_id=session_id)
        components = await _get_runtime_components_builder()(agent_request)
        runtime_progress(step="components", status="就绪", detail="运行时组件构建完成", session_id=session_id)
        checkpoint_manager = _resolve_checkpoint_manager(components)
        repair_llm_client = _resolve_repair_llm_client(components)
        # 创建 Graph 并打包依赖
        graph = build_langgraph_runtime(
            GraphRuntimeDependencies(
                router=components.router,
                supervisor_agent=getattr(components, "supervisor_agent", None),
                planner_agent=getattr(components, "planner_agent", None),
                planner_tool=getattr(components, "planner_tool", None),
                repair_llm_client=repair_llm_client,
                checkpoint_manager=checkpoint_manager,
            )
        )
        # 把原始请求装进 LangGraphState
        initial_state = LangGraphState.create(
            request_id=request_id,
            session_id=session_id,
            user_input=agent_request.user_input,
            runtime_metadata=_build_runtime_metadata(agent_request),
        )
        # 把请求交给工作流去跑，结果统一整理成标准状态对象
        final_state = coerce_langgraph_state(await graph.ainvoke(initial_state))
        finished_at = utc_now()
        metrics_snapshot = build_metrics_snapshot(final_state, finished_at=finished_at)
        request_metrics_snapshot = build_request_metrics_snapshot(metrics_snapshot, recorded_at=finished_at)
        runtime_trace_snapshot = build_runtime_trace_snapshot(final_state, finished_at=finished_at)
        debug_snapshot = build_debug_snapshot(final_state, finished_at=finished_at) if debug else None
        response = _build_agent_response(
            final_state=final_state,
            debug=debug,
            finished_at=finished_at,
            metrics_snapshot=metrics_snapshot,
            runtime_trace_snapshot=runtime_trace_snapshot,
            debug_snapshot=debug_snapshot,
        )
        # 记录到全局 metrics collecto
        await safe_observe(
            "metrics_collector.record",
            _runtime_metrics_collector.record,
            request_metrics_snapshot,
        )
        # 把指标挂到 trace 文件
        await safe_observe(
            "trace_store.attach_metrics",
            _runtime_trace_store.attach_metrics,
            request_id,
            request_metrics_snapshot,
        )
        await safe_observe(
            "trace_store.record_request_event.end",
            _runtime_trace_store.record_request_event,
            request_id=request_id,
            session_id=session_id,
            layer="request",
            event="end",
            metadata={
                "phase": final_state.phase.value,
                "success": response.result.get("success"),
                "steps": response.trace.get("steps", []),
            },
        )
        if debug:
            response.trace["metrics_export"] = await safe_observe(
                "metrics_collector.export_metrics",
                _runtime_metrics_collector.export_metrics,
                default={},
            )
        _total_ms = (time.perf_counter() - request_started_at) * 1000
        runtime_progress(
            step="request",
            status="请求完成" if response.result.get("success") else "请求失败",
            detail=f"阶段={final_state.phase.value} 总耗时={_total_ms:.0f}ms",
            request_id=request_id,
            session_id=session_id,
        )
        runtime_log(
            layer="request",
            event="end",
            data={
                "session_id": session_id,
                "phase": final_state.phase.value,
                "success": response.result.get("success"),
                "steps": response.trace.get("steps", []),
            },
            latency_ms=(time.perf_counter() - request_started_at) * 1000,
            logger=LOGGER,
        )
        return response
    except Exception as exc:
        await safe_observe(
            "trace_store.record_request_event.error",
            _runtime_trace_store.record_request_event,
            request_id=request_id,
            session_id=session_id,
            layer="request",
            event="error",
            metadata={"error": str(exc), "exception_type": type(exc).__name__},
        )
        runtime_log(
            layer="request",
            event="error",
            data={
                "session_id": session_id,
                "error": str(exc),
                "exception_type": type(exc).__name__,
            },
            latency_ms=(time.perf_counter() - request_started_at) * 1000,
            level=logging.ERROR,
            logger=LOGGER,
        )
        raise
    # 清理request上下文
    finally:
        clear_request_id(request_token)
        if components is not None:
            repair_llm_client = getattr(components, "repair_llm_client", None)
            if repair_llm_client is not None and repair_llm_client is not components.client:
                await repair_llm_client.aclose()
            await components.client.aclose()

# 回放模式入口
async def _replay_runtime(agent_request: AgentRequest, *, debug: bool) -> AgentResponse:
    # 优先使用用户传入request_id
    request_id = agent_request.request_id
    session_id = agent_request.session_id
    replay_trace = None
    # 没传找最近一次
    if request_id is None:
        replay_trace = await _runtime_trace_store.load_latest_trace(session_id=session_id)
        if replay_trace is None:
            raise FileNotFoundError("replay trace not found; provide request_id or a known session_id")
        request_id = replay_trace.request_id

    # debug 模式下逐步回放  普通模式下全量回放
    replay_result = await _runtime_replay_engine.replay(
        request_id,
        mode=ReplayMode.STEP_BY_STEP if debug else ReplayMode.FULL,
    )
    if replay_trace is None:
        replay_trace = await _runtime_trace_store.load_trace(request_id)

    # 把 replay_result 里的字典任务状态，转成 AgentTaskState
    task_states = [
        AgentTaskState(
            task_id=task_id,
            status=str(payload.get("status", "UNKNOWN")),
            retry_count=int(payload.get("retry_count", 0)),
            max_retry=int(payload.get("max_retry", 0)),
            depends_on=list(payload.get("depends_on", [])),
            output_key=str(payload.get("output_key", "")),
            tool=str(payload.get("tool", "")),
        )
        for task_id, payload in replay_result.task_states.items()
    ]

    phase = replay_result.steps[-1].phase if replay_result.steps else "UNKNOWN"
    supervisor_route = replay_result.trace_summary.get("supervisor_route")
    if supervisor_route is None and replay_trace.events:
        supervisor_route = replay_trace.events[-1].supervisor_route

    trace: dict[str, Any] = {
        "phase": phase,
        "supervisor_route": supervisor_route,
        "session_id": replay_result.session_id,
        "replay": True,
        "replay_mode": replay_result.mode.value,
        "trace_summary": replay_result.trace_summary,
        "replay_snapshot": build_replay_snapshot(replay_trace, replay_result).model_dump(mode="json"),
    }
    if debug:
        last_step = replay_result.steps[-1] if replay_result.steps else None
        trace.update(
            {
                "steps": [f"{step.layer}:{step.event}" for step in replay_result.steps],
                "task_graph": last_step.task_graph if last_step is not None else {},
                "tool_calls": last_step.tool_calls if last_step is not None else [],
                "latency": replay_trace.metrics.latency if replay_trace.metrics is not None else {},
                "metrics": replay_trace.metrics.model_dump(mode="json") if replay_trace.metrics is not None else None,
                "replay_steps": [step.model_dump(mode="json") for step in replay_result.steps],
            }
        )

    result_payload = replay_result.final_output if isinstance(replay_result.final_output, dict) else {}
    return AgentResponse(
        request_id=replay_result.request_id,
        result=result_payload,
        task_states=task_states,
        trace=trace,
    )


def set_runtime_components_builder(builder: RuntimeComponentsBuilder) -> None:
    global _runtime_components_builder
    _runtime_components_builder = builder


def reset_runtime_components_builder() -> None:
    global _runtime_components_builder
    _runtime_components_builder = None


def set_runtime_trace_store(trace_store: LocalExecutionTraceStore) -> None:
    global _runtime_trace_store, _runtime_replay_engine
    _runtime_trace_store = trace_store
    _runtime_replay_engine = RuntimeReplayEngine(trace_store=trace_store)

# 允许外部换掉默认builder  用途：单元测试、场景切换策略
def set_runtime_metrics_collector(collector: RuntimeMetricsCollector) -> None:
    global _runtime_metrics_collector
    _runtime_metrics_collector = collector

# trace store、replay engine、metrics collector 恢复默认
def reset_runtime_observability() -> None:
    global _runtime_trace_store, _runtime_replay_engine, _runtime_metrics_collector
    _runtime_trace_store = LocalExecutionTraceStore(_DEFAULT_TRACE_DIRECTORY)
    _runtime_replay_engine = RuntimeReplayEngine(trace_store=_runtime_trace_store)
    _runtime_metrics_collector = RuntimeMetricsCollector()

# 判断是否注入builder,默认build_env_runtime_components
def export_metrics() -> dict[str, Any]:
    return _runtime_metrics_collector.export_metrics()


def _get_runtime_components_builder() -> RuntimeComponentsBuilder:
    if _runtime_components_builder is not None:
        return _runtime_components_builder

    return build_env_runtime_components


def _resolve_checkpoint_manager(components: Any) -> RuntimeCheckpointManager:
    # 组件中又就用现成的，没有就绑定trace_store没有就新建
    existing_manager = getattr(components, "checkpoint_manager", None)
    if existing_manager is not None:
        if getattr(existing_manager, "trace_store", None) is None:
            existing_manager.trace_store = _runtime_trace_store
        return existing_manager
    return RuntimeCheckpointManager(
        store=LocalCheckpointStore(Path("outputs") / "runtime_checkpoints"),
        enabled=True,
        trace_store=_runtime_trace_store,
    )


def _resolve_repair_llm_client(components: Any) -> Any:
    if not hasattr(components, "repair_llm_client"):
        raise ValueError("runtime components must explicitly provide repair_llm_client")
    return getattr(components, "repair_llm_client")

# 把内部任务状态转成 AgentTaskState 列表
def _build_agent_response(
    *,
    final_state: LangGraphState,
    debug: bool,
    finished_at: Any,
    metrics_snapshot: MetricsSnapshot,
    runtime_trace_snapshot: RuntimeTraceSnapshot,
    debug_snapshot: DebugSnapshot | None,
) -> AgentResponse:
    # 把 runtime 内部维护的 task 对象，翻译成 API 约定好的任务状态格式
    task_states = [
        AgentTaskState(
            task_id=task_id,
            status=_status_value(task.status),
            retry_count=task.retry_count,
            max_retry=task.max_retry,
            depends_on=list(task.depends_on),
            output_key=task.output_key,
            tool=task.tool,
        )
        for task_id, task in final_state.context.tasks.items()
    ]

    trace: dict[str, Any] = {
        "phase": final_state.phase.value,
        "supervisor_route": final_state.supervisor_route,
        "session_id": final_state.context.runtime.session_id,
        "trace_summary": {
            "request_id": runtime_trace_snapshot.request_id,
            "status": runtime_trace_snapshot.status,
            "started_at": runtime_trace_snapshot.started_at,
            "finished_at": runtime_trace_snapshot.finished_at,
            "latency_ms": runtime_trace_snapshot.latency_ms,
        },
        "metrics_snapshot": metrics_snapshot.model_dump(mode="json"),
    }
    if debug:
        node_timings = final_state.metadata.get("node_timings", {})
        trace.update(
            {
                "steps": list(node_timings.keys()),
                "task_graph": build_task_graph_trace(final_state),
                "tool_calls": build_tool_calls(final_state),
                "latency": build_latency_breakdown(node_timings),
                "metrics": metrics_snapshot.model_dump(mode="json"),
                "node_timings": node_timings,
                "latest_route_decision": (
                    final_state.latest_route_decision.model_dump(mode="json")
                    if final_state.latest_route_decision is not None
                    else None
                ),
                "errors": [error.model_dump(mode="json") for error in final_state.context.errors],
                "tool_invocations": [
                    item.model_dump(mode="json") for item in final_state.context.tool_call_chain
                ],
                "execution_history": [
                    item.model_dump(mode="json") for item in final_state.context.execution_history
                ],
                "last_checkpoint": final_state.context.shared_data.get("last_checkpoint"),
                "completed_nodes": list(final_state.metadata.get("completed_nodes", [])),
                "replay_available": True,
                "runtime_trace": runtime_trace_snapshot.model_dump(mode="json"),
                "debug_snapshot": debug_snapshot.model_dump(mode="json") if debug_snapshot is not None else None,
                "observed_at": finished_at.isoformat(),
            }
        )

    result_payload = final_state.final_response if isinstance(final_state.final_response, dict) else {}
    return AgentResponse(
        request_id=final_state.context.runtime.request_id,
        result=result_payload,
        task_states=task_states,
        trace=trace,
    )
def _status_value(status: Any) -> str:
    if isinstance(status, Enum):
        return str(status.value)
    return str(status)


def _is_clarification_response(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    return bool(payload.get("need_clarification"))
