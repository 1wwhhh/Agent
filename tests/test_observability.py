from __future__ import annotations

import ast
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib.util
import logging
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import types
from uuid import uuid4

import pytest

from app.observability import LocalExecutionTraceStore, RuntimeMetricsCollector, RuntimeReplayEngine, safe_observe
from app.observability.builders import (
    build_debug_snapshot,
    build_metrics_snapshot,
    build_replay_snapshot,
    build_request_metrics_snapshot,
    build_runtime_trace_snapshot,
    build_task_graph_trace,
)
from app.observability.compact import compact_observability_payload
from app.schemas.context import ErrorRecord, ExecutionRecord, ToolCallRecord
from app.schemas.graph import GraphPhase
from app.schemas.observability import ReplayMode
from app.schemas.task import TaskModel, TaskStatus
from app.state import LangGraphState
from tests.support.models import RuntimeTestInput
from tests.support.test_case_generator import TestCaseGenerator
from tests.support.tools import RuntimeTestTool, StaticStructuredLLMClient

SERVICE_PATH = Path("app/api/service.py")
BUILDERS_PATH = Path("app/observability/builders.py")
API_DIR = Path("app/api")
_LOADED_API_MODULES: tuple[ModuleType, ModuleType, ModuleType] | None = None


class BrokenRecordRequestTraceStore(LocalExecutionTraceStore):
    async def record_request_event(self, *args, **kwargs):
        raise OSError("record_request_event failed")


class BrokenAttachMetricsTraceStore(LocalExecutionTraceStore):
    async def attach_metrics(self, request_id: str, metrics):
        raise OSError("attach_metrics failed")


class BrokenMetricsCollector(RuntimeMetricsCollector):
    def record(self, snapshot) -> None:
        raise RuntimeError("metrics collector failed")


def _load_module(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_service_modules() -> tuple[ModuleType, ModuleType, ModuleType]:
    global _LOADED_API_MODULES
    if _LOADED_API_MODULES is not None:
        return _LOADED_API_MODULES

    package = types.ModuleType("app.api")
    package.__path__ = [str(API_DIR.resolve())]  # type: ignore[attr-defined]
    sys.modules["app.api"] = package

    schemas_module = _load_module("app.api.schemas", API_DIR / "schemas.py")
    runtime_module = _load_module("app.api.runtime", API_DIR / "runtime.py")
    _load_module("app.api.bootstrap", API_DIR / "bootstrap.py")
    service_module = _load_module("app.api.service", API_DIR / "service.py")
    _LOADED_API_MODULES = (schemas_module, runtime_module, service_module)
    return _LOADED_API_MODULES


def _build_service_case_registry() -> dict[str, RuntimeTestInput]:
    cases = [
        *TestCaseGenerator.system_simple_task_cases(),
        *TestCaseGenerator.system_complex_dag_cases(),
    ]
    return {case.user_input: case for case in cases}


async def _build_test_runtime_components(agent_request) -> SimpleNamespace:
    from app.agents import SupervisorAgent
    from app.planner import LLMTaskPlanner
    from app.router import TaskRouter
    from app.router.capability import capability_from_tool

    registry = _build_service_case_registry()
    runtime_input = registry.get(agent_request.user_input)
    if runtime_input is None:
        raise ValueError(f"no system test case registered for input: {agent_request.user_input}")

    client = StaticStructuredLLMClient(test_input=runtime_input)
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
    supervisor_agent = SupervisorAgent(client=client) if runtime_input.use_supervisor_agent else None
    planner_agent = LLMTaskPlanner(client=client) if runtime_input.use_planner_agent else None
    return SimpleNamespace(
        client=client,
        repair_llm_client=client,
        router=router,
        supervisor_agent=supervisor_agent,
        planner_agent=planner_agent,
        planner_tool=None,
        checkpoint_manager=None,
    )


def _build_sample_state() -> tuple[LangGraphState, datetime, datetime]:
    started_at = datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
    finished_at = started_at + timedelta(milliseconds=2500)

    state = LangGraphState.create(
        request_id="req_observability",
        session_id="sess_observability",
        user_input="Analyze the request and explain the outcome.",
    )
    state.context.runtime.timestamp = started_at

    task_1 = TaskModel(
        task_id="task_1",
        task_name="collect_inputs",
        description="Collect inputs.",
        tool="llm_reason_tool",
        output_key="inputs",
        status=TaskStatus.SUCCESS,
    )
    task_2 = TaskModel(
        task_id="task_2",
        task_name="draft_answer",
        description="Draft the answer.",
        tool="text_generate_tool",
        output_key="draft",
        depends_on=["task_1"],
        retry_count=1,
        max_retry=2,
        status=TaskStatus.SUCCESS,
    )
    task_3 = TaskModel(
        task_id="task_3",
        task_name="route_guarded",
        description="Task denied by router.",
        tool="text_generate_tool",
        output_key="denied",
        depends_on=["task_1"],
        status=TaskStatus.CANCELLED,
    )
    task_4 = TaskModel(
        task_id="task_4",
        task_name="slow_branch",
        description="Timeout branch.",
        tool="llm_reason_tool",
        output_key="slow_output",
        depends_on=["task_2"],
        status=TaskStatus.TIMEOUT,
    )
    task_5 = TaskModel(
        task_id="task_5",
        task_name="broken_branch",
        description="Failed branch.",
        tool="text_generate_tool",
        output_key="broken_output",
        depends_on=["task_2"],
        status=TaskStatus.FAILED,
    )

    state.planned_tasks = [task_1, task_2, task_3, task_4, task_5]
    for task in state.planned_tasks:
        state.context.register_task(task)
        state.agent_state.sync_task_status(task)

    state.context.set_task_result("inputs", {"text": "inputs"})
    state.context.set_task_result("draft", {"text": "draft"})
    state.context.set_shared_value("raw_plan_text", '{"tasks": ["draft_answer"]}')
    state.context.set_shared_value("parsed_plan", {"tasks": [{"task_id": "task_1"}, {"task_id": "task_2"}]})
    state.context.set_shared_value("planner_prompt", {"name": "planner_prompt", "version": "p4"})
    state.context.set_shared_value("last_checkpoint", {"checkpoint_id": "ckpt-1"})
    state.context.set_shared_value(
        "executor_nonfatal_errors",
        [{"task_id": "task_4", "message": "executor timeout ignored for observability snapshot"}],
    )

    state.context.add_execution_record(
        ExecutionRecord(task_id="task_1", status=TaskStatus.RUNNING, attempt=0, recorded_at=started_at)
    )
    state.context.add_execution_record(
        ExecutionRecord(
            task_id="task_1",
            status=TaskStatus.SUCCESS,
            attempt=0,
            recorded_at=started_at + timedelta(milliseconds=100),
        )
    )
    state.context.add_execution_record(
        ExecutionRecord(
            task_id="task_2",
            status=TaskStatus.RUNNING,
            attempt=0,
            recorded_at=started_at + timedelta(milliseconds=200),
        )
    )
    state.context.add_execution_record(
        ExecutionRecord(
            task_id="task_2",
            status=TaskStatus.RETRY,
            attempt=0,
            recorded_at=started_at + timedelta(milliseconds=300),
        )
    )
    state.context.add_execution_record(
        ExecutionRecord(
            task_id="task_2",
            status=TaskStatus.RUNNING,
            attempt=1,
            recorded_at=started_at + timedelta(milliseconds=400),
        )
    )
    state.context.add_execution_record(
        ExecutionRecord(
            task_id="task_2",
            status=TaskStatus.SUCCESS,
            attempt=1,
            recorded_at=started_at + timedelta(milliseconds=500),
        )
    )
    state.context.add_execution_record(
        ExecutionRecord(
            task_id="task_3",
            status=TaskStatus.CANCELLED,
            attempt=0,
            recorded_at=started_at + timedelta(milliseconds=600),
        )
    )
    state.context.add_execution_record(
        ExecutionRecord(
            task_id="task_4",
            status=TaskStatus.TIMEOUT,
            attempt=0,
            recorded_at=started_at + timedelta(milliseconds=700),
        )
    )
    state.context.add_execution_record(
        ExecutionRecord(
            task_id="task_5",
            status=TaskStatus.FAILED,
            attempt=0,
            recorded_at=started_at + timedelta(milliseconds=800),
        )
    )

    state.context.add_tool_call(
        ToolCallRecord(
            tool_name="llm_reason_tool",
            task_id="task_1",
            status="SUCCESS",
            started_at=started_at,
            finished_at=started_at + timedelta(milliseconds=80),
            latency_ms=80,
            metadata={"input": {"prompt": "Collect inputs"}, "output": {"text": "inputs"}},
        )
    )
    state.context.add_tool_call(
        ToolCallRecord(
            tool_name="text_generate_tool",
            task_id="task_2",
            status="SUCCESS",
            started_at=started_at + timedelta(milliseconds=200),
            finished_at=started_at + timedelta(milliseconds=480),
            latency_ms=280,
            metadata={"input": {"prompt": "Draft the answer"}, "output": {"text": "draft"}},
        )
    )

    state.context.add_error(
        ErrorRecord(source="executor", message="task_4 timed out", task_id="task_4", details={"type": "timeout"})
    )
    state.context.add_error(
        ErrorRecord(source="router", message="task_3 denied", task_id="task_3", details={"type": "denied"})
    )

    metadata = {
        "task_failures": [
            {"task_id": "task_4", "failure_type": "TIMEOUT"},
            {"task_id": "task_5", "failure_type": "FAILED"},
        ],
        "parser_repair_history": [
            {"attempt": 1, "reason": "invalid_json", "repaired": True},
        ],
        "routing_history": [
            {"task_id": "task_1", "routing_result": "ALLOWED"},
            {"task_id": "task_3", "routing_result": "DENIED"},
        ],
        "llm_calls": [
            {"operation": "planner", "success": True, "fallback_used": False},
            {"operation": "repair", "success": True, "fallback_used": True},
        ],
        "node_timings": {
            "planner": {"duration_ms": 12.5},
            "parser": {"duration_ms": 7.5},
            "executor": {"duration_ms": 125.0},
            "aggregator": {"duration_ms": 5.0},
        },
        "completed_nodes": ["planner", "parser", "executor", "aggregator"],
    }
    state.metadata.update(metadata)
    state.final_response = {
        "success": False,
        "answer": "partial output",
        "metadata": deepcopy(metadata),
    }
    state.context.final_output = deepcopy(state.final_response)
    state.phase = GraphPhase.FAILED
    state.agent_state.final_output_ready = True
    return state, started_at, finished_at


def _state_signatures(state: LangGraphState) -> tuple[dict[str, str], dict[str, object], dict[str, object]]:
    task_statuses = {task_id: task.status for task_id, task in state.context.tasks.items()}
    return task_statuses, deepcopy(state.context.task_results), deepcopy(state.metadata)


def _prepare_dir(name: str) -> Path:
    path = Path("outputs") / "test_observability_runtime" / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def configure_service_runtime():
    schemas_module, _, service_module = _load_service_modules()
    service_module.set_runtime_components_builder(_build_test_runtime_components)
    try:
        yield schemas_module, service_module
    finally:
        service_module.reset_runtime_components_builder()
        service_module.reset_runtime_observability()


def test_runtime_trace_snapshot_aggregates_metadata_and_latency() -> None:
    state, started_at, finished_at = _build_sample_state()

    snapshot = build_runtime_trace_snapshot(state, finished_at=finished_at)

    assert snapshot.request_id == "req_observability"
    assert snapshot.session_id == "sess_observability"
    assert snapshot.started_at == started_at.isoformat()
    assert snapshot.finished_at == finished_at.isoformat()
    assert snapshot.status == "FAILED"
    assert snapshot.latency_ms == 2500
    assert snapshot.parser_repair_history == state.metadata["parser_repair_history"]
    assert snapshot.task_failures == state.metadata["task_failures"]
    assert snapshot.routing_history == state.metadata["routing_history"]
    assert snapshot.llm_calls == state.metadata["llm_calls"]
    assert snapshot.metrics.llm_call_count == 2
    assert snapshot.metrics.fallback_count == 1
    assert snapshot.metrics.parser_repair_count == 1
    assert snapshot.metrics.routing_denied_count == 1
    assert snapshot.debug_events[0] == {"type": "completed_node", "node": "planner"}
    assert snapshot.debug_events[-1]["type"] == "executor_nonfatal_error"


def test_compact_observability_payload_summarizes_large_tool_results() -> None:
    payload = {
        "task_results": {
            "dept_plan_completion": {
                "dept_plan_followups": [{"plan_id": index, "plan_text": "x" * 2000} for index in range(50)],
                "dept_plan_completion_context_text": "y" * 5000,
                "pairing_summary": {"total_plans": 50},
            },
            "final_result": {"text": "z" * 5000},
        }
    }

    compacted = compact_observability_payload(payload, max_string_chars=100, max_list_items=5)
    task_results = compacted["task_results"]

    dept_result = task_results["dept_plan_completion"]
    assert dept_result["dept_plan_followups"]["_count"] == 50
    assert dept_result["dept_plan_completion_context_text"]["_original_chars"] == 5000
    assert task_results["final_result"]["text"]["_original_chars"] == 5000


def test_task_graph_trace_contains_nodes_edges_and_execution_sequence() -> None:
    state, _, _ = _build_sample_state()

    task_graph = build_task_graph_trace(state)

    assert len(task_graph["nodes"]) == 5
    assert {"from": "task_1", "to": "task_2"} in task_graph["edges"]
    assert {"from": "task_2", "to": "task_4"} in task_graph["edges"]
    assert task_graph["execution_sequence"][0]["task_id"] == "task_1"
    assert task_graph["execution_sequence"][3]["status"] == "RETRY"
    assert task_graph["retry_tasks"] == ["task_2"]
    assert task_graph["cancelled_tasks"] == ["task_3"]
    assert task_graph["timeout_tasks"] == ["task_4"]


def test_metrics_snapshot_aggregates_counts_rates_and_latency() -> None:
    state, _, finished_at = _build_sample_state()

    metrics = build_metrics_snapshot(state, finished_at=finished_at)

    assert metrics.total_tasks == 5
    assert metrics.successful_tasks == 2
    assert metrics.failed_tasks == 1
    assert metrics.cancelled_tasks == 1
    assert metrics.timeout_tasks == 1
    assert metrics.retry_count == 1
    assert metrics.llm_call_count == 2
    assert metrics.fallback_count == 1
    assert metrics.parser_repair_count == 1
    assert metrics.routing_denied_count == 1
    assert metrics.runtime_latency_ms == 2500
    assert metrics.task_success_rate == pytest.approx(0.4)
    assert metrics.retry_rate == pytest.approx(0.2)
    assert metrics.context_consistency_rate == pytest.approx(1.0)
    assert metrics.latency["total"] == pytest.approx(150.0)


@pytest.mark.asyncio
async def test_replay_snapshot_reuses_existing_trace_and_replay_data() -> None:
    state, _, finished_at = _build_sample_state()
    trace_dir = Path("outputs") / "test_observability_traces" / uuid4().hex
    trace_store = LocalExecutionTraceStore(trace_dir)

    await trace_store.record_request_event(
        request_id=state.context.runtime.request_id,
        session_id=state.context.runtime.session_id,
        layer="request",
        event="start",
        metadata={"input": state.context.runtime.user_input, "debug": True},
    )
    await trace_store.record_state_event(state=state, source_layer="runtime", event="end")
    metrics = build_metrics_snapshot(state, finished_at=finished_at)
    await trace_store.attach_metrics(
        state.context.runtime.request_id,
        build_request_metrics_snapshot(metrics, recorded_at=finished_at),
    )

    replay_engine = RuntimeReplayEngine(trace_store=trace_store)
    replay_result = await replay_engine.replay(state.context.runtime.request_id, mode=ReplayMode.STEP_BY_STEP)
    trace = await trace_store.load_trace(state.context.runtime.request_id)

    snapshot = build_replay_snapshot(trace, replay_result)

    assert snapshot.request_id == state.context.runtime.request_id
    assert snapshot.raw_user_input == state.context.runtime.user_input
    assert snapshot.planner_raw_output == state.context.shared_data["raw_plan_text"]
    assert snapshot.repaired_output == state.context.shared_data["parsed_plan"]
    assert snapshot.routing_decisions == state.metadata["routing_history"]
    assert snapshot.llm_calls == state.metadata["llm_calls"]
    assert snapshot.final_output == state.final_response
    assert snapshot.runtime_context_snapshot["runtime"]["request_id"] == state.context.runtime.request_id


def test_debug_snapshot_and_builders_are_read_only() -> None:
    state, _, finished_at = _build_sample_state()
    task_statuses_before, task_results_before, metadata_before = _state_signatures(state)

    debug_snapshot = build_debug_snapshot(state, finished_at=finished_at)
    runtime_trace_snapshot = build_runtime_trace_snapshot(state, finished_at=finished_at)
    metrics_snapshot = build_metrics_snapshot(state, finished_at=finished_at)

    assert debug_snapshot.planner_output["raw_plan_text"] == state.context.shared_data["raw_plan_text"]
    assert debug_snapshot.repair_attempts == state.metadata["parser_repair_history"]
    assert debug_snapshot.routing_decisions == state.metadata["routing_history"]
    assert debug_snapshot.llm_calls == state.metadata["llm_calls"]
    assert debug_snapshot.metrics.runtime_latency_ms == 2500
    assert runtime_trace_snapshot.metrics.runtime_latency_ms == metrics_snapshot.runtime_latency_ms
    assert _state_signatures(state) == (task_statuses_before, task_results_before, metadata_before)


def test_service_uses_observability_builders_as_single_source() -> None:
    source = SERVICE_PATH.read_text(encoding="utf-8")

    assert "def _build_metrics_snapshot" not in source
    assert "def _build_task_graph" not in source
    assert "def _build_tool_calls" not in source
    assert "def _build_latency_breakdown" not in source
    assert "def _build_metrics" not in source
    assert "def _calculate_dag_correctness" not in source
    assert "build_metrics_snapshot(" in source
    assert "build_runtime_trace_snapshot(" in source
    assert "build_debug_snapshot(" in source
    assert "build_task_graph_trace(" in source
    assert "build_tool_calls(" in source
    assert "build_latency_breakdown(" in source
    assert "safe_observe(" in source
    assert "graph.ainvoke(" in source


def test_observability_builders_do_not_depend_on_llm_control_layers() -> None:
    tree = ast.parse(BUILDERS_PATH.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module)

    forbidden = {
        "app.llm.retry",
        "app.llm.fallback",
        "app.llm.circuit_breaker",
    }
    assert imports.isdisjoint(forbidden)


@pytest.mark.asyncio
async def test_safe_observe_swallows_observability_exception_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)

    def _boom():
        raise RuntimeError("safe wrapper failure")

    result = await safe_observe("unit.safe_observe", _boom, default={"ok": False})

    assert result == {"ok": False}
    assert "unit.safe_observe" in caplog.text
    assert "safe wrapper failure" in caplog.text


@pytest.mark.asyncio
async def test_run_runtime_survives_trace_store_record_request_event_failure(
    configure_service_runtime,
    caplog: pytest.LogCaptureFixture,
) -> None:
    schemas_module, service_module = configure_service_runtime
    caplog.set_level(logging.WARNING)
    runtime_input = TestCaseGenerator.system_simple_task_cases()[0]
    service_module.set_runtime_components_builder(_build_test_runtime_components)
    service_module.set_runtime_trace_store(BrokenRecordRequestTraceStore(_prepare_dir("broken_record_trace")))
    service_module.set_runtime_metrics_collector(RuntimeMetricsCollector())

    response = await service_module.run_runtime(
        schemas_module.AgentRequest(user_input=runtime_input.user_input, session_id="obs_record_failure"),
        debug=True,
    )

    assert response.result["success"] is True
    assert response.trace["supervisor_route"] == "SIMPLE_TASK"
    assert response.trace["metrics_export"]["total_requests"] == 1
    assert "trace_store.record_request_event.start" in caplog.text


@pytest.mark.asyncio
async def test_run_runtime_survives_attach_metrics_failure(
    configure_service_runtime,
    caplog: pytest.LogCaptureFixture,
) -> None:
    schemas_module, service_module = configure_service_runtime
    caplog.set_level(logging.WARNING)
    runtime_input = TestCaseGenerator.system_simple_task_cases()[0]
    service_module.set_runtime_components_builder(_build_test_runtime_components)
    service_module.set_runtime_trace_store(BrokenAttachMetricsTraceStore(_prepare_dir("broken_attach_metrics")))
    service_module.set_runtime_metrics_collector(RuntimeMetricsCollector())

    response = await service_module.run_runtime(
        schemas_module.AgentRequest(user_input=runtime_input.user_input, session_id="obs_attach_failure"),
        debug=True,
    )

    assert response.result["success"] is True
    assert any(item.status == "SUCCESS" for item in response.task_states)
    assert response.trace["metrics_export"]["total_requests"] == 1
    assert "trace_store.attach_metrics" in caplog.text


@pytest.mark.asyncio
async def test_run_runtime_survives_metrics_collector_record_failure(
    configure_service_runtime,
    caplog: pytest.LogCaptureFixture,
) -> None:
    schemas_module, service_module = configure_service_runtime
    caplog.set_level(logging.WARNING)
    runtime_input = TestCaseGenerator.system_simple_task_cases()[0]
    service_module.set_runtime_components_builder(_build_test_runtime_components)
    service_module.set_runtime_trace_store(LocalExecutionTraceStore(_prepare_dir("broken_metrics_collector") / "traces"))
    service_module.set_runtime_metrics_collector(BrokenMetricsCollector())

    response = await service_module.run_runtime(
        schemas_module.AgentRequest(user_input=runtime_input.user_input, session_id="obs_metrics_failure"),
        debug=True,
    )

    assert response.result["success"] is True
    assert response.trace["metrics_export"]["total_requests"] == 0
    assert "metrics_collector.record" in caplog.text
