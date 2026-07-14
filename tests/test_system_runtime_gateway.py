from __future__ import annotations

import asyncio
from collections.abc import Iterable
from pathlib import Path
import shutil
from uuid import uuid4

import httpx
import pytest

from app.api.schemas import AgentRequest
from app.api.server import app
from app.api.service import (
    reset_runtime_components_builder,
    reset_runtime_observability,
    set_runtime_components_builder,
    set_runtime_metrics_collector,
    set_runtime_trace_store,
)
from app.observability import LocalExecutionTraceStore, RuntimeMetricsCollector
from tests.support.system_runtime import build_system_runtime_components
from tests.support.test_case_generator import TestCaseGenerator

ITERATIONS_PER_CASE = 100
PARALLEL_SESSIONS = 10


@pytest.fixture(autouse=True)
def configure_system_runtime_builder():
    set_runtime_components_builder(build_system_runtime_components)
    set_runtime_trace_store(LocalExecutionTraceStore(_prepare_dir("system_runtime_gateway") / "traces"))
    set_runtime_metrics_collector(RuntimeMetricsCollector())
    try:
        yield
    finally:
        reset_runtime_components_builder()
        reset_runtime_observability()


async def setup_graph(runtime_input=None):
    return runtime_input or TestCaseGenerator.system_simple_task_cases()[0]


async def run_runtime(runtime_input):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await _run_case_iterations(client, runtime_input)


def assert_task_states(result, *, expected_statuses):
    task_states = {item["task_id"]: item["status"] for item in result["task_states"]}
    for task_id, expected_status in expected_statuses.items():
        assert task_states.get(task_id) == expected_status


def assert_context_state(result, *, expected_output_keys=None, missing_output_keys=None):
    task_results = result["result"].get("task_results", {})
    for key in expected_output_keys or []:
        assert key in task_results, f"expected output key {key} missing"
    for key in missing_output_keys or []:
        assert key not in task_results, f"unexpected output key {key} found"


def assert_execution_order(result, *, before_after_pairs=None, blocked_tasks=None):
    execution_history = result["trace"]["execution_history"]
    running_indices: dict[str, int] = {}
    for index, record in enumerate(execution_history):
        if record["status"] == "RUNNING" and record["task_id"] not in running_indices:
            running_indices[record["task_id"]] = index

    for before_task, after_task in before_after_pairs or []:
        assert before_task in running_indices
        assert after_task in running_indices
        assert running_indices[before_task] < running_indices[after_task]

    task_states = {item["task_id"]: item["status"] for item in result["task_states"]}
    for task_id in blocked_tasks or []:
        assert task_states[task_id] in {"CANCELLED", "PENDING", "FAILED", "TIMEOUT"}


@pytest.mark.asyncio
async def test_simple_task_set_is_stable_for_100_iterations():
    simple_cases = TestCaseGenerator.system_simple_task_cases()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        for runtime_input in simple_cases:
            results = await _run_case_iterations(client, runtime_input)
            _assert_debug_trace_shape(results)
            _assert_unique_request_ids(results)
            _assert_no_context_leakage(results)
            for result in results:
                assert result["result"]["success"] is True
                assert result["result"]["phase"] == "COMPLETED"
                assert result["result"]["final_output_ready"] is True
                assert result["trace"]["supervisor_route"] == "SIMPLE_TASK"
                assert "planner" not in result["trace"]["steps"]
                assert_task_states(result, expected_statuses={"task_1": "SUCCESS"})
                assert result["trace"]["metrics"]["task_success_rate"] == 1.0
                assert sum(item["retry_count"] for item in result["task_states"]) <= 5
                assert result["trace"]["metrics"]["context_consistency_rate"] == 1.0


@pytest.mark.asyncio
async def test_complex_dag_task_set_is_stable_for_100_iterations_with_parallel_sessions():
    complex_cases = TestCaseGenerator.system_complex_dag_cases()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        for runtime_input in complex_cases:
            results = await _run_case_iterations(client, runtime_input)
            _assert_debug_trace_shape(results)
            _assert_unique_request_ids(results)
            _assert_no_context_leakage(results)
            for result in results:
                assert result["result"]["success"] is True
                assert result["result"]["phase"] == "COMPLETED"
                assert result["result"]["final_output_ready"] is True
                assert result["trace"]["supervisor_route"] == "COMPLEX_TASK"
                assert "planner" in result["trace"]["steps"]
                assert "queue" in result["trace"]["steps"]
                assert "executor" in result["trace"]["steps"]
                _assert_task_graph_is_valid_dag(result["trace"]["task_graph"])
                _assert_tool_calls_shape(result["trace"]["tool_calls"])
                assert result["trace"]["metrics"]["dag_correctness_rate"] == 1.0
                assert sum(item["retry_count"] for item in result["task_states"]) <= 5
                assert result["trace"]["metrics"]["context_consistency_rate"] == 1.0
                if runtime_input.name == "system_complex_report":
                    assert_task_states(result, expected_statuses={"task_1": "SUCCESS", "task_2": "SUCCESS"})
                    assert_execution_order(result, before_after_pairs=[("task_1", "task_2")])
                if runtime_input.name == "system_complex_parallel_merge":
                    assert_task_states(
                        result,
                        expected_statuses={"task_a": "SUCCESS", "task_b": "SUCCESS", "task_merge": "SUCCESS"},
                    )
                    assert_execution_order(result, before_after_pairs=[("task_a", "task_merge"), ("task_b", "task_merge")])


async def _run_case_iterations(client: httpx.AsyncClient, runtime_input, *, iterations: int = ITERATIONS_PER_CASE):
    results = []
    for batch_start in range(0, iterations, PARALLEL_SESSIONS):
        batch_size = min(PARALLEL_SESSIONS, iterations - batch_start)
        tasks = [
            _post_run(
                client,
                AgentRequest(
                    user_input=runtime_input.user_input,
                    session_id=f"{runtime_input.name}_session_{batch_start + index}",
                ),
            )
            for index in range(batch_size)
        ]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
    return results


async def _post_run(client: httpx.AsyncClient, request: AgentRequest) -> dict:
    response = await client.post("/run?debug=true", json=request.model_dump(mode="json"))
    assert response.status_code == 200, response.text
    return response.json()


def _assert_debug_trace_shape(results: Iterable[dict]) -> None:
    for result in results:
        trace = result["trace"]
        assert "steps" in trace
        assert "task_graph" in trace
        assert "tool_calls" in trace
        assert "latency" in trace
        assert "metrics" in trace


def _assert_unique_request_ids(results: Iterable[dict]) -> None:
    request_ids = [result["request_id"] for result in results]
    assert len(request_ids) == len(set(request_ids))


def _assert_no_context_leakage(results: Iterable[dict]) -> None:
    for result in results:
        session_id = result["trace"]["session_id"]
        assert result["result"]["session_id"] == session_id
        task_graph = result["trace"]["task_graph"]
        node_ids = {node["task_id"] for node in task_graph["nodes"]}
        for edge in task_graph["edges"]:
            assert edge["from"] in node_ids
            assert edge["to"] in node_ids


def _assert_task_graph_is_valid_dag(task_graph: dict) -> None:
    node_ids = {node["task_id"] for node in task_graph["nodes"]}
    adjacency = {node_id: [] for node_id in node_ids}
    indegree = {node_id: 0 for node_id in node_ids}

    for edge in task_graph["edges"]:
        assert edge["from"] in node_ids
        assert edge["to"] in node_ids
        adjacency[edge["from"]].append(edge["to"])
        indegree[edge["to"]] += 1

    ready = [node_id for node_id, count in indegree.items() if count == 0]
    visited = 0
    while ready:
        node_id = ready.pop()
        visited += 1
        for target in adjacency[node_id]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    assert visited == len(node_ids), "task_graph contains a cycle or orphan dependency"


def _assert_tool_calls_shape(tool_calls: list[dict]) -> None:
    for item in tool_calls:
        assert "tool_name" in item
        assert "input" in item
        assert "output" in item
        assert "timestamp" in item


def _prepare_dir(name: str) -> Path:
    path = Path("outputs") / "phase3_tests" / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path
