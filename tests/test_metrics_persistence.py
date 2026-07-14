from __future__ import annotations

import httpx
from pathlib import Path
import shutil
import pytest
from uuid import uuid4

from app.api.schemas import AgentRequest
from app.api.server import app
from app.api.service import (
    export_metrics,
    reset_runtime_components_builder,
    set_runtime_components_builder,
    set_runtime_metrics_collector,
    set_runtime_trace_store,
)
from app.observability import LocalExecutionTraceStore, RuntimeMetricsCollector
from tests.support.system_runtime import build_system_runtime_components
from tests.support.test_case_generator import TestCaseGenerator


@pytest.fixture(autouse=True)
def configure_runtime_observability():
    set_runtime_components_builder(build_system_runtime_components)
    set_runtime_trace_store(LocalExecutionTraceStore(_prepare_dir("metrics_persistence") / "traces"))
    set_runtime_metrics_collector(RuntimeMetricsCollector())
    try:
        yield
    finally:
        reset_runtime_components_builder()


@pytest.mark.asyncio
async def test_metrics_are_persisted_and_exportable():
    cases = [
        TestCaseGenerator.system_simple_task_cases()[0],
        TestCaseGenerator.system_complex_dag_cases()[0],
    ]
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        request_ids: list[str] = []
        for case in cases:
            response = await client.post(
                "/run?debug=true",
                json=AgentRequest(user_input=case.user_input, session_id=f"{case.name}_session").model_dump(mode="json"),
            )
            assert response.status_code == 200, response.text
            request_ids.append(response.json()["request_id"])

    exported = export_metrics()
    assert exported["total_requests"] == 2
    assert "averages" in exported
    assert "latency" in exported["averages"]
    for request_id in request_ids:
        assert request_id in exported["requests"]


def _prepare_dir(name: str) -> Path:
    path = Path("outputs") / "phase3_tests" / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path
