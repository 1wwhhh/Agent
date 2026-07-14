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
def configure_replay_runtime():
    set_runtime_components_builder(build_system_runtime_components)
    set_runtime_trace_store(LocalExecutionTraceStore(_prepare_dir("replay_system") / "traces"))
    set_runtime_metrics_collector(RuntimeMetricsCollector())
    try:
        yield
    finally:
        reset_runtime_components_builder()


@pytest.mark.asyncio
async def test_replay_reconstructs_previous_request_without_reexecution():
    runtime_input = TestCaseGenerator.system_complex_dag_cases()[0]
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        original_response = await client.post(
            "/run?debug=true",
            json=AgentRequest(user_input=runtime_input.user_input, session_id="replay_session").model_dump(mode="json"),
        )
        assert original_response.status_code == 200, original_response.text
        original_payload = original_response.json()
        metrics_before_replay = export_metrics()
        assert metrics_before_replay["total_requests"] == 1

        replay_response = await client.post(
            "/run?debug=true&replay=true",
            json=AgentRequest(
                user_input="replay previous request",
                session_id="replay_session",
                request_id=original_payload["request_id"],
            ).model_dump(mode="json"),
        )
        assert replay_response.status_code == 200, replay_response.text
        replay_payload = replay_response.json()

    metrics_after_replay = export_metrics()
    assert metrics_after_replay["total_requests"] == 1
    assert replay_payload["request_id"] == original_payload["request_id"]
    assert replay_payload["result"] == original_payload["result"]
    assert replay_payload["trace"]["replay"] is True
    assert replay_payload["trace"]["replay_mode"] == "STEP_BY_STEP"
    assert replay_payload["trace"]["replay_steps"]
    assert replay_payload["trace"]["metrics"]["request_id"] == original_payload["request_id"]


def _prepare_dir(name: str) -> Path:
    path = Path("outputs") / "phase3_tests" / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path
