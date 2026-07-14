from __future__ import annotations

import json
import logging

import httpx
import pytest

from app.api.schemas import AgentRequest
from app.api.server import app
from app.api.service import reset_runtime_components_builder, set_runtime_components_builder
from app.utils.logging import ALLOWED_EVENTS
from tests.support.system_runtime import build_system_runtime_components
from tests.support.test_case_generator import TestCaseGenerator


@pytest.fixture(autouse=True)
def configure_system_runtime_builder():
    set_runtime_components_builder(build_system_runtime_components)
    try:
        yield
    finally:
        reset_runtime_components_builder()


@pytest.mark.asyncio
async def test_runtime_logs_use_structured_schema_and_propagate_request_id(caplog):
    caplog.set_level(logging.DEBUG, logger="agent_runtime")
    runtime_input = TestCaseGenerator.system_complex_dag_cases()[0]

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/run?debug=true",
            json=AgentRequest(
                user_input=runtime_input.user_input,
                session_id="logging_case_session",
            ).model_dump(mode="json"),
        )

    assert response.status_code == 200, response.text
    body = response.json()
    runtime_logs = [record for record in caplog.records if record.name == "agent_runtime"]
    assert runtime_logs

    parsed_logs = [json.loads(record.getMessage()) for record in runtime_logs]
    for item in parsed_logs:
        assert set(item.keys()) == {"request_id", "layer", "event", "data", "latency_ms", "timestamp"}
        assert item["request_id"] == body["request_id"]
        assert item["event"] in ALLOWED_EVENTS
        assert isinstance(item["data"], dict)

    layers = {item["layer"] for item in parsed_logs}
    assert {"request", "supervisor", "planner", "parser", "queue", "executor", "aggregator", "task"} <= layers

