from __future__ import annotations

import importlib

from fastapi.testclient import TestClient

from app.api.schemas import AgentResponse, AgentTaskState
from app.api.server import app

api_router_module = importlib.import_module("app.api.router")


def test_post_run_gateway_returns_runtime_payload(monkeypatch):
    async def fake_run_runtime(agent_request, *, debug: bool = False, replay: bool = False):
        assert agent_request.client_timezone == "Asia/Shanghai"
        return AgentResponse(
            request_id="req_test",
            result={"success": True, "echo": agent_request.user_input},
            task_states=[
                AgentTaskState(
                    task_id="task_1",
                    status="SUCCESS",
                    retry_count=0,
                    max_retry=1,
                    depends_on=[],
                    output_key="final_result",
                    tool="text_generate_tool",
                )
            ],
            trace={"phase": "COMPLETED", "debug": debug, "replay": replay},
        )

    monkeypatch.setattr(api_router_module, "run_runtime", fake_run_runtime)

    client = TestClient(app)
    response = client.post(
        "/run?debug=true",
        json={"user_input": "hello", "session_id": "sess_1", "client_timezone": "Asia/Shanghai"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["request_id"] == "req_test"
    assert payload["result"]["echo"] == "hello"
    assert payload["task_states"][0]["status"] == "SUCCESS"
    assert payload["trace"]["debug"] is True
    assert payload["trace"]["replay"] is False
