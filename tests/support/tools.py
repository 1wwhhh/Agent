from __future__ import annotations

import asyncio
import json
from datetime import timezone
from typing import Any

from app.agents import SupervisorAgent
from app.planner import LLMTaskPlanner
from app.schemas.context import ContextStore
from app.schemas.llm import LLMFunctionCall, LLMRequest, LLMResponse
from app.schemas.task import utc_now
from app.tools.base import BaseTool
from app.tools.llm_client import LLMClient
from tests.support.models import PlannerScenario, RuntimeTestInput, TaskBehavior, TaskSpec


def build_planner_json(test_input: RuntimeTestInput) -> str:
    tasks = []
    created_at = utc_now().astimezone(timezone.utc).isoformat()

    for task in test_input.task_specs:
        payload = {
            "prompt": task.prompt,
            "task_id": task.task_id,
            "behavior": task.behavior.value,
            "read_keys": list(task.read_keys),
            "delay_seconds": task.delay_seconds,
            **task.metadata,
        }
        tasks.append(
            {
                "task_id": task.task_id,
                "task_name": task.task_name,
                "description": task.description,
                "tool": task.tool,
                "input": payload,
                "output_key": task.output_key,
                "idempotency_key": task.idempotency_key,
                "irreversible": task.irreversible,
                "depends_on": list(task.depends_on),
                "priority": task.priority,
                "status": "PENDING",
                "retry_count": 0,
                "max_retry": task.max_retry,
                "timeout": task.timeout,
                "created_at": created_at,
            }
        )

    return json.dumps({"goal": test_input.user_input, "tasks": tasks})


class PlannerTestTool(BaseTool):
    scenario: PlannerScenario
    test_input: RuntimeTestInput

    async def _arun(self, payload: dict[str, Any], context: ContextStore | None = None):
        if self.test_input.planner_raw_output is not None:
            return self.build_result(success=True, output={"text": self.test_input.planner_raw_output})
        if self.scenario == PlannerScenario.INVALID_JSON:
            return self.build_result(success=True, output={"text": '{"goal": "bad", "tasks": ['})
        if self.scenario == PlannerScenario.INVALID_SCHEMA:
            invalid_payload = {"goal": self.test_input.user_input, "tasks": [{"task_name": "missing_task_id"}]}
            return self.build_result(success=True, output={"text": json.dumps(invalid_payload)})
        return self.build_result(success=True, output={"text": build_planner_json(self.test_input)})


class RuntimeTestTool(BaseTool):
    def get_routing_capability(self) -> dict[str, Any]:
        capability = super().get_routing_capability()
        if self.name == "llm_reason_tool":
            capability["supported_task_types"] = ["reasoning"]
            capability["default_task_type"] = "reasoning"
            capability["supported_tags"] = list(self.tags)
        elif self.name == "text_generate_tool":
            capability["supported_task_types"] = ["text_generation"]
            capability["default_task_type"] = "text_generation"
            capability["supported_tags"] = list(self.tags)
        return capability

    async def _arun(self, payload: dict[str, Any], context: ContextStore | None = None):
        behavior = TaskBehavior(payload.get("behavior", TaskBehavior.SUCCESS.value))
        task_id = str(payload.get("task_id", "unknown"))
        prompt = str(payload.get("prompt", ""))
        delay_seconds = float(payload.get("delay_seconds", 0.0))
        read_keys = [str(item) for item in payload.get("read_keys", [])]

        if delay_seconds:
            await asyncio.sleep(delay_seconds)

        consumed_inputs = self._collect_inputs(context, read_keys)
        self._append_runtime_trace(
            context,
            {
                "task_id": task_id,
                "tool_name": self.name,
                "behavior": behavior.value,
                "consumed_keys": read_keys,
                "consumed_inputs": consumed_inputs,
            },
        )

        if behavior == TaskBehavior.TIMEOUT:
            await asyncio.sleep(float(payload.get("timeout_sleep_seconds", 1.2)))
            return self.build_result(success=True, output={"text": "timeout-unreachable"})
        if behavior == TaskBehavior.TOOL_EXCEPTION:
            raise RuntimeError(f"tool exception for {task_id}")
        if behavior == TaskBehavior.TOOL_FAILURE:
            return self.build_result(success=False, error=f"tool failure for {task_id}")
        if behavior == TaskBehavior.SHARED_KEY_WRITE:
            if context is None:
                raise RuntimeError("context is required for shared key write behavior")
            shared_key = str(payload.get("shared_key", "shared_conflict_key"))
            context.set_shared_value(shared_key, {"task_id": task_id, "prompt": prompt}, allow_overwrite=False)
            return self.build_result(
                success=True,
                output={"text": f"shared-write::{prompt}", "task_id": task_id, "shared_key": shared_key},
            )
        if behavior == TaskBehavior.FAIL_ONCE:
            attempts = self._increment_attempt_counter(context, task_id)
            if attempts == 1:
                return self.build_result(success=False, error=f"transient failure for {task_id}")
        if behavior == TaskBehavior.EXECUTOR_CRASH:
            return self.build_result(
                success=True,
                output={"text": f"executor-crash::{prompt}", "consumed_inputs": consumed_inputs},
                metadata={
                    "usage": {
                        "model_name": "broken-usage",
                        "prompt_tokens": -1,
                        "completion_tokens": 1,
                        "total_tokens": 0,
                    }
                },
            )

        return self.build_result(
            success=True,
            output={
                "text": f"{self.name}::{prompt}",
                "task_id": task_id,
                "consumed_inputs": consumed_inputs,
            },
            metadata={
                "usage": {
                    "model_name": self.name,
                    "prompt_tokens": 3,
                    "completion_tokens": 5,
                    "total_tokens": 8,
                }
            },
        )

    def _collect_inputs(self, context: ContextStore | None, read_keys: list[str]) -> dict[str, Any]:
        if context is None:
            return {}
        return {key: context.task_results.get(key) for key in read_keys}

    def _increment_attempt_counter(self, context: ContextStore | None, task_id: str) -> int:
        if context is None:
            return 1
        return context.increment_shared_counter("_attempt_counters", task_id)

    def _append_runtime_trace(self, context: ContextStore | None, record: dict[str, Any]) -> None:
        if context is None:
            return
        context.append_shared_list("runtime_tool_trace", {**record, "recorded_at": utc_now().isoformat()})


class StaticStructuredLLMClient(LLMClient):
    def __init__(self, *, test_input: RuntimeTestInput) -> None:
        super().__init__(timeout_seconds=30, model_name="static-structured-client", model_version="test-v1")
        self.test_input = test_input

    async def _generate(self, request: LLMRequest) -> LLMResponse:
        function_name = request.tool_choice or (request.function_schemas[0].name if request.function_schemas else None)
        if function_name == "route_user_request":
            route = self.test_input.force_route or ("COMPLEX_TASK" if self.test_input.task_specs else "SIMPLE_TASK")
            payload = {
                "route": route,
                "complexity": "complex" if route == "COMPLEX_TASK" else "simple",
                "needs_planning": route == "COMPLEX_TASK",
                "reason": "structured supervisor test response",
            }
        elif function_name == "emit_task_plan":
            payload = json.loads(build_planner_json(self.test_input))
        else:
            payload = {"text": "unsupported function", "summary": "unsupported", "key_points": []}

        function_schema = request.function_schemas[0] if request.function_schemas else None
        return LLMResponse(
            text=json.dumps({"tool_name": function_name, "arguments": payload}),
            model_name=self.model_name,
            model_version=self.model_version,
            request_id=request.request_id,
            session_id=request.session_id,
            trace_id=request.trace_id,
            prompt_name=request.prompt_name,
            prompt_version=request.prompt_version,
            function_call=LLMFunctionCall(
                tool_name=function_name or "unknown",
                arguments=payload,
                schema_name=function_schema.schema_name if function_schema is not None else None,
                schema_version=function_schema.schema_version if function_schema is not None else None,
            ),
            raw_response={"provider": "static_structured_client"},
        )


class ParserRepairTestLLMClient(LLMClient):
    def __init__(self, *, test_input: RuntimeTestInput) -> None:
        super().__init__(timeout_seconds=30, model_name="parser-repair-test-client", model_version="test-v1")
        self.test_input = test_input
        self.call_count = 0

    async def _generate(self, request: LLMRequest) -> LLMResponse:
        self.call_count += 1
        responses = list(self.test_input.parser_repair_outputs)
        if not responses:
            raise RuntimeError("parser repair output is not configured for this test case")
        index = min(self.call_count - 1, len(responses) - 1)
        payload = responses[index]
        return LLMResponse(
            text=payload,
            model_name=self.model_name,
            model_version=self.model_version,
            request_id=request.request_id,
            session_id=request.session_id,
            trace_id=request.trace_id,
            prompt_name=request.prompt_name,
            prompt_version=request.prompt_version,
            raw_response={"provider": "parser_repair_test_client", "call_count": self.call_count},
        )


def build_supervisor_agent(test_input: RuntimeTestInput) -> SupervisorAgent:
    return SupervisorAgent(client=StaticStructuredLLMClient(test_input=test_input))


def build_planner_agent(test_input: RuntimeTestInput) -> LLMTaskPlanner:
    return LLMTaskPlanner(client=StaticStructuredLLMClient(test_input=test_input))
