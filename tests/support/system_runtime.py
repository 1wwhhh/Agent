from __future__ import annotations

from functools import partial
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.api.schemas import AgentRequest
from app.agents import SupervisorAgent
from app.planner import LLMTaskPlanner
from app.router import TaskRouter
from app.router.capability import capability_from_tool
from app.tools.base import BaseTool
from tests.support.models import RuntimeTestInput
from tests.support.test_case_generator import TestCaseGenerator
from tests.support.tools import RuntimeTestTool, StaticStructuredLLMClient


class SystemRuntimeComponents(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", validate_assignment=True)

    client: StaticStructuredLLMClient = Field(...)
    repair_llm_client: StaticStructuredLLMClient = Field(...)
    router: TaskRouter = Field(...)
    supervisor_agent: SupervisorAgent | None = Field(default=None)
    planner_agent: LLMTaskPlanner | None = Field(default=None)
    planner_tool: BaseTool | None = Field(default=None)


def build_system_case_registry() -> dict[str, RuntimeTestInput]:
    cases = [
        *TestCaseGenerator.system_simple_task_cases(),
        *TestCaseGenerator.system_complex_dag_cases(),
    ]
    return {case.user_input: case for case in cases}


async def build_system_runtime_components(agent_request: AgentRequest) -> SystemRuntimeComponents:
    registry = build_system_case_registry()
    runtime_input = registry.get(agent_request.user_input)
    if runtime_input is None:
        raise ValueError(f"no predefined system test case registered for input: {agent_request.user_input}")

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
    return SystemRuntimeComponents(
        client=client,
        repair_llm_client=client,
        router=router,
        supervisor_agent=supervisor_agent,
        planner_agent=planner_agent,
    )
