from __future__ import annotations

from collections.abc import Sequence
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.prompts import PromptRegistry, ToolDefinition, build_default_prompt_registry, build_task_planner_prompt
from app.schemas.llm import LLMFunctionSchema, LLMMessage, LLMRequest
from app.schemas.planner import TaskPlan
from app.tools.function_calling import FunctionCallingAdapter
from app.tools.llm_client import LLMClient
from app.utils import runtime_progress


class LLMPlannerResult(BaseModel):
    """结构化的 Planner 输出结果，同时包含模型追踪元数据。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", validate_assignment=True)

    task_plan: TaskPlan = Field(...)
    prompt_name: str = Field(..., min_length=1)
    prompt_version: str = Field(..., min_length=1)
    trace_id: str = Field(..., min_length=1)
    model_name: str | None = Field(default=None)
    model_version: str | None = Field(default=None)


class LLMTaskPlanner:
    """基于大模型的任务规划器，负责输出结构化的 TaskPlan DAG。"""

    def __init__(
        self,
        *,
        client: LLMClient,
        prompt_registry: PromptRegistry | None = None,
        function_adapter: FunctionCallingAdapter | None = None,
        model_name: str | None = None,
        temperature: float = 0.2,
        max_validation_retries: int = 1,
    ) -> None:
        self.client = client
        self.prompt_registry = prompt_registry or build_default_prompt_registry()
        self.function_adapter = function_adapter or FunctionCallingAdapter()
        self.model_name = model_name
        self.temperature = temperature
        self.max_validation_retries = max_validation_retries

    async def plan(
        self,
        *,
        user_input: str,
        request_id: str,
        session_id: str,
        tools: Sequence[ToolDefinition | dict[str, object]],
        context_summary: str | None = None,
    ) -> LLMPlannerResult:
        prompt_bundle = build_task_planner_prompt(
            user_input=user_input,
            tools=tools,
            context_summary=context_summary,
        )
        trace_id = f"{request_id}:planner:{uuid4().hex}"
        request = LLMRequest(
            prompt=prompt_bundle.user_prompt,
            system_prompt=prompt_bundle.system_prompt,
            messages=[
                LLMMessage(role="system", content=prompt_bundle.system_prompt),
                LLMMessage(role="user", content=prompt_bundle.user_prompt),
            ],
            model_name=self.model_name,
            temperature=self.temperature,
            request_id=request_id,
            session_id=session_id,
            trace_id=trace_id,
            prompt_name=prompt_bundle.prompt_name,
            prompt_version=prompt_bundle.prompt_version,
            response_schema_name=TaskPlan.__name__,
            response_schema_version="v1",
            max_validation_retries=self.max_validation_retries,
            metadata={"component": "planner", "operation": "planner"},
        )
        result = await self.function_adapter.invoke_structured(
            client=self.client,
            request=request,
            function_schema=LLMFunctionSchema(
                name="emit_task_plan",
                description="Return the executable TaskPlan DAG for the runtime.",
                parameters_schema=TaskPlan.model_json_schema(),
                schema_name=TaskPlan.__name__,
                schema_version="v1",
            ),
            output_model=TaskPlan,
        )
        task_plan = result.output
        runtime_progress(step="planner:目标", status="已确认", detail=task_plan.goal)
        for task in task_plan.tasks:
            deps_str = f" ← [{', '.join(task.depends_on)}]" if task.depends_on else ""
            runtime_progress(
                step=f"  └─ {task.task_id}",
                status="任务",
                detail=f"{task.task_name} | 工具={task.tool}{deps_str}",
            )
        return LLMPlannerResult(
            task_plan=result.output,
            prompt_name=prompt_bundle.prompt_name,
            prompt_version=prompt_bundle.prompt_version,
            trace_id=result.response.trace_id or trace_id,
            model_name=result.response.model_name,
            model_version=result.response.model_version,
        )
