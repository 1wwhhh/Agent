from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.prompts import (
    PromptRegistry,
    SUPERVISOR_PROMPT_NAME,
    SUPERVISOR_PROMPT_VERSION,
    build_default_prompt_registry,
    build_supervisor_prompt,
)
from app.schemas.llm import LLMFunctionSchema, LLMMessage, LLMRequest
from app.schemas.supervisor import SupervisorDecision
from app.tools.function_calling import FunctionCallingAdapter
from app.tools.llm_client import LLMClient
from app.utils import runtime_progress


class SupervisorAgentResult(BaseModel):
    """结构化的 Supervisor 判定结果，同时携带请求链路追踪所需的请求元数据。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", validate_assignment=True)

    decision: SupervisorDecision = Field(...)
    prompt_name: str = Field(..., min_length=1)
    prompt_version: str = Field(..., min_length=1)
    trace_id: str = Field(..., min_length=1)
    model_name: str | None = Field(default=None)
    model_version: str | None = Field(default=None)


class SupervisorAgent:
    """基于大模型能力的 Supervisor 封装，负责输出结构化的任务入口路由决策。"""

    def __init__(
        self,
        *,
        client: LLMClient,
        prompt_registry: PromptRegistry | None = None,
        function_adapter: FunctionCallingAdapter | None = None,
        model_name: str | None = None,
        temperature: float = 0.1,
        max_validation_retries: int = 1,
    ) -> None:
        self.client = client
        self.prompt_registry = prompt_registry or build_default_prompt_registry()
        self.function_adapter = function_adapter or FunctionCallingAdapter()
        self.model_name = model_name
        self.temperature = temperature
        self.max_validation_retries = max_validation_retries
        
    # 对用户输入做分类判断
    async def classify(
        self,
        *,
        user_input: str,
        request_id: str,
        session_id: str,
        context_summary: str | None = None,
    ) -> SupervisorAgentResult:
        prompt_bundle = build_supervisor_prompt(user_input=user_input, context_summary=context_summary)
        trace_id = f"{request_id}:supervisor:{uuid4().hex}"
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
            prompt_name=SUPERVISOR_PROMPT_NAME,
            prompt_version=SUPERVISOR_PROMPT_VERSION,
            response_schema_name=SupervisorDecision.__name__,
            response_schema_version="v1",
            max_validation_retries=self.max_validation_retries,
            metadata={"component": "supervisor"},
        )
        result = await self.function_adapter.invoke_structured(
            client=self.client,
            request=request,
            function_schema=LLMFunctionSchema(
                name="route_user_request",
                description="Return the supervisor route decision for the request.",
                parameters_schema=SupervisorDecision.model_json_schema(),
                schema_name=SupervisorDecision.__name__,
                schema_version="v1",
            ),
            output_model=SupervisorDecision,
        )
        decision = result.output
        runtime_progress(
            step="supervisor",
            status="决策详情",
            detail="已完成请求类型判断",
        )
        return SupervisorAgentResult(
            decision=result.output,
            prompt_name=prompt_bundle.prompt_name,
            prompt_version=prompt_bundle.prompt_version,
            trace_id=result.response.trace_id or trace_id,
            model_name=result.response.model_name,
            model_version=result.response.model_version,
        )
