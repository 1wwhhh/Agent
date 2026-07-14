from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.context import TokenUsage


class LLMMessage(BaseModel):
    """面向模型供应商请求的标准化对话消息模型。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    role: str = Field(..., min_length=1, description="Message role, such as system, user, or assistant.")
    content: str = Field(..., min_length=1, description="Message content.")
    name: str | None = Field(default=None, description="Optional message participant name.")


class LLMFunctionSchema(BaseModel):
    """暴露给模型使用的标准化函数或工具 Schema。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    name: str = Field(..., min_length=1, description="Stable function name exposed to the model.")
    description: str = Field(..., min_length=1, description="Human-readable function description.")
    parameters_schema: dict[str, Any] = Field(..., description="JSON schema for function arguments.")
    schema_name: str = Field(..., min_length=1, description="Semantic schema name for tracing.")
    schema_version: str = Field(..., min_length=1, description="Schema version identifier.")


class LLMFunctionCall(BaseModel):
    """模型选择后的标准化函数调用载荷。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    tool_name: str = Field(..., min_length=1, description="Selected function or tool name.")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Validated JSON arguments payload.")
    schema_name: str | None = Field(default=None, description="Schema name used to validate arguments.")
    schema_version: str | None = Field(default=None, description="Schema version used to validate arguments.")


class LLMRequest(BaseModel):
    """与供应商无关的 LLM 请求契约。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    prompt: str | None = Field(default=None, description="Primary user prompt.")
    system_prompt: str | None = Field(default=None, description="Optional system instruction.")
    messages: list[LLMMessage] = Field(default_factory=list, description="Optional prebuilt message list.")
    model_name: str | None = Field(default=None, description="Preferred model name.")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="Sampling temperature.")
    max_tokens: int | None = Field(default=None, gt=0, description="Maximum completion tokens.")
    timeout_seconds: int | None = Field(default=None, gt=0, description="Optional per-request timeout override.")
    stream: bool = Field(default=False, description="Whether streaming output is requested.")
    request_id: str | None = Field(default=None, description="Runtime request identifier for tracing.")
    session_id: str | None = Field(default=None, description="Runtime session identifier for tracing.")
    trace_id: str | None = Field(default=None, description="Per-call tracing id for observability.")
    prompt_name: str | None = Field(default=None, description="Prompt template name.")
    prompt_version: str | None = Field(default=None, description="Prompt template version.")
    response_schema_name: str | None = Field(default=None, description="Expected structured response schema name.")
    response_schema_version: str | None = Field(default=None, description="Expected structured response schema version.")
    function_schemas: list[LLMFunctionSchema] = Field(default_factory=list, description="Available structured functions.")
    tool_choice: str | None = Field(default=None, description="Preferred function name when function calling is used.")
    max_validation_retries: int = Field(default=1, ge=0, description="Retries allowed after schema validation errors.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional provider metadata.")

    @model_validator(mode="after")
    def validate_prompt_or_messages(self) -> "LLMRequest":
        if not self.prompt and not self.messages:
            raise ValueError("LLM 请求必须至少提供 prompt 或 messages 之一。")
        return self


class LLMResponse(BaseModel):
    """与供应商无关的标准化 LLM 响应。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    text: str = Field(..., min_length=1, description="Primary response text.")
    model_name: str | None = Field(default=None, description="Resolved model name used by the provider.")
    model_version: str | None = Field(default=None, description="Resolved model version used by the provider.")
    finish_reason: str | None = Field(default=None, description="Provider finish reason if available.")
    request_id: str | None = Field(default=None, description="Request identifier echoed from the request.")
    session_id: str | None = Field(default=None, description="Session identifier echoed from the request.")
    trace_id: str | None = Field(default=None, description="Per-call tracing identifier.")
    prompt_name: str | None = Field(default=None, description="Prompt template name used for the call.")
    prompt_version: str | None = Field(default=None, description="Prompt template version used for the call.")
    function_call: LLMFunctionCall | None = Field(default=None, description="Normalized function call returned by the model.")
    usage: TokenUsage | None = Field(default=None, description="Normalized token usage record.")
    raw_response: dict[str, Any] = Field(default_factory=dict, description="Optional raw provider payload.")


class LLMResponseChunk(BaseModel):
    """客户端在流式输出时发出的可选响应分片。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    delta_text: str = Field(default="", description="Incremental text delta.")
    trace_id: str | None = Field(default=None, description="Per-call tracing identifier.")
    raw_chunk: dict[str, Any] = Field(default_factory=dict, description="Optional raw provider chunk payload.")
