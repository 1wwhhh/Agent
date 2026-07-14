from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentRequest(BaseModel):
    """Runtime 网关的 HTTP 请求载荷模型。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    user_input: str = Field(..., min_length=1)
    session_id: str | None = Field(default=None)
    request_id: str | None = Field(default=None)
    client_timezone: str | None = Field(default=None)


class AgentTaskState(BaseModel):
    """API 层返回的可序列化任务状态模型。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    task_id: str = Field(..., min_length=1)
    status: str = Field(..., min_length=1)
    retry_count: int = Field(..., ge=0)
    max_retry: int = Field(..., ge=0)
    depends_on: list[str] = Field(default_factory=list)
    output_key: str = Field(..., min_length=1)
    tool: str = Field(..., min_length=1)


class AgentResponse(BaseModel):
    """Runtime 网关的 HTTP 响应载荷模型。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    request_id: str = Field(..., min_length=1)
    result: dict[str, Any] = Field(default_factory=dict)
    task_states: list[AgentTaskState] = Field(default_factory=list)
    trace: dict[str, Any] = Field(default_factory=dict)
