from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.router import ToolRouteDecision
from app.schemas.task import TaskStatus
from app.schemas.tool import ToolResult


class ExecutorErrorDetail(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    code: str = Field(..., min_length=1, description="Stable executor error code.")
    message: str = Field(..., min_length=1, description="Human-readable executor failure message.")
    details: dict[str, Any] = Field(default_factory=dict, description="Structured executor error metadata.")


class TaskExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    task_id: str = Field(..., min_length=1)
    output_key: str = Field(..., min_length=1)
    idempotency_key: str | None = Field(default=None, description="Logical task idempotency key.")
    attempt_key: str | None = Field(default=None, description="Execution-attempt key for the current run.")
    success: bool = Field(..., description="Whether the task reached SUCCESS.")
    final_status: TaskStatus = Field(..., description="Final task status after this execution attempt.")
    attempt: int = Field(..., ge=0, description="Retry attempt index after execution handling.")
    routed_tool: str | None = Field(default=None, description="Resolved tool used by the executor.")
    retry_scheduled: bool = Field(default=False, description="Whether the task was moved back to RETRY.")
    error_message: str | None = Field(default=None, description="Execution error when unsuccessful.")
    route_decision: ToolRouteDecision | None = Field(default=None, description="Structured router decision.")
    tool_result: ToolResult | None = Field(default=None, description="Raw normalized tool result when available.")
    started_at: datetime = Field(..., description="Execution start time.")
    finished_at: datetime = Field(..., description="Execution end time.")
