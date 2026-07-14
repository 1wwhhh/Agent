from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.task import utc_now


class ToolResult(BaseModel):
    """执行器层使用的标准化工具执行结果。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    tool_name: str = Field(..., min_length=1, description="Tool name that produced the result.")
    success: bool = Field(..., description="Whether the tool execution succeeded.")
    output: Any | None = Field(default=None, description="Tool output payload.")
    error: str | None = Field(default=None, description="Failure message when execution is unsuccessful.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Structured execution metadata.")
    started_at: datetime = Field(default_factory=utc_now, description="Execution start time.")
    finished_at: datetime = Field(default_factory=utc_now, description="Execution finish time.")
    latency_ms: int | None = Field(default=None, ge=0, description="Measured execution latency in milliseconds.")

    @model_validator(mode="after")
    def validate_consistency(self) -> "ToolResult":
        if self.success and self.error:
            raise ValueError("successful tool result must not contain error")
        if not self.success and not self.error:
            raise ValueError("failed tool result must contain error")
        return self
