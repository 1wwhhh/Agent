from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.planner import TaskPlan
from app.schemas.task import TaskModel, utc_now


class ParserErrorDetail(BaseModel):
    """Structured parser failure metadata."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    code: str = Field(..., min_length=1, description="Stable parser error code.")
    message: str = Field(..., min_length=1, description="Human-readable parse failure message.")
    stage: str = Field(..., min_length=1, description="Parser stage that failed.")
    raw_text: str | None = Field(default=None, description="Original model output when safe to store.")
    details: dict[str, Any] = Field(default_factory=dict, description="Structured parse error metadata.")


class TaskParserResult(BaseModel):
    """Validated runtime task parse result."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    raw_plan: TaskPlan = Field(..., description="Validated planner output contract.")
    tasks: list[TaskModel] = Field(..., min_length=1, description="Runtime task models derived from the plan.")


class RepairType(str, Enum):
    INVALID_JSON = "INVALID_JSON"
    INVALID_SCHEMA = "INVALID_SCHEMA"
    MISSING_FIELD = "MISSING_FIELD"
    INVALID_DEPENDENCY = "INVALID_DEPENDENCY"
    UNSUPPORTED_TAGS = "UNSUPPORTED_TAGS"
    INTERNAL_KNOWLEDGE_REQUIRES_RAG = "INTERNAL_KNOWLEDGE_REQUIRES_RAG"
    UNSUPPORTED_MODEL_NAME = "UNSUPPORTED_MODEL_NAME"


class RepairResult(BaseModel):
    """Structured parser repair attempt record."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    success: bool = Field(..., description="Whether this repair attempt produced a valid TaskPlan.")
    repair_type: RepairType = Field(..., description="Classified parser failure type.")
    retry_count: int = Field(..., ge=1, description="1-based repair attempt index.")
    repaired_output: str | None = Field(default=None, description="Raw repaired planner output.")
    error_message: str | None = Field(default=None, description="Failure message for this repair attempt.")
    output_contract_violation: bool = Field(
        default=False,
        description="Whether the repair output violated the strict JSON-only contract.",
    )
    violation_reason: str | None = Field(
        default=None,
        description="Structured contract violation reason when the repair output is not strict JSON.",
    )
    latency_ms: float = Field(..., ge=0.0, description="End-to-end attempt latency in milliseconds.")
    timestamp: datetime = Field(default_factory=utc_now, description="Repair attempt timestamp.")
