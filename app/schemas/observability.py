from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.task import utc_now


class ReplayMode(str, Enum):
    FULL = "FULL"
    STEP_BY_STEP = "STEP_BY_STEP"


class RequestMetricsSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    request_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    phase: str = Field(..., min_length=1)
    supervisor_route: str | None = Field(default=None)
    task_success_rate: float = Field(..., ge=0.0)
    dag_correctness_rate: float = Field(..., ge=0.0)
    retry_rate: float = Field(..., ge=0.0)
    retry_count: int = Field(..., ge=0)
    context_consistency_rate: float = Field(..., ge=0.0)
    latency: dict[str, float] = Field(default_factory=dict)
    recorded_at: str = Field(default_factory=lambda: utc_now().isoformat())


class PersistedTraceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    request_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    sequence: int = Field(..., ge=0)
    layer: str = Field(..., min_length=1)
    event: str = Field(..., min_length=1)
    timestamp: str = Field(default_factory=lambda: utc_now().isoformat())
    phase: str | None = Field(default=None)
    current_node: str | None = Field(default=None)
    last_completed_node: str | None = Field(default=None)
    supervisor_route: str | None = Field(default=None)
    node_execution_order: list[str] = Field(default_factory=list)
    task_states: dict[str, dict[str, Any]] = Field(default_factory=dict)
    task_graph: dict[str, Any] = Field(default_factory=dict)
    execution_history: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    context_snapshot: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PersistedExecutionTrace(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    request_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    created_at: str = Field(default_factory=lambda: utc_now().isoformat())
    updated_at: str = Field(default_factory=lambda: utc_now().isoformat())
    events: list[PersistedTraceEvent] = Field(default_factory=list)
    metrics: RequestMetricsSnapshot | None = Field(default=None)


class ReplayStep(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    sequence: int = Field(..., ge=0)
    layer: str = Field(..., min_length=1)
    event: str = Field(..., min_length=1)
    phase: str | None = Field(default=None)
    task_states: dict[str, dict[str, Any]] = Field(default_factory=dict)
    task_graph: dict[str, Any] = Field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    context_snapshot: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(..., min_length=1)


class ReplayResult(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    request_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    mode: ReplayMode = Field(...)
    steps: list[ReplayStep] = Field(default_factory=list)
    final_output: dict[str, Any] = Field(default_factory=dict)
    task_states: dict[str, dict[str, Any]] = Field(default_factory=dict)
    trace_summary: dict[str, Any] = Field(default_factory=dict)
