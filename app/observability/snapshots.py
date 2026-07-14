from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MetricsSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    request_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    phase: str = Field(..., min_length=1)
    supervisor_route: str | None = Field(default=None)
    total_tasks: int = Field(default=0, ge=0)
    successful_tasks: int = Field(default=0, ge=0)
    failed_tasks: int = Field(default=0, ge=0)
    cancelled_tasks: int = Field(default=0, ge=0)
    timeout_tasks: int = Field(default=0, ge=0)
    retry_count: int = Field(default=0, ge=0)
    llm_call_count: int = Field(default=0, ge=0)
    fallback_count: int = Field(default=0, ge=0)
    parser_repair_count: int = Field(default=0, ge=0)
    routing_denied_count: int = Field(default=0, ge=0)
    runtime_latency_ms: int = Field(default=0, ge=0)
    task_success_rate: float = Field(default=0.0, ge=0.0)
    dag_correctness_rate: float = Field(default=0.0, ge=0.0)
    retry_rate: float = Field(default=0.0, ge=0.0)
    context_consistency_rate: float = Field(default=0.0, ge=0.0)
    latency: dict[str, float] = Field(default_factory=dict)


class RuntimeTraceSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    request_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    started_at: str = Field(..., min_length=1)
    finished_at: str = Field(..., min_length=1)
    status: str = Field(..., min_length=1)
    latency_ms: int = Field(default=0, ge=0)
    task_graph: dict[str, Any] = Field(default_factory=dict)
    task_results: dict[str, Any] = Field(default_factory=dict)
    task_failures: list[dict[str, Any]] = Field(default_factory=list)
    parser_repair_history: list[dict[str, Any]] = Field(default_factory=list)
    routing_history: list[dict[str, Any]] = Field(default_factory=list)
    llm_calls: list[dict[str, Any]] = Field(default_factory=list)
    metrics: MetricsSnapshot = Field(...)
    errors: list[dict[str, Any]] = Field(default_factory=list)
    debug_events: list[dict[str, Any]] = Field(default_factory=list)


class ReplaySnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    request_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    raw_user_input: str | None = Field(default=None)
    planner_raw_output: Any | None = Field(default=None)
    repaired_output: Any | None = Field(default=None)
    task_graph: dict[str, Any] = Field(default_factory=dict)
    runtime_context_snapshot: dict[str, Any] = Field(default_factory=dict)
    routing_decisions: list[dict[str, Any]] = Field(default_factory=list)
    llm_calls: list[dict[str, Any]] = Field(default_factory=list)
    final_output: Any | None = Field(default=None)
    errors: list[dict[str, Any]] = Field(default_factory=list)


class DebugSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    request_id: str = Field(..., min_length=1)
    planner_output: dict[str, Any] = Field(default_factory=dict)
    repair_attempts: list[dict[str, Any]] = Field(default_factory=list)
    routing_decisions: list[dict[str, Any]] = Field(default_factory=list)
    executor_events: list[dict[str, Any]] = Field(default_factory=list)
    llm_calls: list[dict[str, Any]] = Field(default_factory=list)
    metrics: MetricsSnapshot = Field(...)
    errors: list[dict[str, Any]] = Field(default_factory=list)
