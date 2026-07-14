from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PlannerScenario(str, Enum):
    VALID = "valid"
    INVALID_JSON = "invalid_json"
    INVALID_SCHEMA = "invalid_schema"


class TaskBehavior(str, Enum):
    SUCCESS = "success"
    FAIL_ONCE = "fail_once"
    TOOL_FAILURE = "tool_failure"
    TOOL_EXCEPTION = "tool_exception"
    TIMEOUT = "timeout"
    EXECUTOR_CRASH = "executor_crash"
    SHARED_KEY_WRITE = "shared_key_write"


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    task_id: str = Field(..., min_length=1)
    task_name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    tool: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1)
    output_key: str = Field(..., min_length=1)
    idempotency_key: str | None = Field(default=None)
    irreversible: bool = Field(default=False)
    depends_on: list[str] = Field(default_factory=list)
    priority: int = Field(default=10, ge=0)
    max_retry: int = Field(default=1, ge=0)
    timeout: int = Field(default=60, gt=0)
    behavior: TaskBehavior = Field(default=TaskBehavior.SUCCESS)
    read_keys: list[str] = Field(default_factory=list)
    delay_seconds: float = Field(default=0.0, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuntimeTestInput(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)

    name: str = Field(..., min_length=1)
    user_input: str = Field(..., min_length=1)
    force_route: str | None = Field(default=None)
    planner_scenario: PlannerScenario = Field(default=PlannerScenario.VALID)
    planner_raw_output: str | None = Field(default=None)
    parser_repair_outputs: list[str] = Field(default_factory=list)
    task_specs: list[TaskSpec] = Field(default_factory=list)
    queue_max_concurrency: int = Field(default=4, gt=0)
    planner_model_name: str | None = Field(default=None)
    planner_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    use_supervisor_agent: bool = Field(default=False)
    use_planner_agent: bool = Field(default=False)
    runtime_metadata: dict[str, Any] = Field(default_factory=dict)
    graph_metadata: dict[str, Any] = Field(default_factory=dict)


class RuntimeMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    task_success_rate: float = Field(..., ge=0.0, le=1.0)
    dag_correctness_rate: float = Field(..., ge=0.0, le=1.0)
    retry_rate: float = Field(..., ge=0.0)
    context_consistency_rate: float = Field(..., ge=0.0, le=1.0)
    latencies_ms: dict[str, float] = Field(default_factory=dict)


class TestResult(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    final_output: Any | None = Field(default=None)
    task_states: dict[str, dict[str, Any]] = Field(default_factory=dict)
    context_snapshot: dict[str, Any] = Field(default_factory=dict)
    execution_trace: dict[str, Any] = Field(default_factory=dict)
    metrics: RuntimeMetrics = Field(...)
