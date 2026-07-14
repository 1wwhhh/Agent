from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.task import TaskStatus


class PlannerTask(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    task_id: str = Field(..., min_length=1)
    task_name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    task_type: str | None = Field(default=None)
    tool: str = Field(..., min_length=1)
    input: dict[str, Any] = Field(...)
    output_key: str = Field(..., min_length=1)
    idempotency_key: str | None = Field(default=None)
    irreversible: bool = Field(default=False)
    depends_on: list[str] = Field(...)
    priority: int = Field(..., ge=0)
    tags: list[str] = Field(default_factory=list)
    status: TaskStatus = Field(...)
    retry_count: int = Field(..., ge=0)
    max_retry: int = Field(..., ge=0)
    timeout: int = Field(..., gt=0)
    created_at: datetime = Field(...)

    @field_validator("depends_on")
    @classmethod
    def normalize_dependencies(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for dependency in value:
            dependency_id = dependency.strip()
            if not dependency_id:
                raise ValueError("depends_on cannot contain empty task ids")
            if dependency_id not in seen:
                normalized.append(dependency_id)
                seen.add(dependency_id)
        return normalized

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for tag in value:
            tag_value = tag.strip()
            if not tag_value:
                raise ValueError("tags cannot contain empty values")
            if tag_value not in seen:
                normalized.append(tag_value)
                seen.add(tag_value)
        return normalized

    @model_validator(mode="after")
    def validate_relationships(self) -> "PlannerTask":
        if self.task_id in self.depends_on:
            raise ValueError("task cannot depend on itself")
        if self.retry_count > self.max_retry:
            raise ValueError("retry_count cannot be greater than max_retry")
        if self.idempotency_key is not None and not self.idempotency_key.strip():
            raise ValueError("idempotency_key cannot be blank")
        if self.task_type is not None and not self.task_type.strip():
            raise ValueError("task_type cannot be blank")
        return self


class TaskPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    goal: str = Field(..., min_length=1)
    tasks: list[PlannerTask] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_unique_task_ids(self) -> "TaskPlan":
        task_ids = [task.task_id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task ids must be unique")
        return self
