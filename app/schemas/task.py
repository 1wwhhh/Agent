from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    RETRY = "RETRY"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"


class RetryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    retry_count: int = Field(default=0, ge=0)
    max_retry: int = Field(default=2, ge=0)

    def can_retry(self) -> bool:
        return self.retry_count < self.max_retry

    def mark_retry(self) -> "RetryModel":
        self.retry_count += 1
        return self


class TaskModel(RetryModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
        str_strip_whitespace=True,
    )

    task_id: str = Field(..., min_length=1)
    task_name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    task_type: str | None = Field(default=None, description="Optional routing task type.")
    tool: str = Field(..., min_length=1)
    input: dict[str, Any] = Field(default_factory=dict)
    output_key: str = Field(..., min_length=1)
    idempotency_key: str | None = Field(default=None)
    irreversible: bool = Field(default=False)
    depends_on: list[str] = Field(default_factory=list)
    priority: int = Field(default=100, ge=0)
    tags: list[str] = Field(default_factory=list)
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    timeout: int = Field(default=60, gt=0)
    created_at: datetime = Field(default_factory=utc_now)

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
    def validate_task_relationships(self) -> "TaskModel":
        if self.task_id in self.depends_on:
            raise ValueError("task cannot depend on itself")
        if self.idempotency_key is not None and not self.idempotency_key.strip():
            raise ValueError("idempotency_key cannot be blank")
        if self.task_type is not None and not self.task_type.strip():
            raise ValueError("task_type cannot be blank")
        return self
