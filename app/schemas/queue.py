from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class QueueErrorDetail(BaseModel):
    """描述调度失败的结构化队列错误模型。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    code: str = Field(..., min_length=1, description="Stable queue error code.")
    message: str = Field(..., min_length=1, description="Human-readable scheduling error message.")
    details: dict[str, Any] = Field(default_factory=dict, description="Structured scheduling error metadata.")


class QueueSnapshot(BaseModel):
    """用于上下文持久化与调试的可序列化队列快照。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    total_tasks: int = Field(..., ge=0)
    max_concurrency: int = Field(..., gt=0)
    available_slots: int = Field(..., ge=0)
    ready_task_ids: list[str] = Field(default_factory=list)
    blocked_task_ids: list[str] = Field(default_factory=list)
    queued_task_ids: list[str] = Field(default_factory=list)
    running_task_ids: list[str] = Field(default_factory=list)
    retry_task_ids: list[str] = Field(default_factory=list)
    completed_task_ids: list[str] = Field(default_factory=list)
    failed_task_ids: list[str] = Field(default_factory=list)
