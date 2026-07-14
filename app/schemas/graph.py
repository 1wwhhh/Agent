from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GraphPhase(str, Enum):
    """在 LangGraph 各节点之间流转时跟踪的高层运行阶段。"""

    INITIALIZED = "INITIALIZED"
    SUPERVISED = "SUPERVISED"
    PLANNED = "PLANNED"
    PARSED = "PARSED"
    QUEUED = "QUEUED"
    ROUTED = "ROUTED"
    EXECUTING = "EXECUTING"
    AGGREGATING = "AGGREGATING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class GraphStateSnapshot(BaseModel):
    """用于检查点与调试的可序列化 LangGraph 状态快照。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    request_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    phase: GraphPhase = Field(..., description="Current runtime phase.")
    current_node: str | None = Field(default=None, description="Currently active graph node.")
    last_completed_node: str | None = Field(default=None, description="Most recently completed graph node.")
    supervisor_route: str | None = Field(default=None, description="Supervisor decision such as SIMPLE_TASK or COMPLEX_TASK.")
    current_task_id: str | None = Field(default=None, description="Currently executing task id if any.")
    pending_task_ids: list[str] = Field(default_factory=list)
    completed_task_ids: list[str] = Field(default_factory=list)
    failed_task_ids: list[str] = Field(default_factory=list)
    final_output_ready: bool = Field(default=False)
    final_output: Any | None = Field(default=None)
