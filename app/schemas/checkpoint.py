from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.task import utc_now


class RuntimeCheckpoint(BaseModel):
    """单次运行时快照的可序列化结构。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    checkpoint_id: str = Field(..., min_length=1, description="检查点唯一标识。")
    request_id: str = Field(..., min_length=1, description="关联请求 ID。")
    session_id: str = Field(..., min_length=1, description="关联会话 ID。")
    source_layer: str = Field(..., min_length=1, description="产生日志/快照的运行时层。")
    event: str = Field(..., min_length=1, description="产生日志/快照时的事件类型。")
    created_at: datetime = Field(default_factory=utc_now, description="快照创建时间。")
    snapshot_payload: dict[str, Any] = Field(default_factory=dict, description="完整图状态快照。")
    metadata: dict[str, Any] = Field(default_factory=dict, description="额外快照元数据。")
