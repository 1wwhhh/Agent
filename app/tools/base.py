from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.context import ContextStore
from app.schemas.task import utc_now
from app.schemas.tool import ToolResult


class BaseTool(BaseModel, ABC):
    """所有 Runtime 工具都必须继承的基础接口。"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    name: str = Field(..., min_length=1, description="Stable registry name for routing.")
    description: str = Field(..., min_length=1, description="Human-readable tool description.")
    timeout: int = Field(default=60, gt=0, description="Maximum execution time in seconds.")
    enabled: bool = Field(default=True, description="Whether the tool can be routed.")
    tags: list[str] = Field(default_factory=list, description="Optional routing and audit tags.")
    schema_version: str = Field(default="v1", min_length=1, description="Version of the tool input/output schema.")

    def run(self, payload: dict[str, Any], context: ContextStore | None = None) -> ToolResult:
        """为尚未运行事件循环的环境提供的同步包装入口。"""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun(payload=payload, context=context))

        raise RuntimeError("run() cannot be used inside an active event loop; use await arun() instead")

    async def arun(self, payload: dict[str, Any], context: ContextStore | None = None) -> ToolResult:
        """供执行器调用的异步运行入口。"""
        if not self.enabled:
            raise RuntimeError(f"tool '{self.name}' is disabled")

        started_at = utc_now()
        timeout_seconds = self.resolve_timeout(payload)
        result = await asyncio.wait_for(
            self._arun(payload=payload, context=context),
            timeout=timeout_seconds,
        )
        return self._finalize_result(result=result, started_at=started_at)

    @abstractmethod
    async def _arun(self, payload: dict[str, Any], context: ContextStore | None = None) -> ToolResult:
        """异步执行工具，并返回标准化 ToolResult。"""

    def get_input_schema(self) -> dict[str, Any]:
        """返回当前工具输入载荷的 JSON Schema。"""
        return {
            "type": "object",
            "additionalProperties": True,
        }

    def get_output_schema(self) -> dict[str, Any]:
        """返回当前工具输出载荷的 JSON Schema。"""
        return {
            "type": "object",
            "additionalProperties": True,
        }

    def get_routing_capability(self) -> dict[str, Any]:
        return {
            "tool_name": self.name,
            "enabled": self.enabled,
            "supported_task_types": [],
            "default_task_type": None,
            "supported_tags": [],
            "required_permissions": [],
            "allowed_roles": [],
            "max_concurrency": 64,
            "supports_streaming": False,
            "supports_retry": True,
            "supports_timeout": True,
        }

    def build_result(
        self,
        *,
        success: bool,
        output: Any | None = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
    ) -> ToolResult:
        """供子类构造标准化 ToolResult 的辅助方法。"""
        result_started_at = started_at or utc_now()
        result_finished_at = finished_at or utc_now()
        latency_ms = max(
            0,
            int((result_finished_at - result_started_at).total_seconds() * 1000),
        )
        return ToolResult(
            tool_name=self.name,
            success=success,
            output=output,
            error=error,
            metadata=metadata or {},
            started_at=result_started_at,
            finished_at=result_finished_at,
            latency_ms=latency_ms,
        )

    def _finalize_result(self, result: ToolResult, started_at: datetime) -> ToolResult:
        finished_at = utc_now()
        latency_ms = max(0, int((finished_at - started_at).total_seconds() * 1000))
        return result.model_copy(
            update={
                "tool_name": self.name,
                "started_at": started_at,
                "finished_at": finished_at,
                "latency_ms": latency_ms,
            }
        )

    def resolve_timeout(self, payload: dict[str, Any]) -> int:
        timeout_value = payload.get("tool_timeout_seconds")
        if timeout_value is None:
            timeout_value = payload.get("timeout_seconds", self.timeout)
        try:
            resolved_timeout = int(timeout_value)
        except (TypeError, ValueError):
            resolved_timeout = self.timeout
        return resolved_timeout if resolved_timeout > 0 else self.timeout
