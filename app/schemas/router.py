from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RouterErrorDetail(BaseModel):
    """描述任务到工具路由失败的结构化路由错误模型。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    code: str = Field(..., min_length=1, description="Stable router error code.")
    message: str = Field(..., min_length=1, description="Human-readable routing failure message.")
    details: dict[str, Any] = Field(default_factory=dict, description="Structured routing error metadata.")


class ToolRouteCandidate(BaseModel):
    """路由阶段纳入评估的候选工具。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    tool_name: str = Field(..., min_length=1, description="Candidate tool registry name.")
    available: bool = Field(..., description="Whether the tool exists in the registry.")
    enabled: bool = Field(..., description="Whether the candidate tool is enabled for routing.")
    score: int = Field(..., ge=0, description="Relative routing score assigned by the router.")
    reason: str = Field(..., min_length=1, description="Why the router considered this candidate.")


class ToolRouteDecision(BaseModel):
    """任务路由器返回的可序列化路由决策。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    task_id: str = Field(..., min_length=1, description="Task identifier being routed.")
    requested_tool: str = Field(..., min_length=1, description="Primary tool requested by the task.")
    selected_tool: str = Field(..., min_length=1, description="Tool selected by the router.")
    routing_mode: str = Field(..., min_length=1, description="Routing mode such as static, dynamic, or failover.")
    reason: str = Field(..., min_length=1, description="High-level reason for the selected route.")
    candidate_tools: list[ToolRouteCandidate] = Field(default_factory=list, description="Ordered candidate tools.")
    failover_chain: list[str] = Field(default_factory=list, description="Candidate tool names skipped before selection.")
