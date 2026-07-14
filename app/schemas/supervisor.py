from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SupervisorDecision(BaseModel):
    """结构化的 Supervisor 路由决策模型。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    route: str = Field(..., pattern="^(SIMPLE_TASK|COMPLEX_TASK)$")
    complexity: str = Field(..., pattern="^(simple|complex)$")
    needs_planning: bool = Field(..., description="Whether the request should go through planner decomposition.")
    reason: str = Field(..., min_length=1, description="Short structured rationale for observability.")
