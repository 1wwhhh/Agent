from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PromptTemplate(BaseModel):
    """注册在 prompts 目录下的带版本 Prompt 模板。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    name: str = Field(..., min_length=1)
    version: str = Field(..., min_length=1)
    description: str = Field(default="", description="Human-readable template description.")
    system_template: str = Field(..., min_length=1)
    user_template: str = Field(..., min_length=1)


class RenderedPrompt(BaseModel):
    """传入 LLM 请求的已渲染 Prompt 载荷。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    name: str = Field(..., min_length=1)
    version: str = Field(..., min_length=1)
    system_prompt: str = Field(..., min_length=1)
    user_prompt: str = Field(..., min_length=1)
    variables: dict[str, Any] = Field(default_factory=dict)

