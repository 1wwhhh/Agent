from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.schemas.llm import LLMRequest, LLMResponse
from app.tools.llm_client import LLMClient


@runtime_checkable
class LLMProvider(Protocol):
    """面向 LLM 工具的异步供应商接口约定。"""

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """根据给定请求生成标准化的 LLM 响应。"""


__all__ = ["LLMClient", "LLMProvider"]
