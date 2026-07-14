from __future__ import annotations

import json

from app.adapters.base import ModelAdapter
from app.schemas.llm import LLMRequest
from app.utils import configure_runtime_logger


class DeepSeekAdapter(ModelAdapter):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.logger = configure_runtime_logger()

    @property
    def provider_name(self) -> str:
        return "deepseek"

    def build_payload(self, *, request: LLMRequest, stream: bool) -> dict[str, object]:
        payload = super().build_payload(request=request, stream=stream)
        payload["tool_choice"] = self._normalize_tool_choice(
            request_tool_choice=request.tool_choice,
            payload_tool_choice=payload.get("tool_choice"),
        )
        if request.function_schemas:
            payload.setdefault("thinking", {"type": "disabled"})
        self.logger.debug("DeepSeek request payload: %s", json.dumps(payload, ensure_ascii=False, indent=2))
        return payload

    def _normalize_tool_choice(self, *, request_tool_choice: object, payload_tool_choice: object) -> object:
        if isinstance(request_tool_choice, str) and request_tool_choice in {"auto", "none", "required"}:
            return request_tool_choice
        if request_tool_choice is None and payload_tool_choice is None:
            return "auto"
        if isinstance(payload_tool_choice, dict):
            return payload_tool_choice
        if isinstance(payload_tool_choice, str) and payload_tool_choice in {"auto", "none", "required"}:
            return payload_tool_choice
        return "auto"
