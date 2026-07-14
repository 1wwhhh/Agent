from __future__ import annotations

from app.adapters.base import ModelAdapter


class QwenAdapter(ModelAdapter):
    @property
    def provider_name(self) -> str:
        return "qwen"

    def extra_headers(self) -> dict[str, str]:
        if not self.config.organization:
            return {}
        return {"OpenAI-Organization": self.config.organization}

    def supports_function_calling_with_streaming(self) -> bool:
        return False

    def stream_options(self) -> dict[str, object]:
        return {"stream_options": {"include_usage": True}}
