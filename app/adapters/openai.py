from __future__ import annotations

from app.adapters.base import ModelAdapter


class OpenAIAdapter(ModelAdapter):
    @property
    def provider_name(self) -> str:
        return "openai"

    def extra_headers(self) -> dict[str, str]:
        if not self.config.organization:
            return {}
        return {"OpenAI-Organization": self.config.organization}

    def stream_options(self) -> dict[str, object]:
        return {"stream_options": {"include_usage": True}}
