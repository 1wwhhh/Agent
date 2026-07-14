from __future__ import annotations

from app.adapters.registry import ModelRegistry, build_default_model_registry
from app.schemas.model import RuntimeLLMConfig
from app.tools.llm_client import LLMClient


class ModelRouter:
    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or build_default_model_registry()

    def build_client(self, config: RuntimeLLMConfig) -> LLMClient:
        adapters = [self.registry.build_adapter(config.primary)]
        adapters.extend(self.registry.build_adapter(item) for item in config.fallbacks)
        return LLMClient(
            adapters=adapters,
            timeout_seconds=config.primary.timeout_seconds,
            model_name=config.primary.model_name,
            model_version=None,
            supports_streaming=any(adapter.supports_streaming for adapter in adapters),
            max_retries=config.primary.max_retries,
        )
