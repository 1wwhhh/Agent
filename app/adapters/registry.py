from __future__ import annotations

from typing import Any

from app.adapters.base import ModelAdapter
from app.adapters.deepseek import DeepSeekAdapter
from app.adapters.openai import OpenAIAdapter
from app.adapters.qwen import QwenAdapter
from app.schemas.model import ModelConfig, ModelProvider


class ModelRegistry:
    def __init__(self) -> None:
        self._adapter_types: dict[ModelProvider, type[ModelAdapter]] = {}

    def register(self, provider: ModelProvider, adapter_type: type[ModelAdapter]) -> None:
        self._adapter_types[provider] = adapter_type

    def build_adapter(self, config: ModelConfig, **kwargs: Any) -> ModelAdapter:
        adapter_type = self._adapter_types.get(config.provider)
        if adapter_type is None:
            raise ValueError(f"no model adapter registered for provider '{config.provider.value}'")
        return adapter_type(config=config, **kwargs)


def build_default_model_registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.register(ModelProvider.DEEPSEEK, DeepSeekAdapter)
    registry.register(ModelProvider.QWEN, QwenAdapter)
    registry.register(ModelProvider.OPENAI, OpenAIAdapter)
    return registry
