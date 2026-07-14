from __future__ import annotations

from pydantic import Field

from app.schemas.model import ModelConfig, ModelProvider


class DeepSeekConfig(ModelConfig):
    provider: ModelProvider = Field(default=ModelProvider.DEEPSEEK)
    base_url: str = Field(default="https://api.deepseek.com", min_length=1)
    model_name: str = Field(default="deepseek-v4-flash", min_length=1)

    @classmethod
    def from_env(cls) -> "DeepSeekConfig":
        return cls.model_validate(ModelConfig.from_env(ModelProvider.DEEPSEEK).model_dump())
