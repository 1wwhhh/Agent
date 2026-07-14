from __future__ import annotations

from pydantic import Field

from app.schemas.model import ModelConfig, ModelProvider


class QwenConfig(ModelConfig):
    provider: ModelProvider = Field(default=ModelProvider.QWEN)
    base_url: str = Field(default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1", min_length=1)
    model_name: str = Field(default="qwen-plus", min_length=1)

    @classmethod
    def from_env(cls) -> "QwenConfig":
        return cls.model_validate(ModelConfig.from_env(ModelProvider.QWEN).model_dump())
