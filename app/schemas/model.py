from __future__ import annotations

import os
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ModelProvider(str, Enum):
    DEEPSEEK = "deepseek"
    QWEN = "qwen"
    OPENAI = "openai"


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    provider: ModelProvider = Field(...)
    api_key: str = Field(..., min_length=1)
    base_url: str = Field(..., min_length=1)
    model_name: str = Field(..., min_length=1)
    timeout_seconds: int = Field(default=60, gt=0)
    max_retries: int = Field(default=2, ge=0)
    organization: str | None = Field(default=None)

    @classmethod
    def from_env(cls, provider: ModelProvider | str | None = None) -> "ModelConfig":
        resolved_provider = resolve_model_provider(provider)
        return cls(
            provider=resolved_provider,
            api_key=_read_provider_env(resolved_provider, "api_key", required=True),
            base_url=_read_provider_env(resolved_provider, "base_url", required=False),
            model_name=_read_provider_env(resolved_provider, "model_name", required=False),
            timeout_seconds=int(_read_provider_env(resolved_provider, "timeout_seconds", required=False)),
            max_retries=int(_read_provider_env(resolved_provider, "max_retries", required=False)),
            organization=_read_provider_env(resolved_provider, "organization", required=False) or None,
        )


class RuntimeLLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    primary: ModelConfig = Field(...)
    fallbacks: list[ModelConfig] = Field(default_factory=list)

    @classmethod
    def from_env(cls) -> "RuntimeLLMConfig":
        primary_provider = resolve_model_provider()
        fallback_providers = _resolve_fallback_providers(primary_provider)
        return cls(
            primary=ModelConfig.from_env(primary_provider),
            fallbacks=[ModelConfig.from_env(provider) for provider in fallback_providers],
        )


def resolve_model_provider(provider: ModelProvider | str | None = None) -> ModelProvider:
    if provider is not None:
        return provider if isinstance(provider, ModelProvider) else ModelProvider(str(provider).strip().lower())

    configured_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if configured_provider:
        return ModelProvider(configured_provider)

    for candidate in available_model_providers():
        return candidate

    return ModelProvider.DEEPSEEK


def available_model_providers() -> list[ModelProvider]:
    available: list[ModelProvider] = []
    for provider in ModelProvider:
        try:
            _read_provider_env(provider, "api_key", required=True)
        except ValueError:
            continue
        available.append(provider)
    return available


def _resolve_fallback_providers(primary_provider: ModelProvider) -> list[ModelProvider]:
    explicit_fallbacks = os.getenv("LLM_FALLBACK_PROVIDERS", "").strip()
    if explicit_fallbacks:
        providers = [
            ModelProvider(item.strip().lower())
            for item in explicit_fallbacks.split(",")
            if item.strip()
        ]
    else:
        providers = [provider for provider in available_model_providers() if provider != primary_provider]

    deduplicated: list[ModelProvider] = []
    for provider in providers:
        if provider == primary_provider or provider in deduplicated:
            continue
        deduplicated.append(provider)
    return deduplicated


def _read_provider_env(provider: ModelProvider, field_name: str, *, required: bool) -> str:
    aliases = _provider_env_aliases(provider)[field_name]
    for key in aliases:
        value = os.getenv(key, "").strip()
        if value:
            return value

    if required:
        raise ValueError(f"{aliases[0]} is required to initialize the {provider.value} client")

    return _provider_defaults(provider)[field_name]


def _provider_env_aliases(provider: ModelProvider) -> dict[str, tuple[str, ...]]:
    if provider == ModelProvider.DEEPSEEK:
        return {
            "api_key": ("DEEPSEEK_API_KEY", "LLM_API_KEY"),
            "base_url": ("DEEPSEEK_BASE_URL", "LLM_BASE_URL"),
            "model_name": ("DEEPSEEK_MODEL_NAME", "LLM_MODEL_NAME"),
            "timeout_seconds": ("DEEPSEEK_TIMEOUT_SECONDS", "LLM_TIMEOUT_SECONDS"),
            "max_retries": ("DEEPSEEK_MAX_RETRIES", "LLM_MAX_RETRIES"),
            "organization": ("DEEPSEEK_ORGANIZATION", "LLM_ORGANIZATION"),
        }
    if provider == ModelProvider.QWEN:
        return {
            "api_key": ("DASHSCOPE_API_KEY", "LLM_API_KEY"),
            "base_url": ("DASHSCOPE_BASE_URL", "LLM_BASE_URL"),
            "model_name": ("QWEN_MODEL_NAME", "LLM_MODEL_NAME"),
            "timeout_seconds": ("QWEN_TIMEOUT_SECONDS", "LLM_TIMEOUT_SECONDS"),
            "max_retries": ("QWEN_MAX_RETRIES", "LLM_MAX_RETRIES"),
            "organization": ("DASHSCOPE_ORGANIZATION", "LLM_ORGANIZATION"),
        }
    return {
        "api_key": ("OPENAI_API_KEY", "LLM_API_KEY"),
        "base_url": ("OPENAI_BASE_URL", "LLM_BASE_URL"),
        "model_name": ("OPENAI_MODEL_NAME", "LLM_MODEL_NAME"),
        "timeout_seconds": ("OPENAI_TIMEOUT_SECONDS", "LLM_TIMEOUT_SECONDS"),
        "max_retries": ("OPENAI_MAX_RETRIES", "LLM_MAX_RETRIES"),
        "organization": ("OPENAI_ORGANIZATION", "LLM_ORGANIZATION"),
    }


def _provider_defaults(provider: ModelProvider) -> dict[str, str]:
    if provider == ModelProvider.DEEPSEEK:
        return {
            "base_url": "https://api.deepseek.com",
            "model_name": "deepseek-v4-flash",
            "timeout_seconds": "60",
            "max_retries": "2",
            "organization": "",
        }
    if provider == ModelProvider.QWEN:
        return {
            "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "model_name": "qwen-plus",
            "timeout_seconds": "60",
            "max_retries": "2",
            "organization": "",
        }
    return {
        "base_url": "https://api.openai.com/v1",
        "model_name": "gpt-4.1-mini",
        "timeout_seconds": "60",
        "max_retries": "2",
        "organization": "",
    }
