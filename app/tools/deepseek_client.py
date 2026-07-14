from __future__ import annotations

import httpx

from app.adapters import DeepSeekAdapter, ModelAdapterError
from app.schemas.deepseek import DeepSeekConfig
from app.tools.llm_client import LLMClient

DeepSeekClientError = ModelAdapterError


class DeepSeekLLMClient(LLMClient):
    def __init__(
        self,
        *,
        config: DeepSeekConfig,
        http_client: httpx.AsyncClient | None = None,
        supports_streaming: bool = True,
    ) -> None:
        super().__init__(
            adapters=[
                DeepSeekAdapter(
                    config=config,
                    http_client=http_client,
                    supports_streaming=supports_streaming,
                )
            ],
            timeout_seconds=config.timeout_seconds,
            model_name=config.model_name,
            model_version=None,
            supports_streaming=supports_streaming,
            max_retries=config.max_retries,
        )
