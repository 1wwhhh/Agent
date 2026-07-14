from __future__ import annotations

import httpx

from app.adapters import ModelAdapterError, QwenAdapter
from app.schemas.qwen import QwenConfig
from app.tools.llm_client import LLMClient

QwenClientError = ModelAdapterError


class QwenLLMClient(LLMClient):
    def __init__(
        self,
        *,
        config: QwenConfig,
        http_client: httpx.AsyncClient | None = None,
        supports_streaming: bool = True,
    ) -> None:
        super().__init__(
            adapters=[
                QwenAdapter(
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
