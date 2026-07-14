from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator

from app.schemas.llm import LLMRequest, LLMResponse, LLMResponseChunk
from app.tools.llm_client import CircuitBreakerOpenError, LLMClient, LLMClientError
from app.utils import runtime_log, runtime_progress


class FailoverLLMClient(LLMClient):
    def __init__(
        self,
        *,
        clients: list[LLMClient],
        provider_names: list[str] | None = None,
    ) -> None:
        if not clients:
            raise ValueError("FailoverLLMClient requires at least one provider client")
        primary = clients[0]
        super().__init__(
            timeout_seconds=primary.timeout_seconds,
            circuit_breaker_config=primary.circuit_breaker_config,
            model_name=primary.model_name,
            model_version=primary.model_version,
            supports_streaming=any(client.supports_streaming for client in clients),
            max_retries=0,
        )
        self.clients = clients
        self.provider_names = provider_names or [client.model_name or f"provider_{index}" for index, client in enumerate(clients)]

    async def _generate(self, request: LLMRequest) -> LLMResponse:
        last_error: Exception | None = None
        attempted_providers: list[str] = []

        for index, (provider_name, client) in enumerate(zip(self.provider_names, self.clients)):
            attempted_providers.append(provider_name)
            attempt_started_at = time.perf_counter()
            runtime_log(
                layer="llm_client",
                event="execute",
                data={
                    "action": "failover_attempt",
                    "provider": provider_name,
                    "attempt": index + 1,
                    "total_providers": len(self.clients),
                },
                logger=self.logger,
            )
            runtime_progress(
                step="llm",
                status="failover 尝试",
                detail=f"提供商={provider_name} ({index + 1}/{len(self.clients)})",
            )
            try:
                response = await client.generate(request)
                _duration_ms = (time.perf_counter() - attempt_started_at) * 1000
                raw_response = dict(response.raw_response)
                raw_response["selected_provider"] = provider_name
                raw_response["attempted_providers"] = list(attempted_providers)
                runtime_log(
                    layer="llm_client",
                    event="success",
                    data={
                        "action": "failover_success",
                        "provider": provider_name,
                        "attempt": index + 1,
                        "attempted_providers": list(attempted_providers),
                    },
                    latency_ms=_duration_ms,
                    logger=self.logger,
                )
                return response.model_copy(update={"raw_response": raw_response})
            except (CircuitBreakerOpenError, LLMClientError) as exc:
                _duration_ms = (time.perf_counter() - attempt_started_at) * 1000
                last_error = exc
                runtime_log(
                    layer="llm_client",
                    event="error",
                    data={
                        "action": "failover_provider_failed",
                        "provider": provider_name,
                        "attempt": index + 1,
                        "error": str(exc),
                        "will_try_next": index + 1 < len(self.clients),
                    },
                    latency_ms=_duration_ms,
                    level=logging.WARNING,
                    logger=self.logger,
                )
                continue

        runtime_log(
            layer="llm_client",
            event="error",
            data={
                "action": "failover_all_failed",
                "attempted_providers": list(attempted_providers),
                "error": str(last_error),
            },
            level=logging.ERROR,
            logger=self.logger,
        )
        raise LLMClientError(f"all llm providers failed: {last_error}")

    async def _stream(self, request: LLMRequest) -> AsyncIterator[LLMResponseChunk]:
        last_error: Exception | None = None
        attempted_providers: list[str] = []

        for index, (provider_name, client) in enumerate(zip(self.provider_names, self.clients)):
            if not client.supports_streaming:
                runtime_log(
                    layer="llm_client",
                    event="execute",
                    data={
                        "action": "failover_stream_skip",
                        "provider": provider_name,
                        "reason": "streaming_not_supported",
                    },
                    logger=self.logger,
                )
                continue

            attempted_providers.append(provider_name)
            attempt_started_at = time.perf_counter()
            runtime_log(
                layer="llm_client",
                event="execute",
                data={
                    "action": "failover_stream_attempt",
                    "provider": provider_name,
                    "attempt": len(attempted_providers),
                },
                logger=self.logger,
            )
            try:
                async for chunk in client.stream(request):
                    yield chunk
                _duration_ms = (time.perf_counter() - attempt_started_at) * 1000
                runtime_log(
                    layer="llm_client",
                    event="success",
                    data={
                        "action": "failover_stream_success",
                        "provider": provider_name,
                        "attempted_providers": list(attempted_providers),
                    },
                    latency_ms=_duration_ms,
                    logger=self.logger,
                )
                return
            except (CircuitBreakerOpenError, LLMClientError) as exc:
                _duration_ms = (time.perf_counter() - attempt_started_at) * 1000
                last_error = exc
                runtime_log(
                    layer="llm_client",
                    event="error",
                    data={
                        "action": "failover_stream_provider_failed",
                        "provider": provider_name,
                        "error": str(exc),
                        "will_try_next": index + 1 < len(self.clients),
                    },
                    latency_ms=_duration_ms,
                    level=logging.WARNING,
                    logger=self.logger,
                )
                continue

        runtime_log(
            layer="llm_client",
            event="error",
            data={
                "action": "failover_stream_all_failed",
                "attempted_providers": list(attempted_providers),
                "error": str(last_error),
            },
            level=logging.ERROR,
            logger=self.logger,
        )
        raise LLMClientError(f"all streaming llm providers failed: {last_error}")

    async def aclose(self) -> None:
        for client in self.clients:
            await client.aclose()
