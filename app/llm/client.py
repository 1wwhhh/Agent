from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import httpx

from app.llm.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from app.llm.exceptions import (
    CircuitBreakerOpenError,
    LLMAuthenticationError,
    LLMClientError,
    LLMInvalidResponseError,
    LLMProviderRetryableError,
    LLMProviderTimeoutError,
    LLMProviderUnavailableError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from app.llm.fallback import FallbackResult, execute_with_fallback
from app.llm.retry import LLMRetryPolicy
from app.schemas.llm import LLMRequest, LLMResponse, LLMResponseChunk
from app.utils import configure_runtime_logger, runtime_log, runtime_progress

if TYPE_CHECKING:
    from app.adapters.base import ModelAdapter

FAIL_FAST_TIMEOUT_MARKER = "provider_timeout_fail_fast"
RETRYABLE_ERROR_MARKER = "provider_retryable_error"

_PROVIDER_SEMAPHORES: dict[str, asyncio.Semaphore] = {
    "deepseek": asyncio.Semaphore(5),
    "qwen": asyncio.Semaphore(10),
}
_DEFAULT_PROVIDER_SEMAPHORE = asyncio.Semaphore(10)


def _get_provider_semaphore(provider_name: str) -> asyncio.Semaphore:
    return _PROVIDER_SEMAPHORES.get(provider_name, _DEFAULT_PROVIDER_SEMAPHORE)


@dataclass
class ProviderExecutionState:
    adapter: "ModelAdapter"
    circuit_breaker: CircuitBreaker

    @property
    def provider_name(self) -> str:
        return self.adapter.provider_name


class LLMClient:
    def __init__(
        self,
        *,
        timeout_seconds: int = 60,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
        model_name: str | None = None,
        model_version: str | None = None,
        supports_streaming: bool = False,
        max_retries: int = 0,
        retry_backoff_base_seconds: float = 0.5,
        adapters: list["ModelAdapter"] | None = None,
        llm_call_recorder: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig()
        self.model_name = model_name
        self.model_version = model_version
        self.supports_streaming = supports_streaming or any(item.supports_streaming for item in (adapters or []))
        self.max_retries = max(0, max_retries)
        self.retry_backoff_base_seconds = max(0.0, retry_backoff_base_seconds)
        self.retry_policy = LLMRetryPolicy(max_retry=self.max_retries, backoff_base_seconds=self.retry_backoff_base_seconds)
        self._provider_states = [
            ProviderExecutionState(adapter=item, circuit_breaker=CircuitBreaker(self.circuit_breaker_config))
            for item in (adapters or [])
        ]
        self._standalone_breaker = CircuitBreaker(self.circuit_breaker_config)
        self.llm_call_recorder = llm_call_recorder
        self.logger = configure_runtime_logger()

    def set_llm_call_recorder(self, llm_call_recorder: Callable[[dict[str, object]], None] | None) -> None:
        self.llm_call_recorder = llm_call_recorder

    async def generate(self, request: LLMRequest) -> LLMResponse:
        started_at = time.perf_counter()
        operation = self._resolve_operation(request)
        try:
            if self._provider_states:
                fallback_result = await self._generate_with_providers(request=request, operation=operation)
                response = self._finalize_provider_response(
                    response=fallback_result.value,
                    attempted_providers=fallback_result.attempted_providers,
                    selected_provider=fallback_result.provider,
                )
                response = self._finalize_response(response=response, request=request)
                fallback_used = fallback_result.fallback_used
                retry_count = int(response.raw_response.get("retry_count", 0))
                provider = fallback_result.provider
            else:
                response, retry_count = await self._execute_standalone_generate(request=request, operation=operation)
                response = self._finalize_response(response=response, request=request)
                fallback_used = False
                provider = None
        except Exception as exc:
            normalized = self._normalize_error(
                error=exc,
                provider=None,
                model=request.model_name or self.model_name,
                operation=operation,
                timeout_seconds=request.timeout_seconds or self.timeout_seconds,
            )
            self._record_call(
                request=request,
                provider=getattr(normalized, "provider", None),
                model=getattr(normalized, "model", None) or request.model_name or self.model_name,
                operation=operation,
                latency_ms=(time.perf_counter() - started_at) * 1000,
                success=False,
                error_type=getattr(normalized, "error_type", type(normalized).__name__),
                fallback_used=False,
                retry_count=0,
            )
            raise normalized
        self._record_call(
            request=request,
            provider=provider or str(response.raw_response.get("selected_provider") or "") or None,
            model=response.model_name or request.model_name or self.model_name,
            operation=operation,
            latency_ms=(time.perf_counter() - started_at) * 1000,
            success=True,
            error_type=None,
            fallback_used=fallback_used,
            retry_count=retry_count,
        )
        return response

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMResponseChunk]:
        if not self.supports_streaming:
            raise NotImplementedError("streaming is not supported by this LLM client")
        if self._provider_states:
            async for chunk in self._stream_with_providers(request):
                yield chunk
            return
        self._standalone_breaker.before_request()
        async for chunk in self._stream(request):
            yield chunk

    async def _generate_with_providers(self, *, request: LLMRequest, operation: str) -> FallbackResult[LLMResponse]:
        provider_map = {state.provider_name: state for state in self._provider_states}
        return await execute_with_fallback(
            providers=[state.provider_name for state in self._provider_states],
            operation=lambda provider_name, index: self._execute_provider_generate(
                provider_state=provider_map[provider_name],
                request=request,
                operation=operation,
                provider_index=index,
            ),
        )

    async def _execute_provider_generate(
        self,
        *,
        provider_state: ProviderExecutionState,
        request: LLMRequest,
        operation: str,
        provider_index: int,
    ) -> LLMResponse:
        timeout_seconds = request.timeout_seconds or provider_state.adapter.config.timeout_seconds
        provider_state.circuit_breaker.before_request()
        runtime_progress(
            step="llm",
            status="request",
            detail="模型请求已发送",
        )

        retry_limit = 0 if self._should_disable_retries(request) else provider_state.adapter.config.max_retries
        retry_policy = LLMRetryPolicy(
            max_retry=retry_limit,
            backoff_base_seconds=self.retry_backoff_base_seconds,
        )
        semaphore = _get_provider_semaphore(provider_state.provider_name)

        async with semaphore:
            for attempt in range(retry_limit + 1):
                try:
                    response = await asyncio.wait_for(provider_state.adapter.generate(request), timeout=timeout_seconds)
                except Exception as exc:
                    normalized = self._normalize_error(
                        error=exc,
                        provider=provider_state.provider_name,
                        model=request.model_name or provider_state.adapter.config.model_name,
                        operation=operation,
                        timeout_seconds=timeout_seconds,
                    )
                    compatibility_error = self._compatibility_error(request=request, error=normalized)
                    decision = retry_policy.classify(compatibility_error, retry_count=attempt)
                    if decision.should_retry:
                        await asyncio.sleep(decision.backoff_seconds)
                        continue
                    provider_state.circuit_breaker.record_failure()
                    raise compatibility_error

                provider_state.circuit_breaker.record_success()
                raw_response = dict(response.raw_response)
                raw_response["retry_count"] = attempt
                raw_response["provider_index"] = provider_index
                return response.model_copy(
                    update={
                        "model_name": response.model_name or provider_state.adapter.config.model_name,
                        "raw_response": raw_response,
                    }
                )
        raise LLMProviderUnavailableError(
            f"provider '{provider_state.provider_name}' exhausted without response",
            provider=provider_state.provider_name,
            model=request.model_name or provider_state.adapter.config.model_name,
            operation=operation,
        )

    async def _execute_standalone_generate(self, *, request: LLMRequest, operation: str) -> tuple[LLMResponse, int]:
        timeout_seconds = request.timeout_seconds or self.timeout_seconds
        self._standalone_breaker.before_request()
        for attempt in range(self.max_retries + 1):
            try:
                response = await asyncio.wait_for(self._generate(request), timeout=timeout_seconds)
            except Exception as exc:
                normalized = self._normalize_error(
                    error=exc,
                    provider=None,
                    model=request.model_name or self.model_name,
                    operation=operation,
                    timeout_seconds=timeout_seconds,
                )
                decision = self.retry_policy.classify(normalized, retry_count=attempt)
                if decision.should_retry:
                    await asyncio.sleep(decision.backoff_seconds)
                    continue
                self._standalone_breaker.record_failure()
                raise normalized
            self._standalone_breaker.record_success()
            return response, attempt
        raise LLMClientError("llm request failed after retries")

    async def _stream_with_providers(self, request: LLMRequest) -> AsyncIterator[LLMResponseChunk]:
        last_error: Exception | None = None
        for provider_state in self._provider_states:
            if not provider_state.adapter.supports_streaming:
                continue
            try:
                provider_state.circuit_breaker.before_request()
            except CircuitBreakerOpenError as exc:
                last_error = exc
                continue
            async for chunk in provider_state.adapter.stream(request):
                yield chunk.model_copy(update={"trace_id": chunk.trace_id or request.trace_id})
            provider_state.circuit_breaker.record_success()
            return
        if last_error is not None:
            raise last_error
        raise LLMClientError("all streaming llm providers failed")

    def _normalize_error(
        self,
        *,
        error: Exception,
        provider: str | None,
        model: str | None,
        operation: str,
        timeout_seconds: int,
    ) -> Exception:
        if isinstance(error, LLMClientError):
            return error
        if isinstance(error, (asyncio.TimeoutError, httpx.TimeoutException)):
            return LLMTimeoutError(
                f"provider '{provider or 'default'}' timed out after {timeout_seconds}s",
                provider=provider,
                model=model,
                operation=operation,
                original_exception=error,
            )
        if isinstance(error, httpx.HTTPStatusError):
            status_code = error.response.status_code
            if status_code in {401, 403}:
                return LLMAuthenticationError(
                    f"provider '{provider or 'default'}' authentication failed ({status_code})",
                    provider=provider,
                    model=model,
                    operation=operation,
                    original_exception=error,
                )
            if status_code == 429:
                return LLMRateLimitError(
                    f"provider '{provider or 'default'}' rate limited the request",
                    provider=provider,
                    model=model,
                    operation=operation,
                    original_exception=error,
                )
            if 500 <= status_code < 600:
                return LLMProviderUnavailableError(
                    f"provider '{provider or 'default'}' is unavailable ({status_code})",
                    provider=provider,
                    model=model,
                    operation=operation,
                    original_exception=error,
                )
        if isinstance(error, (httpx.ConnectError, httpx.RemoteProtocolError, httpx.NetworkError)):
            return LLMProviderUnavailableError(
                f"provider '{provider or 'default'}' is unavailable ({error})",
                provider=provider,
                model=model,
                operation=operation,
                original_exception=error,
            )
        if isinstance(error, ValueError):
            return LLMInvalidResponseError(
                str(error),
                provider=provider,
                model=model,
                operation=operation,
                original_exception=error,
            )
        return LLMProviderUnavailableError(
            str(error),
            provider=provider,
            model=model,
            operation=operation,
            original_exception=error,
        )

    def _compatibility_error(self, *, request: LLMRequest, error: Exception) -> Exception:
        if not isinstance(error, LLMClientError):
            return error
        if self._should_disable_retries(request):
            context = error.context.__dict__
            if isinstance(error, LLMTimeoutError) and not isinstance(error, LLMProviderTimeoutError):
                return LLMProviderTimeoutError(f"{FAIL_FAST_TIMEOUT_MARKER}: {error}", **context)
            if isinstance(error, (LLMRateLimitError, LLMProviderUnavailableError)) and not isinstance(
                error,
                LLMProviderRetryableError,
            ):
                return LLMProviderRetryableError(f"{RETRYABLE_ERROR_MARKER}: {error}", **context)
        return error

    def _resolve_operation(self, request: LLMRequest) -> str:
        metadata = request.metadata
        operation = metadata.get("operation")
        if isinstance(operation, str) and operation.strip():
            return operation.strip()
        component = str(metadata.get("component") or "").strip().lower()
        if component == "planner":
            return "planner"
        if component == "parser_repair":
            return "repair"
        if metadata.get("tool_name"):
            return "tool"
        return "structured"

    def _record_call(
        self,
        *,
        request: LLMRequest,
        provider: str | None,
        model: str | None,
        operation: str,
        latency_ms: float,
        success: bool,
        error_type: str | None,
        fallback_used: bool,
        retry_count: int,
    ) -> None:
        if self.llm_call_recorder is None:
            return
        record = {
            "request_id": request.request_id,
            "provider": provider,
            "model": model,
            "operation": operation,
            "latency_ms": int(max(0.0, latency_ms)),
            "success": success,
            "error_type": error_type,
            "fallback_used": fallback_used,
            "retry_count": retry_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self.llm_call_recorder(record)
        except Exception as exc:
            runtime_log(
                layer="llm_client",
                event="error",
                data={
                    "action": "llm_call_recorder_failed",
                    "request_id": request.request_id,
                    "operation": operation,
                    "exception_type": type(exc).__name__,
                    "error": str(exc),
                },
                level=logging.DEBUG,
                logger=self.logger,
            )

    def _should_disable_retries(self, request: LLMRequest) -> bool:
        return bool(request.metadata.get("fail_fast_timeout") or request.metadata.get("disable_provider_retries"))

    def _finalize_provider_response(
        self,
        *,
        response: LLMResponse,
        attempted_providers: list[str],
        selected_provider: str | None,
    ) -> LLMResponse:
        raw_response = dict(response.raw_response)
        raw_response["selected_provider"] = selected_provider
        raw_response["attempted_providers"] = list(attempted_providers)
        return response.model_copy(update={"raw_response": raw_response})

    def _finalize_response(self, *, response: LLMResponse, request: LLMRequest) -> LLMResponse:
        return response.model_copy(
            update={
                "request_id": response.request_id or request.request_id,
                "session_id": response.session_id or request.session_id,
                "trace_id": response.trace_id or request.trace_id,
                "prompt_name": response.prompt_name or request.prompt_name,
                "prompt_version": response.prompt_version or request.prompt_version,
                "model_name": response.model_name or request.model_name or self.model_name,
                "model_version": response.model_version or self.model_version,
            }
        )

    async def _generate(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError("adapter-backed clients should not call _generate directly")

    async def _stream(self, request: LLMRequest) -> AsyncIterator[LLMResponseChunk]:
        raise NotImplementedError("streaming is not supported by this LLM client")

    async def aclose(self) -> None:
        for provider_state in self._provider_states:
            await provider_state.adapter.aclose()
