from __future__ import annotations

from dataclasses import dataclass

from app.llm.exceptions import (
    LLMAuthenticationError,
    LLMCircuitOpenError,
    LLMClientError,
    LLMInvalidResponseError,
    LLMProviderUnavailableError,
    LLMRateLimitError,
    LLMTimeoutError,
)


@dataclass(frozen=True)
class RetryDecision:
    retryable: bool
    should_retry: bool
    retry_count: int
    backoff_seconds: float
    reason: str


class LLMRetryPolicy:
    def __init__(self, *, max_retry: int = 0, backoff_base_seconds: float = 0.5) -> None:
        self.max_retry = max(0, max_retry)
        self.backoff_base_seconds = max(0.0, backoff_base_seconds)

    def classify(self, error: Exception, *, retry_count: int) -> RetryDecision:
        retryable = self.is_retryable(error)
        should_retry = retryable and retry_count < self.max_retry
        return RetryDecision(
            retryable=retryable,
            should_retry=should_retry,
            retry_count=retry_count,
            backoff_seconds=self.backoff_seconds(retry_count) if should_retry else 0.0,
            reason=self._reason(error=error, retryable=retryable, should_retry=should_retry),
        )

    def is_retryable(self, error: Exception) -> bool:
        return isinstance(error, (LLMTimeoutError, LLMRateLimitError, LLMProviderUnavailableError)) and not isinstance(
            error,
            (LLMAuthenticationError, LLMInvalidResponseError, LLMCircuitOpenError),
        )

    def backoff_seconds(self, retry_count: int) -> float:
        if self.backoff_base_seconds <= 0:
            return 0.0
        return min(5.0, self.backoff_base_seconds * (2**max(0, retry_count)))

    def _reason(self, *, error: Exception, retryable: bool, should_retry: bool) -> str:
        if not retryable:
            if isinstance(error, LLMAuthenticationError):
                return "authentication errors are not retryable"
            if isinstance(error, LLMInvalidResponseError):
                return "invalid provider responses are not retryable"
            if isinstance(error, LLMCircuitOpenError):
                return "open circuit falls through to provider fallback only"
            if isinstance(error, LLMClientError):
                return error.error_type
            return type(error).__name__
        if should_retry:
            return "retryable llm failure"
        return "retry budget exhausted"
