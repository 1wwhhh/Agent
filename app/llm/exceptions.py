from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMErrorContext:
    provider: str | None
    model: str | None
    operation: str | None


class LLMClientError(Exception):
    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        operation: str | None = None,
        error_type: str | None = None,
        original_exception: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.operation = operation
        self.error_type = error_type or type(self).__name__
        self.original_exception = original_exception

    @property
    def context(self) -> LLMErrorContext:
        return LLMErrorContext(provider=self.provider, model=self.model, operation=self.operation)


class LLMTimeoutError(LLMClientError):
    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(message, error_type=kwargs.pop("error_type", "LLM_TIMEOUT"), **kwargs)


class LLMRateLimitError(LLMClientError):
    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(message, error_type=kwargs.pop("error_type", "LLM_RATE_LIMIT"), **kwargs)


class LLMProviderUnavailableError(LLMClientError):
    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(message, error_type=kwargs.pop("error_type", "LLM_PROVIDER_UNAVAILABLE"), **kwargs)


class LLMInvalidResponseError(LLMClientError):
    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(message, error_type=kwargs.pop("error_type", "LLM_INVALID_RESPONSE"), **kwargs)


class LLMAuthenticationError(LLMClientError):
    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(message, error_type=kwargs.pop("error_type", "LLM_AUTHENTICATION"), **kwargs)


class LLMCircuitOpenError(LLMClientError):
    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(message, error_type=kwargs.pop("error_type", "LLM_CIRCUIT_OPEN"), **kwargs)


class CircuitBreakerOpenError(LLMCircuitOpenError):
    pass


class LLMProviderTimeoutError(LLMTimeoutError):
    pass


class LLMProviderRetryableError(LLMProviderUnavailableError):
    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(message, error_type="LLM_PROVIDER_RETRYABLE", **kwargs)
