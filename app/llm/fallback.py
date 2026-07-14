from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from app.llm.exceptions import (
    LLMClientError,
    LLMCircuitOpenError,
    LLMProviderUnavailableError,
    LLMRateLimitError,
    LLMTimeoutError,
)

T = TypeVar("T")


@dataclass(frozen=True)
class FallbackResult(Generic[T]):
    value: T
    fallback_used: bool
    provider: str | None
    attempted_providers: list[str]


def is_fallbackable_error(error: Exception) -> bool:
    return isinstance(
        error,
        (
            LLMTimeoutError,
            LLMRateLimitError,
            LLMProviderUnavailableError,
            LLMCircuitOpenError,
        ),
    )


async def execute_with_fallback(
    *,
    providers: Sequence[str],
    operation: Callable[[str, int], Awaitable[T]],
) -> FallbackResult[T]:
    last_error: LLMClientError | None = None
    attempted_providers: list[str] = []
    for index, provider in enumerate(providers):
        attempted_providers.append(provider)
        try:
            value = await operation(provider, index)
        except LLMClientError as exc:
            if not is_fallbackable_error(exc):
                raise
            last_error = exc
            continue
        return FallbackResult(
            value=value,
            fallback_used=index > 0,
            provider=provider,
            attempted_providers=attempted_providers,
        )
    if last_error is not None:
        raise last_error
    raise LLMClientError("no llm providers configured")
