from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from app.llm.exceptions import LLMInvalidResponseError
from app.schemas.llm import LLMMessage, LLMRequest


@dataclass(frozen=True)
class StructuredOutputResult:
    output: BaseModel
    raw_output: str
    parsed_payload: dict[str, Any]
    attempts_used: int


def parse_json_output(
    *,
    raw_output: str,
    output_model: type[BaseModel],
    provider: str | None = None,
    model: str | None = None,
    operation: str | None = None,
) -> StructuredOutputResult:
    try:
        parsed_payload = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise LLMInvalidResponseError(
            f"invalid JSON output: {exc}",
            provider=provider,
            model=model,
            operation=operation,
            error_type="INVALID_JSON",
            original_exception=exc,
        ) from exc
    if not isinstance(parsed_payload, dict):
        raise LLMInvalidResponseError(
            "structured output must be a JSON object",
            provider=provider,
            model=model,
            operation=operation,
            error_type="INVALID_JSON",
        )
    try:
        parsed_model = output_model.model_validate(parsed_payload)
    except ValidationError as exc:
        raise LLMInvalidResponseError(
            f"structured output schema validation failed: {exc}",
            provider=provider,
            model=model,
            operation=operation,
            error_type="SCHEMA_VALIDATION_FAILED",
            original_exception=exc,
        ) from exc
    return StructuredOutputResult(
        output=parsed_model,
        raw_output=raw_output,
        parsed_payload=parsed_payload,
        attempts_used=1,
    )


async def invoke_structured_output(
    *,
    generate: Callable[[LLMRequest], Awaitable[str]],
    request: LLMRequest,
    output_model: type[BaseModel],
) -> StructuredOutputResult:
    last_error: LLMInvalidResponseError | None = None
    attempt_request = request
    max_attempts = max(1, request.max_validation_retries + 1)
    for attempt in range(1, max_attempts + 1):
        raw_output = await generate(attempt_request)
        try:
            result = parse_json_output(
                raw_output=raw_output,
                output_model=output_model,
                provider=str(attempt_request.metadata.get("provider") or "") or None,
                model=attempt_request.model_name,
                operation=str(attempt_request.metadata.get("operation") or "structured"),
            )
        except LLMInvalidResponseError as exc:
            last_error = exc
            if attempt >= max_attempts:
                break
            retry_messages = list(attempt_request.messages)
            retry_messages.append(
                LLMMessage(
                    role="system",
                    content=f"Previous structured output failed validation: {exc}. Return only valid JSON.",
                )
            )
            attempt_request = attempt_request.model_copy(update={"messages": retry_messages})
            continue
        return StructuredOutputResult(
            output=result.output,
            raw_output=result.raw_output,
            parsed_payload=result.parsed_payload,
            attempts_used=attempt,
        )
    if last_error is not None:
        raise last_error
    raise LLMInvalidResponseError("structured output validation failed")
