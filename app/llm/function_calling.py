from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.llm.exceptions import LLMInvalidResponseError
from app.schemas.llm import LLMFunctionCall, LLMFunctionSchema, LLMMessage, LLMRequest, LLMResponse


class FunctionCallingAdapterError(LLMInvalidResponseError):
    pass


class FunctionCallingResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", validate_assignment=True)

    output: Any = Field(...)
    function_call: LLMFunctionCall = Field(...)
    response: LLMResponse = Field(...)
    attempts_used: int = Field(..., ge=1)


StructuredLLMResult = FunctionCallingResult


class FunctionCallingAdapter:
    async def invoke_structured(
        self,
        *,
        client,
        request: LLMRequest,
        function_schema: LLMFunctionSchema,
        output_model: type[BaseModel],
    ) -> FunctionCallingResult:
        attempt_request = request.model_copy(
            update={
                "function_schemas": [function_schema],
                "tool_choice": function_schema.name,
                "response_schema_name": function_schema.schema_name,
                "response_schema_version": function_schema.schema_version,
                "metadata": {
                    **request.metadata,
                    "operation": request.metadata.get("operation", "function_calling"),
                },
            }
        )
        max_attempts = max(1, attempt_request.max_validation_retries + 1)
        last_error: FunctionCallingAdapterError | None = None

        for attempt in range(1, max_attempts + 1):
            response = await client.generate(attempt_request)
            try:
                function_call = self.extract_function_call(response=response, function_schema=function_schema)
                parsed_output = output_model.model_validate(function_call.arguments)
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                last_error = self._build_error(
                    exc=exc,
                    response=response,
                    function_schema=function_schema,
                    operation=str(attempt_request.metadata.get("operation") or "function_calling"),
                )
                if attempt >= max_attempts:
                    break
                attempt_request = self._build_retry_request(
                    request=attempt_request,
                    error_message=str(last_error),
                    function_schema=function_schema,
                )
                continue
            return FunctionCallingResult(
                output=parsed_output,
                function_call=function_call,
                response=response,
                attempts_used=attempt,
            )

        if last_error is not None:
            raise last_error
        raise FunctionCallingAdapterError("function calling validation failed")

    def extract_function_call(self, *, response: LLMResponse, function_schema: LLMFunctionSchema) -> LLMFunctionCall:
        if response.function_call is not None:
            function_call = response.function_call
        else:
            payload = json.loads(response.text)
            if not isinstance(payload, dict):
                raise ValueError("invalid function call payload")
            function_call = LLMFunctionCall.model_validate(payload)

        if function_call.tool_name != function_schema.name:
            raise ValueError(f"unsupported function '{function_call.tool_name}'")
        if not isinstance(function_call.arguments, dict):
            raise ValueError("missing arguments")

        return function_call.model_copy(
            update={
                "schema_name": function_call.schema_name or function_schema.schema_name,
                "schema_version": function_call.schema_version or function_schema.schema_version,
            }
        )

    def _build_retry_request(
        self,
        *,
        request: LLMRequest,
        error_message: str,
        function_schema: LLMFunctionSchema,
    ) -> LLMRequest:
        retry_messages = list(request.messages)
        retry_messages.append(
            LLMMessage(
                role="system",
                content=(
                    "Previous function call output was invalid. "
                    f"Return only the supported function '{function_schema.name}' with valid arguments. "
                    f"Error: {error_message}"
                ),
            )
        )
        return request.model_copy(update={"messages": retry_messages})

    def _build_error(
        self,
        *,
        exc: Exception,
        response: LLMResponse,
        function_schema: LLMFunctionSchema,
        operation: str,
    ) -> FunctionCallingAdapterError:
        error_type = "INVALID_FUNCTION_CALL"
        if isinstance(exc, ValidationError):
            validation_errors = exc.errors()
            if any(str(item.get("type", "")).endswith("missing") for item in validation_errors):
                error_type = "MISSING_ARGUMENTS"
            else:
                error_type = "INVALID_FUNCTION_ARGUMENTS"
        elif "unsupported function" in str(exc):
            error_type = "UNSUPPORTED_FUNCTION"
        elif "missing arguments" in str(exc):
            error_type = "MISSING_ARGUMENTS"
        return FunctionCallingAdapterError(
            str(exc),
            provider=str(response.raw_response.get("selected_provider") or "") or None,
            model=response.model_name,
            operation=operation,
            error_type=error_type,
            original_exception=exc,
        )
