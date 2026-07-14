from __future__ import annotations

import asyncio
import hashlib
import json
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.schemas.context import TokenUsage
from app.schemas.llm import LLMFunctionCall, LLMRequest, LLMResponse, LLMResponseChunk
from app.schemas.model import ModelConfig
from app.utils import configure_runtime_logger, runtime_log, runtime_progress


class ModelAdapterError(Exception):
    pass


class ModelAdapter(ABC):
    def __init__(
        self,
        *,
        config: ModelConfig,
        http_client: httpx.AsyncClient | None = None,
        supports_streaming: bool = True,
    ) -> None:
        self.config = config
        self.supports_streaming = supports_streaming
        self._owns_http_client = http_client is None
        self.http_client = http_client or httpx.AsyncClient(timeout=None)
        self.logger = configure_runtime_logger()

    @property
    @abstractmethod
    def provider_name(self) -> str:
        raise NotImplementedError

    async def generate(self, request: LLMRequest) -> LLMResponse:
        payload = self.build_payload(request=request, stream=False)
        timeout = self.build_http_timeout(request=request)
        response: httpx.Response | None = None
        try:
            response = await self.http_client.post(
                self.chat_completions_url,
                headers=self.build_headers(request=request),
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            return self.parse_response(raw_payload=response.json(), request=request)
        except (asyncio.CancelledError, httpx.TimeoutException):
            runtime_log(
                layer=f"{self.provider_name}_adapter",
                event="timeout",
                data={
                    "action": "request_cleanup",
                    "provider": self.provider_name,
                    "timeout_config": self.describe_timeout(request=request),
                },
                logger=self.logger,
            )
            raise
        finally:
            if response is not None:
                await response.aclose()

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMResponseChunk]:
        if not self.supports_streaming:
            raise NotImplementedError(f"streaming is not supported by provider {self.provider_name}")

        payload = self.build_payload(request=request, stream=True)
        timeout = self.build_http_timeout(request=request)
        try:
            async with self.http_client.stream(
                "POST",
                self.chat_completions_url,
                headers=self.build_headers(request=request),
                json=payload,
                timeout=timeout,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    chunk = self.parse_stream_chunk(line=line, request=request)
                    if chunk is not None:
                        yield chunk
        except (asyncio.CancelledError, httpx.TimeoutException):
            runtime_log(
                layer=f"{self.provider_name}_adapter",
                event="timeout",
                data={
                    "action": "stream_cleanup",
                    "provider": self.provider_name,
                    "timeout_config": self.describe_timeout(request=request),
                },
                logger=self.logger,
            )
            raise

    async def aclose(self) -> None:
        if self._owns_http_client:
            await self.http_client.aclose()

    def should_retry(self, error: Exception) -> bool:
        if isinstance(error, httpx.HTTPStatusError):
            status_code = error.response.status_code
            return status_code == 429 or 500 <= status_code < 600
        return isinstance(error, (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError))

    @property
    def chat_completions_url(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/chat/completions"

    def build_headers(self, *, request: LLMRequest) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(
            {
                key: self._http_header_value(value)
                for key, value in self.extra_headers().items()
                if self._http_header_value(value)
            }
        )
        if request.trace_id:
            headers["X-Trace-Id"] = self._http_header_value(request.trace_id)
        if request.request_id:
            headers["X-Request-Id"] = self._http_header_value(request.request_id)
        if request.session_id:
            headers["X-Session-Id"] = self._http_header_value(request.session_id)
        return headers

    def extra_headers(self) -> dict[str, str]:
        return {}

    def _http_header_value(self, value: object, *, max_length: int = 512) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        safe = re.sub(r"[^A-Za-z0-9._~:/-]+", "-", text)
        safe = re.sub(r"-{2,}", "-", safe).strip("-")
        if safe == text and len(safe) <= max_length:
            return safe
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        prefix_limit = max(0, max_length - len(digest) - 1)
        prefix = safe[:prefix_limit].strip("-")
        return f"{prefix}-{digest}" if prefix else digest

    def build_payload(self, *, request: LLMRequest, stream: bool) -> dict[str, Any]:
        if stream and request.function_schemas and not self.supports_function_calling_with_streaming():
            raise ModelAdapterError(
                f"{self.provider_name} tools/function calling cannot be used with stream=True"
            )

        messages = request.messages or self.build_messages_from_request(request)
        payload: dict[str, Any] = {
            "model": request.model_name or self.config.model_name,
            "messages": [
                message.model_dump(exclude_none=True) if hasattr(message, "model_dump") else dict(message)
                for message in messages
            ],
            "temperature": request.temperature,
            "stream": stream,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if stream:
            payload.update(self.stream_options())

        if request.function_schemas:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": schema.name,
                        "description": schema.description,
                        "parameters": schema.parameters_schema,
                    },
                }
                for schema in request.function_schemas
            ]
        if request.tool_choice:
            payload["tool_choice"] = {
                "type": "function",
                "function": {"name": request.tool_choice},
            }
        return payload

    def supports_function_calling_with_streaming(self) -> bool:
        return True

    def stream_options(self) -> dict[str, Any]:
        return {}

    def build_http_timeout(self, *, request: LLMRequest) -> httpx.Timeout:
        total_timeout = float(request.timeout_seconds or self.config.timeout_seconds)
        connect_timeout = min(10.0, total_timeout)
        pool_timeout = min(10.0, total_timeout)
        write_timeout = min(30.0, total_timeout)
        read_timeout = total_timeout
        return httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=write_timeout,
            pool=pool_timeout,
        )

    def describe_timeout(self, *, request: LLMRequest) -> dict[str, float]:
        timeout = self.build_http_timeout(request=request)
        return {
            "connect": float(timeout.connect),
            "read": float(timeout.read),
            "write": float(timeout.write),
            "pool": float(timeout.pool),
        }

    def build_messages_from_request(self, request: LLMRequest) -> list[Any]:
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        if request.prompt:
            messages.append({"role": "user", "content": request.prompt})
        if not messages:
            raise ModelAdapterError(f"{self.provider_name} request requires messages or prompt content")
        return messages

    def parse_response(self, *, raw_payload: dict[str, Any], request: LLMRequest) -> LLMResponse:
        choices = raw_payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ModelAdapterError(f"{self.provider_name} response does not contain choices")

        first_choice = choices[0]
        message = first_choice.get("message") or {}
        reasoning_content = str(message.get("reasoning_content") or "").strip()
        if reasoning_content:
            runtime_progress(
                step=f"{self.provider_name}:模型思考",
                status="推理过程",
                detail=reasoning_content[:5000],
            )
        tool_calls = message.get("tool_calls") or []
        function_call = self.parse_function_call(tool_calls)
        content = message.get("content")
        response_text = str(content or "").strip()
        if not response_text and function_call is not None:
            response_text = json.dumps(
                {
                    "tool_name": function_call.tool_name,
                    "arguments": function_call.arguments,
                },
                ensure_ascii=True,
            )
        if not response_text:
            response_text = "{}"

        usage_payload = raw_payload.get("usage")
        usage = None
        if isinstance(usage_payload, dict):
            usage = TokenUsage.model_validate(
                {
                    "model_name": raw_payload.get("model") or request.model_name or self.config.model_name,
                    "prompt_tokens": usage_payload.get("prompt_tokens", 0),
                    "completion_tokens": usage_payload.get("completion_tokens", 0),
                    "total_tokens": usage_payload.get("total_tokens", 0),
                }
            )

        system_fingerprint = str(raw_payload.get("system_fingerprint") or "").strip() or None
        return LLMResponse(
            text=response_text,
            model_name=raw_payload.get("model") or request.model_name or self.config.model_name,
            model_version=system_fingerprint,
            finish_reason=first_choice.get("finish_reason"),
            request_id=request.request_id,
            session_id=request.session_id,
            trace_id=request.trace_id,
            prompt_name=request.prompt_name,
            prompt_version=request.prompt_version,
            function_call=function_call,
            usage=usage,
            raw_response=raw_payload,
        )

    def parse_stream_chunk(self, *, line: str, request: LLMRequest) -> LLMResponseChunk | None:
        normalized_line = line.strip()
        if not normalized_line:
            return None
        if normalized_line.startswith(":"):
            return None
        if not normalized_line.startswith("data:"):
            return None

        data = normalized_line[5:].strip()
        if not data or data == "[DONE]":
            return None

        raw_chunk = json.loads(data)
        choices = raw_chunk.get("choices") or []
        if not choices:
            return None

        delta = choices[0].get("delta") or {}
        return LLMResponseChunk(
            delta_text=str(delta.get("content") or ""),
            trace_id=request.trace_id,
            raw_chunk=raw_chunk,
        )

    def parse_function_call(self, tool_calls: list[Any]) -> LLMFunctionCall | None:
        if not tool_calls:
            return None

        first_tool_call = tool_calls[0]
        function_block = first_tool_call.get("function") if isinstance(first_tool_call, dict) else None
        if not isinstance(function_block, dict):
            raise ModelAdapterError(f"{self.provider_name} tool call does not contain a function block")

        arguments_raw = function_block.get("arguments", "{}")
        if isinstance(arguments_raw, str):
            arguments = self.parse_function_arguments(arguments_raw)
        elif isinstance(arguments_raw, dict):
            arguments = arguments_raw
        else:
            raise ModelAdapterError(
                f"{self.provider_name} function call arguments must be a JSON string or object"
            )

        return LLMFunctionCall(
            tool_name=str(function_block.get("name") or "").strip(),
            arguments=arguments if isinstance(arguments, dict) else {},
        )

    def parse_function_arguments(self, arguments_raw: str) -> Any:
        try:
            return json.loads(arguments_raw)
        except json.JSONDecodeError as exc:
            first_error = exc

        extracted = self.extract_json_object(arguments_raw)
        candidates = [extracted] if extracted != arguments_raw else []
        candidates.append(arguments_raw)
        for candidate in candidates:
            repaired = self.repair_json_object_text(candidate)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                continue
        loose_arguments = self.extract_loose_function_arguments(arguments_raw)
        if loose_arguments:
            return loose_arguments
        raise first_error

    def extract_json_object(self, text: str) -> str:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return text[start : end + 1]
        return text

    def repair_json_object_text(self, text: str) -> str:
        repaired = text.strip()
        repaired = repaired.replace("，", ",").replace("：", ":")
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        repaired = re.sub(
            r'(["}\]\d])\s+("[-A-Za-z0-9_\u4e00-\u9fff]+"\s*:)',
            r"\1,\2",
            repaired,
        )
        repaired = re.sub(
            r'(["}\]\d])\s*("[-A-Za-z0-9_\u4e00-\u9fff]+"\s*:)',
            r"\1,\2",
            repaired,
        )
        return repaired

    def extract_loose_function_arguments(self, text: str) -> dict[str, Any] | None:
        keys = ("plan_id", "status", "reason", "evidence")
        pattern = re.compile(
            r'["\']?(plan_id|status|reason|evidence)["\']?\s*[:：]\s*'
            r"(.*?)"
            r'(?=(?:[,，;；\n]\s*)?["\']?(?:plan_id|status|reason|evidence)["\']?\s*[:：]|\s*$)',
            re.DOTALL,
        )
        extracted: dict[str, str] = {}
        for match in pattern.finditer(text):
            key = match.group(1)
            value = self.clean_loose_function_argument_value(match.group(2))
            if value:
                extracted[key] = value
        if {"plan_id", "status", "reason", "evidence"}.issubset(extracted):
            return extracted
        return None

    def clean_loose_function_argument_value(self, value: str) -> str:
        cleaned = value.strip()
        cleaned = cleaned.strip(",，;；")
        cleaned = cleaned.strip()
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
            cleaned = cleaned[1:-1]
        return cleaned.strip()
