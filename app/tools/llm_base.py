from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import os
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.prompts import PromptRegistry, RenderedPrompt, build_default_prompt_registry
from app.schemas.context import ContextStore
from app.schemas.llm import LLMFunctionSchema, LLMMessage, LLMRequest
from app.schemas.tool import ToolResult
from app.tools.base import BaseTool
from app.tools.function_calling import FunctionCallingAdapter
from app.tools.llm_client import LLMClient
from app.utils import runtime_log, runtime_progress


class BaseLLMTool(BaseTool, ABC):
    """基于模型供应商实现的 LLM 工具公共基类。"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    client: Any = Field(..., description="Injected async LLM client implementation.")
    prompt_registry: PromptRegistry = Field(default_factory=build_default_prompt_registry)
    function_adapter: FunctionCallingAdapter = Field(default_factory=FunctionCallingAdapter)
    default_model_name: str | None = Field(default=None, description="Default model name when payload omits one.")
    default_temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="Default sampling temperature.")
    default_max_tokens: int | None = Field(default=None, gt=0, description="Default completion token budget.")
    default_timeout_seconds: int | None = Field(default=None, gt=0, description="Optional per-request timeout override.")
    prompt_name: str = Field(..., min_length=1, description="Registered prompt template name.")
    prompt_version: str = Field(default="v1", min_length=1, description="Prompt template version.")
    response_schema_version: str = Field(default="v1", min_length=1, description="Structured response schema version.")
    function_name: str = Field(..., min_length=1, description="Function name returned by the model.")
    function_description: str = Field(..., min_length=1, description="Function description exposed to the model.")
    max_schema_retries: int = Field(default=1, ge=0, description="Retries when structured output validation fails.")

    @model_validator(mode="after")
    def validate_client(self) -> "BaseLLMTool":
        if not isinstance(self.client, LLMClient):
            raise TypeError("client must implement the LLMClient abstraction")
        return self

    async def _arun(self, payload: dict[str, Any], context: ContextStore | None = None) -> ToolResult:
        try:
            resolved_payload, injection_results = self.resolve_payload_templates(payload=payload, context=context)
            rendered_prompt = self.render_prompt(payload=resolved_payload, context=context)
            self.log_prompt_render(
                payload=resolved_payload,
                rendered_prompt=rendered_prompt,
                context=context,
                injection_results=injection_results,
            )
            self._log_visible_llm_input(rendered_prompt=rendered_prompt, context=context)
            request = self.build_request(rendered_prompt=rendered_prompt, payload=resolved_payload, context=context)
            structured_result = await self.function_adapter.invoke_structured(
                client=self.client,
                request=request,
                function_schema=self.build_function_schema(),
                output_model=self.get_output_model(),
            )
            output_model = self.sanitize_output_model(
                structured_result.output,
                payload=resolved_payload,
                context=context,
            )
            _output_dump = output_model.model_dump(mode="json")
            _text = str(_output_dump.get("text") or _output_dump.get("summary") or _output_dump)
            runtime_progress(
                step="模型输出" if self.name == "text_generate_tool" else f"{self.name}:输出内容",
                status="模型生成",
                detail=_text[:5000],
            )
            return self.build_success_result(
                request=request,
                rendered_prompt=rendered_prompt,
                output_model=output_model,
                raw_response=structured_result.response.raw_response,
                model_name=structured_result.response.model_name,
                model_version=structured_result.response.model_version,
                finish_reason=structured_result.response.finish_reason,
                usage=structured_result.response.usage.model_dump(exclude_none=True)
                if structured_result.response.usage is not None
                else None,
                attempts_used=structured_result.attempts_used,
                function_schema=self.build_function_schema(),
                trace_id=structured_result.response.trace_id,
            )
        except Exception as exc:
            return self.build_result(
                success=False,
                error=str(exc),
                metadata={"payload": payload},
            )

    @abstractmethod
    def build_prompt_variables(self, payload: dict[str, Any], context: ContextStore | None = None) -> dict[str, Any]:
        """把原始任务载荷转换为 Prompt 模板变量。"""

    @abstractmethod
    def get_output_model(self) -> type[BaseModel]:
        """返回用于校验结构化 LLM 输出的 Pydantic 模型。"""

    def sanitize_output_model(
        self,
        output_model: BaseModel,
        *,
        payload: dict[str, Any],
        context: ContextStore | None,
    ) -> BaseModel:
        return output_model

    def render_prompt(self, payload: dict[str, Any], context: ContextStore | None = None) -> RenderedPrompt:
        return self.prompt_registry.render(
            self.prompt_name,
            version=self.prompt_version,
            variables=self.build_prompt_variables(payload=payload, context=context),
        )

    def resolve_payload_templates(
        self,
        *,
        payload: dict[str, Any],
        context: ContextStore | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if context is None:
            return dict(payload), []

        injection_results: list[dict[str, Any]] = []
        resolved_payload = self._resolve_template_value(payload, context=context, injection_results=injection_results)
        return resolved_payload if isinstance(resolved_payload, dict) else dict(payload), injection_results

    def _log_visible_llm_input(
        self,
        *,
        rendered_prompt: RenderedPrompt,
        context: ContextStore | None,
    ) -> None:
        if self.name == "text_generate_tool" and not os.getenv("LLM_VISIBLE_INPUT_MAX_CHARS", "").strip():
            user_input = context.runtime.user_input.strip() if context is not None else ""
            detail = f"用户输入={user_input[:500]} | 模型提示与上下文已隐藏"
            runtime_progress(
                step="模型输入",
                status="发送给模型",
                detail=detail,
                request_id=context.runtime.request_id if context is not None else None,
                session_id=context.runtime.session_id if context is not None else None,
            )
            return

        user_prompt = rendered_prompt.user_prompt
        max_chars = self._visible_llm_input_max_chars()
        detail = user_prompt if max_chars <= 0 else user_prompt[:max_chars]
        if max_chars > 0 and len(user_prompt) > max_chars:
            detail = f"{detail}... [truncated {len(user_prompt) - max_chars} chars]"
        runtime_progress(
            step="模型输入" if self.name == "text_generate_tool" else f"{self.name}:输入内容",
            status="发送给模型",
            detail=detail,
            request_id=context.runtime.request_id if context is not None else None,
            session_id=context.runtime.session_id if context is not None else None,
        )

    def _visible_llm_input_max_chars(self) -> int:
        env_value = os.getenv("LLM_VISIBLE_INPUT_MAX_CHARS", "").strip()
        if env_value:
            try:
                return int(env_value)
            except ValueError:
                pass
        return 0 if self.name == "text_generate_tool" else 5000

    def log_prompt_render(
        self,
        *,
        payload: dict[str, Any],
        rendered_prompt: RenderedPrompt,
        context: ContextStore | None,
        injection_results: list[dict[str, Any]],
    ) -> None:
        template = self.prompt_registry.get(self.prompt_name, version=self.prompt_version)
        runtime_log(
            layer=self.name,
            event="execute",
            data={
                "template_before": {
                    "system_template": template.system_template,
                    "user_template": template.user_template,
                },
                "rendered_prompt": {
                    "system_prompt": rendered_prompt.system_prompt,
                    "user_prompt": rendered_prompt.user_prompt,
                },
                "current_context_keys": context.list_render_context_keys() if context is not None else [],
                "output_key_injections": injection_results,
                "payload": payload,
            },
            level=logging.DEBUG,
        )

    def _resolve_template_value(
        self,
        value: Any,
        *,
        context: ContextStore,
        injection_results: list[dict[str, Any]],
    ) -> Any:
        if isinstance(value, str):
            rendered, replacements = context.render_template_string(value)
            if replacements:
                injection_results.extend(replacements)
            return rendered
        if isinstance(value, dict):
            return {
                key: self._resolve_template_value(item, context=context, injection_results=injection_results)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._resolve_template_value(item, context=context, injection_results=injection_results) for item in value]
        return value

    def build_request(
        self,
        *,
        rendered_prompt: RenderedPrompt,
        payload: dict[str, Any],
        context: ContextStore | None = None,
    ) -> LLMRequest:
        return LLMRequest(
            prompt=rendered_prompt.user_prompt,
            system_prompt=rendered_prompt.system_prompt,
            messages=[
                LLMMessage(role="system", content=rendered_prompt.system_prompt),
                LLMMessage(role="user", content=rendered_prompt.user_prompt),
            ],
            model_name=str(payload.get("model_name") or self.default_model_name or "").strip() or None,
            temperature=float(payload.get("temperature", self.default_temperature)),
            max_tokens=payload.get("max_tokens", self.default_max_tokens),
            timeout_seconds=int(payload.get("timeout_seconds", self.default_timeout_seconds or self.timeout)),
            request_id=context.runtime.request_id if context is not None else None,
            session_id=context.runtime.session_id if context is not None else None,
            trace_id=self.build_trace_id(context=context),
            prompt_name=rendered_prompt.name,
            prompt_version=rendered_prompt.version,
            response_schema_name=self.get_output_model().__name__,
            response_schema_version=self.response_schema_version,
            max_validation_retries=self.max_schema_retries,
            metadata={
                "tool_name": self.name,
                "operation": "tool",
                "tool_schema_version": self.schema_version,
                "response_schema_version": self.response_schema_version,
                **payload.get("metadata", {}),
            },
        )

    def build_function_schema(self) -> LLMFunctionSchema:
        return LLMFunctionSchema(
            name=self.function_name,
            description=self.function_description,
            parameters_schema=self.get_output_model().model_json_schema(),
            schema_name=self.get_output_model().__name__,
            schema_version=self.response_schema_version,
        )

    def build_success_result(
        self,
        *,
        request: LLMRequest,
        rendered_prompt: RenderedPrompt,
        output_model: BaseModel,
        raw_response: dict[str, Any],
        model_name: str | None,
        model_version: str | None,
        finish_reason: str | None,
        usage: dict[str, Any] | None,
        attempts_used: int,
        function_schema: LLMFunctionSchema,
        trace_id: str | None,
    ) -> ToolResult:
        metadata: dict[str, Any] = {
            "request": request.model_dump(exclude_none=True),
            "prompt_name": rendered_prompt.name,
            "prompt_version": rendered_prompt.version,
            "model_name": model_name,
            "model_version": model_version,
            "finish_reason": finish_reason,
            "raw_response": raw_response,
            "tool_schema_version": self.schema_version,
            "response_schema_name": function_schema.schema_name,
            "response_schema_version": function_schema.schema_version,
            "function_name": function_schema.name,
            "attempts_used": attempts_used,
            "trace_id": trace_id,
        }
        if usage is not None:
            metadata["usage"] = usage

        return self.build_result(
            success=True,
            output=output_model.model_dump(mode="json"),
            metadata=metadata,
        )

    def build_trace_id(self, *, context: ContextStore | None = None) -> str:
        request_id = context.runtime.request_id if context is not None else "no_request"
        return f"{request_id}:{self.name}:{uuid4().hex}"
