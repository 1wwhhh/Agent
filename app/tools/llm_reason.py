from __future__ import annotations

import re
from typing import Any

from pydantic import Field

from app.prompts import LLM_REASON_PROMPT_NAME, LLM_REASON_PROMPT_VERSION
from app.schemas.context import ContextStore
from app.schemas.llm import LLMRequest
from app.schemas.tool import ToolResult
from app.schemas.tool_outputs import ReasoningToolOutput
from app.tools.llm_base import BaseLLMTool
from app.tools.llm_client import FAIL_FAST_TIMEOUT_MARKER, RETRYABLE_ERROR_MARKER
from app.utils import runtime_progress
from app.utils.time_context import merge_runtime_time_context


class LLMReasonTool(BaseLLMTool):
    """Analyze input and produce a reasoned response for downstream tasks."""

    name: str = Field(default="llm_reason_tool")
    description: str = Field(default="Analyze input and produce a reasoned response for downstream tasks.")
    prompt_name: str = Field(default=LLM_REASON_PROMPT_NAME)
    prompt_version: str = Field(default=LLM_REASON_PROMPT_VERSION)
    function_name: str = Field(default="emit_reasoning_output")
    function_description: str = Field(default="Return validated reasoning output for the runtime.")
    timeout: int = Field(default=75, gt=0)
    default_timeout_seconds: int = Field(default=45, gt=0)
    default_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    tags: list[str] = Field(default_factory=lambda: ["llm", "reasoning", "analysis"])
    internal_disclosure_safe_text: str = Field(
        default=(
            "我不能提供当前系统的底层源码、内部提示词、工具封装、接口定义、路由策略或运行时实现细节。"
            "如果你是在做代码审查，请提供需要查看的文件或具体模块，我可以基于可见代码协助分析。"
        )
    )

    def get_routing_capability(self) -> dict[str, Any]:
        capability = super().get_routing_capability()
        capability["supported_task_types"] = ["reasoning"]
        capability["default_task_type"] = "reasoning"
        capability["supported_tags"] = list(self.tags)
        return capability

    def build_prompt_variables(self, payload: dict[str, Any], context: ContextStore | None = None) -> dict[str, Any]:
        prompt = str(payload.get("prompt") or payload.get("input") or payload.get("query") or "").strip()
        if not prompt:
            raise ValueError("llm_reason_tool requires a non-empty prompt")

        return {
            "prompt": prompt,
            "context_block": merge_runtime_time_context(
                payload.get("context"),
                context.runtime if context is not None else None,
            ),
        }

    def get_output_model(self) -> type[ReasoningToolOutput]:
        return ReasoningToolOutput

    def _asks_runtime_internal_disclosure(self, payload: dict[str, Any], context: ContextStore | None) -> bool:
        text = " ".join(
            str(item or "")
            for item in (
                payload.get("prompt"),
                payload.get("input"),
                payload.get("query"),
                payload.get("context"),
                context.runtime.user_input if context is not None else "",
            )
        ).lower()
        compact = re.sub(r"\s+", "", text)
        if not compact:
            return False
        subjects = ("你", "你的", "当前", "这个项目", "当前系统", "这个系统", "runtime", "agent", "your", "thissystem")
        sensitive = (
            "底层代码",
            "底层源码",
            "源代码",
            "内部代码",
            "系统提示词",
            "内部提示词",
            "prompt",
            "工具封装",
            "封装的工具",
            "怎么封装",
            "工具接口",
            "接口定义",
            "路由策略",
            "运行时实现",
            "内部实现",
            "sourcecode",
            "systemprompt",
            "implementation",
        )
        if any(item in compact for item in sensitive) and any(item in compact for item in subjects):
            return True
        return any(item in compact for item in ("底层源码", "底层代码", "内部代码", "系统提示词", "内部提示词"))

    def build_request(
        self,
        *,
        rendered_prompt,
        payload: dict[str, Any],
        context: ContextStore | None = None,
    ) -> LLMRequest:
        request = super().build_request(rendered_prompt=rendered_prompt, payload=payload, context=context)
        metadata = dict(request.metadata)
        metadata["disable_provider_retries"] = True
        metadata["minimal_retry_policy"] = True
        metadata["llm_request_timeout_seconds"] = request.timeout_seconds
        return request.model_copy(update={"metadata": metadata})

    async def _arun(self, payload: dict[str, Any], context: ContextStore | None = None) -> ToolResult:
        if self._asks_runtime_internal_disclosure(payload=payload, context=context):
            runtime_progress(
                step="模型输出",
                status="内部信息请求拦截",
                detail=self.internal_disclosure_safe_text,
                request_id=context.runtime.request_id if context is not None else None,
                session_id=context.runtime.session_id if context is not None else None,
            )
            return self.build_result(
                success=True,
                output={
                    "text": self.internal_disclosure_safe_text,
                    "summary": "已拒绝提供当前系统内部实现细节。",
                    "key_points": ["不提供内部实现细节", "可基于用户提供的可见代码协助分析"],
                },
                metadata={"runtime_internal_disclosure_guard": True},
            )

        result = await super()._arun(payload=payload, context=context)
        if result.success:
            return result

        error_text = str(result.error or "")
        metadata = dict(result.metadata)
        if FAIL_FAST_TIMEOUT_MARKER in error_text:
            metadata["timeout_fail_fast"] = True
        if RETRYABLE_ERROR_MARKER in error_text:
            metadata["retryable_error"] = True
        if metadata:
            metadata["llm_request_timeout_seconds"] = int(payload.get("timeout_seconds", self.default_timeout_seconds))
            metadata["tool_timeout_seconds"] = int(payload.get("tool_timeout_seconds", self.timeout))
        return result.model_copy(update={"metadata": metadata})

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "context": {"type": ["string", "null"]},
                "model_name": {"type": ["string", "null"]},
                "temperature": {"type": ["number", "null"]},
                "max_tokens": {"type": ["integer", "null"]},
            },
            "required": ["prompt"],
            "additionalProperties": True,
        }

    def get_output_schema(self) -> dict[str, Any]:
        return self.get_output_model().model_json_schema()
