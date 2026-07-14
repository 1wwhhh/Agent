from __future__ import annotations

import json
import os
import re
from typing import Any
from uuid import uuid4

import httpx
from pydantic import Field

from app.prompts import (
    RAG_SEARCH_INTENT_PROMPT_NAME,
    RAG_SEARCH_INTENT_PROMPT_VERSION,
    PromptRegistry,
    build_default_prompt_registry,
)
from app.rag.query_parser import ParsedSearchQuery, parse_search_query
from app.schemas.context import ContextStore
from app.schemas.llm import LLMFunctionSchema, LLMMessage, LLMRequest
from app.schemas.tool import ToolResult
from app.schemas.tool_outputs import RagSearchIntentToolOutput
from app.tools.base import BaseTool
from app.tools.function_calling import FunctionCallingAdapter
from app.tools.llm_client import LLMClient
from app.utils import runtime_log, runtime_progress

ALLOWED_SOURCE_TYPES = {"pdf", "word", "ppt", "excel"}
FULL_DOCUMENT_FETCH_MODES = {"full", "full_doc", "full_document", "all", "document"}
SEARCH_ONLY_FETCH_MODES = {"search", "semantic", "top_k", "chunks"}
EXCEL_FULL_FETCH_KEYWORDS = (
    "对比",
    "比较",
    "冲突",
    "判断",
    "核对",
    "检查",
    "统计",
    "汇总",
    "分析",
    "每个人",
    "每人",
    "每个日期",
    "每天",
    "每日",
    "日期",
    "工作内容",
    "计划",
    "完成情况",
    "延期",
)
PLAN_TRACKING_COMPLETION_KEYWORDS = (
    "是否完成",
    "有没有完成",
    "完成",
    "完成情况",
    "完成率",
    "哪些完成",
    "哪些已完成",
    "哪些没完成",
    "哪些未完成",
    "没完成",
    "未完成",
    "落地",
    "落实",
    "闭环",
    "延期",
)
PLAN_TRACKING_TIME_KEYWORDS = (
    "上周",
    "上一周",
    "上星期",
    "上个周",
    "本周",
    "这个周",
    "下周",
    "下一周",
    "本月",
    "这个月",
    "上个月",
    "月度",
    "每周",
)
DEPT_PLAN_SCOPE_KEYWORDS = ("三七计划", "计划书", "部门计划", "月度计划")
NATURAL_LANGUAGE_DOC_ID_MARKERS = (
    "怎么",
    "如何",
    "什么",
    "哪些",
    "是否",
    "有没有",
    "完成",
    "落地",
    "情况",
    "状态",
    "分析",
    "判断",
    "内容",
    "资料",
    "信息",
)
DESCRIPTIVE_DOCUMENT_KIND_MARKERS = (
    "专利",
    "权利要求",
    "说明书",
    "技术方案",
    "技术交底书",
    "交底书",
)
DOCUMENT_CONTENT_QUESTION_MARKERS = (
    "内容",
    "资料",
    "信息",
    "包含",
    "有什么",
    "有哪些",
    "怎么写",
    "如何写",
    "权利要求",
    "技术方案",
    "摘要",
)



def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        parsed = int(raw_value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


class RAGSearchTool(BaseTool):
    name: str = Field(default="rag_search_tool")
    description: str = Field(default="Search company knowledge base through existing RAG /search API.")
    timeout: int = Field(default=60, gt=0)
    tags: list[str] = Field(default_factory=lambda: ["rag", "search", "knowledge_base"])
    client: LLMClient | None = Field(default=None)
    prompt_registry: PromptRegistry = Field(default_factory=build_default_prompt_registry)
    function_adapter: FunctionCallingAdapter = Field(default_factory=FunctionCallingAdapter)
    use_llm_intent: bool = Field(default=True)
    intent_timeout_seconds: int = Field(default=20, gt=0)
    intent_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    joined_context_max_chars: int = Field(default_factory=lambda: _env_int("RAG_JOINED_CONTEXT_MAX_CHARS", 12000), gt=0)
    chunk_text_max_chars: int = Field(default_factory=lambda: _env_int("RAG_CHUNK_TEXT_MAX_CHARS", 4500), gt=0)
    max_context_chunks: int = Field(default_factory=lambda: _env_int("RAG_MAX_CONTEXT_CHUNKS", 1), gt=0)

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "prompt": {"type": "string"},
                "input": {"type": "string"},
                "top_k": {"type": ["integer", "string", "null"]},
                "source_type": {"type": ["string", "null"]},
                "doc_id": {"type": ["string", "null"]},
                "relative_path": {"type": ["string", "null"]},
                "absolute_path": {"type": ["string", "null"]},
                "fetch_all": {"type": ["boolean", "string", "null"]},
                "fetch_mode": {"type": ["string", "null"]},
                "weight_m3": {"type": ["number", "string", "null"]},
                "weight_zh": {"type": ["number", "string", "null"]},
                "weight_sparse": {"type": ["number", "string", "null"]},
                "rag_base_url": {"type": ["string", "null"]},
                "use_llm_intent": {"type": ["boolean", "string", "null"]},
                "intent_timeout_seconds": {"type": ["integer", "string", "null"]},
                "intent_temperature": {"type": ["number", "string", "null"]},
            },
            "required": [],
            "additionalProperties": True,
        }

    def get_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "chunks": {"type": "array"},
                "joined_context": {"type": "string"},
                "joined_context_is_preview": {"type": "boolean"},
                "context_batches": {"type": "array"},
                "summary": {"type": "string"},
                "low_relevance": {"type": "boolean"},
                "top_score": {"type": "number"},
                "threshold": {"type": "number"},
                "llm_search_intent": {"type": ["object", "null"]},
                "search_intent_mode": {"type": "string"},
                "fetch_all_requested": {"type": "boolean"},
                "fetch_mode": {"type": ["string", "null"]},
                "relative_path": {"type": ["string", "null"]},
                "absolute_path": {"type": ["string", "null"]},
                "full_document": {"type": "boolean"},
                "sheets": {"type": "array"},
            },
            "required": ["query", "chunks", "joined_context", "summary", "context_batches"],
            "additionalProperties": True,
        }

    def get_routing_capability(self) -> dict[str, Any]:
        capability = super().get_routing_capability()
        capability["supported_task_types"] = ["rag_search"]
        capability["default_task_type"] = "rag_search"
        capability["supported_tags"] = list(self.tags)
        return capability

    async def _arun(self, payload: dict[str, Any], context: ContextStore | None = None) -> ToolResult:
        tool_query = str(payload.get("query") or payload.get("prompt") or payload.get("input") or "").strip()
        raw_query, raw_query_source = self._select_raw_query_for_intent(tool_query=tool_query, context=context)
        if not raw_query:
            return self.build_result(
                success=False,
                error="rag_search_tool requires non-empty query",
                metadata={"rag_route": "/search", "payload": payload},
            )

        rag_base_url = str(payload.get("rag_base_url") or os.getenv("RAG_BASE_URL", "")).strip().rstrip("/")
        if not rag_base_url:
            return self.build_result(
                success=False,
                error="RAG_BASE_URL is not configured",
                metadata={"rag_route": "/search", "payload": payload},
            )

        parsed_query, query_parse_error = self._parse_search_query(raw_query)
        rule_query = self._build_request_query(raw_query=raw_query, parsed_query=parsed_query)
        parsed_source_type = parsed_query.source_type if parsed_query is not None else None
        parsed_doc_id = parsed_query.doc_id if parsed_query is not None else None
        parsed_scope_keyword = parsed_query.scope_keyword if parsed_query is not None else None
        parsed_relative_path = parsed_query.relative_path if parsed_query is not None else None
        parsed_absolute_path = parsed_query.absolute_path if parsed_query is not None else None
        llm_intent, llm_intent_error = await self._extract_search_intent(
            raw_query=raw_query,
            payload=payload,
            context=context,
        )
        intent_query = llm_intent.query if llm_intent is not None else None
        intent_source_type = llm_intent.source_type if llm_intent is not None else None
        intent_doc_id = llm_intent.doc_id if llm_intent is not None else None
        intent_relative_path = llm_intent.relative_path if llm_intent is not None else None
        intent_absolute_path = llm_intent.absolute_path if llm_intent is not None else None
        source_type, ignored_source_type = self._normalize_source_type(payload.get("source_type"))
        relative_path = self._first_non_empty(payload.get("relative_path"), parsed_relative_path, intent_relative_path)
        absolute_path = self._first_non_empty(payload.get("absolute_path"), parsed_absolute_path, intent_absolute_path)
        path_scope_present = bool(relative_path or absolute_path)
        if path_scope_present:
            source_type = source_type or parsed_source_type
            doc_id = self._first_non_empty(payload.get("doc_id"), parsed_doc_id)
        else:
            source_type = source_type or parsed_source_type or intent_source_type
            doc_id = self._select_doc_id(
                raw_query=raw_query,
                payload_doc_id=payload.get("doc_id"),
                parsed_doc_id=parsed_doc_id,
                intent_doc_id=intent_doc_id,
                parsed_scope_keyword=parsed_scope_keyword,
            )
        query, query_adjustment = self._select_request_query(
            raw_query=raw_query,
            intent_query=intent_query,
            rule_query=rule_query,
            source_type=source_type,
            doc_id=doc_id,
        )
        fetch_all_requested, fetch_mode = self._resolve_fetch_mode(
            payload=payload,
            raw_query=raw_query,
            query=query,
            source_type=source_type,
            doc_id=doc_id,
            relative_path=relative_path,
            absolute_path=absolute_path,
        )
        search_intent_mode = "llm" if llm_intent is not None else ("rules_fallback" if llm_intent_error else "rules")
        query_metadata = self._build_query_metadata(
            raw_query=raw_query,
            request_query=query,
            parsed_query=parsed_query,
            query_parse_error=query_parse_error,
            llm_intent=llm_intent,
            llm_intent_error=llm_intent_error,
            search_intent_mode=search_intent_mode,
            query_adjustment=query_adjustment,
            fetch_all_requested=fetch_all_requested,
            fetch_mode=fetch_mode,
        )
        if relative_path:
            query_metadata["relative_path"] = relative_path
        if absolute_path:
            query_metadata["absolute_path"] = absolute_path
        if raw_query_source != "tool_input":
            query_metadata["tool_query"] = tool_query
            query_metadata["raw_query_source"] = raw_query_source
        try:
            request_body = {
                "query": query,
                "top_k": self._resolve_top_k(payload=payload, fetch_all_requested=fetch_all_requested),
                "source_type": source_type,
                "doc_id": doc_id,
                "weight_m3": self._safe_float(payload.get("weight_m3"), default=0.5),
                "weight_zh": self._safe_float(payload.get("weight_zh"), default=0.4),
                "weight_sparse": self._safe_float(payload.get("weight_sparse"), default=0.2),
            }
            if relative_path:
                request_body["relative_path"] = relative_path
            if absolute_path:
                request_body["absolute_path"] = absolute_path
            if fetch_all_requested:
                request_body["fetch_all"] = True
                request_body["fetch_mode"] = fetch_mode or "full_document"
            if context is not None:
                runtime_progress(
                    step="rag_search:request",
                    status="prepared",
                    detail=json.dumps(
                        {
                            "query": query,
                            "source_type": source_type,
                            "doc_id": doc_id,
                            "relative_path": relative_path,
                            "absolute_path": absolute_path,
                            "fetch_all": fetch_all_requested,
                            "fetch_mode": fetch_mode,
                            "query_adjustment": query_adjustment,
                            "search_intent_mode": search_intent_mode,
                            "raw_query_source": raw_query_source,
                        },
                        ensure_ascii=False,
                    ),
                    request_id=context.runtime.request_id,
                    session_id=context.runtime.session_id,
                )
        except Exception as exc:
            return self.build_result(
                success=False,
                error=f"invalid rag search parameters: {exc}",
                metadata={
                    "rag_route": "/search",
                    "payload": payload,
                    **query_metadata,
                    "exception_type": type(exc).__name__,
                },
            )

        try:
            async with httpx.AsyncClient(timeout=self.resolve_timeout(payload)) as client:
                response = await client.post(f"{rag_base_url}/search", json=request_body)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            return self.build_result(
                success=False,
                error=f"RAG search request failed: {exc}",
                metadata=self._build_request_metadata(
                    rag_route="/search",
                    rag_base_url=rag_base_url,
                    request_body=request_body,
                    exception_type=type(exc).__name__,
                    ignored_source_type=ignored_source_type,
                    **query_metadata,
                ),
            )

        results_raw = data.get("results")
        results = results_raw if isinstance(results_raw, list) else []
        sheets_raw = data.get("sheets")
        sheets = sheets_raw if isinstance(sheets_raw, list) else []
        if not results and sheets:
            results = self._build_chunks_from_sheets(sheets=sheets, doc_id=doc_id)
        context_batches, preview_joined_context, source_chunk_ids = self._build_context_batches(results)
        summary = f"RAG 检索完成，共 {len(results)} 个 chunk，已分成 {len(context_batches)} 个 context batch。"

        return self.build_result(
            success=True,
            output={
                "query": str(data.get("query", query)),
                "raw_query": raw_query,
                "parsed_search_query": (
                    parsed_query.model_dump(mode="json") if parsed_query is not None else None
                ),
                "llm_search_intent": llm_intent.model_dump(mode="json") if llm_intent is not None else None,
                "search_intent_mode": search_intent_mode,
                "fetch_all_requested": fetch_all_requested,
                "fetch_mode": fetch_mode,
                "relative_path": relative_path,
                "absolute_path": absolute_path,
                "full_document": bool(data.get("full_document", False)),
                "sheets": sheets,
                "chunks": results,
                "joined_context": preview_joined_context,
                "joined_context_is_preview": True,
                "context_batches": context_batches,
                "summary": summary,
                "low_relevance": bool(data.get("low_relevance", False)),
                "top_score": self._safe_float(data.get("top_score"), default=0.0),
                "threshold": self._safe_float(data.get("threshold"), default=0.0),
            },
            metadata=self._build_request_metadata(
                rag_route="/search",
                rag_base_url=rag_base_url,
                total=self._safe_int(data.get("total"), default=len(results)),
                ignored_source_type=ignored_source_type,
                full_document=bool(data.get("full_document", False)),
                **query_metadata,
                joined_context_truncated=len(context_batches) > 1,
                joined_context_chars=len(preview_joined_context),
                source_chunks_used=len(source_chunk_ids),
            ),
        )

    def _safe_int(self, value: Any, *, default: int) -> int:
        if value is None or value == "":
            return default
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed > 0 else default

    def _safe_float(self, value: Any, *, default: float) -> float:
        if value is None or value == "":
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    async def _extract_search_intent(
        self,
        *,
        raw_query: str,
        payload: dict[str, Any],
        context: ContextStore | None,
    ) -> tuple[RagSearchIntentToolOutput | None, str | None]:
        if self.client is None or not self._should_use_llm_intent(payload):
            return None, None

        try:
            rendered_prompt = self.prompt_registry.render(
                RAG_SEARCH_INTENT_PROMPT_NAME,
                version=RAG_SEARCH_INTENT_PROMPT_VERSION,
                variables={"raw_query": raw_query},
            )
            request_id = context.runtime.request_id if context is not None else None
            session_id = context.runtime.session_id if context is not None else None
            trace_prefix = request_id or "rag_search"
            intent_timeout = self._safe_int(
                payload.get("intent_timeout_seconds"),
                default=self.intent_timeout_seconds,
            )
            intent_temperature = max(
                0.0,
                min(2.0, self._safe_float(payload.get("intent_temperature"), default=self.intent_temperature)),
            )
            request = LLMRequest(
                prompt=rendered_prompt.user_prompt,
                system_prompt=rendered_prompt.system_prompt,
                messages=[
                    LLMMessage(role="system", content=rendered_prompt.system_prompt),
                    LLMMessage(role="user", content=rendered_prompt.user_prompt),
                ],
                model_name=payload.get("model_name"),
                temperature=intent_temperature,
                timeout_seconds=intent_timeout,
                request_id=request_id,
                session_id=session_id,
                trace_id=f"{trace_prefix}:rag_search_intent:{uuid4().hex}",
                prompt_name=rendered_prompt.name,
                prompt_version=rendered_prompt.version,
                response_schema_name=RagSearchIntentToolOutput.__name__,
                response_schema_version="v1",
                max_validation_retries=1,
                metadata={
                    "operation": "rag_search_intent",
                    "tool_name": self.name,
                },
            )
            result = await self.function_adapter.invoke_structured(
                client=self.client,
                request=request,
                function_schema=LLMFunctionSchema(
                    name="extract_rag_search_intent",
                    description="Extract query, source_type, doc_id, relative_path, absolute_path, and confidence for RAG search.",
                    parameters_schema=RagSearchIntentToolOutput.model_json_schema(),
                    schema_name=RagSearchIntentToolOutput.__name__,
                    schema_version="v1",
                ),
                output_model=RagSearchIntentToolOutput,
            )
            intent = result.output
            if not isinstance(intent, RagSearchIntentToolOutput):
                intent = RagSearchIntentToolOutput.model_validate(intent)
            intent_query_adjustment = None
            if (
                intent.source_type == "excel"
                and intent.doc_id
                and self._excel_intent_query_semantic_drifted(raw_query=raw_query, intent_query=intent.query)
            ):
                fallback_query = self._strip_doc_reference_from_query(raw_query=raw_query, doc_id=intent.doc_id)
                if fallback_query and fallback_query != intent.query:
                    intent = intent.model_copy(update={"query": fallback_query})
                    intent_query_adjustment = "excel_intent_query_drift_replaced"
            runtime_progress(
                step="rag_search:intent",
                status="llm_result",
                detail=json.dumps(intent.model_dump(mode="json"), ensure_ascii=False),
                request_id=request_id,
                session_id=session_id,
            )
            runtime_log(
                layer="rag_search_intent",
                event="success",
                data={
                    "intent": intent.model_dump(mode="json"),
                    "intent_query_adjustment": intent_query_adjustment,
                    "attempts_used": result.attempts_used,
                    "prompt_name": rendered_prompt.name,
                    "prompt_version": rendered_prompt.version,
                },
            )
            return intent, None
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            request_id = context.runtime.request_id if context is not None else None
            session_id = context.runtime.session_id if context is not None else None
            runtime_progress(
                step="rag_search:intent",
                status="fallback",
                detail=f"llm_intent_error={self._truncate_metadata_text(error_message)} stage=rules",
                request_id=request_id,
                session_id=session_id,
            )
            runtime_log(
                layer="rag_search_intent",
                event="error",
                data={
                    "error": error_message,
                    "stage": "fallback_to_rules",
                },
            )
            return None, error_message

    def _select_raw_query_for_intent(
        self,
        *,
        tool_query: str,
        context: ContextStore | None,
    ) -> tuple[str, str]:
        if not tool_query or context is None:
            return tool_query, "tool_input"

        user_input = str(getattr(context.runtime, "user_input", "") or "").strip()
        if not user_input or user_input == tool_query:
            return tool_query, "tool_input"

        if self._should_use_runtime_user_input(tool_query=tool_query, user_input=user_input):
            return user_input, "runtime_user_input"

        return tool_query, "tool_input"

    def _should_use_runtime_user_input(self, *, tool_query: str, user_input: str) -> bool:
        if tool_query not in user_input:
            return False
        if len(user_input) <= len(tool_query) + 8:
            return False

        remaining_user_objective = user_input.replace(tool_query, " ", 1)
        if self._looks_like_file_reference_input(tool_query):
            return self._has_task_objective_text(remaining_user_objective)

        return self._has_document_reference_text(user_input) and (
            self._has_task_objective_text(tool_query) or self._has_task_objective_text(remaining_user_objective)
        )

    def _looks_like_file_reference_input(self, value: str) -> bool:
        text = value.strip()
        if not text:
            return False
        return bool(
            re.search(
                r"(?:\.(?:xlsx?|docx?|pdf|pptx?)|__(?:xlsx?|docx?|pdf|pptx?))\s*$",
                text,
                flags=re.IGNORECASE,
            )
        )

    def _has_document_reference_text(self, value: str) -> bool:
        text = value.strip()
        if not text:
            return False
        return bool(
            re.search(
                r"(?:\.(?:xlsx?|docx?|pdf|pptx?)|__(?:xlsx?|docx?|pdf|pptx?))",
                text,
                flags=re.IGNORECASE,
            )
        )

    def _has_task_objective_text(self, value: str) -> bool:
        text = value.strip()
        if not text:
            return False
        task_keywords = (
            "分析",
            "比较",
            "对比",
            "判断",
            "查找",
            "提取",
            "统计",
            "汇总",
            "核对",
            "识别",
            "找出",
            "筛选",
            "排序",
            "列出",
            "归纳",
            "每个人",
            "每个日期",
            "本周工作",
            "下周计划",
            "下周的计划",
            "是否完成",
            "完成情况",
        )
        return any(keyword in text for keyword in task_keywords)

    def _should_use_llm_intent(self, payload: dict[str, Any]) -> bool:
        value = payload.get("use_llm_intent")
        if value is None:
            return self.use_llm_intent
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() not in {"0", "false", "no", "off"}

    def _first_non_empty(self, *values: Any) -> str | None:
        for value in values:
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return None

    def _select_doc_id(
        self,
        *,
        raw_query: str,
        payload_doc_id: Any,
        parsed_doc_id: str | None,
        intent_doc_id: str | None,
        parsed_scope_keyword: str | None,
    ) -> str | None:
        explicit_doc_id = self._first_non_empty(payload_doc_id, parsed_doc_id)
        if explicit_doc_id:
            return explicit_doc_id

        if intent_doc_id and self._doc_id_candidate_is_safe(
            raw_query=raw_query,
            candidate=intent_doc_id,
            parsed_scope_keyword=parsed_scope_keyword,
        ):
            return intent_doc_id.strip()

        if parsed_scope_keyword and self._scope_keyword_can_be_doc_id(
            raw_query=raw_query,
            scope_keyword=parsed_scope_keyword,
        ):
            return parsed_scope_keyword.strip()

        return None

    def _doc_id_candidate_is_safe(
        self,
        *,
        raw_query: str,
        candidate: str,
        parsed_scope_keyword: str | None = None,
    ) -> bool:
        doc_id = candidate.strip()
        if not doc_id:
            return False
        if self._looks_like_file_reference_input(doc_id):
            return True
        if parsed_scope_keyword and doc_id == parsed_scope_keyword:
            return self._scope_keyword_can_be_doc_id(raw_query=raw_query, scope_keyword=doc_id)
        return self._scope_keyword_can_be_doc_id(raw_query=raw_query, scope_keyword=doc_id)

    def _scope_keyword_can_be_doc_id(self, *, raw_query: str, scope_keyword: str) -> bool:
        candidate = scope_keyword.strip()
        if not candidate:
            return False
        if self._looks_like_file_reference_input(candidate):
            return True
        if "\\" in candidate or "/" in candidate:
            return False
        if len(candidate) > 40:
            return False
        if re.search(r"[,，。;；!?！？]", candidate):
            return False
        if any(marker in candidate for marker in NATURAL_LANGUAGE_DOC_ID_MARKERS):
            return False

        raw = raw_query.strip()
        if any(kind in candidate for kind in DESCRIPTIVE_DOCUMENT_KIND_MARKERS):
            if any(marker in raw for marker in DOCUMENT_CONTENT_QUESTION_MARKERS):
                return False

        return True

    def _truncate_metadata_text(self, text: str, *, limit: int = 1000) -> str:
        if len(text) <= limit:
            return text
        return f"{text[:limit]}... [truncated {len(text) - limit} chars]"

    def _parse_search_query(self, raw_query: str) -> tuple[ParsedSearchQuery | None, str | None]:
        try:
            return parse_search_query(raw_query), None
        except (TypeError, ValueError) as exc:
            return None, str(exc)

    def _build_request_query(self, *, raw_query: str, parsed_query: ParsedSearchQuery | None) -> str:
        if parsed_query is None:
            return raw_query
        return parsed_query.query

    def _select_request_query(
        self,
        *,
        raw_query: str,
        intent_query: str | None,
        rule_query: str | None,
        source_type: str | None,
        doc_id: str | None,
    ) -> tuple[str, str | None]:
        rule_candidate = self._build_rule_request_query(
            raw_query=raw_query,
            rule_query=rule_query,
            doc_id=doc_id,
        )
        candidate = self._first_non_empty(rule_candidate, intent_query, raw_query) or raw_query

        normalized_intent_query = str(intent_query or "").strip()
        if normalized_intent_query and rule_candidate and normalized_intent_query != rule_candidate:
            if doc_id and self._looks_like_doc_title_query(query=normalized_intent_query, doc_id=doc_id):
                return candidate, "doc_title_query_replaced"
            if source_type == "excel" and doc_id:
                return candidate, "excel_rule_query_preferred_over_llm"
            return candidate, "rule_query_preferred_over_llm"

        return candidate, None

    def _build_rule_request_query(
        self,
        *,
        raw_query: str,
        rule_query: str | None,
        doc_id: str | None,
    ) -> str | None:
        stripped_query = self._strip_doc_reference_from_query(raw_query=raw_query, doc_id=doc_id) if doc_id else None
        if doc_id and self._looks_like_file_reference_input(doc_id) and stripped_query and stripped_query != raw_query:
            return stripped_query
        return self._first_non_empty(rule_query, stripped_query if stripped_query != raw_query else None, raw_query)

    def _excel_intent_query_semantic_drifted(self, *, raw_query: str, intent_query: str) -> bool:
        raw_text = raw_query.lower()
        intent_text = intent_query.lower()
        raw_has_completion_objective = any(
            keyword in raw_text
            for keyword in (
                "是否完成",
                "完成情况",
                "下周计划",
                "下周的计划",
                "下一周",
                "本周工作",
                "同时完成",
            )
        )
        intent_invented_conflict = "冲突" in intent_text and "冲突" not in raw_text
        overlap_keywords = ("同时完成", "好几个人", "多人", "多个人", "相同", "相近", "类似")
        intent_invented_overlap = any(keyword in intent_text for keyword in overlap_keywords) and not any(
            keyword in raw_text for keyword in overlap_keywords
        )
        return raw_has_completion_objective and (intent_invented_conflict or intent_invented_overlap)

    def _looks_like_doc_title_query(self, *, query: str, doc_id: str) -> bool:
        query_key = self._normalize_title_fragment(query)
        doc_key = self._normalize_title_fragment(doc_id)
        if len(query_key) < 2 or len(doc_key) < 2:
            return False
        return query_key == doc_key or query_key in doc_key

    def _normalize_title_fragment(self, value: str) -> str:
        normalized = value.strip().lower()
        normalized = re.sub(r"\.[a-z0-9]+$", " ", normalized)
        normalized = re.sub(r"[\s_\-.,，。:：;；、()（）\[\]【】<>《》]+", "", normalized)
        return normalized

    def _strip_doc_reference_from_query(self, *, raw_query: str, doc_id: str) -> str | None:
        text = raw_query
        doc_id = doc_id.strip()
        if doc_id:
            text = re.sub(re.escape(doc_id), " ", text, flags=re.IGNORECASE)
            basename = re.sub(r"\.[A-Za-z0-9]+$", "", doc_id).strip()
            if basename and basename != doc_id:
                text = re.sub(re.escape(basename), " ", text, flags=re.IGNORECASE)
        text = re.sub(
            r"^(?:请\s*)?(?:帮我\s*)?(?:查询|检索|查找|搜索|找到|根据|基于|依据|按照)(?:一下)?",
            " ",
            text,
        )
        text = re.sub(r"^\s*(?:中的|里的|里面的|中|里|里面|的)", " ", text)
        text = re.sub(r"(?:这个|该)?(?:excel|xlsx?|表格|文件|文档)(?:中|里|里面|的)?", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"^\s*(?:中的|里的|里面的|中|里|里面|的)", " ", text)
        text = re.sub(r"^[\s,，。:：;；、]+|[\s,，。:：;；、]+$", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text or None

    def _resolve_fetch_mode(
        self,
        *,
        payload: dict[str, Any],
        raw_query: str,
        query: str,
        source_type: str | None,
        doc_id: str | None,
        relative_path: str | None,
        absolute_path: str | None,
    ) -> tuple[bool, str | None]:
        explicit_fetch_all = self._safe_bool(payload.get("fetch_all"))
        raw_fetch_mode = str(payload.get("fetch_mode") or "").strip().lower()
        if self._looks_like_plan_tracking_query(raw_query=raw_query, query=query):
            return False, raw_fetch_mode if raw_fetch_mode in SEARCH_ONLY_FETCH_MODES else "search"
        if explicit_fetch_all is not None:
            return explicit_fetch_all, "full_document" if explicit_fetch_all else (raw_fetch_mode or None)
        if raw_fetch_mode in FULL_DOCUMENT_FETCH_MODES:
            return True, "full_document"
        if raw_fetch_mode in SEARCH_ONLY_FETCH_MODES:
            return False, raw_fetch_mode
        if relative_path or absolute_path:
            return True, "full_document"
        if self._should_auto_fetch_full_document(
            raw_query=raw_query,
            query=query,
            source_type=source_type,
            doc_id=doc_id,
        ):
            return True, "full_document"
        return False, raw_fetch_mode or None

    def _resolve_top_k(self, *, payload: dict[str, Any], fetch_all_requested: bool) -> int:
        explicit_top_k = payload.get("top_k")
        requested_top_k = self._safe_int(explicit_top_k, default=10)
        if fetch_all_requested:
            return max(requested_top_k, _env_int("RAG_FULL_DOCUMENT_TOP_K", 2000))
        return requested_top_k


    def _safe_bool(self, value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if value is None or value == "":
            return None
        raw_value = str(value).strip().lower()
        if raw_value in {"1", "true", "yes", "y", "on"}:
            return True
        if raw_value in {"0", "false", "no", "n", "off"}:
            return False
        return None

    def _should_auto_fetch_full_document(
        self,
        *,
        raw_query: str,
        query: str,
        source_type: str | None,
        doc_id: str | None,
    ) -> bool:
        if source_type != "excel" or not doc_id:
            return False
        haystack = f"{raw_query} {query}"
        return any(keyword in haystack for keyword in EXCEL_FULL_FETCH_KEYWORDS)

    def _looks_like_plan_tracking_query(self, *, raw_query: str, query: str) -> bool:
        normalized = re.sub(r"\s+", "", f"{raw_query}{query}")
        if not normalized:
            return False

        has_completion = any(keyword in normalized for keyword in PLAN_TRACKING_COMPLETION_KEYWORDS)
        has_dept_plan_scope = any(keyword in normalized for keyword in DEPT_PLAN_SCOPE_KEYWORDS)
        if has_dept_plan_scope and (
            has_completion or any(keyword in normalized for keyword in ("执行情况", "卡住", "卡着", "没动"))
        ):
            return True

        has_plan_scope = "计划" in normalized or "周报" in normalized
        has_time_scope = any(keyword in normalized for keyword in PLAN_TRACKING_TIME_KEYWORDS) or bool(
            re.search(r"(?:\d{4}年)?\d{1,2}月|[一二三四五六七八九十]+月份?", normalized)
        )
        has_work_completion_intent = (
            any(keyword in normalized for keyword in ("工作", "完成记录", "完成内容"))
            and any(keyword in normalized for keyword in ("完成", "没完成", "未完成"))
        )
        return (has_plan_scope or has_work_completion_intent) and has_completion and has_time_scope

    def _build_query_metadata(
        self,
        *,
        raw_query: str,
        request_query: str,
        parsed_query: ParsedSearchQuery | None,
        query_parse_error: str | None,
        llm_intent: RagSearchIntentToolOutput | None,
        llm_intent_error: str | None,
        search_intent_mode: str,
        query_adjustment: str | None = None,
        fetch_all_requested: bool = False,
        fetch_mode: str | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "raw_query": raw_query,
            "request_query": request_query,
            "search_intent_mode": search_intent_mode,
            "fetch_all_requested": fetch_all_requested,
        }
        if fetch_mode:
            metadata["fetch_mode"] = fetch_mode
        if query_adjustment:
            metadata["query_adjustment"] = query_adjustment
        if parsed_query is not None:
            metadata["parsed_search_query"] = parsed_query.model_dump(mode="json")
        if query_parse_error:
            metadata["query_parse_error"] = query_parse_error
        if llm_intent is not None:
            metadata["llm_search_intent"] = llm_intent.model_dump(mode="json")
        if llm_intent_error:
            metadata["llm_search_intent_error"] = self._truncate_metadata_text(llm_intent_error)
        return metadata

    def _normalize_source_type(self, value: Any) -> tuple[str | None, str | None]:
        if value is None:
            return None, None

        raw_value = str(value).strip().lower()
        if not raw_value:
            return None, None

        if raw_value in ALLOWED_SOURCE_TYPES:
            return raw_value, None

        return None, raw_value

    def _build_request_metadata(self, *, ignored_source_type: str | None = None, **metadata: Any) -> dict[str, Any]:
        if ignored_source_type:
            metadata["ignored_source_type"] = ignored_source_type
        return metadata

    def _build_chunks_from_sheets(self, *, sheets: list[Any], doc_id: str | None) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        for sheet_index, sheet in enumerate(sheets, start=1):
            if not isinstance(sheet, dict):
                continue
            sheet_name = str(sheet.get("sheet_name") or sheet.get("name") or f"sheet_{sheet_index}").strip()
            rows_raw = sheet.get("rows")
            rows = rows_raw if isinstance(rows_raw, list) else []
            for row_offset, row in enumerate(rows, start=1):
                if not isinstance(row, dict):
                    continue
                row_index = row.get("row_index") or row.get("index") or row_offset
                values_raw = row.get("values")
                values = values_raw if isinstance(values_raw, dict) else row
                cell_parts = [
                    f"{key}: {value}"
                    for key, value in values.items()
                    if key not in {"row_index", "index"} and value not in (None, "")
                ]
                if not cell_parts:
                    continue
                chunk_id = f"{doc_id or 'excel'}:{sheet_name}:row_{row_index}"
                chunks.append(
                    {
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "context_text": f"sheet={sheet_name} row={row_index}\n" + "; ".join(cell_parts),
                        "source_type": "excel",
                        "sheet_name": sheet_name,
                        "row_index": row_index,
                        "score": 1.0,
                    }
                )
        return chunks

    def _build_context_batches(self, results: list[Any]) -> tuple[list[dict[str, Any]], str, set[str]]:
        segments = self._build_context_segments(results)
        batches: list[dict[str, Any]] = []
        current_segments: list[dict[str, Any]] = []
        current_chars = 0
        current_chunk_ids: list[str] = []
        current_source_ids: set[str] = set()
        preview_joined_context = ""
        source_chunk_ids: set[str] = set()

        for segment in segments:
            segment_text = segment["segment_text"]
            segment_chars = len(segment_text)

            if current_segments and (
                len(current_source_ids) >= self.max_context_chunks
                or current_chars + segment_chars + 2 > self.joined_context_max_chars
            ):
                batch = self._finalize_batch(
                    batch_id=f"batch_{len(batches) + 1}",
                    segments=current_segments,
                    chunk_ids=current_chunk_ids,
                    source_chunk_ids=current_source_ids,
                )
                if not preview_joined_context:
                    preview_joined_context = batch["joined_context"]
                batches.append(batch)
                current_segments = []
                current_chars = 0
                current_chunk_ids = []
                current_source_ids = set()

            current_segments.append(segment)
            current_chars += segment_chars + 2
            current_chunk_ids.append(segment["chunk_id"])
            current_source_ids.add(segment["source_chunk_id"])
            source_chunk_ids.add(segment["source_chunk_id"])

        if current_segments:
            batch = self._finalize_batch(
                batch_id=f"batch_{len(batches) + 1}",
                segments=current_segments,
                chunk_ids=current_chunk_ids,
                source_chunk_ids=current_source_ids,
            )
            if not preview_joined_context:
                preview_joined_context = batch["joined_context"]
            batches.append(batch)

        return batches, preview_joined_context, source_chunk_ids

    def _build_context_segments(self, results: list[Any]) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        for index, item in enumerate(results, start=1):
            if not isinstance(item, dict):
                continue

            raw_text = str(item.get("context_text") or item.get("text") or "").strip()
            if not raw_text:
                continue

            source = item.get("relative_path") or item.get("section_path") or item.get("doc_id") or "unknown"
            page_id = item.get("page_id")
            score = item.get("score")
            header_parts = [f"[{index}] source={source}"]
            if page_id is not None:
                header_parts.append(f"page={page_id}")
            if score is not None:
                header_parts.append(f"score={score}")
            header = " | ".join(header_parts)
            source_chunk_id = str(item.get("chunk_id") or item.get("id") or item.get("doc_id") or f"chunk_{index}")
            part_limit = min(self.chunk_text_max_chars, self.joined_context_max_chars)
            raw_parts = [raw_text[i : i + part_limit] for i in range(0, len(raw_text), part_limit)] or [""]
            for part_index, part in enumerate(raw_parts, start=1):
                part_header = header
                if len(raw_parts) > 1:
                    part_header = f"{header} | part={part_index}/{len(raw_parts)}"
                segments.append(
                    {
                        "chunk_id": f"{source_chunk_id}#part_{part_index}" if len(raw_parts) > 1 else source_chunk_id,
                        "source_chunk_id": source_chunk_id,
                        "segment_text": f"{part_header}\n{part}" if part_header else part,
                    }
                )
        return segments

    def _finalize_batch(
        self,
        *,
        batch_id: str,
        segments: list[dict[str, Any]],
        chunk_ids: list[str],
        source_chunk_ids: set[str],
    ) -> dict[str, Any]:
        joined_context = "\n\n".join(segment["segment_text"] for segment in segments)
        if len(joined_context) > self.joined_context_max_chars:
            joined_context = joined_context[: self.joined_context_max_chars]
        return {
            "batch_id": batch_id,
            "chunk_ids": list(chunk_ids),
            "source_chunk_ids": sorted(source_chunk_ids),
            "source_count": len(source_chunk_ids),
            "joined_context": joined_context,
            "chars": len(joined_context),
        }
