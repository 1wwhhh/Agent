from __future__ import annotations

import json
import os
from typing import Any

import httpx
from pydantic import Field

from app.schemas.context import ContextStore
from app.schemas.tool import ToolResult
from app.tools.base import BaseTool
from app.utils import runtime_progress

AB_CASE_SEARCH_ROUTE = "/monthly/cases/search"
_BOOLEAN_REQUEST_FIELDS = (
    "include_review",
    "include_formula_score",
    "include_evidence",
    "include_vector_text",
)
_DISPLAY_FIELD_PRIORITY = (
    "example_id",
    "similarity",
    "doc_id",
    "source_file",
    "case_class",
    "case_no",
    "event_text",
    "reason_text",
    "score_reason",
    "score_delta",
    "score_text",
    "evidence_text",
    "needs_review",
    "vector_text",
    "case_type",
    "score",
    "final_score",
    "standard_score",
    "deduction_score",
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


class ABCaseSearchTool(BaseTool):
    name: str = Field(default="ab_case_search_tool")
    description: str = Field(
        default=(
            "检索 A/B 案例评分样例；A案例表示好事奖励，B案例表示坏事惩罚。"
            "通过 RAG 后端 /monthly/cases/search 返回相似案例、相似度和完整案例字段。"
        )
    )
    timeout: int = Field(default=60, gt=0)
    tags: list[str] = Field(default_factory=lambda: ["ab_case", "case_search", "rag", "business"])
    default_top_k: int = Field(default_factory=lambda: _env_int("AB_CASE_SEARCH_DEFAULT_TOP_K", 8), gt=0)
    max_top_k: int = Field(default_factory=lambda: _env_int("AB_CASE_SEARCH_MAX_TOP_K", 50), gt=0)
    context_max_results: int = Field(default_factory=lambda: _env_int("AB_CASE_CONTEXT_MAX_RESULTS", 8), gt=0)
    context_field_max_chars: int = Field(default_factory=lambda: _env_int("AB_CASE_CONTEXT_FIELD_MAX_CHARS", 1200), gt=0)

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": ["string", "null"],
                    "description": "兼容字段。没有传 event_text / reason_text 时，用它作为完整检索文本。",
                },
                "prompt": {"type": "string"},
                "input": {"type": "string"},
                "event_text": {"type": ["string", "null"], "description": "待匹配事件。推荐传这个。"},
                "reason_text": {"type": ["string", "null"], "description": "待匹配缘由/原因。"},
                "case_class": {
                    "type": ["string", "null"],
                    "enum": ["A", "B", "a", "b", None],
                    "description": "只查 A 类或 B 类案例；A=好事奖励，B=坏事惩罚。",
                },
                "case_type": {
                    "type": ["string", "null"],
                    "enum": ["A", "B", "a", "b", None],
                    "description": "兼容旧字段。会被映射为 case_class。",
                },
                "top_k": {"type": ["integer", "string", "null"], "default": 8, "minimum": 1, "maximum": 50},
                "include_review": {"type": ["boolean", "string", "null"], "default": False},
                "include_formula_score": {"type": ["boolean", "string", "null"], "default": False},
                "include_evidence": {"type": ["boolean", "string", "null"], "default": True},
                "include_vector_text": {"type": ["boolean", "string", "null"], "default": False},
                "min_score": {"type": ["number", "string", "null"], "minimum": 0.0, "maximum": 1.0},
                "similarity_threshold": {
                    "type": ["number", "string", "null"],
                    "description": "兼容旧字段。会被映射为 min_score。",
                },
                "threshold": {
                    "type": ["number", "string", "null"],
                    "description": "兼容旧字段。会被映射为 min_score。",
                },
                "rag_base_url": {"type": ["string", "null"]},
                "ab_case_base_url": {"type": ["string", "null"]},
            },
            "additionalProperties": True,
        }

    def get_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "query_text": {"type": "string"},
                "collection": {"type": "string"},
                "results": {"type": "array"},
                "total": {"type": "integer"},
                "returned": {"type": "integer"},
                "top_score": {"type": "number"},
                "case_context_text": {
                    "type": "string",
                    "description": "面向 text_generate_tool 的 A/B 案例压缩上下文。",
                },
                "low_relevance": {"type": "boolean"},
                "top_similarity": {"type": "number"},
                "min_score": {"type": ["number", "null"]},
                "threshold": {"type": "number"},
            },
            "required": ["query", "query_text", "results", "total", "returned", "top_score", "case_context_text"],
            "additionalProperties": True,
        }

    def get_routing_capability(self) -> dict[str, Any]:
        capability = super().get_routing_capability()
        capability["supported_task_types"] = ["ab_case_search"]
        capability["default_task_type"] = "ab_case_search"
        capability["supported_tags"] = list(self.tags)
        return capability

    async def _arun(self, payload: dict[str, Any], context: ContextStore | None = None) -> ToolResult:
        query = self._resolve_query(payload=payload, context=context)
        event_text = self._optional_str(payload.get("event_text"))
        reason_text = self._optional_str(payload.get("reason_text"))
        if not query and not event_text and not reason_text:
            return self.build_result(
                success=False,
                error="ab_case_search_tool requires query, event_text, or reason_text",
                metadata={"rag_route": AB_CASE_SEARCH_ROUTE, "payload": payload},
            )

        base_url = self._resolve_base_url(payload)
        if not base_url:
            return self.build_result(
                success=False,
                error="AB_CASE_RAG_BASE_URL or RAG_BASE_URL is not configured",
                metadata={"rag_route": AB_CASE_SEARCH_ROUTE, "payload": payload},
            )

        try:
            request_body = self._build_request_body(
                payload=payload,
                query=query,
                event_text=event_text,
                reason_text=reason_text,
            )
            if context is not None:
                runtime_progress(
                    step="ab_case_search:request",
                    status="prepared",
                    detail=json.dumps(
                        {
                            "query": request_body.get("query"),
                            "top_k": request_body.get("top_k"),
                            "case_class": request_body.get("case_class"),
                            "min_score": request_body.get("min_score"),
                            "event_text_present": bool(event_text),
                            "reason_text_present": bool(reason_text),
                        },
                        ensure_ascii=False,
                    ),
                    request_id=context.runtime.request_id,
                    session_id=context.runtime.session_id,
                )
        except Exception as exc:
            return self.build_result(
                success=False,
                error=f"invalid A/B case search parameters: {exc}",
                metadata={
                    "rag_route": AB_CASE_SEARCH_ROUTE,
                    "payload": payload,
                    "exception_type": type(exc).__name__,
                },
            )

        try:
            async with httpx.AsyncClient(timeout=self.resolve_timeout(payload)) as client:
                response = await client.post(f"{base_url}{AB_CASE_SEARCH_ROUTE}", json=request_body)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            return self.build_result(
                success=False,
                error=f"A/B case search request failed: {exc}",
                metadata={
                    "rag_route": AB_CASE_SEARCH_ROUTE,
                    "rag_base_url": base_url,
                    "request_body": request_body,
                    "exception_type": type(exc).__name__,
                },
            )

        output = self._build_output(
            data=data,
            fallback_query=query or event_text or reason_text or "",
            request_min_score=request_body.get("min_score"),
        )
        return self.build_result(
            success=True,
            output=output,
            metadata={
                "rag_route": AB_CASE_SEARCH_ROUTE,
                "rag_base_url": base_url,
                "request_body": request_body,
                "total": output["total"],
                "returned": output["returned"],
                "top_score": output["top_score"],
                "top_similarity": output["top_similarity"],
                "min_score": output["min_score"],
                "threshold": output["threshold"],
                "low_relevance": output["low_relevance"],
            },
        )

    def _resolve_query(self, *, payload: dict[str, Any], context: ContextStore | None) -> str:
        query = self._first_non_empty(payload.get("query"), payload.get("prompt"), payload.get("input"))
        if query:
            return query
        if context is not None:
            return str(context.runtime.user_input or "").strip()
        return ""

    def _resolve_base_url(self, payload: dict[str, Any]) -> str:
        return str(
            payload.get("ab_case_base_url")
            or payload.get("rag_base_url")
            or os.getenv("AB_CASE_RAG_BASE_URL", "")
            or os.getenv("RAG_BASE_URL", "")
        ).strip().rstrip("/")

    def _build_request_body(
        self,
        *,
        payload: dict[str, Any],
        query: str,
        event_text: str | None,
        reason_text: str | None,
    ) -> dict[str, Any]:
        request_body: dict[str, Any] = {
            "top_k": self._resolve_top_k(payload),
        }
        if query:
            request_body["query"] = query
        if event_text:
            request_body["event_text"] = event_text
        if reason_text:
            request_body["reason_text"] = reason_text

        case_class = self._resolve_case_class(payload)
        if case_class:
            request_body["case_class"] = case_class

        min_score = self._resolve_min_score(payload)
        if min_score is not None:
            request_body["min_score"] = min_score

        for field_name in _BOOLEAN_REQUEST_FIELDS:
            value = self._optional_bool(payload.get(field_name), field_name=field_name)
            if value is not None:
                request_body[field_name] = value
        return request_body

    def _build_output(self, *, data: Any, fallback_query: str, request_min_score: Any = None) -> dict[str, Any]:
        response = data if isinstance(data, dict) else {}
        results = self._extract_results(response)
        min_score = self._resolve_response_min_score(response=response, request_min_score=request_min_score)
        threshold = min_score if min_score is not None else 0.0
        top_similarity = self._resolve_top_similarity(response=response, results=results)
        top_score = top_similarity
        low_relevance = self._resolve_low_relevance(
            response=response,
            results=results,
            top_similarity=top_similarity,
            threshold=threshold,
        )
        query_text = str(response.get("query_text") or response.get("query") or fallback_query).strip()
        total = self._safe_non_negative_int(response.get("total"), default=len(results))
        returned = self._safe_non_negative_int(response.get("returned"), default=len(results))
        collection = str(response.get("collection") or "").strip()
        case_context_text = self._resolve_case_context_text(
            response=response,
            query=query_text,
            collection=collection,
            results=results,
            total=total,
            returned=returned,
            top_similarity=top_similarity,
            threshold=threshold,
            low_relevance=low_relevance,
        )
        return {
            "collection": collection,
            "query_text": query_text,
            "query": query_text,
            "results": results,
            "total": total,
            "returned": returned,
            "top_score": top_score,
            "case_context_text": case_context_text,
            "low_relevance": low_relevance,
            "top_similarity": top_similarity,
            "min_score": min_score,
            "threshold": threshold,
        }

    def _extract_results(self, response: dict[str, Any]) -> list[Any]:
        raw_results = response.get("results")
        if isinstance(raw_results, list):
            return raw_results
        raw_cases = response.get("cases")
        if isinstance(raw_cases, list):
            return raw_cases
        return []

    def _resolve_top_similarity(self, *, response: dict[str, Any], results: list[Any]) -> float:
        explicit = self._optional_float(response.get("top_similarity"))
        if explicit is None:
            explicit = self._optional_float(response.get("top_score"))
        if explicit is not None:
            return explicit
        values = [
            value
            for value in (self._result_similarity(item) if isinstance(item, dict) else None for item in results)
            if value is not None
        ]
        return max(values) if values else 0.0

    def _result_similarity(self, item: dict[str, Any]) -> float | None:
        for key in ("similarity", "top_similarity", "search_score"):
            value = self._optional_float(item.get(key))
            if value is not None:
                return value
        score = self._optional_float(item.get("score"))
        if score is not None and 0.0 <= score <= 1.0:
            return score
        return None

    def _resolve_low_relevance(
        self,
        *,
        response: dict[str, Any],
        results: list[Any],
        top_similarity: float,
        threshold: float,
    ) -> bool:
        raw_value = response.get("low_relevance")
        if isinstance(raw_value, bool):
            return raw_value
        if threshold > 0:
            return top_similarity < threshold
        return not results

    def _resolve_case_context_text(
        self,
        *,
        response: dict[str, Any],
        query: str,
        collection: str,
        results: list[Any],
        total: int,
        returned: int,
        top_similarity: float,
        threshold: float,
        low_relevance: bool,
    ) -> str:
        for key in ("case_context_text", "joined_context", "context_text"):
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return self._build_case_context_text(
            query=query,
            collection=collection,
            results=results,
            total=total,
            returned=returned,
            top_similarity=top_similarity,
            threshold=threshold,
            low_relevance=low_relevance,
        )

    def _build_case_context_text(
        self,
        *,
        query: str,
        collection: str,
        results: list[Any],
        total: int,
        returned: int,
        top_similarity: float,
        threshold: float,
        low_relevance: bool,
    ) -> str:
        lines = [
            "A/B 案例检索结果（A案例=好事奖励，B案例=坏事惩罚）",
            f"Milvus集合: {collection or '未提供'}",
            f"查询文本: {query or '未提供'}",
            f"命中数量: {total}",
            f"返回数量: {returned}",
            f"最高相似度: {top_similarity:.4f}",
            f"最低相似度过滤: {threshold:.4f}" if threshold > 0 else "最低相似度过滤: 未设置",
            f"低相关: {'是' if low_relevance else '否'}",
        ]
        if not results:
            lines.append("未检索到相似 A/B 案例。")
            return "\n".join(lines)

        for index, item in enumerate(results[: self.context_max_results], start=1):
            lines.append("")
            lines.append(f"案例 {index}:")
            if isinstance(item, dict):
                for key in self._ordered_display_keys(item):
                    value = item.get(key)
                    if value is None or value == "":
                        continue
                    lines.append(f"- {key}: {self._clip_text(value)}")
            else:
                lines.append(f"- value: {self._clip_text(item)}")

        truncated_count = max(0, len(results) - self.context_max_results)
        if truncated_count:
            lines.append("")
            lines.append(f"另有 {truncated_count} 条结果未写入压缩上下文，完整字段仍保留在 results。")
        return "\n".join(lines)

    def _ordered_display_keys(self, item: dict[str, Any]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for key in _DISPLAY_FIELD_PRIORITY:
            if key in item and key not in seen:
                ordered.append(key)
                seen.add(key)
        for key in item:
            if key not in seen:
                ordered.append(key)
                seen.add(key)
        return ordered

    def _resolve_top_k(self, payload: dict[str, Any]) -> int:
        top_k = self._safe_positive_int(payload.get("top_k"), default=self.default_top_k)
        return max(1, min(top_k, self.max_top_k))

    def _resolve_case_class(self, payload: dict[str, Any]) -> str | None:
        raw_value = self._first_non_empty(payload.get("case_class"), payload.get("case_type"))
        if not raw_value:
            return None
        normalized = raw_value.strip().upper()
        if normalized.startswith("A"):
            return "A"
        if normalized.startswith("B"):
            return "B"
        if "奖励" in raw_value or "好事" in raw_value or "正向" in raw_value:
            return "A"
        if "惩罚" in raw_value or "坏事" in raw_value or "处罚" in raw_value or "负向" in raw_value:
            return "B"
        raise ValueError("case_class must be A or B")

    def _resolve_min_score(self, payload: dict[str, Any]) -> float | None:
        for key in ("min_score", "similarity_threshold", "threshold"):
            value = self._optional_float(payload.get(key))
            if value is None:
                continue
            if not 0.0 <= value <= 1.0:
                raise ValueError("min_score must be between 0.0 and 1.0")
            return value
        return None

    def _resolve_response_min_score(self, *, response: dict[str, Any], request_min_score: Any) -> float | None:
        for value in (
            response.get("min_score"),
            response.get("threshold"),
            response.get("similarity_threshold"),
            request_min_score,
        ):
            parsed = self._optional_float(value)
            if parsed is not None and 0.0 <= parsed <= 1.0:
                return parsed
        return None

    def _first_non_empty(self, *values: Any) -> str:
        for value in values:
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _optional_str(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _safe_positive_int(self, value: Any, *, default: int) -> int:
        if value is None or value == "":
            return default
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed > 0 else default

    def _safe_non_negative_int(self, value: Any, *, default: int) -> int:
        if value is None or value == "":
            return default
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed >= 0 else default

    def _safe_float(self, value: Any, *, default: float) -> float:
        parsed = self._optional_float(value)
        return parsed if parsed is not None else default

    def _optional_float(self, value: Any) -> float | None:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _optional_bool(self, value: Any, *, field_name: str) -> bool | None:
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "y", "on", "是", "包含"}:
            return True
        if text in {"false", "0", "no", "n", "off", "否", "不包含"}:
            return False
        raise ValueError(f"{field_name} must be a boolean")

    def _clip_text(self, value: Any) -> str:
        text = str(value)
        if len(text) <= self.context_field_max_chars:
            return text
        return f"{text[: self.context_field_max_chars]}... [truncated {len(text) - self.context_field_max_chars} chars]"
