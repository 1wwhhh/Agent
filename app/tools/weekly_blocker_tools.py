from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.context import ContextStore
from app.schemas.llm import LLMFunctionSchema, LLMMessage, LLMRequest
from app.schemas.tool import ToolResult
from app.tools.base import BaseTool
from app.tools.function_calling import FunctionCallingAdapter
from app.utils import runtime_progress


WEEKLY_BLOCKER_TOOL_TAGS = ["llm", "business", "weekly_report", "weekly_blocker"]
CURRENT_BLOCKER_CLASSIFICATIONS = {"current_blocker", "mixed_current_blocker"}
TRACE_BLOCKER_CLASSIFICATIONS = {
    "no_current_blocker",
    "historical_or_resolved",
    "ambiguous",
    "empty",
}
ALL_BLOCKER_CLASSIFICATIONS = CURRENT_BLOCKER_CLASSIFICATIONS | TRACE_BLOCKER_CLASSIFICATIONS
HISTORICAL_TRACE_STATUSES = {"resolved", "likely_continuing", "insufficient_evidence", "not_a_blocker"}


class WeeklyBlockerClassificationItem(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    user_name: str = Field(..., description="Person name from the input weekly report row.")
    department: str | None = Field(default=None, description="Department from the input weekly report row.")
    raw_risk_and_help: str = Field(default="", description="Original target-week blocker/risk/help text.")
    classification: str = Field(
        default="ambiguous",
        description="current_blocker | no_current_blocker | mixed_current_blocker | historical_or_resolved | ambiguous | empty",
    )
    has_effective_current_blocker: bool = Field(default=False)
    effective_blocker_text: str = Field(default="")
    needs_trace: bool = Field(default=True)
    reason: str = Field(default="")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class WeeklyBlockerClassificationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    items: list[WeeklyBlockerClassificationItem] = Field(default_factory=list)


class HistoricalBlockerTraceJudgementItem(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    candidate_id: str = Field(..., description="Historical blocker candidate id from the input.")
    status: str = Field(default="insufficient_evidence", description="resolved | likely_continuing | insufficient_evidence | not_a_blocker")
    reason: str = Field(default="")
    evidence: str = Field(default="")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class HistoricalBlockerTraceJudgementOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    historical_blocker_followups: list[HistoricalBlockerTraceJudgementItem] = Field(default_factory=list)


class _WeeklyBlockerLLMTool(BaseTool):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    client: Any = Field(..., exclude=True)
    function_adapter: FunctionCallingAdapter = Field(default_factory=FunctionCallingAdapter)
    default_model_name: str | None = Field(default=None)
    tags: list[str] = Field(default_factory=lambda: list(WEEKLY_BLOCKER_TOOL_TAGS))
    timeout: int = Field(default=120, gt=0)

    def get_routing_capability(self) -> dict[str, Any]:
        capability = super().get_routing_capability()
        capability["supported_task_types"] = [self.name]
        capability["default_task_type"] = self.name
        capability["supported_tags"] = list(self.tags)
        capability["max_concurrency"] = _env_int("WEEKLY_BLOCKER_LLM_MAX_CONCURRENCY", 4)
        return capability

    def _build_trace_id(self, *, context: ContextStore | None, suffix: str) -> str:
        request_id = context.runtime.request_id if context is not None else "no_request"
        return f"{request_id}:{self.name}:{suffix}:{uuid4().hex}"

    def _request_model_name(self, payload: Mapping[str, Any]) -> str | None:
        return str(payload.get("model_name") or self.default_model_name or "").strip() or None

    def _request_max_tokens(self, payload: Mapping[str, Any], key: str, default: int) -> int | None:
        value = _optional_positive_int(payload.get(key))
        return value or _env_int(key.upper(), default)


class ClassifyWeeklyBlockersTool(_WeeklyBlockerLLMTool):
    name: str = Field(default="classify_weekly_blockers")
    description: str = Field(default="语义判断目标周员工自填卡点是否包含当前有效卡点，并决定是否需要计划追溯。")

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "weekly_reports_output_key": {
                    "type": ["string", "null"],
                    "default": "weekly_reports",
                    "description": "上游 query_weekly_reports 的 output_key。",
                },
                "limit": {"type": ["integer", "null"], "default": 500},
                "model_name": {"type": ["string", "null"]},
            },
            "additionalProperties": True,
        }

    def get_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "items": {"type": "array"},
                "count": {"type": "integer"},
                "source_output_key": {"type": ["string", "null"]},
                "classification_summary": {"type": "object"},
                "fallback_used": {"type": "boolean"},
            },
            "required": ["items", "count", "classification_summary", "fallback_used"],
            "additionalProperties": True,
        }

    async def _arun(self, payload: dict[str, Any], context: ContextStore | None = None) -> ToolResult:
        source_output_key = str(payload.get("weekly_reports_output_key") or "weekly_reports").strip()
        try:
            people = _collect_weekly_report_people(
                _resolve_task_output(
                    payload=payload,
                    context=context,
                    output_key=source_output_key,
                    direct_keys=("weekly_reports", "items"),
                )
            )
            people = people[: _bounded_limit(payload.get("limit"), default=500, maximum=2000)]
            if not people:
                output = _classification_tool_output(
                    items=[],
                    source_output_key=source_output_key,
                    fallback_used=False,
                    fallback_reason=None,
                )
                return self.build_result(success=True, output=output, metadata={"count": 0, "fallback_used": False})

            try:
                output_model = await self._invoke_classifier(payload=payload, context=context, people=people)
                items = _normalize_classification_items(people=people, llm_items=output_model.items)
                fallback_used = False
                fallback_reason = None
            except Exception as exc:
                items = [_fallback_classification_item(person, reason=f"LLM 分类失败，保守进入追溯：{exc}") for person in people]
                fallback_used = True
                fallback_reason = str(exc)

            output = _classification_tool_output(
                items=items,
                source_output_key=source_output_key,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
            )
            runtime_progress(
                step="classify_weekly_blockers:语义分类",
                status="完成" if not fallback_used else "LLM 失败，使用保守兜底",
                detail=json.dumps(output["classification_summary"], ensure_ascii=False),
                request_id=context.runtime.request_id if context is not None else None,
                session_id=context.runtime.session_id if context is not None else None,
            )
            return self.build_result(
                success=True,
                output=output,
                metadata={"count": len(items), "fallback_used": fallback_used},
            )
        except Exception as exc:
            return self.build_result(
                success=False,
                error=f"classify_weekly_blockers failed: {exc}",
                metadata={"exception_type": type(exc).__name__},
            )

    async def _invoke_classifier(
        self,
        *,
        payload: Mapping[str, Any],
        context: ContextStore | None,
        people: list[dict[str, Any]],
    ) -> WeeklyBlockerClassificationOutput:
        compact_people = [
            {
                "user_name": person["user_name"],
                "department": person.get("department"),
                "raw_risk_and_help": _clip_text(person.get("raw_risk_and_help"), 1200),
                "report_dates": person.get("report_dates", []),
            }
            for person in people
        ]
        system_prompt = (
            "你是周报卡点字段的语义分类器，只判断目标周员工自填卡点文本是否包含当前仍有效的卡点/风险/求助。\n"
            "分类只能使用：current_blocker、no_current_blocker、mixed_current_blocker、historical_or_resolved、ambiguous、empty。\n"
            "不是非空就算卡点：如果只是“无/暂无/没有卡点、无风险、无需协助”等，判 no_current_blocker。\n"
            "也不是出现“无卡点”就一定没有卡点：如果同一句同时写了具体待协助、阻塞、未到货、待确认、无法推进等，判 mixed_current_blocker，并抽取具体卡点。\n"
            "如果文本只描述历史问题已解决、已完成、已闭环，判 historical_or_resolved。\n"
            "current_blocker 和 mixed_current_blocker 的 needs_trace 必须为 false；empty、no_current_blocker、historical_or_resolved、ambiguous 的 needs_trace 必须为 true。\n"
            "effective_blocker_text 只保留当前有效卡点，不要包含“暂无卡点”等否定套话。必须通过函数返回结构化结果。"
        )
        user_prompt = (
            "请分类以下目标周员工自填卡点文本。\n"
            "Input JSON:\n"
            f"{json.dumps({'people': compact_people}, ensure_ascii=False, indent=2)}"
        )
        request = LLMRequest(
            prompt=user_prompt,
            system_prompt=system_prompt,
            messages=[
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ],
            model_name=self._request_model_name(payload),
            temperature=0.0,
            max_tokens=self._request_max_tokens(payload, "classification_max_tokens", 6000),
            timeout_seconds=_env_int("WEEKLY_BLOCKER_CLASSIFY_TIMEOUT_SECONDS", 120),
            request_id=context.runtime.request_id if context is not None else None,
            session_id=context.runtime.session_id if context is not None else None,
            trace_id=self._build_trace_id(context=context, suffix="classification"),
            prompt_name="weekly_blocker_classification_prompt",
            prompt_version="v1",
            response_schema_name=WeeklyBlockerClassificationOutput.__name__,
            response_schema_version="v1",
            max_validation_retries=1,
            metadata={
                "tool_name": self.name,
                "operation": "weekly_blocker_classification",
                "person_count": len(people),
            },
        )
        function_schema = LLMFunctionSchema(
            name="emit_weekly_blocker_classification",
            description="Return semantic classifications for target-week weekly report blocker texts.",
            parameters_schema=WeeklyBlockerClassificationOutput.model_json_schema(),
            schema_name=WeeklyBlockerClassificationOutput.__name__,
            schema_version="v1",
        )
        structured_result = await self.function_adapter.invoke_structured(
            client=self.client,
            request=request,
            function_schema=function_schema,
            output_model=WeeklyBlockerClassificationOutput,
        )
        return structured_result.output


class JudgeWeeklyBlockerTraceTool(_WeeklyBlockerLLMTool):
    name: str = Field(default="judge_weekly_blocker_trace")
    description: str = Field(default="判断历史卡点在后续周报中是否解决，并生成最终周报卡点压缩上下文。")

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "weekly_blocker_classification_output_key": {
                    "type": ["string", "null"],
                    "default": "weekly_blocker_classification",
                },
                "weekly_plan_comparison_output_key": {
                    "type": ["string", "null"],
                    "default": "weekly_plan_comparison",
                },
                "model_name": {"type": ["string", "null"]},
            },
            "additionalProperties": True,
        }

    def get_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "historical_blocker_followups": {"type": "array"},
                "weekly_blocker_context_text": {"type": "string"},
                "fallback_used": {"type": "boolean"},
            },
            "required": ["historical_blocker_followups", "weekly_blocker_context_text", "fallback_used"],
            "additionalProperties": True,
        }

    async def _arun(self, payload: dict[str, Any], context: ContextStore | None = None) -> ToolResult:
        classification_key = str(payload.get("weekly_blocker_classification_output_key") or "weekly_blocker_classification").strip()
        comparison_key = str(payload.get("weekly_plan_comparison_output_key") or "weekly_plan_comparison").strip()
        try:
            classification_output = _resolve_required_context_output(context=context, output_key=classification_key)
            comparison_output = _resolve_required_context_output(context=context, output_key=comparison_key)
            candidates = _historical_candidates_from_comparison(comparison_output)
            if candidates:
                try:
                    llm_output = await self._invoke_historical_trace_judge(
                        payload=payload,
                        context=context,
                        candidates=candidates,
                    )
                    followups = _normalize_historical_judgements(candidates=candidates, llm_items=llm_output.historical_blocker_followups)
                    fallback_used = False
                    fallback_reason = None
                except Exception as exc:
                    followups = [_fallback_historical_judgement(candidate, reason=f"LLM 追溯判断失败：{exc}") for candidate in candidates]
                    fallback_used = True
                    fallback_reason = str(exc)
            else:
                followups = []
                fallback_used = False
                fallback_reason = None

            context_text = _format_final_weekly_blocker_context(
                classification_output=classification_output,
                comparison_output=comparison_output,
                historical_followups=followups,
            )
            output = {
                "historical_blocker_followups": followups,
                "weekly_blocker_context_text": context_text,
                "fallback_used": fallback_used,
                "fallback_reason": fallback_reason,
                "source_output_keys": {
                    "classification": classification_key,
                    "comparison": comparison_key,
                },
            }
            runtime_progress(
                step="judge_weekly_blocker_trace:历史卡点追溯判断",
                status="完成" if not fallback_used else "LLM 失败，使用保守兜底",
                detail=json.dumps(
                    {
                        "historical_candidate_count": len(candidates),
                        "historical_followup_count": len(followups),
                        "context_chars": len(context_text),
                    },
                    ensure_ascii=False,
                ),
                request_id=context.runtime.request_id if context is not None else None,
                session_id=context.runtime.session_id if context is not None else None,
            )
            return self.build_result(
                success=True,
                output=output,
                metadata={
                    "historical_candidate_count": len(candidates),
                    "historical_followup_count": len(followups),
                    "fallback_used": fallback_used,
                },
            )
        except Exception as exc:
            return self.build_result(
                success=False,
                error=f"judge_weekly_blocker_trace failed: {exc}",
                metadata={"exception_type": type(exc).__name__},
            )

    async def _invoke_historical_trace_judge(
        self,
        *,
        payload: Mapping[str, Any],
        context: ContextStore | None,
        candidates: list[dict[str, Any]],
    ) -> HistoricalBlockerTraceJudgementOutput:
        compact_candidates = [
            {
                "candidate_id": candidate["candidate_id"],
                "user_name": candidate.get("user_name"),
                "department": candidate.get("department"),
                "report_date": candidate.get("report_date"),
                "raw_blocker_text": _clip_text(candidate.get("raw_risk_and_help"), 1000),
                "later_done_items": [
                    {
                        "report_date": item.get("report_date"),
                        "item_type": item.get("item_type"),
                        "done_text": _clip_text(item.get("done_text"), 260),
                    }
                    for item in candidate.get("followup_done_items", [])
                    if isinstance(item, Mapping)
                ][:8],
            }
            for candidate in candidates
        ]
        system_prompt = (
            "你是周报历史卡点追溯判断器。\n"
            "只依据输入 JSON 中历史卡点原文和后续完成记录判断，不得引入外部知识，不得编造。\n"
            "status 只能是：resolved、likely_continuing、insufficient_evidence、not_a_blocker。\n"
            "如果历史文本本身只是无卡点表述，判 not_a_blocker。\n"
            "如果后续完成记录明确解决或推进到闭环，判 resolved。\n"
            "如果后续仍显示等待、未完成、继续受阻，判 likely_continuing。\n"
            "如果无法从后续完成记录判断，判 insufficient_evidence。\n"
            "必须通过函数返回结构化结果。"
        )
        user_prompt = (
            "请判断以下历史卡点是否在之后完成或仍持续。\n"
            "Input JSON:\n"
            f"{json.dumps({'candidates': compact_candidates}, ensure_ascii=False, indent=2)}"
        )
        request = LLMRequest(
            prompt=user_prompt,
            system_prompt=system_prompt,
            messages=[
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ],
            model_name=self._request_model_name(payload),
            temperature=0.0,
            max_tokens=self._request_max_tokens(payload, "trace_judge_max_tokens", 6000),
            timeout_seconds=_env_int("WEEKLY_BLOCKER_TRACE_JUDGE_TIMEOUT_SECONDS", 120),
            request_id=context.runtime.request_id if context is not None else None,
            session_id=context.runtime.session_id if context is not None else None,
            trace_id=self._build_trace_id(context=context, suffix="historical_trace"),
            prompt_name="weekly_blocker_historical_trace_judgement_prompt",
            prompt_version="v1",
            response_schema_name=HistoricalBlockerTraceJudgementOutput.__name__,
            response_schema_version="v1",
            max_validation_retries=1,
            metadata={
                "tool_name": self.name,
                "operation": "weekly_blocker_historical_trace_judgement",
                "candidate_count": len(candidates),
            },
        )
        function_schema = LLMFunctionSchema(
            name="emit_weekly_blocker_historical_trace_judgement",
            description="Return semantic follow-up judgements for historical weekly blocker candidates.",
            parameters_schema=HistoricalBlockerTraceJudgementOutput.model_json_schema(),
            schema_name=HistoricalBlockerTraceJudgementOutput.__name__,
            schema_version="v1",
        )
        structured_result = await self.function_adapter.invoke_structured(
            client=self.client,
            request=request,
            function_schema=function_schema,
            output_model=HistoricalBlockerTraceJudgementOutput,
        )
        return structured_result.output


def _resolve_task_output(
    *,
    payload: Mapping[str, Any],
    context: ContextStore | None,
    output_key: str,
    direct_keys: tuple[str, ...],
) -> Any:
    if context is not None and output_key in context.task_results:
        return context.task_results[output_key]
    for key in direct_keys:
        if key in payload:
            value = payload[key]
            if key == "items" and isinstance(value, list):
                return {"items": value}
            return value
    if context is None:
        raise ValueError(f"runtime context is required when {output_key!r} is not supplied directly")
    raise ValueError(f"upstream output not found: {output_key}")


def _resolve_required_context_output(*, context: ContextStore | None, output_key: str) -> Any:
    if context is None:
        raise ValueError(f"{output_key} requires runtime context")
    if output_key not in context.task_results:
        raise ValueError(f"upstream output not found: {output_key}")
    return context.task_results[output_key]


def _collect_weekly_report_people(output: Any) -> list[dict[str, Any]]:
    raw_items = output.get("items") if isinstance(output, Mapping) else output
    if not isinstance(raw_items, list):
        return []
    grouped: dict[tuple[str, str | None], dict[str, Any]] = {}
    for item in raw_items:
        if not isinstance(item, Mapping):
            continue
        user_name = _optional_str(item.get("user_name"))
        if not user_name:
            continue
        department = _optional_str(item.get("department"))
        key = (user_name, department)
        group = grouped.setdefault(
            key,
            {
                "user_name": user_name,
                "department": department,
                "raw_risk_and_help_values": [],
                "report_dates": [],
                "source_row_count": 0,
            },
        )
        group["source_row_count"] += 1
        raw_text = _optional_str(item.get("risk_and_help"))
        if raw_text and raw_text not in group["raw_risk_and_help_values"]:
            group["raw_risk_and_help_values"].append(raw_text)
        report_date = _optional_str(item.get("report_date"))
        if report_date and report_date not in group["report_dates"]:
            group["report_dates"].append(report_date)

    people: list[dict[str, Any]] = []
    for group in grouped.values():
        raw_values = group.pop("raw_risk_and_help_values")
        group["raw_risk_and_help"] = "\n".join(raw_values).strip()
        people.append(group)
    return people


def _classification_tool_output(
    *,
    items: list[dict[str, Any]],
    source_output_key: str | None,
    fallback_used: bool,
    fallback_reason: str | None,
) -> dict[str, Any]:
    summary = {
        "total_people": len(items),
        "current_blocker_count": sum(1 for item in items if item.get("classification") == "current_blocker"),
        "mixed_current_blocker_count": sum(1 for item in items if item.get("classification") == "mixed_current_blocker"),
        "needs_trace_count": sum(1 for item in items if item.get("needs_trace")),
        "direct_current_blocker_count": sum(1 for item in items if item.get("has_effective_current_blocker")),
        "empty_count": sum(1 for item in items if item.get("classification") == "empty"),
        "ambiguous_count": sum(1 for item in items if item.get("classification") == "ambiguous"),
    }
    return {
        "items": items,
        "count": len(items),
        "source_output_key": source_output_key,
        "classification_summary": summary,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
    }


def _normalize_classification_items(
    *,
    people: list[dict[str, Any]],
    llm_items: list[WeeklyBlockerClassificationItem],
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str | None], WeeklyBlockerClassificationItem] = {
        (_optional_str(item.user_name) or "", _optional_str(item.department)): item for item in llm_items
    }
    normalized: list[dict[str, Any]] = []
    for person in people:
        key = (_optional_str(person.get("user_name")) or "", _optional_str(person.get("department")))
        llm_item = by_key.get(key)
        if llm_item is None:
            normalized.append(_fallback_classification_item(person, reason="LLM 未返回该人员分类，保守进入追溯。"))
            continue
        classification = str(llm_item.classification or "ambiguous").strip()
        if classification not in ALL_BLOCKER_CLASSIFICATIONS:
            classification = "ambiguous"
        raw_text = str(person.get("raw_risk_and_help") or llm_item.raw_risk_and_help or "").strip()
        effective_text = str(llm_item.effective_blocker_text or "").strip()
        if classification in CURRENT_BLOCKER_CLASSIFICATIONS:
            if not effective_text:
                effective_text = raw_text
            has_effective = bool(effective_text)
            needs_trace = not has_effective
            if not has_effective:
                classification = "ambiguous"
                needs_trace = True
        else:
            has_effective = False
            effective_text = ""
            needs_trace = True
        normalized.append(
            {
                "user_name": person["user_name"],
                "department": person.get("department"),
                "raw_risk_and_help": raw_text,
                "classification": classification,
                "has_effective_current_blocker": has_effective,
                "effective_blocker_text": effective_text,
                "needs_trace": needs_trace,
                "reason": str(llm_item.reason or "").strip(),
                "confidence": float(llm_item.confidence or 0.0),
                "source_row_count": person.get("source_row_count", 1),
                "report_dates": person.get("report_dates", []),
            }
        )
    return normalized


def _fallback_classification_item(person: Mapping[str, Any], *, reason: str) -> dict[str, Any]:
    raw_text = str(person.get("raw_risk_and_help") or "").strip()
    return {
        "user_name": person.get("user_name"),
        "department": person.get("department"),
        "raw_risk_and_help": raw_text,
        "classification": "empty" if not raw_text else "ambiguous",
        "has_effective_current_blocker": False,
        "effective_blocker_text": "",
        "needs_trace": True,
        "reason": reason,
        "confidence": 0.0,
        "source_row_count": person.get("source_row_count", 1),
        "report_dates": person.get("report_dates", []),
    }


def _historical_candidates_from_comparison(comparison_output: Any) -> list[dict[str, Any]]:
    if not isinstance(comparison_output, Mapping):
        return []
    candidates = comparison_output.get("historical_blocker_candidates")
    if not isinstance(candidates, list):
        return []
    return [dict(candidate) for candidate in candidates if isinstance(candidate, Mapping)]


def _normalize_historical_judgements(
    *,
    candidates: list[dict[str, Any]],
    llm_items: list[HistoricalBlockerTraceJudgementItem],
) -> list[dict[str, Any]]:
    by_id = {str(item.candidate_id): item for item in llm_items}
    normalized: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id") or "")
        llm_item = by_id.get(candidate_id)
        if llm_item is None:
            normalized.append(_fallback_historical_judgement(candidate, reason="LLM 未返回该历史卡点判断。"))
            continue
        status = str(llm_item.status or "insufficient_evidence").strip()
        if status not in HISTORICAL_TRACE_STATUSES:
            status = "insufficient_evidence"
        normalized.append(
            {
                "candidate_id": candidate_id,
                "user_name": candidate.get("user_name"),
                "department": candidate.get("department"),
                "report_date": candidate.get("report_date"),
                "raw_risk_and_help": candidate.get("raw_risk_and_help"),
                "status": status,
                "reason": str(llm_item.reason or "").strip(),
                "evidence": str(llm_item.evidence or "").strip(),
                "confidence": float(llm_item.confidence or 0.0),
                "followup_done_items": candidate.get("followup_done_items", []),
            }
        )
    return normalized


def _fallback_historical_judgement(candidate: Mapping[str, Any], *, reason: str) -> dict[str, Any]:
    return {
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "user_name": candidate.get("user_name"),
        "department": candidate.get("department"),
        "report_date": candidate.get("report_date"),
        "raw_risk_and_help": candidate.get("raw_risk_and_help"),
        "status": "insufficient_evidence",
        "reason": reason,
        "evidence": "",
        "confidence": 0.0,
        "followup_done_items": candidate.get("followup_done_items", []),
    }


def _format_final_weekly_blocker_context(
    *,
    classification_output: Any,
    comparison_output: Any,
    historical_followups: list[dict[str, Any]],
) -> str:
    classifications = _classification_items_from_output(classification_output)
    comparison = comparison_output if isinstance(comparison_output, Mapping) else {}
    date_range = comparison.get("query_date_range", {}) if isinstance(comparison, Mapping) else {}
    trace_scope = comparison.get("trace_scope", {}) if isinstance(comparison, Mapping) else {}
    target_start = date_range.get("this_week_start", "") if isinstance(date_range, Mapping) else ""
    target_end = date_range.get("this_week_end", "") if isinstance(date_range, Mapping) else ""
    windows = date_range.get("trace_windows", []) if isinstance(date_range, Mapping) else []
    lines = [
        "周报卡点语义判断上下文",
        f"- 目标周: {target_start} 至 {target_end}",
        "- 判断原则: 员工明确写出的当前有效卡点优先；未填写、无当前卡点、历史已解决或语义不明确的人员才使用计划和历史卡点追溯证据。",
    ]
    if isinstance(windows, list) and windows:
        window_text = "；".join(
            f"窗口{item.get('window_index')}: {item.get('source_plan_week_start')} 至 {item.get('source_plan_week_end')} -> {item.get('followup_start')} 至 {item.get('followup_end')}"
            for item in windows
            if isinstance(item, Mapping)
        )
        if window_text:
            lines.append(f"- 追溯窗口: {window_text}")
    if isinstance(trace_scope, Mapping):
        traced = trace_scope.get("traced_people", [])
        skipped = trace_scope.get("skipped_people", [])
        lines.append(
            f"- 人员范围: 当前有效员工自填卡点 {len(skipped) if isinstance(skipped, list) else 0} 人；进入追溯 {len(traced) if isinstance(traced, list) else 0} 人。"
        )
    lines.append("")

    followups_by_person = _group_plan_followups_by_person(comparison.get("plan_followups", []))
    historical_by_person = _group_historical_followups_by_person(historical_followups)

    for item in classifications:
        name = item.get("user_name") or "未知人员"
        department = item.get("department") or "未填写部门"
        key = (str(name), _optional_str(department))
        lines.append(f"人员: {name} / {department}")
        if item.get("has_effective_current_blocker"):
            lines.append("目标周员工自填卡点: 有当前有效卡点")
            lines.append(f"卡点内容: {_clip_text(item.get('effective_blocker_text'), 900)}")
            if item.get("classification") == "mixed_current_blocker":
                lines.append("说明: 原文同时包含无卡点表述和具体卡点，已只保留具体卡点。")
        else:
            classification = item.get("classification") or "ambiguous"
            reason = item.get("reason") or ""
            lines.append(f"目标周员工自填卡点: 未确认当前有效卡点（{classification}）")
            if reason:
                lines.append(f"语义判断说明: {_clip_text(reason, 300)}")
            person_followups = followups_by_person.get(key) or []
            if person_followups:
                lines.append("计划追溯证据:")
                for index, followup in enumerate(person_followups[:8], start=1):
                    window = followup.get("trace_window") if isinstance(followup.get("trace_window"), Mapping) else {}
                    window_label = f"窗口{window.get('window_index')}" if window else "追溯窗口"
                    lines.append(f"  {index}. {window_label} 计划: {_clip_text(followup.get('plan_text'), 260) or '无'}")
                    done_items = followup.get("done_items", [])
                    if isinstance(done_items, list) and done_items:
                        for done in done_items[:4]:
                            if isinstance(done, Mapping):
                                lines.append(f"     后续完成: {_clip_text(done.get('done_text'), 220) or '无'}")
                    else:
                        lines.append("     后续完成: 未找到匹配完成记录")
            else:
                lines.append("计划追溯证据: 未找到可用于追溯的计划/完成配对")
        person_historical = historical_by_person.get(key) or []
        if person_historical:
            lines.append("历史卡点追溯:")
            for index, followup in enumerate(person_historical[:6], start=1):
                lines.append(
                    f"  {index}. {followup.get('report_date') or '未知日期'}: {_clip_text(followup.get('raw_risk_and_help'), 360)}"
                )
                lines.append(
                    f"     后续判断: {_historical_status_label(followup.get('status'))}；依据: {_clip_text(followup.get('reason') or followup.get('evidence'), 300) or '无明确依据'}"
                )
        lines.append("")
    return "\n".join(lines).strip()


def _classification_items_from_output(output: Any) -> list[dict[str, Any]]:
    if not isinstance(output, Mapping):
        return []
    items = output.get("items")
    if not isinstance(items, list):
        return []
    return [dict(item) for item in items if isinstance(item, Mapping)]


def _group_plan_followups_by_person(plan_followups: Any) -> dict[tuple[str, str | None], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str | None], list[dict[str, Any]]] = {}
    if not isinstance(plan_followups, list):
        return grouped
    for item in plan_followups:
        if not isinstance(item, Mapping):
            continue
        user_name = _optional_str(item.get("user_name"))
        if not user_name:
            continue
        grouped.setdefault((user_name, _optional_str(item.get("department"))), []).append(dict(item))
    return grouped


def _group_historical_followups_by_person(
    followups: list[dict[str, Any]],
) -> dict[tuple[str, str | None], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str | None], list[dict[str, Any]]] = {}
    for item in followups:
        user_name = _optional_str(item.get("user_name"))
        if not user_name:
            continue
        grouped.setdefault((user_name, _optional_str(item.get("department"))), []).append(item)
    return grouped


def _historical_status_label(value: Any) -> str:
    status = str(value or "").strip()
    return {
        "resolved": "后续已有解决/闭环迹象",
        "likely_continuing": "后续仍可能持续",
        "insufficient_evidence": "后续证据不足",
        "not_a_blocker": "历史文本不是有效卡点",
    }.get(status, "后续证据不足")


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _bounded_limit(value: Any, *, default: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(1, min(number, maximum))


def _optional_positive_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _clip_text(value: Any, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."
