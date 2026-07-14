from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field

from app.prompts import (
    RAG_ANSWER_PROMPT_NAME,
    RAG_ANSWER_PROMPT_VERSION,
    TEXT_GENERATE_PROMPT_NAME,
    TEXT_GENERATE_PROMPT_VERSION,
)
from app.schemas.context import ContextStore
from app.schemas.llm import LLMFunctionSchema, LLMMessage, LLMRequest
from app.schemas.tool import ToolResult
from app.schemas.tool_outputs import TextGenerateToolOutput
from app.tools.llm_base import BaseLLMTool
from app.tools.llm_client import FAIL_FAST_TIMEOUT_MARKER
from app.utils import runtime_progress
from app.utils.time_context import merge_runtime_time_context


WEEKLY_PLAN_STATUS_VALUES = {"已完成", "部分完成", "未完成", "证据不足"}
DEPT_PLAN_JUDGEMENT_FAILED_STATUS = "判断失败"
DEPT_PLAN_STATUS_VALUES = {*WEEKLY_PLAN_STATUS_VALUES, DEPT_PLAN_JUDGEMENT_FAILED_STATUS}

DEPT_PLAN_CANDIDATE_CONFIDENCE_LABELS = {
    "high": "高",
    "medium": "中",
    "low": "低",
}

DEPT_PLAN_CANDIDATE_SOURCE_LABELS = {
    "owner_keyword_trace": "负责人和关键词均匹配的周报线索",
    "owner_trace": "负责人周报线索",
    "owner_cross_department_keyword_trace": "负责人跨部门且关键词相关的周报线索",
    "owner_cross_department_trace": "负责人跨部门周报线索",
    "keyword_support_fallback": "关键词相关候选",
    "recency_fallback": "按时间兜底候选",
    "collaboration_weekly_report_hint": "非负责人主表周报协作辅助线索",
    "possible_weekly_evidence": "备选复核材料",
}

DEPT_PLAN_FILTER_REASON_LABELS = {
    "department_mismatch": "部门不匹配",
    "no_owner_or_keyword_support": "缺少负责人或关键词支持",
    "not_selected_due_to_candidate_cap": "候选数量截断未展示",
    "owner_cross_department_possible": "负责人跨部门备选证据",
    "filtered_from_strong_candidates": "未进入强候选",
    "possible_evidence": "备选证据",
}

DEPT_PLAN_OPL_CANDIDATE_SOURCE_LABELS = {
    "owner_opl_issue_keyword_trace": "负责人和关键词均匹配的 OPL 问题线索",
    "owner_opl_issue_trace": "负责人匹配的 OPL 问题线索",
    "tracker_opl_issue_keyword_trace": "跟踪人和关键词均匹配的 OPL 问题线索",
    "tracker_opl_issue_trace": "跟踪人匹配的 OPL 问题线索",
    "department_opl_issue_keyword_trace": "部门和关键词均匹配的 OPL 问题线索",
    "keyword_opl_issue_trace": "关键词匹配的 OPL 问题线索",
    "opl_issue_trace": "OPL 问题线索",
    "possible_opl_evidence": "备选 OPL 问题线索",
}

DEPT_PLAN_OPL_FILTER_REASON_LABELS = {
    "department_mismatch": "部门、负责人和关键词均不支持",
    "no_owner_or_keyword_support": "缺少负责人或关键词支持",
    "weak_opl_match": "OPL 关联较弱",
    "not_selected_due_to_candidate_cap": "候选数量截断未展示",
    "possible_opl_evidence": "备选 OPL 证据",
}

DEPT_PLAN_OPL_STATUS_GROUP_LABELS = {
    "open": "未闭环",
    "closed": "已闭环",
    "unknown": "状态不明确",
}

DEPT_PLAN_SELF_EVAL_TYPE_LABELS = {
    "achievement": "完成项",
    "unfinished": "未完成项",
    "risk": "风险项",
    "reason": "原因说明",
    "next_action": "后续动作",
    "resolved": "已解决项",
    "unresolved": "未解决项",
}

DEPT_PLAN_INTERNAL_BOOL_LABELS = {
    "owner_match": (("负责人匹配", "负责人不匹配")),
    "department_match": (("部门匹配", "部门不匹配")),
    "owner_cross_department": (("负责人跨部门周报", "非负责人跨部门周报")),
    "strong_keyword_support": (("有明确关键词支持", "缺少明确关键词支持")),
    "business_keyword_support": (("有具体业务关键词支持", "缺少具体业务关键词支持")),
    "tracker_match": (("跟踪人匹配", "跟踪人不匹配")),
    "open_issue": (("未闭环问题", "非未闭环问题")),
    "closed_issue": (("已闭环问题", "非已闭环问题")),
    "high_priority": (("高优先级", "非高优先级")),
    "has_keyword_support": (("有关键词支持", "缺少关键词支持")),
    "exact_phrase": (("有原词组命中", "无原词组命中")),
}

DEPT_PLAN_INTERNAL_TOKEN_LABELS = {
    "possible_weekly_evidence": "备选复核材料",
    "weekly_match_audit": "周报匹配审计",
    "weekly_done_candidates": "周报候选",
    "self_eval_candidates": "自评候选",
    "opl_issue_candidates": "OPL 相关问题",
    "possible_opl_evidence": "OPL 备选复核材料",
    "opl_match_audit": "OPL 匹配审计",
    "candidate_confidence": "候选可信度",
    "candidate_source": "候选来源",
    "strong_keyword_support": "明确关键词支持",
    "business_keyword_support": "具体业务关键词支持",
    "has_keyword_support": "关键词支持",
    "specific_overlap_keywords": "具体命中词",
    "filter_reason": "过滤原因",
    "overlap_keywords": "重合关键词",
    "tracker_match": "跟踪人匹配",
    "status_group": "闭环状态",
    "open_issue": "未闭环问题",
    "closed_issue": "已闭环问题",
    "high_priority": "高优先级",
    "issue_ref": "OPL 问题编号",
    "source_issue_id": "OPL 源问题 ID",
    "issue_date": "问题日期",
    "follow_date": "跟踪日期",
    "issue_description": "问题描述",
    "solution_progress": "解决措施/最新进展",
    "priority": "优先级",
    "tracker_user": "跟踪人",
    "assembly": "机构/总成",
    "remark": "备注",
    "coverage": "覆盖度",
    "exact_phrase": "原词组命中",
    "source_doc_id": "来源文档",
    "source_chunk_id": "来源证据片段",
    "owner_match_count": "负责人匹配数量",
    "tracker_match_count": "跟踪人匹配数量",
    "owner_candidate_count": "负责人候选数量",
    "owner_cross_department_count": "负责人跨部门周报数量",
    "owner_cross_department_selected_count": "入选的负责人跨部门周报数量",
    "owner_selected_count": "入选的负责人周报数量",
    "non_owner_selected_count": "入选的非负责人周报数量",
    "department_filtered_count": "部门范围内候选数量",
    "department_compatible_count": "部门兼容数量",
    "keyword_match_count": "关键词匹配数量",
    "strong_keyword_match_count": "明确关键词匹配数量",
    "business_keyword_match_count": "具体业务关键词匹配数量",
    "raw_opl_issue_count": "OPL 问题池数量",
    "open_issue_count": "未闭环 OPL 数量",
    "high_priority_open_issue_count": "高优先级未闭环 OPL 数量",
    "candidate_pool_count": "候选池数量",
    "selected_before_cap_count": "截断前入选数量",
    "selected_count": "入选强候选数量",
    "selected_candidate_limit": "强候选展示上限",
    "non_owner_candidate_limit": "非负责人候选展示上限",
    "owner_candidate_over_limit_count": "超出展示上限但仍保留的负责人周报数量",
    "capped_candidate_count": "因数量上限未展示的候选数量",
    "possible_evidence_count": "备选复核数量",
    "possible_evidence_before_cap_count": "截断前备选复核数量",
    "filtered_counts": "过滤原因统计",
    "top_unfinished_items": "重点未完成事项",
    "status_counts": "状态统计",
    "department_summaries": "部门汇总",
    "total_plans": "计划总数",
    "unfinished_or_attention_count": "未完全完成/需关注数量",
    "weekly_evidence_candidate_count": "周报候选数量",
    "self_eval_candidate_count": "自评候选数量",
    "opl_issue_count": "OPL 问题取数数量",
    "opl_issue_candidate_count": "计划相关 OPL 候选数量",
    "possible_opl_evidence_count": "OPL 备选复核数量",
    "open_opl_issue_candidate_count": "未闭环 OPL 候选数量",
    "high_priority_open_opl_issue_candidate_count": "高优先级未闭环 OPL 候选数量",
    "plans_with_opl_issue_candidates": "有关联 OPL 的计划数量",
    "batch_error_count": "批次错误数量",
    "batch_errors": "批次错误",
    "generation_method": "生成方式",
    "owner_user": "负责人",
    "user_name": "人员",
    "report_date": "周报日期",
    "done_text": "完成内容",
    "item_type": "事项类型",
    "item_text": "事项内容",
    "plan_text": "计划内容",
    "plan_id": "计划编号",
    "plan_month": "计划月份",
    "due_date": "截止日期",
}

DEPT_PLAN_INTERNAL_VALUE_LABELS = {
    **DEPT_PLAN_CANDIDATE_SOURCE_LABELS,
    **DEPT_PLAN_FILTER_REASON_LABELS,
    **DEPT_PLAN_OPL_CANDIDATE_SOURCE_LABELS,
    **DEPT_PLAN_OPL_FILTER_REASON_LABELS,
    **DEPT_PLAN_OPL_STATUS_GROUP_LABELS,
    **DEPT_PLAN_SELF_EVAL_TYPE_LABELS,
    "open": "未闭环",
    "closed": "已闭环",
    "unknown": "状态不明确",
}


class WeeklyPlanBatchJudgementItem(BaseModel):
    plan_id: str = Field(..., description="Input plan id.")
    status: str = Field(..., description="已完成、部分完成、未完成或证据不足。")
    reason: str = Field(..., description="简短判断原因。")
    evidence: str = Field(default="", description="支撑该判断的完成项摘要；无则为空。")


class WeeklyPlanBatchJudgementOutput(BaseModel):
    judgements: list[WeeklyPlanBatchJudgementItem] = Field(
        default_factory=list,
        description="Judgement for each input item, preserving input order where possible.",
    )


class DeptPlanBatchJudgementItem(BaseModel):
    plan_id: str = Field(..., description="Input dept plan id.")
    status: str = Field(..., description="已完成、部分完成、未完成或证据不足。")
    reason: str = Field(..., description="简短判断原因。")
    evidence: str = Field(default="", description="支撑该判断的周报、自评或 OPL 证据摘要；无则为空。")


class DeptPlanBatchJudgementOutput(BaseModel):
    judgements: list[DeptPlanBatchJudgementItem] = Field(
        default_factory=list,
        description="Judgement for each input department plan item.",
    )


class DeptPlanSingleJudgementOutput(BaseModel):
    plan_id: str = Field(..., description="Input dept plan id.")
    status: str = Field(..., description="已完成、部分完成、未完成或证据不足。")
    reason: str = Field(..., description="简短判断原因。")
    evidence: str = Field(default="", description="支撑该判断的周报、自评或 OPL 证据摘要；无则为空。")


class DeptPlanOwnerEvidenceReportItem(BaseModel):
    report_id: str = Field(..., description="Input owner weekly report id.")
    report_date: str = Field(default="", description="Weekly report date.")
    submitter: str = Field(default="", description="Weekly report submitter.")
    is_related: bool = Field(default=False, description="Whether this report is semantically related to the plan.")
    evidence_snippets: list[str] = Field(
        default_factory=list,
        description="Original report snippets that support the relation or completion judgement.",
    )
    completion_signal: str = Field(default="", description="Completion, progress, or no-completion signal.")
    blockage_signal: str = Field(default="", description="Blocking, risk, pending, or unfinished signal.")
    relation_reason: str = Field(default="", description="Short reason why the report is or is not related.")


class DeptPlanOwnerEvidenceExtractionOutput(BaseModel):
    plan_id: str = Field(..., description="Input dept plan id.")
    reports: list[DeptPlanOwnerEvidenceReportItem] = Field(
        default_factory=list,
        description="Evidence extraction result for each input owner weekly report.",
    )


class DeptPlanOwnerGroupEvidenceItem(BaseModel):
    plan_id: str = Field(..., description="Input dept plan id.")
    report_id: str = Field(..., description="Input owner weekly report id.")
    report_date: str = Field(default="", description="Weekly report date.")
    submitter: str = Field(default="", description="Weekly report submitter.")
    is_related: bool = Field(default=False, description="Whether this report is semantically related to this plan.")
    evidence_snippets: list[str] = Field(
        default_factory=list,
        description="Original report snippets that support the relation or completion judgement.",
    )
    completion_signal: str = Field(default="", description="Completion, progress, or no-completion signal.")
    blockage_signal: str = Field(default="", description="Blocking, risk, pending, or unfinished signal.")
    relation_reason: str = Field(default="", description="Short reason why the report is or is not related.")


class DeptPlanOwnerGroupEvidenceExtractionOutput(BaseModel):
    items: list[DeptPlanOwnerGroupEvidenceItem] = Field(
        default_factory=list,
        description="Evidence extraction result for each plan/report pair in the input owner group.",
    )


class WeeklyPlanPersonSummaryItem(BaseModel):
    user_name: str = Field(..., description="Person name.")
    department: str = Field(default="", description="Department name.")
    summary: str = Field(..., description="Concise Chinese person-level completion summary.")
    blocked_or_unfinished: list[str] = Field(
        default_factory=list,
        description="Key unfinished or blocked plan summaries for this person.",
    )
    follow_up_suggestions: list[str] = Field(
        default_factory=list,
        description="Suggested follow-up actions for this person.",
    )


class WeeklyPlanPersonSummaryOutput(BaseModel):
    people: list[WeeklyPlanPersonSummaryItem] = Field(
        default_factory=list,
        description="Person-level summaries for the supplied grouped plan judgements.",
    )


class TextGenerateTool(BaseLLMTool):
    """面向自然语言生成任务优化的 LLM 工具。"""

    name: str = Field(default="text_generate_tool")
    description: str = Field(default="Generate polished text from the provided prompt and context.")
    prompt_name: str = Field(default=TEXT_GENERATE_PROMPT_NAME)
    prompt_version: str = Field(default=TEXT_GENERATE_PROMPT_VERSION)
    function_name: str = Field(default="emit_text_generation_output")
    function_description: str = Field(default="Return validated text-generation output for the runtime.")
    timeout: int = Field(default=75, gt=0)
    default_timeout_seconds: int = Field(default=45, gt=0)
    default_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    tags: list[str] = Field(default_factory=lambda: ["llm", "generation", "text"])

    INTERNAL_DISCLOSURE_SAFE_TEXT: str = Field(
        default=(
            "我不能提供当前系统的底层源码、内部提示词、工具封装、接口定义、路由策略或运行时实现细节。"
            "我可以说明对外可用能力：根据你的问题生成回答；在授权范围内查询和汇总业务数据或知识库内容；"
            "对你提供的代码片段、公开项目或明确授权的模块做解释和排查。"
            "如果你是在做代码审查，请直接提供需要查看的文件或具体模块，我可以基于可见代码协助分析。"
        )
    )

    def get_routing_capability(self) -> dict[str, Any]:
        capability = super().get_routing_capability()
        capability["supported_task_types"] = ["text_generation"]
        capability["default_task_type"] = "text_generation"
        capability["supported_tags"] = list(self.tags)
        return capability

    def build_prompt_variables(self, payload: dict[str, Any], context: ContextStore | None = None) -> dict[str, Any]:
        prompt = str(payload.get("prompt") or payload.get("input") or payload.get("query") or "").strip()
        if not prompt:
            raise ValueError("text_generate_tool 需要非空的 prompt。")
        if self._asks_weekly_blockers(
            self._deterministic_report_haystack(payload=payload, context=context)
        ):
            prompt = self._sanitize_weekly_blocker_user_text(prompt)

        style = payload.get("style")
        audience = payload.get("audience")
        context_block = merge_runtime_time_context(
            payload.get("context"),
            context.runtime if context is not None else None,
        )
        if self._asks_weekly_blockers(
            self._deterministic_report_haystack(payload=payload, context=context)
        ):
            context_block = self._sanitize_weekly_blocker_user_text(context_block)
        variables = {
            "prompt": prompt,
            "style": str(style or "None"),
            "audience": str(audience or "None"),
            "context_block": context_block,
        }
        if self._is_rag_grounded(payload.get("rag_grounded")):
            variables["user_question"] = self._resolve_rag_user_question(
                payload=payload,
                context=context,
                fallback=prompt,
            )
        return variables

    def render_prompt(self, payload: dict[str, Any], context: ContextStore | None = None):
        rag_grounded = self._is_rag_grounded(payload.get("rag_grounded"))
        prompt_name = RAG_ANSWER_PROMPT_NAME if rag_grounded else self.prompt_name
        prompt_version = RAG_ANSWER_PROMPT_VERSION if rag_grounded else self.prompt_version
        return self.prompt_registry.render(
            prompt_name,
            version=prompt_version,
            variables=self.build_prompt_variables(payload=payload, context=context),
        )

    def _is_rag_grounded(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _resolve_rag_user_question(
        self,
        *,
        payload: dict[str, Any],
        context: ContextStore | None,
        fallback: str,
    ) -> str:
        if context is not None and context.runtime.user_input.strip():
            return context.runtime.user_input.strip()
        for key in ("query", "question", "prompt", "input"):
            value = payload.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return fallback

    def get_output_model(self) -> type[TextGenerateToolOutput]:
        return TextGenerateToolOutput

    def sanitize_output_model(
        self,
        output_model: BaseModel,
        *,
        payload: dict[str, Any],
        context: ContextStore | None,
    ) -> BaseModel:
        if not isinstance(output_model, TextGenerateToolOutput):
            return output_model
        if self._asks_runtime_internal_disclosure(payload=payload, context=context):
            return output_model.model_copy(update={"text": self.INTERNAL_DISCLOSURE_SAFE_TEXT})
        if not self._asks_weekly_blockers(
            self._deterministic_report_haystack(payload=payload, context=context)
        ):
            return output_model
        sanitized_text = self._sanitize_weekly_blocker_user_text(output_model.text)
        if sanitized_text == output_model.text:
            return output_model
        return output_model.model_copy(update={"text": sanitized_text})

    def build_request(
        self,
        *,
        rendered_prompt,
        payload: dict[str, Any],
        context: ContextStore | None = None,
    ) -> LLMRequest:
        request = super().build_request(rendered_prompt=rendered_prompt, payload=payload, context=context)
        metadata = dict(request.metadata)
        metadata["fail_fast_timeout"] = True
        metadata["llm_request_timeout_seconds"] = request.timeout_seconds
        metadata["rag_grounded"] = self._is_rag_grounded(payload.get("rag_grounded"))
        return request.model_copy(update={"metadata": metadata})

    async def _arun(self, payload: dict[str, Any], context: ContextStore | None = None) -> ToolResult:
        if self._asks_runtime_internal_disclosure(payload=payload, context=context):
            runtime_progress(
                step="模型输出",
                status="内部信息请求拦截",
                detail=self.INTERNAL_DISCLOSURE_SAFE_TEXT,
                request_id=context.runtime.request_id if context is not None else None,
                session_id=context.runtime.session_id if context is not None else None,
            )
            return self.build_result(
                success=True,
                output={
                    "text": self.INTERNAL_DISCLOSURE_SAFE_TEXT,
                    "audience": str(payload.get("audience") or "business_user"),
                    "style": str(payload.get("style") or "concise"),
                },
                metadata={
                    "runtime_internal_disclosure_guard": True,
                    "rag_grounded": self._is_rag_grounded(payload.get("rag_grounded")),
                },
            )

        mysql_fallback_report = self._resolve_mysql_weekly_plan_fallback_report(payload=payload, context=context)
        if mysql_fallback_report:
            runtime_progress(
                step="模型输出",
                status="MySQL周计划兜底报告",
                detail=mysql_fallback_report[:5000],
                request_id=context.runtime.request_id if context is not None else None,
                session_id=context.runtime.session_id if context is not None else None,
            )
            return self.build_result(
                success=True,
                output={
                    "text": mysql_fallback_report,
                    "audience": str(payload.get("audience") or "business_user"),
                    "style": str(payload.get("style") or "structured"),
                },
                metadata={
                    "mysql_weekly_plan_fallback_report": True,
                    "report_source": "context.task_results.weekly_plan_comparison.final_report",
                    "rag_grounded": self._is_rag_grounded(payload.get("rag_grounded")),
                },
            )

        mysql_dept_plan_report = await self._try_mysql_dept_plan_llm_report(payload=payload, context=context)
        if mysql_dept_plan_report is not None:
            report_text, report_metadata = mysql_dept_plan_report
            runtime_progress(
                step="模型输出",
                status="MySQL三七计划LLM分批判断",
                detail=report_text[:5000],
                request_id=context.runtime.request_id if context is not None else None,
                session_id=context.runtime.session_id if context is not None else None,
            )
            return self.build_result(
                success=True,
                output={
                    "text": report_text,
                    "audience": str(payload.get("audience") or "business_user"),
                    "style": str(payload.get("style") or "structured"),
                },
                metadata={
                    "mysql_dept_plan_llm_judgement": True,
                    "rag_grounded": self._is_rag_grounded(payload.get("rag_grounded")),
                    **report_metadata,
                },
            )

        mysql_weekly_plan_report = await self._try_mysql_weekly_plan_llm_report(payload=payload, context=context)
        if mysql_weekly_plan_report is not None:
            report_text, report_metadata = mysql_weekly_plan_report
            runtime_progress(
                step="模型输出",
                status="MySQL周计划LLM分批判断",
                detail=report_text[:5000],
                request_id=context.runtime.request_id if context is not None else None,
                session_id=context.runtime.session_id if context is not None else None,
            )
            return self.build_result(
                success=True,
                output={
                    "text": report_text,
                    "audience": str(payload.get("audience") or "business_user"),
                    "style": str(payload.get("style") or "structured"),
                },
                metadata={
                    "mysql_weekly_plan_llm_judgement": True,
                    "rag_grounded": self._is_rag_grounded(payload.get("rag_grounded")),
                    **report_metadata,
                },
            )

        result = await super()._arun(payload=payload, context=context)
        if result.success:
            metadata = dict(result.metadata)
            if self._weekly_blocker_result_text_was_sanitized(result=result):
                metadata["weekly_blocker_user_text_sanitized"] = True
                return result.model_copy(update={"metadata": metadata})
            return result

        error_text = str(result.error or "")
        if FAIL_FAST_TIMEOUT_MARKER not in error_text:
            return result

        metadata = dict(result.metadata)
        metadata["fail_fast_timeout"] = True
        metadata["llm_request_timeout_seconds"] = int(payload.get("timeout_seconds", self.default_timeout_seconds))
        metadata["tool_timeout_seconds"] = int(payload.get("tool_timeout_seconds", self.timeout))
        return result.model_copy(update={"metadata": metadata})

    def _weekly_blocker_result_text_was_sanitized(self, *, result: ToolResult) -> bool:
        output = result.output
        if not isinstance(output, Mapping):
            return False
        text = output.get("text")
        if not isinstance(text, str):
            return False
        request = result.metadata.get("request") if isinstance(result.metadata, Mapping) else None
        raw_response = result.metadata.get("raw_response") if isinstance(result.metadata, Mapping) else None
        return self._weekly_blocker_text_contains_internal_terms(request) or self._weekly_blocker_text_contains_internal_terms(raw_response)

    def _weekly_blocker_text_contains_internal_terms(self, value: Any) -> bool:
        text = json.dumps(value, ensure_ascii=False, default=str) if isinstance(value, (Mapping, list)) else str(value or "")
        return "risk_and_help" in text or "weekly_blocker_context_text" in text or "plan_followups" in text

    def _asks_runtime_internal_disclosure(
        self,
        *,
        payload: Mapping[str, Any],
        context: ContextStore | None,
    ) -> bool:
        text = self._deterministic_report_haystack(payload=dict(payload), context=context).lower()
        compact = re.sub(r"\s+", "", text)
        if not compact:
            return False

        strong_patterns = (
            "底层代码",
            "底层源码",
            "源代码",
            "内部代码",
            "源码",
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
            "jsonschema",
            "schema",
            "functioncalling",
            "函数调用",
        )
        runtime_subjects = (
            "你",
            "你的",
            "现在你",
            "当前",
            "这个项目",
            "你这个项目",
            "当前项目",
            "这个系统",
            "当前系统",
            "对话系统",
            "工具链",
            "运行时",
            "runtime",
            "agent",
            "平台",
            "模型",
        )
        if any(pattern in compact for pattern in strong_patterns) and any(
            subject in compact for subject in runtime_subjects
        ):
            return True

        ambiguous_runtime_source_requests = (
            "底层源码",
            "底层代码",
            "内部代码",
            "系统提示词",
            "内部提示词",
        )
        if any(pattern in compact for pattern in ambiguous_runtime_source_requests):
            return True

        project_tool_requests = (
            "项目所有功能",
            "所有的功能",
            "工具怎么封装",
            "怎么封装的工具",
            "封装方式",
            "工具定义",
        )
        return any(pattern in compact for pattern in project_tool_requests) and any(
            subject in compact for subject in runtime_subjects
        )

    def _sanitize_weekly_blocker_user_text(self, value: Any) -> str:
        text = str(value or "")
        if not text:
            return ""

        replacements = [
            ("有效 risk_and_help", "有效员工自填卡点"),
            ("risk_and_help 是有效卡点/风险/求助内容", "员工填写了有效卡点/风险/求助内容"),
            ("非空 risk_and_help 是员工自填卡点", "已填写卡点为员工自填卡点"),
            ("risk_and_help 字段非空", "员工已填写卡点"),
            ("risk_and_help 全部为空", "所有人均未填写卡点"),
            ("risk_and_help 非空", "员工已填写卡点"),
            ("非空 risk_and_help", "员工已填写卡点"),
            ("risk_and_help 为空的人员", "未填写卡点的人员"),
            ("risk_and_help 为空", "未填写卡点"),
            ("risk_and_help", "员工自填卡点"),
            ("weekly_blocker_context_text 压缩字段", "压缩后的逐人卡点证据"),
            ("weekly_blocker_context_text", "逐人卡点证据"),
            ("plan_followups", "计划追溯证据"),
        ]
        for raw, label in replacements:
            text = text.replace(raw, label)
        return text.strip()

    def _resolve_mysql_weekly_plan_fallback_report(
        self,
        *,
        payload: dict[str, Any],
        context: ContextStore | None,
    ) -> str | None:
        if context is None:
            return None

        haystack = self._deterministic_report_haystack(payload=payload, context=context)
        if not self._asks_weekly_plan_completion(haystack):
            return None
        requested_context = str(payload.get("context") or "")
        for key, output in context.task_results.items():
            if not isinstance(output, dict):
                continue
            if key != "weekly_plan_comparison" and "plan_tracking_context_text" not in output:
                continue
            report = str(output.get("final_report") or "").strip()
            report_type = str(output.get("final_report_type") or "").strip()
            if not report or report_type != "weekly_plan_completion":
                continue
            if self._has_mysql_weekly_plan_followups(output):
                continue
            if requested_context and key not in requested_context and "plan_tracking_context_text" not in requested_context:
                continue
            return report
        return None

    async def _try_mysql_weekly_plan_llm_report(
        self,
        *,
        payload: dict[str, Any],
        context: ContextStore | None,
    ) -> tuple[str, dict[str, Any]] | None:
        if context is None:
            return None
        if self._asks_weekly_blockers(self._deterministic_report_haystack(payload=payload, context=context)):
            return None
        source_key, source_output = self._resolve_mysql_weekly_plan_source(payload=payload, context=context)
        if source_key is None or source_output is None:
            return None

        followups = [
            item for item in source_output.get("plan_followups", [])
            if isinstance(item, Mapping)
        ] if isinstance(source_output.get("plan_followups"), list) else []
        if not followups:
            return None

        batch_size = self._env_int("MYSQL_WEEKLY_LLM_JUDGE_BATCH_SIZE", 12)
        batch_concurrency = self._env_int("MYSQL_WEEKLY_LLM_JUDGE_CONCURRENCY", 4)
        batches = [
            followups[index : index + batch_size]
            for index in range(0, len(followups), batch_size)
        ]
        semaphore = asyncio.Semaphore(max(1, batch_concurrency))
        batch_errors: list[dict[str, Any]] = []

        async def run_batch(batch_index: int, batch_items: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
            async with semaphore:
                try:
                    return await self._invoke_weekly_plan_judge_batch(
                        context=context,
                        payload=payload,
                        batch_index=batch_index,
                        batch_items=batch_items,
                    )
                except Exception as exc:
                    batch_errors.append(
                        {
                            "batch_index": batch_index,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "item_count": len(batch_items),
                        }
                    )
                    return [
                        self._fallback_weekly_plan_judgement(item=item, reason=f"LLM 批次判断失败：{type(exc).__name__}")
                        for item in batch_items
                    ]

        batch_results = await asyncio.gather(
            *(run_batch(batch_index, batch) for batch_index, batch in enumerate(batches, start=1))
        )
        judgements = [item for batch in batch_results for item in batch]
        ordered_judgements = self._ordered_weekly_plan_judgements(
            followups=followups,
            judgements=judgements,
        )
        merge_error: dict[str, Any] | None = None
        layered_merge_used = False
        person_summary_count = 0
        try:
            report_text, layered_metadata = await self._invoke_weekly_plan_layered_merge_report(
                context=context,
                payload=payload,
                source_output=source_output,
                ordered_judgements=ordered_judgements,
                batch_errors=batch_errors,
            )
            layered_merge_used = True
            person_summary_count = int(layered_metadata.get("person_summary_count") or 0)
            merge_report_by_llm = True
        except Exception as exc:
            merge_error = {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            try:
                report_text = await self._invoke_weekly_plan_merge_report(
                    context=context,
                    payload=payload,
                    source_output=source_output,
                    ordered_judgements=ordered_judgements,
                    batch_errors=batch_errors,
                )
                merge_report_by_llm = True
            except Exception as fallback_exc:
                merge_error = {
                    "error_type": type(fallback_exc).__name__,
                    "error": str(fallback_exc),
                    "layered_error": merge_error,
                }
                report_text = self._format_mysql_weekly_plan_llm_report(
                    source_output=source_output,
                    followups=followups,
                    judgements=judgements,
                )
                merge_report_by_llm = False
        return report_text, {
            "mysql_weekly_plan_source_key": source_key,
            "plan_followup_count": len(followups),
            "llm_batch_count": len(batches),
            "llm_batch_size": batch_size,
            "llm_batch_concurrency": batch_concurrency,
            "llm_batch_error_count": len(batch_errors),
            "llm_batch_errors": batch_errors[:10],
            "llm_merge_report": merge_report_by_llm,
            "llm_merge_error": merge_error,
            "llm_layered_merge": layered_merge_used,
            "llm_person_summary_count": person_summary_count,
        }

    async def _try_mysql_dept_plan_llm_report(
        self,
        *,
        payload: dict[str, Any],
        context: ContextStore | None,
    ) -> tuple[str, dict[str, Any]] | None:
        if context is None:
            return None
        source_key, source_output = self._resolve_mysql_dept_plan_source(payload=payload, context=context)
        if source_key is None or source_output is None:
            return None

        followups = [
            item for item in source_output.get("dept_plan_followups", [])
            if isinstance(item, Mapping)
        ] if isinstance(source_output.get("dept_plan_followups"), list) else []
        if not followups:
            return None

        followups, owner_evidence_metadata, owner_evidence_failed_judgements = (
            await self._attach_dept_plan_owner_evidence_extractions(
                followups=followups,
                source_output=source_output,
                context=context,
                payload=payload,
            )
        )
        judgement_ready_followups = [
            item
            for item in followups
            if not item.get("owner_evidence_extraction_failed")
        ]
        cached_judgements, cache_metadata = self._load_dept_plan_judgement_cache(followups=judgement_ready_followups)
        cached_plan_ids = {item.get("plan_id") for item in cached_judgements}
        pending_followups = [
            item
            for item in judgement_ready_followups
            if self._dept_plan_followup_id(item) not in cached_plan_ids
        ]
        batch_size = self._env_int("MYSQL_DEPT_PLAN_LLM_JUDGE_BATCH_SIZE", 6)
        batch_concurrency = self._env_int("MYSQL_DEPT_PLAN_LLM_JUDGE_CONCURRENCY", 3)
        batch_max_attempts = self._env_int("MYSQL_DEPT_PLAN_LLM_JUDGE_MAX_ATTEMPTS", 2)
        batch_max_chars = self._env_int("MYSQL_DEPT_PLAN_LLM_JUDGE_BATCH_MAX_CHARS", 24000)
        recovery_max_attempts = self._env_int("MYSQL_DEPT_PLAN_LLM_JUDGE_RECOVERY_MAX_ATTEMPTS", 2)
        batch_retry_delay_seconds = self._env_float("MYSQL_DEPT_PLAN_LLM_JUDGE_RETRY_DELAY_SECONDS", 1.5)
        batch_circuit_retry_delay_seconds = self._env_float(
            "MYSQL_DEPT_PLAN_LLM_JUDGE_CIRCUIT_RETRY_DELAY_SECONDS",
            31.0,
        )
        batches, batch_plan = self._build_dept_plan_judge_batches(
            followups=pending_followups,
            max_items=batch_size,
            max_chars=batch_max_chars,
        )
        semaphore = asyncio.Semaphore(max(1, batch_concurrency))
        batch_errors: list[dict[str, Any]] = []
        batch_retry_events: list[dict[str, Any]] = []
        batch_rescue_events: list[dict[str, Any]] = []
        batch_recovery_successes: list[dict[str, Any]] = []
        cache_store_count = 0

        async def invoke_batch_with_retries(
            *,
            batch_index: int,
            batch_items: list[Mapping[str, Any]],
            max_attempts: int,
            compact_mode: str,
            stage: str,
        ) -> tuple[list[dict[str, Any]] | None, Exception | None]:
            attempts = max(1, max_attempts)
            last_exc: Exception | None = None
            for attempt in range(1, attempts + 1):
                try:
                    return await self._invoke_dept_plan_judge_batch(
                        context=context,
                        payload=payload,
                        batch_index=batch_index,
                        batch_items=batch_items,
                        attempt=attempt,
                        compact_mode=compact_mode,
                        batch_stage=stage,
                    ), None
                except Exception as exc:
                    last_exc = exc
                    if attempt >= attempts:
                        break
                    retry_delay_seconds = (
                        batch_circuit_retry_delay_seconds
                        if type(exc).__name__ == "CircuitBreakerOpenError"
                        else batch_retry_delay_seconds
                    )
                    batch_retry_events.append(
                        {
                            "batch_index": batch_index,
                            "stage": stage,
                            "compact_mode": compact_mode,
                            "attempt": attempt,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "item_count": len(batch_items),
                            "retry_delay_seconds": retry_delay_seconds,
                        }
                    )
                    if retry_delay_seconds > 0:
                        await asyncio.sleep(retry_delay_seconds)
            return None, last_exc

        async def run_batch(batch_index: int, batch_items: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
            nonlocal cache_store_count
            async with semaphore:
                result, error = await invoke_batch_with_retries(
                    batch_index=batch_index,
                    batch_items=batch_items,
                    max_attempts=batch_max_attempts,
                    compact_mode="primary",
                    stage="primary",
                )
                if result is not None:
                    cache_store_count += self._store_dept_plan_judgement_cache_for_results(
                        batch_items=batch_items,
                        judgements=result,
                    )
                    return result

                if len(batch_items) > 1:
                    batch_rescue_events.append(
                        {
                            "batch_index": batch_index,
                            "stage": "split_after_batch_failure",
                            "error_type": type(error).__name__ if error is not None else "UnknownError",
                            "error": str(error) if error is not None else "",
                            "item_count": len(batch_items),
                            "plan_ids": [self._dept_plan_followup_id(item) for item in batch_items],
                        }
                    )
                    rescued: list[dict[str, Any]] = []
                    for split_index, item in enumerate(batch_items, start=1):
                        split_stage = f"split_{split_index}"
                        split_result, split_error = await invoke_batch_with_retries(
                            batch_index=batch_index,
                            batch_items=[item],
                            max_attempts=batch_max_attempts,
                            compact_mode="primary",
                            stage=split_stage,
                        )
                        if split_result is not None:
                            cache_store_count += self._store_dept_plan_judgement_cache_for_results(
                                batch_items=[item],
                                judgements=split_result,
                            )
                            rescued.extend(split_result)
                            continue
                        recovery_result, recovery_error = await invoke_batch_with_retries(
                            batch_index=batch_index,
                            batch_items=[item],
                            max_attempts=recovery_max_attempts,
                            compact_mode="recovery",
                            stage=f"recovery_{split_index}",
                        )
                        if recovery_result is not None:
                            batch_recovery_successes.append(
                                {
                                    "batch_index": batch_index,
                                    "stage": f"recovery_{split_index}",
                                    "plan_id": self._dept_plan_followup_id(item),
                                }
                            )
                            cache_store_count += self._store_dept_plan_judgement_cache_for_results(
                                batch_items=[item],
                                judgements=recovery_result,
                            )
                            rescued.extend(recovery_result)
                            continue
                        rescued.append(
                            self._record_failed_dept_plan_batch_item(
                                item=item,
                                batch_index=batch_index,
                                attempts=batch_max_attempts + recovery_max_attempts,
                                error=recovery_error or split_error or error,
                                batch_errors=batch_errors,
                                reason_prefix="LLM 批次拆分和压缩救援后仍失败",
                            )
                        )
                    return rescued

                recovery_result, recovery_error = await invoke_batch_with_retries(
                    batch_index=batch_index,
                    batch_items=batch_items,
                    max_attempts=recovery_max_attempts,
                    compact_mode="recovery",
                    stage="recovery",
                )
                if recovery_result is not None:
                    batch_recovery_successes.append(
                        {
                            "batch_index": batch_index,
                            "stage": "recovery",
                            "plan_ids": [self._dept_plan_followup_id(item) for item in batch_items],
                        }
                    )
                    cache_store_count += self._store_dept_plan_judgement_cache_for_results(
                        batch_items=batch_items,
                        judgements=recovery_result,
                    )
                    return recovery_result

                return [
                    self._record_failed_dept_plan_batch_item(
                        item=item,
                        batch_index=batch_index,
                        attempts=batch_max_attempts + recovery_max_attempts,
                        error=recovery_error or error,
                        batch_errors=batch_errors,
                        reason_prefix="LLM 批次判断和压缩救援后仍失败",
                    )
                    for item in batch_items
                ]

        if batches:
            batch_results = await asyncio.gather(
                *(run_batch(batch_index, batch) for batch_index, batch in enumerate(batches, start=1))
            )
            judgements = [item for batch in batch_results for item in batch]
        else:
            judgements = []
        judgements = [*owner_evidence_failed_judgements, *cached_judgements, *judgements]
        ordered_judgements = self._ordered_dept_plan_judgements(
            followups=followups,
            judgements=judgements,
        )
        ordered_judgements, opl_closed_completion_override_count = (
            self._apply_dept_plan_closed_opl_completion_rule(
                followups=followups,
                ordered_judgements=ordered_judgements,
            )
        )
        judgement_failed_items = [
            item
            for item in ordered_judgements
            if self._normalize_dept_plan_status(item.get("status")) == DEPT_PLAN_JUDGEMENT_FAILED_STATUS
        ]
        merge_error: dict[str, Any] | None = None
        judgement_complete = not batch_errors and not judgement_failed_items
        if not judgement_complete:
            report_text = self._format_mysql_dept_plan_judgement_incomplete_report(
                source_output=source_output,
                ordered_judgements=ordered_judgements,
                batch_errors=batch_errors,
                failed_items=judgement_failed_items,
            )
            merge_report_by_llm = False
        else:
            try:
                report_text = await self._invoke_dept_plan_merge_report(
                    context=context,
                    payload=payload,
                    source_output=source_output,
                    ordered_judgements=ordered_judgements,
                    batch_errors=batch_errors,
                )
                merge_report_by_llm = True
                report_text = self._replace_dept_plan_simple_attention_lists(
                    report_text=report_text,
                    ordered_judgements=ordered_judgements,
                )
                detail_text = self._format_dept_plan_detail_section(
                    source_output=source_output,
                    ordered_judgements=ordered_judgements,
                )
                report_text = f"{report_text.rstrip()}\n\n{detail_text}".strip()
            except Exception as exc:
                merge_error = {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                report_text = self._format_mysql_dept_plan_llm_report(
                    source_output=source_output,
                    ordered_judgements=ordered_judgements,
                )
                merge_report_by_llm = False
        report_text = self._ensure_dept_plan_report_data_source_note(
            report_text=report_text,
            source_output=source_output,
        )
        report_text = self._sanitize_dept_plan_user_text(report_text)
        cache_metadata["store_count"] = cache_store_count
        return report_text, {
            "mysql_dept_plan_source_key": source_key,
            "dept_plan_followup_count": len(followups),
            "llm_batch_count": len(batches),
            "llm_batch_size": batch_size,
            "llm_batch_dynamic": True,
            "llm_batch_max_chars": batch_max_chars,
            "llm_batch_plan": batch_plan[:20],
            "llm_judgement_cache": cache_metadata,
            "llm_pending_followup_count": len(pending_followups),
            "llm_owner_evidence_extraction": owner_evidence_metadata,
            "llm_batch_concurrency": batch_concurrency,
            "llm_batch_max_attempts": batch_max_attempts,
            "llm_recovery_max_attempts": recovery_max_attempts,
            "llm_batch_retry_delay_seconds": batch_retry_delay_seconds,
            "llm_batch_circuit_retry_delay_seconds": batch_circuit_retry_delay_seconds,
            "llm_batch_retry_count": len(batch_retry_events),
            "llm_batch_retry_events": batch_retry_events[:10],
            "llm_batch_rescue_count": len(batch_rescue_events),
            "llm_batch_rescue_events": batch_rescue_events[:10],
            "llm_batch_recovery_success_count": len(batch_recovery_successes),
            "llm_batch_recovery_successes": batch_recovery_successes[:10],
            "llm_batch_error_count": len(batch_errors),
            "llm_batch_errors": batch_errors[:10],
            "llm_opl_closed_completion_override_count": opl_closed_completion_override_count,
            "llm_judgement_complete": judgement_complete,
            "llm_judgement_failed_count": len(judgement_failed_items),
            "llm_merge_report": merge_report_by_llm,
            "llm_merge_error": merge_error,
        }

    def _resolve_mysql_dept_plan_source(
        self,
        *,
        payload: dict[str, Any],
        context: ContextStore,
    ) -> tuple[str | None, dict[str, Any] | None]:
        requested_context = str(payload.get("context") or "")
        haystack = self._deterministic_report_haystack(payload=payload, context=context)
        asks_completion = self._asks_dept_plan_completion(haystack)
        for key, output in context.task_results.items():
            if not isinstance(output, dict):
                continue
            if not isinstance(output.get("dept_plan_followups"), list):
                continue
            requested_compact_context = "dept_plan_completion_context_text" in requested_context
            requested_full_output = key in requested_context and asks_completion
            if requested_compact_context or requested_full_output or asks_completion:
                return key, output
        return None, None

    async def _attach_dept_plan_owner_evidence_extractions(
        self,
        *,
        followups: list[Mapping[str, Any]],
        source_output: Mapping[str, Any] | None,
        context: ContextStore,
        payload: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
        metadata = {
            "enabled": self._dept_plan_owner_evidence_extraction_enabled(),
            "plan_count": len(followups),
            "plans_with_owner_reports": 0,
            "owner_report_count": 0,
            "unique_owner_report_count": 0,
            "owner_group_count": 0,
            "grouped_extraction": False,
            "llm_call_count": 0,
            "cache_hit_count": 0,
            "cache_store_count": 0,
            "failure_count": 0,
            "failures": [],
        }
        prepared = [dict(item) for item in followups]
        if not metadata["enabled"]:
            for item in prepared:
                item["owner_weekly_llm_evidence"] = self._dept_plan_empty_owner_evidence(item=item)
            return prepared, metadata, []

        owner_report_pool = self._dept_plan_owner_report_pool_for_evidence(source_output)
        if owner_report_pool:
            return await self._attach_dept_plan_owner_evidence_extractions_by_owner_group(
                followups=prepared,
                owner_report_pool=owner_report_pool,
                context=context,
                payload=payload,
                metadata=metadata,
            )

        max_attempts = self._env_int("MYSQL_DEPT_PLAN_LLM_EVIDENCE_MAX_ATTEMPTS", 2)
        retry_delay_seconds = self._env_float("MYSQL_DEPT_PLAN_LLM_EVIDENCE_RETRY_DELAY_SECONDS", 1.0)
        chunk_size = max(1, self._env_int("MYSQL_DEPT_PLAN_LLM_EVIDENCE_CHUNK_SIZE", 12))
        concurrency = max(1, self._env_int("MYSQL_DEPT_PLAN_LLM_EVIDENCE_CONCURRENCY", 5))
        semaphore = asyncio.Semaphore(concurrency)

        async def run_item(item: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
            owner_reports = self._dept_plan_owner_weekly_reports_for_evidence(item)
            metadata["owner_report_count"] += len(owner_reports)
            if not owner_reports:
                item["owner_weekly_llm_evidence"] = self._dept_plan_empty_owner_evidence(item=item)
                return item, None
            metadata["plans_with_owner_reports"] += 1

            cached_evidence = self._load_dept_plan_owner_evidence_cache(item=item, owner_reports=owner_reports)
            if cached_evidence is not None:
                item["owner_weekly_llm_evidence"] = cached_evidence
                metadata["cache_hit_count"] += 1
                return item, None

            try:
                extracted_reports: list[dict[str, Any]] = []
                for chunk_index, chunk in enumerate(
                    self._chunk_dept_plan_owner_reports(owner_reports, size=chunk_size),
                    start=1,
                ):
                    last_exc: Exception | None = None
                    for attempt in range(1, max(1, max_attempts) + 1):
                        try:
                            metadata["llm_call_count"] += 1
                            chunk_reports = await self._invoke_dept_plan_owner_evidence_extraction(
                                context=context,
                                payload=payload,
                                item=item,
                                reports=chunk,
                                chunk_index=chunk_index,
                                attempt=attempt,
                            )
                            extracted_reports.extend(chunk_reports)
                            break
                        except Exception as exc:
                            last_exc = exc
                            if attempt >= max(1, max_attempts):
                                raise
                            if retry_delay_seconds > 0:
                                await asyncio.sleep(retry_delay_seconds)
                    else:
                        if last_exc is not None:
                            raise last_exc
                owner_evidence = self._build_dept_plan_owner_evidence_summary(
                    item=item,
                    owner_reports=owner_reports,
                    extracted_reports=extracted_reports,
                )
                item["owner_weekly_llm_evidence"] = owner_evidence
                metadata["cache_store_count"] += self._store_dept_plan_owner_evidence_cache(
                    item=item,
                    owner_reports=owner_reports,
                    owner_evidence=owner_evidence,
                )
                return item, None
            except Exception as exc:
                item["owner_evidence_extraction_failed"] = True
                error = {
                    "plan_id": self._dept_plan_followup_id(item),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                metadata["failure_count"] += 1
                metadata["failures"].append(error)
                failed = self._failed_dept_plan_judgement(
                    item=item,
                    reason=(
                        "负责人周报证据抽取失败，不能把该计划计为证据不足；"
                        "需要重新触发该计划的负责人周报抽取和完成判断。"
                    ),
                )
                return item, failed

        async def run_with_semaphore(item: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
            async with semaphore:
                return await run_item(item)

        results = await asyncio.gather(*(run_with_semaphore(item) for item in prepared))
        extracted_followups = [item for item, _failed in results]
        failed_judgements = [failed for _item, failed in results if failed is not None]
        metadata["failures"] = metadata["failures"][:10]
        return extracted_followups, metadata, failed_judgements

    async def _attach_dept_plan_owner_evidence_extractions_by_owner_group(
        self,
        *,
        followups: list[dict[str, Any]],
        owner_report_pool: list[dict[str, Any]],
        context: ContextStore,
        payload: dict[str, Any],
        metadata: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
        metadata["grouped_extraction"] = True
        metadata["unique_owner_report_count"] = len(owner_report_pool)
        report_by_id = {
            str(report.get("report_id") or ""): report
            for report in owner_report_pool
            if str(report.get("report_id") or "").strip()
        }
        plans_by_owner = self._dept_plan_owner_groups_for_evidence(followups=followups, report_by_id=report_by_id)
        metadata["owner_group_count"] = len(plans_by_owner)
        max_attempts = self._env_int("MYSQL_DEPT_PLAN_LLM_EVIDENCE_MAX_ATTEMPTS", 2)
        retry_delay_seconds = self._env_float("MYSQL_DEPT_PLAN_LLM_EVIDENCE_RETRY_DELAY_SECONDS", 1.0)
        chunk_size = max(1, self._env_int("MYSQL_DEPT_PLAN_LLM_EVIDENCE_CHUNK_SIZE", 12))
        plan_chunk_size = max(1, self._env_int("MYSQL_DEPT_PLAN_LLM_EVIDENCE_PLAN_CHUNK_SIZE", 4))
        concurrency = max(1, self._env_int("MYSQL_DEPT_PLAN_LLM_EVIDENCE_CONCURRENCY", 5))
        metadata["owner_report_chunk_size"] = chunk_size
        metadata["owner_plan_chunk_size"] = plan_chunk_size
        metadata["owner_evidence_concurrency"] = concurrency
        semaphore = asyncio.Semaphore(concurrency)
        failed_by_plan_id: dict[str, dict[str, Any]] = {}

        for item in followups:
            owner_reports = self._dept_plan_owner_weekly_reports_for_evidence(item, report_by_id=report_by_id)
            metadata["owner_report_count"] += len(owner_reports)
            if owner_reports:
                metadata["plans_with_owner_reports"] += 1

        for item in followups:
            owner_reports = self._dept_plan_owner_weekly_reports_for_evidence(item, report_by_id=report_by_id)
            if not owner_reports:
                item["owner_weekly_llm_evidence"] = self._dept_plan_empty_owner_evidence(item=item)
                continue
            cached_evidence = self._load_dept_plan_owner_evidence_cache(item=item, owner_reports=owner_reports)
            if cached_evidence is not None:
                item["owner_weekly_llm_evidence"] = cached_evidence
                metadata["cache_hit_count"] += 1

        async def run_group(owner_name: str, group_items: list[dict[str, Any]]) -> None:
            pending_items = [item for item in group_items if not isinstance(item.get("owner_weekly_llm_evidence"), Mapping)]
            if not pending_items:
                return
            request_chunk_index = 0
            for plan_chunk in self._chunk_dept_plan_owner_items(pending_items, size=plan_chunk_size):
                owner_reports = self._dept_plan_owner_reports_for_group(
                    items=plan_chunk,
                    report_by_id=report_by_id,
                    owner_name=owner_name,
                )
                if not owner_reports:
                    for item in plan_chunk:
                        item["owner_weekly_llm_evidence"] = self._dept_plan_empty_owner_evidence(item=item)
                    continue
                extracted_by_plan: dict[str, list[dict[str, Any]]] = {
                    self._dept_plan_followup_id(item): []
                    for item in plan_chunk
                }
                try:
                    for chunk in self._chunk_dept_plan_owner_reports(owner_reports, size=chunk_size):
                        request_chunk_index += 1
                        last_exc: Exception | None = None
                        for attempt in range(1, max(1, max_attempts) + 1):
                            try:
                                metadata["llm_call_count"] += 1
                                chunk_items = await self._invoke_dept_plan_owner_group_evidence_extraction(
                                    context=context,
                                    payload=payload,
                                    owner_name=owner_name,
                                    items=plan_chunk,
                                    reports=chunk,
                                    chunk_index=request_chunk_index,
                                    attempt=attempt,
                                )
                                for extracted in chunk_items:
                                    plan_id = str(extracted.get("plan_id") or "")
                                    if plan_id in extracted_by_plan:
                                        extracted_by_plan[plan_id].append(extracted)
                                break
                            except Exception as exc:
                                last_exc = exc
                                if attempt >= max(1, max_attempts):
                                    raise
                                if retry_delay_seconds > 0:
                                    await asyncio.sleep(retry_delay_seconds)
                        else:
                            if last_exc is not None:
                                raise last_exc
                    for item in plan_chunk:
                        plan_id = self._dept_plan_followup_id(item)
                        item_reports = self._dept_plan_owner_weekly_reports_for_evidence(item, report_by_id=report_by_id)
                        plan_extracted = list(extracted_by_plan.get(plan_id, []))
                        extracted_report_ids = {str(report.get("report_id") or "") for report in plan_extracted}
                        for report in item_reports:
                            report_id = str(report.get("report_id") or "")
                            if report_id and report_id not in extracted_report_ids:
                                plan_extracted.append(
                                    {
                                        "plan_id": plan_id,
                                        "report_id": report_id,
                                        "report_date": report.get("日期") or "",
                                        "submitter": report.get("提交人") or "",
                                        "is_related": False,
                                        "evidence_snippets": [],
                                        "completion_signal": "",
                                        "blockage_signal": "",
                                        "relation_reason": "该周报未返回相关证据。",
                                    }
                                )
                        owner_evidence = self._build_dept_plan_owner_evidence_summary(
                            item=item,
                            owner_reports=item_reports,
                            extracted_reports=plan_extracted,
                        )
                        item["owner_weekly_llm_evidence"] = owner_evidence
                        metadata["cache_store_count"] += self._store_dept_plan_owner_evidence_cache(
                            item=item,
                            owner_reports=item_reports,
                            owner_evidence=owner_evidence,
                        )
                except Exception as exc:
                    for item in plan_chunk:
                        item["owner_evidence_extraction_failed"] = True
                        plan_id = self._dept_plan_followup_id(item)
                        error = {
                            "plan_id": plan_id,
                            "owner_name": owner_name,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                        metadata["failure_count"] += 1
                        metadata["failures"].append(error)
                        failed_by_plan_id[plan_id] = self._failed_dept_plan_judgement(
                            item=item,
                            reason=(
                                "负责人周报证据抽取失败，不能把该计划计为证据不足；"
                                "需要重新触发该负责人计划小批次的周报抽取和完成判断。"
                            ),
                        )

        async def run_group_with_semaphore(owner_name: str, group_items: list[dict[str, Any]]) -> None:
            async with semaphore:
                await run_group(owner_name, group_items)

        await asyncio.gather(
            *(
                run_group_with_semaphore(owner_name, group_items)
                for owner_name, group_items in plans_by_owner.items()
            )
        )
        for item in followups:
            if not isinstance(item.get("owner_weekly_llm_evidence"), Mapping) and not item.get("owner_evidence_extraction_failed"):
                item["owner_weekly_llm_evidence"] = self._dept_plan_empty_owner_evidence(item=item)
        metadata["failures"] = metadata["failures"][:10]
        return followups, metadata, list(failed_by_plan_id.values())

    async def _invoke_dept_plan_owner_group_evidence_extraction(
        self,
        *,
        context: ContextStore,
        payload: dict[str, Any],
        owner_name: str,
        items: list[Mapping[str, Any]],
        reports: list[Mapping[str, Any]],
        chunk_index: int,
        attempt: int,
    ) -> list[dict[str, Any]]:
        compact_items = [
            {
                "plan_id": self._dept_plan_followup_id(item),
                "部门": str(item.get("department") or "未填写部门"),
                "负责人": str(item.get("owner_user") or ""),
                "计划月份": str(item.get("plan_month") or ""),
                "截止日期": str(item.get("due_date") or ""),
                "周次目标": str(item.get("target") or ""),
                "计划内容": str(item.get("plan_text") or "").strip(),
                "完成证据标准": self._dept_plan_completion_evidence_standards(item.get("plan_text")),
            }
            for item in items
        ]
        input_payload = {
            "owner_name": owner_name,
            "plans": compact_items,
            "owner_weekly_reports": reports,
            "rules": [
                "这些周报是按负责人和日期范围从 weekly_reports 主表取出的完整原文，不得因为缺少关键词就忽略。",
                "必须对每条计划逐条阅读本批周报原文，判断每份周报是否和该计划语义相关。",
                "相关时必须摘录原文证据片段，不要改写成总结。",
                "只证明准备、讨论、推进的内容不能当作计划完成证据。",
                "输出 items 至少覆盖所有相关的 plan_id/report_id；不相关且无证据的组合可以不输出。",
            ],
        }
        system_prompt = (
            "你是三七计划负责人周报证据抽取器。\n"
            "只依据输入中的同一负责人计划组和负责人周报原文工作，不得引入外部知识。\n"
            "你必须以计划为单位判断每份周报是否相关；不允许依赖关键词过滤。\n"
            "如果相关，请从原文摘录能证明完成、部分推进、卡点或未完成的短句或短段；不要改写原文。\n"
            "必须通过函数返回结构化结果，items 使用 plan_id 和 report_id 对应输入。"
        )
        user_prompt = (
            "请抽取以下负责人计划组周报中与各计划相关的原文证据。\n"
            "Dept Plan Owner Group Evidence JSON:\n"
            f"{json.dumps(input_payload, ensure_ascii=False, indent=2)}"
        )
        request = LLMRequest(
            prompt=user_prompt,
            system_prompt=system_prompt,
            messages=[
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ],
            model_name=str(payload.get("model_name") or self.default_model_name or "").strip() or None,
            temperature=0.0,
            max_tokens=self._optional_positive_int(payload.get("evidence_extract_max_tokens")),
            timeout_seconds=self._env_int("MYSQL_DEPT_PLAN_LLM_EVIDENCE_TIMEOUT_SECONDS", 120),
            request_id=context.runtime.request_id,
            session_id=context.runtime.session_id,
            trace_id=(
                f"{self.build_trace_id(context=context)}:dept_plan_owner_group_evidence:"
                f"{owner_name}:chunk:{chunk_index}:attempt:{attempt}"
            ),
            prompt_name="mysql_dept_plan_owner_group_evidence_extraction_prompt",
            prompt_version="v1",
            response_schema_name=DeptPlanOwnerGroupEvidenceExtractionOutput.__name__,
            response_schema_version="v1",
            max_validation_retries=1,
            metadata={
                "tool_name": self.name,
                "operation": "mysql_dept_plan_owner_group_evidence_extraction",
                "owner_name": owner_name,
                "chunk_index": chunk_index,
                "attempt": attempt,
                "plan_count": len(items),
                "report_count": len(reports),
            },
        )
        function_schema = LLMFunctionSchema(
            name="emit_dept_plan_owner_group_evidence_extraction",
            description="Extract original evidence snippets from one owner's weekly reports for multiple department plans.",
            parameters_schema=DeptPlanOwnerGroupEvidenceExtractionOutput.model_json_schema(),
            schema_name=DeptPlanOwnerGroupEvidenceExtractionOutput.__name__,
            schema_version="v1",
        )
        structured_result = await self.function_adapter.invoke_structured(
            client=self.client,
            request=request,
            function_schema=function_schema,
            output_model=DeptPlanOwnerGroupEvidenceExtractionOutput,
        )
        return self._normalize_dept_plan_owner_group_evidence_extraction(
            items=items,
            reports=reports,
            output=structured_result.output,
        )

    async def _invoke_dept_plan_owner_evidence_extraction(
        self,
        *,
        context: ContextStore,
        payload: dict[str, Any],
        item: Mapping[str, Any],
        reports: list[Mapping[str, Any]],
        chunk_index: int,
        attempt: int,
    ) -> list[dict[str, Any]]:
        plan_id = self._dept_plan_followup_id(item)
        input_payload = {
            "plan": {
                "plan_id": plan_id,
                "部门": str(item.get("department") or "未填写部门"),
                "负责人": str(item.get("owner_user") or ""),
                "计划月份": str(item.get("plan_month") or ""),
                "截止日期": str(item.get("due_date") or ""),
                "周次目标": str(item.get("target") or ""),
                "计划内容": str(item.get("plan_text") or "").strip(),
                "完成证据标准": self._dept_plan_completion_evidence_standards(item.get("plan_text")),
            },
            "owner_weekly_reports": reports,
            "rules": [
                "这些周报是按负责人和日期范围取出的，不得因为缺少关键词就忽略。",
                "必须逐条阅读完整周报原文，判断是否和计划语义相关。",
                "相关时必须摘录原文证据片段，不要改写成总结。",
                "只证明准备、讨论、推进的内容不能当作计划完成证据。",
            ],
        }
        system_prompt = (
            "你是三七计划负责人周报证据抽取器。\n"
            "只依据输入中的计划和负责人周报原文工作，不得引入外部知识。\n"
            "你必须逐条判断每份周报是否与计划语义相关；不允许依赖关键词过滤。\n"
            "如果相关，请从原文摘录能证明完成、部分推进、卡点或未完成的短句或短段；不要改写原文。\n"
            "如果不相关，is_related 返回 false，并简短说明原因。\n"
            "必须通过函数返回结构化结果，reports 要覆盖每个输入 report_id。"
        )
        user_prompt = (
            "请抽取以下负责人周报中与计划相关的原文证据。\n"
            "Dept Plan Owner Evidence JSON:\n"
            f"{json.dumps(input_payload, ensure_ascii=False, indent=2)}"
        )
        request = LLMRequest(
            prompt=user_prompt,
            system_prompt=system_prompt,
            messages=[
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ],
            model_name=str(payload.get("model_name") or self.default_model_name or "").strip() or None,
            temperature=0.0,
            max_tokens=self._optional_positive_int(payload.get("evidence_extract_max_tokens")),
            timeout_seconds=self._env_int("MYSQL_DEPT_PLAN_LLM_EVIDENCE_TIMEOUT_SECONDS", 120),
            request_id=context.runtime.request_id,
            session_id=context.runtime.session_id,
            trace_id=(
                f"{self.build_trace_id(context=context)}:dept_plan_owner_evidence:"
                f"{plan_id}:chunk:{chunk_index}:attempt:{attempt}"
            ),
            prompt_name="mysql_dept_plan_owner_evidence_extraction_prompt",
            prompt_version="v1",
            response_schema_name=DeptPlanOwnerEvidenceExtractionOutput.__name__,
            response_schema_version="v1",
            max_validation_retries=1,
            metadata={
                "tool_name": self.name,
                "operation": "mysql_dept_plan_owner_evidence_extraction",
                "plan_id": plan_id,
                "chunk_index": chunk_index,
                "attempt": attempt,
                "report_count": len(reports),
            },
        )
        function_schema = LLMFunctionSchema(
            name="emit_dept_plan_owner_evidence_extraction",
            description="Extract original evidence snippets from owner weekly reports for one department plan.",
            parameters_schema=DeptPlanOwnerEvidenceExtractionOutput.model_json_schema(),
            schema_name=DeptPlanOwnerEvidenceExtractionOutput.__name__,
            schema_version="v1",
        )
        structured_result = await self.function_adapter.invoke_structured(
            client=self.client,
            request=request,
            function_schema=function_schema,
            output_model=DeptPlanOwnerEvidenceExtractionOutput,
        )
        return self._normalize_dept_plan_owner_evidence_extraction(
            item=item,
            reports=reports,
            output=structured_result.output,
        )

    async def _invoke_dept_plan_judge_batch(
        self,
        *,
        context: ContextStore,
        payload: dict[str, Any],
        batch_index: int,
        batch_items: list[Mapping[str, Any]],
        attempt: int = 1,
        compact_mode: str = "primary",
        batch_stage: str = "primary",
    ) -> list[dict[str, Any]]:
        if len(batch_items) == 1:
            return await self._invoke_dept_plan_judge_single(
                context=context,
                payload=payload,
                batch_index=batch_index,
                item=batch_items[0],
                attempt=attempt,
                compact_mode=compact_mode,
                batch_stage=batch_stage,
            )
        compact_items = [
            self._compact_dept_plan_followup_for_llm(item, mode=compact_mode)
            for item in batch_items
        ]
        system_prompt = self._dept_plan_judge_system_prompt(single=False)
        user_prompt = (
            "请判断以下三七计划/部门计划是否完成。\n"
            "Dept Plan Batch JSON:\n"
            f"{json.dumps({'batch_index': batch_index, 'items': compact_items}, ensure_ascii=False, indent=2)}"
        )
        request = LLMRequest(
            prompt=user_prompt,
            system_prompt=system_prompt,
            messages=[
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ],
            model_name=str(payload.get("model_name") or self.default_model_name or "").strip() or None,
            temperature=0.0,
            max_tokens=self._optional_positive_int(payload.get("judge_max_tokens")),
            timeout_seconds=self._env_int("MYSQL_DEPT_PLAN_LLM_JUDGE_TIMEOUT_SECONDS", 120),
            request_id=context.runtime.request_id,
            session_id=context.runtime.session_id,
            trace_id=(
                f"{self.build_trace_id(context=context)}:dept_plan_batch:"
                f"{batch_index}:{batch_stage}:attempt:{attempt}"
            ),
            prompt_name="mysql_dept_plan_batch_judgement_prompt",
            prompt_version="v1",
            response_schema_name=DeptPlanBatchJudgementOutput.__name__,
            response_schema_version="v1",
            max_validation_retries=1,
            metadata={
                "tool_name": self.name,
                "operation": "mysql_dept_plan_batch_judgement",
                "batch_index": batch_index,
                "batch_stage": batch_stage,
                "attempt": attempt,
                "batch_item_count": len(batch_items),
                "compact_mode": compact_mode,
            },
        )
        function_schema = LLMFunctionSchema(
            name="emit_dept_plan_batch_judgement",
            description="Return structured semantic completion judgements for department plan evidence.",
            parameters_schema=DeptPlanBatchJudgementOutput.model_json_schema(),
            schema_name=DeptPlanBatchJudgementOutput.__name__,
            schema_version="v1",
        )
        structured_result = await self.function_adapter.invoke_structured(
            client=self.client,
            request=request,
            function_schema=function_schema,
            output_model=DeptPlanBatchJudgementOutput,
        )
        normalized = self._normalize_dept_plan_batch_judgements(
            batch_items=batch_items,
            output=structured_result.output,
        )
        missing_ids = [
            item["plan_id"]
            for item in normalized
            if item.get("status") == DEPT_PLAN_JUDGEMENT_FAILED_STATUS
        ]
        if missing_ids:
            raise ValueError(f"LLM 未返回这些计划的判断：{', '.join(missing_ids)}")
        return normalized

    async def _invoke_dept_plan_judge_single(
        self,
        *,
        context: ContextStore,
        payload: dict[str, Any],
        batch_index: int,
        item: Mapping[str, Any],
        attempt: int = 1,
        compact_mode: str = "primary",
        batch_stage: str = "primary",
    ) -> list[dict[str, Any]]:
        compact_item = self._compact_dept_plan_followup_for_llm(item, mode=compact_mode)
        system_prompt = self._dept_plan_judge_system_prompt(single=True)
        user_prompt = (
            "请判断以下单条三七计划/部门计划是否完成。\n"
            "Dept Plan Item JSON:\n"
            f"{json.dumps({'batch_index': batch_index, 'item': compact_item}, ensure_ascii=False, indent=2)}"
        )
        request = LLMRequest(
            prompt=user_prompt,
            system_prompt=system_prompt,
            messages=[
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ],
            model_name=str(payload.get("model_name") or self.default_model_name or "").strip() or None,
            temperature=0.0,
            max_tokens=self._optional_positive_int(payload.get("judge_max_tokens")),
            timeout_seconds=self._env_int("MYSQL_DEPT_PLAN_LLM_JUDGE_TIMEOUT_SECONDS", 120),
            request_id=context.runtime.request_id,
            session_id=context.runtime.session_id,
            trace_id=(
                f"{self.build_trace_id(context=context)}:dept_plan_single:"
                f"{batch_index}:{batch_stage}:attempt:{attempt}"
            ),
            prompt_name="mysql_dept_plan_single_judgement_prompt",
            prompt_version="v1",
            response_schema_name=DeptPlanSingleJudgementOutput.__name__,
            response_schema_version="v1",
            max_validation_retries=1,
            metadata={
                "tool_name": self.name,
                "operation": "mysql_dept_plan_batch_judgement",
                "batch_index": batch_index,
                "batch_stage": batch_stage,
                "attempt": attempt,
                "batch_item_count": 1,
                "compact_mode": compact_mode,
                "single_item_schema": True,
            },
        )
        function_schema = LLMFunctionSchema(
            name="emit_dept_plan_single_judgement",
            description="Return one structured semantic completion judgement for one department plan item.",
            parameters_schema=DeptPlanSingleJudgementOutput.model_json_schema(),
            schema_name=DeptPlanSingleJudgementOutput.__name__,
            schema_version="v1",
        )
        structured_result = await self.function_adapter.invoke_structured(
            client=self.client,
            request=request,
            function_schema=function_schema,
            output_model=DeptPlanSingleJudgementOutput,
        )
        output = structured_result.output
        batch_output = DeptPlanBatchJudgementOutput(
            judgements=[
                DeptPlanBatchJudgementItem(
                    plan_id=output.plan_id,
                    status=output.status,
                    reason=output.reason,
                    evidence=output.evidence,
                )
            ]
        )
        normalized = self._normalize_dept_plan_batch_judgements(
            batch_items=[item],
            output=batch_output,
        )
        if normalized and normalized[0].get("status") == DEPT_PLAN_JUDGEMENT_FAILED_STATUS:
            raise ValueError(f"LLM 未返回该计划的判断：{self._dept_plan_followup_id(item)}")
        return normalized

    def _dept_plan_judge_system_prompt(self, *, single: bool) -> str:
        output_instruction = (
            "必须通过函数返回结构化结果，字段直接对应这 1 条输入计划。"
            if single
            else "必须通过函数返回结构化结果，judgements 与输入 items 一一对应。"
        )
        return (
            "你是三七计划/部门月度计划完成情况语义判断器。\n"
            "只依据输入 JSON 中每条计划、第一层LLM抽出的负责人周报原文证据片段、负责人本人月度考核记录、OPL问题闭环证据和少量非负责人协作辅助判断，不得引入外部知识，不得编造；plan_id 仅用于对应输出。\n"
            "status 只能是：已完成、部分完成、未完成、证据不足。\n"
            "负责人本人周报已经先由第一层LLM按负责人和日期范围逐条读取完整原文，并抽取原文证据片段；最终判断必须优先依据这些负责人原文证据片段。\n"
            "负责人本人月度考核记录是按计划负责人姓名从 employee_self_eval_reports/items 直接查出的该员工当月考核表，不是部门候选搜索结果；必须只把对应负责人的考核内容作为该计划证据，不得用其他员工考核替代。\n"
            "OPL问题闭环证据来自 OPL 问题清单跟踪表；未闭环 OPL 只能作为风险/卡点证据，不能单独等同计划未完成；已闭环 OPL 只有在解决措施或最新进展明确对应计划目标时才可作为完成判断证据。\n"
            "不能因为负责人证据缺少关键词就排除；关键词只可作为非负责人协作周报的辅助排序线索，不能替代语义判断。\n"
            "负责人周报只能作为候选证据，不能因为提交人是负责人就直接判已完成；只有原文证据明确覆盖计划目标且表达完成结果时才能判已完成。\n"
            "非负责人协作周报只能作为辅助证据；低可信、按时间兜底、只命中泛词的协作材料不能单独判定已完成。\n"
            "备选复核材料中，负责人本人提交且标注可作为补充判断证据的材料，需要纳入判断；低可信、非负责人、只命中泛词的备选材料只能作为人工复核线索，不能单独判定已完成。\n"
            "如果输入项包含“完成证据标准”，必须优先按该标准判断；只证明准备、推进、环境部署或讨论的证据，不能替代计划目标本身的完成证据。\n"
            "如果输入项标注为压缩救援证据，说明上一次结构化输出失败；你仍需基于压缩后的全部候选摘要判断，必须返回合法结构化结果。\n"
            "采购/购买/新增购置类计划：需看到下单、采购完成、到货、验收、入库、付款、合同/供应商确认等采购闭环证据；环境部署、测试、培训、数据训练或使用痕迹，除非明确说明采购设备已到货/已验收，否则不能单独证明“购买”已完成。\n"
            "测试/验证/冒烟/回归/转测/问题单类计划：需看到实际执行测试/验证/回归、测试结果、测试报告、缺陷回归关闭或问题单处理闭环；仅准备测试、计划测试、待测试、提出问题但未验证闭环，不能判已完成。\n"
            "制度/流程/规范/文档/培训类计划：需看到制度流程已定稿、评审通过、发布归档、培训已执行或落地结果；仅讨论、梳理、草稿、拟定、待评审、待培训，不能判已完成。\n"
            "周报证据、负责人月度考核或已闭环 OPL 的解决进展明确覆盖计划目标且没有未完成/待处理表述时，可判已完成；即使周报和负责人月度考核都没有出现该计划，只要高相关已闭环 OPL 的解决措施明确覆盖计划目标，也应判已完成；只覆盖部分动作、仍在进行、存在卡点或超出截止周期时判部分完成；负责人月度考核中的未完成项、未解决项、原因说明或未闭环 OPL 明确对应计划风险时，可作为未完成/部分完成风险依据；没有相关证据时判证据不足。\n"
            "reason 和 evidence 必须使用面向业务用户的自然中文，不得出现英文变量名、JSON key、true/false、low/medium/high 等内部字段值。\n"
            f"{output_instruction}"
        )

    async def _invoke_dept_plan_merge_report(
        self,
        *,
        context: ContextStore,
        payload: dict[str, Any],
        source_output: Mapping[str, Any],
        ordered_judgements: list[dict[str, Any]],
        batch_errors: list[dict[str, Any]],
    ) -> str:
        report_payload = self._build_dept_plan_merge_payload(
            context=context,
            source_output=source_output,
            ordered_judgements=ordered_judgements,
            batch_errors=batch_errors,
        )
        system_prompt = (
            "你是三七计划完成情况总报告生成器。\n"
            "只依据输入 JSON 中的总体统计、部门统计和每条计划的结构化判断生成中文报告。\n"
            "不得重新判定计划 status，不得修改每条计划的 status，不得编造证据。\n"
            "报告的数据来源或判定方式必须明确包含周报证据、负责人本人月度考核记录和 OPL 问题清单/问题闭环证据；如输入 JSON 有 OPL 统计，需客观说明 OPL 已纳入风险/闭环判断；未闭环 OPL 不得被夸大为单独未完成依据。\n"
            "必须明确区分业务计划项状态与 Agent Runtime 执行状态；不要把证据不足或业务未完成表述成系统任务未完成。\n"
            "输出应先给总体结论，再按部门汇总；明确未完成和证据不足清单由后端按结构化结果追加或替换。\n"
            "最终文本必须面向业务用户，不得出现英文变量名、JSON key、true/false、low/medium/high 等内部字段值。\n"
            "必须通过函数 emit_text_generation_output 返回最终报告文本。"
        )
        user_prompt = (
            "请生成三七计划完成情况总报告。不要展开全部明细表。\n"
            "Dept Plan Report JSON:\n"
            f"{json.dumps(report_payload, ensure_ascii=False, indent=2)}"
        )
        request = LLMRequest(
            prompt=user_prompt,
            system_prompt=system_prompt,
            messages=[
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ],
            model_name=str(payload.get("model_name") or self.default_model_name or "").strip() or None,
            temperature=float(payload.get("merge_temperature", 0.2)),
            max_tokens=(
                self._optional_positive_int(payload.get("merge_max_tokens"))
                or self._env_int("MYSQL_DEPT_PLAN_LLM_MERGE_MAX_TOKENS", 9000)
            ),
            timeout_seconds=self._env_int("MYSQL_DEPT_PLAN_LLM_MERGE_TIMEOUT_SECONDS", 240),
            request_id=context.runtime.request_id,
            session_id=context.runtime.session_id,
            trace_id=f"{self.build_trace_id(context=context)}:dept_plan_merge_report",
            prompt_name="mysql_dept_plan_merge_report_prompt",
            prompt_version="v1",
            response_schema_name=TextGenerateToolOutput.__name__,
            response_schema_version="v1",
            max_validation_retries=1,
            metadata={
                "tool_name": self.name,
                "operation": "mysql_dept_plan_merge_report",
                "dept_plan_followup_count": len(ordered_judgements),
                "batch_error_count": len(batch_errors),
            },
        )
        function_schema = LLMFunctionSchema(
            name=self.function_name,
            description="Return the final department plan completion summary report text.",
            parameters_schema=TextGenerateToolOutput.model_json_schema(),
            schema_name=TextGenerateToolOutput.__name__,
            schema_version="v1",
        )
        structured_result = await self.function_adapter.invoke_structured(
            client=self.client,
            request=request,
            function_schema=function_schema,
            output_model=TextGenerateToolOutput,
        )
        return self._sanitize_dept_plan_user_text(structured_result.output.text.strip())

    def _resolve_mysql_weekly_plan_source(
        self,
        *,
        payload: dict[str, Any],
        context: ContextStore,
    ) -> tuple[str | None, dict[str, Any] | None]:
        requested_context = str(payload.get("context") or "")
        haystack = self._deterministic_report_haystack(payload=payload, context=context)
        asks_completion = self._asks_weekly_plan_completion(haystack)
        if self._asks_weekly_blockers(haystack):
            return None, None
        for key, output in context.task_results.items():
            if not isinstance(output, dict):
                continue
            if not isinstance(output.get("plan_followups"), list):
                continue
            requested_tracking_context = "plan_tracking_context_text" in requested_context
            requested_full_output_for_completion = key in requested_context and asks_completion
            if requested_tracking_context or requested_full_output_for_completion or asks_completion:
                return key, output
        return None, None

    async def _invoke_weekly_plan_judge_batch(
        self,
        *,
        context: ContextStore,
        payload: dict[str, Any],
        batch_index: int,
        batch_items: list[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        compact_items = [self._compact_weekly_plan_followup_for_llm(item) for item in batch_items]
        system_prompt = (
            "你是周报计划完成情况语义判断器。\n"
            "只依据输入 JSON 中每条计划、后续完成项和候选完成项判断，不得引入外部知识，不得编造。\n"
            "后端已经完成同人、部门和日期配对；你只做语义判断。\n"
            "status 只能是：已完成、部分完成、未完成、证据不足。\n"
            "有明确完成证据才判已完成；只体现部分动作或同项目不同任务时判部分完成；有后续记录但无相关证据判未完成；缺少后续记录或证据无法判断时判证据不足。\n"
            "必须通过函数返回结构化结果，judgements 与输入 items 一一对应。"
        )
        user_prompt = (
            "请判断以下周计划是否在后续完成记录中完成。\n"
            "Batch JSON:\n"
            f"{json.dumps({'batch_index': batch_index, 'items': compact_items}, ensure_ascii=False, indent=2)}"
        )
        request = LLMRequest(
            prompt=user_prompt,
            system_prompt=system_prompt,
            messages=[
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ],
            model_name=str(payload.get("model_name") or self.default_model_name or "").strip() or None,
            temperature=0.0,
            max_tokens=self._optional_positive_int(payload.get("judge_max_tokens")),
            timeout_seconds=self._env_int("MYSQL_WEEKLY_LLM_JUDGE_TIMEOUT_SECONDS", 120),
            request_id=context.runtime.request_id,
            session_id=context.runtime.session_id,
            trace_id=f"{self.build_trace_id(context=context)}:weekly_plan_batch:{batch_index}",
            prompt_name="mysql_weekly_plan_batch_judgement_prompt",
            prompt_version="v1",
            response_schema_name=WeeklyPlanBatchJudgementOutput.__name__,
            response_schema_version="v1",
            max_validation_retries=1,
            metadata={
                "tool_name": self.name,
                "operation": "mysql_weekly_plan_batch_judgement",
                "batch_index": batch_index,
                "batch_item_count": len(batch_items),
            },
        )
        function_schema = LLMFunctionSchema(
            name="emit_weekly_plan_batch_judgement",
            description="Return structured semantic completion judgements for a batch of weekly plan followups.",
            parameters_schema=WeeklyPlanBatchJudgementOutput.model_json_schema(),
            schema_name=WeeklyPlanBatchJudgementOutput.__name__,
            schema_version="v1",
        )
        structured_result = await self.function_adapter.invoke_structured(
            client=self.client,
            request=request,
            function_schema=function_schema,
            output_model=WeeklyPlanBatchJudgementOutput,
        )
        return self._normalize_weekly_plan_batch_judgements(
            batch_items=batch_items,
            output=structured_result.output,
        )

    async def _invoke_weekly_plan_layered_merge_report(
        self,
        *,
        context: ContextStore,
        payload: dict[str, Any],
        source_output: Mapping[str, Any],
        ordered_judgements: list[dict[str, Any]],
        batch_errors: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any]]:
        person_groups = self._group_weekly_plan_judgements_by_person(ordered_judgements)
        person_summaries = await self._invoke_weekly_plan_person_summaries(
            context=context,
            payload=payload,
            source_output=source_output,
            person_groups=person_groups,
        )
        report_text = await self._invoke_weekly_plan_summary_report(
            context=context,
            payload=payload,
            source_output=source_output,
            ordered_judgements=ordered_judgements,
            person_summaries=person_summaries,
            batch_errors=batch_errors,
        )
        detail_text = self._format_mysql_weekly_plan_detail_section(
            source_output=source_output,
            ordered_judgements=ordered_judgements,
        )
        return f"{report_text.rstrip()}\n\n{detail_text}".strip(), {
            "person_summary_count": len(person_summaries),
        }

    async def _invoke_weekly_plan_person_summaries(
        self,
        *,
        context: ContextStore,
        payload: dict[str, Any],
        source_output: Mapping[str, Any],
        person_groups: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        group_size = self._env_int("MYSQL_WEEKLY_LLM_PERSON_SUMMARY_GROUP_SIZE", 4)
        concurrency = self._env_int("MYSQL_WEEKLY_LLM_PERSON_SUMMARY_CONCURRENCY", 4)
        batches = [
            person_groups[index : index + group_size]
            for index in range(0, len(person_groups), group_size)
        ]
        semaphore = asyncio.Semaphore(max(1, concurrency))
        summaries: list[dict[str, Any]] = []

        async def run_batch(batch_index: int, groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
            async with semaphore:
                return await self._invoke_weekly_plan_person_summary_batch(
                    context=context,
                    payload=payload,
                    source_output=source_output,
                    batch_index=batch_index,
                    groups=groups,
                )

        batch_results = await asyncio.gather(
            *(run_batch(batch_index, groups) for batch_index, groups in enumerate(batches, start=1))
        )
        for result in batch_results:
            summaries.extend(result)
        return summaries

    async def _invoke_weekly_plan_person_summary_batch(
        self,
        *,
        context: ContextStore,
        payload: dict[str, Any],
        source_output: Mapping[str, Any],
        batch_index: int,
        groups: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        summary_payload = {
            "batch_index": batch_index,
            "user_question": context.runtime.user_input,
            "date_range": self._weekly_plan_date_range_payload(source_output),
            "people": groups,
        }
        system_prompt = (
            "你是周计划完成人员小结生成器。\n"
            "只依据输入 JSON 中每个人的结构化计划判断结果生成中文小结。\n"
            "不得修改计划 status，不得重新判定完成状态，不得编造证据。\n"
            "每个人 summary 控制在 120 字以内；blocked_or_unfinished 最多 5 条；follow_up_suggestions 最多 3 条。\n"
            "必须通过函数 emit_weekly_plan_person_summary 返回结构化结果。"
        )
        user_prompt = (
            "请按人员汇总计划完成情况、长期未完成/卡住事项和建议跟进点。\n"
            "Person Summary JSON:\n"
            f"{json.dumps(summary_payload, ensure_ascii=False, indent=2)}"
        )
        request = LLMRequest(
            prompt=user_prompt,
            system_prompt=system_prompt,
            messages=[
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ],
            model_name=str(payload.get("model_name") or self.default_model_name or "").strip() or None,
            temperature=float(payload.get("person_summary_temperature", 0.2)),
            max_tokens=(
                self._optional_positive_int(payload.get("person_summary_max_tokens"))
                or self._env_int("MYSQL_WEEKLY_LLM_PERSON_SUMMARY_MAX_TOKENS", 6000)
            ),
            timeout_seconds=self._env_int("MYSQL_WEEKLY_LLM_PERSON_SUMMARY_TIMEOUT_SECONDS", 180),
            request_id=context.runtime.request_id,
            session_id=context.runtime.session_id,
            trace_id=f"{self.build_trace_id(context=context)}:weekly_plan_person_summary:{batch_index}",
            prompt_name="mysql_weekly_plan_person_summary_prompt",
            prompt_version="v1",
            response_schema_name=WeeklyPlanPersonSummaryOutput.__name__,
            response_schema_version="v1",
            max_validation_retries=1,
            metadata={
                "tool_name": self.name,
                "operation": "mysql_weekly_plan_person_summary",
                "batch_index": batch_index,
                "person_count": len(groups),
            },
        )
        function_schema = LLMFunctionSchema(
            name="emit_weekly_plan_person_summary",
            description="Return person-level weekly plan completion summaries.",
            parameters_schema=WeeklyPlanPersonSummaryOutput.model_json_schema(),
            schema_name=WeeklyPlanPersonSummaryOutput.__name__,
            schema_version="v1",
        )
        structured_result = await self.function_adapter.invoke_structured(
            client=self.client,
            request=request,
            function_schema=function_schema,
            output_model=WeeklyPlanPersonSummaryOutput,
        )
        return [
            {
                "user_name": item.user_name,
                "department": item.department,
                "summary": self._clip_text(item.summary, 220),
                "blocked_or_unfinished": [self._clip_text(value, 180) for value in item.blocked_or_unfinished[:5]],
                "follow_up_suggestions": [self._clip_text(value, 180) for value in item.follow_up_suggestions[:3]],
            }
            for item in structured_result.output.people
        ]

    async def _invoke_weekly_plan_summary_report(
        self,
        *,
        context: ContextStore,
        payload: dict[str, Any],
        source_output: Mapping[str, Any],
        ordered_judgements: list[dict[str, Any]],
        person_summaries: list[dict[str, Any]],
        batch_errors: list[dict[str, Any]],
    ) -> str:
        report_payload = self._build_weekly_plan_layered_report_payload(
            context=context,
            source_output=source_output,
            ordered_judgements=ordered_judgements,
            person_summaries=person_summaries,
            batch_errors=batch_errors,
        )
        system_prompt = (
            "你是周报计划完成情况总报告生成器。\n"
            "只依据输入 JSON 中的总体统计、人员小结和重点未完成项生成中文报告。\n"
            "不得重新判定计划 status，不得编造证据。\n"
            "输出只包含总体结论、按人汇总、长期未完成/卡住重点、跟进建议；不要输出完整明细表，完整明细由后端追加。\n"
            "必须通过函数 emit_text_generation_output 返回最终报告文本。"
        )
        user_prompt = (
            "请生成计划完成情况总报告。不要展开全部明细表。\n"
            "Layered Report JSON:\n"
            f"{json.dumps(report_payload, ensure_ascii=False, indent=2)}"
        )
        request = LLMRequest(
            prompt=user_prompt,
            system_prompt=system_prompt,
            messages=[
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ],
            model_name=str(payload.get("model_name") or self.default_model_name or "").strip() or None,
            temperature=float(payload.get("merge_temperature", 0.2)),
            max_tokens=(
                self._optional_positive_int(payload.get("merge_max_tokens"))
                or self._env_int("MYSQL_WEEKLY_LLM_MERGE_MAX_TOKENS", 12000)
            ),
            timeout_seconds=self._env_int("MYSQL_WEEKLY_LLM_MERGE_TIMEOUT_SECONDS", 240),
            request_id=context.runtime.request_id,
            session_id=context.runtime.session_id,
            trace_id=f"{self.build_trace_id(context=context)}:weekly_plan_layered_merge_report",
            prompt_name="mysql_weekly_plan_layered_merge_report_prompt",
            prompt_version="v1",
            response_schema_name=TextGenerateToolOutput.__name__,
            response_schema_version="v1",
            max_validation_retries=1,
            metadata={
                "tool_name": self.name,
                "operation": "mysql_weekly_plan_layered_merge_report",
                "plan_followup_count": len(ordered_judgements),
                "person_summary_count": len(person_summaries),
                "batch_error_count": len(batch_errors),
            },
        )
        function_schema = LLMFunctionSchema(
            name=self.function_name,
            description="Return the final weekly plan completion summary report text.",
            parameters_schema=TextGenerateToolOutput.model_json_schema(),
            schema_name=TextGenerateToolOutput.__name__,
            schema_version="v1",
        )
        structured_result = await self.function_adapter.invoke_structured(
            client=self.client,
            request=request,
            function_schema=function_schema,
            output_model=TextGenerateToolOutput,
        )
        return structured_result.output.text.strip()

    async def _invoke_weekly_plan_merge_report(
        self,
        *,
        context: ContextStore,
        payload: dict[str, Any],
        source_output: Mapping[str, Any],
        ordered_judgements: list[dict[str, Any]],
        batch_errors: list[dict[str, Any]],
    ) -> str:
        report_payload = self._build_weekly_plan_merge_payload(
            context=context,
            source_output=source_output,
            ordered_judgements=ordered_judgements,
            batch_errors=batch_errors,
        )
        system_prompt = (
            "你是周报计划完成情况报告生成器。\n"
            "你只能依据输入 JSON 中的日期、统计和每条计划的结构化判断生成中文报告。\n"
            "不得重新判定完成状态，不得修改每条计划的 status，不得编造证据。\n"
            "汇总数字必须与 JSON.summary.status_counts 和 total_plans 保持一致。\n"
            "报告应先给总体结论，再列出未完全完成/需关注项，最后给出全部计划明细。\n"
            "全部计划明细不得遗漏 items 中任一计划；若条目较多，也至少列出每条计划的人员、状态、计划内容和判断依据。\n"
            "必须通过函数 emit_text_generation_output 返回最终报告文本。"
        )
        user_prompt = (
            "请根据以下结构化周计划判断结果生成最终报告。\n"
            "Report JSON:\n"
            f"{json.dumps(report_payload, ensure_ascii=False, indent=2)}"
        )
        request = LLMRequest(
            prompt=user_prompt,
            system_prompt=system_prompt,
            messages=[
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ],
            model_name=str(payload.get("model_name") or self.default_model_name or "").strip() or None,
            temperature=float(payload.get("merge_temperature", 0.2)),
            max_tokens=(
                self._optional_positive_int(payload.get("merge_max_tokens"))
                or self._env_int("MYSQL_WEEKLY_LLM_MERGE_MAX_TOKENS", 6000)
            ),
            timeout_seconds=self._env_int("MYSQL_WEEKLY_LLM_MERGE_TIMEOUT_SECONDS", 32000),
            request_id=context.runtime.request_id,
            session_id=context.runtime.session_id,
            trace_id=f"{self.build_trace_id(context=context)}:weekly_plan_merge_report",
            prompt_name="mysql_weekly_plan_merge_report_prompt",
            prompt_version="v1",
            response_schema_name=TextGenerateToolOutput.__name__,
            response_schema_version="v1",
            max_validation_retries=1,
            metadata={
                "tool_name": self.name,
                "operation": "mysql_weekly_plan_merge_report",
                "plan_followup_count": len(ordered_judgements),
                "batch_error_count": len(batch_errors),
            },
        )
        function_schema = LLMFunctionSchema(
            name=self.function_name,
            description="Return the final weekly plan completion report text.",
            parameters_schema=TextGenerateToolOutput.model_json_schema(),
            schema_name=TextGenerateToolOutput.__name__,
            schema_version="v1",
        )
        structured_result = await self.function_adapter.invoke_structured(
            client=self.client,
            request=request,
            function_schema=function_schema,
            output_model=TextGenerateToolOutput,
        )
        return structured_result.output.text.strip()

    def _build_weekly_plan_merge_payload(
        self,
        *,
        context: ContextStore,
        source_output: Mapping[str, Any],
        ordered_judgements: list[dict[str, Any]],
        batch_errors: list[dict[str, Any]],
    ) -> dict[str, Any]:
        date_range = source_output.get("query_date_range") if isinstance(source_output.get("query_date_range"), Mapping) else {}
        pairing_summary = source_output.get("pairing_summary") if isinstance(source_output.get("pairing_summary"), Mapping) else {}
        status_counts = self._weekly_plan_status_counts(ordered_judgements)
        unfinished_count = sum(status_counts.get(status, 0) for status in ("部分完成", "未完成", "证据不足"))
        return {
            "user_question": context.runtime.user_input,
            "date_range": {
                "plan_start": date_range.get("last_week_start", ""),
                "plan_end": date_range.get("last_week_end", ""),
                "completion_start": date_range.get("this_week_start", ""),
                "completion_end": date_range.get("this_week_end", ""),
            },
            "summary": {
                "total_plans": len(ordered_judgements),
                "weekly_pair_count": pairing_summary.get("weekly_pair_count", 0),
                "status_counts": status_counts,
                "unfinished_or_attention_count": unfinished_count,
                "batch_error_count": len(batch_errors),
                "generation_method": "大模型先分批判断每条计划完成状态，再由大模型基于结构化判断结果生成本报告。",
            },
            "items": [
                {
                    "index": index,
                    "user_name": self._clip_text(item.get("user_name"), 40),
                    "department": self._clip_text(item.get("department"), 60),
                    "plan_date": item.get("plan_date"),
                    "completion_date": item.get("completion_date"),
                    "status": item.get("status"),
                    "plan_text": self._clip_text(item.get("plan_text"), 180),
                    "reason": self._clip_text(item.get("reason"), 160),
                    "evidence": self._clip_text(item.get("evidence"), 120),
                }
                for index, item in enumerate(ordered_judgements, start=1)
            ],
            "batch_errors": batch_errors[:10],
        }

    def _build_weekly_plan_layered_report_payload(
        self,
        *,
        context: ContextStore,
        source_output: Mapping[str, Any],
        ordered_judgements: list[dict[str, Any]],
        person_summaries: list[dict[str, Any]],
        batch_errors: list[dict[str, Any]],
    ) -> dict[str, Any]:
        pairing_summary = source_output.get("pairing_summary") if isinstance(source_output.get("pairing_summary"), Mapping) else {}
        status_counts = self._weekly_plan_status_counts(ordered_judgements)
        unfinished = [
            item for item in ordered_judgements
            if self._normalize_weekly_plan_status(item.get("status")) in {"部分完成", "未完成", "证据不足"}
        ]
        return {
            "user_question": context.runtime.user_input,
            "date_range": self._weekly_plan_date_range_payload(source_output),
            "summary": {
                "total_plans": len(ordered_judgements),
                "weekly_pair_count": pairing_summary.get("weekly_pair_count", 0),
                "status_counts": status_counts,
                "unfinished_or_attention_count": len(unfinished),
                "batch_error_count": len(batch_errors),
                "generation_method": "大模型分批判断计划状态，按人员生成小结，再由大模型生成总报告；完整明细由后端基于结构化结果追加。",
            },
            "person_summaries": person_summaries,
            "top_unfinished_items": [
                {
                    "user_name": self._clip_text(item.get("user_name"), 40),
                    "department": self._clip_text(item.get("department"), 60),
                    "plan_date": item.get("plan_date"),
                    "completion_date": item.get("completion_date"),
                    "status": item.get("status"),
                    "plan_text": self._clip_text(item.get("plan_text"), 160),
                    "reason": self._clip_text(item.get("reason"), 160),
                }
                for item in unfinished[:80]
            ],
            "batch_errors": batch_errors[:10],
        }

    def _build_dept_plan_merge_payload(
        self,
        *,
        context: ContextStore,
        source_output: Mapping[str, Any],
        ordered_judgements: list[dict[str, Any]],
        batch_errors: list[dict[str, Any]],
    ) -> dict[str, Any]:
        query_scope = source_output.get("query_scope") if isinstance(source_output.get("query_scope"), Mapping) else {}
        pairing_summary = source_output.get("pairing_summary") if isinstance(source_output.get("pairing_summary"), Mapping) else {}
        status_counts = self._dept_plan_status_counts(ordered_judgements)
        unfinished = [
            item for item in ordered_judgements
            if self._normalize_dept_plan_status(item.get("status")) in {"部分完成", "未完成", "证据不足", DEPT_PLAN_JUDGEMENT_FAILED_STATUS}
        ]
        return {
            "user_question": context.runtime.user_input,
            "query_scope": {
                "month": query_scope.get("month", ""),
                "department": query_scope.get("department") or "全部部门",
                "weekly_evidence_start": query_scope.get("weekly_evidence_start", ""),
                "weekly_evidence_end": query_scope.get("weekly_evidence_end", ""),
                "include_opl": bool(query_scope.get("include_opl")),
            },
            "summary": {
                "total_plans": len(ordered_judgements),
                "status_counts": status_counts,
                "unfinished_or_attention_count": len(unfinished),
                "weekly_evidence_candidate_count": pairing_summary.get("weekly_evidence_candidate_count", 0),
                "self_eval_candidate_count": pairing_summary.get("self_eval_candidate_count", 0),
                "opl_issue_count": pairing_summary.get("opl_issue_count", 0),
                "opl_issue_candidate_count": pairing_summary.get("opl_issue_candidate_count", 0),
                "possible_opl_evidence_count": pairing_summary.get("possible_opl_evidence_count", 0),
                "open_opl_issue_candidate_count": pairing_summary.get("open_opl_issue_candidate_count", 0),
                "high_priority_open_opl_issue_candidate_count": pairing_summary.get(
                    "high_priority_open_opl_issue_candidate_count",
                    0,
                ),
                "plans_with_opl_issue_candidates": pairing_summary.get("plans_with_opl_issue_candidates", 0),
                "batch_error_count": len(batch_errors),
                "generation_method": "大模型分批判断每条三七计划状态，综合负责人周报、负责人月度考核记录和 OPL 问题闭环证据，再由大模型基于结构化判断结果生成总报告；完整明细由后端追加。",
            },
            "data_sources": self._dept_plan_data_sources_payload(source_output),
            "department_summaries": self._group_dept_plan_judgements_by_department(ordered_judgements),
            "top_unfinished_items": [
                {
                    "department": self._clip_text(item.get("department"), 60),
                    "owner_user": self._clip_text(item.get("owner_user"), 40),
                    "due_date": item.get("due_date"),
                    "target": item.get("target"),
                    "status": item.get("status"),
                    "plan_text": self._clip_text(item.get("plan_text"), 180),
                    "reason": self._clip_text(item.get("reason"), 180),
                    "evidence": self._clip_text(item.get("evidence"), 140),
                }
                for item in unfinished[:80]
            ],
            "batch_errors": batch_errors[:10],
        }

    def _dept_plan_data_sources_payload(self, source_output: Mapping[str, Any]) -> dict[str, Any]:
        query_scope = source_output.get("query_scope") if isinstance(source_output.get("query_scope"), Mapping) else {}
        pairing_summary = source_output.get("pairing_summary") if isinstance(source_output.get("pairing_summary"), Mapping) else {}
        return {
            "周报证据": {
                "范围": f"{query_scope.get('weekly_evidence_start', '')} 至 {query_scope.get('weekly_evidence_end', '')}",
                "负责人完整周报": pairing_summary.get("owner_weekly_report_count", 0),
                "协作辅助候选": pairing_summary.get("weekly_evidence_candidate_count", 0),
            },
            "负责人月度考核记录": {
                "负责人考核明细引用": pairing_summary.get("owner_self_eval_item_count", 0),
                "旧版自评候选": pairing_summary.get("self_eval_candidate_count", 0),
                "缺失负责人考核计划数": pairing_summary.get("plans_missing_owner_self_eval", 0),
            },
            "OPL问题闭环证据": {
                "已纳入": self._dept_plan_has_opl_source(source_output),
                "OPL问题取数": pairing_summary.get("opl_issue_count", 0),
                "计划相关OPL候选": pairing_summary.get("opl_issue_candidate_count", 0),
                "OPL备选复核": pairing_summary.get("possible_opl_evidence_count", 0),
                "未闭环OPL候选": pairing_summary.get("open_opl_issue_candidate_count", 0),
                "高优先级未闭环OPL候选": pairing_summary.get(
                    "high_priority_open_opl_issue_candidate_count",
                    0,
                ),
                "有关联OPL的计划数": pairing_summary.get("plans_with_opl_issue_candidates", 0),
                "使用规则": (
                    "未闭环 OPL 作为风险/卡点证据，不能单独等同计划未完成；"
                    "已闭环 OPL 只有解决进展对应计划目标时才作为完成判断证据。"
                ),
            },
        }

    def _dept_plan_has_opl_source(self, source_output: Mapping[str, Any]) -> bool:
        query_scope = source_output.get("query_scope") if isinstance(source_output.get("query_scope"), Mapping) else {}
        if query_scope.get("include_opl"):
            return True
        pairing_summary = source_output.get("pairing_summary") if isinstance(source_output.get("pairing_summary"), Mapping) else {}
        opl_summary_keys = (
            "opl_issue_count",
            "opl_issue_candidate_count",
            "possible_opl_evidence_count",
            "open_opl_issue_candidate_count",
            "high_priority_open_opl_issue_candidate_count",
            "plans_with_opl_issue_candidates",
        )
        if any(key in pairing_summary for key in opl_summary_keys):
            return True
        followups = source_output.get("dept_plan_followups")
        if isinstance(followups, list):
            return any(
                isinstance(item, Mapping)
                and (
                    "opl_issue_candidates" in item
                    or "possible_opl_evidence" in item
                    or "opl_match_audit" in item
                )
                for item in followups
            )
        return False

    def _format_dept_plan_data_source_line(self, source_output: Mapping[str, Any]) -> str:
        query_scope = source_output.get("query_scope") if isinstance(source_output.get("query_scope"), Mapping) else {}
        pairing_summary = source_output.get("pairing_summary") if isinstance(source_output.get("pairing_summary"), Mapping) else {}
        weekly_range = f"{query_scope.get('weekly_evidence_start', '')} 至 {query_scope.get('weekly_evidence_end', '')}".strip()
        weekly_part = f"周报证据（{weekly_range}）" if weekly_range != "至" else "周报证据"
        parts = [weekly_part, "负责人月度考核记录"]
        if self._dept_plan_has_opl_source(source_output):
            opl_detail = (
                f"OPL 问题清单/问题闭环证据（取数 {pairing_summary.get('opl_issue_count', 0)} 条，"
                f"计划相关候选 {pairing_summary.get('opl_issue_candidate_count', 0)} 条，"
                f"未闭环候选 {pairing_summary.get('open_opl_issue_candidate_count', 0)} 条）"
            )
            parts.append(opl_detail)
        return f"**数据来源：** {'、'.join(parts)}"

    def _ensure_dept_plan_report_data_source_note(
        self,
        *,
        report_text: str,
        source_output: Mapping[str, Any],
    ) -> str:
        text = str(report_text or "").strip()
        if not text:
            return text
        source_line = self._format_dept_plan_data_source_line(source_output)
        lines = text.splitlines()
        for index, line in enumerate(lines):
            if "数据来源" in line:
                lines[index] = source_line
                return "\n".join(lines).strip()

        insert_at = 1 if lines else 0
        while insert_at < len(lines) and not lines[insert_at].strip():
            insert_at += 1
        merged = [*lines[:insert_at], source_line, "", *lines[insert_at:]]
        return "\n".join(merged).strip()

    def _group_dept_plan_judgements_by_department(self, ordered_judgements: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in ordered_judgements:
            department = str(item.get("department") or "未填写部门")
            grouped.setdefault(department, []).append(item)
        summaries: list[dict[str, Any]] = []
        for department, items in grouped.items():
            status_counts = self._dept_plan_status_counts(items)
            unfinished = [
                item for item in items
                if self._normalize_dept_plan_status(item.get("status")) in {"部分完成", "未完成", "证据不足", DEPT_PLAN_JUDGEMENT_FAILED_STATUS}
            ]
            summaries.append(
                {
                    "department": department,
                    "total_plans": len(items),
                    "status_counts": status_counts,
                    "unfinished_or_attention_count": len(unfinished),
                    "representative_unfinished": [
                        {
                            "owner_user": self._clip_text(item.get("owner_user"), 40),
                            "status": item.get("status"),
                            "plan_text": self._clip_text(item.get("plan_text"), 150),
                            "reason": self._clip_text(item.get("reason"), 140),
                        }
                        for item in unfinished[:8]
                    ],
                }
            )
        return summaries

    def _weekly_plan_date_range_payload(self, source_output: Mapping[str, Any]) -> dict[str, Any]:
        date_range = source_output.get("query_date_range") if isinstance(source_output.get("query_date_range"), Mapping) else {}
        return {
            "plan_start": date_range.get("last_week_start", ""),
            "plan_end": date_range.get("last_week_end", ""),
            "completion_start": date_range.get("this_week_start", ""),
            "completion_end": date_range.get("this_week_end", ""),
        }

    def _group_weekly_plan_judgements_by_person(self, ordered_judgements: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for item in ordered_judgements:
            key = (str(item.get("user_name") or "未知人员"), str(item.get("department") or "未填写部门"))
            grouped.setdefault(key, []).append(item)
        groups: list[dict[str, Any]] = []
        for (user_name, department), items in grouped.items():
            status_counts = self._weekly_plan_status_counts(items)
            unfinished = [
                item for item in items
                if self._normalize_weekly_plan_status(item.get("status")) in {"部分完成", "未完成", "证据不足"}
            ]
            groups.append(
                {
                    "user_name": user_name,
                    "department": department,
                    "summary": {
                        "total_plans": len(items),
                        "status_counts": status_counts,
                        "unfinished_or_attention_count": len(unfinished),
                    },
                    "items": [
                        {
                            "plan_date": item.get("plan_date"),
                            "completion_date": item.get("completion_date"),
                            "status": item.get("status"),
                            "plan_text": self._clip_text(item.get("plan_text"), 160),
                            "reason": self._clip_text(item.get("reason"), 140),
                            "evidence": self._clip_text(item.get("evidence"), 100),
                        }
                        for item in items[:40]
                    ],
                    "truncated_item_count": max(0, len(items) - 40),
                }
            )
        return groups

    def _compact_weekly_plan_followup_for_llm(self, item: Mapping[str, Any]) -> dict[str, Any]:
        done_items = item.get("done_items", [])
        candidates = item.get("candidate_done_items", [])
        return {
            "plan_id": self._plan_followup_id(item),
            "user_name": str(item.get("user_name") or "未知人员"),
            "department": str(item.get("department") or "未填写部门"),
            "plan_date": str(item.get("plan_date") or ""),
            "completion_date": str(item.get("completion_date") or ""),
            "plan_text": self._clip_text(item.get("plan_text"), 300),
            "done_items": [
                {
                    "done_text": self._clip_text(done.get("done_text"), 220),
                    "report_date": done.get("report_date"),
                }
                for done in done_items[:3]
                if isinstance(done, Mapping)
            ] if isinstance(done_items, list) else [],
            "candidate_done_items": [
                {
                    "done_text": self._clip_text(candidate.get("done_text"), 220),
                    "report_date": candidate.get("report_date"),
                    "overlap_keywords": candidate.get("overlap_keywords", [])[:8]
                    if isinstance(candidate.get("overlap_keywords"), list)
                    else [],
                }
                for candidate in candidates[:3]
                if isinstance(candidate, Mapping)
            ] if isinstance(candidates, list) else [],
        }

    def _compact_dept_plan_followup_for_llm(self, item: Mapping[str, Any], *, mode: str = "full") -> dict[str, Any]:
        weekly_candidates = item.get("weekly_done_candidates", [])
        possible_weekly_evidence = item.get("possible_weekly_evidence", [])
        weekly_match_audit = item.get("weekly_match_audit", {})
        self_eval_candidates = item.get("self_eval_candidates", [])
        owner_self_eval = self._dept_plan_owner_self_eval_for_llm(item)
        opl_evidence = self._dept_plan_opl_issues_for_llm(item)
        weekly_candidate_limit = self._env_int("MYSQL_DEPT_PLAN_LLM_WEEKLY_CANDIDATE_MAX_ITEMS", 128)
        review_evidence_limit = self._env_int("MYSQL_DEPT_PLAN_LLM_REVIEW_EVIDENCE_MAX_ITEMS", 24)
        llm_weekly_candidates = self._dept_plan_weekly_candidates_for_llm(
            candidates=weekly_candidates,
            limit=weekly_candidate_limit,
        )
        if mode == "primary":
            return self._compact_dept_plan_followup_for_primary_llm(
                item=item,
                weekly_candidates=llm_weekly_candidates,
                possible_weekly_evidence=possible_weekly_evidence,
                weekly_match_audit=weekly_match_audit,
                owner_self_eval=owner_self_eval,
                self_eval_candidates=self_eval_candidates,
                opl_evidence=opl_evidence,
            )
        if mode == "recovery":
            return self._compact_dept_plan_followup_for_recovery_llm(
                item=item,
                weekly_candidates=llm_weekly_candidates,
                possible_weekly_evidence=possible_weekly_evidence,
                weekly_match_audit=weekly_match_audit,
                owner_self_eval=owner_self_eval,
                self_eval_candidates=self_eval_candidates,
                opl_evidence=opl_evidence,
            )
        return {
            "plan_id": self._dept_plan_followup_id(item),
            "部门": str(item.get("department") or "未填写部门"),
            "负责人": str(item.get("owner_user") or ""),
            "计划月份": str(item.get("plan_month") or ""),
            "截止日期": str(item.get("due_date") or ""),
            "周次目标": str(item.get("target") or ""),
            "计划内容": self._clip_text(item.get("plan_text"), 320),
            "完成证据标准": self._dept_plan_completion_evidence_standards(item.get("plan_text")),
            "周报候选选择规则": "负责人本人周报全部保留给模型逐条判断；非负责人周报按相关性和数量上限补充。",
            "周报候选": [
                {
                    "完成内容": self._clip_text(candidate.get("done_text"), 240),
                    "周报日期": candidate.get("report_date"),
                    "提交人": candidate.get("user_name"),
                    "是否负责人本人提交": self._dept_plan_yes_no(candidate.get("owner_match")),
                    "是否与计划部门兼容": self._dept_plan_yes_no(candidate.get("department_match")),
                    "是否负责人跨部门周报": self._dept_plan_yes_no(candidate.get("owner_cross_department")),
                    "是否有明确业务命中": self._dept_plan_yes_no(
                        candidate.get("business_keyword_support") or candidate.get("strong_keyword_support")
                    ),
                    "候选可信度": self._dept_plan_confidence_label(candidate.get("candidate_confidence")),
                    "候选来源": self._dept_plan_candidate_source_label(candidate.get("candidate_source")),
                    "重合关键词": candidate.get("overlap_keywords", [])[:8]
                    if isinstance(candidate.get("overlap_keywords"), list)
                    else [],
                    "具体命中词": candidate.get("specific_overlap_keywords", [])[:8]
                    if isinstance(candidate.get("specific_overlap_keywords"), list)
                    else [],
                }
                for candidate in llm_weekly_candidates
                if isinstance(candidate, Mapping)
            ] if isinstance(weekly_candidates, list) else [],
            "备选复核材料": [
                {
                    "完成内容": self._clip_text(candidate.get("done_text"), 180),
                    "周报日期": candidate.get("report_date"),
                    "提交人": candidate.get("user_name"),
                    "复核原因": self._dept_plan_filter_reason_label(candidate.get("filter_reason")),
                    "是否负责人本人提交": self._dept_plan_yes_no(candidate.get("owner_match")),
                    "是否与计划部门兼容": self._dept_plan_yes_no(candidate.get("department_match")),
                    "是否负责人跨部门周报": self._dept_plan_yes_no(candidate.get("owner_cross_department")),
                    "是否有明确业务命中": self._dept_plan_yes_no(
                        candidate.get("business_keyword_support") or candidate.get("strong_keyword_support")
                    ),
                    "可否作为完成判断证据": self._dept_plan_review_evidence_usage_label(candidate),
                    "重合关键词": candidate.get("overlap_keywords", [])[:6]
                    if isinstance(candidate.get("overlap_keywords"), list)
                    else [],
                    "具体命中词": candidate.get("specific_overlap_keywords", [])[:6]
                    if isinstance(candidate.get("specific_overlap_keywords"), list)
                    else [],
                }
                for candidate in possible_weekly_evidence[:review_evidence_limit]
                if isinstance(candidate, Mapping)
            ] if isinstance(possible_weekly_evidence, list) else [],
            "周报匹配审计": self._dept_plan_weekly_match_audit_for_llm(weekly_match_audit),
            "负责人月度考核记录": owner_self_eval,
            "OPL问题闭环证据": opl_evidence,
            "旧版自评候选": [
                {
                    "类型": self._dept_plan_self_eval_type_label(candidate.get("item_type")),
                    "内容": self._clip_text(candidate.get("item_text"), 240),
                    "重合关键词": candidate.get("overlap_keywords", [])[:8]
                    if isinstance(candidate.get("overlap_keywords"), list)
                    else [],
                }
                for candidate in self_eval_candidates[:4]
                if isinstance(candidate, Mapping)
            ] if isinstance(self_eval_candidates, list) else [],
        }

    def _compact_dept_plan_followup_for_primary_llm(
        self,
        *,
        item: Mapping[str, Any],
        weekly_candidates: list[Mapping[str, Any]],
        possible_weekly_evidence: Any,
        weekly_match_audit: Any,
        owner_self_eval: dict[str, Any],
        self_eval_candidates: Any,
        opl_evidence: dict[str, Any],
    ) -> dict[str, Any]:
        review_evidence_limit = self._env_int("MYSQL_DEPT_PLAN_LLM_REVIEW_EVIDENCE_MAX_ITEMS", 12)
        non_owner_candidates = self._dept_plan_non_owner_weekly_candidates_for_llm(
            candidates=weekly_candidates,
            limit=self._env_int("MYSQL_DEPT_PLAN_LLM_NON_OWNER_CANDIDATE_MAX_ITEMS", 12),
        )
        owner_evidence = self._dept_plan_owner_evidence_for_llm(item)
        return {
            "plan_id": self._dept_plan_followup_id(item),
            "部门": str(item.get("department") or "未填写部门"),
            "负责人": str(item.get("owner_user") or ""),
            "计划月份": str(item.get("plan_month") or ""),
            "截止日期": str(item.get("due_date") or ""),
            "周次目标": str(item.get("target") or ""),
            "计划内容": self._clip_text(item.get("plan_text"), 280),
            "完成证据标准": self._dept_plan_completion_evidence_standards(item.get("plan_text")),
            "负责人周报证据规则": "负责人周报已由第一层LLM按负责人和日期范围逐条读取完整原文并抽取原文证据；最终判断必须以这些原文证据为主。",
            "负责人周报证据抽取": owner_evidence,
            "非负责人协作周报规则": "非负责人周报只作为跨部门协作辅助证据，按相关性排序补充，不能单独证明负责人计划已完成。",
            "非负责人协作周报": [
                {
                    "report_id": candidate.get("report_id"),
                    "日期": candidate.get("report_date"),
                    "提交人": candidate.get("user_name"),
                    "主表周报摘要": self._clip_text(candidate.get("done_text"), 150),
                    "命中点": candidate.get("specific_overlap_keywords", [])[:6]
                    if isinstance(candidate.get("specific_overlap_keywords"), list)
                    else [],
                    "是否有明确业务命中": self._dept_plan_yes_no(
                        candidate.get("business_keyword_support") or candidate.get("strong_keyword_support")
                    ),
                    "完成信号": candidate.get("completion_signal") or self._dept_plan_completion_signal_label(candidate.get("done_text")),
                    "卡点或未完成信号": candidate.get("blockage_signal") or "",
                }
                for candidate in non_owner_candidates
                if isinstance(candidate, Mapping)
            ],
            "备选复核材料": [
                {
                    "日期": candidate.get("report_date"),
                    "提交人": candidate.get("user_name"),
                    "完成内容摘要": self._clip_text(candidate.get("done_text"), 120),
                    "复核原因": self._dept_plan_filter_reason_label(candidate.get("filter_reason")),
                    "可否作为完成判断证据": self._dept_plan_review_evidence_usage_label(candidate),
                }
                for candidate in possible_weekly_evidence[:review_evidence_limit]
                if isinstance(candidate, Mapping)
            ] if isinstance(possible_weekly_evidence, list) else [],
            "周报匹配审计": self._dept_plan_weekly_match_audit_for_recovery_llm(weekly_match_audit),
            "负责人月度考核记录": owner_self_eval,
            "OPL问题闭环证据": opl_evidence,
            "旧版自评候选": [
                {
                    "类型": self._dept_plan_self_eval_type_label(candidate.get("item_type")),
                    "内容": self._clip_text(candidate.get("item_text"), 180),
                }
                for candidate in self_eval_candidates[:4]
                if isinstance(candidate, Mapping)
            ] if isinstance(self_eval_candidates, list) else [],
        }

    def _compact_dept_plan_followup_for_recovery_llm(
        self,
        *,
        item: Mapping[str, Any],
        weekly_candidates: list[Mapping[str, Any]],
        possible_weekly_evidence: Any,
        weekly_match_audit: Any,
        owner_self_eval: dict[str, Any],
        self_eval_candidates: Any,
        opl_evidence: dict[str, Any],
    ) -> dict[str, Any]:
        review_evidence_limit = self._env_int("MYSQL_DEPT_PLAN_LLM_RECOVERY_REVIEW_EVIDENCE_MAX_ITEMS", 8)
        non_owner_candidates = self._dept_plan_non_owner_weekly_candidates_for_llm(
            candidates=weekly_candidates,
            limit=self._env_int("MYSQL_DEPT_PLAN_LLM_RECOVERY_NON_OWNER_CANDIDATE_MAX_ITEMS", 6),
        )
        owner_evidence = self._dept_plan_owner_evidence_for_llm(item)
        return {
            "plan_id": self._dept_plan_followup_id(item),
            "部门": str(item.get("department") or "未填写部门"),
            "负责人": str(item.get("owner_user") or ""),
            "计划月份": str(item.get("plan_month") or ""),
            "截止日期": str(item.get("due_date") or ""),
            "周次目标": str(item.get("target") or ""),
            "计划内容": self._clip_text(item.get("plan_text"), 260),
            "完成证据标准": self._dept_plan_completion_evidence_standards(item.get("plan_text")),
            "证据压缩说明": (
                "这是结构化输出失败后的压缩救援输入；负责人周报仍以第一层LLM抽取的原文证据为主，"
                "非负责人协作材料只作辅助。请必须返回合法结构化结果。"
            ),
            "负责人周报证据抽取": owner_evidence,
            "非负责人协作周报": [
                {
                    "report_id": candidate.get("report_id"),
                    "主表周报摘要": self._clip_text(candidate.get("done_text"), 120),
                    "周报日期": candidate.get("report_date"),
                    "提交人": candidate.get("user_name"),
                    "明确业务命中": self._dept_plan_yes_no(
                        candidate.get("business_keyword_support") or candidate.get("strong_keyword_support")
                    ),
                    "命中词": candidate.get("specific_overlap_keywords", [])[:5]
                    if isinstance(candidate.get("specific_overlap_keywords"), list)
                    else [],
                    "完成信号": candidate.get("completion_signal") or self._dept_plan_completion_signal_label(candidate.get("done_text")),
                    "卡点或未完成信号": candidate.get("blockage_signal") or "",
                }
                for candidate in non_owner_candidates
                if isinstance(candidate, Mapping)
            ],
            "备选复核材料": [
                {
                    "完成内容": self._clip_text(candidate.get("done_text"), 100),
                    "周报日期": candidate.get("report_date"),
                    "提交人": candidate.get("user_name"),
                    "复核原因": self._dept_plan_filter_reason_label(candidate.get("filter_reason")),
                    "可否作为完成判断证据": self._dept_plan_review_evidence_usage_label(candidate),
                }
                for candidate in possible_weekly_evidence[:review_evidence_limit]
                if isinstance(candidate, Mapping)
            ] if isinstance(possible_weekly_evidence, list) else [],
            "周报匹配审计": self._dept_plan_weekly_match_audit_for_recovery_llm(weekly_match_audit),
            "负责人月度考核记录": owner_self_eval,
            "OPL问题闭环证据": opl_evidence,
            "旧版自评候选": [
                {
                    "类型": self._dept_plan_self_eval_type_label(candidate.get("item_type")),
                    "内容": self._clip_text(candidate.get("item_text"), 180),
                }
                for candidate in self_eval_candidates[:4]
                if isinstance(candidate, Mapping)
            ] if isinstance(self_eval_candidates, list) else [],
        }

    def _dept_plan_opl_issues_for_llm(self, item: Mapping[str, Any]) -> dict[str, Any]:
        candidates_raw = item.get("opl_issue_candidates", [])
        possible_raw = item.get("possible_opl_evidence", [])
        audit_raw = item.get("opl_match_audit", {})
        candidates = candidates_raw if isinstance(candidates_raw, list) else []
        possible = possible_raw if isinstance(possible_raw, list) else []
        candidate_limit = self._env_int("MYSQL_DEPT_PLAN_LLM_OPL_CANDIDATE_MAX_ITEMS", 8)
        review_limit = self._env_int("MYSQL_DEPT_PLAN_LLM_OPL_REVIEW_MAX_ITEMS", 5)
        return {
            "使用规则": (
                "未闭环 OPL 只能作为风险/卡点证据，不能单独等同计划未完成；"
                "已闭环 OPL 只有解决措施或最新进展明确对应计划目标时，才可作为完成判断证据。"
            ),
            "相关OPL问题": [
                self._dept_plan_compact_opl_issue_for_llm(issue, text_limit=180)
                for issue in candidates[:candidate_limit]
                if isinstance(issue, Mapping)
            ],
            "备选OPL复核": [
                self._dept_plan_compact_opl_issue_for_llm(issue, text_limit=120, include_filter_reason=True)
                for issue in possible[:review_limit]
                if isinstance(issue, Mapping)
            ],
            "OPL匹配审计": self._dept_plan_opl_match_audit_for_llm(audit_raw),
        }

    def _dept_plan_compact_opl_issue_for_llm(
        self,
        issue: Mapping[str, Any],
        *,
        text_limit: int,
        include_filter_reason: bool = False,
    ) -> dict[str, Any]:
        overlap = issue.get("specific_overlap_keywords") or issue.get("overlap_keywords") or []
        result = {
            "问题编号": str(issue.get("issue_ref") or issue.get("source_no") or issue.get("source_issue_id") or ""),
            "问题日期": issue.get("issue_date"),
            "跟踪日期": issue.get("follow_date"),
            "部门": issue.get("department"),
            "机构/总成": issue.get("assembly"),
            "状态": issue.get("status") or "未填写",
            "闭环状态": self._dept_plan_opl_status_label(issue),
            "优先级": issue.get("priority") or "未填写",
            "是否高优先级": self._dept_plan_yes_no(issue.get("high_priority")),
            "负责人": issue.get("owner_user") or "未填写",
            "跟踪人": issue.get("tracker_user") or "未填写",
            "问题描述": self._clip_text(issue.get("issue_description"), text_limit),
            "解决措施/最新进展": self._clip_text(issue.get("solution_progress"), text_limit),
            "备注": self._clip_text(issue.get("remark"), max(80, text_limit // 2)),
            "匹配关系": [
                label
                for enabled, label in (
                    (issue.get("owner_match"), "负责人匹配"),
                    (issue.get("tracker_match"), "跟踪人匹配"),
                    (issue.get("department_match"), "部门匹配"),
                    (
                        issue.get("business_keyword_support") or issue.get("strong_keyword_support"),
                        "计划关键词匹配",
                    ),
                )
                if enabled
            ],
            "候选可信度": self._dept_plan_confidence_label(issue.get("candidate_confidence")),
            "候选来源": self._dept_plan_opl_candidate_source_label(issue.get("candidate_source")),
            "命中词": overlap[:6] if isinstance(overlap, list) else [],
            "可用于判断": self._dept_plan_opl_evidence_usage_label(issue),
        }
        if include_filter_reason:
            result["复核原因"] = self._dept_plan_opl_filter_reason_label(issue.get("filter_reason"))
        return {key: value for key, value in result.items() if value not in (None, "", [])}

    def _dept_plan_opl_status_label(self, issue: Mapping[str, Any]) -> str:
        if issue.get("open_issue"):
            return "未闭环"
        if issue.get("closed_issue"):
            return "已闭环"
        raw = str(issue.get("status_group") or "unknown").strip().lower()
        return DEPT_PLAN_OPL_STATUS_GROUP_LABELS.get(raw, "状态不明确")

    def _dept_plan_opl_evidence_usage_label(self, issue: Mapping[str, Any]) -> str:
        if issue.get("open_issue") or self._dept_plan_opl_status_label(issue) == "未闭环":
            priority_note = "高优先级" if issue.get("high_priority") else ""
            prefix = f"{priority_note}未闭环问题" if priority_note else "未闭环问题"
            return f"{prefix}，只可作为风险/卡点证据，不能单独等同计划未完成"
        if issue.get("closed_issue") or self._dept_plan_opl_status_label(issue) == "已闭环":
            return "已闭环问题；解决进展明确对应计划目标时，可作为完成判断证据"
        return "状态不明确，只能作为人工复核线索"

    def _dept_plan_opl_match_audit_for_llm(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            return {}
        raw_to_label = {
            "raw_opl_issue_count": "OPL 问题池数量",
            "department_compatible_count": "部门兼容数量",
            "owner_match_count": "负责人匹配数量",
            "tracker_match_count": "跟踪人匹配数量",
            "keyword_match_count": "关键词匹配数量",
            "strong_keyword_match_count": "明确关键词匹配数量",
            "business_keyword_match_count": "具体业务关键词匹配数量",
            "open_issue_count": "未闭环 OPL 数量",
            "high_priority_open_issue_count": "高优先级未闭环 OPL 数量",
            "candidate_pool_count": "候选池数量",
            "selected_before_cap_count": "截断前入选数量",
            "selected_count": "入选 OPL 候选数量",
            "selected_candidate_limit": "OPL 候选展示上限",
            "capped_candidate_count": "因数量上限未展示的 OPL 数量",
            "possible_evidence_count": "备选 OPL 复核数量",
            "possible_evidence_before_cap_count": "截断前备选 OPL 复核数量",
        }
        audit = {
            label: value.get(key)
            for key, label in raw_to_label.items()
            if value.get(key) not in (None, "", {})
        }
        filtered_counts = value.get("filtered_counts")
        if isinstance(filtered_counts, Mapping):
            audit["过滤原因统计"] = {
                self._dept_plan_opl_filter_reason_label(reason): count
                for reason, count in filtered_counts.items()
            }
        owner_names = value.get("owner_names")
        if isinstance(owner_names, list) and owner_names:
            audit["计划负责人匹配名"] = [str(item) for item in owner_names[:8] if str(item or "").strip()]
        if value.get("rule"):
            audit["匹配规则"] = value.get("rule")
        return audit

    def _dept_plan_owner_self_eval_for_llm(self, item: Mapping[str, Any]) -> dict[str, Any]:
        reports_raw = item.get("owner_self_eval_reports", [])
        items_raw = item.get("owner_self_eval_items", [])
        missing_raw = item.get("missing_owner_self_eval_users", [])
        reports = reports_raw if isinstance(reports_raw, list) else []
        items = items_raw if isinstance(items_raw, list) else []
        missing_users = [
            str(user).strip()
            for user in missing_raw
            if str(user or "").strip()
        ] if isinstance(missing_raw, list) else []
        detail_limit = self._env_int("MYSQL_DEPT_PLAN_LLM_OWNER_SELF_EVAL_MAX_ITEMS", 16)
        return {
            "来源规则": "按计划负责人姓名直接查询该员工当月 employee_self_eval_reports/items；不是部门候选搜索。",
            "考核表": [
                {
                    "员工": report.get("user_name"),
                    "部门": report.get("department"),
                    "岗位": report.get("position"),
                    "月份": report.get("month"),
                    "工作任务平均完成率": report.get("work_avg_completion_rate"),
                    "管理项平均得分": report.get("management_avg_score"),
                    "上级评分": report.get("leader_rating_score"),
                    "行政评分": report.get("admin_rating_score"),
                    "sheet": report.get("sheet_name"),
                }
                for report in reports[:6]
                if isinstance(report, Mapping)
            ],
            "明细": [
                {
                    "员工": detail.get("user_name"),
                    "区块": detail.get("section"),
                    "类型": self._dept_plan_self_eval_type_label(detail.get("item_type")),
                    "主事项": self._clip_text(detail.get("item_text"), 160),
                    "计划/目标": self._clip_text(detail.get("plan_text"), 180),
                    "完成结果": self._clip_text(detail.get("result_text"), 220),
                    "完成时间": detail.get("completion_time"),
                    "完成率": detail.get("completion_rate"),
                    "未完成内容": self._clip_text(detail.get("unfinished_text"), 160),
                    "未解决问题": self._clip_text(detail.get("unresolved_text"), 160),
                    "原因说明": self._clip_text(detail.get("reason_text"), 160),
                    "效果/影响": self._clip_text(detail.get("effect_text"), 120),
                    "来源行": detail.get("source_row"),
                }
                for detail in items[:detail_limit]
                if isinstance(detail, Mapping)
            ],
            "缺失负责人": missing_users,
        }

    def _dept_plan_owner_report_pool_for_evidence(self, source_output: Mapping[str, Any] | None) -> list[dict[str, Any]]:
        if not isinstance(source_output, Mapping):
            return []
        raw_pool = source_output.get("owner_weekly_reports_pool", [])
        if not isinstance(raw_pool, list):
            return []
        reports: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw_report in raw_pool:
            if not isinstance(raw_report, Mapping):
                continue
            report_id = str(raw_report.get("report_id") or "").strip()
            raw_text = str(
                raw_report.get("周报原文")
                or raw_report.get("raw_text")
                or raw_report.get("evidence_text")
                or ""
            ).strip()
            if not report_id or not raw_text or report_id in seen:
                continue
            seen.add(report_id)
            reports.append(
                {
                    "report_id": report_id,
                    "日期": raw_report.get("日期") or raw_report.get("report_date") or "",
                    "提交人": raw_report.get("提交人") or raw_report.get("user_name") or "",
                    "部门": raw_report.get("部门") or raw_report.get("department") or "",
                    "事项类型": raw_report.get("事项类型") or raw_report.get("item_type") or "weekly_report",
                    "周报事项": raw_report.get("周报事项") or "",
                    "周报原文": raw_text,
                    "source_doc_id": raw_report.get("source_doc_id"),
                    "source_chunk_id": raw_report.get("source_chunk_id"),
                }
            )
        reports.sort(key=lambda row: (str(row.get("日期") or ""), str(row.get("提交人") or ""), str(row.get("report_id") or "")))
        return reports

    def _dept_plan_owner_groups_for_evidence(
        self,
        *,
        followups: list[dict[str, Any]],
        report_by_id: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for item in followups:
            reports = self._dept_plan_owner_weekly_reports_for_evidence(item, report_by_id=report_by_id)
            report_submitters = {
                str(report.get("提交人") or "").strip()
                for report in reports
                if str(report.get("提交人") or "").strip()
            }
            owner_names = self._dept_plan_owner_names_for_item(item)
            group_name = "、".join(sorted(set(owner_names) or report_submitters)) or "未填写负责人"
            groups.setdefault(group_name, []).append(item)
        return groups

    def _dept_plan_owner_names_for_item(self, item: Mapping[str, Any]) -> list[str]:
        raw_names = item.get("owner_names")
        if isinstance(raw_names, list):
            names = [str(name).strip() for name in raw_names if str(name or "").strip()]
            if names:
                return names
        owner_text = str(item.get("owner_user") or "")
        parts = [part.strip() for part in re.split(r"[、,，;/；|\s]+", owner_text) if part.strip()]
        return list(dict.fromkeys(parts))

    def _dept_plan_owner_reports_for_group(
        self,
        *,
        items: list[Mapping[str, Any]],
        report_by_id: Mapping[str, Mapping[str, Any]],
        owner_name: str,
    ) -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in items:
            for report in self._dept_plan_owner_weekly_reports_for_evidence(item, report_by_id=report_by_id):
                report_id = str(report.get("report_id") or "")
                if not report_id or report_id in seen:
                    continue
                submitter = str(report.get("提交人") or "").strip()
                if submitter and not self._dept_plan_owner_group_contains_submitter(owner_name, submitter):
                    continue
                seen.add(report_id)
                reports.append(report)
        reports.sort(key=lambda row: (str(row.get("日期") or ""), str(row.get("report_id") or "")))
        return reports

    def _dept_plan_owner_group_contains_submitter(self, owner_name: str, submitter: str) -> bool:
        group_names = {
            name.strip()
            for name in re.split(r"[、,，;/；|\s]+", owner_name)
            if name.strip()
        }
        return not group_names or submitter in group_names

    def _dept_plan_owner_weekly_reports_for_evidence(
        self,
        item: Mapping[str, Any],
        *,
        report_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        if report_by_id:
            refs = item.get("owner_weekly_report_refs") or item.get("owner_weekly_reports") or []
            reports: list[dict[str, Any]] = []
            seen: set[str] = set()
            if isinstance(refs, list):
                for ref in refs:
                    if not isinstance(ref, Mapping):
                        continue
                    report_id = str(ref.get("report_id") or "").strip()
                    report = report_by_id.get(report_id)
                    if not report_id or report_id in seen or not isinstance(report, Mapping):
                        continue
                    seen.add(report_id)
                    reports.append(dict(report))
            reports.sort(key=lambda row: (str(row.get("日期") or ""), str(row.get("report_id") or "")))
            if reports:
                return reports

        owner_reports = item.get("owner_weekly_reports", [])
        if isinstance(owner_reports, list) and owner_reports:
            reports = self._normalize_dept_plan_owner_weekly_reports_for_evidence(owner_reports)
            if reports:
                return reports

        candidates = item.get("weekly_done_candidates", [])
        if not isinstance(candidates, list):
            return []
        reports: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        for candidate in candidates:
            if not isinstance(candidate, Mapping) or not candidate.get("owner_match"):
                continue
            done_text = str(candidate.get("done_text") or "").strip()
            raw_text = str(candidate.get("周报原文") or candidate.get("evidence_text") or done_text).strip()
            if not raw_text:
                continue
            identity = (
                candidate.get("report_date"),
                candidate.get("user_name"),
                candidate.get("item_type"),
                done_text,
                raw_text,
                candidate.get("source_doc_id"),
                candidate.get("source_chunk_id"),
            )
            if identity in seen:
                continue
            seen.add(identity)
            reports.append(
                {
                    "report_id": f"r{len(reports) + 1}",
                    "日期": candidate.get("report_date") or "",
                    "提交人": candidate.get("user_name") or "",
                    "部门": candidate.get("department") or "",
                    "事项类型": candidate.get("item_type") or "",
                    "周报事项": done_text,
                    "周报原文": raw_text,
                    "source_doc_id": candidate.get("source_doc_id"),
                    "source_chunk_id": candidate.get("source_chunk_id"),
                }
            )
        reports.sort(key=lambda row: (str(row.get("日期") or ""), str(row.get("report_id") or "")))
        for index, report in enumerate(reports, start=1):
            report["report_id"] = f"r{index}"
        return reports

    def _normalize_dept_plan_owner_group_evidence_extraction(
        self,
        *,
        items: list[Mapping[str, Any]],
        reports: list[Mapping[str, Any]],
        output: DeptPlanOwnerGroupEvidenceExtractionOutput,
    ) -> list[dict[str, Any]]:
        valid_plan_ids = {self._dept_plan_followup_id(item) for item in items}
        report_by_id = {str(report.get("report_id") or ""): report for report in reports}
        normalized: list[dict[str, Any]] = []
        for raw in output.items:
            plan_id = str(raw.plan_id)
            report_id = str(raw.report_id)
            if plan_id not in valid_plan_ids or report_id not in report_by_id:
                continue
            report = report_by_id[report_id]
            snippets = [
                self._clip_text(self._sanitize_dept_plan_user_text(snippet), 600)
                for snippet in raw.evidence_snippets
                if str(snippet or "").strip()
            ][:4]
            normalized.append(
                {
                    "plan_id": plan_id,
                    "report_id": report_id,
                    "report_date": raw.report_date or str(report.get("日期") or ""),
                    "submitter": raw.submitter or str(report.get("提交人") or ""),
                    "is_related": bool(raw.is_related),
                    "evidence_snippets": snippets,
                    "completion_signal": self._clip_text(
                        self._sanitize_dept_plan_user_text(raw.completion_signal),
                        120,
                    ),
                    "blockage_signal": self._clip_text(
                        self._sanitize_dept_plan_user_text(raw.blockage_signal),
                        120,
                    ),
                    "relation_reason": self._clip_text(
                        self._sanitize_dept_plan_user_text(raw.relation_reason),
                        160,
                    ),
                }
            )
        return normalized

    def _normalize_dept_plan_owner_weekly_reports_for_evidence(self, owner_reports: list[Any]) -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        for raw_report in owner_reports:
            if not isinstance(raw_report, Mapping):
                continue
            item_text = str(
                raw_report.get("周报事项")
                or raw_report.get("item_text")
                or raw_report.get("done_text")
                or ""
            ).strip()
            raw_text = str(
                raw_report.get("周报原文")
                or raw_report.get("raw_text")
                or raw_report.get("evidence_text")
                or item_text
            ).strip()
            if not raw_text:
                continue
            identity = (
                raw_report.get("日期") or raw_report.get("report_date"),
                raw_report.get("提交人") or raw_report.get("user_name"),
                raw_report.get("部门") or raw_report.get("department"),
                raw_report.get("事项类型") or raw_report.get("item_type"),
                item_text,
                raw_text,
                raw_report.get("source_doc_id"),
                raw_report.get("source_chunk_id"),
            )
            if identity in seen:
                continue
            seen.add(identity)
            reports.append(
                {
                    "report_id": f"r{len(reports) + 1}",
                    "日期": raw_report.get("日期") or raw_report.get("report_date") or "",
                    "提交人": raw_report.get("提交人") or raw_report.get("user_name") or "",
                    "部门": raw_report.get("部门") or raw_report.get("department") or "",
                    "事项类型": raw_report.get("事项类型") or raw_report.get("item_type") or "",
                    "周报事项": item_text,
                    "周报原文": raw_text,
                    "source_doc_id": raw_report.get("source_doc_id"),
                    "source_chunk_id": raw_report.get("source_chunk_id"),
                }
            )
        reports.sort(key=lambda row: (str(row.get("日期") or ""), str(row.get("report_id") or "")))
        for index, report in enumerate(reports, start=1):
            report["report_id"] = f"r{index}"
        return reports

    def _chunk_dept_plan_owner_reports(
        self,
        reports: list[Mapping[str, Any]],
        *,
        size: int,
    ) -> list[list[Mapping[str, Any]]]:
        chunk_size = max(1, size)
        return [reports[index : index + chunk_size] for index in range(0, len(reports), chunk_size)]

    def _chunk_dept_plan_owner_items(
        self,
        items: list[Mapping[str, Any]],
        *,
        size: int,
    ) -> list[list[Mapping[str, Any]]]:
        chunk_size = max(1, size)
        return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]

    def _normalize_dept_plan_owner_evidence_extraction(
        self,
        *,
        item: Mapping[str, Any],
        reports: list[Mapping[str, Any]],
        output: DeptPlanOwnerEvidenceExtractionOutput,
    ) -> list[dict[str, Any]]:
        expected_plan_id = self._dept_plan_followup_id(item)
        if str(output.plan_id) != expected_plan_id:
            raise ValueError(f"负责人周报证据抽取返回了错误计划编号：{output.plan_id}")
        raw_by_id = {str(report.report_id): report for report in output.reports}
        normalized: list[dict[str, Any]] = []
        missing_report_ids: list[str] = []
        for report in reports:
            report_id = str(report.get("report_id") or "")
            raw = raw_by_id.get(report_id)
            if raw is None:
                missing_report_ids.append(report_id)
                continue
            snippets = [
                self._clip_text(self._sanitize_dept_plan_user_text(snippet), 600)
                for snippet in raw.evidence_snippets
                if str(snippet or "").strip()
            ][:4]
            normalized.append(
                {
                    "report_id": report_id,
                    "report_date": raw.report_date or str(report.get("日期") or ""),
                    "submitter": raw.submitter or str(report.get("提交人") or ""),
                    "is_related": bool(raw.is_related),
                    "evidence_snippets": snippets,
                    "completion_signal": self._clip_text(
                        self._sanitize_dept_plan_user_text(raw.completion_signal),
                        120,
                    ),
                    "blockage_signal": self._clip_text(
                        self._sanitize_dept_plan_user_text(raw.blockage_signal),
                        120,
                    ),
                    "relation_reason": self._clip_text(
                        self._sanitize_dept_plan_user_text(raw.relation_reason),
                        160,
                    ),
                }
            )
        if missing_report_ids:
            raise ValueError(f"负责人周报证据抽取未覆盖这些周报：{', '.join(missing_report_ids)}")
        return normalized

    def _build_dept_plan_owner_evidence_summary(
        self,
        *,
        item: Mapping[str, Any],
        owner_reports: list[Mapping[str, Any]],
        extracted_reports: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        related_items = [
            dict(report)
            for report in extracted_reports
            if report.get("is_related")
            or report.get("evidence_snippets")
            or str(report.get("completion_signal") or "").strip()
            or str(report.get("blockage_signal") or "").strip()
        ]
        return {
            "plan_id": self._dept_plan_followup_id(item),
            "owner_report_count": len(owner_reports),
            "reviewed_report_count": len(extracted_reports),
            "related_report_count": len(related_items),
            "unrelated_report_count": max(0, len(owner_reports) - len(related_items)),
            "all_owner_reports_reviewed": len(owner_reports) == len(extracted_reports),
            "evidence_items": related_items,
        }

    def _dept_plan_empty_owner_evidence(self, *, item: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "plan_id": self._dept_plan_followup_id(item),
            "owner_report_count": 0,
            "reviewed_report_count": 0,
            "related_report_count": 0,
            "unrelated_report_count": 0,
            "all_owner_reports_reviewed": True,
            "evidence_items": [],
        }

    def _dept_plan_owner_evidence_for_llm(self, item: Mapping[str, Any]) -> dict[str, Any]:
        evidence = item.get("owner_weekly_llm_evidence")
        if not isinstance(evidence, Mapping):
            return self._dept_plan_empty_owner_evidence(item=item)
        evidence_items = evidence.get("evidence_items", [])
        return {
            "负责人周报总数": evidence.get("owner_report_count", 0),
            "已由LLM逐条读取数量": evidence.get("reviewed_report_count", 0),
            "相关证据周报数量": evidence.get("related_report_count", 0),
            "不相关周报数量": evidence.get("unrelated_report_count", 0),
            "是否已覆盖全部负责人周报": self._dept_plan_yes_no(evidence.get("all_owner_reports_reviewed")),
            "相关原文证据": [
                {
                    "report_id": report.get("report_id"),
                    "日期": report.get("report_date"),
                    "提交人": report.get("submitter"),
                    "原文片段": [
                        self._clip_text(snippet, 500)
                        for snippet in report.get("evidence_snippets", [])
                        if str(snippet or "").strip()
                    ][:3],
                    "完成信号": report.get("completion_signal"),
                    "卡点或未完成信号": report.get("blockage_signal"),
                    "相关原因": report.get("relation_reason"),
                }
                for report in evidence_items
                if isinstance(report, Mapping)
            ],
        }

    def _dept_plan_non_owner_weekly_candidates_for_llm(
        self,
        *,
        candidates: Any,
        limit: int,
    ) -> list[Mapping[str, Any]]:
        if not isinstance(candidates, list):
            return []
        valid_candidates = [
            candidate
            for candidate in candidates
            if isinstance(candidate, Mapping) and not candidate.get("owner_match")
        ]
        return valid_candidates[: max(0, limit)]

    def _dept_plan_weekly_candidates_for_llm(self, *, candidates: Any, limit: int) -> list[Mapping[str, Any]]:
        if not isinstance(candidates, list):
            return []
        valid_candidates = [candidate for candidate in candidates if isinstance(candidate, Mapping)]
        owner_candidates = [candidate for candidate in valid_candidates if candidate.get("owner_match")]
        non_owner_candidates = [candidate for candidate in valid_candidates if not candidate.get("owner_match")]
        non_owner_limit = self._env_int(
            "MYSQL_DEPT_PLAN_LLM_NON_OWNER_CANDIDATE_MAX_ITEMS",
            max(1, min(12, int(limit))),
        )
        selected_ids = {id(candidate) for candidate in owner_candidates}
        return [
            *owner_candidates,
            *[
                candidate
                for candidate in non_owner_candidates[:non_owner_limit]
                if id(candidate) not in selected_ids
            ],
        ]

    def _build_dept_plan_judge_batches(
        self,
        *,
        followups: list[Mapping[str, Any]],
        max_items: int,
        max_chars: int,
    ) -> tuple[list[list[Mapping[str, Any]]], list[dict[str, Any]]]:
        item_limit = max(1, max_items)
        char_limit = max(1, max_chars)
        batches: list[list[Mapping[str, Any]]] = []
        plan: list[dict[str, Any]] = []
        current: list[Mapping[str, Any]] = []
        current_chars = 0
        current_department: str | None = None

        def flush_current(**extra: Any) -> None:
            nonlocal current, current_chars, current_department
            if not current:
                return
            batches.append(current)
            plan.append(
                {
                    "batch_index": len(batches),
                    "item_count": len(current),
                    "compact_chars": current_chars,
                    "department": current_department,
                    "plan_ids": [self._dept_plan_followup_id(row) for row in current],
                    **extra,
                }
            )
            current = []
            current_chars = 0
            current_department = None

        for item in followups:
            compact_chars = self._dept_plan_compact_chars(item)
            force_single = self._dept_plan_requires_single_judge(item=item, compact_chars=compact_chars)
            item_department = str(item.get("department") or "未填写部门")
            should_flush = bool(
                current
                and (
                    item_department != current_department
                    or
                    force_single
                    or len(current) >= item_limit
                    or current_chars + compact_chars > char_limit
                )
            )
            if should_flush:
                flush_current(department_boundary=item_department != current_department)
            current.append(item)
            current_chars += compact_chars
            current_department = item_department
            if force_single or compact_chars >= char_limit:
                flush_current(
                    oversized_single=compact_chars >= char_limit,
                    forced_single_large_evidence=force_single,
                )
        if current:
            flush_current()
        return batches, plan

    def _dept_plan_compact_chars(self, item: Mapping[str, Any]) -> int:
        compact = self._compact_dept_plan_followup_for_llm(item, mode="primary")
        return len(json.dumps(compact, ensure_ascii=False, separators=(",", ":")))

    def _load_dept_plan_judgement_cache(self, *, followups: list[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        metadata = {
            "enabled": self._dept_plan_judgement_cache_enabled(),
            "hit_count": 0,
            "miss_count": 0,
            "store_count": 0,
        }
        if not metadata["enabled"]:
            metadata["miss_count"] = len(followups)
            return [], metadata
        cached: list[dict[str, Any]] = []
        for item in followups:
            cache_file = self._dept_plan_judgement_cache_file(item)
            if not cache_file or not os.path.exists(cache_file):
                metadata["miss_count"] += 1
                continue
            try:
                with open(cache_file, encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, json.JSONDecodeError):
                metadata["miss_count"] += 1
                continue
            judgement = payload.get("judgement") if isinstance(payload, Mapping) else None
            if not isinstance(judgement, Mapping):
                metadata["miss_count"] += 1
                continue
            status = self._normalize_dept_plan_status(judgement.get("status"))
            if status == DEPT_PLAN_JUDGEMENT_FAILED_STATUS:
                metadata["miss_count"] += 1
                continue
            cached.append(
                {
                    "plan_id": self._dept_plan_followup_id(item),
                    "department": str(item.get("department") or "未填写部门"),
                    "owner_user": str(item.get("owner_user") or ""),
                    "plan_month": str(item.get("plan_month") or ""),
                    "due_date": str(item.get("due_date") or ""),
                    "target": str(item.get("target") or ""),
                    "plan_text": str(item.get("plan_text") or "").strip(),
                    "status": status,
                    "reason": self._clip_text(self._sanitize_dept_plan_user_text(judgement.get("reason")), 240)
                    or "使用已缓存的结构化判断。",
                    "evidence": self._clip_text(self._sanitize_dept_plan_user_text(judgement.get("evidence")), 220),
                }
            )
            metadata["hit_count"] += 1
        return cached, metadata

    def _store_dept_plan_judgement_cache_for_results(
        self,
        *,
        batch_items: list[Mapping[str, Any]],
        judgements: list[dict[str, Any]],
    ) -> int:
        if not self._dept_plan_judgement_cache_enabled():
            return 0
        stored_count = 0
        item_by_id = {self._dept_plan_followup_id(item): item for item in batch_items}
        for judgement in judgements:
            status = self._normalize_dept_plan_status(judgement.get("status"))
            if status == DEPT_PLAN_JUDGEMENT_FAILED_STATUS:
                continue
            item = item_by_id.get(str(judgement.get("plan_id") or ""))
            if item is None:
                continue
            cache_file = self._dept_plan_judgement_cache_file(item)
            if not cache_file:
                continue
            payload = {
                "schema_version": "dept_plan_judgement_cache_v3",
                "cache_key": self._dept_plan_judgement_cache_key(item),
                "plan_id": self._dept_plan_followup_id(item),
                "judgement": {
                    "status": status,
                    "reason": str(judgement.get("reason") or "").strip(),
                    "evidence": str(judgement.get("evidence") or "").strip(),
                },
            }
            try:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                temp_file = f"{cache_file}.tmp"
                with open(temp_file, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False, indent=2)
                os.replace(temp_file, cache_file)
                stored_count += 1
            except OSError:
                continue
        return stored_count

    def _dept_plan_judgement_cache_enabled(self) -> bool:
        raw = os.getenv("MYSQL_DEPT_PLAN_LLM_JUDGE_CACHE_ENABLED", "1").strip().lower()
        return raw not in {"0", "false", "no", "off"}

    def _dept_plan_judgement_cache_file(self, item: Mapping[str, Any]) -> str:
        return os.path.join(self._dept_plan_judgement_cache_dir(), f"{self._dept_plan_judgement_cache_key(item)}.json")

    def _dept_plan_judgement_cache_dir(self) -> str:
        return os.getenv(
            "MYSQL_DEPT_PLAN_LLM_JUDGE_CACHE_DIR",
            os.path.join(os.getcwd(), "outputs", "dept_plan_judgement_cache"),
        )

    def _dept_plan_judgement_cache_key(self, item: Mapping[str, Any]) -> str:
        payload = {
            "schema_version": "dept_plan_judgement_cache_v3",
            "plan_id": self._dept_plan_followup_id(item),
            "compact": self._compact_dept_plan_followup_for_llm(item, mode="primary"),
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _dept_plan_owner_evidence_extraction_enabled(self) -> bool:
        raw = os.getenv("MYSQL_DEPT_PLAN_LLM_EVIDENCE_EXTRACTION_ENABLED", "1").strip().lower()
        return raw not in {"0", "false", "no", "off"}

    def _dept_plan_owner_evidence_cache_enabled(self) -> bool:
        raw = os.getenv("MYSQL_DEPT_PLAN_LLM_EVIDENCE_CACHE_ENABLED", "1").strip().lower()
        return raw not in {"0", "false", "no", "off"}

    def _load_dept_plan_owner_evidence_cache(
        self,
        *,
        item: Mapping[str, Any],
        owner_reports: list[Mapping[str, Any]],
    ) -> dict[str, Any] | None:
        if not self._dept_plan_owner_evidence_cache_enabled():
            return None
        cache_file = self._dept_plan_owner_evidence_cache_file(item=item, owner_reports=owner_reports)
        if not cache_file or not os.path.exists(cache_file):
            return None
        try:
            with open(cache_file, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        evidence = payload.get("owner_evidence") if isinstance(payload, Mapping) else None
        if isinstance(evidence, Mapping):
            return dict(evidence)
        return None

    def _store_dept_plan_owner_evidence_cache(
        self,
        *,
        item: Mapping[str, Any],
        owner_reports: list[Mapping[str, Any]],
        owner_evidence: Mapping[str, Any],
    ) -> int:
        if not self._dept_plan_owner_evidence_cache_enabled():
            return 0
        cache_file = self._dept_plan_owner_evidence_cache_file(item=item, owner_reports=owner_reports)
        if not cache_file:
            return 0
        payload = {
            "schema_version": "dept_plan_owner_evidence_cache_v1",
            "cache_key": self._dept_plan_owner_evidence_cache_key(item=item, owner_reports=owner_reports),
            "plan_id": self._dept_plan_followup_id(item),
            "owner_evidence": dict(owner_evidence),
        }
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            temp_file = f"{cache_file}.tmp"
            with open(temp_file, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            os.replace(temp_file, cache_file)
            return 1
        except OSError:
            return 0

    def _dept_plan_owner_evidence_cache_file(
        self,
        *,
        item: Mapping[str, Any],
        owner_reports: list[Mapping[str, Any]],
    ) -> str:
        return os.path.join(
            self._dept_plan_owner_evidence_cache_dir(),
            f"{self._dept_plan_owner_evidence_cache_key(item=item, owner_reports=owner_reports)}.json",
        )

    def _dept_plan_owner_evidence_cache_dir(self) -> str:
        return os.getenv(
            "MYSQL_DEPT_PLAN_LLM_EVIDENCE_CACHE_DIR",
            os.path.join(os.getcwd(), "outputs", "dept_plan_owner_evidence_cache"),
        )

    def _dept_plan_owner_evidence_cache_key(
        self,
        *,
        item: Mapping[str, Any],
        owner_reports: list[Mapping[str, Any]],
    ) -> str:
        payload = {
            "schema_version": "dept_plan_owner_evidence_cache_v1",
            "plan_id": self._dept_plan_followup_id(item),
            "owner_user": str(item.get("owner_user") or ""),
            "plan_text": str(item.get("plan_text") or "").strip(),
            "plan_month": str(item.get("plan_month") or ""),
            "reports": [
                {
                    "report_id": report.get("report_id"),
                    "date": report.get("日期"),
                    "submitter": report.get("提交人"),
                    "text": report.get("周报原文"),
                    "source_doc_id": report.get("source_doc_id"),
                    "source_chunk_id": report.get("source_chunk_id"),
                }
                for report in owner_reports
            ],
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _dept_plan_requires_single_judge(self, *, item: Mapping[str, Any], compact_chars: int) -> bool:
        weekly_candidates = item.get("weekly_done_candidates")
        if not isinstance(weekly_candidates, list):
            return False
        owner_count = sum(1 for candidate in weekly_candidates if isinstance(candidate, Mapping) and candidate.get("owner_match"))
        total_threshold = self._env_int("MYSQL_DEPT_PLAN_LLM_SINGLE_CANDIDATE_THRESHOLD", 80)
        owner_threshold = self._env_int("MYSQL_DEPT_PLAN_LLM_SINGLE_OWNER_CANDIDATE_THRESHOLD", 60)
        char_threshold = self._env_int("MYSQL_DEPT_PLAN_LLM_SINGLE_COMPACT_CHARS_THRESHOLD", 18000)
        return bool(
            len(weekly_candidates) >= total_threshold
            or owner_count >= owner_threshold
            or compact_chars >= char_threshold
        )

    def _dept_plan_completion_evidence_standards(self, plan_text: Any) -> list[str]:
        text = str(plan_text or "").strip()
        compact = re.sub(r"\s+", "", text)
        standards = [
            "通用标准：完成证据必须明确覆盖计划目标本身；仅能证明准备、推进、讨论或相关背景的内容，不能单独判已完成。",
        ]
        if any(token in compact for token in ("采购", "购买", "购置", "新增购买", "下单", "到货", "验收", "入库")):
            standards.append(
                "采购类计划：需看到下单、采购完成、到货、验收、入库、付款、合同或供应商确认等采购闭环证据；环境部署、测试、培训、数据训练或使用痕迹，除非明确说明采购对象已到货或已验收，否则不能证明购买完成。"
            )
        if any(token in compact for token in ("测试", "验证", "冒烟", "回归", "转测", "问题单", "bug", "BUG", "Bug")):
            standards.append(
                "测试类计划：需看到实际执行测试/验证/回归、测试结果、测试报告、缺陷回归关闭或问题单处理闭环；仅准备测试、计划测试、待测试、提出问题但未验证闭环，不能判已完成。"
            )
        if any(token in compact for token in ("制度", "流程", "规范", "文档", "资料", "培训", "知识库")):
            standards.append(
                "制度/文档/培训类计划：需看到制度流程或资料已定稿、评审通过、发布归档、培训已执行或产生落地结果；仅讨论、梳理、草稿、拟定、待评审、待培训，不能判已完成。"
            )
        return standards

    def _dept_plan_weekly_match_audit_for_llm(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            return {}
        audit = {
            "部门范围内候选数量": value.get("department_filtered_count", 0),
            "负责人匹配数量": value.get("owner_match_count", 0),
            "负责人候选数量": value.get("owner_candidate_count", 0),
            "负责人跨部门周报数量": value.get("owner_cross_department_count", 0),
            "入选的负责人跨部门周报数量": value.get("owner_cross_department_selected_count", 0),
            "入选的负责人周报数量": value.get("owner_selected_count", 0),
            "入选的非负责人周报数量": value.get("non_owner_selected_count", 0),
            "关键词匹配数量": value.get("keyword_match_count", 0),
            "明确关键词匹配数量": value.get("strong_keyword_match_count", 0),
            "具体业务关键词匹配数量": value.get("business_keyword_match_count", 0),
            "候选池数量": value.get("candidate_pool_count", 0),
            "入选强候选数量": value.get("selected_count", 0),
            "强候选展示上限": value.get("selected_candidate_limit", 0),
            "非负责人候选展示上限": value.get("non_owner_candidate_limit", 0),
            "超出展示上限但仍保留的负责人周报数量": value.get("owner_candidate_over_limit_count", 0),
            "因数量上限未展示的候选数量": value.get("capped_candidate_count", 0),
            "备选复核数量": value.get("possible_evidence_count", 0),
            "截断前备选复核数量": value.get("possible_evidence_before_cap_count", 0),
        }
        filtered_counts = value.get("filtered_counts")
        if isinstance(filtered_counts, Mapping):
            audit["过滤原因统计"] = {
                self._dept_plan_filter_reason_label(reason): count
                for reason, count in filtered_counts.items()
            }
        return {
            key: count
            for key, count in audit.items()
            if count not in (None, "", {})
        }

    def _dept_plan_weekly_match_audit_for_recovery_llm(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            return {}
        keys = (
            "owner_match_count",
            "owner_candidate_count",
            "owner_cross_department_count",
            "owner_selected_count",
            "non_owner_selected_count",
            "selected_count",
            "possible_evidence_count",
        )
        labels = self._dept_plan_weekly_match_audit_for_llm(value)
        compact: dict[str, Any] = {}
        raw_to_label = {
            "owner_match_count": "负责人匹配数量",
            "owner_candidate_count": "负责人候选数量",
            "owner_cross_department_count": "负责人跨部门周报数量",
            "owner_selected_count": "入选的负责人周报数量",
            "non_owner_selected_count": "入选的非负责人周报数量",
            "selected_count": "入选强候选数量",
            "possible_evidence_count": "备选复核数量",
        }
        for key in keys:
            label = raw_to_label[key]
            if label in labels:
                compact[label] = labels[label]
        return compact

    def _record_failed_dept_plan_batch_item(
        self,
        *,
        item: Mapping[str, Any],
        batch_index: int,
        attempts: int,
        error: Exception | None,
        batch_errors: list[dict[str, Any]],
        reason_prefix: str,
    ) -> dict[str, Any]:
        plan_id = self._dept_plan_followup_id(item)
        error_type = type(error).__name__ if error is not None else "UnknownError"
        error_text = str(error) if error is not None else ""
        batch_errors.append(
            {
                "batch_index": batch_index,
                "attempts": attempts,
                "error_type": error_type,
                "error": error_text,
                "item_count": 1,
                "failed_plan_ids": [plan_id],
            }
        )
        return self._failed_dept_plan_judgement(
            item=item,
            reason=(
                f"{reason_prefix}（{error_type}）。"
                "这是模型调用或结构化解析失败，不能计为业务证据不足；需要重新触发该计划判断。"
            ),
        )

    def _dept_plan_completion_signal_label(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return "未明确"
        compact = re.sub(r"\s+", "", text)
        unfinished_markers = (
            "未完成", "未通过", "待", "尚未", "无法", "问题", "卡", "风险", "继续", "推进中", "进行中", "准备",
        )
        completion_markers = (
            "完成", "已完成", "闭环", "通过", "输出", "发布", "归档", "定稿", "验收", "到货", "入库", "下单",
            "解决", "修复", "回归", "验证",
        )
        has_completion = any(marker in compact for marker in completion_markers)
        has_unfinished = any(marker in compact for marker in unfinished_markers)
        if has_completion and not has_unfinished:
            return "有完成结果表述"
        if has_completion and has_unfinished:
            return "有完成表述但仍需核对待处理内容"
        return "未明确完成结果"

    def _dept_plan_yes_no(self, value: Any) -> str:
        if isinstance(value, bool):
            return "是" if value else "否"
        raw = str(value or "").strip().lower()
        if raw in {"1", "true", "yes", "y", "是"}:
            return "是"
        if raw in {"0", "false", "no", "n", "否"}:
            return "否"
        return "未标注"

    def _dept_plan_confidence_label(self, value: Any) -> str:
        raw = str(value or "low").strip().lower()
        return DEPT_PLAN_CANDIDATE_CONFIDENCE_LABELS.get(raw, raw or "低")

    def _dept_plan_candidate_source_label(self, value: Any) -> str:
        raw = str(value or "recency_fallback").strip()
        return DEPT_PLAN_CANDIDATE_SOURCE_LABELS.get(raw, raw or "按时间兜底候选")

    def _dept_plan_filter_reason_label(self, value: Any) -> str:
        raw = str(value or "possible_evidence").strip()
        return DEPT_PLAN_FILTER_REASON_LABELS.get(raw, raw or "备选证据")

    def _dept_plan_opl_candidate_source_label(self, value: Any) -> str:
        raw = str(value or "opl_issue_trace").strip()
        return DEPT_PLAN_OPL_CANDIDATE_SOURCE_LABELS.get(raw, raw or "OPL 问题线索")

    def _dept_plan_opl_filter_reason_label(self, value: Any) -> str:
        raw = str(value or "possible_opl_evidence").strip()
        return DEPT_PLAN_OPL_FILTER_REASON_LABELS.get(raw, raw or "备选 OPL 证据")

    def _dept_plan_review_evidence_usage_label(self, candidate: Mapping[str, Any]) -> str:
        confidence = str(candidate.get("candidate_confidence") or "").strip().lower()
        has_clear_business_match = bool(candidate.get("business_keyword_support") or candidate.get("strong_keyword_support"))
        if candidate.get("owner_match") and (has_clear_business_match or confidence in {"high", "medium"}):
            return "可结合计划内容作为补充判断证据"
        return "只能作为人工复核线索，不能单独证明完成"

    def _dept_plan_self_eval_type_label(self, value: Any) -> str:
        raw = str(value or "").strip()
        return DEPT_PLAN_SELF_EVAL_TYPE_LABELS.get(raw, raw or "未标注类型")

    def _sanitize_dept_plan_user_text(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""

        for token, labels in DEPT_PLAN_INTERNAL_BOOL_LABELS.items():
            token_pattern = rf"`?{re.escape(token)}`?"
            text = re.sub(
                rf"(?<![A-Za-z0-9_]){token_pattern}\s*(?:为|=|:|是)\s*(?:true|True|TRUE|是|匹配)(?![A-Za-z0-9_])",
                labels[0],
                text,
            )
            text = re.sub(
                rf"(?<![A-Za-z0-9_]){token_pattern}\s*(?:为|=|:|是)\s*(?:false|False|FALSE|否|不匹配)(?![A-Za-z0-9_])",
                labels[1],
                text,
            )

        for confidence, label in DEPT_PLAN_CANDIDATE_CONFIDENCE_LABELS.items():
            text = re.sub(
                rf"(?<![A-Za-z0-9_])`?candidate_confidence`?\s*(?:为|=|:|是)\s*{re.escape(confidence)}(?![A-Za-z0-9_])",
                f"候选可信度为{label}",
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(
                rf"候选可信度\s*(?:为|=|:|是)\s*{re.escape(confidence)}(?![A-Za-z0-9_])",
                f"候选可信度为{label}",
                text,
                flags=re.IGNORECASE,
            )

        for token, label in sorted(DEPT_PLAN_INTERNAL_TOKEN_LABELS.items(), key=lambda item: len(item[0]), reverse=True):
            text = re.sub(
                rf"(?<![A-Za-z0-9_])`?{re.escape(token)}`?(?![A-Za-z0-9_])",
                label,
                text,
            )

        for raw, label in sorted(DEPT_PLAN_INTERNAL_VALUE_LABELS.items(), key=lambda item: len(item[0]), reverse=True):
            text = re.sub(
                rf"(?<![A-Za-z0-9_])`?{re.escape(raw)}`?(?![A-Za-z0-9_])",
                label,
                text,
            )

        text = re.sub(r"无\s*owner\s*用户", "未找到计划负责人本人提交的周报", text, flags=re.IGNORECASE)
        text = re.sub(r"owner\s*用户", "计划负责人", text, flags=re.IGNORECASE)
        text = re.sub(r"(?<![A-Za-z0-9_])owner(?![A-Za-z0-9_])", "负责人", text, flags=re.IGNORECASE)
        text = re.sub(r"(?<![A-Za-z0-9_])true(?![A-Za-z0-9_])", "是", text, flags=re.IGNORECASE)
        text = re.sub(r"(?<![A-Za-z0-9_])false(?![A-Za-z0-9_])", "否", text, flags=re.IGNORECASE)
        return text

    def _normalize_weekly_plan_batch_judgements(
        self,
        *,
        batch_items: list[Mapping[str, Any]],
        output: WeeklyPlanBatchJudgementOutput,
    ) -> list[dict[str, Any]]:
        raw_by_id = {
            str(item.plan_id): item
            for item in output.judgements
        }
        normalized: list[dict[str, Any]] = []
        for item in batch_items:
            plan_id = self._plan_followup_id(item)
            raw = raw_by_id.get(plan_id)
            if raw is None:
                normalized.append(self._fallback_weekly_plan_judgement(item=item, reason="LLM 未返回该计划的判断。"))
                continue
            status = self._normalize_weekly_plan_status(raw.status)
            normalized.append(
                {
                    "plan_id": plan_id,
                    "user_name": str(item.get("user_name") or "未知人员"),
                    "department": str(item.get("department") or "未填写部门"),
                    "plan_date": str(item.get("plan_date") or ""),
                    "completion_date": str(item.get("completion_date") or "未找到"),
                    "plan_text": str(item.get("plan_text") or "").strip(),
                    "status": status,
                    "reason": self._clip_text(raw.reason, 220) or "LLM 未提供判断原因。",
                    "evidence": self._clip_text(raw.evidence, 180),
                }
            )
        return normalized

    def _normalize_dept_plan_batch_judgements(
        self,
        *,
        batch_items: list[Mapping[str, Any]],
        output: DeptPlanBatchJudgementOutput,
    ) -> list[dict[str, Any]]:
        raw_by_id = {
            str(item.plan_id): item
            for item in output.judgements
        }
        normalized: list[dict[str, Any]] = []
        for item in batch_items:
            plan_id = self._dept_plan_followup_id(item)
            raw = raw_by_id.get(plan_id)
            if raw is None:
                normalized.append(
                    self._failed_dept_plan_judgement(
                        item=item,
                        reason="LLM 未返回该计划的结构化判断；该情况属于模型输出异常，不能计为业务证据不足。",
                    )
                )
                continue
            status = self._normalize_dept_plan_status(raw.status)
            reason = self._sanitize_dept_plan_user_text(raw.reason)
            evidence = self._sanitize_dept_plan_user_text(raw.evidence)
            normalized.append(
                {
                    "plan_id": plan_id,
                    "department": str(item.get("department") or "未填写部门"),
                    "owner_user": str(item.get("owner_user") or ""),
                    "plan_month": str(item.get("plan_month") or ""),
                    "due_date": str(item.get("due_date") or ""),
                    "target": str(item.get("target") or ""),
                    "plan_text": str(item.get("plan_text") or "").strip(),
                    "status": status,
                    "reason": self._clip_text(reason, 240) or "LLM 未提供判断原因。",
                    "evidence": self._clip_text(evidence, 220),
                }
            )
        return normalized

    def _fallback_weekly_plan_judgement(self, *, item: Mapping[str, Any], reason: str) -> dict[str, Any]:
        return {
            "plan_id": self._plan_followup_id(item),
            "user_name": str(item.get("user_name") or "未知人员"),
            "department": str(item.get("department") or "未填写部门"),
            "plan_date": str(item.get("plan_date") or ""),
            "completion_date": str(item.get("completion_date") or "未找到"),
            "plan_text": str(item.get("plan_text") or "").strip(),
            "status": "证据不足",
            "reason": reason,
            "evidence": "",
        }

    def _fallback_dept_plan_judgement(self, *, item: Mapping[str, Any], reason: str) -> dict[str, Any]:
        return {
            "plan_id": self._dept_plan_followup_id(item),
            "department": str(item.get("department") or "未填写部门"),
            "owner_user": str(item.get("owner_user") or ""),
            "plan_month": str(item.get("plan_month") or ""),
            "due_date": str(item.get("due_date") or ""),
            "target": str(item.get("target") or ""),
            "plan_text": str(item.get("plan_text") or "").strip(),
            "status": "证据不足",
            "reason": reason,
            "evidence": "",
        }

    def _failed_dept_plan_judgement(self, *, item: Mapping[str, Any], reason: str) -> dict[str, Any]:
        return {
            "plan_id": self._dept_plan_followup_id(item),
            "department": str(item.get("department") or "未填写部门"),
            "owner_user": str(item.get("owner_user") or ""),
            "plan_month": str(item.get("plan_month") or ""),
            "due_date": str(item.get("due_date") or ""),
            "target": str(item.get("target") or ""),
            "plan_text": str(item.get("plan_text") or "").strip(),
            "status": DEPT_PLAN_JUDGEMENT_FAILED_STATUS,
            "reason": reason,
            "evidence": "",
        }

    def _format_mysql_weekly_plan_llm_report(
        self,
        *,
        source_output: Mapping[str, Any],
        followups: list[Mapping[str, Any]],
        judgements: list[dict[str, Any]],
    ) -> str:
        date_range = source_output.get("query_date_range") if isinstance(source_output.get("query_date_range"), Mapping) else {}
        pairing_summary = source_output.get("pairing_summary") if isinstance(source_output.get("pairing_summary"), Mapping) else {}
        plan_start = date_range.get("last_week_start", "")
        plan_end = date_range.get("last_week_end", "")
        done_start = date_range.get("this_week_start", "")
        done_end = date_range.get("this_week_end", "")
        ordered = self._ordered_weekly_plan_judgements(followups=followups, judgements=judgements)
        status_counts = self._weekly_plan_status_counts(ordered)

        lines = [
            "计划完成情况报告",
            "",
            "汇总：",
            f"- 计划范围：{plan_start} 至 {plan_end}",
            f"- 完成记录范围：{done_start} 至 {done_end}",
            f"- 共追踪计划 {len(ordered)} 条，周配对 {pairing_summary.get('weekly_pair_count', 0)} 组。",
            f"- 已完成 {status_counts.get('已完成', 0)} 条，部分完成 {status_counts.get('部分完成', 0)} 条，未完成 {status_counts.get('未完成', 0)} 条，证据不足 {status_counts.get('证据不足', 0)} 条。",
            "- 判定方式：后端先按人员、部门和日期配对计划与完成记录，再由大模型分批做语义判断；本报告因大模型汇总生成失败，由后端按结构化判断兜底生成。",
            "",
        ]
        unfinished = [item for item in ordered if item["status"] in {"部分完成", "未完成", "证据不足"}]
        if unfinished:
            lines.extend(
                [
                    "未完全完成/需关注：",
                    "| 人员 | 部门 | 计划日期 | 后续完成日期 | 状态 | 计划内容 | 判断依据 |",
                    "|---|---|---|---|---|---|---|",
                ]
            )
            for item in unfinished:
                lines.append(self._weekly_plan_report_row(item))
            lines.append("")

        lines.extend(
            [
                "全部计划明细：",
                "| 人员 | 部门 | 计划日期 | 后续完成日期 | 状态 | 计划内容 | 判断依据 |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for item in ordered:
            lines.append(self._weekly_plan_report_row(item))
        return "\n".join(lines).strip()

    def _format_mysql_weekly_plan_detail_section(
        self,
        *,
        source_output: Mapping[str, Any],
        ordered_judgements: list[dict[str, Any]],
    ) -> str:
        date_range = source_output.get("query_date_range") if isinstance(source_output.get("query_date_range"), Mapping) else {}
        lines = [
            "完整计划明细",
            "",
            f"计划范围：{date_range.get('last_week_start', '')} 至 {date_range.get('last_week_end', '')}",
            f"完成记录范围：{date_range.get('this_week_start', '')} 至 {date_range.get('this_week_end', '')}",
            "",
            "| # | 人员 | 部门 | 计划日期 | 后续完成日期 | 状态 | 计划内容 | 判断依据 |",
            "|---:|---|---|---|---|---|---|---|",
        ]
        for index, item in enumerate(ordered_judgements, start=1):
            evidence = str(item.get("evidence") or "").strip()
            reason = str(item.get("reason") or "").strip()
            basis = f"{reason}；依据：{evidence}" if evidence else reason
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(index),
                        self._markdown_table_cell(item.get("user_name")),
                        self._markdown_table_cell(item.get("department")),
                        self._markdown_table_cell(item.get("plan_date")),
                        self._markdown_table_cell(item.get("completion_date")),
                        self._markdown_table_cell(item.get("status")),
                        self._markdown_table_cell(self._clip_text(item.get("plan_text"), 140)),
                        self._markdown_table_cell(self._clip_text(basis, 200)),
                    ]
                )
                + " |"
            )
        return "\n".join(lines).strip()

    def _format_mysql_dept_plan_llm_report(
        self,
        *,
        source_output: Mapping[str, Any],
        ordered_judgements: list[dict[str, Any]],
    ) -> str:
        query_scope = source_output.get("query_scope") if isinstance(source_output.get("query_scope"), Mapping) else {}
        status_counts = self._dept_plan_status_counts(ordered_judgements)
        lines = [
            "三七计划完成情况报告",
            "",
            "汇总：",
            "- Runtime 执行说明：本报告中的已完成、部分完成、未完成、证据不足均指业务计划项状态，不代表 Agent Runtime 任务失败。",
            f"- 计划月份：{query_scope.get('month', '')}",
            f"- 部门范围：{query_scope.get('department') or '全部部门'}",
            (
                f"- 共核对计划 {len(ordered_judgements)} 条，已完成 {status_counts.get('已完成', 0)} 条，"
                f"部分完成 {status_counts.get('部分完成', 0)} 条，未完成 {status_counts.get('未完成', 0)} 条，"
                f"证据不足 {status_counts.get('证据不足', 0)} 条。"
            ),
            "- 判定方式：后端先从 MySQL 取计划、周报、负责人本人月度考核记录和 OPL 问题闭环证据，再由大模型分批做语义判断；本报告因大模型汇总生成失败，由后端按结构化判断兜底生成。",
            "",
        ]
        attention = [
            item
            for item in ordered_judgements
            if self._normalize_dept_plan_status(item.get("status")) in {"部分完成", "未完成", "证据不足"}
        ]
        if attention:
            lines.extend(
                [
                    "未完全完成/需关注：",
                    "| 部门 | 负责人 | 截止/周次 | 状态 | 计划内容 | 判断依据 |",
                    "|---|---|---|---|---|---|",
                ]
            )
            for item in attention:
                lines.append(self._dept_plan_report_row(item))
            lines.append("")
        lines.append(self._format_dept_plan_detail_section(source_output=source_output, ordered_judgements=ordered_judgements))
        return "\n".join(lines).strip()

    def _format_mysql_dept_plan_judgement_incomplete_report(
        self,
        *,
        source_output: Mapping[str, Any],
        ordered_judgements: list[dict[str, Any]],
        batch_errors: list[dict[str, Any]],
        failed_items: list[dict[str, Any]],
    ) -> str:
        query_scope = source_output.get("query_scope") if isinstance(source_output.get("query_scope"), Mapping) else {}
        failed_ids = {str(item.get("plan_id") or "") for item in failed_items}
        lines = [
            "三七计划完成情况报告（待复核）",
            "",
            "一致性校验未通过：",
            f"- 计划月份：{query_scope.get('month', '')}",
            f"- 部门范围：{query_scope.get('department') or '全部部门'}",
            f"- 共收到计划 {len(ordered_judgements)} 条，其中 {len(failed_items)} 条没有得到可靠的 LLM 结构化判断。",
            "- 本次不输出最终完成率，也不输出最终完成统计；失败项不会被计为证据不足。",
            "- 请重试失败批次，或在模型服务恢复后重新生成报告。",
            "",
        ]
        if batch_errors:
            lines.extend(["失败批次：", "| 批次 | 尝试次数 | 影响计划编号 | 错误类型 |", "|---:|---:|---|---|"])
            for error in batch_errors[:20]:
                plan_ids = error.get("failed_plan_ids")
                if isinstance(plan_ids, list):
                    plan_id_text = "、".join(str(item) for item in plan_ids[:12])
                else:
                    plan_id_text = "-"
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            self._markdown_table_cell(error.get("batch_index")),
                            self._markdown_table_cell(error.get("attempts")),
                            self._markdown_table_cell(plan_id_text),
                            self._markdown_table_cell(error.get("error_type")),
                        ]
                    )
                    + " |"
                )
            lines.append("")
        if failed_items:
            lines.extend(
                [
                    "未完成模型判断的计划：",
                    "| # | 部门 | 负责人 | 截止/周次 | 状态 | 计划内容 | 说明 |",
                    "|---:|---|---|---|---|---|---|",
                ]
            )
            for index, item in enumerate(failed_items[:80], start=1):
                due_target = " / ".join(
                    value
                    for value in (
                        str(item.get("due_date") or "").strip(),
                        str(item.get("target") or "").strip(),
                    )
                    if value
                )
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(index),
                            self._markdown_table_cell(item.get("department")),
                            self._markdown_table_cell(item.get("owner_user")),
                            self._markdown_table_cell(due_target),
                            self._markdown_table_cell(item.get("status")),
                            self._markdown_table_cell(self._clip_text(item.get("plan_text"), 140)),
                            self._markdown_table_cell(self._clip_text(item.get("reason"), 180)),
                        ]
                    )
                    + " |"
                )
            lines.append("")
        completed_judgement_count = sum(
            1
            for item in ordered_judgements
            if str(item.get("plan_id") or "") not in failed_ids
        )
        lines.extend(
            [
                f"已得到结构化判断的计划：{completed_judgement_count} 条。",
                "这些判断仅作为排查参考，因存在失败批次，本次不会据此发布最终完成率。",
            ]
        )
        return "\n".join(lines).strip()

    def _replace_dept_plan_simple_attention_lists(
        self,
        *,
        report_text: str,
        ordered_judgements: list[dict[str, Any]],
    ) -> str:
        unfinished_section = self._format_dept_plan_simple_status_list(
            heading="### （一）明确未完成事项",
            status="未完成",
            ordered_judgements=ordered_judgements,
        )
        insufficient_section = self._format_dept_plan_simple_status_list(
            heading="### （三）证据不足需人工核查事项",
            status="证据不足",
            ordered_judgements=ordered_judgements,
        )
        text = str(report_text or "").strip()
        if not text:
            return "\n\n".join(["## 三、重点未完成/证据不足事项", unfinished_section, insufficient_section])

        text, replaced_unfinished = self._replace_markdown_subsection(
            text=text,
            heading_keywords=("明确未完成",),
            replacement=unfinished_section,
        )
        text, replaced_insufficient = self._replace_markdown_subsection(
            text=text,
            heading_keywords=("证据不足",),
            replacement=insufficient_section,
        )
        if replaced_unfinished and replaced_insufficient:
            return text

        attention_section = "\n\n".join(
            [
                "## 三、重点未完成/证据不足事项",
                unfinished_section,
                insufficient_section,
            ]
        )
        return f"{text.rstrip()}\n\n{attention_section}".strip()

    def _replace_markdown_subsection(
        self,
        *,
        text: str,
        heading_keywords: tuple[str, ...],
        replacement: str,
    ) -> tuple[str, bool]:
        lines = text.splitlines()
        start_index: int | None = None
        for index, line in enumerate(lines):
            compact = line.strip()
            if compact.startswith("###") and all(keyword in compact for keyword in heading_keywords):
                start_index = index
                break
        if start_index is None:
            return text, False

        end_index = len(lines)
        for index in range(start_index + 1, len(lines)):
            compact = lines[index].strip()
            if compact.startswith("###") or compact.startswith("## "):
                end_index = index
                break
        merged = [
            *lines[:start_index],
            *replacement.splitlines(),
            *lines[end_index:],
        ]
        return "\n".join(merged).strip(), True

    def _format_dept_plan_simple_status_list(
        self,
        *,
        heading: str,
        status: str,
        ordered_judgements: list[dict[str, Any]],
    ) -> str:
        items = [
            item
            for item in ordered_judgements
            if self._normalize_dept_plan_status(item.get("status")) == status
        ]
        lines = [f"{heading}（{len(items)}项）"]
        if not items:
            lines.append("无。")
            return "\n".join(lines)

        for index, item in enumerate(items, start=1):
            owner = str(item.get("owner_user") or "").strip() or "未明确负责人"
            department = str(item.get("department") or "").strip()
            plan_text = self._clip_text(item.get("plan_text"), 120)
            suffix = f"（{department}）" if department else ""
            lines.append(f"{index}. {owner}：{plan_text}{suffix}")
        return "\n".join(lines)

    def _format_dept_plan_detail_section(
        self,
        *,
        source_output: Mapping[str, Any],
        ordered_judgements: list[dict[str, Any]],
    ) -> str:
        query_scope = source_output.get("query_scope") if isinstance(source_output.get("query_scope"), Mapping) else {}
        pairing_summary = source_output.get("pairing_summary") if isinstance(source_output.get("pairing_summary"), Mapping) else {}
        lines = [
            "完整三七计划明细",
            "",
            f"计划月份：{query_scope.get('month', '')}",
            f"周报证据范围：{query_scope.get('weekly_evidence_start', '')} 至 {query_scope.get('weekly_evidence_end', '')}",
        ]
        if self._dept_plan_has_opl_source(source_output):
            lines.append(
                "OPL 问题闭环证据："
                f"取数 {pairing_summary.get('opl_issue_count', 0)} 条，"
                f"计划相关候选 {pairing_summary.get('opl_issue_candidate_count', 0)} 条，"
                f"未闭环候选 {pairing_summary.get('open_opl_issue_candidate_count', 0)} 条；"
                "未闭环 OPL 只作为风险/卡点证据，不能单独等同计划未完成。"
            )
        lines.extend(
            [
                "",
                "| # | 部门 | 负责人 | 截止/周次 | 状态 | 计划内容 | 判断依据 |",
                "|---:|---|---|---|---|---|---|",
            ]
        )
        for index, item in enumerate(ordered_judgements, start=1):
            evidence = str(item.get("evidence") or "").strip()
            reason = str(item.get("reason") or "").strip()
            basis = f"{reason}；依据：{evidence}" if evidence else reason
            due_target = " / ".join(
                value for value in (str(item.get("due_date") or "").strip(), str(item.get("target") or "").strip()) if value
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(index),
                        self._markdown_table_cell(item.get("department")),
                        self._markdown_table_cell(item.get("owner_user")),
                        self._markdown_table_cell(due_target),
                        self._markdown_table_cell(item.get("status")),
                        self._markdown_table_cell(self._clip_text(item.get("plan_text"), 140)),
                        self._markdown_table_cell(self._clip_text(basis, 220)),
                    ]
                )
                + " |"
            )
        return "\n".join(lines).strip()

    def _ordered_weekly_plan_judgements(
        self,
        *,
        followups: list[Mapping[str, Any]],
        judgements: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        judged_by_id = {item["plan_id"]: item for item in judgements}
        return [
            judged_by_id.get(self._plan_followup_id(item))
            or self._fallback_weekly_plan_judgement(item=item, reason="未生成判断。")
            for item in followups
        ]

    def _ordered_dept_plan_judgements(
        self,
        *,
        followups: list[Mapping[str, Any]],
        judgements: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        judged_by_id = {item["plan_id"]: item for item in judgements}
        return [
            judged_by_id.get(self._dept_plan_followup_id(item))
            or self._failed_dept_plan_judgement(item=item, reason="未生成结构化判断；该情况属于系统判断缺失，不能计为业务证据不足。")
            for item in followups
        ]

    def _apply_dept_plan_closed_opl_completion_rule(
        self,
        *,
        followups: list[Mapping[str, Any]],
        ordered_judgements: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int]:
        followup_by_id = {self._dept_plan_followup_id(item): item for item in followups}
        adjusted: list[dict[str, Any]] = []
        override_count = 0
        for judgement in ordered_judgements:
            status = self._normalize_dept_plan_status(judgement.get("status"))
            if status in {"已完成", DEPT_PLAN_JUDGEMENT_FAILED_STATUS}:
                adjusted.append(judgement)
                continue
            item = followup_by_id.get(str(judgement.get("plan_id") or ""))
            support = self._dept_plan_closed_opl_completion_support(item) if item is not None else None
            if support is None:
                adjusted.append(judgement)
                continue
            issue_ref = str(
                support.get("issue_ref")
                or support.get("source_issue_id")
                or support.get("source_no")
                or "OPL记录"
            ).strip()
            progress = self._clip_text(
                support.get("solution_progress")
                or support.get("evidence_text")
                or support.get("remark")
                or support.get("issue_description"),
                140,
            )
            updated = dict(judgement)
            updated["status"] = "已完成"
            updated["reason"] = self._clip_text(
                "周报和负责人月度考核未找到该计划的完成证据；"
                "但关联 OPL 问题已闭环，且解决措施明确覆盖计划目标，按 OPL 闭环规则判为已完成。",
                240,
            )
            updated["evidence"] = self._clip_text(
                f"{issue_ref} 状态为已闭环，解决措施/最新进展：{progress}",
                220,
            )
            adjusted.append(updated)
            override_count += 1
        return adjusted, override_count

    def _dept_plan_closed_opl_completion_support(self, item: Mapping[str, Any]) -> Mapping[str, Any] | None:
        if self._dept_plan_has_weekly_or_self_eval_evidence(item):
            return None
        candidates_raw = item.get("opl_issue_candidates", [])
        if not isinstance(candidates_raw, list):
            return None
        candidates = [candidate for candidate in candidates_raw if isinstance(candidate, Mapping)]
        if any(self._dept_plan_open_opl_candidate_conflicts(candidate) for candidate in candidates):
            return None
        closed_candidates = [
            candidate
            for candidate in candidates
            if self._dept_plan_closed_opl_candidate_supports_completion(candidate)
        ]
        return closed_candidates[0] if closed_candidates else None

    def _dept_plan_has_weekly_or_self_eval_evidence(self, item: Mapping[str, Any]) -> bool:
        owner_evidence = item.get("owner_weekly_llm_evidence")
        if isinstance(owner_evidence, Mapping):
            if self._safe_positive_int(owner_evidence.get("related_report_count")) > 0:
                return True
            evidence_items = owner_evidence.get("evidence_items", [])
            if isinstance(evidence_items, list):
                for evidence_item in evidence_items:
                    if not isinstance(evidence_item, Mapping):
                        continue
                    if evidence_item.get("is_related"):
                        return True
                    if self._has_non_empty_list(evidence_item.get("evidence_snippets")):
                        return True
                    if str(evidence_item.get("completion_signal") or "").strip():
                        return True
                    if str(evidence_item.get("blockage_signal") or "").strip():
                        return True
        for key in (
            "weekly_done_candidates",
            "possible_weekly_evidence",
            "owner_self_eval_items",
            "self_eval_candidates",
        ):
            if self._has_non_empty_list(item.get(key)):
                return True
        return False

    def _dept_plan_open_opl_candidate_conflicts(self, candidate: Mapping[str, Any]) -> bool:
        if not candidate.get("open_issue") and self._dept_plan_opl_status_label(candidate) != "未闭环":
            return False
        return self._dept_plan_opl_candidate_is_highly_related(candidate)

    def _dept_plan_closed_opl_candidate_supports_completion(self, candidate: Mapping[str, Any]) -> bool:
        if candidate.get("open_issue") or self._dept_plan_opl_status_label(candidate) != "已闭环":
            return False
        progress_text = " ".join(
            str(candidate.get(key) or "").strip()
            for key in ("solution_progress", "evidence_text", "remark")
            if str(candidate.get(key) or "").strip()
        )
        if not progress_text:
            return False
        return self._dept_plan_opl_candidate_is_highly_related(candidate)

    def _dept_plan_opl_candidate_is_highly_related(self, candidate: Mapping[str, Any]) -> bool:
        keyword_support = bool(
            candidate.get("business_keyword_support")
            or candidate.get("strong_keyword_support")
            or candidate.get("exact_phrase")
        )
        person_support = bool(candidate.get("owner_match") or candidate.get("tracker_match"))
        department_support = bool(candidate.get("department_match"))
        confidence = str(candidate.get("candidate_confidence") or "").strip().lower()
        if keyword_support and (person_support or department_support):
            return True
        return confidence == "high" and (keyword_support or person_support or department_support)

    def _has_non_empty_list(self, value: Any) -> bool:
        return isinstance(value, list) and any(item not in (None, "", [], {}) for item in value)

    def _safe_positive_int(self, value: Any) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            return 0
        return max(0, number)

    def _weekly_plan_status_counts(self, items: list[Mapping[str, Any]]) -> dict[str, int]:
        status_counts = {status: 0 for status in ("已完成", "部分完成", "未完成", "证据不足")}
        for item in items:
            status = self._normalize_weekly_plan_status(item.get("status"))
            status_counts[status] = status_counts.get(status, 0) + 1
        return status_counts

    def _dept_plan_status_counts(self, items: list[Mapping[str, Any]]) -> dict[str, int]:
        status_counts = {status: 0 for status in ("已完成", "部分完成", "未完成", "证据不足", DEPT_PLAN_JUDGEMENT_FAILED_STATUS)}
        for item in items:
            status = self._normalize_dept_plan_status(item.get("status"))
            status_counts[status] = status_counts.get(status, 0) + 1
        return status_counts

    def _dept_plan_report_row(self, item: Mapping[str, Any]) -> str:
        evidence = str(item.get("evidence") or "").strip()
        reason = str(item.get("reason") or "").strip()
        basis = f"{reason}；依据：{evidence}" if evidence else reason
        due_target = " / ".join(
            value for value in (str(item.get("due_date") or "").strip(), str(item.get("target") or "").strip()) if value
        )
        return (
            "| "
            + " | ".join(
                [
                    self._markdown_table_cell(item.get("department")),
                    self._markdown_table_cell(item.get("owner_user")),
                    self._markdown_table_cell(due_target),
                    self._markdown_table_cell(item.get("status")),
                    self._markdown_table_cell(self._clip_text(item.get("plan_text"), 120)),
                    self._markdown_table_cell(self._clip_text(basis, 200)),
                ]
            )
            + " |"
        )

    def _weekly_plan_report_row(self, item: Mapping[str, Any]) -> str:
        evidence = str(item.get("evidence") or "").strip()
        reason = str(item.get("reason") or "").strip()
        basis = f"{reason}；依据：{evidence}" if evidence else reason
        return (
            "| "
            + " | ".join(
                [
                    self._markdown_table_cell(item.get("user_name")),
                    self._markdown_table_cell(item.get("department")),
                    self._markdown_table_cell(item.get("plan_date")),
                    self._markdown_table_cell(item.get("completion_date")),
                    self._markdown_table_cell(item.get("status")),
                    self._markdown_table_cell(self._clip_text(item.get("plan_text"), 120)),
                    self._markdown_table_cell(self._clip_text(basis, 180)),
                ]
            )
            + " |"
        )

    def _plan_followup_id(self, item: Mapping[str, Any]) -> str:
        value = item.get("plan_id")
        if value not in (None, ""):
            return str(value)
        return "|".join(
            str(part or "")
            for part in (
                item.get("user_name"),
                item.get("department"),
                item.get("plan_date"),
                item.get("plan_text"),
            )
        )

    def _dept_plan_followup_id(self, item: Mapping[str, Any]) -> str:
        value = item.get("plan_id")
        if value not in (None, ""):
            return str(value)
        return "|".join(
            str(part or "")
            for part in (
                item.get("department"),
                item.get("owner_user"),
                item.get("due_date"),
                item.get("plan_text"),
            )
        )

    def _normalize_weekly_plan_status(self, value: Any) -> str:
        text = str(value or "").strip()
        return text if text in WEEKLY_PLAN_STATUS_VALUES else "证据不足"

    def _normalize_dept_plan_status(self, value: Any) -> str:
        text = str(value or "").strip()
        return text if text in DEPT_PLAN_STATUS_VALUES else "证据不足"

    def _markdown_table_cell(self, value: Any) -> str:
        text = str(value or "").replace("\n", " ").replace("|", "\\|").strip()
        return text or "-"

    def _clip_text(self, value: Any, max_chars: int) -> str:
        text = str(value or "").strip()
        if len(text) <= max_chars:
            return text
        return f"{text[:max_chars].rstrip()}..."

    def _env_int(self, name: str, default: int) -> int:
        raw = os.getenv(name, "").strip()
        if not raw:
            return default
        try:
            parsed = int(raw)
        except ValueError:
            return default
        return parsed if parsed > 0 else default

    def _env_float(self, name: str, default: float) -> float:
        raw = os.getenv(name, "").strip()
        if not raw:
            return default
        try:
            parsed = float(raw)
        except ValueError:
            return default
        return parsed if parsed > 0 else default

    def _optional_positive_int(self, value: Any) -> int | None:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    def _has_mysql_weekly_plan_followups(self, output: Mapping[str, Any]) -> bool:
        followups = output.get("plan_followups")
        return isinstance(followups, list) and bool(followups)

    def _deterministic_report_haystack(self, *, payload: dict[str, Any], context: ContextStore | None) -> str:
        return " ".join(
            str(item or "")
            for item in (
                payload.get("prompt"),
                payload.get("query"),
                payload.get("question"),
                payload.get("context"),
                context.runtime.user_input if context is not None else "",
            )
        )

    def _asks_weekly_plan_completion(self, text: str) -> bool:
        if self._asks_weekly_work_completion(text):
            return True
        has_plan_scope = "计划" in text or "下周" in text or "周报" in text
        if not has_plan_scope:
            return False
        if any(keyword in text for keyword in ("是否完成", "有没有完成", "完成情况", "哪些完成", "哪些没完成", "没完成", "未完成", "闭环", "落实")):
            return True
        return "完成" in text and ("是否" in text or "本周工作" in text or "下一周" in text)

    def _asks_dept_plan_completion(self, text: str) -> bool:
        if "dept_plan_completion_context_text" in text:
            return True
        has_dept_plan_scope = any(keyword in text for keyword in ("三七计划", "计划书", "部门计划", "月度计划"))
        has_completion = any(
            keyword in text
            for keyword in ("完成", "没完成", "未完成", "完成率", "落地", "落实", "执行", "闭环")
        )
        return has_dept_plan_scope and has_completion

    def _asks_weekly_blockers(self, text: str) -> bool:
        if "weekly_blocker_context_text" in text:
            return True
        has_blocker_intent = any(keyword in text for keyword in ("卡点", "风险", "求助", "阻塞", "卡在", "卡住"))
        if "risk_and_help" in text and has_blocker_intent:
            return True
        has_week_scope = any(keyword in text for keyword in ("上周", "上一周", "上星期", "上个周", "本周", "这个周", "周报"))
        return has_week_scope and has_blocker_intent

    def _asks_weekly_work_completion(self, text: str) -> bool:
        has_week_scope = any(keyword in text for keyword in ("上周", "上一周", "上星期", "上个周", "本周", "这个周"))
        has_work_scope = "工作" in text or "完成记录" in text or "完成内容" in text
        has_completion = any(keyword in text for keyword in ("完成", "没完成", "未完成", "有没有完成", "是否完成"))
        return has_week_scope and has_work_scope and has_completion

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "context": {"type": ["string", "null"]},
                "style": {"type": ["string", "null"]},
                "audience": {"type": ["string", "null"]},
                "model_name": {"type": ["string", "null"]},
                "temperature": {"type": ["number", "null"]},
                "max_tokens": {"type": ["integer", "null"]},
                "rag_grounded": {"type": ["boolean", "string", "null"]},
                "query": {"type": ["string", "null"]},
                "question": {"type": ["string", "null"]},
            },
            "required": ["prompt"],
            "additionalProperties": True,
        }

    def get_output_schema(self) -> dict[str, Any]:
        return self.get_output_model().model_json_schema()
