from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field

from app.schemas.context import ContextStore
from app.schemas.tool import ToolResult
from app.tools.base import BaseTool
from app.utils import runtime_progress

try:  # Optional dependency. The tools fail clearly when it is missing.
    import pymysql
    from pymysql.cursors import DictCursor
except Exception:  # pragma: no cover - depends on deployment environment
    pymysql = None
    DictCursor = None


MYSQL_BUSINESS_TAGS = ["mysql", "business", "weekly_report"]
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DEFAULT_DEPARTMENT_ALIASES = (
    "产品部",
    "软件部",
    "行政部",
    "研发部",
    "测试部",
    "财务部",
    "人事部",
    "市场部",
    "销售部",
    "生产部",
    "采购部",
    "质量部",
    "技术部",
    "运营部",
    "工程部",
    "项目部",
    "算法部",
    "硬件部",
    "结构部",
    "电气",
    "电气部",
    "机械部",
    "装配制造",
    "装配制造部",
    "售后部",
    "客服部",
    "商务部",
    "仓储部",
    "物流部",
    "计划部",
    "工艺部",
    "总办",
    "办公室",
)
_DEPARTMENT_PATH_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9_]+(?:[/\\][\u4e00-\u9fffA-Za-z0-9_]+)+部")
_DEFAULT_DEPARTMENT_CANONICAL_ALIASES: dict[str, tuple[str, ...]] = {
    "产品": ("产品部", "电气", "电气部"),
    "软件": ("软件专业", "软件部"),
    "机械": ("机械专业", "机械设计部", "机械部"),
    "制造": ("生产制造部", "制造部", "生产部", "装配制造", "装配制造部"),
    "总办": ("总办", "干部群", "行政部", "办公室"),
    "质量": ("质量部",),
}
_DEFAULT_PERSON_ALIASES: dict[str, tuple[str, ...]] = {
    "李华": ("李华", "厉害"),
    "张超": ("张超", "大张超", "小张超"),
    "朱亦曼": ("朱亦曼", "朱一曼"),
    "马丽娜": ("马丽娜", "马立娜"),
}
_OWNER_TITLE_SUFFIXES = ("总", "主管", "经理", "主任", "老师")
_OWNER_NOISE_TOKENS = {
    "已做",
    "做完",
    "判断",
    "兼容",
    "后续",
    "改用",
    "进行",
    "识别",
    "优化",
    "完成",
    "设备",
    "算法",
    "目前",
    "为了",
    "全体",
    "厉害",
}
_COLLECTIVE_OWNER_TOKENS = {
    "全体",
}
_NO_BLOCKER_COMPACT_MARKERS = (
    "无卡点",
    "暂无卡点",
    "没有卡点",
    "无风险",
    "暂无风险",
    "没有风险",
    "无求助",
    "暂无求助",
    "没有求助",
    "无需协调",
    "无需协助",
    "不需要协调",
    "不需要协助",
    "无问题",
    "暂无问题",
    "没有问题",
    "卡点无",
    "卡点暂无",
    "卡点没有",
    "风险无",
    "风险暂无",
    "求助无",
    "求助暂无",
)
_NO_BLOCKER_COMPACT_EXACT = {
    "无",
    "暂无",
    "没有",
    "无卡点",
    "暂无卡点",
    "没有卡点",
    "目前无卡点",
    "目前暂无卡点",
    "目前没有卡点",
    "无风险",
    "暂无风险",
    "没有风险",
    "无求助",
    "暂无求助",
    "没有求助",
    "无问题",
    "暂无问题",
    "没有问题",
    "无卡点无需协调",
    "目前无卡点无需协调",
    "目前没有卡点无需协调",
    "工作有序开展暂无卡点",
}
_ACTIONABLE_BLOCKER_COMPACT_MARKERS = (
    "暂未",
    "尚未",
    "未完成",
    "未到",
    "未收到",
    "未解决",
    "未闭环",
    "未安装",
    "未开始",
    "未开展",
    "无法",
    "没法",
    "不能",
    "不稳定",
    "异常",
    "偏差",
    "延期",
    "延迟",
    "影响",
    "受阻",
    "阻塞",
    "卡住",
    "卡在",
    "困难",
    "失败",
    "报错",
    "暂停",
    "停滞",
    "缺少",
    "缺失",
    "不足",
    "装不上",
    "找不到",
    "依赖",
    "等待",
    "待确认",
    "待处理",
    "待协调",
    "待联调",
    "待测试",
    "待调试",
    "待到货",
    "待物料",
    "待软件",
    "待机械",
    "待采购",
    "待客户",
    "待领导",
)
_ACTIONABLE_BLOCKER_REGEXES = (
    re.compile(r"(?<![不无])需要.{0,18}(协助|协调|支持|确认|联调|处理|解决|配合|沟通|调整|优化|排查|提供)"),
    re.compile(r"(?<![不无])需.{0,18}(协助|协调|支持|确认|联调|处理|解决|配合|沟通|调整|优化|排查|提供)"),
)
_NEGATED_ACTION_PREFIXES = ("无", "暂无", "没有", "未见", "未发现", "没有发现", "不存在", "不", "无需", "不需要")


@dataclass(frozen=True)
class WeeklyReportSchema:
    table: str = "weekly_report_items"
    reports_table: str = "weekly_reports"
    user_name: str = "user_name"
    department: str = "department"
    report_date: str = "report_date"
    item_type: str = "item_type"
    item_text: str = "item_text"
    risk_and_help: str = "risk_and_help"
    this_week_raw: str = "this_week_raw"
    next_week_raw: str = "next_week_raw"
    report_end: str = "week_end"
    evidence_text: str = "evidence_text"
    source_doc_id: str = "source_doc_id"
    source_chunk_id: str = "source_chunk_id"
    sort_order: str = "sort_order"
    id: str = "id"

    @classmethod
    def from_env(cls) -> "WeeklyReportSchema":
        return cls(
            table=_env_name("MYSQL_WEEKLY_ITEMS_TABLE", cls.table),
            reports_table=_env_name("MYSQL_WEEKLY_REPORTS_TABLE", cls.reports_table),
            user_name=_env_name("MYSQL_WEEKLY_USER_COLUMN", cls.user_name),
            department=_env_name("MYSQL_WEEKLY_DEPARTMENT_COLUMN", cls.department),
            report_date=_env_name("MYSQL_WEEKLY_REPORT_DATE_COLUMN", cls.report_date),
            item_type=_env_name("MYSQL_WEEKLY_ITEM_TYPE_COLUMN", cls.item_type),
            item_text=_env_name("MYSQL_WEEKLY_ITEM_TEXT_COLUMN", cls.item_text),
            risk_and_help=_env_name("MYSQL_WEEKLY_RISK_AND_HELP_COLUMN", cls.risk_and_help),
            this_week_raw=_env_name("MYSQL_WEEKLY_THIS_WEEK_RAW_COLUMN", cls.this_week_raw),
            next_week_raw=_env_name("MYSQL_WEEKLY_NEXT_WEEK_RAW_COLUMN", cls.next_week_raw),
            report_end=_env_name("MYSQL_WEEKLY_REPORT_END_DATE_COLUMN", cls.report_end),
            evidence_text=_env_name("MYSQL_WEEKLY_EVIDENCE_TEXT_COLUMN", cls.evidence_text),
            source_doc_id=_env_name("MYSQL_WEEKLY_SOURCE_DOC_ID_COLUMN", cls.source_doc_id),
            source_chunk_id=_env_name("MYSQL_WEEKLY_SOURCE_CHUNK_ID_COLUMN", cls.source_chunk_id),
            sort_order=_env_name("MYSQL_WEEKLY_SORT_ORDER_COLUMN", cls.sort_order),
            id=_env_name("MYSQL_WEEKLY_ID_COLUMN", cls.id),
        )

    def validate(self) -> None:
        for value in self.__dict__.values():
            _validate_identifier(value)


@dataclass(frozen=True)
class DeptPlanSchema:
    table: str = "dept_plan_items"
    id: str = "id"
    doc_id: str = "doc_id"
    month: str = "month"
    department: str = "department"
    plan_text: str = "plan_text"
    owner_user: str = "owner_user"
    due_date: str = "due_date"
    target: str = "target"
    slide_no: str = "slide_no"
    evidence_text: str = "evidence_text"
    evidence_chunk_id: str = "evidence_chunk_id"

    @classmethod
    def from_env(cls) -> "DeptPlanSchema":
        return cls(
            table=_env_name("MYSQL_DEPT_PLAN_ITEMS_TABLE", cls.table),
            id=_env_name("MYSQL_DEPT_PLAN_ID_COLUMN", cls.id),
            doc_id=_env_name("MYSQL_DEPT_PLAN_DOC_ID_COLUMN", cls.doc_id),
            month=_env_name("MYSQL_DEPT_PLAN_MONTH_COLUMN", cls.month),
            department=_env_name("MYSQL_DEPT_PLAN_DEPARTMENT_COLUMN", cls.department),
            plan_text=_env_name("MYSQL_DEPT_PLAN_TEXT_COLUMN", cls.plan_text),
            owner_user=_env_name("MYSQL_DEPT_PLAN_OWNER_COLUMN", cls.owner_user),
            due_date=_env_name("MYSQL_DEPT_PLAN_DUE_DATE_COLUMN", cls.due_date),
            target=_env_name("MYSQL_DEPT_PLAN_TARGET_COLUMN", cls.target),
            slide_no=_env_name("MYSQL_DEPT_PLAN_SLIDE_NO_COLUMN", cls.slide_no),
            evidence_text=_env_name("MYSQL_DEPT_PLAN_EVIDENCE_TEXT_COLUMN", cls.evidence_text),
            evidence_chunk_id=_env_name("MYSQL_DEPT_PLAN_EVIDENCE_CHUNK_ID_COLUMN", cls.evidence_chunk_id),
        )

    def validate(self) -> None:
        for value in self.__dict__.values():
            _validate_identifier(value)


@dataclass(frozen=True)
class DeptSelfEvalSchema:
    table: str = "dept_self_eval_items"
    id: str = "id"
    doc_id: str = "doc_id"
    month: str = "month"
    department: str = "department"
    item_type: str = "item_type"
    item_text: str = "item_text"
    evidence_text: str = "evidence_text"
    evidence_chunk_id: str = "evidence_chunk_id"

    @classmethod
    def from_env(cls) -> "DeptSelfEvalSchema":
        return cls(
            table=_env_name("MYSQL_DEPT_SELF_EVAL_ITEMS_TABLE", cls.table),
            id=_env_name("MYSQL_DEPT_SELF_EVAL_ID_COLUMN", cls.id),
            doc_id=_env_name("MYSQL_DEPT_SELF_EVAL_DOC_ID_COLUMN", cls.doc_id),
            month=_env_name("MYSQL_DEPT_SELF_EVAL_MONTH_COLUMN", cls.month),
            department=_env_name("MYSQL_DEPT_SELF_EVAL_DEPARTMENT_COLUMN", cls.department),
            item_type=_env_name("MYSQL_DEPT_SELF_EVAL_ITEM_TYPE_COLUMN", cls.item_type),
            item_text=_env_name("MYSQL_DEPT_SELF_EVAL_ITEM_TEXT_COLUMN", cls.item_text),
            evidence_text=_env_name("MYSQL_DEPT_SELF_EVAL_EVIDENCE_TEXT_COLUMN", cls.evidence_text),
            evidence_chunk_id=_env_name("MYSQL_DEPT_SELF_EVAL_EVIDENCE_CHUNK_ID_COLUMN", cls.evidence_chunk_id),
        )

    def validate(self) -> None:
        for value in self.__dict__.values():
            _validate_identifier(value)


@dataclass(frozen=True)
class EmployeeSelfEvalSchema:
    reports_table: str = "employee_self_eval_reports"
    items_table: str = "employee_self_eval_items"
    report_id: str = "id"
    item_report_id: str = "report_id"
    doc_id: str = "doc_id"
    month: str = "month"
    user_name: str = "user_name"
    position: str = "position"
    department: str = "department"
    file_path: str = "file_path"
    sheet_name: str = "sheet_name"
    work_avg_completion_rate: str = "work_avg_completion_rate"
    management_avg_score: str = "management_avg_score"
    leader_rating_score: str = "leader_rating_score"
    admin_rating_score: str = "admin_rating_score"
    item_id: str = "id"
    section: str = "section"
    item_type: str = "item_type"
    item_text: str = "item_text"
    plan_text: str = "plan_text"
    result_text: str = "result_text"
    completion_time: str = "completion_time"
    completion_rate: str = "completion_rate"
    contact_user: str = "contact_user"
    unfinished_text: str = "unfinished_text"
    unresolved_text: str = "unresolved_text"
    reason_text: str = "reason_text"
    effect_text: str = "effect_text"
    source_sheet: str = "source_sheet"
    source_row: str = "source_row"
    evidence_text: str = "evidence_text"
    evidence_chunk_id: str = "evidence_chunk_id"

    @classmethod
    def from_env(cls) -> "EmployeeSelfEvalSchema":
        return cls(
            reports_table=_env_name("MYSQL_EMPLOYEE_SELF_EVAL_REPORTS_TABLE", cls.reports_table),
            items_table=_env_name("MYSQL_EMPLOYEE_SELF_EVAL_ITEMS_TABLE", cls.items_table),
            report_id=_env_name("MYSQL_EMPLOYEE_SELF_EVAL_REPORT_ID_COLUMN", cls.report_id),
            item_report_id=_env_name("MYSQL_EMPLOYEE_SELF_EVAL_ITEM_REPORT_ID_COLUMN", cls.item_report_id),
            doc_id=_env_name("MYSQL_EMPLOYEE_SELF_EVAL_DOC_ID_COLUMN", cls.doc_id),
            month=_env_name("MYSQL_EMPLOYEE_SELF_EVAL_MONTH_COLUMN", cls.month),
            user_name=_env_name("MYSQL_EMPLOYEE_SELF_EVAL_USER_COLUMN", cls.user_name),
            position=_env_name("MYSQL_EMPLOYEE_SELF_EVAL_POSITION_COLUMN", cls.position),
            department=_env_name("MYSQL_EMPLOYEE_SELF_EVAL_DEPARTMENT_COLUMN", cls.department),
            file_path=_env_name("MYSQL_EMPLOYEE_SELF_EVAL_FILE_PATH_COLUMN", cls.file_path),
            sheet_name=_env_name("MYSQL_EMPLOYEE_SELF_EVAL_SHEET_COLUMN", cls.sheet_name),
            work_avg_completion_rate=_env_name(
                "MYSQL_EMPLOYEE_SELF_EVAL_WORK_AVG_COMPLETION_RATE_COLUMN",
                cls.work_avg_completion_rate,
            ),
            management_avg_score=_env_name(
                "MYSQL_EMPLOYEE_SELF_EVAL_MANAGEMENT_AVG_SCORE_COLUMN",
                cls.management_avg_score,
            ),
            leader_rating_score=_env_name(
                "MYSQL_EMPLOYEE_SELF_EVAL_LEADER_RATING_SCORE_COLUMN",
                cls.leader_rating_score,
            ),
            admin_rating_score=_env_name(
                "MYSQL_EMPLOYEE_SELF_EVAL_ADMIN_RATING_SCORE_COLUMN",
                cls.admin_rating_score,
            ),
            item_id=_env_name("MYSQL_EMPLOYEE_SELF_EVAL_ITEM_ID_COLUMN", cls.item_id),
            section=_env_name("MYSQL_EMPLOYEE_SELF_EVAL_SECTION_COLUMN", cls.section),
            item_type=_env_name("MYSQL_EMPLOYEE_SELF_EVAL_ITEM_TYPE_COLUMN", cls.item_type),
            item_text=_env_name("MYSQL_EMPLOYEE_SELF_EVAL_ITEM_TEXT_COLUMN", cls.item_text),
            plan_text=_env_name("MYSQL_EMPLOYEE_SELF_EVAL_PLAN_TEXT_COLUMN", cls.plan_text),
            result_text=_env_name("MYSQL_EMPLOYEE_SELF_EVAL_RESULT_TEXT_COLUMN", cls.result_text),
            completion_time=_env_name("MYSQL_EMPLOYEE_SELF_EVAL_COMPLETION_TIME_COLUMN", cls.completion_time),
            completion_rate=_env_name("MYSQL_EMPLOYEE_SELF_EVAL_COMPLETION_RATE_COLUMN", cls.completion_rate),
            contact_user=_env_name("MYSQL_EMPLOYEE_SELF_EVAL_CONTACT_USER_COLUMN", cls.contact_user),
            unfinished_text=_env_name("MYSQL_EMPLOYEE_SELF_EVAL_UNFINISHED_TEXT_COLUMN", cls.unfinished_text),
            unresolved_text=_env_name("MYSQL_EMPLOYEE_SELF_EVAL_UNRESOLVED_TEXT_COLUMN", cls.unresolved_text),
            reason_text=_env_name("MYSQL_EMPLOYEE_SELF_EVAL_REASON_TEXT_COLUMN", cls.reason_text),
            effect_text=_env_name("MYSQL_EMPLOYEE_SELF_EVAL_EFFECT_TEXT_COLUMN", cls.effect_text),
            source_sheet=_env_name("MYSQL_EMPLOYEE_SELF_EVAL_SOURCE_SHEET_COLUMN", cls.source_sheet),
            source_row=_env_name("MYSQL_EMPLOYEE_SELF_EVAL_SOURCE_ROW_COLUMN", cls.source_row),
            evidence_text=_env_name("MYSQL_EMPLOYEE_SELF_EVAL_EVIDENCE_TEXT_COLUMN", cls.evidence_text),
            evidence_chunk_id=_env_name("MYSQL_EMPLOYEE_SELF_EVAL_EVIDENCE_CHUNK_ID_COLUMN", cls.evidence_chunk_id),
        )

    def validate(self) -> None:
        for value in self.__dict__.values():
            _validate_identifier(value)


@dataclass(frozen=True)
class OplIssueSchema:
    table: str = "opl_issue_items"
    id: str = "id"
    doc_id: str = "doc_id"
    file_path: str = "file_path"
    sheet_name: str = "sheet_name"
    source_row: str = "source_row"
    source_no: str = "source_no"
    department: str = "department"
    assembly: str = "assembly"
    issue_date: str = "issue_date"
    issue_description: str = "issue_description"
    status: str = "status"
    solution_progress: str = "solution_progress"
    priority: str = "priority"
    tracker_user: str = "tracker_user"
    owner_user: str = "owner_user"
    follow_date: str = "follow_date"
    remark: str = "remark"
    evidence_text: str = "evidence_text"
    evidence_chunk_id: str = "evidence_chunk_id"

    @classmethod
    def from_env(cls) -> "OplIssueSchema":
        return cls(
            table=_env_name("MYSQL_OPL_ISSUES_TABLE", cls.table),
            id=_env_name("MYSQL_OPL_ISSUE_ID_COLUMN", cls.id),
            doc_id=_env_name("MYSQL_OPL_ISSUE_DOC_ID_COLUMN", cls.doc_id),
            file_path=_env_name("MYSQL_OPL_ISSUE_FILE_PATH_COLUMN", cls.file_path),
            sheet_name=_env_name("MYSQL_OPL_ISSUE_SHEET_COLUMN", cls.sheet_name),
            source_row=_env_name("MYSQL_OPL_ISSUE_SOURCE_ROW_COLUMN", cls.source_row),
            source_no=_env_name("MYSQL_OPL_ISSUE_SOURCE_NO_COLUMN", cls.source_no),
            department=_env_name("MYSQL_OPL_ISSUE_DEPARTMENT_COLUMN", cls.department),
            assembly=_env_name("MYSQL_OPL_ISSUE_ASSEMBLY_COLUMN", cls.assembly),
            issue_date=_env_name("MYSQL_OPL_ISSUE_DATE_COLUMN", cls.issue_date),
            issue_description=_env_name("MYSQL_OPL_ISSUE_DESCRIPTION_COLUMN", cls.issue_description),
            status=_env_name("MYSQL_OPL_ISSUE_STATUS_COLUMN", cls.status),
            solution_progress=_env_name("MYSQL_OPL_ISSUE_SOLUTION_PROGRESS_COLUMN", cls.solution_progress),
            priority=_env_name("MYSQL_OPL_ISSUE_PRIORITY_COLUMN", cls.priority),
            tracker_user=_env_name("MYSQL_OPL_ISSUE_TRACKER_COLUMN", cls.tracker_user),
            owner_user=_env_name("MYSQL_OPL_ISSUE_OWNER_COLUMN", cls.owner_user),
            follow_date=_env_name("MYSQL_OPL_ISSUE_FOLLOW_DATE_COLUMN", cls.follow_date),
            remark=_env_name("MYSQL_OPL_ISSUE_REMARK_COLUMN", cls.remark),
            evidence_text=_env_name("MYSQL_OPL_ISSUE_EVIDENCE_TEXT_COLUMN", cls.evidence_text),
            evidence_chunk_id=_env_name("MYSQL_OPL_ISSUE_EVIDENCE_CHUNK_ID_COLUMN", cls.evidence_chunk_id),
        )

    def validate(self) -> None:
        for value in self.__dict__.values():
            _validate_identifier(value)


class MySQLBusinessClient:
    def __init__(
        self,
        schema: WeeklyReportSchema | None = None,
        dept_plan_schema: DeptPlanSchema | None = None,
        dept_self_eval_schema: DeptSelfEvalSchema | None = None,
        employee_self_eval_schema: EmployeeSelfEvalSchema | None = None,
        opl_issue_schema: OplIssueSchema | None = None,
    ) -> None:
        self.schema = schema or WeeklyReportSchema.from_env()
        self.dept_plan_schema = dept_plan_schema or DeptPlanSchema.from_env()
        self.dept_self_eval_schema = dept_self_eval_schema or DeptSelfEvalSchema.from_env()
        self.employee_self_eval_schema = employee_self_eval_schema or EmployeeSelfEvalSchema.from_env()
        self.opl_issue_schema = opl_issue_schema or OplIssueSchema.from_env()
        self.schema.validate()
        self.dept_plan_schema.validate()
        self.dept_self_eval_schema.validate()
        self.employee_self_eval_schema.validate()
        self.opl_issue_schema.validate()

    async def query_weekly_reports(
        self,
        *,
        user_name: str | None,
        department: str | None,
        start_date: str | None,
        end_date: str | None,
        item_type: str | None,
        limit: int,
        record_level: str = "items",
        include_evidence_text: bool = True,
    ) -> list[dict[str, Any]]:
        if record_level == "reports":
            sql, params = self._build_weekly_reports_sql(
                user_name=user_name,
                department=department,
                start_date=start_date,
                end_date=end_date,
                limit=limit,
            )
        else:
            sql, params = self._build_weekly_items_sql(
                user_name=user_name,
                department=department,
                start_date=start_date,
                end_date=end_date,
                item_type=item_type,
                limit=limit,
                include_evidence_text=include_evidence_text,
            )
        return await self._query(sql, params)

    async def query_weekly_report_items_lightweight(
        self,
        *,
        user_name: str | None,
        department: str | None,
        start_date: str | None,
        end_date: str | None,
        item_type: str | None,
        limit: int,
        include_evidence_text: bool = True,
    ) -> list[dict[str, Any]]:
        sql, params = self._build_weekly_items_lightweight_sql(
            user_name=user_name,
            department=department,
            start_date=start_date,
            end_date=end_date,
            item_type=item_type,
            limit=limit,
            include_evidence_text=include_evidence_text,
        )
        return await self._query(sql, params)

    async def query_weekly_reports_for_users(
        self,
        *,
        user_names: list[str],
        start_date: str | None,
        end_date: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        per_user_limit = max(
            1,
            min(_env_int("MYSQL_DEPT_PLAN_OWNER_REPORTS_PER_OWNER_LIMIT", 64), limit),
        )
        for user_name in user_names:
            reports.extend(
                await self.query_weekly_reports(
                    user_name=user_name,
                    department=None,
                    start_date=start_date,
                    end_date=end_date,
                    item_type=None,
                    limit=per_user_limit,
                    record_level="reports",
                    include_evidence_text=False,
                )
            )
        return _dedupe_rows(reports)

    async def compare_weekly_plan_done(
        self,
        *,
        user_name: str | None,
        department: str | None,
        last_week_start: str,
        last_week_end: str,
        this_week_start: str,
        this_week_end: str,
        limit: int,
    ) -> dict[str, Any]:
        completed_start = _earliest_date(last_week_start, this_week_start)
        plans = await self.query_weekly_reports(
            user_name=user_name,
            department=department,
            start_date=last_week_start,
            end_date=last_week_end,
            item_type="next_week_plan",
            limit=limit,
            include_evidence_text=True,
        )
        completed = await self.query_weekly_reports(
            user_name=user_name,
            department=department,
            start_date=completed_start,
            end_date=this_week_end,
            item_type="this_week_work",
            limit=limit,
            include_evidence_text=True,
        )
        completed = _dedupe_rows(
            [
                *completed,
                *await self.query_weekly_reports(
                    user_name=user_name,
                    department=department,
                    start_date=completed_start,
                    end_date=this_week_end,
                    item_type="this_week_done",
                    limit=limit,
                    include_evidence_text=True,
                ),
            ]
        )
        followup_evidence = _build_weekly_plan_followup_evidence(plans=plans, completed=completed)
        return {
            "last_week_plans": plans,
            "this_week_completed": completed,
            "candidate_matches": self._candidate_matches(plans=plans, completed=completed),
            **followup_evidence,
        }

    async def monthly_department_analysis(
        self,
        *,
        department: str,
        month: str,
        limit: int,
    ) -> dict[str, Any]:
        start_date, end_date = _month_range(month)
        weekly_items = await self.query_weekly_reports(
            user_name=None,
            department=department,
            start_date=start_date,
            end_date=end_date,
            item_type=None,
            limit=limit,
            include_evidence_text=True,
        )
        opl_issues = await self.query_opl_issues(
            department=department,
            start_date=start_date,
            end_date=end_date,
            status=None,
            owner_user=None,
            priority=None,
            doc_id=None,
            keyword=None,
            owner_names=[],
            include_open_before_start=True,
            limit=limit,
        )
        return {
            "dept_plans": [item for item in weekly_items if item.get("item_type") in {"department_plan", "dept_plan"}],
            "weekly_items": [
                item for item in weekly_items if item.get("item_type") in {"this_week_work", "this_week_done", "next_week_plan"}
            ],
            "self_eval_items": [item for item in weekly_items if item.get("item_type") in {"self_eval", "department_self_eval"}],
            "opl_issues": opl_issues,
            "opl_summary": _summarize_opl_issues(opl_issues),
        }

    async def query_dept_plan_items(
        self,
        *,
        month: str,
        department: str | None,
        doc_id: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        sql, params = self._build_dept_plan_items_sql(
            month=month,
            department=department,
            doc_id=doc_id,
            limit=limit,
        )
        return await self._query(sql, params)

    async def query_dept_self_eval_items(
        self,
        *,
        month: str,
        department: str | None,
        item_type: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        sql, params = self._build_dept_self_eval_items_sql(
            month=month,
            department=department,
            item_type=item_type,
            limit=limit,
        )
        return await self._query(sql, params)

    async def query_employee_self_eval_items_for_users(
        self,
        *,
        month: str,
        user_names: list[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        if not user_names:
            return []
        sql, params = self._build_employee_self_eval_items_for_users_sql(
            month=month,
            user_names=user_names,
            limit=limit,
        )
        return await self._query(sql, params)

    async def query_opl_issues(
        self,
        *,
        department: str | None,
        start_date: str | None,
        end_date: str | None,
        status: str | None,
        owner_user: str | None,
        priority: str | None,
        doc_id: str | None,
        keyword: str | None,
        owner_names: list[str] | None = None,
        include_open_before_start: bool = True,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        sql, params = self._build_opl_issues_sql(
            department=department,
            start_date=start_date,
            end_date=end_date,
            status=status,
            owner_user=owner_user,
            priority=priority,
            doc_id=doc_id,
            keyword=keyword,
            owner_names=owner_names or [],
            include_open_before_start=include_open_before_start,
            limit=limit,
        )
        return await self._query(sql, params)

    async def compare_dept_plan_completion(
        self,
        *,
        month: str,
        department: str | None,
        doc_id: str | None,
        include_weekly: bool,
        include_self_eval: bool,
        followup_days: int,
        limit: int,
        include_opl: bool = True,
    ) -> dict[str, Any]:
        month_start, month_end = _month_range(month)
        evidence_end = _add_days(month_end, followup_days)
        plans = await self.query_dept_plan_items(
            month=month,
            department=department,
            doc_id=doc_id,
            limit=limit,
        )
        collaboration_weekly_reports: list[dict[str, Any]] = []
        owner_weekly_reports: list[dict[str, Any]] = []
        owner_query_names: list[str] = _dept_plan_owner_names(plans)
        if include_weekly:
            weekly_limit = _expanded_query_limit(limit)
            collaboration_weekly_reports = await self.query_weekly_reports(
                user_name=None,
                department=department,
                start_date=month_start,
                end_date=evidence_end,
                item_type=None,
                limit=weekly_limit,
                record_level="reports",
                include_evidence_text=False,
            )
            owner_query_names = _dept_plan_owner_names(
                plans,
                known_user_names=_person_names_from_rows(
                    collaboration_weekly_reports,
                    keys=("user_name", "提交人"),
                ),
            )
            owner_weekly_reports = await self.query_weekly_reports_for_users(
                user_names=owner_query_names,
                start_date=month_start,
                end_date=evidence_end,
                limit=weekly_limit,
            )
        owner_self_eval_items: list[dict[str, Any]] | None = None
        if include_self_eval:
            owner_self_eval_items = await self.query_employee_self_eval_items_for_users(
                month=month,
                user_names=owner_query_names,
                limit=limit,
            )
        opl_issues: list[dict[str, Any]] = []
        if include_opl:
            opl_issues = await self.query_opl_issues(
                department=department,
                start_date=month_start,
                end_date=evidence_end,
                status=None,
                owner_user=None,
                priority=None,
                doc_id=None,
                keyword=None,
                owner_names=[],
                include_open_before_start=True,
                limit=_expanded_query_limit(limit),
            )
        result = _build_dept_plan_completion_evidence(
            plans=plans,
            weekly_completed=[],
            collaboration_weekly_reports=collaboration_weekly_reports,
            owner_weekly_reports=owner_weekly_reports,
            owner_self_eval_items=owner_self_eval_items,
            opl_issues=opl_issues,
            known_user_names=owner_query_names,
        )
        result["query_scope"] = {
            "month": month,
            "department": department,
            "doc_id": doc_id,
            "plan_month_start": month_start,
            "plan_month_end": month_end,
            "weekly_evidence_start": month_start,
            "weekly_evidence_end": evidence_end,
            "followup_days": followup_days,
            "include_weekly": include_weekly,
            "include_self_eval": include_self_eval,
            "include_opl": include_opl,
        }
        _attach_dept_plan_completion_context(result=result)
        return result

    def _build_dept_plan_items_sql(
        self,
        *,
        month: str,
        department: str | None,
        doc_id: str | None,
        limit: int,
    ) -> tuple[str, list[Any]]:
        s = self.dept_plan_schema
        selected_columns = [
            ("p", s.id, "id"),
            ("p", s.doc_id, "doc_id"),
            ("p", s.month, "month"),
            ("p", s.department, "department"),
            ("p", s.plan_text, "plan_text"),
            ("p", s.owner_user, "owner_user"),
            ("p", s.due_date, "due_date"),
            ("p", s.target, "target"),
            ("p", s.slide_no, "slide_no"),
            ("p", s.evidence_text, "evidence_text"),
            ("p", s.evidence_chunk_id, "evidence_chunk_id"),
        ]
        selected_sql_parts = [
            f"{table_alias}.`{column}` AS `{alias}`" for table_alias, column, alias in selected_columns
        ]
        where = [f"p.`{s.month}` = %s"]
        params: list[Any] = [month]
        if department:
            department_filter, department_params = _department_filter_sql(
                column=f"p.`{s.department}`",
                department=department,
            )
            where.append(department_filter)
            params.extend(department_params)
        if doc_id:
            where.append(f"p.`{s.doc_id}` = %s")
            params.append(doc_id)
        sql = (
            f"SELECT {', '.join(selected_sql_parts)} "
            f"FROM `{s.table}` p "
            f"WHERE {' AND '.join(where)} "
            f"ORDER BY p.`{s.department}` ASC, p.`{s.due_date}` ASC, p.`{s.id}` ASC "
            "LIMIT %s"
        )
        params.append(max(1, min(limit, _env_int("MYSQL_BUSINESS_MAX_LIMIT", 2000))))
        return sql, params

    def _build_dept_self_eval_items_sql(
        self,
        *,
        month: str,
        department: str | None,
        item_type: str | None,
        limit: int,
    ) -> tuple[str, list[Any]]:
        s = self.dept_self_eval_schema
        selected_columns = [
            ("e", s.id, "id"),
            ("e", s.doc_id, "doc_id"),
            ("e", s.month, "month"),
            ("e", s.department, "department"),
            ("e", s.item_type, "item_type"),
            ("e", s.item_text, "item_text"),
            ("e", s.evidence_text, "evidence_text"),
            ("e", s.evidence_chunk_id, "evidence_chunk_id"),
        ]
        selected_sql_parts = [
            f"{table_alias}.`{column}` AS `{alias}`" for table_alias, column, alias in selected_columns
        ]
        where = [f"e.`{s.month}` = %s"]
        params: list[Any] = [month]
        if department:
            department_filter, department_params = _department_filter_sql(
                column=f"e.`{s.department}`",
                department=department,
            )
            where.append(department_filter)
            params.extend(department_params)
        if item_type:
            where.append(f"e.`{s.item_type}` = %s")
            params.append(_resolve_self_eval_item_type_alias(item_type))
        sql = (
            f"SELECT {', '.join(selected_sql_parts)} "
            f"FROM `{s.table}` e "
            f"WHERE {' AND '.join(where)} "
            f"ORDER BY e.`{s.department}` ASC, e.`{s.id}` ASC "
            "LIMIT %s"
        )
        params.append(max(1, min(limit, _env_int("MYSQL_BUSINESS_MAX_LIMIT", 2000))))
        return sql, params

    def _build_employee_self_eval_items_for_users_sql(
        self,
        *,
        month: str,
        user_names: list[str],
        limit: int,
    ) -> tuple[str, list[Any]]:
        s = self.employee_self_eval_schema
        query_names = _employee_self_eval_query_names(user_names)
        if not query_names:
            return "SELECT 1 AS `empty_result` WHERE 1 = 0", []

        selected_sql_parts = [
            f"r.`{s.report_id}` AS `report_id`",
            f"i.`{s.item_id}` AS `item_id`",
            f"COALESCE(i.`{s.doc_id}`, r.`{s.doc_id}`) AS `doc_id`",
            f"COALESCE(i.`{s.month}`, r.`{s.month}`) AS `month`",
            f"COALESCE(i.`{s.user_name}`, r.`{s.user_name}`) AS `user_name`",
            f"COALESCE(i.`{s.position}`, r.`{s.position}`) AS `position`",
            f"COALESCE(i.`{s.department}`, r.`{s.department}`) AS `department`",
            f"r.`{s.file_path}` AS `file_path`",
            f"r.`{s.sheet_name}` AS `report_sheet_name`",
            f"r.`{s.work_avg_completion_rate}` AS `work_avg_completion_rate`",
            f"r.`{s.management_avg_score}` AS `management_avg_score`",
            f"r.`{s.leader_rating_score}` AS `leader_rating_score`",
            f"r.`{s.admin_rating_score}` AS `admin_rating_score`",
            f"i.`{s.section}` AS `section`",
            f"i.`{s.item_type}` AS `item_type`",
            f"i.`{s.item_text}` AS `item_text`",
            f"i.`{s.plan_text}` AS `plan_text`",
            f"i.`{s.result_text}` AS `result_text`",
            f"i.`{s.completion_time}` AS `completion_time`",
            f"i.`{s.completion_rate}` AS `completion_rate`",
            f"i.`{s.contact_user}` AS `contact_user`",
            f"i.`{s.unfinished_text}` AS `unfinished_text`",
            f"i.`{s.unresolved_text}` AS `unresolved_text`",
            f"i.`{s.reason_text}` AS `reason_text`",
            f"i.`{s.effect_text}` AS `effect_text`",
            f"i.`{s.source_sheet}` AS `source_sheet`",
            f"i.`{s.source_row}` AS `source_row`",
            f"i.`{s.evidence_text}` AS `evidence_text`",
            f"i.`{s.evidence_chunk_id}` AS `evidence_chunk_id`",
        ]
        placeholders = ", ".join(["%s"] * len(query_names))
        sql = (
            f"SELECT {', '.join(selected_sql_parts)} "
            f"FROM `{s.reports_table}` r "
            f"LEFT JOIN `{s.items_table}` i ON i.`{s.item_report_id}` = r.`{s.report_id}` "
            f"WHERE r.`{s.month}` = %s AND r.`{s.user_name}` IN ({placeholders}) "
            f"ORDER BY r.`{s.user_name}` ASC, r.`{s.report_id}` ASC, i.`{s.source_row}` ASC, i.`{s.item_id}` ASC "
            "LIMIT %s"
        )
        params: list[Any] = [month, *query_names, max(1, min(limit, _env_int("MYSQL_BUSINESS_MAX_LIMIT", 2000)))]
        return sql, params

    def _build_opl_issues_sql(
        self,
        *,
        department: str | None,
        start_date: str | None,
        end_date: str | None,
        status: str | None,
        owner_user: str | None,
        priority: str | None,
        doc_id: str | None,
        keyword: str | None,
        owner_names: list[str],
        include_open_before_start: bool,
        limit: int,
    ) -> tuple[str, list[Any]]:
        s = self.opl_issue_schema
        selected_columns = [
            ("o", s.id, "id"),
            ("o", s.doc_id, "doc_id"),
            ("o", s.file_path, "file_path"),
            ("o", s.sheet_name, "sheet_name"),
            ("o", s.source_row, "source_row"),
            ("o", s.source_no, "source_no"),
            ("o", s.department, "department"),
            ("o", s.assembly, "assembly"),
            ("o", s.issue_date, "issue_date"),
            ("o", s.issue_description, "issue_description"),
            ("o", s.status, "status"),
            ("o", s.solution_progress, "solution_progress"),
            ("o", s.priority, "priority"),
            ("o", s.tracker_user, "tracker_user"),
            ("o", s.owner_user, "owner_user"),
            ("o", s.follow_date, "follow_date"),
            ("o", s.remark, "remark"),
            ("o", s.evidence_text, "evidence_text"),
            ("o", s.evidence_chunk_id, "evidence_chunk_id"),
        ]
        selected_sql_parts = [
            f"{table_alias}.`{column}` AS `{alias}`" for table_alias, column, alias in selected_columns
        ]
        where: list[str] = []
        params: list[Any] = []
        if start_date and end_date:
            if include_open_before_start:
                open_filter, open_params = _opl_open_status_sql(f"o.`{s.status}`")
                where.append(
                    "("
                    f"(o.`{s.issue_date}` BETWEEN %s AND %s) "
                    f"OR (o.`{s.follow_date}` BETWEEN %s AND %s) "
                    f"OR (o.`{s.issue_date}` <= %s AND {open_filter})"
                    ")"
                )
                params.extend([start_date, end_date, start_date, end_date, end_date, *open_params])
            else:
                where.append(
                    "("
                    f"(o.`{s.issue_date}` BETWEEN %s AND %s) "
                    f"OR (o.`{s.follow_date}` BETWEEN %s AND %s)"
                    ")"
                )
                params.extend([start_date, end_date, start_date, end_date])
        elif start_date:
            where.append(f"(o.`{s.issue_date}` >= %s OR o.`{s.follow_date}` >= %s)")
            params.extend([start_date, start_date])
        elif end_date:
            where.append(f"(o.`{s.issue_date}` <= %s OR o.`{s.follow_date}` <= %s)")
            params.extend([end_date, end_date])
        if department:
            department_filter, department_params = _department_filter_sql(
                column=f"o.`{s.department}`",
                department=department,
            )
            where.append(department_filter)
            params.extend(department_params)
        if status:
            status_filter, status_params = _opl_status_filter_sql(f"o.`{s.status}`", status)
            where.append(status_filter)
            params.extend(status_params)
        if owner_user:
            where.append(f"(o.`{s.owner_user}` = %s OR o.`{s.tracker_user}` = %s)")
            params.extend([owner_user, owner_user])
        elif owner_names:
            query_names = _employee_self_eval_query_names(owner_names)
            if query_names:
                placeholders = ", ".join(["%s"] * len(query_names))
                where.append(f"(o.`{s.owner_user}` IN ({placeholders}) OR o.`{s.tracker_user}` IN ({placeholders}))")
                params.extend([*query_names, *query_names])
        if priority:
            where.append(f"o.`{s.priority}` = %s")
            params.append(priority)
        if doc_id:
            where.append(f"o.`{s.doc_id}` = %s")
            params.append(doc_id)
        if keyword:
            like_value = f"%{keyword}%"
            where.append(
                "("
                f"o.`{s.issue_description}` LIKE %s "
                f"OR o.`{s.solution_progress}` LIKE %s "
                f"OR o.`{s.assembly}` LIKE %s "
                f"OR o.`{s.remark}` LIKE %s"
                ")"
            )
            params.extend([like_value, like_value, like_value, like_value])

        where_sql = f" WHERE {' AND '.join(where)}" if where else ""
        sql = (
            f"SELECT {', '.join(selected_sql_parts)} "
            f"FROM `{s.table}` o"
            f"{where_sql} "
            f"ORDER BY o.`{s.issue_date}` ASC, o.`{s.follow_date}` ASC, o.`{s.id}` ASC "
            "LIMIT %s"
        )
        params.append(max(1, min(limit, _env_int("MYSQL_BUSINESS_MAX_LIMIT", 2000))))
        return sql, params

    def _build_weekly_items_sql(
        self,
        *,
        user_name: str | None,
        department: str | None,
        start_date: str | None,
        end_date: str | None,
        item_type: str | None,
        limit: int,
        include_evidence_text: bool = True,
    ) -> tuple[str, list[Any]]:
        s = self.schema
        selected_columns = [
            ("i", s.id, "id"),
            ("i", s.user_name, "user_name"),
            ("i", s.department, "department"),
            ("i", s.report_date, "report_date"),
            ("i", s.item_type, "item_type"),
            ("i", s.item_text, "item_text"),
            ("r", s.risk_and_help, "risk_and_help"),
            ("i", s.source_doc_id, "source_doc_id"),
            ("i", s.source_chunk_id, "source_chunk_id"),
        ]
        selected_sql_parts = [
            f"{table_alias}.`{column}` AS `{alias}`" for table_alias, column, alias in selected_columns
        ]
        if include_evidence_text:
            selected_sql_parts.append(f"i.`{s.evidence_text}` AS `evidence_text`")
        else:
            selected_sql_parts.append("NULL AS `evidence_text`")
        where: list[str] = []
        params: list[Any] = []
        if user_name:
            where.append(f"i.`{s.user_name}` = %s")
            params.append(user_name)
        if department:
            department_filter, department_params = _department_filter_sql(
                column=f"i.`{s.department}`",
                department=department,
            )
            where.append(department_filter)
            params.extend(department_params)
        if start_date:
            where.append(f"i.`{s.report_date}` >= %s")
            params.append(start_date)
        if end_date:
            where.append(f"i.`{s.report_date}` <= %s")
            params.append(end_date)
        if item_type:
            where.append(f"i.`{s.item_type}` = %s")
            params.append(_resolve_item_type_alias(item_type))

        where_sql = f" WHERE {' AND '.join(where)}" if where else ""
        sql = (
            f"SELECT {', '.join(selected_sql_parts)} "
            f"FROM `{s.table}` i "
            f"LEFT JOIN `{s.reports_table}` r "
            f"ON r.`{s.source_doc_id}` = i.`{s.source_doc_id}` "
            f"AND r.`{s.user_name}` = i.`{s.user_name}` "
            f"AND r.`{s.report_date}` = i.`{s.report_date}`"
            f"{where_sql} "
            f"ORDER BY i.`{s.report_date}` ASC, i.`{s.user_name}` ASC, i.`{s.sort_order}` ASC "
            "LIMIT %s"
        )
        params.append(max(1, min(limit, _env_int("MYSQL_BUSINESS_MAX_LIMIT", 2000))))
        return sql, params

    def _build_weekly_items_lightweight_sql(
        self,
        *,
        user_name: str | None,
        department: str | None,
        start_date: str | None,
        end_date: str | None,
        item_type: str | None,
        limit: int,
        include_evidence_text: bool = True,
    ) -> tuple[str, list[Any]]:
        s = self.schema
        selected_columns = [
            ("i", s.id, "id"),
            ("i", s.user_name, "user_name"),
            ("i", s.department, "department"),
            ("i", s.report_date, "report_date"),
            ("i", s.item_type, "item_type"),
            ("i", s.item_text, "item_text"),
            ("i", s.source_doc_id, "source_doc_id"),
            ("i", s.source_chunk_id, "source_chunk_id"),
        ]
        selected_sql_parts = [
            f"{table_alias}.`{column}` AS `{alias}`" for table_alias, column, alias in selected_columns
        ]
        selected_sql_parts.append("NULL AS `risk_and_help`")
        if include_evidence_text:
            selected_sql_parts.append(f"i.`{s.evidence_text}` AS `evidence_text`")
        else:
            selected_sql_parts.append("NULL AS `evidence_text`")
        where: list[str] = []
        params: list[Any] = []
        if user_name:
            where.append(f"i.`{s.user_name}` = %s")
            params.append(user_name)
        if department:
            department_filter, department_params = _department_filter_sql(
                column=f"i.`{s.department}`",
                department=department,
            )
            where.append(department_filter)
            params.extend(department_params)
        if start_date:
            where.append(f"i.`{s.report_date}` >= %s")
            params.append(start_date)
        if end_date:
            where.append(f"i.`{s.report_date}` <= %s")
            params.append(end_date)
        if item_type:
            where.append(f"i.`{s.item_type}` = %s")
            params.append(_resolve_item_type_alias(item_type))

        where_sql = f" WHERE {' AND '.join(where)}" if where else ""
        sql = (
            f"SELECT {', '.join(selected_sql_parts)} "
            f"FROM `{s.table}` i "
            f"{where_sql} "
            "LIMIT %s"
        )
        params.append(max(1, min(limit, _env_int("MYSQL_BUSINESS_MAX_LIMIT", 2000))))
        return sql, params

    def _build_weekly_reports_sql(
        self,
        *,
        user_name: str | None,
        department: str | None,
        start_date: str | None,
        end_date: str | None,
        limit: int,
    ) -> tuple[str, list[Any]]:
        s = self.schema
        selected_columns = [
            ("r", s.id, "id"),
            ("r", s.user_name, "user_name"),
            ("r", s.department, "department"),
            ("r", s.report_date, "report_date"),
            ("r", s.report_end, "report_end"),
            ("r", s.this_week_raw, "this_week_raw"),
            ("r", s.next_week_raw, "next_week_raw"),
            ("r", s.risk_and_help, "risk_and_help"),
            ("r", s.source_doc_id, "source_doc_id"),
            ("r", s.source_chunk_id, "source_chunk_id"),
        ]
        selected_sql_parts = [
            f"{table_alias}.`{column}` AS `{alias}`" for table_alias, column, alias in selected_columns
        ]
        selected_sql_parts.extend(["'weekly_report' AS `item_type`", "NULL AS `item_text`", "NULL AS `evidence_text`"])
        where: list[str] = []
        params: list[Any] = []
        if user_name:
            where.append(f"r.`{s.user_name}` = %s")
            params.append(user_name)
        if department:
            department_filter, department_params = _department_filter_sql(
                column=f"r.`{s.department}`",
                department=department,
            )
            where.append(department_filter)
            params.extend(department_params)
        if start_date:
            where.append(f"r.`{s.report_date}` >= %s")
            params.append(start_date)
        if end_date:
            where.append(f"r.`{s.report_date}` <= %s")
            params.append(end_date)

        where_sql = f" WHERE {' AND '.join(where)}" if where else ""
        sql = (
            f"SELECT {', '.join(selected_sql_parts)} "
            f"FROM `{s.reports_table}` r"
            f"{where_sql} "
            f"ORDER BY r.`{s.report_date}` ASC, r.`{s.user_name}` ASC, r.`{s.id}` ASC "
            "LIMIT %s"
        )
        params.append(max(1, min(limit, _env_int("MYSQL_BUSINESS_MAX_LIMIT", 2000))))
        return sql, params

    async def _query(self, sql: str, params: list[Any]) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._query_sync, sql, params)

    def _query_sync(self, sql: str, params: list[Any]) -> list[dict[str, Any]]:
        if pymysql is None or DictCursor is None:
            raise RuntimeError("pymysql is not installed; install pymysql to enable MySQL business tools")

        connection = pymysql.connect(
            host=_required_env("MYSQL_HOST"),
            port=_env_int("MYSQL_PORT", 3306),
            user=_required_env("MYSQL_USER"),
            password=_required_env("MYSQL_PASSWORD"),
            database=_required_env("MYSQL_DATABASE"),
            charset=os.getenv("MYSQL_CHARSET", "utf8mb4"),
            cursorclass=DictCursor,
            read_timeout=_env_int("MYSQL_READ_TIMEOUT", 15),
            write_timeout=_env_int("MYSQL_WRITE_TIMEOUT", 15),
            connect_timeout=_env_int("MYSQL_CONNECT_TIMEOUT", 5),
        )
        try:
            with connection.cursor() as cursor:
                cursor.execute(sql, params)
                rows = cursor.fetchall()
        finally:
            connection.close()
        return [_normalize_row(row) for row in rows]

    def _candidate_matches(
        self,
        *,
        plans: list[dict[str, Any]],
        completed: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for plan in plans:
            plan_text = str(plan.get("item_text") or "")
            plan_user = _optional_str(plan.get("user_name"))
            plan_date = _row_date(plan)
            plan_tokens = _keyword_tokens(plan_text)
            if not plan_tokens:
                continue
            for done in completed:
                if not _same_user(plan_user, _optional_str(done.get("user_name"))):
                    continue
                if not _departments_compatible(plan.get("department"), done.get("department")):
                    continue
                done_date = _row_date(done)
                if plan_date and done_date and done_date <= plan_date:
                    continue
                done_text = str(done.get("item_text") or "")
                done_tokens = _keyword_tokens(done_text)
                if not done_tokens:
                    continue
                overlap = sorted(plan_tokens & done_tokens)
                if not overlap:
                    continue
                matches.append(
                    {
                        "plan_text": plan_text,
                        "done_text": done_text,
                        "user_name": plan.get("user_name"),
                        "department": plan.get("department"),
                        "plan_date": plan_date,
                        "done_date": done_date,
                        "overlap_keywords": overlap[:20],
                        "plan_evidence_text": _clip_text(plan.get("evidence_text"), 400),
                        "done_evidence_text": _clip_text(done.get("evidence_text"), 400),
                    }
                )
        return matches[: _env_int("MYSQL_BUSINESS_MAX_CANDIDATE_MATCHES", 200)]


class _MySQLBusinessTool(BaseTool):
    timeout: int = Field(default=30, gt=0)
    tags: list[str] = Field(default_factory=lambda: list(MYSQL_BUSINESS_TAGS))
    client: MySQLBusinessClient | None = Field(default=None, exclude=True)

    def get_routing_capability(self) -> dict[str, Any]:
        capability = super().get_routing_capability()
        capability["supported_task_types"] = [self.name]
        capability["default_task_type"] = self.name
        capability["supported_tags"] = list(self.tags)
        capability["max_concurrency"] = _env_int("MYSQL_BUSINESS_MAX_CONCURRENCY", 8)
        return capability

    def _client(self) -> MySQLBusinessClient:
        if self.client is None:
            self.client = MySQLBusinessClient()
        return self.client

    def _limit(self, payload: Mapping[str, Any], *, default: int = 500) -> int:
        return max(1, min(_safe_int(payload.get("limit"), default), _env_int("MYSQL_BUSINESS_MAX_LIMIT", 2000)))


class QueryWeeklyReportsTool(_MySQLBusinessTool):
    name: str = Field(default="query_weekly_reports")
    description: str = Field(default="查询结构化周报明细，如本周完成、下周计划、员工自填卡点 risk_and_help、部门或个人指定日期范围内的周报事项。")

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "user_name": {"type": ["string", "null"]},
                "department": {"type": ["string", "null"]},
                "start_date": {"type": ["string", "null"], "description": "YYYY-MM-DD"},
                "end_date": {"type": ["string", "null"], "description": "YYYY-MM-DD"},
                "item_type": {
                    "type": ["string", "null"],
                    "enum": ["this_week_work", "this_week_done", "next_week_plan", "self_eval", "department_plan", "dept_plan", "department_self_eval", None],
                },
                "record_level": {
                    "type": ["string", "null"],
                    "enum": ["items", "reports", None],
                    "default": "items",
                    "description": "items 查询 weekly_report_items 拆分事项；reports 查询 weekly_reports 主表，一人一周一条，适合卡点 risk_and_help。",
                },
                "include_evidence_text": {
                    "type": ["boolean", "null"],
                    "default": True,
                    "description": "是否返回 evidence_text。大范围查询时可设为 false，避免重复完整周报原文导致上下文过大。",
                },
                "limit": {"type": ["integer", "null"], "default": 500},
            },
            "additionalProperties": False,
        }

    def get_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "items": {"type": "array"},
                "count": {"type": "integer"},
            },
            "required": ["items", "count"],
            "additionalProperties": True,
        }

    async def _arun(self, payload: dict[str, Any], context: ContextStore | None = None) -> ToolResult:
        try:
            department = _optional_str(payload.get("department")) or _infer_department_from_runtime_context(context)
            record_level = _optional_record_level(payload.get("record_level")) or "items"
            if (
                "record_level" not in payload
                and _optional_item_type(payload.get("item_type")) is None
                and context is not None
                and _looks_like_weekly_blocker_query(context.runtime.user_input)
            ):
                record_level = "reports"
            items = await self._client().query_weekly_reports(
                user_name=_optional_str(payload.get("user_name")),
                department=department,
                start_date=_optional_date(payload.get("start_date")),
                end_date=_optional_date(payload.get("end_date")),
                item_type=_optional_item_type(payload.get("item_type")),
                limit=self._limit(payload),
                record_level=record_level,
                include_evidence_text=_optional_bool(payload.get("include_evidence_text"), default=True),
            )
        except Exception as exc:
            return self.build_result(
                success=False,
                error=f"query_weekly_reports failed: {exc}",
                metadata={"exception_type": type(exc).__name__},
            )
        return self.build_result(
            success=True,
            output={"items": items, "count": len(items)},
            metadata={"count": len(items)},
        )


class CompareWeeklyPlanDoneTool(_MySQLBusinessTool):
    name: str = Field(default="compare_weekly_plan_done")
    description: str = Field(default="查询上周下周计划与本周完成记录，用于判断计划是否完成或统计完成率。")

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "user_name": {"type": ["string", "null"]},
                "department": {"type": ["string", "null"]},
                "last_week_start": {"type": "string", "description": "YYYY-MM-DD"},
                "last_week_end": {"type": "string", "description": "YYYY-MM-DD"},
                "this_week_start": {"type": "string", "description": "YYYY-MM-DD"},
                "this_week_end": {"type": "string", "description": "YYYY-MM-DD"},
                "trace_only_empty_risk_from_output_key": {
                    "type": ["string", "null"],
                    "description": "可选。上游 query_weekly_reports 任务的 output_key。设置后，仅追溯该结果中员工自填卡点为空或仅表示无卡点/无风险/无求助的人员。",
                },
                "weekly_blocker_classification_output_key": {
                    "type": ["string", "null"],
                    "description": "可选。上游 classify_weekly_blockers 的 output_key。设置后，按 LLM 语义分类结果仅追溯 needs_trace=true 的人员。",
                },
                "trace_weeks": {
                    "type": ["integer", "null"],
                    "default": 1,
                    "description": "卡点追溯周数。默认追溯上一周；周报卡点兜底链路使用 2，分别追溯上一周和上上周两个窗口。",
                },
                "include_historical_blockers": {
                    "type": ["boolean", "null"],
                    "default": False,
                    "description": "是否收集被追溯人员前 1-2 周历史卡点候选，并交给后续 LLM 判断是否已解决。",
                },
                "limit": {"type": ["integer", "null"], "default": 500},
            },
            "required": ["last_week_start", "last_week_end", "this_week_start", "this_week_end"],
            "additionalProperties": False,
        }

    def get_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "last_week_plans": {"type": "array"},
                "this_week_completed": {"type": "array"},
                "candidate_matches": {"type": "array"},
                "weekly_pairs": {"type": "array"},
                "plan_followups": {"type": "array"},
                "pairing_summary": {"type": "object"},
                "query_date_range": {"type": "object"},
                "trace_scope": {"type": "object"},
                "trace_windows": {"type": "array"},
                "direct_blocker_people": {"type": "array"},
                "historical_blocker_candidates": {"type": "array"},
                "weekly_blocker_context_text": {
                    "type": "string",
                    "description": "面向 text_generate_tool 的周报卡点压缩上下文，避免注入完整周报大对象。",
                },
                "weekly_blocker_people": {"type": "array"},
                "plan_tracking_context_text": {
                    "type": "string",
                    "description": "面向 text_generate_tool 的计划追踪压缩上下文，避免注入完整 weekly_plan_comparison 大对象。",
                },
                "final_report": {
                    "type": "string",
                    "description": "计划完成追踪的确定性报告，可由 text_generate_tool 直接返回，避免大范围计划判断超时。",
                },
                "final_report_type": {"type": "string"},
            },
            "required": [
                "last_week_plans",
                "this_week_completed",
                "candidate_matches",
                "weekly_pairs",
                "plan_followups",
                "pairing_summary",
            ],
            "additionalProperties": True,
        }

    async def _arun(self, payload: dict[str, Any], context: ContextStore | None = None) -> ToolResult:
        trace_source_key: str | None = None
        classification_source_key: str | None = None
        try:
            department = _optional_str(payload.get("department")) or _infer_department_from_runtime_context(context)
            date_range = _normalize_compare_weekly_date_range(
                last_week_start=_required_date(payload.get("last_week_start"), "last_week_start"),
                last_week_end=_required_date(payload.get("last_week_end"), "last_week_end"),
                this_week_start=_required_date(payload.get("this_week_start"), "this_week_start"),
                this_week_end=_required_date(payload.get("this_week_end"), "this_week_end"),
                user_input=context.runtime.user_input if context is not None else None,
                runtime_timestamp=context.runtime.timestamp if context is not None else None,
                timezone_name=str(context.runtime.metadata.get("client_timezone") or "UTC") if context is not None else "UTC",
            )
            classification_source_key = _optional_str(payload.get("weekly_blocker_classification_output_key"))
            trace_source_key = _optional_str(payload.get("trace_only_empty_risk_from_output_key"))
            if classification_source_key:
                result = await self._compare_weekly_blocker_classified_people(
                    context=context,
                    classification_output_key=classification_source_key,
                    user_name=_optional_str(payload.get("user_name")),
                    department=department,
                    date_range=date_range,
                    trace_weeks=max(1, min(_safe_int(payload.get("trace_weeks"), 1), 4)),
                    include_historical_blockers=_optional_bool(payload.get("include_historical_blockers"), default=False),
                    limit=self._limit(payload),
                )
            elif trace_source_key:
                result = await self._compare_only_empty_risk_people(
                    context=context,
                    source_output_key=trace_source_key,
                    user_name=_optional_str(payload.get("user_name")),
                    department=department,
                    date_range=date_range,
                    limit=self._limit(payload),
                )
            else:
                result = await self._client().compare_weekly_plan_done(
                    user_name=_optional_str(payload.get("user_name")),
                    department=department,
                    last_week_start=date_range["last_week_start"],
                    last_week_end=date_range["last_week_end"],
                    this_week_start=date_range["this_week_start"],
                    this_week_end=date_range["this_week_end"],
                    limit=self._limit(payload),
                )
                result["query_date_range"] = date_range
        except Exception as exc:
            return self.build_result(
                success=False,
                error=f"compare_weekly_plan_done failed: {exc}",
                metadata={"exception_type": type(exc).__name__},
            )
        _attach_plan_tracking_context(
            result=result,
            include_final_report=trace_source_key is None
            and classification_source_key is None
            and (context is None or _asks_plan_completion_tracking(context.runtime.user_input)),
        )
        runtime_progress(
            step=f"{self.name}:计划完成证据输出",
            status="结构化配对证据",
            detail=json.dumps(_build_compare_weekly_progress_detail(result), ensure_ascii=False, default=str),
            request_id=context.runtime.request_id if context is not None else None,
            session_id=context.runtime.session_id if context is not None else None,
        )
        trace_scope = result.get("trace_scope") if isinstance(result.get("trace_scope"), dict) else {}
        return self.build_result(
            success=True,
            output=result,
            metadata={
                "last_week_plan_count": len(result["last_week_plans"]),
                "this_week_completed_count": len(result["this_week_completed"]),
                "candidate_match_count": len(result["candidate_matches"]),
                "plan_followup_count": len(result.get("plan_followups", [])),
                "trace_filtered_by_risk_and_help": bool(trace_scope.get("trace_filtered_by_risk_and_help")),
                "trace_user_count": len(trace_scope.get("traced_people", [])) if isinstance(trace_scope.get("traced_people"), list) else 0,
                "trace_skipped_user_count": len(trace_scope.get("skipped_people", [])) if isinstance(trace_scope.get("skipped_people"), list) else 0,
            },
        )

    async def _compare_only_empty_risk_people(
        self,
        *,
        context: ContextStore | None,
        source_output_key: str,
        user_name: str | None,
        department: str | None,
        date_range: dict[str, Any],
        limit: int,
    ) -> dict[str, Any]:
        if context is None:
            raise ValueError("trace_only_empty_risk_from_output_key requires runtime context")
        if source_output_key not in context.task_results:
            raise ValueError(f"upstream weekly report output not found: {source_output_key}")

        trace_scope = _build_empty_risk_trace_scope(
            context.task_results[source_output_key],
            user_name=user_name,
            department=department,
            source_output_key=source_output_key,
        )
        traced_people = trace_scope["traced_people"]
        if not traced_people:
            result = _empty_weekly_plan_done_result(
                date_range=date_range,
                trace_scope=trace_scope,
                skip_reason="所有匹配人员均填写了有效卡点/风险/求助内容，或上游周报结果中没有可追溯人员；未执行计划追溯查询。",
            )
            _attach_weekly_blocker_context(result=result, weekly_reports_output=context.task_results[source_output_key])
            return result

        results: list[dict[str, Any]] = []
        client = self._client()
        for person in traced_people:
            results.append(
                await client.compare_weekly_plan_done(
                    user_name=person["user_name"],
                    department=person.get("department"),
                    last_week_start=date_range["last_week_start"],
                    last_week_end=date_range["last_week_end"],
                    this_week_start=date_range["this_week_start"],
                    this_week_end=date_range["this_week_end"],
                    limit=limit,
                )
            )
        result = _merge_weekly_plan_done_results(results=results, date_range=date_range, trace_scope=trace_scope)
        _attach_weekly_blocker_context(result=result, weekly_reports_output=context.task_results[source_output_key])
        return result

    async def _compare_weekly_blocker_classified_people(
        self,
        *,
        context: ContextStore | None,
        classification_output_key: str,
        user_name: str | None,
        department: str | None,
        date_range: dict[str, Any],
        trace_weeks: int,
        include_historical_blockers: bool,
        limit: int,
    ) -> dict[str, Any]:
        if context is None:
            raise ValueError("weekly_blocker_classification_output_key requires runtime context")
        if classification_output_key not in context.task_results:
            raise ValueError(f"upstream weekly blocker classification output not found: {classification_output_key}")

        classification_output = context.task_results[classification_output_key]
        trace_scope = _build_classified_weekly_blocker_trace_scope(
            classification_output,
            user_name=user_name,
            department=department,
            classification_output_key=classification_output_key,
        )
        trace_windows = _build_weekly_blocker_trace_windows(date_range=date_range, trace_weeks=trace_weeks)
        date_range = {**date_range, "trace_windows": trace_windows}
        traced_people = trace_scope["traced_people"]
        if not traced_people:
            result = _empty_weekly_plan_done_result(
                date_range=date_range,
                trace_scope=trace_scope,
                skip_reason="所有匹配人员均已通过语义分类确认为当前有效员工自填卡点，或上游分类结果中没有可追溯人员；未执行计划追溯查询。",
            )
            result["trace_windows"] = trace_windows
            result["historical_blocker_candidates"] = []
            _attach_weekly_blocker_context_from_classification(
                result=result,
                classification_output=classification_output,
            )
            return result

        results: list[dict[str, Any]] = []
        client = self._client()
        for person in traced_people:
            for window in trace_windows:
                window_result = await client.compare_weekly_plan_done(
                    user_name=person["user_name"],
                    department=person.get("department"),
                    last_week_start=window["source_plan_week_start"],
                    last_week_end=window["source_plan_week_end"],
                    this_week_start=window["followup_start"],
                    this_week_end=window["followup_end"],
                    limit=limit,
                )
                _annotate_weekly_trace_window(result=window_result, trace_window=window)
                results.append(window_result)
        result = _merge_weekly_plan_done_results(results=results, date_range=date_range, trace_scope=trace_scope)
        result["trace_windows"] = trace_windows
        if include_historical_blockers:
            result["historical_blocker_candidates"] = await self._query_historical_blocker_candidates(
                traced_people=traced_people,
                date_range=date_range,
                trace_weeks=trace_weeks,
                limit=limit,
            )
        else:
            result["historical_blocker_candidates"] = []
        _attach_weekly_blocker_context_from_classification(
            result=result,
            classification_output=classification_output,
        )
        return result

    async def _query_historical_blocker_candidates(
        self,
        *,
        traced_people: list[dict[str, Any]],
        date_range: dict[str, Any],
        trace_weeks: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        target_start = _parse_date(str(date_range["this_week_start"]))
        target_end = _parse_date(str(date_range["this_week_end"]))
        historical_start = target_start - timedelta(days=7 * max(1, trace_weeks))
        historical_end = target_start - timedelta(days=1)
        client = self._client()
        candidates: list[dict[str, Any]] = []
        for person in traced_people:
            reports = await client.query_weekly_reports(
                user_name=person["user_name"],
                department=person.get("department"),
                start_date=historical_start.isoformat(),
                end_date=historical_end.isoformat(),
                item_type=None,
                limit=limit,
                record_level="reports",
                include_evidence_text=False,
            )
            done_rows = _dedupe_rows(
                [
                    *await client.query_weekly_reports(
                        user_name=person["user_name"],
                        department=person.get("department"),
                        start_date=historical_start.isoformat(),
                        end_date=target_end.isoformat(),
                        item_type="this_week_work",
                        limit=limit,
                        record_level="items",
                        include_evidence_text=True,
                    ),
                    *await client.query_weekly_reports(
                        user_name=person["user_name"],
                        department=person.get("department"),
                        start_date=historical_start.isoformat(),
                        end_date=target_end.isoformat(),
                        item_type="this_week_done",
                        limit=limit,
                        record_level="items",
                        include_evidence_text=True,
                    ),
                ]
            )
            for report in reports:
                if not isinstance(report, Mapping):
                    continue
                raw_text = _optional_str(report.get("risk_and_help"))
                if not raw_text:
                    continue
                report_date_text = _row_date(report) or _optional_str(report.get("report_date"))
                if not report_date_text:
                    continue
                report_date = _parse_date(report_date_text)
                followup_items = [
                    _compact_historical_done_item(row)
                    for row in done_rows
                    if isinstance(row, Mapping)
                    and (done_date_text := (_row_date(row) or _optional_str(row.get("report_date"))))
                    and report_date < _parse_date(done_date_text) <= target_end
                ][:8]
                candidates.append(
                    {
                        "candidate_id": f"historical_blocker_{len(candidates) + 1}",
                        "user_name": person["user_name"],
                        "department": person.get("department"),
                        "report_date": report_date_text,
                        "raw_risk_and_help": _clip_text(raw_text, 1200),
                        "followup_done_items": followup_items,
                    }
                )
        return candidates[:limit]


class MonthlyDepartmentAnalysisTool(_MySQLBusinessTool):
    name: str = Field(default="monthly_department_analysis")
    description: str = Field(default="查询部门月度计划、周报事项和自评事项，用于部门月度计划完成情况或自评一致性分析。")

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "department": {"type": "string"},
                "month": {"type": "string", "description": "YYYY-MM"},
                "limit": {"type": ["integer", "null"], "default": 1000},
            },
            "required": ["department", "month"],
            "additionalProperties": False,
        }

    def get_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "dept_plans": {"type": "array"},
                "weekly_items": {"type": "array"},
                "self_eval_items": {"type": "array"},
                "opl_issues": {"type": "array"},
                "opl_summary": {"type": "object"},
            },
            "required": ["dept_plans", "weekly_items", "self_eval_items", "opl_issues", "opl_summary"],
            "additionalProperties": True,
        }

    async def _arun(self, payload: dict[str, Any], context: ContextStore | None = None) -> ToolResult:
        try:
            result = await self._client().monthly_department_analysis(
                department=_required_str(payload.get("department"), "department"),
                month=_required_month(payload.get("month")),
                limit=self._limit(payload, default=1000),
            )
        except Exception as exc:
            return self.build_result(
                success=False,
                error=f"monthly_department_analysis failed: {exc}",
                metadata={"exception_type": type(exc).__name__},
            )
        return self.build_result(
            success=True,
            output=result,
            metadata={
                "dept_plan_count": len(result["dept_plans"]),
                "weekly_item_count": len(result["weekly_items"]),
                "self_eval_item_count": len(result["self_eval_items"]),
                "opl_issue_count": len(result.get("opl_issues", [])),
            },
        )


class QueryOplIssuesTool(_MySQLBusinessTool):
    name: str = Field(default="query_opl_issues")
    description: str = Field(default="查询 OPL 问题清单跟踪表，用于按月份、部门、负责人、状态、优先级和关键词分析问题闭环情况。")
    tags: list[str] = Field(default_factory=lambda: ["mysql", "business", "opl_issue"])

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "month": {"type": ["string", "null"], "description": "可选，YYYY-MM；提供后会自动转为当月日期范围。"},
                "start_date": {"type": ["string", "null"], "description": "YYYY-MM-DD；month 为空时使用。"},
                "end_date": {"type": ["string", "null"], "description": "YYYY-MM-DD；month 为空时使用。"},
                "department": {"type": ["string", "null"]},
                "status": {"type": ["string", "null"], "description": "可传具体状态，也可传 open/closed 语义状态。"},
                "owner_user": {"type": ["string", "null"], "description": "问题负责人或跟踪人。"},
                "priority": {"type": ["string", "null"]},
                "doc_id": {"type": ["string", "null"]},
                "keyword": {"type": ["string", "null"], "description": "匹配问题描述、解决进展、机构总成和备注。"},
                "include_open_before_start": {
                    "type": ["boolean", "null"],
                    "default": True,
                    "description": "按月份分析时是否纳入本月前提出但仍未闭环的问题。",
                },
                "limit": {"type": ["integer", "null"], "default": 1000},
            },
            "additionalProperties": False,
        }

    def get_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "items": {"type": "array"},
                "summary": {"type": "object"},
                "count": {"type": "integer"},
            },
            "required": ["items", "summary", "count"],
            "additionalProperties": True,
        }

    async def _arun(self, payload: dict[str, Any], context: ContextStore | None = None) -> ToolResult:
        try:
            month = _optional_month(payload.get("month"))
            if month:
                start_date, end_date = _month_range(month)
            else:
                start_date = _optional_date(payload.get("start_date"))
                end_date = _optional_date(payload.get("end_date"))
            department = _optional_str(payload.get("department")) or _infer_department_from_runtime_context(context)
            items = await self._client().query_opl_issues(
                department=department,
                start_date=start_date,
                end_date=end_date,
                status=_optional_str(payload.get("status")),
                owner_user=_optional_str(payload.get("owner_user")),
                priority=_optional_str(payload.get("priority")),
                doc_id=_optional_str(payload.get("doc_id")),
                keyword=_optional_str(payload.get("keyword")),
                owner_names=[],
                include_open_before_start=_optional_bool(payload.get("include_open_before_start"), default=True),
                limit=self._limit(payload, default=1000),
            )
        except Exception as exc:
            return self.build_result(
                success=False,
                error=f"query_opl_issues failed: {exc}",
                metadata={"exception_type": type(exc).__name__},
            )
        summary = _summarize_opl_issues(items)
        return self.build_result(
            success=True,
            output={"items": items, "summary": summary, "count": len(items)},
            metadata={"count": len(items), **summary},
        )


class CompareDeptPlanCompletionTool(_MySQLBusinessTool):
    name: str = Field(default="compare_dept_plan_completion")
    description: str = Field(default="核对三七计划书或部门月度计划是否完成：查询 dept_plan_items，并直连负责人本人月度考核记录、周报证据和 OPL 问题闭环证据，完成状态由后续 LLM 判断。")
    tags: list[str] = Field(default_factory=lambda: ["mysql", "business", "dept_plan", "weekly_report", "self_eval", "opl_issue"])

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "month": {"type": "string", "description": "计划归属月份，格式 YYYY-MM。"},
                "department": {"type": ["string", "null"], "description": "可选部门；为空时查询该月所有部门。"},
                "doc_id": {"type": ["string", "null"], "description": "可选计划书 doc_id；用于只核对某一份计划书。"},
                "include_weekly": {"type": ["boolean", "null"], "default": True},
                "include_self_eval": {"type": ["boolean", "null"], "default": True},
                "include_opl": {
                    "type": ["boolean", "null"],
                    "default": True,
                    "description": "是否纳入 OPL 问题清单；默认纳入，用于识别相关问题是否未闭环和解决进展。",
                },
                "followup_days": {
                    "type": ["integer", "null"],
                    "default": 7,
                    "description": "月末后继续查多少天周报完成记录；普通月度核对默认只保留短宽限窗口，用户明确要求后续追踪时可放宽。",
                },
                "limit": {"type": ["integer", "null"], "default": 1000},
            },
            "required": ["month"],
            "additionalProperties": False,
        }

    def get_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "dept_plans": {"type": "array"},
                "weekly_items": {"type": "array"},
                "owner_weekly_reports_pool": {"type": "array"},
                "collaboration_weekly_reports_pool": {"type": "array"},
                "owner_self_eval_items": {"type": "array"},
                "self_eval_items": {"type": "array"},
                "opl_issues": {"type": "array"},
                "opl_issue_pool": {"type": "array"},
                "dept_plan_followups": {"type": "array"},
                "pairing_summary": {"type": "object"},
                "query_scope": {"type": "object"},
                "dept_plan_completion_context_text": {
                    "type": "string",
                    "description": "面向 text_generate_tool 的三七计划完成核对压缩上下文。",
                },
            },
            "required": ["dept_plans", "weekly_items", "self_eval_items", "dept_plan_followups", "pairing_summary"],
            "additionalProperties": True,
        }

    async def _arun(self, payload: dict[str, Any], context: ContextStore | None = None) -> ToolResult:
        try:
            department = _optional_str(payload.get("department")) or _infer_department_from_runtime_context(context)
            result = await self._client().compare_dept_plan_completion(
                month=_required_month(payload.get("month")),
                department=department,
                doc_id=_optional_str(payload.get("doc_id")),
                include_weekly=_optional_bool(payload.get("include_weekly"), default=True),
                include_self_eval=_optional_bool(payload.get("include_self_eval"), default=True),
                followup_days=self._followup_days(payload, context=context),
                limit=self._limit(payload, default=1000),
                include_opl=_optional_bool(payload.get("include_opl"), default=True),
            )
        except Exception as exc:
            return self.build_result(
                success=False,
                error=f"compare_dept_plan_completion failed: {exc}",
                metadata={"exception_type": type(exc).__name__},
            )
        runtime_progress(
            step=f"{self.name}:计划完成证据输出",
            status="结构化三七计划证据",
            detail=json.dumps(_build_dept_plan_progress_detail(result), ensure_ascii=False, default=str),
            request_id=context.runtime.request_id if context is not None else None,
            session_id=context.runtime.session_id if context is not None else None,
        )
        return self.build_result(
            success=True,
            output=result,
            metadata={
                "dept_plan_count": len(result.get("dept_plans", [])),
                "weekly_item_count": len(result.get("weekly_items", [])),
                "owner_weekly_report_count": len(result.get("owner_weekly_reports_pool", [])),
                "collaboration_weekly_report_count": len(result.get("collaboration_weekly_reports_pool", [])),
                "owner_self_eval_item_count": len(result.get("owner_self_eval_items", [])),
                "self_eval_item_count": len(result.get("self_eval_items", [])),
                "opl_issue_count": len(result.get("opl_issues", [])),
                "opl_issue_pool_count": len(result.get("opl_issue_pool", [])),
                "opl_issue_candidate_count": result.get("pairing_summary", {}).get("opl_issue_candidate_count", 0)
                if isinstance(result.get("pairing_summary"), Mapping)
                else 0,
                "dept_plan_followup_count": len(result.get("dept_plan_followups", [])),
                "context_chars": len(str(result.get("dept_plan_completion_context_text") or "")),
            },
        )

    def _followup_days(self, payload: Mapping[str, Any], *, context: ContextStore | None = None) -> int:
        requested = _optional_non_negative_int(
            payload.get("followup_days"),
            default=_dept_plan_default_followup_days(),
        )
        user_input = context.runtime.user_input if context is not None else None
        return _normalize_dept_plan_followup_days(
            user_input=user_input,
            requested_days=requested,
        )


def _env_name(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required to use MySQL business tools")
    return value


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_non_negative_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


def _expanded_query_limit(limit: int) -> int:
    multiplier = _env_int("MYSQL_DEPT_PLAN_EVIDENCE_LIMIT_MULTIPLIER", 3)
    max_limit = _env_int("MYSQL_DEPT_PLAN_EVIDENCE_MAX_LIMIT", _env_int("MYSQL_BUSINESS_MAX_LIMIT", 2000))
    return max(1, min(max(1, limit) * max(1, multiplier), max_limit))


def _safe_int(value: Any, default: int) -> int:
    if value in (None, ""):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _optional_non_negative_int(value: Any, *, default: int) -> int:
    if value in (None, ""):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _dept_plan_default_followup_days() -> int:
    return _env_non_negative_int("MYSQL_DEPT_PLAN_DEFAULT_FOLLOWUP_DAYS", 7)


def _normalize_dept_plan_followup_days(*, user_input: str | None, requested_days: int) -> int:
    max_days = _env_int("MYSQL_DEPT_PLAN_MAX_FOLLOWUP_DAYS", 62)
    requested_days = max(0, min(requested_days, max_days))
    normalized = str(user_input or "").strip()
    if not normalized:
        return requested_days

    strict_month_tokens = (
        "只看本月",
        "仅看本月",
        "限本月",
        "只统计本月",
        "只查本月",
        "不要查后续",
        "不查后续",
        "不看后续",
        "不要包含后续",
        "不要六月",
        "不看六月",
        "排除六月",
        "不要6月",
        "不看6月",
        "排除6月",
    )
    if any(token in normalized for token in strict_month_tokens):
        return 0
    if re.search(r"(?:只|仅|单独|限)\D{0,6}(?:本月|\d{1,2}月|[一二三四五六七八九十]+月)", normalized):
        return 0

    explicit_days = _explicit_dept_plan_followup_days(normalized)
    if explicit_days is not None:
        return max(0, min(explicit_days, max_days))

    long_followup_tokens = (
        "截至现在",
        "截止现在",
        "到现在",
        "至今",
        "目前为止",
        "现在为止",
        "截至今天",
        "截止今天",
        "截至今日",
        "截止今日",
        "后续完成",
        "后续周报",
        "后续进展",
        "追踪后续",
        "继续追踪",
        "后来完成",
    )
    if any(token in normalized for token in long_followup_tokens):
        return max(requested_days, min(31, max_days))

    # Older planner prompts used 31 days for every monthly dept-plan query. For
    # ordinary "某月三七计划是否完成" questions, cap that legacy value to a short
    # grace window so a whole extra month does not dominate candidate matching.
    return min(requested_days, _dept_plan_default_followup_days(), max_days)


def _explicit_dept_plan_followup_days(text: str) -> int | None:
    if any(token in text for token in ("后续一个月", "往后一个月", "之后一个月", "后面一个月", "接下来一个月")):
        return 31
    if any(token in text for token in ("后续两周", "往后两周", "之后两周", "后面两周", "接下来两周")):
        return 14
    if any(token in text for token in ("后续一周", "往后一周", "之后一周", "后面一周", "接下来一周")):
        return 7
    match = re.search(r"(?:后续|往后|之后|后面|接下来)\s*(?P<days>\d{1,3})\s*天", text)
    if match:
        return int(match.group("days"))
    return None


def _validate_identifier(value: str) -> None:
    if not _IDENTIFIER_RE.match(value):
        raise ValueError(f"unsafe MySQL identifier: {value}")


def _normalize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (datetime, date)):
            normalized[key] = value.isoformat()
        elif key == "evidence_text":
            normalized[key] = _clip_text(value, _env_int("MYSQL_BUSINESS_EVIDENCE_TEXT_MAX_CHARS", 800))
        elif key in {"this_week_raw", "next_week_raw"}:
            normalized[key] = _clip_text(value, _env_int("MYSQL_BUSINESS_RAW_TEXT_MAX_CHARS", 12000))
        else:
            normalized[key] = value
    return normalized


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _infer_department_from_runtime_context(context: ContextStore | None) -> str | None:
    if context is None:
        return None
    return _infer_department_from_text(context.runtime.user_input)


def _infer_department_from_text(text: str | None) -> str | None:
    if not text:
        return None
    full_path_match = _DEPARTMENT_PATH_RE.search(text)
    if full_path_match:
        return full_path_match.group(0)

    aliases = _department_aliases()
    matches = [alias for alias in aliases if alias and alias in text]
    if not matches:
        return None
    return max(matches, key=len)


def _department_aliases() -> tuple[str, ...]:
    custom_aliases = [
        item.strip()
        for item in os.getenv("MYSQL_DEPARTMENT_ALIASES", "").split(",")
        if item.strip()
    ]
    aliases = [*custom_aliases, *_DEFAULT_DEPARTMENT_ALIASES]
    return tuple(dict.fromkeys(aliases))


def _department_canonical_aliases() -> dict[str, tuple[str, ...]]:
    aliases = dict(_DEFAULT_DEPARTMENT_CANONICAL_ALIASES)
    raw = os.getenv("MYSQL_DEPARTMENT_CANONICAL_ALIASES", "").strip()
    if raw:
        for group in raw.split(";"):
            parts = [part.strip() for part in group.split("=") if part.strip()]
            if len(parts) != 2:
                continue
            canonical, values = parts
            alias_values = tuple(item.strip() for item in values.split("|") if item.strip())
            if alias_values:
                aliases[canonical] = alias_values
    return aliases


def _canonical_department(value: Any) -> str | None:
    text = _optional_str(value)
    if text is None:
        return None
    tail = re.split(r"[/\\]", text)[-1]
    for canonical, aliases in _department_canonical_aliases().items():
        if any(text == alias or tail == alias or text.endswith(f"/{alias}") or text.endswith(f"\\{alias}") for alias in aliases):
            return canonical
    return tail if _is_short_department_name(tail) else text


def _department_filter_sql(*, column: str, department: str) -> tuple[str, list[Any]]:
    aliases = _department_query_aliases(department)
    clauses: list[str] = []
    params: list[Any] = []
    for alias in aliases:
        if _is_short_department_name(alias):
            clauses.append(f"({column} = %s OR {column} LIKE %s OR {column} LIKE %s)")
            params.extend([alias, f"%/{alias}", f"%\\{alias}"])
        else:
            clauses.append(f"{column} = %s")
            params.append(alias)
    return f"({' OR '.join(clauses)})", params


def _department_query_aliases(department: str) -> list[str]:
    aliases = [department]
    requested_canonical = _canonical_department(department)
    if requested_canonical is not None:
        for canonical, values in _department_canonical_aliases().items():
            if canonical == requested_canonical:
                aliases.extend(values)
                break
    return list(dict.fromkeys(alias for alias in aliases if alias))


def _departments_match(row_department: Any, requested_department: Any) -> bool:
    row_text = _optional_str(row_department)
    requested_text = _optional_str(requested_department)
    if requested_text is None:
        return True
    if row_text is None:
        return False
    if row_text == requested_text:
        return True
    row_canonical = _canonical_department(row_text)
    requested_canonical = _canonical_department(requested_text)
    if row_canonical is not None and row_canonical == requested_canonical:
        return True
    return _is_short_department_name(requested_text) and row_text.endswith(f"/{requested_text}")


def _departments_compatible(left_department: Any, right_department: Any) -> bool:
    left_text = _optional_str(left_department)
    right_text = _optional_str(right_department)
    if left_text is None or right_text is None:
        return True
    if left_text == right_text:
        return True
    left_canonical = _canonical_department(left_text)
    right_canonical = _canonical_department(right_text)
    if left_canonical is not None and left_canonical == right_canonical:
        return True
    if _is_short_department_name(left_text) and right_text.endswith(f"/{left_text}"):
        return True
    return _is_short_department_name(right_text) and left_text.endswith(f"/{right_text}")


def _same_user(left_user: str | None, right_user: str | None) -> bool:
    return bool(left_user and right_user and _canonical_person_name(left_user) == _canonical_person_name(right_user))


def _person_aliases() -> dict[str, tuple[str, ...]]:
    aliases = {canonical: tuple(values) for canonical, values in _DEFAULT_PERSON_ALIASES.items()}
    raw_json = os.getenv("MYSQL_PERSON_ALIASES_JSON")
    if raw_json:
        try:
            loaded = json.loads(raw_json)
        except json.JSONDecodeError:
            loaded = None
        if isinstance(loaded, Mapping):
            for raw_canonical, raw_aliases in loaded.items():
                canonical = str(raw_canonical or "").strip()
                if not canonical:
                    continue
                values: list[str] = [canonical]
                if isinstance(raw_aliases, str):
                    values.extend(part.strip() for part in re.split(r"[、,，;/；|+＋\s]+", raw_aliases) if part.strip())
                elif isinstance(raw_aliases, list):
                    values.extend(str(part).strip() for part in raw_aliases if str(part or "").strip())
                existing = list(aliases.get(canonical, ()))
                aliases[canonical] = tuple(dict.fromkeys([*existing, *values]))

    raw_pairs = os.getenv("MYSQL_PERSON_ALIAS_PAIRS")
    if raw_pairs:
        for pair in re.split(r"[;,；\n]+", raw_pairs):
            if not pair.strip():
                continue
            if "=" in pair:
                raw_alias, raw_canonical = pair.split("=", 1)
            elif ":" in pair:
                raw_alias, raw_canonical = pair.split(":", 1)
            else:
                continue
            alias = raw_alias.strip()
            canonical = raw_canonical.strip()
            if not alias or not canonical:
                continue
            aliases[canonical] = tuple(dict.fromkeys([*aliases.get(canonical, ()), canonical, alias]))
    return aliases


def _canonical_person_name(value: str) -> str:
    text = str(value or "").strip()
    for canonical, aliases in _person_aliases().items():
        if text == canonical or text in aliases:
            return canonical
    return text


def _is_short_department_name(value: str) -> bool:
    return "/" not in value and "\\" not in value


def _classify_risk_and_help(value: Any) -> str:
    text = _optional_str(value)
    if text is None:
        return "empty"
    compact = _compact_risk_and_help_statement(text)
    if _looks_like_no_blocker_statement(compact) and not _has_actionable_blocker_signal(compact):
        return "no_blocker_statement"
    return "actionable_blocker"


def _risk_and_help_has_actionable_blocker(value: Any) -> bool:
    return _classify_risk_and_help(value) == "actionable_blocker"


def _compact_risk_and_help_statement(value: str) -> str:
    return re.sub(r"[\s,，.。!！?？;；:：、/\\|·`~…（）()【】\[\]{}《》<>\"'“”‘’\-—_]+", "", str(value or "").lower())


def _looks_like_no_blocker_statement(compact_text: str) -> bool:
    if not compact_text:
        return False
    if compact_text in _NO_BLOCKER_COMPACT_EXACT:
        return True
    return any(marker in compact_text for marker in _NO_BLOCKER_COMPACT_MARKERS)


def _has_actionable_blocker_signal(compact_text: str) -> bool:
    if not compact_text:
        return False
    if any(_has_non_negated_marker(compact_text, marker) for marker in _ACTIONABLE_BLOCKER_COMPACT_MARKERS):
        return True
    return any(pattern.search(compact_text) for pattern in _ACTIONABLE_BLOCKER_REGEXES)


def _has_non_negated_marker(compact_text: str, marker: str) -> bool:
    start = 0
    while True:
        index = compact_text.find(marker, start)
        if index < 0:
            return False
        prefix = compact_text[max(0, index - 4) : index]
        if not any(prefix.endswith(negative) for negative in _NEGATED_ACTION_PREFIXES):
            return True
        start = index + len(marker)


def _optional_bool(value: Any, *, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _optional_record_level(value: Any) -> str | None:
    text = _optional_str(value)
    if text is None:
        return None
    if text not in {"items", "reports"}:
        raise ValueError(f"unsupported record_level: {text}")
    return text


def _looks_like_weekly_blocker_query(text: str | None) -> bool:
    if not text:
        return False
    return any(keyword in text for keyword in ("卡点", "风险", "求助", "阻塞", "卡在", "问题"))


def _weekly_report_items_from_output(output: Any) -> list[dict[str, Any]]:
    raw_items = output.get("items") if isinstance(output, Mapping) else output
    if not isinstance(raw_items, list):
        return []
    return [dict(item) for item in raw_items if isinstance(item, Mapping)]


def _build_empty_risk_trace_scope(
    weekly_reports_output: Any,
    *,
    user_name: str | None,
    department: str | None,
    source_output_key: str,
) -> dict[str, Any]:
    items = _weekly_report_items_from_output(weekly_reports_output)
    grouped: dict[tuple[str, str | None], dict[str, Any]] = {}
    for item in items:
        row_user = _optional_str(item.get("user_name"))
        row_department = _optional_str(item.get("department"))
        if not row_user:
            continue
        if user_name and row_user != user_name:
            continue
        if department and not _departments_match(row_department, department):
            continue

        key = (row_user, row_department)
        group = grouped.setdefault(
            key,
            {
                "user_name": row_user,
                "department": row_department,
                "row_count": 0,
                "has_actionable_risk_and_help": False,
                "risk_and_help_classification": "empty",
                "risk_and_help_preview": None,
            },
        )
        group["row_count"] += 1
        classification = _classify_risk_and_help(item.get("risk_and_help"))
        if classification == "actionable_blocker":
            group["has_actionable_risk_and_help"] = True
            group["risk_and_help_classification"] = classification
            if group["risk_and_help_preview"] is None:
                group["risk_and_help_preview"] = str(item.get("risk_and_help")).strip()[:200]
        elif group["risk_and_help_classification"] == "empty":
            group["risk_and_help_classification"] = classification

    people = list(grouped.values())
    traced_people = [
        {
            "user_name": str(item["user_name"]),
            "department": item.get("department"),
            "source_row_count": item["row_count"],
            "reason": item["risk_and_help_classification"],
        }
        for item in people
        if not item["has_actionable_risk_and_help"]
    ]
    skipped_people = [
        {
            "user_name": str(item["user_name"]),
            "department": item.get("department"),
            "source_row_count": item["row_count"],
            "reason": "actionable_risk_and_help",
            "risk_and_help_preview": item.get("risk_and_help_preview"),
        }
        for item in people
        if item["has_actionable_risk_and_help"]
    ]
    return {
        "trace_filtered_by_risk_and_help": True,
        "source_output_key": source_output_key,
        "source_item_count": len(items),
        "matched_people_count": len(people),
        "traced_people": traced_people,
        "skipped_people": skipped_people,
        "filter_rule": "仅追溯目标周员工自填卡点为空，或仅填写无卡点/无风险/无求助表述的人员。",
    }


def _weekly_blocker_classification_items_from_output(output: Any) -> list[dict[str, Any]]:
    raw_items = output.get("items") if isinstance(output, Mapping) else output
    if not isinstance(raw_items, list):
        return []
    return [dict(item) for item in raw_items if isinstance(item, Mapping)]


def _build_classified_weekly_blocker_trace_scope(
    classification_output: Any,
    *,
    user_name: str | None,
    department: str | None,
    classification_output_key: str,
) -> dict[str, Any]:
    items = _weekly_blocker_classification_items_from_output(classification_output)
    matched_people: list[dict[str, Any]] = []
    traced_people: list[dict[str, Any]] = []
    skipped_people: list[dict[str, Any]] = []
    for item in items:
        row_user = _optional_str(item.get("user_name"))
        row_department = _optional_str(item.get("department"))
        if not row_user:
            continue
        if user_name and row_user != user_name:
            continue
        if department and not _departments_match(row_department, department):
            continue
        matched_people.append(item)
        classification = _optional_str(item.get("classification")) or "ambiguous"
        needs_trace = _optional_bool(item.get("needs_trace"), default=True)
        has_effective = _optional_bool(item.get("has_effective_current_blocker"), default=False)
        effective_text = _optional_str(item.get("effective_blocker_text"))
        person = {
            "user_name": row_user,
            "department": row_department,
            "source_row_count": _safe_int(item.get("source_row_count"), 1),
            "classification": classification,
            "reason": _optional_str(item.get("reason")) or classification,
        }
        if needs_trace:
            traced_people.append(person)
        else:
            skipped_people.append(
                {
                    **person,
                    "reason": "effective_current_blocker_by_llm",
                    "has_effective_current_blocker": has_effective,
                    "effective_blocker_preview": (effective_text or _optional_str(item.get("raw_risk_and_help")) or "")[:200],
                }
            )
    return {
        "trace_filtered_by_risk_and_help": True,
        "trace_filtered_by_weekly_blocker_classification": True,
        "classification_output_key": classification_output_key,
        "source_output_key": classification_output.get("source_output_key") if isinstance(classification_output, Mapping) else None,
        "source_item_count": len(items),
        "matched_people_count": len(matched_people),
        "traced_people": traced_people,
        "skipped_people": skipped_people,
        "filter_rule": "按 classify_weekly_blockers 的语义分类结果，仅追溯 needs_trace=true 的人员；当前有效员工自填卡点直接保留。",
    }


def _build_weekly_blocker_trace_windows(*, date_range: dict[str, Any], trace_weeks: int) -> list[dict[str, Any]]:
    target_start = _parse_date(str(date_range["this_week_start"]))
    target_end = _parse_date(str(date_range["this_week_end"]))
    windows: list[dict[str, Any]] = []
    for window_index in range(1, max(1, trace_weeks) + 1):
        source_start = target_start - timedelta(days=7 * window_index)
        source_end = target_start - timedelta(days=7 * (window_index - 1) + 1)
        followup_start = source_end + timedelta(days=1)
        windows.append(
            {
                "window_index": window_index,
                "source_plan_week_start": source_start.isoformat(),
                "source_plan_week_end": source_end.isoformat(),
                "followup_start": followup_start.isoformat(),
                "followup_end": target_end.isoformat(),
            }
        )
    return windows


def _annotate_weekly_trace_window(*, result: dict[str, Any], trace_window: dict[str, Any]) -> None:
    for key in ("last_week_plans", "this_week_completed", "candidate_matches", "weekly_pairs", "plan_followups"):
        rows = result.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict):
                row["trace_window"] = dict(trace_window)


def _attach_weekly_blocker_context_from_classification(
    *,
    result: dict[str, Any],
    classification_output: Any,
) -> None:
    people = _build_weekly_blocker_people_from_classification(
        classification_output=classification_output,
        plan_followups=result.get("plan_followups", []),
        trace_scope=result.get("trace_scope", {}),
    )
    result["weekly_blocker_people"] = people
    result["direct_blocker_people"] = [
        {
            "user_name": person.get("user_name"),
            "department": person.get("department"),
            "blocker_text": person.get("risk_and_help"),
            "classification": person.get("classification"),
        }
        for person in people
        if person.get("risk_and_help")
    ]
    result["weekly_blocker_context_text"] = _format_weekly_blocker_context(
        people=people,
        date_range=result.get("query_date_range", {}),
        pairing_summary=result.get("pairing_summary", {}),
    )


def _build_weekly_blocker_people_from_classification(
    *,
    classification_output: Any,
    plan_followups: Any,
    trace_scope: Any,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str | None], dict[str, Any]] = {}
    for item in _weekly_blocker_classification_items_from_output(classification_output):
        row_user = _optional_str(item.get("user_name"))
        row_department = _optional_str(item.get("department"))
        if not row_user:
            continue
        key = (row_user, row_department)
        effective_text = _optional_str(item.get("effective_blocker_text"))
        has_effective = _optional_bool(item.get("has_effective_current_blocker"), default=False)
        grouped[key] = {
            "user_name": row_user,
            "department": row_department,
            "source": "employee_risk_and_help" if has_effective and effective_text else "inferred_from_plan_followups",
            "risk_and_help": effective_text if has_effective else None,
            "classification": _optional_str(item.get("classification")),
            "source_row_count": _safe_int(item.get("source_row_count"), 1),
            "inferred_followups": [],
        }

    followups_by_person: dict[tuple[str, str | None], list[dict[str, Any]]] = {}
    if isinstance(plan_followups, list):
        for followup in plan_followups:
            if not isinstance(followup, Mapping):
                continue
            row_user = _optional_str(followup.get("user_name"))
            row_department = _optional_str(followup.get("department"))
            if not row_user:
                continue
            followups_by_person.setdefault((row_user, row_department), []).append(_compact_plan_followup(followup))

    traced_people = trace_scope.get("traced_people", []) if isinstance(trace_scope, Mapping) else []
    for item in traced_people:
        if not isinstance(item, Mapping):
            continue
        row_user = _optional_str(item.get("user_name"))
        row_department = _optional_str(item.get("department"))
        if not row_user:
            continue
        grouped.setdefault(
            (row_user, row_department),
            {
                "user_name": row_user,
                "department": row_department,
                "source": "inferred_from_plan_followups",
                "risk_and_help": None,
                "classification": _optional_str(item.get("classification")),
                "source_row_count": _safe_int(item.get("source_row_count"), 1),
                "inferred_followups": [],
            },
        )

    for key, followups in followups_by_person.items():
        grouped.setdefault(
            key,
            {
                "user_name": key[0],
                "department": key[1],
                "source": "inferred_from_plan_followups",
                "risk_and_help": None,
                "classification": None,
                "source_row_count": 0,
                "inferred_followups": [],
            },
        )
        grouped[key]["inferred_followups"] = followups[:8]
    return list(grouped.values())


def _compact_historical_done_item(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "report_date": _row_date(row) or row.get("report_date"),
        "item_type": row.get("item_type"),
        "done_text": _clip_text(row.get("item_text"), 320),
        "evidence_text": _clip_text(row.get("evidence_text"), 400),
    }


def _attach_plan_tracking_context(*, result: dict[str, Any], include_final_report: bool = True) -> None:
    pairing_summary = result.get("pairing_summary")
    if include_final_report and isinstance(pairing_summary, dict):
        pairing_summary["judgement_owner"] = "deterministic_mysql_report"
        pairing_summary["judgement_instruction"] = "MySQL 工具基于计划内容、下一期完成项和候选匹配项生成保守的确定性完成情况报告。"
    result["plan_tracking_context_text"] = _format_plan_tracking_context(
        plan_followups=result.get("plan_followups", []),
        date_range=result.get("query_date_range", {}),
        pairing_summary=result.get("pairing_summary", {}),
    )
    if include_final_report:
        result["final_report"] = _format_plan_tracking_final_report(
            plan_followups=result.get("plan_followups", []),
            date_range=result.get("query_date_range", {}),
            pairing_summary=result.get("pairing_summary", {}),
        )
        result["final_report_type"] = "weekly_plan_completion"


def _build_compare_weekly_progress_detail(result: Mapping[str, Any]) -> dict[str, Any]:
    trace_scope = result.get("trace_scope") if isinstance(result.get("trace_scope"), Mapping) else {}
    traced_people = trace_scope.get("traced_people", []) if isinstance(trace_scope, Mapping) else []
    skipped_people = trace_scope.get("skipped_people", []) if isinstance(trace_scope, Mapping) else []
    return {
        "query_date_range": result.get("query_date_range", {}),
        "last_week_plan_count": len(result.get("last_week_plans", [])) if isinstance(result.get("last_week_plans"), list) else 0,
        "this_week_completed_count": len(result.get("this_week_completed", [])) if isinstance(result.get("this_week_completed"), list) else 0,
        "candidate_match_count": len(result.get("candidate_matches", [])) if isinstance(result.get("candidate_matches"), list) else 0,
        "weekly_pair_count": len(result.get("weekly_pairs", [])) if isinstance(result.get("weekly_pairs"), list) else 0,
        "plan_followup_count": len(result.get("plan_followups", [])) if isinstance(result.get("plan_followups"), list) else 0,
        "trace_filtered_by_risk_and_help": bool(trace_scope.get("trace_filtered_by_risk_and_help")) if isinstance(trace_scope, Mapping) else False,
        "trace_user_count": len(traced_people) if isinstance(traced_people, list) else 0,
        "trace_skipped_user_count": len(skipped_people) if isinstance(skipped_people, list) else 0,
        "weekly_blocker_context_chars": len(str(result.get("weekly_blocker_context_text") or "")),
        "plan_tracking_context_chars": len(str(result.get("plan_tracking_context_text") or "")),
    }


def _format_plan_tracking_context(
    *,
    plan_followups: Any,
    date_range: Any,
    pairing_summary: Any,
) -> str:
    if isinstance(date_range, Mapping):
        plan_start = date_range.get("last_week_start", "")
        plan_end = date_range.get("last_week_end", "")
        done_start = date_range.get("this_week_start", "")
        done_end = date_range.get("this_week_end", "")
    else:
        plan_start = plan_end = done_start = done_end = ""
    if isinstance(pairing_summary, Mapping):
        total_plans = pairing_summary.get("total_plans", 0)
        plans_without_followup = pairing_summary.get("plans_without_followup_records", 0)
        weekly_pair_count = pairing_summary.get("weekly_pair_count", 0)
    else:
        total_plans = plans_without_followup = weekly_pair_count = 0

    lines = [
        "计划追踪压缩上下文",
        f"- 计划日期范围: {plan_start} 至 {plan_end}",
        f"- 完成记录日期范围: {done_start} 至 {done_end}",
        f"- 统计: 计划 {total_plans} 条，周配对 {weekly_pair_count} 组，无后续完成记录 {plans_without_followup} 条。",
        "- 判断规则: 请只依据每条计划的计划内容、后续完成项和候选完成项，判断已完成、部分完成、未完成或证据不足。",
        "",
    ]
    if not isinstance(plan_followups, list) or not plan_followups:
        lines.append("无可评估计划。")
        return "\n".join(lines).strip()

    max_items = _env_int("MYSQL_PLAN_TRACKING_CONTEXT_MAX_ITEMS", 160)
    for index, followup in enumerate(plan_followups[:max_items], start=1):
        if not isinstance(followup, Mapping):
            continue
        lines.append(f"计划 {index}:")
        lines.append(f"  人员: {followup.get('user_name') or '未知人员'} / {followup.get('department') or '未填写部门'}")
        lines.append(f"  计划周: {followup.get('plan_date') or '未知'}")
        lines.append(f"  后续完成周: {followup.get('completion_date') or '未找到'}")
        lines.append(f"  计划内容: {_clip_text(followup.get('plan_text'), 260)}")
        done_items = followup.get("done_items", [])
        if isinstance(done_items, list) and done_items:
            for done_index, done in enumerate(done_items[:2], start=1):
                if not isinstance(done, Mapping):
                    continue
                lines.append(f"  后续完成 {done_index}: {_clip_text(done.get('done_text'), 180)}")
        else:
            lines.append("  后续完成: 未找到下一期完成记录")

        candidates = followup.get("candidate_done_items", [])
        if isinstance(candidates, list) and candidates:
            for candidate_index, candidate in enumerate(candidates[:2], start=1):
                if not isinstance(candidate, Mapping):
                    continue
                keywords = candidate.get("overlap_keywords", [])
                keyword_text = "、".join(str(item) for item in keywords[:8]) if isinstance(keywords, list) else ""
                suffix = f"；重合关键词: {keyword_text}" if keyword_text else ""
                lines.append(f"  候选完成 {candidate_index}: {_clip_text(candidate.get('done_text'), 160)}{suffix}")
        lines.append("")
    if isinstance(plan_followups, list) and len(plan_followups) > max_items:
        lines.append(f"还有 {len(plan_followups) - max_items} 条计划未展开，请提示用户缩小人员、部门或月份范围。")
    return "\n".join(lines).strip()


def _format_plan_tracking_final_report(
    *,
    plan_followups: Any,
    date_range: Any,
    pairing_summary: Any,
) -> str:
    followups = [item for item in plan_followups if isinstance(item, Mapping)] if isinstance(plan_followups, list) else []
    if isinstance(date_range, Mapping):
        plan_start = date_range.get("last_week_start", "")
        plan_end = date_range.get("last_week_end", "")
        done_start = date_range.get("this_week_start", "")
        done_end = date_range.get("this_week_end", "")
    else:
        plan_start = plan_end = done_start = done_end = ""

    judged = [_judge_plan_followup(item) for item in followups]
    status_counts: dict[str, int] = {"已完成": 0, "部分完成": 0, "未完成": 0, "证据不足": 0}
    for item in judged:
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1

    if isinstance(pairing_summary, Mapping):
        weekly_pair_count = pairing_summary.get("weekly_pair_count", 0)
    else:
        weekly_pair_count = 0

    lines = [
        "计划完成情况报告",
        "",
        "汇总：",
        f"- 计划范围：{plan_start} 至 {plan_end}",
        f"- 完成记录范围：{done_start} 至 {done_end}",
        f"- 共追踪计划 {len(followups)} 条，周配对 {weekly_pair_count} 组。",
        f"- 已完成 {status_counts.get('已完成', 0)} 条，部分完成 {status_counts.get('部分完成', 0)} 条，未完成 {status_counts.get('未完成', 0)} 条，证据不足 {status_counts.get('证据不足', 0)} 条。",
        "- 判定说明：工具只依据结构化周报中的计划内容、下一期完成项和候选完成项进行保守判断；未找到后续周报时标为证据不足。",
        "",
    ]
    unfinished = [item for item in judged if item["status"] in {"未完成", "部分完成", "证据不足"}]
    if unfinished:
        lines.extend(
            [
                "未完全完成/需关注：",
                "| 人员 | 部门 | 计划日期 | 后续完成日期 | 状态 | 计划内容 | 依据 |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for item in unfinished:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _markdown_table_cell(item["user_name"]),
                        _markdown_table_cell(item["department"]),
                        _markdown_table_cell(item["plan_date"]),
                        _markdown_table_cell(item["completion_date"]),
                        _markdown_table_cell(item["status"]),
                        _markdown_table_cell(_clip_text(item["plan_text"], 120)),
                        _markdown_table_cell(_clip_text(item["reason"], 160)),
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.extend(
        [
            "全部计划明细：",
            "| 人员 | 部门 | 计划日期 | 后续完成日期 | 状态 | 计划内容 | 依据 |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for item in judged:
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_table_cell(item["user_name"]),
                    _markdown_table_cell(item["department"]),
                    _markdown_table_cell(item["plan_date"]),
                    _markdown_table_cell(item["completion_date"]),
                    _markdown_table_cell(item["status"]),
                    _markdown_table_cell(_clip_text(item["plan_text"], 120)),
                    _markdown_table_cell(_clip_text(item["reason"], 160)),
                ]
            )
            + " |"
        )
    return "\n".join(lines).strip()


def _judge_plan_followup(followup: Mapping[str, Any]) -> dict[str, str]:
    plan_text = str(followup.get("plan_text") or "").strip()
    done_items = followup.get("done_items", [])
    candidate_items = followup.get("candidate_done_items", [])
    done_texts = [
        str(item.get("done_text") or "").strip()
        for item in done_items
        if isinstance(item, Mapping) and str(item.get("done_text") or "").strip()
    ] if isinstance(done_items, list) else []
    candidate_rows = [item for item in candidate_items if isinstance(item, Mapping)] if isinstance(candidate_items, list) else []

    if not followup.get("completion_date"):
        status = "证据不足"
        reason = "未找到该计划之后的周报完成记录。"
    elif not done_texts:
        status = "证据不足"
        reason = "找到了后续周报日期，但没有可用的本周完成事项。"
    else:
        best_candidate = candidate_rows[0] if candidate_rows else {}
        match_metrics = _candidate_judgement_metrics(plan_text=plan_text, candidate=best_candidate)
        overlap_keywords = match_metrics.get("overlap_keywords", [])
        item_overlap_keywords = match_metrics.get("item_overlap_keywords", [])
        strong_overlap_keywords = _strong_overlap_keywords(overlap_keywords)
        strong_item_overlap_keywords = _strong_overlap_keywords(item_overlap_keywords)
        keyword_count = len(strong_overlap_keywords)
        item_keyword_count = len(strong_item_overlap_keywords)
        coverage = _safe_float(match_metrics.get("coverage"), 0.0)
        item_coverage = _safe_float(match_metrics.get("item_coverage"), 0.0)
        exact_phrase = bool(match_metrics.get("exact_phrase"))
        item_exact_phrase = bool(match_metrics.get("item_exact_phrase"))
        longest_overlap = _safe_int(match_metrics.get("longest_overlap"), 0)
        item_longest_overlap = _safe_int(match_metrics.get("item_longest_overlap"), 0)
        best_done_text = str(best_candidate.get("done_text") or done_texts[0] or "").strip() if isinstance(best_candidate, Mapping) else done_texts[0]
        reason_keywords = strong_item_overlap_keywords or strong_overlap_keywords
        if (
            exact_phrase
            or item_exact_phrase
            or (item_keyword_count >= 3 and item_coverage >= 0.45)
            or (item_keyword_count >= 2 and keyword_count >= 3 and item_coverage >= 0.35)
        ):
            status = "已完成"
            reason = f"后续完成记录与计划有明确关键词重合：{_format_keywords(reason_keywords)}；完成项：{_clip_text(best_done_text, 120)}"
        elif (
            item_keyword_count >= 2
            or (keyword_count >= 3 and coverage >= 0.3)
            or (item_keyword_count >= 1 and max(longest_overlap, item_longest_overlap) >= 6)
        ):
            status = "部分完成"
            reason = f"后续完成记录与计划有部分关键词重合：{_format_keywords(reason_keywords)}；完成项：{_clip_text(best_done_text, 120)}"
        else:
            status = "未完成"
            reason = "后续周报有完成记录，但未找到与该计划直接相关的候选完成项。"

    return {
        "user_name": str(followup.get("user_name") or "未知人员"),
        "department": str(followup.get("department") or "未填写部门"),
        "plan_date": str(followup.get("plan_date") or ""),
        "completion_date": str(followup.get("completion_date") or "未找到"),
        "plan_text": plan_text,
        "status": status,
        "reason": reason,
    }


def _format_keywords(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    return "、".join(str(item) for item in value[:8])


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _candidate_judgement_metrics(*, plan_text: str, candidate: Mapping[str, Any]) -> dict[str, Any]:
    metrics = {
        "overlap_keywords": candidate.get("overlap_keywords", []),
        "item_overlap_keywords": candidate.get("item_overlap_keywords", []),
        "coverage": candidate.get("coverage"),
        "item_coverage": candidate.get("item_coverage"),
        "exact_phrase": candidate.get("exact_phrase"),
        "item_exact_phrase": candidate.get("item_exact_phrase"),
        "longest_overlap": candidate.get("longest_overlap"),
        "item_longest_overlap": candidate.get("item_longest_overlap"),
    }
    if all(metrics.get(key) is not None for key in ("coverage", "item_coverage", "longest_overlap", "item_longest_overlap")):
        return metrics

    done_text = str(candidate.get("done_text") or "").strip()
    if not done_text:
        return metrics
    rescored = _score_done_matches(
        plan_text,
        [
            {
                "item_text": done_text,
                "evidence_text": candidate.get("evidence_text") or "",
                "report_date": candidate.get("report_date"),
                "source_doc_id": candidate.get("source_doc_id"),
                "source_chunk_id": candidate.get("source_chunk_id"),
            }
        ],
    )
    if rescored:
        for key, value in rescored[0].items():
            metrics[key] = value
    return metrics


def _strong_overlap_keywords(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    strong: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        if not text or _is_weak_overlap_keyword(text):
            continue
        normalized = _compact_text(text)
        if normalized in seen:
            continue
        seen.add(normalized)
        strong.append(text)
    return strong


def _specific_business_overlap_keywords(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    specific: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        if not text:
            continue
        normalized = _compact_text(text)
        if (
            normalized in seen
            or normalized in _GENERIC_DEPT_PLAN_MATCH_TOKENS
            or normalized in _WEAK_MATCH_TOKENS
        ):
            continue
        if len(normalized) <= 2 and normalized not in _SPECIFIC_SHORT_BUSINESS_TOKENS:
            continue
        seen.add(normalized)
        specific.append(text)
    return specific


def _has_specific_business_keyword_support(*groups: Any) -> bool:
    values: list[str] = []
    for group in groups:
        if isinstance(group, list):
            values.extend(str(item or "").strip() for item in group)
    normalized_values = {_compact_text(value) for value in values if str(value or "").strip()}
    if not normalized_values:
        return False
    specific_terms = {
        value
        for value in normalized_values
        if value
        and value not in _GENERIC_DEPT_PLAN_MATCH_TOKENS
        and value not in _WEAK_MATCH_TOKENS
        and (len(value) > 2 or value in _SPECIFIC_SHORT_BUSINESS_TOKENS)
    }
    if not specific_terms:
        return False
    if any(len(value) >= 4 for value in specific_terms):
        return True
    action_terms = normalized_values & _DEPT_PLAN_ACTION_TOKENS
    return bool(action_terms and specific_terms)


def _is_weak_overlap_keyword(value: str) -> bool:
    normalized = _compact_text(value)
    if normalized in _GENERIC_DEPT_PLAN_MATCH_TOKENS:
        return True
    if normalized in _IMPORTANT_SHORT_TOKENS:
        return False
    if normalized in _WEAK_MATCH_TOKENS:
        return True
    if len(normalized) <= 2:
        return True
    if normalized in {"进行安装", "相关工作", "其他工作", "持续推进"}:
        return True
    if "设备" in normalized and any(char.isdigit() for char in normalized) and len(normalized) <= 6:
        return True
    return normalized.startswith(".") and "设备" in normalized


def _markdown_table_cell(value: Any) -> str:
    text = str(value or "").replace("\n", " ").replace("|", "\\|").strip()
    return text or "-"


def _attach_weekly_blocker_context(*, result: dict[str, Any], weekly_reports_output: Any) -> None:
    people = _build_weekly_blocker_people(
        weekly_reports_output=weekly_reports_output,
        plan_followups=result.get("plan_followups", []),
        trace_scope=result.get("trace_scope", {}),
    )
    result["weekly_blocker_people"] = people
    result["weekly_blocker_context_text"] = _format_weekly_blocker_context(
        people=people,
        date_range=result.get("query_date_range", {}),
        pairing_summary=result.get("pairing_summary", {}),
    )


def _build_weekly_blocker_people(
    *,
    weekly_reports_output: Any,
    plan_followups: Any,
    trace_scope: Any,
) -> list[dict[str, Any]]:
    items = _weekly_report_items_from_output(weekly_reports_output)
    grouped: dict[tuple[str, str | None], dict[str, Any]] = {}
    for item in items:
        row_user = _optional_str(item.get("user_name"))
        row_department = _optional_str(item.get("department"))
        if not row_user:
            continue
        key = (row_user, row_department)
        group = grouped.setdefault(
            key,
            {
                "user_name": row_user,
                "department": row_department,
                "source": "inferred_from_plan_followups",
                "risk_and_help": None,
                "source_row_count": 0,
                "inferred_followups": [],
            },
        )
        group["source_row_count"] += 1
        risk_text = _optional_str(item.get("risk_and_help"))
        if risk_text and _risk_and_help_has_actionable_blocker(risk_text) and group["risk_and_help"] is None:
            group["source"] = "employee_risk_and_help"
            group["risk_and_help"] = risk_text

    followups_by_person: dict[tuple[str, str | None], list[dict[str, Any]]] = {}
    if isinstance(plan_followups, list):
        for followup in plan_followups:
            if not isinstance(followup, Mapping):
                continue
            row_user = _optional_str(followup.get("user_name"))
            row_department = _optional_str(followup.get("department"))
            if not row_user:
                continue
            followups_by_person.setdefault((row_user, row_department), []).append(_compact_plan_followup(followup))

    traced_people = trace_scope.get("traced_people", []) if isinstance(trace_scope, Mapping) else []
    traced_keys = {
        (_optional_str(item.get("user_name")), _optional_str(item.get("department")))
        for item in traced_people
        if isinstance(item, Mapping) and _optional_str(item.get("user_name"))
    }
    for key in traced_keys:
        if key not in grouped:
            grouped[key] = {
                "user_name": key[0],
                "department": key[1],
                "source": "inferred_from_plan_followups",
                "risk_and_help": None,
                "source_row_count": 0,
                "inferred_followups": [],
            }

    for key, followups in followups_by_person.items():
        if key not in grouped:
            grouped[key] = {
                "user_name": key[0],
                "department": key[1],
                "source": "inferred_from_plan_followups",
                "risk_and_help": None,
                "source_row_count": 0,
                "inferred_followups": [],
            }
        grouped[key]["inferred_followups"] = followups[:8]

    return list(grouped.values())


def _compact_plan_followup(followup: Mapping[str, Any]) -> dict[str, Any]:
    done_items = followup.get("done_items", [])
    compact_done: list[dict[str, Any]] = []
    if isinstance(done_items, list):
        for item in done_items[:5]:
            if not isinstance(item, Mapping):
                continue
            overlap_keywords = item.get("overlap_keywords", [])
            compact_done.append(
                {
                    "done_text": _clip_text(item.get("done_text"), 240),
                    "report_date": item.get("report_date"),
                    "overlap_keywords": overlap_keywords[:10] if isinstance(overlap_keywords, list) else [],
                }
            )
    return {
        "plan_text": _clip_text(followup.get("plan_text"), 260),
        "plan_date": followup.get("plan_date"),
        "trace_window": followup.get("trace_window"),
        "done_items": compact_done,
    }


def _format_weekly_blocker_context(
    *,
    people: list[dict[str, Any]],
    date_range: Any,
    pairing_summary: Any,
) -> str:
    if isinstance(date_range, Mapping):
        target_start = date_range.get("this_week_start", "")
        target_end = date_range.get("this_week_end", "")
        trace_start = date_range.get("last_week_start", "")
        trace_end = date_range.get("last_week_end", "")
    else:
        target_start = target_end = trace_start = trace_end = ""
    if isinstance(pairing_summary, Mapping):
        traced_count = pairing_summary.get(
            "traced_people_count",
            sum(1 for person in people if not person.get("risk_and_help")),
        )
        skipped_count = pairing_summary.get(
            "skipped_people_count",
            sum(1 for person in people if person.get("risk_and_help")),
        )
    else:
        traced_count = sum(1 for person in people if not person.get("risk_and_help"))
        skipped_count = sum(1 for person in people if person.get("risk_and_help"))
    lines = [
        "周报卡点压缩上下文",
        f"- 目标周: {target_start} 至 {target_end}",
        f"- 追溯计划周: {trace_start} 至 {trace_end}",
        f"- 追溯统计: 未填写或仅填写无卡点表述的人员追溯 {traced_count} 人，有效员工自填卡点直接采用 {skipped_count} 人。",
        "",
    ]
    for person in people:
        name = person.get("user_name") or "未知人员"
        department = person.get("department") or "未填写部门"
        lines.append(f"人员: {name} / {department}")
        if person.get("risk_and_help"):
            lines.append("来源: 员工自填卡点")
            lines.append(f"卡点: {_clip_text(person.get('risk_and_help'), 900)}")
        else:
            lines.append("来源: 未填写卡点，根据上一周计划与目标周完成记录推断")
            followups = person.get("inferred_followups", [])
            if isinstance(followups, list) and followups:
                for index, followup in enumerate(followups[:8], start=1):
                    lines.append(f"推断证据 {index}:")
                    lines.append(f"  计划: {followup.get('plan_text') or '无'}")
                    done_items = followup.get("done_items", [])
                    if isinstance(done_items, list) and done_items:
                        for done in done_items[:5]:
                            if not isinstance(done, Mapping):
                                continue
                            lines.append(f"  后续完成: {done.get('done_text') or '无'}")
                    else:
                        lines.append("  后续完成: 未找到匹配完成记录")
            else:
                lines.append("推断证据: 未找到可用于追溯的计划/完成配对")
        lines.append("")
    return "\n".join(lines).strip()


def _build_dept_plan_opl_issue_pool(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        issue_description = _optional_str(row.get("issue_description"))
        solution_progress = _optional_str(row.get("solution_progress"))
        if not issue_description and not solution_progress:
            continue
        identity = _opl_issue_identity(row)
        if identity in seen:
            continue
        seen.add(identity)
        status_group = _opl_status_group(row.get("status"))
        pool.append(
            {
                "issue_ref": f"opl{len(pool) + 1}",
                "source_issue_id": row.get("id"),
                "doc_id": row.get("doc_id"),
                "file_path": row.get("file_path"),
                "sheet_name": row.get("sheet_name"),
                "source_row": row.get("source_row"),
                "source_no": row.get("source_no"),
                "department": row.get("department"),
                "assembly": row.get("assembly"),
                "issue_date": row.get("issue_date"),
                "issue_description": _clip_text(issue_description, 500),
                "status": row.get("status"),
                "status_group": status_group,
                "solution_progress": _clip_text(solution_progress, 500),
                "priority": row.get("priority"),
                "tracker_user": row.get("tracker_user"),
                "owner_user": row.get("owner_user"),
                "follow_date": row.get("follow_date"),
                "remark": _clip_text(row.get("remark"), 300),
                "open_issue": status_group == "open",
                "closed_issue": status_group == "closed",
                "high_priority": _is_high_priority_opl(row.get("priority")),
                "evidence_text": _clip_text(row.get("evidence_text"), 500),
                "source_doc_id": row.get("doc_id"),
                "source_chunk_id": row.get("evidence_chunk_id"),
            }
        )
    pool.sort(
        key=lambda item: (
            str(item.get("issue_date") or ""),
            str(item.get("follow_date") or ""),
            str(item.get("issue_ref") or ""),
        )
    )
    for index, issue in enumerate(pool, start=1):
        issue["issue_ref"] = f"opl{index}"
    return pool


def _dept_plan_opl_issue_match_result(
    *,
    plan: Mapping[str, Any],
    issue_pool: list[dict[str, Any]],
    owner_names: set[str],
) -> dict[str, Any]:
    plan_department = plan.get("department")
    plan_month = str(plan.get("month") or "").strip()
    plan_text = str(plan.get("plan_text") or "").strip()
    plan_profile = _dept_plan_text_profile(plan_text)
    scored: list[dict[str, Any]] = []
    possible_evidence: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    filtered_counts: dict[str, int] = {}
    department_compatible_count = 0
    owner_match_count = 0
    tracker_match_count = 0
    keyword_match_count = 0
    strong_keyword_match_count = 0
    business_keyword_match_count = 0
    open_issue_count = 0
    high_priority_open_issue_count = 0
    for issue in issue_pool:
        if not isinstance(issue, Mapping):
            continue
        department_match = _departments_compatible(plan_department, issue.get("department"))
        if department_match:
            department_compatible_count += 1
        owner_match = _opl_issue_person_matches(issue.get("owner_user"), owner_names)
        tracker_match = _opl_issue_person_matches(issue.get("tracker_user"), owner_names)
        if owner_match:
            owner_match_count += 1
        if tracker_match:
            tracker_match_count += 1
        if issue.get("open_issue"):
            open_issue_count += 1
            if issue.get("high_priority"):
                high_priority_open_issue_count += 1

        candidate = _build_dept_plan_opl_issue_candidate(
            plan_profile=plan_profile,
            issue=issue,
            department_match=department_match,
            owner_match=owner_match,
            tracker_match=tracker_match,
        )
        if candidate is None:
            continue
        if candidate.get("has_keyword_support"):
            keyword_match_count += 1
        if candidate.get("strong_keyword_support"):
            strong_keyword_match_count += 1
        if candidate.get("business_keyword_support"):
            business_keyword_match_count += 1
        audit_rows.append(candidate)
        if candidate.get("filter_reason"):
            reason = str(candidate.get("filter_reason"))
            filtered_counts[reason] = filtered_counts.get(reason, 0) + 1
            possible_evidence.append(_possible_opl_evidence_from_candidate(candidate))
            continue
        scored.append(candidate)

    selected_candidate_limit = max(1, _env_int("MYSQL_DEPT_PLAN_OPL_CANDIDATE_MAX_ITEMS", 8))
    scored.sort(key=lambda item: _opl_issue_candidate_sort_key(item, plan_month=plan_month), reverse=True)
    selected = scored[:selected_candidate_limit]
    selected_keys = {_opl_issue_candidate_identity(item) for item in selected}
    capped_candidates = [
        item
        for item in scored[selected_candidate_limit:]
        if _opl_issue_candidate_identity(item) not in selected_keys
    ]
    for item in capped_candidates:
        possible_evidence.append(_possible_opl_evidence_from_candidate(item, filter_reason="not_selected_due_to_candidate_cap"))
    possible_evidence_before_cap_count = len(possible_evidence)
    possible_evidence = _dedupe_possible_opl_evidence(possible_evidence)
    possible_evidence.sort(
        key=lambda item: _possible_opl_evidence_sort_key(item, plan_month=plan_month),
        reverse=True,
    )
    possible_evidence = possible_evidence[: _env_int("MYSQL_DEPT_PLAN_POSSIBLE_OPL_MAX_ITEMS", 16)]
    selected_candidates = [_compact_opl_issue_candidate(item) for item in selected]
    return {
        "selected_candidates": selected_candidates,
        "possible_evidence": possible_evidence,
        "audit": {
            "raw_opl_issue_count": len(issue_pool),
            "department_compatible_count": department_compatible_count,
            "owner_match_count": owner_match_count,
            "tracker_match_count": tracker_match_count,
            "keyword_match_count": keyword_match_count,
            "strong_keyword_match_count": strong_keyword_match_count,
            "business_keyword_match_count": business_keyword_match_count,
            "open_issue_count": open_issue_count,
            "high_priority_open_issue_count": high_priority_open_issue_count,
            "candidate_pool_count": len(audit_rows),
            "selected_before_cap_count": len(scored),
            "selected_count": len(selected_candidates),
            "selected_candidate_limit": selected_candidate_limit,
            "capped_candidate_count": len(capped_candidates),
            "possible_evidence_count": len(possible_evidence),
            "possible_evidence_before_cap_count": possible_evidence_before_cap_count,
            "filtered_counts": filtered_counts,
            "owner_names": sorted(owner_names),
            "rule": (
                "OPL 候选按部门、负责人/跟踪人、计划文本与问题描述/解决进展/备注的关键词关联；"
                "未闭环 OPL 作为风险或卡点证据，已解决 OPL 仅在解决措施对应计划目标时作为完成判断证据。"
            ),
            "sample_capped": [_compact_opl_issue_candidate(item) for item in capped_candidates[:3]],
            "sample_filtered": [
                {
                    key: row.get(key)
                    for key in ("filter_reason", "issue_ref", "issue_date", "status", "priority", "issue_description")
                    if row.get(key) not in (None, "", [])
                }
                for row in audit_rows
                if row.get("filter_reason")
            ][:3],
        },
    }


def _build_dept_plan_opl_issue_candidate(
    *,
    plan_profile: Mapping[str, Any],
    issue: Mapping[str, Any],
    department_match: bool,
    owner_match: bool,
    tracker_match: bool,
) -> dict[str, Any] | None:
    issue_text = _opl_issue_match_text(issue)
    if not issue_text:
        return None
    metrics = _weekly_report_text_match_metrics(plan_profile=plan_profile, report_text=issue_text)
    overlap_keywords = metrics.get("overlap_keywords", [])
    strong_keywords = _strong_overlap_keywords(overlap_keywords if isinstance(overlap_keywords, list) else [])
    specific_keywords = _specific_business_overlap_keywords(strong_keywords)
    exact_phrase = bool(metrics.get("exact_phrase"))
    business_keyword_support = _has_specific_business_keyword_support(strong_keywords)
    has_keyword_support = bool(overlap_keywords or exact_phrase)
    strong_keyword_support = bool(
        exact_phrase
        or business_keyword_support
        or len(strong_keywords) >= 3
        or _safe_float(metrics.get("coverage"), 0.0) >= 0.35
    )
    selected = bool(
        ((owner_match or tracker_match) and (department_match or has_keyword_support))
        or (department_match and strong_keyword_support)
        or business_keyword_support
    )
    filter_reason = ""
    if not selected:
        if not department_match and not owner_match and not tracker_match:
            filter_reason = "department_mismatch"
        elif not has_keyword_support:
            filter_reason = "no_owner_or_keyword_support"
        else:
            filter_reason = "weak_opl_match"

    if owner_match and strong_keyword_support:
        candidate_source = "owner_opl_issue_keyword_trace"
        confidence = "high"
    elif owner_match:
        candidate_source = "owner_opl_issue_trace"
        confidence = "medium"
    elif tracker_match and strong_keyword_support:
        candidate_source = "tracker_opl_issue_keyword_trace"
        confidence = "medium"
    elif tracker_match:
        candidate_source = "tracker_opl_issue_trace"
        confidence = "medium"
    elif department_match and strong_keyword_support:
        candidate_source = "department_opl_issue_keyword_trace"
        confidence = "medium"
    else:
        candidate_source = "keyword_opl_issue_trace"
        confidence = "low"

    priority_score = (
        _safe_float(metrics.get("score"), 0.0)
        + (_safe_float(metrics.get("coverage"), 0.0) * 0.8)
        + (2.2 if owner_match else 0.0)
        + (1.3 if tracker_match else 0.0)
        + (0.8 if department_match else 0.0)
        + (0.9 if business_keyword_support else 0.0)
        + (0.8 if issue.get("open_issue") else 0.0)
        + (0.6 if issue.get("high_priority") else 0.0)
        + min(len(specific_keywords) * 0.2, 0.8)
    )
    return {
        "issue_ref": issue.get("issue_ref"),
        "source_issue_id": issue.get("source_issue_id"),
        "issue_date": issue.get("issue_date"),
        "follow_date": issue.get("follow_date"),
        "department": issue.get("department"),
        "assembly": issue.get("assembly"),
        "issue_description": issue.get("issue_description"),
        "status": issue.get("status"),
        "status_group": issue.get("status_group"),
        "solution_progress": issue.get("solution_progress"),
        "priority": issue.get("priority"),
        "tracker_user": issue.get("tracker_user"),
        "owner_user": issue.get("owner_user"),
        "remark": issue.get("remark"),
        "owner_match": owner_match,
        "tracker_match": tracker_match,
        "department_match": department_match,
        "open_issue": bool(issue.get("open_issue")),
        "closed_issue": bool(issue.get("closed_issue")),
        "high_priority": bool(issue.get("high_priority")),
        "strong_keyword_support": strong_keyword_support,
        "business_keyword_support": business_keyword_support,
        "has_keyword_support": has_keyword_support,
        "candidate_confidence": confidence,
        "candidate_source": candidate_source,
        "filter_reason": filter_reason,
        "overlap_keywords": overlap_keywords[:12] if isinstance(overlap_keywords, list) else [],
        "specific_overlap_keywords": specific_keywords[:12],
        "coverage": metrics.get("coverage"),
        "exact_phrase": exact_phrase,
        "priority_score": round(priority_score, 4),
        "evidence_text": _clip_text(issue.get("evidence_text"), 350),
        "source_doc_id": issue.get("source_doc_id"),
        "source_chunk_id": issue.get("source_chunk_id"),
    }


def _compact_opl_issue_candidate(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "issue_ref": item.get("issue_ref"),
        "source_issue_id": item.get("source_issue_id"),
        "issue_date": item.get("issue_date"),
        "follow_date": item.get("follow_date"),
        "department": item.get("department"),
        "assembly": item.get("assembly"),
        "issue_description": _clip_text(item.get("issue_description"), 260),
        "status": item.get("status"),
        "status_group": item.get("status_group"),
        "solution_progress": _clip_text(item.get("solution_progress"), 260),
        "priority": item.get("priority"),
        "tracker_user": item.get("tracker_user"),
        "owner_user": item.get("owner_user"),
        "remark": _clip_text(item.get("remark"), 160),
        "owner_match": bool(item.get("owner_match")),
        "tracker_match": bool(item.get("tracker_match")),
        "department_match": bool(item.get("department_match")),
        "open_issue": bool(item.get("open_issue")),
        "closed_issue": bool(item.get("closed_issue")),
        "high_priority": bool(item.get("high_priority")),
        "strong_keyword_support": bool(item.get("strong_keyword_support")),
        "business_keyword_support": bool(item.get("business_keyword_support")),
        "candidate_confidence": item.get("candidate_confidence") or "low",
        "candidate_source": item.get("candidate_source") or "opl_issue_trace",
        "overlap_keywords": item.get("overlap_keywords", [])[:8] if isinstance(item.get("overlap_keywords"), list) else [],
        "specific_overlap_keywords": item.get("specific_overlap_keywords", [])[:8]
        if isinstance(item.get("specific_overlap_keywords"), list)
        else [],
        "coverage": item.get("coverage"),
        "exact_phrase": bool(item.get("exact_phrase")),
        "evidence_text": _clip_text(item.get("evidence_text"), 260),
        "source_doc_id": item.get("source_doc_id"),
        "source_chunk_id": item.get("source_chunk_id"),
    }


def _possible_opl_evidence_from_candidate(
    row: Mapping[str, Any],
    *,
    filter_reason: str | None = None,
) -> dict[str, Any]:
    item = _compact_opl_issue_candidate(row)
    item["filter_reason"] = filter_reason or row.get("filter_reason") or "possible_opl_evidence"
    item["priority_score"] = row.get("priority_score")
    return item


def _dedupe_possible_opl_evidence(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for item in items:
        identity = _opl_issue_candidate_identity(item)
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(item)
    return deduped


def _opl_issue_candidate_sort_key(row: Mapping[str, Any], *, plan_month: str) -> tuple[Any, ...]:
    open_issue = bool(row.get("open_issue"))
    high_priority = bool(row.get("high_priority"))
    owner_match = bool(row.get("owner_match"))
    tracker_match = bool(row.get("tracker_match"))
    department_match = bool(row.get("department_match"))
    strong_keyword_support = bool(row.get("strong_keyword_support"))
    keyword_support = bool(row.get("has_keyword_support") or row.get("overlap_keywords") or row.get("exact_phrase"))
    in_plan_month = _opl_issue_in_plan_month(row, plan_month=plan_month)
    if owner_match and strong_keyword_support and open_issue:
        evidence_tier = 90
    elif owner_match and open_issue:
        evidence_tier = 84
    elif department_match and strong_keyword_support and open_issue:
        evidence_tier = 78
    elif tracker_match and strong_keyword_support:
        evidence_tier = 72
    elif owner_match or tracker_match:
        evidence_tier = 64
    elif strong_keyword_support and department_match:
        evidence_tier = 58
    elif strong_keyword_support:
        evidence_tier = 44
    elif keyword_support:
        evidence_tier = 24
    else:
        evidence_tier = 0
    return (
        evidence_tier,
        high_priority,
        open_issue,
        in_plan_month,
        _safe_float(row.get("priority_score"), 0.0),
        str(row.get("follow_date") or row.get("issue_date") or ""),
    )


def _possible_opl_evidence_sort_key(row: Mapping[str, Any], *, plan_month: str) -> tuple[Any, ...]:
    reason = str(row.get("filter_reason") or "")
    reason_priority = {
        "not_selected_due_to_candidate_cap": 4,
        "weak_opl_match": 2,
        "no_owner_or_keyword_support": 1,
        "department_mismatch": 0,
    }.get(reason, 0)
    return (
        reason_priority,
        bool(row.get("open_issue")),
        bool(row.get("high_priority")),
        bool(row.get("owner_match")),
        bool(row.get("tracker_match")),
        bool(row.get("strong_keyword_support")),
        _opl_issue_in_plan_month(row, plan_month=plan_month),
        _safe_float(row.get("priority_score"), 0.0),
        str(row.get("follow_date") or row.get("issue_date") or ""),
    )


def _opl_issue_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    row_id = row.get("id")
    if row_id not in (None, ""):
        return ("id", str(row_id))
    return (
        "fields",
        row.get("doc_id"),
        row.get("sheet_name"),
        row.get("source_row"),
        _optional_str(row.get("issue_description")),
    )


def _opl_issue_candidate_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    issue_id = row.get("source_issue_id")
    if issue_id not in (None, ""):
        return ("id", str(issue_id))
    return (
        "fields",
        row.get("issue_date"),
        row.get("department"),
        row.get("owner_user"),
        row.get("tracker_user"),
        row.get("issue_description"),
    )


def _opl_issue_match_text(issue: Mapping[str, Any]) -> str:
    return " ".join(
        str(value or "").strip()
        for value in (
            issue.get("assembly"),
            issue.get("issue_description"),
            issue.get("solution_progress"),
            issue.get("remark"),
            issue.get("evidence_text"),
        )
        if str(value or "").strip()
    )


def _opl_issue_in_plan_month(row: Mapping[str, Any], *, plan_month: str) -> bool:
    month = plan_month[:7]
    if not re.match(r"^\d{4}-\d{2}$", month):
        return True
    return str(row.get("issue_date") or "").startswith(month) or str(row.get("follow_date") or "").startswith(month)


def _opl_issue_person_matches(value: Any, owner_names: set[str]) -> bool:
    if not owner_names:
        return False
    names = _split_owner_users(value)
    if names:
        return any(_canonical_person_name(name) in owner_names for name in names)
    text = _optional_str(value)
    if text is None:
        return False
    return any(_same_user(owner_name, text) for owner_name in owner_names)


def _build_dept_plan_completion_evidence(
    *,
    plans: list[dict[str, Any]],
    weekly_completed: list[dict[str, Any]],
    self_eval_items: list[dict[str, Any]] | None = None,
    owner_self_eval_items: list[dict[str, Any]] | None = None,
    owner_weekly_reports: list[dict[str, Any]] | None = None,
    collaboration_weekly_reports: list[dict[str, Any]] | None = None,
    opl_issues: list[dict[str, Any]] | None = None,
    known_user_names: list[str] | set[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    followups: list[dict[str, Any]] = []
    legacy_self_eval_items = self_eval_items or []
    direct_owner_self_eval_items = owner_self_eval_items or []
    use_direct_owner_self_eval = owner_self_eval_items is not None
    opl_issue_pool = _build_dept_plan_opl_issue_pool(opl_issues or [])
    owner_weekly_reports_pool = _build_dept_plan_weekly_report_pool(
        owner_weekly_reports or [],
        report_id_prefix="or",
    )
    collaboration_source_reports = (
        collaboration_weekly_reports
        if collaboration_weekly_reports is not None
        else _weekly_item_rows_as_report_rows(weekly_completed)
    )
    collaboration_weekly_reports_pool = _build_dept_plan_weekly_report_pool(
        collaboration_source_reports or [],
        report_id_prefix="cr",
    )
    person_registry = _normalize_known_person_names(
        [
            *(known_user_names or []),
            *_dept_plan_owner_names(plans, known_user_names=known_user_names),
            *_person_names_from_rows(plans, keys=("owner_user",)),
            *_person_names_from_rows(owner_weekly_reports_pool, keys=("user_name", "提交人")),
            *_person_names_from_rows(collaboration_weekly_reports_pool, keys=("user_name", "提交人")),
            *_person_names_from_rows(direct_owner_self_eval_items, keys=("user_name",)),
            *_person_names_from_rows(opl_issues or [], keys=("owner_user", "tracker_user")),
        ]
    )
    for index, plan in enumerate(plans, start=1):
        plan_text = str(plan.get("plan_text") or "").strip()
        owner_name_list = _split_owner_users(plan.get("owner_user"), known_user_names=person_registry)
        owner_names = set(owner_name_list)
        owner_user_display, owner_name_audit = _dept_plan_display_owner_user(
            plan.get("owner_user"),
            owner_name_list,
            known_user_names=person_registry,
        )
        owner_report_refs = _dept_plan_report_refs_for_owners(
            report_pool=owner_weekly_reports_pool,
            owner_names=owner_names,
        )
        collaboration_match_result = _dept_plan_collaboration_weekly_match_result(
            plan=plan,
            report_pool=collaboration_weekly_reports_pool,
            owner_names=owner_names,
        )
        weekly_candidates = collaboration_match_result["selected_candidates"]
        if use_direct_owner_self_eval:
            owner_self_eval_reports = _dept_plan_owner_self_eval_reports(
                plan=plan,
                owner_self_eval_items=direct_owner_self_eval_items,
            )
            plan_owner_self_eval_items = _dept_plan_owner_self_eval_items(
                plan=plan,
                owner_self_eval_items=direct_owner_self_eval_items,
            )
            missing_owner_self_eval_users = _dept_plan_missing_owner_self_eval_users(
                plan=plan,
                owner_self_eval_items=direct_owner_self_eval_items,
            )
        else:
            owner_self_eval_reports = []
            plan_owner_self_eval_items = []
            missing_owner_self_eval_users = []
        opl_match_result = _dept_plan_opl_issue_match_result(
            plan=plan,
            issue_pool=opl_issue_pool,
            owner_names=owner_names,
        )
        self_eval_candidates = (
            []
            if use_direct_owner_self_eval
            else _dept_plan_self_eval_candidates(plan=plan, self_eval_items=legacy_self_eval_items)
        )
        followups.append(
            {
                "plan_id": plan.get("id") or index,
                "department": plan.get("department"),
                "owner_user": owner_user_display,
                "owner_user_original": plan.get("owner_user"),
                "owner_name_audit": owner_name_audit,
                "plan_month": plan.get("month"),
                "due_date": plan.get("due_date"),
                "target": plan.get("target"),
                "slide_no": plan.get("slide_no"),
                "plan_text": plan_text,
                "plan_evidence_text": _clip_text(plan.get("evidence_text"), 400),
                "owner_names": owner_name_list,
                "owner_weekly_report_refs": owner_report_refs,
                "owner_weekly_reports": owner_report_refs,
                "weekly_done_candidates": weekly_candidates,
                "possible_weekly_evidence": collaboration_match_result["possible_evidence"],
                "collaboration_weekly_report_refs": collaboration_match_result["selected_report_refs"],
                "weekly_match_audit": collaboration_match_result["audit"],
                "owner_self_eval_reports": owner_self_eval_reports,
                "owner_self_eval_items": plan_owner_self_eval_items,
                "owner_self_eval_found": bool(owner_self_eval_reports or plan_owner_self_eval_items),
                "missing_owner_self_eval_users": missing_owner_self_eval_users,
                "self_eval_candidates": self_eval_candidates,
                "opl_issue_candidates": opl_match_result["selected_candidates"],
                "possible_opl_evidence": opl_match_result["possible_evidence"],
                "opl_match_audit": opl_match_result["audit"],
                "source_doc_id": plan.get("doc_id"),
                "source_chunk_id": plan.get("evidence_chunk_id"),
            }
        )
    return {
        "dept_plans": plans,
        "weekly_items": weekly_completed,
        "owner_weekly_reports_pool": owner_weekly_reports_pool,
        "collaboration_weekly_reports_pool": collaboration_weekly_reports_pool,
        "owner_self_eval_items": direct_owner_self_eval_items,
        "self_eval_items": direct_owner_self_eval_items if use_direct_owner_self_eval else legacy_self_eval_items,
        "opl_issues": opl_issues or [],
        "opl_issue_pool": opl_issue_pool,
        "dept_plan_followups": followups,
        "pairing_summary": {
            "total_plans": len(followups),
            "weekly_evidence_candidate_count": sum(len(item.get("weekly_done_candidates", [])) for item in followups),
            "owner_weekly_report_count": len(owner_weekly_reports_pool),
            "owner_weekly_report_ref_count": sum(len(item.get("owner_weekly_report_refs", [])) for item in followups),
            "collaboration_weekly_report_count": len(collaboration_weekly_reports_pool),
            "collaboration_weekly_report_ref_count": sum(
                len(item.get("collaboration_weekly_report_refs", [])) for item in followups
            ),
            "possible_weekly_evidence_count": sum(len(item.get("possible_weekly_evidence", [])) for item in followups),
            "owner_self_eval_item_count": sum(len(item.get("owner_self_eval_items", [])) for item in followups),
            "owner_self_eval_report_ref_count": sum(len(item.get("owner_self_eval_reports", [])) for item in followups),
            "owner_name_normalized_count": sum(
                1
                for item in followups
                if str(item.get("owner_user") or "") != str(item.get("owner_user_original") or "")
                and item.get("owner_user") not in (None, "")
            ),
            "owner_name_alias_normalization_count": sum(
                len(audit.get("alias_normalizations", []))
                for item in followups
                for audit in [item.get("owner_name_audit") if isinstance(item.get("owner_name_audit"), Mapping) else {}]
            ),
            "owner_name_unresolved_count": sum(
                1
                for item in followups
                for audit in [item.get("owner_name_audit") if isinstance(item.get("owner_name_audit"), Mapping) else {}]
                if audit.get("invalid_owner_text")
            ),
            "opl_issue_count": len(opl_issue_pool),
            "opl_issue_candidate_count": sum(len(item.get("opl_issue_candidates", [])) for item in followups),
            "possible_opl_evidence_count": sum(len(item.get("possible_opl_evidence", [])) for item in followups),
            "open_opl_issue_candidate_count": sum(
                1
                for item in followups
                for candidate in item.get("opl_issue_candidates", [])
                if isinstance(candidate, Mapping) and candidate.get("open_issue")
            ),
            "high_priority_open_opl_issue_candidate_count": sum(
                1
                for item in followups
                for candidate in item.get("opl_issue_candidates", [])
                if isinstance(candidate, Mapping) and candidate.get("open_issue") and candidate.get("high_priority")
            ),
            "plans_missing_owner_self_eval": sum(
                1
                for item in followups
                if item.get("missing_owner_self_eval_users")
            ),
            "self_eval_candidate_count": sum(len(item.get("self_eval_candidates", [])) for item in followups),
            "plans_without_evidence_candidates": sum(
                1
                for item in followups
                if not item.get("owner_weekly_report_refs")
                and not item.get("weekly_done_candidates")
                and not item.get("owner_self_eval_items")
                and not item.get("self_eval_candidates")
                and not item.get("opl_issue_candidates")
            ),
            "plans_with_possible_weekly_evidence": sum(
                1
                for item in followups
                if item.get("possible_weekly_evidence")
            ),
            "plans_with_opl_issue_candidates": sum(
                1
                for item in followups
                if item.get("opl_issue_candidates")
            ),
            "judgement_owner": "backend_llm",
            "judgement_instruction": (
                "MySQL 工具只按负责人、月份和周报日期从 weekly_reports 主表取负责人完整周报原文，"
                "计划项只挂 report_id 引用；非负责人协作材料也来自 weekly_reports 主表，"
                "负责人本人月度考核记录按计划负责人姓名直接查询 employee_self_eval_reports/items，"
                "OPL 问题清单按时间和部门取数后挂到相关计划项，未闭环问题作为风险/卡点证据，"
                "已解决且解决措施对应计划目标时才可作为完成判断证据；"
                "关键词仅用于辅助排序，完成状态必须由 text_generate_tool 后端 LLM 根据原文证据判断。"
            ),
        },
    }


def _weekly_item_rows_as_report_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    report_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        text = _optional_str(row.get("evidence_text")) or _optional_str(row.get("item_text"))
        if not text:
            continue
        report_rows.append(
            {
                "id": row.get("id"),
                "user_name": row.get("user_name"),
                "department": row.get("department"),
                "report_date": row.get("report_date"),
                "report_end": None,
                "this_week_raw": text,
                "next_week_raw": "",
                "risk_and_help": row.get("risk_and_help"),
                "source_doc_id": row.get("source_doc_id"),
                "source_chunk_id": row.get("source_chunk_id"),
            }
        )
    return report_rows


def _build_dept_plan_weekly_report_pool(
    reports: list[dict[str, Any]],
    *,
    report_id_prefix: str,
) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in reports:
        if not isinstance(row, Mapping):
            continue
        raw_text = _format_weekly_report_raw_text(row)
        if not raw_text:
            continue
        identity = _weekly_report_pool_identity(row=row, raw_text=raw_text)
        if identity in seen:
            continue
        seen.add(identity)
        pool.append(
            {
                "report_id": f"{report_id_prefix}{len(pool) + 1}",
                "source_report_id": row.get("id"),
                "日期": row.get("report_date") or "",
                "提交人": row.get("user_name") or "",
                "部门": row.get("department") or "",
                "事项类型": "weekly_report",
                "周报开始": row.get("report_date") or "",
                "周报结束": row.get("report_end") or "",
                "本周完成": _clip_text(row.get("this_week_raw"), _env_int("MYSQL_DEPT_PLAN_REPORT_SECTION_MAX_CHARS", 900)),
                "下周计划": _clip_text(row.get("next_week_raw"), _env_int("MYSQL_DEPT_PLAN_REPORT_SECTION_MAX_CHARS", 900)),
                "风险/求助": _clip_text(row.get("risk_and_help"), _env_int("MYSQL_DEPT_PLAN_REPORT_SECTION_MAX_CHARS", 600)),
                "周报原文": raw_text,
                "source_doc_id": row.get("source_doc_id"),
                "source_chunk_id": row.get("source_chunk_id"),
            }
        )
    pool.sort(
        key=lambda item: (
            str(item.get("日期") or ""),
            str(item.get("提交人") or ""),
            str(item.get("source_report_id") or ""),
            str(item.get("report_id") or ""),
        )
    )
    for index, report in enumerate(pool, start=1):
        report["report_id"] = f"{report_id_prefix}{index}"
    return pool


def _weekly_report_pool_identity(*, row: Mapping[str, Any], raw_text: str) -> tuple[Any, ...]:
    row_id = row.get("id")
    if row_id not in (None, ""):
        return ("id", str(row_id))
    return (
        "fields",
        row.get("report_date"),
        _optional_str(row.get("user_name")),
        _optional_str(row.get("department")),
        raw_text,
        row.get("source_doc_id"),
        row.get("source_chunk_id"),
    )


def _dept_plan_report_refs_for_owners(
    *,
    report_pool: list[dict[str, Any]],
    owner_names: set[str],
) -> list[dict[str, Any]]:
    if not owner_names:
        return []
    refs = [
        _dept_plan_report_ref(report)
        for report in report_pool
        if _canonical_person_name(str(report.get("提交人") or "")) in owner_names
    ]
    refs.sort(key=lambda item: (str(item.get("日期") or ""), str(item.get("提交人") or ""), str(item.get("report_id") or "")))
    return refs


def _dept_plan_report_ref(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "report_id": report.get("report_id"),
        "日期": report.get("日期"),
        "提交人": report.get("提交人"),
        "部门": report.get("部门"),
        "source_doc_id": report.get("source_doc_id"),
        "source_chunk_id": report.get("source_chunk_id"),
    }


def _dept_plan_collaboration_weekly_match_result(
    *,
    plan: Mapping[str, Any],
    report_pool: list[dict[str, Any]],
    owner_names: set[str],
) -> dict[str, Any]:
    plan_department = plan.get("department")
    plan_month = str(plan.get("month") or "").strip()
    plan_text = str(plan.get("plan_text") or "").strip()
    plan_profile = _dept_plan_text_profile(plan_text)
    scored: list[dict[str, Any]] = []
    possible_evidence: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    filtered_counts: dict[str, int] = {}
    owner_report_count = 0
    department_compatible_count = 0
    keyword_match_count = 0
    strong_keyword_match_count = 0
    business_keyword_match_count = 0
    for report in report_pool:
        submitter = _optional_str(report.get("提交人"))
        canonical_submitter = _canonical_person_name(submitter) if submitter else None
        owner_match = bool(canonical_submitter and canonical_submitter in owner_names)
        if owner_match:
            owner_report_count += 1
            continue
        department_match = _departments_compatible(plan_department, report.get("部门"))
        if department_match:
            department_compatible_count += 1
        candidate = _build_dept_plan_collaboration_candidate(
            plan_profile=plan_profile,
            report=report,
            department_match=department_match,
        )
        if candidate is None:
            continue
        audit_rows.append(candidate)
        if candidate.get("has_keyword_support"):
            keyword_match_count += 1
        if candidate.get("strong_keyword_support"):
            strong_keyword_match_count += 1
        if candidate.get("business_keyword_support"):
            business_keyword_match_count += 1
        if not department_match and not candidate.get("strong_keyword_support"):
            filtered_counts["department_mismatch"] = filtered_counts.get("department_mismatch", 0) + 1
            candidate["filter_reason"] = "department_mismatch"
            possible_evidence.append(_possible_weekly_evidence_from_candidate(candidate, filter_reason="department_mismatch"))
            continue
        if not candidate.get("strong_keyword_support") and not candidate.get("business_keyword_support"):
            filtered_counts["no_owner_or_keyword_support"] = filtered_counts.get("no_owner_or_keyword_support", 0) + 1
            candidate["filter_reason"] = "no_owner_or_keyword_support"
            possible_evidence.append(_possible_weekly_evidence_from_candidate(candidate, filter_reason="no_owner_or_keyword_support"))
            continue
        scored.append(candidate)

    selected_candidate_limit = max(0, _env_int("MYSQL_DEPT_PLAN_COLLAB_REPORT_CANDIDATE_MAX_ITEMS", 8))
    scored.sort(
        key=lambda item: _dept_plan_candidate_sort_key(item, plan_month=plan_month),
        reverse=True,
    )
    selected = scored[:selected_candidate_limit]
    selected_keys = {_weekly_candidate_identity(item) for item in selected}
    capped_candidates = [item for item in scored[selected_candidate_limit:] if _weekly_candidate_identity(item) not in selected_keys]
    for item in capped_candidates:
        possible_evidence.append(_possible_weekly_evidence_from_candidate(item, filter_reason="not_selected_due_to_candidate_cap"))
    possible_evidence_before_cap_count = len(possible_evidence)
    possible_evidence = _dedupe_possible_weekly_evidence(possible_evidence)
    possible_evidence.sort(
        key=lambda item: _possible_weekly_evidence_sort_key(item, plan_month=plan_month),
        reverse=True,
    )
    possible_evidence = possible_evidence[: _env_int("MYSQL_DEPT_PLAN_POSSIBLE_EVIDENCE_MAX_ITEMS", 24)]
    selected_candidates = [_compact_dept_plan_collaboration_candidate(item) for item in selected]
    return {
        "selected_candidates": selected_candidates,
        "selected_report_refs": [_dept_plan_report_ref(item) for item in selected],
        "possible_evidence": possible_evidence,
        "audit": {
            "raw_weekly_count": len(report_pool),
            "owner_report_count": owner_report_count,
            "department_filtered_count": department_compatible_count,
            "owner_match_count": 0,
            "owner_candidate_count": 0,
            "owner_cross_department_count": 0,
            "owner_cross_department_keyword_count": 0,
            "owner_cross_department_selected_count": 0,
            "keyword_match_count": keyword_match_count,
            "strong_keyword_match_count": strong_keyword_match_count,
            "business_keyword_match_count": business_keyword_match_count,
            "candidate_pool_count": len(audit_rows),
            "selected_before_cap_count": len(scored),
            "selected_count": len(selected_candidates),
            "owner_selected_count": 0,
            "non_owner_selected_count": len(selected_candidates),
            "selected_candidate_limit": selected_candidate_limit,
            "non_owner_candidate_limit": selected_candidate_limit,
            "owner_candidate_over_limit_count": 0,
            "capped_candidate_count": len(capped_candidates),
            "possible_evidence_count": len(possible_evidence),
            "possible_evidence_before_cap_count": possible_evidence_before_cap_count,
            "filtered_counts": filtered_counts,
            "owner_names": sorted(owner_names),
            "rule": (
                "负责人主证据只引用 owner_weekly_reports_pool 中的 weekly_reports 主表完整原文；"
                "非负责人协作辅助也来自 collaboration_weekly_reports_pool 主表原文，关键词只用于排序。"
            ),
            "sample_capped": [
                {
                    key: row.get(key)
                    for key in ("report_id", "user_name", "department", "report_date", "done_text", "specific_overlap_keywords")
                    if row.get(key) not in (None, "", [])
                }
                for row in capped_candidates[:3]
            ],
            "sample_filtered": [
                {
                    key: row.get(key)
                    for key in ("filter_reason", "report_id", "user_name", "department", "report_date", "done_text")
                    if row.get(key) not in (None, "", [])
                }
                for row in possible_evidence[:3]
            ],
        },
    }


def _build_dept_plan_collaboration_candidate(
    *,
    plan_profile: Mapping[str, Any],
    report: Mapping[str, Any],
    department_match: bool,
) -> dict[str, Any] | None:
    raw_text = str(report.get("周报原文") or "").strip()
    if not raw_text:
        return None
    metrics = _weekly_report_text_match_metrics(plan_profile=plan_profile, report_text=raw_text)
    overlap_keywords = metrics.get("overlap_keywords", [])
    strong_keywords = _strong_overlap_keywords(overlap_keywords if isinstance(overlap_keywords, list) else [])
    specific_keywords = _specific_business_overlap_keywords(strong_keywords)
    exact_phrase = bool(metrics.get("exact_phrase"))
    business_keyword_support = _has_specific_business_keyword_support(strong_keywords)
    strong_keyword_support = bool(
        exact_phrase
        or business_keyword_support
        or len(strong_keywords) >= 3
        or _safe_float(metrics.get("coverage"), 0.0) >= 0.35
    )
    has_keyword_support = bool(overlap_keywords or exact_phrase)
    priority_score = (
        _safe_float(metrics.get("score"), 0.0)
        + (_safe_float(metrics.get("coverage"), 0.0) * 0.8)
        + (0.5 if department_match else 0.0)
        + (0.8 if business_keyword_support else 0.0)
        + min(len(specific_keywords) * 0.2, 0.8)
    )
    confidence = "medium" if strong_keyword_support and department_match else ("low" if has_keyword_support else "low")
    this_week_raw = str(report.get("本周完成") or "").strip()
    done_text = this_week_raw or raw_text
    completion_signal, blockage_signal = _weekly_report_signals(raw_text)
    return {
        "report_id": report.get("report_id"),
        "done_text": _clip_text(done_text, 240),
        "report_date": report.get("日期"),
        "user_name": report.get("提交人"),
        "department": report.get("部门"),
        "item_type": "weekly_report",
        "owner_match": False,
        "department_match": department_match,
        "owner_cross_department": False,
        "strong_keyword_support": strong_keyword_support,
        "business_keyword_support": business_keyword_support,
        "candidate_confidence": confidence,
        "candidate_source": "collaboration_weekly_report_hint",
        "overlap_keywords": overlap_keywords[:12] if isinstance(overlap_keywords, list) else [],
        "item_overlap_keywords": [],
        "specific_overlap_keywords": specific_keywords[:12],
        "coverage": metrics.get("coverage"),
        "item_coverage": metrics.get("coverage"),
        "exact_phrase": exact_phrase,
        "item_exact_phrase": exact_phrase,
        "priority_score": round(priority_score, 4),
        "completion_signal": completion_signal,
        "blockage_signal": blockage_signal,
        "has_keyword_support": has_keyword_support,
        "source_doc_id": report.get("source_doc_id"),
        "source_chunk_id": report.get("source_chunk_id"),
    }


def _weekly_report_text_match_metrics(*, plan_profile: Mapping[str, Any], report_text: str) -> dict[str, Any]:
    plan_compact = str(plan_profile.get("compact") or "")
    plan_tokens = plan_profile.get("tokens")
    if not isinstance(plan_tokens, set):
        plan_tokens = set(plan_tokens or [])
    report_compact = _compact_text(report_text)
    report_tokens = _meaningful_keywords(report_text)
    if not plan_compact or not plan_tokens or not report_compact:
        return {"score": 0.0, "coverage": 0.0, "exact_phrase": False, "longest_overlap": 0, "overlap_keywords": []}
    overlap = sorted(plan_tokens & report_tokens, key=lambda item: (-len(item), item))
    exact_phrase = bool(plan_compact and plan_compact in report_compact)
    if not overlap and not exact_phrase:
        return {"score": 0.0, "coverage": 0.0, "exact_phrase": False, "longest_overlap": 0, "overlap_keywords": []}
    coverage = len(overlap) / max(1, len(plan_tokens))
    longest = max((len(item) for item in overlap), default=0)
    score = coverage + (0.45 if exact_phrase else 0.0) + min(longest / 20, 0.3)
    return {
        "score": round(score, 4),
        "coverage": round(coverage, 4),
        "exact_phrase": exact_phrase,
        "longest_overlap": longest,
        "overlap_keywords": overlap,
    }


def _weekly_report_signals(raw_text: str) -> tuple[str, str]:
    completion_signal = ""
    blockage_signal = ""
    if any(token in raw_text for token in ("完成", "已完成", "闭环", "归档", "上线", "验收", "下单", "发布")):
        completion_signal = "周报原文含完成/闭环类表述"
    if any(token in raw_text for token in ("未完成", "待", "卡点", "风险", "求助", "延期", "阻塞", "出入", "影响")):
        blockage_signal = "周报原文含未完成/待处理/风险类表述"
    return completion_signal, blockage_signal


def _compact_dept_plan_collaboration_candidate(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "report_id": item.get("report_id"),
        "done_text": item.get("done_text"),
        "report_date": item.get("report_date"),
        "user_name": item.get("user_name"),
        "department": item.get("department"),
        "item_type": "weekly_report",
        "owner_match": False,
        "department_match": bool(item.get("department_match")),
        "owner_cross_department": False,
        "strong_keyword_support": bool(item.get("strong_keyword_support")),
        "business_keyword_support": bool(item.get("business_keyword_support")),
        "candidate_confidence": item.get("candidate_confidence") or "low",
        "candidate_source": item.get("candidate_source") or "collaboration_weekly_report_hint",
        "overlap_keywords": item.get("overlap_keywords", [])[:8] if isinstance(item.get("overlap_keywords"), list) else [],
        "item_overlap_keywords": [],
        "specific_overlap_keywords": item.get("specific_overlap_keywords", [])[:8]
        if isinstance(item.get("specific_overlap_keywords"), list)
        else [],
        "coverage": item.get("coverage"),
        "item_coverage": item.get("item_coverage"),
        "exact_phrase": bool(item.get("exact_phrase")),
        "item_exact_phrase": bool(item.get("item_exact_phrase")),
        "completion_signal": item.get("completion_signal"),
        "blockage_signal": item.get("blockage_signal"),
        "source_doc_id": item.get("source_doc_id"),
        "source_chunk_id": item.get("source_chunk_id"),
    }


def _person_names_from_rows(rows: list[Mapping[str, Any]] | None, *, keys: tuple[str, ...]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for row in rows or []:
        if not isinstance(row, Mapping):
            continue
        for key in keys:
            value = _optional_str(row.get(key))
            if value is None:
                continue
            canonical = _canonical_person_name(value)
            if canonical in seen or not _looks_like_plain_person_name(canonical):
                continue
            seen.add(canonical)
            names.append(canonical)
    return names


def _normalize_known_person_names(values: list[str] | set[str] | tuple[str, ...] | None) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = _optional_str(value)
        if text is None:
            continue
        canonical = _canonical_person_name(text)
        if canonical in seen or not _looks_like_plain_person_name(canonical):
            continue
        seen.add(canonical)
        names.append(canonical)
    return names


def _dept_plan_owner_names(
    plans: list[Mapping[str, Any]],
    *,
    known_user_names: list[str] | set[str] | tuple[str, ...] | None = None,
) -> list[str]:
    seed_names = _normalize_known_person_names(known_user_names)
    for plan in plans:
        seed_names.extend(_split_owner_users_without_embedded_scan(plan.get("owner_user")))
    known_names = _normalize_known_person_names(seed_names)
    names: list[str] = []
    seen: set[str] = set()
    for plan in plans:
        for owner_name in _split_owner_users(plan.get("owner_user"), known_user_names=known_names):
            canonical = _canonical_person_name(owner_name)
            if canonical in seen:
                continue
            seen.add(canonical)
            names.append(canonical)
    return names


def _dept_plan_display_owner_user(
    raw_owner_user: Any,
    owner_names: list[str],
    *,
    known_user_names: list[str] | set[str] | tuple[str, ...] | None = None,
) -> tuple[str, dict[str, Any]]:
    original = _optional_str(raw_owner_user) or ""
    known_names = _normalize_known_person_names([*owner_names, *(known_user_names or [])])
    display_names: list[str] = []
    seen: set[str] = set()
    for owner_name in owner_names:
        canonical = _canonical_person_name(owner_name)
        if canonical in seen:
            continue
        seen.add(canonical)
        display_names.append(canonical)
    aliases: list[dict[str, str]] = []
    for raw_part in [part.strip() for part in re.split(r"[、,，;/；|+＋\s]+", original) if part.strip()]:
        canonical = _normalize_owner_name_part(raw_part, known_user_names=known_names)
        if canonical is None:
            embedded = _embedded_known_owner_names(raw_part, known_user_names=known_names)
            canonical = embedded[0] if len(embedded) == 1 else None
        if canonical is None:
            continue
        if raw_part != canonical:
            aliases.append({"original": raw_part, "canonical": canonical})

    invalid_owner_text = ""
    if not display_names and original:
        if original in _COLLECTIVE_OWNER_TOKENS:
            display = original
        else:
            invalid_owner_text = original
            display = "未明确负责人"
    else:
        display = "、".join(display_names)

    audit = {
        "original_owner_user": original,
        "display_owner_user": display,
        "normalized_owner_names": display_names,
        "alias_normalizations": aliases,
        "invalid_owner_text": invalid_owner_text,
    }
    return display, audit


def _employee_self_eval_query_names(user_names: list[str]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for user_name in user_names:
        text = _optional_str(user_name)
        if text is None:
            continue
        canonical = _canonical_person_name(text)
        variants = [canonical]
        variants.extend(_person_aliases().get(canonical, ()))
        for variant in variants:
            normalized = _optional_str(variant)
            if normalized is None or normalized in seen:
                continue
            seen.add(normalized)
            names.append(normalized)
    return names


def _split_owner_users(
    value: Any,
    *,
    known_user_names: list[str] | set[str] | tuple[str, ...] | None = None,
) -> list[str]:
    text = _optional_str(value)
    if text is None:
        return []
    known_names = _normalize_known_person_names(known_user_names)
    parts = [part.strip() for part in re.split(r"[、,，;/；|+＋\s]+", text) if part.strip()]
    deduped: list[str] = []
    seen: set[str] = set()
    for part in parts:
        canonical = _normalize_owner_name_part(part, known_user_names=known_names)
        if canonical is None:
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        deduped.append(canonical)
    if deduped:
        return deduped
    for embedded_name in _embedded_known_owner_names(text, known_user_names=known_names):
        if embedded_name in seen:
            continue
        seen.add(embedded_name)
        deduped.append(embedded_name)
    return deduped


def _split_owner_users_without_embedded_scan(value: Any) -> list[str]:
    text = _optional_str(value)
    if text is None:
        return []
    names: list[str] = []
    seen: set[str] = set()
    for part in [part.strip() for part in re.split(r"[、,，;/；|+＋\s]+", text) if part.strip()]:
        canonical = _normalize_owner_name_part(part)
        if canonical is None or canonical in seen:
            continue
        seen.add(canonical)
        names.append(canonical)
    return names


def _normalize_owner_name_part(
    value: str,
    *,
    known_user_names: list[str] | set[str] | tuple[str, ...] | None = None,
) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if len(text) > 4:
        return None
    if not all("\u4e00" <= char <= "\u9fff" for char in text):
        return None
    canonical = _canonical_person_name(text)
    if canonical != text and _looks_like_plain_person_name(canonical):
        return canonical
    known_names = _normalize_known_person_names(known_user_names)
    if text in known_names:
        return text
    title_name = _resolve_titled_owner_name(text, known_user_names=known_names)
    if title_name:
        return title_name
    close_name = _resolve_close_known_owner_name(text, known_user_names=known_names)
    if close_name:
        return close_name
    if not _looks_like_person_name(text):
        return None
    return text


def _looks_like_plain_person_name(value: str) -> bool:
    text = str(value or "").strip()
    return bool(
        text
        and 2 <= len(text) <= 4
        and all("\u4e00" <= char <= "\u9fff" for char in text)
        and not _looks_like_owner_department_token(text)
        and not _looks_like_owner_title_token(text)
        and text not in _OWNER_NOISE_TOKENS
        and not any(token in text for token in _OWNER_NOISE_TOKENS)
    )


def _looks_like_person_name(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if len(text) > 4:
        return False
    if not all("\u4e00" <= char <= "\u9fff" for char in text):
        return False
    if _looks_like_owner_department_token(text):
        return False
    if _canonical_person_name(text) != text:
        return True
    if _looks_like_owner_title_token(text):
        return False
    if text in _OWNER_NOISE_TOKENS:
        return False
    if any(token in text for token in _OWNER_NOISE_TOKENS):
        return False
    return True


def _looks_like_owner_department_token(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    department_tokens = set(_DEFAULT_DEPARTMENT_ALIASES)
    department_tokens.update(_DEFAULT_DEPARTMENT_CANONICAL_ALIASES.keys())
    for aliases in _DEFAULT_DEPARTMENT_CANONICAL_ALIASES.values():
        department_tokens.update(aliases)
    if text in department_tokens:
        return True
    return text.endswith(("部", "部门", "专业")) and text not in _person_aliases()


def _looks_like_owner_title_token(value: str) -> bool:
    text = str(value or "").strip()
    return any(text.endswith(suffix) and len(text[: -len(suffix)]) == 1 for suffix in _OWNER_TITLE_SUFFIXES)


def _resolve_titled_owner_name(
    value: str,
    *,
    known_user_names: list[str] | set[str] | tuple[str, ...] | None,
) -> str | None:
    text = str(value or "").strip()
    known_names = _normalize_known_person_names(known_user_names)
    if not known_names:
        return None
    for suffix in _OWNER_TITLE_SUFFIXES:
        if not text.endswith(suffix):
            continue
        stem = text[: -len(suffix)]
        if len(stem) != 1:
            continue
        candidates = [name for name in known_names if name.startswith(stem)]
        if len(candidates) == 1:
            return candidates[0]
    return None


def _resolve_close_known_owner_name(
    value: str,
    *,
    known_user_names: list[str] | set[str] | tuple[str, ...] | None,
) -> str | None:
    text = str(value or "").strip()
    known_names = _normalize_known_person_names(known_user_names)
    if not known_names or not _looks_like_person_name(text):
        return None
    candidates = [
        name
        for name in known_names
        if len(name) == len(text)
        and name != text
        and _person_name_distance(text, name) == 1
        and (name[0] == text[0] or name[-1] == text[-1])
    ]
    return candidates[0] if len(candidates) == 1 else None


def _embedded_known_owner_names(
    value: Any,
    *,
    known_user_names: list[str] | set[str] | tuple[str, ...] | None,
) -> list[str]:
    text = _optional_str(value)
    if text is None:
        return []
    known_names = _normalize_known_person_names(known_user_names)
    matches = [
        name
        for name in known_names
        if len(name) >= 2 and name in text
    ]
    return list(dict.fromkeys(sorted(matches, key=len, reverse=True)))


def _person_name_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if abs(len(left) - len(right)) > 1:
        return max(len(left), len(right))
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _build_dept_plan_weekly_index(weekly_completed: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    by_owner: dict[str, list[dict[str, Any]]] = {}
    by_department: dict[str | None, list[dict[str, Any]]] = {}
    for row in weekly_completed:
        prepared = dict(row)
        user_name = _optional_str(row.get("user_name"))
        canonical_user = _canonical_person_name(user_name) if user_name else None
        canonical_department = _canonical_department(row.get("department"))
        done_text = str(row.get("item_text") or row.get("evidence_text") or "").strip()
        evidence_text = str(row.get("evidence_text") or "")
        prepared["_canonical_user_name"] = canonical_user
        prepared["_canonical_department"] = canonical_department
        prepared["_dept_plan_done_text"] = done_text
        prepared["_dept_plan_done_item_compact"] = _compact_text(done_text)
        prepared["_dept_plan_done_compact"] = _compact_text(" ".join([done_text, evidence_text]))
        prepared["_dept_plan_done_item_tokens"] = _meaningful_keywords(done_text)
        prepared["_dept_plan_done_tokens"] = _meaningful_keywords(" ".join([done_text, evidence_text]))
        rows.append(prepared)
        if canonical_user:
            by_owner.setdefault(canonical_user, []).append(prepared)
        by_department.setdefault(canonical_department, []).append(prepared)
    return {
        "rows": rows,
        "by_owner": by_owner,
        "by_department": by_department,
    }


def _dept_plan_text_profile(plan_text: str) -> dict[str, Any]:
    return {
        "text": plan_text,
        "compact": _compact_text(plan_text),
        "tokens": _meaningful_keywords(plan_text),
    }


def _weekly_rows_for_department(*, weekly_index: Mapping[str, Any], plan_department: Any) -> list[dict[str, Any]]:
    rows = weekly_index.get("rows", [])
    if not isinstance(rows, list):
        return []
    plan_text = _optional_str(plan_department)
    if plan_text is None:
        return [row for row in rows if isinstance(row, dict)]
    plan_canonical = _canonical_department(plan_text)
    by_department = weekly_index.get("by_department", {})
    if not isinstance(by_department, Mapping):
        return [
            row
            for row in rows
            if isinstance(row, dict) and _departments_compatible(plan_department, row.get("department"))
        ]
    selected: list[dict[str, Any]] = []
    if plan_canonical in by_department:
        selected.extend(row for row in by_department.get(plan_canonical, []) if isinstance(row, dict))
    selected.extend(row for row in by_department.get(None, []) if isinstance(row, dict))
    if selected:
        return _dedupe_prepared_weekly_rows(selected)
    return [
        row
        for row in rows
        if isinstance(row, dict) and _departments_compatible(plan_department, row.get("department"))
    ]


def _weekly_rows_for_owners(*, weekly_index: Mapping[str, Any], owner_names: set[str]) -> list[dict[str, Any]]:
    by_owner = weekly_index.get("by_owner", {})
    if not isinstance(by_owner, Mapping) or not owner_names:
        return []
    rows: list[dict[str, Any]] = []
    for owner_name in owner_names:
        rows.extend(row for row in by_owner.get(owner_name, []) if isinstance(row, dict))
    return _dedupe_prepared_weekly_rows(rows)


def _dedupe_prepared_weekly_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in rows:
        identity = id(row)
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(row)
    return deduped


def _prepared_weekly_row_metrics(*, plan_profile: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    plan_compact = str(plan_profile.get("compact") or "")
    plan_tokens = plan_profile.get("tokens")
    if not isinstance(plan_tokens, set):
        plan_tokens = set(plan_tokens or [])
    done_text = str(row.get("_dept_plan_done_text") or row.get("item_text") or row.get("evidence_text") or "").strip()
    evidence_text = str(row.get("evidence_text") or "")
    done_item_compact = str(row.get("_dept_plan_done_item_compact") or _compact_text(done_text))
    done_compact = str(row.get("_dept_plan_done_compact") or _compact_text(" ".join([done_text, evidence_text])))
    done_item_tokens = row.get("_dept_plan_done_item_tokens")
    if not isinstance(done_item_tokens, set):
        done_item_tokens = _meaningful_keywords(done_text)
    done_tokens = row.get("_dept_plan_done_tokens")
    if not isinstance(done_tokens, set):
        done_tokens = _meaningful_keywords(" ".join([done_text, evidence_text]))
    if not plan_compact or not plan_tokens:
        return {
            "overlap_keywords": [],
            "item_overlap_keywords": [],
            "coverage": 0.0,
            "item_coverage": 0.0,
            "exact_phrase": False,
            "item_exact_phrase": False,
            "longest_overlap": 0,
            "item_longest_overlap": 0,
            "score": 0.0,
        }
    overlap = sorted(plan_tokens & done_tokens, key=lambda item: (-len(item), item))
    item_overlap = sorted(plan_tokens & done_item_tokens, key=lambda item: (-len(item), item))
    exact_phrase = bool(plan_compact and plan_compact in done_compact)
    item_exact_phrase = bool(plan_compact and plan_compact in done_item_compact)
    if not overlap and not exact_phrase:
        return {
            "overlap_keywords": [],
            "item_overlap_keywords": [],
            "coverage": 0.0,
            "item_coverage": 0.0,
            "exact_phrase": False,
            "item_exact_phrase": False,
            "longest_overlap": 0,
            "item_longest_overlap": 0,
            "score": 0.0,
        }
    coverage = len(overlap) / max(1, len(plan_tokens))
    item_coverage = len(item_overlap) / max(1, len(plan_tokens))
    longest = max((len(item) for item in overlap), default=0)
    item_longest = max((len(item) for item in item_overlap), default=0)
    score = (
        coverage
        + item_coverage * 0.8
        + (0.45 if exact_phrase else 0.0)
        + (0.35 if item_exact_phrase else 0.0)
        + min(longest / 20, 0.3)
        + min(item_longest / 20, 0.25)
    )
    return {
        "score": round(score, 4),
        "coverage": round(coverage, 4),
        "item_coverage": round(item_coverage, 4),
        "exact_phrase": exact_phrase,
        "item_exact_phrase": item_exact_phrase,
        "longest_overlap": longest,
        "item_longest_overlap": item_longest,
        "overlap_keywords": overlap,
        "item_overlap_keywords": item_overlap,
    }


def _build_dept_plan_weekly_candidate(
    *,
    plan_text: str,
    plan_profile: Mapping[str, Any],
    owner_names: set[str],
    row: Mapping[str, Any],
    department_match: bool,
    owner_cross_department: bool,
) -> tuple[float, dict[str, Any], dict[str, Any]] | None:
    user_name = _optional_str(row.get("user_name"))
    done_text = str(row.get("_dept_plan_done_text") or row.get("item_text") or row.get("evidence_text") or "").strip()
    if not done_text and not user_name:
        return None

    match_metrics = _prepared_weekly_row_metrics(plan_profile=plan_profile, row=row)
    overlap_keywords = match_metrics.get("overlap_keywords", [])
    item_overlap_keywords = match_metrics.get("item_overlap_keywords", [])
    exact_phrase = bool(match_metrics.get("exact_phrase"))
    item_exact_phrase = bool(match_metrics.get("item_exact_phrase"))
    has_keyword_support = bool(overlap_keywords or item_overlap_keywords or exact_phrase or item_exact_phrase)
    strong_overlap_keywords = _strong_overlap_keywords(overlap_keywords)
    strong_item_overlap_keywords = _strong_overlap_keywords(item_overlap_keywords)
    item_business_keyword_support = _has_specific_business_keyword_support(item_overlap_keywords)
    evidence_business_keyword_support = _has_specific_business_keyword_support(overlap_keywords)
    business_keyword_support = item_business_keyword_support or (
        bool(item_exact_phrase)
        or (not done_text and evidence_business_keyword_support)
    )
    clear_item_keyword_support = (
        item_exact_phrase
        or item_business_keyword_support
        or len(strong_item_overlap_keywords) >= 3
        or _safe_float(match_metrics.get("item_coverage"), 0.0) >= 0.35
    )
    evidence_only_keyword_support = (
        not done_text
        and (
            exact_phrase
            or (
                len(strong_overlap_keywords) >= 3
                and _safe_float(match_metrics.get("coverage"), 0.0) >= 0.25
            )
        )
    )
    specific_overlap_keywords = _specific_business_overlap_keywords(
        strong_item_overlap_keywords if done_text else [*strong_overlap_keywords, *strong_item_overlap_keywords]
    )
    strong_keyword_support = (
        clear_item_keyword_support
        or evidence_only_keyword_support
    )
    canonical_user_name = row.get("_canonical_user_name") or (_canonical_person_name(user_name) if user_name else None)
    owner_match = bool(canonical_user_name and canonical_user_name in owner_names)

    priority_score = (
        (100.0 if owner_match else 0.0)
        + _safe_float(match_metrics.get("score"), 0.0)
        + (_safe_float(match_metrics.get("item_coverage"), 0.0) * 0.8)
        + (_safe_float(match_metrics.get("coverage"), 0.0) * 0.5)
        + (0.4 if exact_phrase else 0.0)
        + (0.3 if item_exact_phrase else 0.0)
        + (1.2 if item_business_keyword_support else 0.0)
        + (0.25 if evidence_business_keyword_support and not item_business_keyword_support else 0.0)
        + min(len(specific_overlap_keywords) * 0.18, 0.9)
    )

    selected = owner_match or strong_keyword_support
    filter_reason = None if selected else "no_owner_or_keyword_support"
    if owner_cross_department and clear_item_keyword_support:
        candidate_source = "owner_cross_department_keyword_trace"
        candidate_confidence = "medium"
    elif owner_cross_department:
        candidate_source = "owner_cross_department_trace"
        candidate_confidence = "low"
    elif owner_match and clear_item_keyword_support:
        candidate_source = "owner_keyword_trace"
        candidate_confidence = "high"
    elif owner_match and has_keyword_support:
        candidate_source = "owner_trace"
        candidate_confidence = "medium"
    elif owner_match:
        candidate_source = "owner_trace"
        candidate_confidence = "medium"
    else:
        candidate_source = "keyword_support_fallback"
        candidate_confidence = "low"
    candidate = {
        "done_text": done_text,
        "report_date": row.get("report_date"),
        "user_name": user_name,
        "department": row.get("department"),
        "item_type": row.get("item_type"),
        "owner_match": owner_match,
        "department_match": department_match,
        "owner_cross_department": owner_cross_department,
        "strong_keyword_support": strong_keyword_support,
        "business_keyword_support": business_keyword_support,
        "item_business_keyword_support": item_business_keyword_support,
        "candidate_confidence": candidate_confidence,
        "candidate_source": candidate_source,
        "overlap_keywords": overlap_keywords[:12] if isinstance(overlap_keywords, list) else [],
        "item_overlap_keywords": item_overlap_keywords[:12] if isinstance(item_overlap_keywords, list) else [],
        "specific_overlap_keywords": specific_overlap_keywords[:12],
        "coverage": match_metrics.get("coverage"),
        "item_coverage": match_metrics.get("item_coverage"),
        "exact_phrase": exact_phrase,
        "item_exact_phrase": item_exact_phrase,
        "evidence_text": _clip_text(row.get("evidence_text"), 300),
        "source_doc_id": row.get("source_doc_id"),
        "source_chunk_id": row.get("source_chunk_id"),
    }
    audit = {
        "user_name": user_name,
        "department": row.get("department"),
        "report_date": row.get("report_date"),
        "item_type": row.get("item_type"),
        "owner_match": owner_match,
        "department_match": department_match,
        "owner_cross_department": owner_cross_department,
        "has_keyword_support": has_keyword_support,
        "strong_keyword_support": strong_keyword_support,
        "business_keyword_support": business_keyword_support,
        "item_business_keyword_support": item_business_keyword_support,
        "selected": selected,
        "filter_reason": filter_reason,
        "candidate_confidence": candidate_confidence,
        "candidate_source": candidate_source,
        "overlap_keywords": candidate["overlap_keywords"],
        "item_overlap_keywords": candidate["item_overlap_keywords"],
        "specific_overlap_keywords": candidate["specific_overlap_keywords"],
        "done_text": _clip_text(done_text, 160),
    }
    return priority_score, candidate, audit


def _dept_plan_weekly_candidates(*, plan: Mapping[str, Any], weekly_completed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _dept_plan_weekly_match_result(plan=plan, weekly_completed=weekly_completed)["selected_candidates"]


def _dept_plan_weekly_match_result(
    *,
    plan: Mapping[str, Any],
    weekly_completed: list[dict[str, Any]],
    weekly_index: Mapping[str, Any] | None = None,
    owner_weekly_reports: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    plan_department = plan.get("department")
    plan_month = str(plan.get("month") or "").strip()
    plan_text = str(plan.get("plan_text") or "").strip()
    plan_profile = _dept_plan_text_profile(plan_text)
    weekly_index = weekly_index or _build_dept_plan_weekly_index(weekly_completed)
    known_user_names = _normalize_known_person_names(
        [
            *_person_names_from_rows(weekly_completed, keys=("user_name", "提交人")),
            *_person_names_from_rows(owner_weekly_reports or [], keys=("user_name", "提交人")),
        ]
    )
    owner_names = set(_split_owner_users(plan.get("owner_user"), known_user_names=known_user_names))
    indexed_rows = weekly_index.get("rows", []) if isinstance(weekly_index, Mapping) else []
    raw_weekly_count = len(indexed_rows) if isinstance(indexed_rows, list) else len(weekly_completed)
    department_rows = _weekly_rows_for_department(weekly_index=weekly_index, plan_department=plan_department)
    owner_rows = _weekly_rows_for_owners(weekly_index=weekly_index, owner_names=owner_names)
    owner_report_rows = (
        _dept_plan_owner_report_rows_for_evidence(owner_weekly_reports, owner_names=owner_names)
        if owner_weekly_reports is not None
        else _dept_plan_owner_weekly_reports_for_evidence(owner_rows)
    )
    department_row_ids = {id(row) for row in department_rows}
    owner_row_ids = {id(row) for row in owner_rows}
    owner_cross_department_rows = [
        row
        for row in owner_rows
        if id(row) not in department_row_ids and not _departments_compatible(plan_department, row.get("department"))
    ]
    owner_cross_department_ids = {id(row) for row in owner_cross_department_rows}
    candidate_rows = _dedupe_prepared_weekly_rows([*department_rows, *owner_rows])
    scored_candidates: list[dict[str, Any]] = []
    possible_evidence: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    department_filtered_count = len(department_rows)
    owner_match_count = 0
    owner_cross_department_count = 0
    owner_cross_department_keyword_count = 0
    owner_cross_department_selected_count = 0
    keyword_match_count = 0
    strong_keyword_match_count = 0
    business_keyword_match_count = 0
    selected_before_cap_count = 0
    filtered_counts: dict[str, int] = {}
    department_mismatch_filtered_count = max(0, raw_weekly_count - len(department_row_ids | owner_cross_department_ids))
    if department_mismatch_filtered_count:
        filtered_counts["department_mismatch"] = department_mismatch_filtered_count
    for row in candidate_rows:
        department_match = id(row) in department_row_ids or _departments_compatible(plan_department, row.get("department"))
        owner_cross_department = id(row) in owner_cross_department_ids
        scored_candidate = _build_dept_plan_weekly_candidate(
            plan_text=plan_text,
            plan_profile=plan_profile,
            owner_names=owner_names,
            row=row,
            department_match=department_match,
            owner_cross_department=owner_cross_department,
        )
        if scored_candidate is None:
            continue
        priority_score, candidate, audit = scored_candidate
        if audit.get("owner_match"):
            owner_match_count += 1
        if audit.get("owner_cross_department"):
            owner_cross_department_count += 1
            if audit.get("has_keyword_support"):
                owner_cross_department_keyword_count += 1
        if audit.get("has_keyword_support"):
            keyword_match_count += 1
        if audit.get("strong_keyword_support"):
            strong_keyword_match_count += 1
        if audit.get("business_keyword_support"):
            business_keyword_match_count += 1
        audit["priority_score"] = round(priority_score, 4)
        audit_rows.append(audit)
        if audit.get("filter_reason"):
            reason = str(audit["filter_reason"])
            filtered_counts[reason] = filtered_counts.get(reason, 0) + 1
            possible_evidence.append(_possible_weekly_evidence_from_audit(audit))
            continue
        selected_before_cap_count += 1
        candidate["priority_score"] = priority_score
        scored_candidates.append(candidate)

    selected_candidate_limit = max(1, _env_int("MYSQL_DEPT_PLAN_WEEKLY_CANDIDATE_MAX_ITEMS", 128))
    owner_scored_candidates = sorted(
        [item for item in scored_candidates if item.get("owner_match")],
        key=lambda item: _dept_plan_candidate_sort_key(item, plan_month=plan_month),
        reverse=True,
    )
    non_owner_scored_candidates = sorted(
        [item for item in scored_candidates if not item.get("owner_match")],
        key=lambda item: _dept_plan_candidate_sort_key(item, plan_month=plan_month),
        reverse=True,
    )
    non_owner_candidate_limit = max(0, selected_candidate_limit - len(owner_scored_candidates))
    selected = [*owner_scored_candidates, *non_owner_scored_candidates[:non_owner_candidate_limit]]
    selected.sort(
        key=lambda item: _dept_plan_candidate_sort_key(item, plan_month=plan_month),
        reverse=True,
    )
    owner_cross_department_selected_count = sum(1 for item in selected if item.get("owner_cross_department"))
    owner_selected_count = sum(1 for item in selected if item.get("owner_match"))
    non_owner_selected_count = len(selected) - owner_selected_count
    selected_keys = {_weekly_candidate_identity(item) for item in selected}
    capped_candidates = [
        item
        for item in non_owner_scored_candidates
        if _weekly_candidate_identity(item) not in selected_keys
    ]
    for item in sorted(
        capped_candidates,
        key=lambda row: _dept_plan_candidate_sort_key(row, plan_month=plan_month),
        reverse=True,
    ):
        possible_evidence.append(
            _possible_weekly_evidence_from_candidate(
                item,
                filter_reason="not_selected_due_to_candidate_cap",
            )
        )
    possible_evidence_before_cap_count = len(possible_evidence)
    possible_evidence = _dedupe_possible_weekly_evidence(possible_evidence)
    possible_evidence_before_cap_count = len(possible_evidence)
    possible_evidence.sort(
        key=lambda item: _possible_weekly_evidence_sort_key(item, plan_month=plan_month),
        reverse=True,
    )
    possible_evidence = possible_evidence[: _env_int("MYSQL_DEPT_PLAN_POSSIBLE_EVIDENCE_MAX_ITEMS", 24)]
    selected_candidates = [
        {
            "done_text": item.get("done_text"),
            "report_date": item.get("report_date"),
            "user_name": item.get("user_name"),
            "department": item.get("department"),
            "item_type": item.get("item_type"),
            "owner_match": bool(item.get("owner_match")),
            "department_match": bool(item.get("department_match")),
            "owner_cross_department": bool(item.get("owner_cross_department")),
            "strong_keyword_support": bool(item.get("strong_keyword_support")),
            "business_keyword_support": bool(item.get("business_keyword_support")),
            "candidate_confidence": item.get("candidate_confidence") or ("high" if item.get("owner_match") else "low"),
            "candidate_source": item.get("candidate_source") or ("owner_trace" if item.get("owner_match") else "recency_fallback"),
            "overlap_keywords": item.get("overlap_keywords", [])[:12]
            if isinstance(item.get("overlap_keywords"), list)
            else [],
            "item_overlap_keywords": item.get("item_overlap_keywords", [])[:12]
            if isinstance(item.get("item_overlap_keywords"), list)
            else [],
            "specific_overlap_keywords": item.get("specific_overlap_keywords", [])[:12]
            if isinstance(item.get("specific_overlap_keywords"), list)
            else [],
            "coverage": item.get("coverage"),
            "item_coverage": item.get("item_coverage"),
            "exact_phrase": bool(item.get("exact_phrase")),
            "item_exact_phrase": bool(item.get("item_exact_phrase")),
            "evidence_text": _clip_text(item.get("evidence_text"), 300),
            "source_doc_id": item.get("source_doc_id"),
            "source_chunk_id": item.get("source_chunk_id"),
        }
        for item in selected
    ]
    return {
        "selected_candidates": selected_candidates,
        "possible_evidence": possible_evidence,
        "owner_weekly_reports": owner_report_rows,
        "audit": {
            "raw_weekly_count": raw_weekly_count,
            "department_filtered_count": department_filtered_count,
            "owner_match_count": owner_match_count,
            "owner_candidate_count": len(owner_rows),
            "owner_cross_department_count": owner_cross_department_count,
            "owner_cross_department_keyword_count": owner_cross_department_keyword_count,
            "owner_cross_department_selected_count": owner_cross_department_selected_count,
            "keyword_match_count": keyword_match_count,
            "strong_keyword_match_count": strong_keyword_match_count,
            "business_keyword_match_count": business_keyword_match_count,
            "candidate_pool_count": len(candidate_rows),
            "selected_before_cap_count": selected_before_cap_count,
            "selected_count": len(selected_candidates),
            "owner_selected_count": owner_selected_count,
            "non_owner_selected_count": non_owner_selected_count,
            "selected_candidate_limit": selected_candidate_limit,
            "non_owner_candidate_limit": non_owner_candidate_limit,
            "owner_candidate_over_limit_count": max(0, len(owner_scored_candidates) - selected_candidate_limit),
            "capped_candidate_count": len(capped_candidates),
            "possible_evidence_count": len(possible_evidence),
            "possible_evidence_before_cap_count": possible_evidence_before_cap_count,
            "filtered_counts": filtered_counts,
            "owner_names": sorted(owner_names),
            "rule": (
                "负责人完整周报从 weekly_reports 主表按负责人和日期范围全量取出给 LLM 语义判断；"
                "非负责人协作周报按关键词辅助排序并受候选数量上限控制。"
            ),
            "sample_owner_cross_department": [
                {
                    key: row.get(key)
                    for key in (
                        "user_name",
                        "department",
                        "report_date",
                        "done_text",
                        "candidate_source",
                        "overlap_keywords",
                        "specific_overlap_keywords",
                    )
                    if row.get(key) not in (None, "", [])
                }
                for row in audit_rows
                if row.get("owner_cross_department")
            ][:3],
            "sample_capped": [
                {
                    key: row.get(key)
                    for key in (
                        "user_name",
                        "department",
                        "report_date",
                        "done_text",
                        "candidate_source",
                        "overlap_keywords",
                        "specific_overlap_keywords",
                    )
                    if row.get(key) not in (None, "", [])
                }
                for row in sorted(
                    capped_candidates,
                    key=lambda item: _dept_plan_candidate_sort_key(item, plan_month=plan_month),
                    reverse=True,
                )
            ][:3],
            "sample_filtered": [
                {
                    key: row.get(key)
                    for key in ("filter_reason", "user_name", "department", "report_date", "done_text")
                    if row.get(key) not in (None, "")
                }
                for row in audit_rows
                if row.get("filter_reason")
            ][:3],
        },
    }


def _dept_plan_owner_weekly_reports_for_evidence(owner_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in owner_rows:
        item_text = str(row.get("_dept_plan_done_text") or row.get("item_text") or "").strip()
        raw_text = str(row.get("evidence_text") or item_text).strip()
        if not raw_text:
            continue
        identity = (
            row.get("report_date"),
            row.get("user_name"),
            row.get("department"),
            row.get("item_type"),
            item_text,
            raw_text,
            row.get("source_doc_id"),
            row.get("source_chunk_id"),
        )
        if identity in seen:
            continue
        seen.add(identity)
        reports.append(
            {
                "report_id": f"r{len(reports) + 1}",
                "日期": row.get("report_date") or "",
                "提交人": row.get("user_name") or "",
                "部门": row.get("department") or "",
                "事项类型": row.get("item_type") or "",
                "周报事项": item_text,
                "周报原文": raw_text,
                "source_doc_id": row.get("source_doc_id"),
                "source_chunk_id": row.get("source_chunk_id"),
            }
        )
    reports.sort(key=lambda item: (str(item.get("日期") or ""), str(item.get("report_id") or "")))
    for index, report in enumerate(reports, start=1):
        report["report_id"] = f"r{index}"
    return reports


def _dept_plan_owner_report_rows_for_evidence(
    owner_weekly_reports: list[dict[str, Any]],
    *,
    owner_names: set[str],
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    if not owner_names:
        return reports
    for row in owner_weekly_reports:
        user_name = _optional_str(row.get("user_name"))
        if not user_name or _canonical_person_name(user_name) not in owner_names:
            continue
        raw_text = _format_weekly_report_raw_text(row)
        if not raw_text:
            continue
        identity = (
            row.get("report_date"),
            user_name,
            row.get("department"),
            raw_text,
            row.get("source_doc_id"),
            row.get("source_chunk_id"),
        )
        if identity in seen:
            continue
        seen.add(identity)
        reports.append(
            {
                "report_id": f"r{len(reports) + 1}",
                "日期": row.get("report_date") or "",
                "提交人": user_name,
                "部门": row.get("department") or "",
                "事项类型": "weekly_report",
                "周报事项": "",
                "周报原文": raw_text,
                "source_doc_id": row.get("source_doc_id"),
                "source_chunk_id": row.get("source_chunk_id"),
            }
        )
    reports.sort(
        key=lambda item: (
            str(item.get("日期") or ""),
            str(item.get("提交人") or ""),
            str(item.get("report_id") or ""),
        )
    )
    for index, report in enumerate(reports, start=1):
        report["report_id"] = f"r{index}"
    return reports


def _format_weekly_report_raw_text(row: Mapping[str, Any]) -> str:
    sections: list[str] = []
    this_week = _optional_str(row.get("this_week_raw"))
    if this_week:
        sections.append(f"本周完成:\n{this_week}")
    next_week = _optional_str(row.get("next_week_raw"))
    if next_week:
        sections.append(f"下周计划:\n{next_week}")
    risk_and_help = _optional_str(row.get("risk_and_help"))
    if risk_and_help:
        sections.append(f"风险/求助:\n{risk_and_help}")
    if sections:
        return "\n\n".join(sections).strip()
    return str(row.get("evidence_text") or row.get("item_text") or "").strip()


def _weekly_candidate_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("user_name"),
        row.get("department"),
        row.get("report_date"),
        row.get("item_type"),
        row.get("done_text"),
        row.get("source_doc_id"),
        row.get("source_chunk_id"),
    )


def _weekly_candidate_has_keyword_support(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("overlap_keywords")
        or row.get("item_overlap_keywords")
        or row.get("exact_phrase")
        or row.get("item_exact_phrase")
    )


def _weekly_candidate_has_strong_keyword_support(row: Mapping[str, Any]) -> bool:
    if row.get("strong_keyword_support") or row.get("business_keyword_support"):
        return True
    overlap_keywords = row.get("overlap_keywords")
    item_overlap_keywords = row.get("item_overlap_keywords")
    strong_keywords = _strong_overlap_keywords(overlap_keywords if isinstance(overlap_keywords, list) else [])
    strong_item_keywords = _strong_overlap_keywords(item_overlap_keywords if isinstance(item_overlap_keywords, list) else [])
    return _has_specific_business_keyword_support(strong_keywords, strong_item_keywords)


def _weekly_candidate_in_plan_month(row: Mapping[str, Any], *, plan_month: str) -> bool:
    month = plan_month[:7]
    if not re.match(r"^\d{4}-\d{2}$", month):
        return True
    return str(row.get("report_date") or "").startswith(month)


def _dept_plan_candidate_sort_key(row: Mapping[str, Any], *, plan_month: str) -> tuple[Any, ...]:
    owner_match = bool(row.get("owner_match"))
    keyword_support = _weekly_candidate_has_keyword_support(row)
    strong_keyword_support = _weekly_candidate_has_strong_keyword_support(row)
    in_plan_month = _weekly_candidate_in_plan_month(row, plan_month=plan_month)
    department_match = bool(row.get("department_match"))
    owner_cross_department = bool(row.get("owner_cross_department"))
    specific_count = len(row.get("specific_overlap_keywords") or []) if isinstance(row.get("specific_overlap_keywords"), list) else 0
    if owner_match and strong_keyword_support and in_plan_month and department_match:
        evidence_tier = 90
    elif owner_match and strong_keyword_support and in_plan_month:
        evidence_tier = 88
    elif owner_match and strong_keyword_support and department_match:
        evidence_tier = 84
    elif owner_match and strong_keyword_support:
        evidence_tier = 82
    elif strong_keyword_support and department_match and in_plan_month:
        evidence_tier = 76
    elif strong_keyword_support and department_match:
        evidence_tier = 72
    elif owner_match and keyword_support and in_plan_month and department_match:
        evidence_tier = 68
    elif owner_match and keyword_support and in_plan_month:
        evidence_tier = 66
    elif owner_match and in_plan_month and department_match:
        evidence_tier = 64
    elif owner_match and in_plan_month and owner_cross_department:
        evidence_tier = 60
    elif owner_match and department_match:
        evidence_tier = 56
    elif owner_match:
        evidence_tier = 52
    elif strong_keyword_support:
        evidence_tier = 40
    elif keyword_support:
        evidence_tier = 20
    else:
        evidence_tier = 0
    return (
        evidence_tier,
        specific_count,
        _safe_float(row.get("priority_score"), 0.0),
        str(row.get("report_date") or ""),
    )


def _possible_weekly_evidence_sort_key(row: Mapping[str, Any], *, plan_month: str) -> tuple[Any, ...]:
    reason = str(row.get("filter_reason") or "")
    reason_priority = {
        "not_selected_due_to_candidate_cap": 4,
        "owner_cross_department_possible": 3,
        "no_owner_or_keyword_support": 1,
    }.get(reason, 0)
    return (
        reason_priority,
        bool(row.get("owner_match")),
        bool(row.get("owner_cross_department")),
        _weekly_candidate_has_strong_keyword_support(row),
        _weekly_candidate_has_keyword_support(row),
        _weekly_candidate_in_plan_month(row, plan_month=plan_month),
        _safe_float(row.get("priority_score"), 0.0),
        str(row.get("report_date") or ""),
    )


def _possible_weekly_evidence_from_candidate(row: Mapping[str, Any], *, filter_reason: str) -> dict[str, Any]:
    return {
        "done_text": _clip_text(row.get("done_text"), 180),
        "report_date": row.get("report_date"),
        "user_name": row.get("user_name"),
        "department": row.get("department"),
        "item_type": row.get("item_type"),
        "filter_reason": filter_reason,
        "owner_match": bool(row.get("owner_match")),
        "department_match": bool(row.get("department_match")),
        "owner_cross_department": bool(row.get("owner_cross_department")),
        "strong_keyword_support": bool(row.get("strong_keyword_support")),
        "business_keyword_support": bool(row.get("business_keyword_support")),
        "candidate_confidence": row.get("candidate_confidence") or "low",
        "candidate_source": row.get("candidate_source") or "possible_weekly_evidence",
        "overlap_keywords": row.get("overlap_keywords", [])[:8] if isinstance(row.get("overlap_keywords"), list) else [],
        "specific_overlap_keywords": row.get("specific_overlap_keywords", [])[:8]
        if isinstance(row.get("specific_overlap_keywords"), list)
        else [],
        "priority_score": row.get("priority_score"),
    }


def _possible_weekly_evidence_from_audit(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "done_text": _clip_text(row.get("done_text"), 180),
        "report_date": row.get("report_date"),
        "user_name": row.get("user_name"),
        "department": row.get("department"),
        "item_type": row.get("item_type"),
        "filter_reason": row.get("filter_reason") or "filtered_from_strong_candidates",
        "owner_match": bool(row.get("owner_match")),
        "department_match": bool(row.get("department_match")),
        "owner_cross_department": bool(row.get("owner_cross_department")),
        "strong_keyword_support": bool(row.get("strong_keyword_support")),
        "business_keyword_support": bool(row.get("business_keyword_support")),
        "candidate_confidence": row.get("candidate_confidence") or "low",
        "candidate_source": row.get("candidate_source") or "possible_weekly_evidence",
        "overlap_keywords": row.get("overlap_keywords", [])[:8] if isinstance(row.get("overlap_keywords"), list) else [],
        "specific_overlap_keywords": row.get("specific_overlap_keywords", [])[:8]
        if isinstance(row.get("specific_overlap_keywords"), list)
        else [],
        "priority_score": row.get("priority_score"),
    }


def _dedupe_possible_weekly_evidence(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for item in items:
        identity = _weekly_candidate_identity(item)
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(item)
    return deduped


def _dept_plan_owner_self_eval_rows(
    *,
    plan: Mapping[str, Any],
    owner_self_eval_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    owner_names = _split_owner_users(
        plan.get("owner_user"),
        known_user_names=_person_names_from_rows(owner_self_eval_items, keys=("user_name",)),
    )
    if not owner_names:
        return []
    rows: list[dict[str, Any]] = []
    for item in owner_self_eval_items:
        if not isinstance(item, Mapping):
            continue
        item_user_name = _optional_str(item.get("user_name"))
        if not any(_same_user(owner_name, item_user_name) for owner_name in owner_names):
            continue
        rows.append(dict(item))
    rows.sort(
        key=lambda item: (
            str(item.get("user_name") or ""),
            _safe_int(item.get("source_row"), 999999),
            _safe_int(item.get("item_id"), 999999),
        )
    )
    return rows


def _dept_plan_owner_self_eval_reports(
    *,
    plan: Mapping[str, Any],
    owner_self_eval_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for row in _dept_plan_owner_self_eval_rows(plan=plan, owner_self_eval_items=owner_self_eval_items):
        report_id = row.get("report_id") or (row.get("doc_id"), row.get("user_name"), row.get("month"))
        if report_id in seen:
            continue
        seen.add(report_id)
        reports.append(
            {
                "report_id": row.get("report_id"),
                "doc_id": row.get("doc_id"),
                "month": row.get("month"),
                "user_name": row.get("user_name"),
                "department": row.get("department"),
                "position": row.get("position"),
                "sheet_name": row.get("report_sheet_name"),
                "work_avg_completion_rate": row.get("work_avg_completion_rate"),
                "management_avg_score": row.get("management_avg_score"),
                "leader_rating_score": row.get("leader_rating_score"),
                "admin_rating_score": row.get("admin_rating_score"),
            }
        )
    return reports


def _dept_plan_owner_self_eval_items(
    *,
    plan: Mapping[str, Any],
    owner_self_eval_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = _dept_plan_owner_self_eval_rows(plan=plan, owner_self_eval_items=owner_self_eval_items)
    max_items = _env_int("MYSQL_DEPT_PLAN_OWNER_SELF_EVAL_MAX_ITEMS_PER_PLAN", 80)
    items: list[dict[str, Any]] = []
    for row in rows:
        if row.get("item_id") is None:
            continue
        items.append(
            {
                "report_id": row.get("report_id"),
                "item_id": row.get("item_id"),
                "doc_id": row.get("doc_id"),
                "month": row.get("month"),
                "user_name": row.get("user_name"),
                "department": row.get("department"),
                "position": row.get("position"),
                "section": row.get("section"),
                "item_type": row.get("item_type"),
                "item_text": row.get("item_text"),
                "plan_text": row.get("plan_text"),
                "result_text": row.get("result_text"),
                "completion_time": row.get("completion_time"),
                "completion_rate": row.get("completion_rate"),
                "contact_user": row.get("contact_user"),
                "unfinished_text": row.get("unfinished_text"),
                "unresolved_text": row.get("unresolved_text"),
                "reason_text": row.get("reason_text"),
                "effect_text": row.get("effect_text"),
                "source_sheet": row.get("source_sheet"),
                "source_row": row.get("source_row"),
                "evidence_text": _clip_text(row.get("evidence_text"), 400),
                "source_doc_id": row.get("doc_id"),
                "source_chunk_id": row.get("evidence_chunk_id"),
            }
        )
    return items[:max_items]


def _dept_plan_missing_owner_self_eval_users(
    *,
    plan: Mapping[str, Any],
    owner_self_eval_items: list[dict[str, Any]],
) -> list[str]:
    owner_names = _split_owner_users(
        plan.get("owner_user"),
        known_user_names=_person_names_from_rows(owner_self_eval_items, keys=("user_name",)),
    )
    if not owner_names:
        return []
    found_users = {
        _canonical_person_name(str(item.get("user_name") or ""))
        for item in owner_self_eval_items
        if isinstance(item, Mapping) and item.get("user_name")
    }
    return [
        owner_name
        for owner_name in owner_names
        if _canonical_person_name(owner_name) not in found_users
    ]


def _dept_plan_self_eval_candidates(*, plan: Mapping[str, Any], self_eval_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    plan_department = plan.get("department")
    candidate_rows = [
        {
            "item_text": item.get("item_text"),
            "evidence_text": item.get("evidence_text"),
            "report_date": item.get("month"),
            "user_name": None,
            "department": item.get("department"),
            "item_type": item.get("item_type"),
            "source_doc_id": item.get("doc_id"),
            "source_chunk_id": item.get("evidence_chunk_id"),
        }
        for item in self_eval_items
        if _departments_compatible(plan_department, item.get("department"))
    ]
    scored = _score_done_matches(str(plan.get("plan_text") or ""), candidate_rows)
    scored.sort(
        key=lambda item: (
            _self_eval_type_priority(item.get("item_type")),
            _safe_float(item.get("score"), 0.0),
            _safe_float(item.get("item_coverage"), 0.0),
        ),
        reverse=True,
    )
    return [
        {
            "item_type": item.get("item_type"),
            "item_text": item.get("done_text"),
            "month": item.get("report_date"),
            "department": item.get("department"),
            "overlap_keywords": item.get("overlap_keywords", [])[:12]
            if isinstance(item.get("overlap_keywords"), list)
            else [],
            "evidence_text": _clip_text(item.get("evidence_text"), 300),
            "source_doc_id": item.get("source_doc_id"),
            "source_chunk_id": item.get("source_chunk_id"),
        }
        for item in scored[:5]
    ]


def _self_eval_type_priority(value: Any) -> int:
    return {
        "achievement": 5,
        "unfinished": 4,
        "risk": 3,
        "reason": 2,
        "next_action": 1,
    }.get(str(value or "").strip(), 0)


def _attach_dept_plan_completion_context(*, result: dict[str, Any]) -> None:
    result["dept_plan_completion_context_text"] = _format_dept_plan_completion_context(
        followups=result.get("dept_plan_followups", []),
        query_scope=result.get("query_scope", {}),
        pairing_summary=result.get("pairing_summary", {}),
    )


_DEPT_PLAN_FILTER_REASON_LABELS = {
    "department_mismatch": "部门不匹配",
    "no_owner_or_keyword_support": "缺少负责人或关键词支持",
    "not_selected_due_to_candidate_cap": "候选数量截断未展示",
    "owner_cross_department_possible": "负责人跨部门备选证据",
    "filtered_from_strong_candidates": "未进入强候选",
    "possible_evidence": "备选证据",
}

_DEPT_PLAN_SELF_EVAL_TYPE_LABELS = {
    "achievement": "完成项",
    "unfinished": "未完成项",
    "risk": "风险项",
    "reason": "原因说明",
    "next_action": "后续动作",
    "resolved": "已解决项",
    "unresolved": "未解决项",
}


_OPL_CLOSED_STATUS_VALUES = (
    "已解决",
    "已关闭",
    "关闭",
    "完成",
    "已完成",
    "closed",
    "done",
)

_OPL_OPEN_STATUS_VALUES = (
    "待解决",
    "未解决",
    "进行中",
    "跟进中",
    "未闭环",
    "待处理",
    "处理中",
    "open",
)


def _opl_open_status_sql(column: str) -> tuple[str, list[Any]]:
    placeholders = ", ".join(["%s"] * len(_OPL_CLOSED_STATUS_VALUES))
    return (
        f"({column} IS NULL OR TRIM({column}) = '' OR LOWER(TRIM({column})) NOT IN ({placeholders}))",
        [value.lower() for value in _OPL_CLOSED_STATUS_VALUES],
    )


def _opl_closed_status_sql(column: str) -> tuple[str, list[Any]]:
    placeholders = ", ".join(["%s"] * len(_OPL_CLOSED_STATUS_VALUES))
    return (
        f"LOWER(TRIM({column})) IN ({placeholders})",
        [value.lower() for value in _OPL_CLOSED_STATUS_VALUES],
    )


def _opl_status_filter_sql(column: str, status: str) -> tuple[str, list[Any]]:
    normalized = _compact_text(status)
    if normalized in {_compact_text(value) for value in _OPL_OPEN_STATUS_VALUES} or normalized in {"open", "unclosed"}:
        return _opl_open_status_sql(column)
    if normalized in {_compact_text(value) for value in _OPL_CLOSED_STATUS_VALUES} or normalized in {"closed", "done"}:
        return _opl_closed_status_sql(column)
    return f"{column} = %s", [status]


def _opl_status_group(status: Any) -> str:
    normalized = _compact_text(str(status or ""))
    if not normalized:
        return "unknown"
    if normalized in {_compact_text(value) for value in _OPL_CLOSED_STATUS_VALUES}:
        return "closed"
    if normalized in {_compact_text(value) for value in _OPL_OPEN_STATUS_VALUES}:
        return "open"
    if any(token in normalized for token in ("待", "未", "进行", "跟进", "处理中", "open")):
        return "open"
    if any(token in normalized for token in ("解决", "关闭", "完成", "closed", "done")):
        return "closed"
    return "unknown"


def _is_high_priority_opl(priority: Any) -> bool:
    normalized = _compact_text(str(priority or ""))
    if not normalized:
        return False
    return normalized in {"高", "高优先级", "紧急", "p0", "p1", "high"} or "高" in normalized


def _summarize_opl_issues(items: list[dict[str, Any]]) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    by_priority: dict[str, int] = {}
    by_owner: dict[str, int] = {}
    open_count = 0
    closed_count = 0
    high_priority_open_count = 0
    for item in items:
        if not isinstance(item, Mapping):
            continue
        status = str(item.get("status") or "未填写").strip() or "未填写"
        priority = str(item.get("priority") or "未填写").strip() or "未填写"
        owner = str(item.get("owner_user") or "未填写").strip() or "未填写"
        by_status[status] = by_status.get(status, 0) + 1
        by_priority[priority] = by_priority.get(priority, 0) + 1
        by_owner[owner] = by_owner.get(owner, 0) + 1
        status_group = _opl_status_group(status)
        if status_group == "closed":
            closed_count += 1
        else:
            open_count += 1
            if _is_high_priority_opl(priority):
                high_priority_open_count += 1
    return {
        "total": len(items),
        "open_count": open_count,
        "closed_count": closed_count,
        "high_priority_open_count": high_priority_open_count,
        "by_status": by_status,
        "by_priority": by_priority,
        "by_owner": by_owner,
    }


def _dept_plan_filter_reason_label(value: Any) -> str:
    raw = str(value or "possible_evidence").strip()
    return _DEPT_PLAN_FILTER_REASON_LABELS.get(raw, raw or "备选证据")


def _dept_plan_self_eval_type_label(value: Any) -> str:
    raw = str(value or "").strip()
    return _DEPT_PLAN_SELF_EVAL_TYPE_LABELS.get(raw, raw or "未标注类型")


def _opl_filter_reason_label(value: Any) -> str:
    raw = str(value or "").strip()
    labels = {
        "department_mismatch": "部门、负责人和关键词均不支持",
        "no_owner_or_keyword_support": "缺少负责人或关键词支持",
        "weak_opl_match": "OPL 关联较弱",
        "not_selected_due_to_candidate_cap": "候选数量截断未展示",
        "possible_opl_evidence": "备选 OPL 证据",
    }
    return labels.get(raw, raw or "备选 OPL 证据")


def _format_dept_plan_completion_context(
    *,
    followups: Any,
    query_scope: Any,
    pairing_summary: Any,
) -> str:
    scope = query_scope if isinstance(query_scope, Mapping) else {}
    summary = pairing_summary if isinstance(pairing_summary, Mapping) else {}
    lines = [
        "三七计划完成核对压缩上下文",
        f"- 计划月份: {scope.get('month') or ''}",
        f"- 部门范围: {scope.get('department') or '全部部门'}",
        f"- 周报证据范围: {scope.get('weekly_evidence_start') or ''} 至 {scope.get('weekly_evidence_end') or ''}",
        (
            f"- 统计: 计划 {summary.get('total_plans', 0)} 条，"
            f"负责人唯一完整周报 {summary.get('owner_weekly_report_count', 0)} 条，"
            f"计划引用 {summary.get('owner_weekly_report_ref_count', 0)} 次，"
            f"协作唯一周报 {summary.get('collaboration_weekly_report_count', 0)} 条，"
            f"协作辅助候选 {summary.get('weekly_evidence_candidate_count', 0)} 条，"
            f"负责人月度考核明细引用 {summary.get('owner_self_eval_item_count', 0)} 条，"
            f"缺失负责人考核 {summary.get('plans_missing_owner_self_eval', 0)} 项，"
            f"OPL 问题 {summary.get('opl_issue_count', 0)} 条，"
            f"计划相关 OPL 候选 {summary.get('opl_issue_candidate_count', 0)} 条，"
            f"未闭环 OPL 候选 {summary.get('open_opl_issue_candidate_count', 0)} 条，"
            f"高优先级未闭环 OPL 候选 {summary.get('high_priority_open_opl_issue_candidate_count', 0)} 条，"
            f"无候选证据 {summary.get('plans_without_evidence_candidates', 0)} 条。"
        ),
        (
            "- 判断规则: 规则层只按负责人、月份和周报日期取负责人完整周报原文，"
            "不用关键词过滤；计划上只保存 report_id 引用，完整原文保存在唯一周报池。"
            "负责人月度考核记录按计划负责人姓名直接查询 employee_self_eval_reports/items，"
            "不是从全部门自评里做候选搜索；"
            "OPL 问题清单用于追踪相关问题是否暴露、是否未闭环、解决措施是否对应计划目标；"
            "未闭环 OPL 是风险/卡点证据，不能单独等同于计划未完成；"
            "已解决 OPL 只有在解决进展对应计划目标时才可作为完成判断证据；"
            "第一层 LLM 按负责人计划组读取完整周报并抽取原文证据；"
            "关键词仅用于非负责人协作周报排序提示，不能替代语义判断。"
        ),
        "",
    ]
    if not isinstance(followups, list) or not followups:
        lines.append("无可评估的三七计划事项。")
        return "\n".join(lines).strip()

    max_items = _env_int("MYSQL_DEPT_PLAN_CONTEXT_MAX_ITEMS", 180)
    for index, followup in enumerate(followups[:max_items], start=1):
        if not isinstance(followup, Mapping):
            continue
        lines.append(f"计划 {index}:")
        lines.append(f"  部门: {followup.get('department') or '未填写部门'}")
        lines.append(f"  负责人: {followup.get('owner_user') or '未填写'}")
        lines.append(f"  截止/周次: {followup.get('due_date') or '未填写'} / {followup.get('target') or '未填写'}")
        lines.append(f"  计划内容: {_clip_text(followup.get('plan_text'), 260)}")
        owner_reports = followup.get("owner_weekly_report_refs") or followup.get("owner_weekly_reports", [])
        if isinstance(owner_reports, list):
            lines.append(
                f"  负责人完整周报引用: {len(owner_reports)} 条，完整原文在 owner_weekly_reports_pool 中"
            )
        match_audit = followup.get("weekly_match_audit", {})
        if isinstance(match_audit, Mapping):
            lines.append(
                "  周报匹配审计: "
                f"主表协作池 {match_audit.get('raw_weekly_count', 0)} 条，"
                f"部门兼容 {match_audit.get('department_filtered_count', 0)} 条，"
                f"关键词匹配 {match_audit.get('keyword_match_count', 0)} 条，"
                f"具体业务命中 {match_audit.get('business_keyword_match_count', 0)} 条，"
                f"入选非负责人协作候选 {match_audit.get('selected_count', 0)} 条，"
                f"备选复核 {match_audit.get('possible_evidence_count', 0)} 条，"
                f"候选上限 {match_audit.get('selected_candidate_limit', 0)} 条"
            )
        weekly_candidates = followup.get("weekly_done_candidates", [])
        if isinstance(weekly_candidates, list) and weekly_candidates:
            weekly_context_limit = _env_int("MYSQL_DEPT_PLAN_CONTEXT_WEEKLY_CANDIDATE_MAX_ITEMS", 12)
            for candidate_index, candidate in enumerate(weekly_candidates[:weekly_context_limit], start=1):
                if not isinstance(candidate, Mapping):
                    continue
                owner_suffix = "；负责人跨部门匹配" if candidate.get("owner_cross_department") else (
                    "；负责人匹配" if candidate.get("owner_match") else ""
                )
                overlap = candidate.get("specific_overlap_keywords") or candidate.get("overlap_keywords") or []
                overlap_suffix = ""
                if isinstance(overlap, list) and overlap:
                    overlap_suffix = f"；命中: {'、'.join(str(item) for item in overlap[:5])}"
                lines.append(
                    f"  非负责人协作辅助 {candidate_index}: "
                    f"{candidate.get('report_date') or ''} / {candidate.get('user_name') or '未知人员'} / "
                    f"{_clip_text(candidate.get('done_text'), 180)}{owner_suffix}{overlap_suffix}"
                )
        else:
            lines.append("  非负责人协作辅助: 未找到主表协作线索")
        possible_weekly_evidence = followup.get("possible_weekly_evidence", [])
        if isinstance(possible_weekly_evidence, list) and possible_weekly_evidence:
            review_context_limit = _env_int("MYSQL_DEPT_PLAN_CONTEXT_REVIEW_EVIDENCE_MAX_ITEMS", 12)
            for candidate_index, candidate in enumerate(possible_weekly_evidence[:review_context_limit], start=1):
                if not isinstance(candidate, Mapping):
                    continue
                can_support = bool(
                    candidate.get("owner_match")
                    and (
                        candidate.get("strong_keyword_support")
                        or candidate.get("business_keyword_support")
                        or candidate.get("candidate_confidence") in {"high", "medium"}
                    )
                )
                support_note = "可结合计划内容作为补充判断证据" if can_support else "低可信或非负责人材料，不能单独判已完成"
                lines.append(
                    f"  周报备选复核 {candidate_index}: "
                    f"{candidate.get('report_date') or ''} / {candidate.get('user_name') or '未知人员'} / "
                    f"{_clip_text(candidate.get('done_text'), 140)}"
                    f"（原因: {_dept_plan_filter_reason_label(candidate.get('filter_reason'))}"
                    f"{'；负责人跨部门' if candidate.get('owner_cross_department') else ''}；{support_note}）"
                )
        self_eval_candidates = followup.get("self_eval_candidates", [])
        owner_self_eval_reports = followup.get("owner_self_eval_reports", [])
        owner_self_eval_items = followup.get("owner_self_eval_items", [])
        missing_owner_self_eval_users = followup.get("missing_owner_self_eval_users", [])
        if isinstance(owner_self_eval_reports, list) and owner_self_eval_reports:
            report_labels = [
                (
                    f"{report.get('user_name') or '未知人员'}"
                    f"（完成率: {report.get('work_avg_completion_rate') or '未填写'}，"
                    f"上级评分: {report.get('leader_rating_score') or '未填写'}）"
                )
                for report in owner_self_eval_reports[:5]
                if isinstance(report, Mapping)
            ]
            lines.append(f"  负责人月度考核表: {'；'.join(report_labels) if report_labels else '已找到'}")
        if isinstance(owner_self_eval_items, list) and owner_self_eval_items:
            eval_context_limit = _env_int("MYSQL_DEPT_PLAN_CONTEXT_OWNER_SELF_EVAL_MAX_ITEMS", 12)
            for eval_index, item in enumerate(owner_self_eval_items[:eval_context_limit], start=1):
                if not isinstance(item, Mapping):
                    continue
                detail_parts = [
                    f"类型: {_dept_plan_self_eval_type_label(item.get('item_type'))}",
                    f"区块: {item.get('section') or '未填写'}",
                ]
                if item.get("completion_rate"):
                    detail_parts.append(f"完成率: {item.get('completion_rate')}")
                if item.get("plan_text"):
                    detail_parts.append(f"计划/目标: {_clip_text(item.get('plan_text'), 100)}")
                if item.get("result_text"):
                    detail_parts.append(f"完成结果: {_clip_text(item.get('result_text'), 120)}")
                if item.get("unfinished_text"):
                    detail_parts.append(f"未完成: {_clip_text(item.get('unfinished_text'), 100)}")
                if item.get("unresolved_text"):
                    detail_parts.append(f"未解决: {_clip_text(item.get('unresolved_text'), 100)}")
                if item.get("reason_text"):
                    detail_parts.append(f"原因: {_clip_text(item.get('reason_text'), 100)}")
                lines.append(
                    f"  负责人月度考核明细 {eval_index}: "
                    f"{item.get('user_name') or '未知人员'} / "
                    + "；".join(detail_parts)
                )
        elif isinstance(missing_owner_self_eval_users, list) and missing_owner_self_eval_users:
            lines.append(f"  负责人月度考核: 未找到 {'、'.join(str(item) for item in missing_owner_self_eval_users)} 的本月考核表")
        elif isinstance(self_eval_candidates, list) and self_eval_candidates:
            for candidate_index, candidate in enumerate(self_eval_candidates[:3], start=1):
                if not isinstance(candidate, Mapping):
                    continue
                lines.append(
                    f"  旧版自评候选 {candidate_index}: "
                    f"{_dept_plan_self_eval_type_label(candidate.get('item_type'))} / {_clip_text(candidate.get('item_text'), 180)}"
                )
        else:
            lines.append("  负责人月度考核: 未找到负责人本月考核记录")
        opl_candidates = followup.get("opl_issue_candidates", [])
        if isinstance(opl_candidates, list) and opl_candidates:
            opl_context_limit = _env_int("MYSQL_DEPT_PLAN_CONTEXT_OPL_MAX_ITEMS", 8)
            for opl_index, issue in enumerate(opl_candidates[:opl_context_limit], start=1):
                if not isinstance(issue, Mapping):
                    continue
                status_group = issue.get("status_group") or "unknown"
                if issue.get("open_issue"):
                    support_note = "未闭环问题，可作为风险/卡点证据，不能单独等同于计划未完成"
                elif issue.get("closed_issue"):
                    support_note = "已闭环问题；只有解决进展对应计划目标时才可作为完成判断证据"
                else:
                    support_note = "状态不明确，只能作为人工复核线索"
                overlap = issue.get("specific_overlap_keywords") or issue.get("overlap_keywords") or []
                overlap_suffix = ""
                if isinstance(overlap, list) and overlap:
                    overlap_suffix = f"；命中: {'、'.join(str(item) for item in overlap[:5])}"
                lines.append(
                    f"  OPL 相关问题 {opl_index}: "
                    f"{issue.get('issue_date') or ''} / 状态: {issue.get('status') or '未填写'}({status_group}) / "
                    f"优先级: {issue.get('priority') or '未填写'} / 负责人: {issue.get('owner_user') or '未填写'} / "
                    f"跟踪人: {issue.get('tracker_user') or '未填写'} / "
                    f"问题: {_clip_text(issue.get('issue_description'), 160)} / "
                    f"进展: {_clip_text(issue.get('solution_progress'), 160)}"
                    f"{overlap_suffix}（{support_note}）"
                )
        else:
            lines.append("  OPL 相关问题: 未找到匹配的问题清单记录")
        possible_opl_evidence = followup.get("possible_opl_evidence", [])
        if isinstance(possible_opl_evidence, list) and possible_opl_evidence:
            review_limit = _env_int("MYSQL_DEPT_PLAN_CONTEXT_OPL_REVIEW_MAX_ITEMS", 5)
            for opl_index, issue in enumerate(possible_opl_evidence[:review_limit], start=1):
                if not isinstance(issue, Mapping):
                    continue
                lines.append(
                    f"  OPL 备选复核 {opl_index}: "
                    f"{issue.get('issue_date') or ''} / 状态: {issue.get('status') or '未填写'} / "
                    f"优先级: {issue.get('priority') or '未填写'} / "
                    f"{_clip_text(issue.get('issue_description'), 120)}"
                    f"（原因: {_opl_filter_reason_label(issue.get('filter_reason'))}；不能单独判定计划状态）"
                )
        lines.append("")
    if len(followups) > max_items:
        lines.append(f"还有 {len(followups) - max_items} 条计划未展开，请提示用户缩小部门或月份范围。")
    return "\n".join(lines).strip()


def _build_dept_plan_progress_detail(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "query_scope": result.get("query_scope", {}),
        "dept_plan_count": len(result.get("dept_plans", [])) if isinstance(result.get("dept_plans"), list) else 0,
        "weekly_item_count": len(result.get("weekly_items", [])) if isinstance(result.get("weekly_items"), list) else 0,
        "owner_weekly_report_count": len(result.get("owner_weekly_reports_pool", []))
        if isinstance(result.get("owner_weekly_reports_pool"), list)
        else 0,
        "collaboration_weekly_report_count": len(result.get("collaboration_weekly_reports_pool", []))
        if isinstance(result.get("collaboration_weekly_reports_pool"), list)
        else 0,
        "owner_self_eval_item_count": len(result.get("owner_self_eval_items", []))
        if isinstance(result.get("owner_self_eval_items"), list)
        else 0,
        "self_eval_item_count": len(result.get("self_eval_items", [])) if isinstance(result.get("self_eval_items"), list) else 0,
        "opl_issue_count": len(result.get("opl_issues", [])) if isinstance(result.get("opl_issues"), list) else 0,
        "opl_issue_pool_count": len(result.get("opl_issue_pool", []))
        if isinstance(result.get("opl_issue_pool"), list)
        else 0,
        "dept_plan_followup_count": len(result.get("dept_plan_followups", []))
        if isinstance(result.get("dept_plan_followups"), list)
        else 0,
        "context_chars": len(str(result.get("dept_plan_completion_context_text") or "")),
    }


def _clip_text(value: Any, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."

def _empty_weekly_plan_done_result(
    *,
    date_range: dict[str, Any],
    trace_scope: dict[str, Any],
    skip_reason: str,
) -> dict[str, Any]:
    return {
        "last_week_plans": [],
        "this_week_completed": [],
        "candidate_matches": [],
        "weekly_pairs": [],
        "plan_followups": [],
        "pairing_summary": {
            "total_plans": 0,
            "weekly_pair_count": 0,
            "plans_without_followup_records": 0,
            "judgement_owner": "backend_llm",
            "judgement_instruction": "MySQL 工具只提供结构化计划和下一期完成记录配对证据；完成状态必须由 text_generate_tool 后端 LLM 根据 plan_followups 判断。",
            "trace_filtered_by_risk_and_help": True,
            "trace_skipped": True,
            "trace_skip_reason": skip_reason,
        },
        "query_date_range": date_range,
        "trace_scope": trace_scope,
    }


def _merge_weekly_plan_done_results(
    *,
    results: list[dict[str, Any]],
    date_range: dict[str, Any],
    trace_scope: dict[str, Any],
) -> dict[str, Any]:
    merged = {
        "last_week_plans": [],
        "this_week_completed": [],
        "candidate_matches": [],
        "weekly_pairs": [],
        "plan_followups": [],
    }
    plans_without_followup = 0
    for result in results:
        for key in merged:
            value = result.get(key, [])
            if isinstance(value, list):
                merged[key].extend(value)
        summary = result.get("pairing_summary", {})
        if isinstance(summary, Mapping):
            plans_without_followup += _safe_int(summary.get("plans_without_followup_records"), 0)

    merged["pairing_summary"] = {
        "total_plans": len(merged["plan_followups"]),
        "weekly_pair_count": len(merged["weekly_pairs"]),
        "plans_without_followup_records": plans_without_followup,
        "judgement_owner": "backend_llm",
        "judgement_instruction": "MySQL 工具只提供结构化计划和下一期完成记录配对证据；完成状态必须由 text_generate_tool 后端 LLM 根据 plan_followups 判断。",
        "trace_filtered_by_risk_and_help": True,
        "traced_people_count": len(trace_scope.get("traced_people", [])),
        "skipped_people_count": len(trace_scope.get("skipped_people", [])),
    }
    merged["query_date_range"] = date_range
    merged["trace_scope"] = trace_scope
    return merged


def _required_str(value: Any, field_name: str) -> str:
    text = _optional_str(value)
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _optional_date(value: Any) -> str | None:
    text = _optional_str(value)
    if text is None:
        return None
    return _validate_date(text)


def _required_date(value: Any, field_name: str) -> str:
    return _validate_date(_required_str(value, field_name))


def _validate_date(value: str) -> str:
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"date must be YYYY-MM-DD: {value}") from exc
    return value


def _parse_date(value: str) -> date:
    return datetime.strptime(_validate_date(value), "%Y-%m-%d").date()


def _required_month(value: Any) -> str:
    text = _required_str(value, "month")
    try:
        datetime.strptime(text, "%Y-%m")
    except ValueError as exc:
        raise ValueError(f"month must be YYYY-MM: {text}") from exc
    return text


def _optional_month(value: Any) -> str | None:
    text = _optional_str(value)
    if text is None:
        return None
    try:
        datetime.strptime(text, "%Y-%m")
    except ValueError as exc:
        raise ValueError(f"month must be YYYY-MM: {text}") from exc
    return text


def _month_range(month: str) -> tuple[str, str]:
    start = datetime.strptime(month, "%Y-%m").date()
    if start.month == 12:
        next_month = date(start.year + 1, 1, 1)
    else:
        next_month = date(start.year, start.month + 1, 1)
    end = date.fromordinal(next_month.toordinal() - 1)
    return start.isoformat(), end.isoformat()


def _add_days(value: str, days: int) -> str:
    parsed = datetime.strptime(value, "%Y-%m-%d").date()
    return (parsed + timedelta(days=max(0, days))).isoformat()


def _normalize_compare_weekly_date_range(
    *,
    last_week_start: str,
    last_week_end: str,
    this_week_start: str,
    this_week_end: str,
    user_input: str | None,
    runtime_timestamp: datetime | None = None,
    timezone_name: str = "UTC",
) -> dict[str, Any]:
    normalized = {
        "last_week_start": last_week_start,
        "last_week_end": last_week_end,
        "this_week_start": this_week_start,
        "this_week_end": this_week_end,
        "adjusted": False,
        "adjust_reason": "",
    }
    if _asks_previous_week_work_against_original_plan(user_input or ""):
        target_range = _previous_week_range_from_runtime(
            runtime_timestamp=runtime_timestamp,
            timezone_name=timezone_name,
        )
        if target_range is not None:
            target_start, target_end = target_range
            plan_start = target_start - timedelta(days=7)
            plan_end = target_start - timedelta(days=1)
            desired = {
                "last_week_start": plan_start.isoformat(),
                "last_week_end": plan_end.isoformat(),
                "this_week_start": target_start.isoformat(),
                "this_week_end": target_end.isoformat(),
            }
            if any(normalized[key] != value for key, value in desired.items()):
                normalized.update(
                    {
                        **desired,
                        "adjusted": True,
                        "adjust_reason": "用户询问上周工作是否按原计划完成，已按语义改为用上上周 next_week_plan 对比上周完成记录。",
                        "planner_original": {
                            "last_week_start": last_week_start,
                            "last_week_end": last_week_end,
                            "this_week_start": this_week_start,
                            "this_week_end": this_week_end,
                        },
                    }
                )
            return normalized

    if not _asks_monthly_weekly_plan_tracking(user_input or ""):
        return normalized

    start_date = datetime.strptime(last_week_start, "%Y-%m-%d").date()
    end_date = datetime.strptime(last_week_end, "%Y-%m-%d").date()
    this_end_date = datetime.strptime(this_week_end, "%Y-%m-%d").date()
    if start_date.day != 1:
        return normalized
    if (end_date - start_date).days > 10:
        return normalized

    month_start, month_end = _month_range(f"{start_date.year:04d}-{start_date.month:02d}")
    followup_end = date.fromordinal(datetime.strptime(month_end, "%Y-%m-%d").date().toordinal() + 31).isoformat()
    normalized.update(
        {
            "last_week_start": month_start,
            "last_week_end": month_end,
            "this_week_start": month_start,
            "this_week_end": max(this_week_end, followup_end),
            "adjusted": True,
            "adjust_reason": "用户问题是月度每周计划追踪，但 Planner 只给了月初第一周窗口，已扩展为整月计划窗口并覆盖后续一期周报。",
            "planner_original": {
                "last_week_start": last_week_start,
                "last_week_end": last_week_end,
                "this_week_start": this_week_start,
                "this_week_end": this_week_end,
            },
        }
    )
    if datetime.strptime(str(normalized["this_week_end"]), "%Y-%m-%d").date() < this_end_date:
        normalized["this_week_end"] = this_end_date.isoformat()
    return normalized


def _asks_previous_week_work_against_original_plan(text: str) -> bool:
    if not text:
        return False
    normalized = text.strip()
    has_previous_week_scope = any(token in normalized for token in ("上周", "上一周", "上星期", "上个周"))
    has_work_target = any(token in normalized for token in ("工作", "本周完成", "完成内容", "完成记录"))
    has_original_plan = any(token in normalized for token in ("原计划", "按计划", "原定计划", "计划内"))
    mentions_last_week_plan = any(token in normalized for token in ("上周计划", "上周的计划", "上一周计划", "上一周的计划"))
    return has_previous_week_scope and has_work_target and has_original_plan and not mentions_last_week_plan


def _previous_week_range_from_runtime(
    *,
    runtime_timestamp: datetime | None,
    timezone_name: str,
) -> tuple[date, date] | None:
    if runtime_timestamp is None:
        return None
    timestamp = runtime_timestamp
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    try:
        local_tz = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        local_tz = timezone.utc
    local_date = timestamp.astimezone(local_tz).date()
    this_week_start = local_date - timedelta(days=local_date.weekday())
    previous_week_start = this_week_start - timedelta(days=7)
    previous_week_end = this_week_start - timedelta(days=1)
    return previous_week_start, previous_week_end


def _asks_monthly_weekly_plan_tracking(text: str) -> bool:
    if not text:
        return False
    has_month_scope = any(keyword in text for keyword in ("上个月", "本月", "这个月", "月度", "每月")) or bool(
        re.search(r"\d{4}年\d{1,2}月|\d{1,2}月", text)
    )
    has_weekly_plan = "计划" in text and ("每周" in text or "周报" in text or "完成" in text)
    return has_month_scope and has_weekly_plan


def _asks_plan_completion_tracking(text: str | None) -> bool:
    if not text:
        return False
    has_plan = "计划" in text or "下周" in text or "周报" in text
    has_completion = any(
        keyword in text
        for keyword in (
            "是否完成",
            "有没有完成",
            "有没有按原计划完成",
            "按原计划完成",
            "按计划完成",
            "完成情况",
            "哪些完成",
            "哪些没完成",
            "没完成",
            "未完成",
            "完成率",
            "闭环",
            "落实",
        )
    )
    return has_plan and has_completion


def _earliest_date(left: str, right: str) -> str:
    left_date = datetime.strptime(left, "%Y-%m-%d").date()
    right_date = datetime.strptime(right, "%Y-%m-%d").date()
    return min(left_date, right_date).isoformat()


def _build_weekly_plan_followup_evidence(
    *,
    plans: list[dict[str, Any]],
    completed: list[dict[str, Any]],
) -> dict[str, Any]:
    plan_followups: list[dict[str, Any]] = []
    weekly_pair_map: dict[tuple[str, str | None, str, str | None], dict[str, Any]] = {}

    for index, plan in enumerate(plans, start=1):
        done_rows_for_plan = _completed_rows_for_plan(plan=plan, completed=completed)
        done_by_date = _group_rows_by_report_date(done_rows_for_plan)
        done_dates = sorted(done_by_date)
        plan_date = _row_date(plan)
        completion_date = _next_report_date(plan_date, done_dates)
        done_rows = done_by_date.get(completion_date or "", [])
        plan_text = str(plan.get("item_text") or "").strip()
        candidate_done_items = [_format_candidate_match_item(match) for match in _score_done_matches(plan_text, done_rows)[:5]]
        done_items = [_compact_done_item(row) for row in done_rows]

        plan_followups.append(
            {
                "plan_id": plan.get("id") or index,
                "user_name": plan.get("user_name"),
                "department": plan.get("department"),
                "plan_date": plan_date,
                "completion_date": completion_date,
                "plan_text": plan_text,
                "plan_evidence_text": _clip_text(plan.get("evidence_text"), 400),
                "done_items": done_items,
                "candidate_done_items": candidate_done_items,
                "source_doc_id": plan.get("source_doc_id"),
                "source_chunk_id": plan.get("source_chunk_id"),
            }
        )

        pair_department = _optional_str(plan.get("department"))
        pair_key = (str(plan.get("user_name") or ""), pair_department, str(plan_date or ""), completion_date)
        if pair_key not in weekly_pair_map:
            weekly_pair_map[pair_key] = {
                "user_name": plan.get("user_name"),
                "department": plan.get("department"),
                "plan_date": plan_date,
                "completion_date": completion_date,
                "plan_count": 0,
                "done_count": len(done_rows),
                "done_items": done_items,
            }
        weekly_pair_map[pair_key]["plan_count"] += 1

    return {
        "weekly_pairs": list(weekly_pair_map.values()),
        "plan_followups": plan_followups,
        "pairing_summary": {
            "total_plans": len(plan_followups),
            "weekly_pair_count": len(weekly_pair_map),
            "plans_without_followup_records": sum(1 for item in plan_followups if not item.get("completion_date")),
            "judgement_owner": "backend_llm",
            "judgement_instruction": "MySQL 工具只提供结构化计划和下一期完成记录配对证据；完成状态必须由 text_generate_tool 后端 LLM 根据 plan_followups 判断。",
        },
    }


def _format_candidate_match_item(match: dict[str, Any]) -> dict[str, Any]:
    return {
        "done_text": match["done_text"],
        "report_date": match["report_date"],
        "user_name": match.get("user_name"),
        "department": match.get("department"),
        "item_type": match.get("item_type"),
        "overlap_keywords": match["overlap_keywords"][:20],
        "item_overlap_keywords": match.get("item_overlap_keywords", [])[:20],
        "evidence_text": _clip_text(match["evidence_text"], 400),
        "source_doc_id": match["source_doc_id"],
        "source_chunk_id": match["source_chunk_id"],
    }


def _completed_rows_for_plan(*, plan: Mapping[str, Any], completed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    plan_user = _optional_str(plan.get("user_name"))
    if plan_user is None:
        return []
    plan_department = plan.get("department")
    return [
        row
        for row in completed
        if _same_user(plan_user, _optional_str(row.get("user_name")))
        and _departments_compatible(plan_department, row.get("department"))
    ]



def _group_rows_by_report_date(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        row_date = _row_date(row)
        if not row_date:
            continue
        grouped.setdefault(row_date, []).append(row)
    return grouped


def _row_date(row: Mapping[str, Any]) -> str | None:
    value = row.get("report_date")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _next_report_date(plan_date: str | None, done_dates: list[str]) -> str | None:
    if not plan_date:
        return None
    for done_date in done_dates:
        if done_date > plan_date:
            return done_date
    return None



def _score_done_matches(plan_text: str, done_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    plan_compact = _compact_text(plan_text)
    plan_tokens = _meaningful_keywords(plan_text)
    if not plan_compact or not plan_tokens:
        return []

    matches: list[dict[str, Any]] = []
    for done in done_rows:
        done_text = str(done.get("item_text") or "").strip()
        evidence_text = str(done.get("evidence_text") or "")
        done_item_compact = _compact_text(done_text)
        done_compact = _compact_text(" ".join([done_text, evidence_text]))
        done_item_tokens = _meaningful_keywords(done_text)
        done_tokens = _meaningful_keywords(" ".join([done_text, evidence_text]))
        overlap = sorted(plan_tokens & done_tokens, key=lambda item: (-len(item), item))
        item_overlap = sorted(plan_tokens & done_item_tokens, key=lambda item: (-len(item), item))
        exact_phrase = bool(plan_compact and plan_compact in done_compact)
        item_exact_phrase = bool(plan_compact and plan_compact in done_item_compact)
        if not overlap and not exact_phrase:
            continue
        coverage = len(overlap) / max(1, len(plan_tokens))
        item_coverage = len(item_overlap) / max(1, len(plan_tokens))
        longest = max((len(item) for item in overlap), default=0)
        item_longest = max((len(item) for item in item_overlap), default=0)
        score = (
            coverage
            + item_coverage * 0.8
            + (0.45 if exact_phrase else 0.0)
            + (0.35 if item_exact_phrase else 0.0)
            + min(longest / 20, 0.3)
            + min(item_longest / 20, 0.25)
        )
        matches.append(
            {
                "done_text": done_text,
                "report_date": _row_date(done),
                "user_name": done.get("user_name"),
                "department": done.get("department"),
                "item_type": done.get("item_type"),
                "score": round(score, 4),
                "coverage": round(coverage, 4),
                "item_coverage": round(item_coverage, 4),
                "exact_phrase": exact_phrase,
                "item_exact_phrase": item_exact_phrase,
                "longest_overlap": longest,
                "item_longest_overlap": item_longest,
                "overlap_keywords": overlap,
                "item_overlap_keywords": item_overlap,
                "evidence_text": _clip_text(done.get("evidence_text"), 400),
                "source_doc_id": done.get("source_doc_id"),
                "source_chunk_id": done.get("source_chunk_id"),
            }
        )
    return sorted(
        matches,
        key=lambda item: (
            item["item_exact_phrase"],
            item["item_coverage"],
            item["item_longest_overlap"],
            item["score"],
            item["coverage"],
            item["longest_overlap"],
        ),
        reverse=True,
    )



def _compact_done_item(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "done_text": row.get("item_text"),
        "report_date": row.get("report_date"),
        "user_name": row.get("user_name"),
        "department": row.get("department"),
        "item_type": row.get("item_type"),
        "evidence_text": _clip_text(row.get("evidence_text"), 400),
        "source_doc_id": row.get("source_doc_id"),
        "source_chunk_id": row.get("source_chunk_id"),
    }



def _compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _meaningful_keywords(text: str) -> set[str]:
    compact = _compact_text(text)
    raw_tokens = _keyword_tokens(text)
    tokens = {token.lower() for token in raw_tokens if len(token) >= 3 and token.lower() not in _WEAK_MATCH_TOKENS}
    short_tokens = {token for token in _IMPORTANT_SHORT_TOKENS if token in compact}
    synonym_tokens = _dept_plan_synonym_tokens(compact)
    phrases = {
        compact[index : index + size]
        for size in (5, 6, 7, 8)
        for index in range(0, max(0, len(compact) - size + 1))
        if any("\u4e00" <= char <= "\u9fff" for char in compact[index : index + size])
    }
    return tokens | short_tokens | synonym_tokens | phrases


def _dept_plan_synonym_tokens(compact_text: str) -> set[str]:
    if not compact_text:
        return set()
    tokens: set[str] = set()
    for group in _DEPT_PLAN_BUSINESS_SYNONYM_GROUPS:
        normalized_group = tuple(_compact_text(item) for item in group if item)
        if any(item and item in compact_text for item in normalized_group):
            tokens.update(item for item in normalized_group if item)
    return tokens


_IMPORTANT_SHORT_TOKENS = {
    "检索",
    "过滤",
    "策略",
    "压缩",
    "摘要",
    "总结",
    "批次",
    "融合",
    "质量",
    "数据",
    "边界",
    "建设",
    "持久",
    "基础",
    "设计",
    "扫描",
    "上传",
    "保存",
    "文件",
    "存储",
    "储存",
    "方法",
    "逻辑",
    "准确",
    "测试",
    "调试",
    "伺服",
    "模组",
    "转测",
    "冒烟",
    "问题单",
    "回归",
    "全流程",
    "锁类型",
    "专项",
    "招聘",
    "标定",
    "盘库",
    "上线",
    "库存",
    "股权",
    "确权",
    "财务",
    "审计",
    "培训",
    "学习",
    "内部",
    "qc",
    "ui",
    "ug",
    "oa",
    "erp",
    "3d",
    "2d",
    "2.5",
    "hmi",
    "plc",
    "ccs",
    "ecs",
    "bom",
    "bug",
    "联调",
    "部署",
    "封装",
    "推理",
    "算法",
    "图纸",
    "接线",
    "岗位",
    "考核",
    "考评",
    "下单",
    "页面",
    "界面",
    "雷达",
    "派单",
    "数显",
    "黄灯",
    "异常",
    "闭环",
}


_WEAK_MATCH_TOKENS = {
    "完成",
    "优化",
    "推进",
    "继续",
    "进行",
    "进行安装",
    "相关",
    "工作",
    "计划",
    "提升",
    "调整",
    "根据",
    "支持",
    "新增",
    "系统",
    "工具",
}


_GENERIC_DEPT_PLAN_MATCH_TOKENS = {
    "设计",
    "测试",
    "调试",
    "方法",
    "质量",
    "数据",
    "系统",
    "功能",
    "设备",
    "内容",
    "工作",
    "问题",
    "相关",
    "计划",
    "完成",
    "推进",
    "继续",
    "进行",
    "处理",
    "支持",
    "优化",
}


_SPECIFIC_SHORT_BUSINESS_TOKENS = {
    "ui",
    "qc",
    "ug",
    "oa",
    "erp",
    "3d",
    "2d",
    "2.5",
    "hmi",
    "plc",
    "ccs",
    "ecs",
    "bom",
    "bug",
    "伺服",
    "模组",
    "限位",
    "雷达",
    "派单",
    "冒烟",
    "回归",
    "岗位",
    "图纸",
    "接线",
    "部署",
    "封装",
    "算法",
    "标定",
    "盘库",
    "确权",
    "股权",
    "库存",
    "黄灯",
    "数显",
    "联调",
    "闭环",
    "下单",
}


_DEPT_PLAN_ACTION_TOKENS = {
    "测试",
    "调试",
    "设计",
    "回归",
    "冒烟",
    "下单",
    "部署",
    "上线",
    "编写",
    "修改",
    "招聘",
    "盘库",
    "确权",
    "联调",
    "标定",
    "验证",
    "封装",
    "集成",
}


_DEPT_PLAN_BUSINESS_SYNONYM_GROUPS = (
    ("电路图", "电气图纸", "接线图", "接线图纸", "图纸修改"),
    ("岗位描述", "岗位说明", "岗位说明书", "岗位职责"),
    ("考核操作方法", "考核操作办法", "考核办法", "考评内容", "考核标准"),
    ("触摸屏", "hmi", "画面模板", "触摸屏画面", "界面"),
    ("ui", "界面", "页面"),
    ("封装", "集成部署", "部署", "推理部署", "上线"),
    ("细分类", "锁类型细分类", "粗细分类"),
    ("服务器", "119设备服务器", "边缘设备", "边缘计算盒子"),
    ("下单", "图纸下单", "加工图纸", "外协制作", "制作"),
    ("问题单", "提单", "回归", "回归测试"),
    ("转测", "冒烟", "冒烟测试", "自验"),
    ("派单", "mqtt", "ccs", "ecs"),
    ("雷达", "引导", "数显", "黄灯", "红绿灯", "双雷达"),
    ("异常", "异常处理", "异常闭环", "上报异常", "闭环"),
    ("电柜", "动力柜", "网络柜"),
    ("盘库", "库存", "erp", "bom"),
    ("股权", "确权", "财产份额"),
    ("qc", "质量部", "质量"),
    ("相机", "相机支架", "标定", "坐标"),
    ("伺服", "模组", "限位", "轴"),
    ("技术宝典", "操作手册", "技术文档", "文档"),
)


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        identity = _row_identity(row)
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(row)
    return deduped


def _row_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    row_id = row.get("id")
    if row_id not in (None, ""):
        return ("id", str(row_id))
    return (
        "fields",
        _optional_str(row.get("user_name")),
        _optional_str(row.get("department")),
        _row_date(row),
        _optional_str(row.get("item_type")),
        _optional_str(row.get("item_text")),
        _optional_str(row.get("source_doc_id")),
        _optional_str(row.get("source_chunk_id")),
    )


def _resolve_item_type_alias(item_type: str) -> str:
    aliases = {
        "this_week_work": _env_name("MYSQL_WEEKLY_THIS_WEEK_ITEM_TYPE", "this_week_work"),
        "next_week_plan": _env_name("MYSQL_WEEKLY_NEXT_WEEK_ITEM_TYPE", "next_week_plan"),
        "self_eval": _env_name("MYSQL_WEEKLY_SELF_EVAL_ITEM_TYPE", "self_eval"),
        "department_plan": _env_name("MYSQL_WEEKLY_DEPARTMENT_PLAN_ITEM_TYPE", "department_plan"),
        "dept_plan": _env_name("MYSQL_WEEKLY_DEPARTMENT_PLAN_ITEM_TYPE", "dept_plan"),
        "department_self_eval": _env_name("MYSQL_WEEKLY_DEPARTMENT_SELF_EVAL_ITEM_TYPE", "department_self_eval"),
    }
    return aliases.get(item_type, item_type)


def _resolve_self_eval_item_type_alias(item_type: str) -> str:
    aliases = {
        "achievement": _env_name("MYSQL_DEPT_SELF_EVAL_ACHIEVEMENT_ITEM_TYPE", "achievement"),
        "unfinished": _env_name("MYSQL_DEPT_SELF_EVAL_UNFINISHED_ITEM_TYPE", "unfinished"),
        "reason": _env_name("MYSQL_DEPT_SELF_EVAL_REASON_ITEM_TYPE", "reason"),
        "risk": _env_name("MYSQL_DEPT_SELF_EVAL_RISK_ITEM_TYPE", "risk"),
        "next_action": _env_name("MYSQL_DEPT_SELF_EVAL_NEXT_ACTION_ITEM_TYPE", "next_action"),
    }
    return aliases.get(item_type, item_type)


def _optional_item_type(value: Any) -> str | None:
    text = _optional_str(value)
    if text is None:
        return None
    allowed = {
        "this_week_work",
        "this_week_done",
        "next_week_plan",
        "self_eval",
        "department_plan",
        "dept_plan",
        "department_self_eval",
    }
    if text not in allowed:
        raise ValueError(f"unsupported item_type: {text}")
    return text


def _keyword_tokens(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", text)
    tokens = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", compact))
    phrases = {
        compact[index : index + 4]
        for index in range(0, max(0, len(compact) - 3))
        if any("\u4e00" <= char <= "\u9fff" for char in compact[index : index + 4])
    }
    return tokens | phrases
