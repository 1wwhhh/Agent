from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.prompts.task_planner import ToolDefinition, build_task_planner_prompt
from app.schemas.llm import LLMFunctionCall, LLMRequest, LLMResponse
from app.state import LangGraphState
from app.tools.mysql_business_tools import (
    CompareDeptPlanCompletionTool,
    CompareWeeklyPlanDoneTool,
    DeptPlanSchema,
    DeptSelfEvalSchema,
    EmployeeSelfEvalSchema,
    MySQLBusinessClient,
    QueryWeeklyReportsTool,
    WeeklyReportSchema,
    _attach_dept_plan_completion_context,
    _add_days,
    _build_compare_weekly_progress_detail,
    _build_dept_plan_completion_evidence,
    _build_weekly_plan_followup_evidence,
    _classify_risk_and_help,
    _dept_plan_weekly_match_result,
    _dept_plan_weekly_candidates,
    _departments_compatible,
    _split_owner_users,
    _judge_plan_followup,
    _month_range,
    _normalize_compare_weekly_date_range,
)
from app.tools.weekly_blocker_tools import ClassifyWeeklyBlockersTool, JudgeWeeklyBlockerTraceTool


def test_weekly_report_sql_uses_parameterized_filters() -> None:
    client = MySQLBusinessClient(schema=WeeklyReportSchema())

    sql, params = client._build_weekly_items_sql(
        user_name="张三",
        department="九工机器/研发部",
        start_date="2026-05-01",
        end_date="2026-05-31",
        item_type="this_week_work",
        limit=50,
    )

    assert "weekly_report_items" in sql
    assert "weekly_reports" in sql
    assert "LEFT JOIN" in sql
    assert "张三" not in sql
    assert "研发部" not in sql
    assert "r.`risk_and_help` AS `risk_and_help`" in sql
    assert sql.count("%s") == len(params)
    assert params[:-1] == ["张三", "九工机器/研发部", "2026-05-01", "2026-05-31", "this_week_work"]
    assert params[-1] == 50


def test_weekly_report_sql_matches_short_department_suffix() -> None:
    client = MySQLBusinessClient(schema=WeeklyReportSchema())

    sql, params = client._build_weekly_items_sql(
        user_name=None,
        department="产品部",
        start_date="2026-06-15",
        end_date="2026-06-21",
        item_type=None,
        limit=500,
    )

    assert "(i.`department` = %s OR i.`department` LIKE %s OR i.`department` LIKE %s)" in sql
    assert params == ["产品部", "%/产品部", "%\\产品部", "2026-06-15", "2026-06-21", 500]


def test_weekly_report_sql_can_query_report_level_rows() -> None:
    client = MySQLBusinessClient(schema=WeeklyReportSchema())

    sql, params = client._build_weekly_reports_sql(
        user_name=None,
        department="产品部",
        start_date="2026-06-15",
        end_date="2026-06-21",
        limit=500,
    )

    assert "FROM `weekly_reports` r" in sql
    assert "this_week_raw" in sql
    assert "next_week_raw" in sql
    assert "risk_and_help" in sql
    assert "'weekly_report' AS `item_type`" in sql
    assert "(r.`department` = %s OR r.`department` LIKE %s OR r.`department` LIKE %s)" in sql
    assert params == ["产品部", "%/产品部", "%\\产品部", "2026-06-15", "2026-06-21", 500]


def test_weekly_report_sql_can_omit_repeated_evidence_text() -> None:
    client = MySQLBusinessClient(schema=WeeklyReportSchema())

    sql, params = client._build_weekly_items_sql(
        user_name=None,
        department=None,
        start_date="2026-06-15",
        end_date="2026-06-21",
        item_type=None,
        limit=500,
        include_evidence_text=False,
    )

    assert "NULL AS `evidence_text`" in sql
    assert "i.`evidence_text` AS `evidence_text`" not in sql
    assert params == ["2026-06-15", "2026-06-21", 500]


def test_weekly_report_schema_rejects_unsafe_identifiers() -> None:
    schema = WeeklyReportSchema(table="weekly_report_items; DROP TABLE users")

    with pytest.raises(ValueError):
        MySQLBusinessClient(schema=schema)


def test_dept_plan_sql_uses_parameterized_filters() -> None:
    client = MySQLBusinessClient(
        schema=WeeklyReportSchema(),
        dept_plan_schema=DeptPlanSchema(),
        dept_self_eval_schema=DeptSelfEvalSchema(),
    )

    sql, params = client._build_dept_plan_items_sql(
        month="2026-05",
        department="质量部",
        doc_id="质量部三七计划_pptx",
        limit=50,
    )

    assert "FROM `dept_plan_items` p" in sql
    assert "质量部" not in sql
    assert "质量部三七计划_pptx" not in sql
    assert sql.count("%s") == len(params)
    assert params == ["2026-05", "质量部", "%/质量部", "%\\质量部", "质量部三七计划_pptx", 50]


def test_dept_self_eval_sql_uses_parameterized_filters() -> None:
    client = MySQLBusinessClient(
        schema=WeeklyReportSchema(),
        dept_plan_schema=DeptPlanSchema(),
        dept_self_eval_schema=DeptSelfEvalSchema(),
    )

    sql, params = client._build_dept_self_eval_items_sql(
        month="2026-05",
        department="质量部",
        item_type="achievement",
        limit=100,
    )

    assert "FROM `dept_self_eval_items` e" in sql
    assert "achievement" not in sql
    assert sql.count("%s") == len(params)
    assert params == ["2026-05", "质量部", "%/质量部", "%\\质量部", "achievement", 100]


def test_employee_self_eval_sql_uses_owner_names_not_department_candidate_search() -> None:
    client = MySQLBusinessClient(
        schema=WeeklyReportSchema(),
        dept_plan_schema=DeptPlanSchema(),
        dept_self_eval_schema=DeptSelfEvalSchema(),
        employee_self_eval_schema=EmployeeSelfEvalSchema(),
    )

    sql, params = client._build_employee_self_eval_items_for_users_sql(
        month="2026-05",
        user_names=["张三", "朱一曼"],
        limit=100,
    )

    assert "FROM `employee_self_eval_reports` r" in sql
    assert "LEFT JOIN `employee_self_eval_items` i" in sql
    assert "r.`user_name` IN (%s, %s, %s)" in sql
    assert "department" not in sql.lower().split("where", 1)[1]
    assert "张三" not in sql
    assert "朱一曼" not in sql
    assert sql.count("%s") == len(params)
    assert params == ["2026-05", "张三", "朱亦曼", "朱一曼", 100]


def test_mysql_tools_expose_expected_routing_capabilities() -> None:
    query_tool = QueryWeeklyReportsTool()
    compare_tool = CompareWeeklyPlanDoneTool()
    dept_plan_tool = CompareDeptPlanCompletionTool()

    assert query_tool.get_routing_capability()["default_task_type"] == "query_weekly_reports"
    assert compare_tool.get_routing_capability()["default_task_type"] == "compare_weekly_plan_done"
    assert dept_plan_tool.get_routing_capability()["default_task_type"] == "compare_dept_plan_completion"
    assert "mysql" in query_tool.get_routing_capability()["supported_tags"]


def test_planner_prompt_contains_mysql_business_guidance() -> None:
    tool = QueryWeeklyReportsTool()
    capability = tool.get_routing_capability()
    planner = build_task_planner_prompt(
        user_input="研发部 5 月计划完成率是多少",
        tools=[
            ToolDefinition(
                name=tool.name,
                description=tool.description,
                input_schema=tool.get_input_schema(),
                output_schema=tool.get_output_schema(),
                supported_task_types=capability["supported_task_types"],
                default_task_type=capability["default_task_type"],
                supported_tags=capability["supported_tags"],
            ),
            ToolDefinition(
                name="text_generate_tool",
                description="Generate final text",
                input_schema={"type": "object", "properties": {"prompt": {"type": "string"}}},
                output_schema={"type": "object", "properties": {"text": {"type": "string"}}},
                supported_task_types=["text_generation"],
                default_task_type="text_generation",
                supported_tags=["llm", "generation", "text"],
            ),
        ],
        planning_timestamp="2026-06-18T00:00:00+00:00",
    )

    assert "MySQL Business Tool Guidance" in planner.user_prompt
    assert "优先使用 MySQL 业务工具" in planner.user_prompt
    assert "risk_and_help" in planner.user_prompt
    assert "员工自填卡点字段" in planner.user_prompt
    assert "classify_weekly_blockers" in planner.user_prompt
    assert "judge_weekly_blocker_trace" in planner.user_prompt
    assert "compare_weekly_plan_done 必须依赖" in planner.user_prompt
    assert "不能用字段非空" in planner.user_prompt
    assert "trace_weeks=2" in planner.user_prompt
    assert "weekly_blocker_classification_output_key" in planner.user_prompt
    assert "weekly_blocker_context_text" in planner.user_prompt
    assert "weekly_blocker_trace_judgement.weekly_blocker_context_text" in planner.user_prompt
    assert "record_level=\"reports\"" in planner.user_prompt
    assert "plan_tracking_context_text" in planner.user_prompt
    assert "compare_dept_plan_completion" in planner.user_prompt
    assert "dept_plan_completion_context_text" in planner.user_prompt
    assert "不要把 {{weekly_reports}}" in planner.user_prompt
    assert "不要把完整 {{weekly_plan_comparison}}" in planner.user_prompt
    assert "compare_weekly_plan_done 必须依赖 query_weekly_reports 和 classify_weekly_blockers" in planner.user_prompt
    assert "不要创建任意 SQL 任务" in planner.user_prompt


class FakeTrackingBusinessClient(MySQLBusinessClient):
    def __init__(self) -> None:
        super().__init__(schema=WeeklyReportSchema())
        self.compare_calls: list[dict[str, object]] = []

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
    ) -> dict[str, object]:
        self.compare_calls.append(
            {
                "user_name": user_name,
                "department": department,
                "last_week_start": last_week_start,
                "last_week_end": last_week_end,
                "this_week_start": this_week_start,
                "this_week_end": this_week_end,
                "limit": limit,
            }
        )
        return {
            "last_week_plans": [{"user_name": user_name, "department": department, "item_text": "下周计划"}],
            "this_week_completed": [],
            "candidate_matches": [],
            "weekly_pairs": [],
            "plan_followups": [{"user_name": user_name, "department": department, "plan_text": "下周计划"}],
            "pairing_summary": {
                "total_plans": 1,
                "weekly_pair_count": 0,
                "plans_without_followup_records": 1,
                "judgement_owner": "backend_llm",
            },
        }


class FakeHistoricalTrackingBusinessClient(FakeTrackingBusinessClient):
    def __init__(self) -> None:
        super().__init__()
        self.query_calls: list[dict[str, object]] = []

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
    ) -> list[dict[str, object]]:
        self.query_calls.append(
            {
                "user_name": user_name,
                "department": department,
                "start_date": start_date,
                "end_date": end_date,
                "item_type": item_type,
                "limit": limit,
                "record_level": record_level,
                "include_evidence_text": include_evidence_text,
            }
        )
        if record_level == "reports":
            return [
                {
                    "user_name": user_name,
                    "department": department,
                    "report_date": "2026-06-08",
                    "risk_and_help": "供应商接口资料未提供，联调受阻",
                }
            ]
        if item_type in {"this_week_work", "this_week_done"}:
            return [
                {
                    "id": f"{item_type}-1",
                    "user_name": user_name,
                    "department": department,
                    "report_date": "2026-06-18",
                    "item_type": item_type,
                    "item_text": "已收到供应商接口资料并完成联调验证",
                    "evidence_text": "已收到供应商接口资料并完成联调验证",
                }
            ]
        return []


class FakeWeeklyBlockerLLMClient:
    def __init__(self, arguments: dict[str, object] | None = None, *, fail: bool = False) -> None:
        self.arguments = arguments or {}
        self.fail = fail
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("LLM unavailable")
        tool_name = request.tool_choice or (request.function_schemas[0].name if request.function_schemas else "emit")
        return LLMResponse(
            text="{}",
            model_name=request.model_name or "fake-llm",
            request_id=request.request_id,
            session_id=request.session_id,
            trace_id=request.trace_id,
            function_call=LLMFunctionCall(
                tool_name=tool_name,
                arguments=self.arguments,
            ),
        )


class FakeQueryBusinessClient(MySQLBusinessClient):
    def __init__(self) -> None:
        super().__init__(schema=WeeklyReportSchema())
        self.query_calls: list[dict[str, object]] = []

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
    ) -> list[dict[str, object]]:
        self.query_calls.append(
            {
                "user_name": user_name,
                "department": department,
                "start_date": start_date,
                "end_date": end_date,
                "item_type": item_type,
                "limit": limit,
                "record_level": record_level,
                "include_evidence_text": include_evidence_text,
            }
        )
        return []


class FakeCompareQueryBusinessClient(MySQLBusinessClient):
    def __init__(self) -> None:
        super().__init__(schema=WeeklyReportSchema())
        self.query_calls: list[dict[str, object]] = []

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
    ) -> list[dict[str, object]]:
        self.query_calls.append(
            {
                "user_name": user_name,
                "department": department,
                "start_date": start_date,
                "end_date": end_date,
                "item_type": item_type,
                "limit": limit,
                "record_level": record_level,
                "include_evidence_text": include_evidence_text,
            }
        )
        if item_type == "next_week_plan":
            return [
                {
                    "id": 101,
                    "user_name": "张三",
                    "department": "产品部",
                    "report_date": "2026-06-14",
                    "item_type": "next_week_plan",
                    "item_text": "完成扫码枪多类型扫码复测",
                    "evidence_text": "下周计划：完成扫码枪多类型扫码复测",
                }
            ]
        if item_type == "this_week_done":
            return [
                {
                    "id": 201,
                    "user_name": "张三",
                    "department": "产品部",
                    "report_date": "2026-06-21",
                    "item_type": "this_week_done",
                    "item_text": "完成扫码枪多类型扫码复测并记录问题",
                    "evidence_text": "本周完成：完成扫码枪多类型扫码复测并记录问题",
                }
            ]
        return []


class FakeDeptPlanBusinessClient(MySQLBusinessClient):
    def __init__(self) -> None:
        super().__init__(
            schema=WeeklyReportSchema(),
            dept_plan_schema=DeptPlanSchema(),
            dept_self_eval_schema=DeptSelfEvalSchema(),
        )
        self.compare_calls: list[dict[str, object]] = []

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
    ) -> dict[str, object]:
        self.compare_calls.append(
            {
                "month": month,
                "department": department,
                "doc_id": doc_id,
                "include_weekly": include_weekly,
                "include_self_eval": include_self_eval,
                "followup_days": followup_days,
                "limit": limit,
            }
        )
        result = _build_dept_plan_completion_evidence(
            plans=[
                {
                    "id": 1,
                    "doc_id": "总办三七计划_pptx",
                    "month": month,
                    "department": department or "总办",
                    "plan_text": "完成制度流程梳理",
                    "owner_user": "张三",
                    "due_date": "2026-05-31",
                    "target": "第四周",
                    "slide_no": 3,
                    "evidence_text": "第四周：完成制度流程梳理（张三）",
                }
            ],
            weekly_completed=[],
            owner_weekly_reports=[
                {
                    "id": 11,
                    "department": department or "总办",
                    "user_name": "张三",
                    "report_date": "2026-05-29",
                    "this_week_raw": "完成制度流程梳理并提交评审。",
                    "next_week_raw": "继续推进制度发布后的培训。",
                    "risk_and_help": "",
                    "source_doc_id": "weekly_doc_11",
                    "source_chunk_id": "weekly_chunk_11",
                }
            ],
            collaboration_weekly_reports=[],
            self_eval_items=[
                {
                    "id": 21,
                    "department": department or "总办",
                    "item_type": "achievement",
                    "item_text": "完成制度流程梳理",
                    "evidence_text": "已完成制度流程梳理",
                }
            ],
        )
        month_start, month_end = _month_range(month)
        result["query_scope"] = {
            "month": month,
            "department": department,
            "doc_id": doc_id,
            "weekly_evidence_start": month_start,
            "weekly_evidence_end": _add_days(month_end, followup_days),
            "followup_days": followup_days,
        }
        _attach_dept_plan_completion_context(result=result)
        return result


class FakeDeptPlanSourceBusinessClient(MySQLBusinessClient):
    def __init__(self) -> None:
        super().__init__(
            schema=WeeklyReportSchema(),
            dept_plan_schema=DeptPlanSchema(),
            dept_self_eval_schema=DeptSelfEvalSchema(),
        )
        self.owner_report_calls: list[dict[str, object]] = []

    async def query_dept_plan_items(
        self,
        *,
        month: str,
        department: str | None,
        doc_id: str | None,
        limit: int,
    ) -> list[dict[str, object]]:
        return [
            {
                "id": 1,
                "doc_id": doc_id or "产品部三七计划_pptx",
                "month": month,
                "department": department or "产品部",
                "plan_text": "完成扫码枪采购闭环",
                "owner_user": "张三、李四",
                "due_date": "2026-05-31",
                "target": "第四周",
                "slide_no": 2,
                "evidence_text": "第四周：完成扫码枪采购闭环",
            }
        ]

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
    ) -> list[dict[str, object]]:
        return []

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
    ) -> list[dict[str, object]]:
        return []

    async def query_weekly_reports_for_users(
        self,
        *,
        user_names: list[str],
        start_date: str | None,
        end_date: str | None,
        limit: int,
    ) -> list[dict[str, object]]:
        self.owner_report_calls.append(
            {
                "user_names": user_names,
                "start_date": start_date,
                "end_date": end_date,
                "limit": limit,
            }
        )
        return [
            {
                "id": 101,
                "user_name": "张三",
                "department": "产品部",
                "report_date": "2026-05-24",
                "this_week_raw": "关键完成证据：扫码枪已下单并完成验收。",
                "next_week_raw": "继续跟进扫码枪使用反馈。",
                "risk_and_help": "",
                "source_doc_id": "weekly_doc_101",
                "source_chunk_id": "weekly_chunk_101",
            }
        ]

    async def query_dept_self_eval_items(
        self,
        *,
        month: str,
        department: str | None,
        item_type: str | None,
        limit: int,
    ) -> list[dict[str, object]]:
        return []


@pytest.mark.asyncio
async def test_query_weekly_reports_infers_department_when_planner_omits_it() -> None:
    client = FakeQueryBusinessClient()
    tool = QueryWeeklyReportsTool(client=client)
    state = LangGraphState.create(
        request_id="req_infer_department_query",
        session_id="sess_infer_department_query",
        user_input="上周产品部大家都卡在哪些问题上？",
    )

    result = await tool._arun(
        payload={
            "department": None,
            "start_date": "2026-06-15",
            "end_date": "2026-06-21",
            "item_type": None,
            "record_level": "reports",
            "include_evidence_text": False,
            "limit": 50,
        },
        context=state.context,
    )

    assert result.success is True
    assert client.query_calls[0]["department"] == "产品部"
    assert client.query_calls[0]["record_level"] == "reports"


@pytest.mark.asyncio
async def test_compare_dept_plan_completion_tool_infers_department_and_outputs_context() -> None:
    client = FakeDeptPlanBusinessClient()
    tool = CompareDeptPlanCompletionTool(client=client)
    state = LangGraphState.create(
        request_id="req_dept_plan_completion",
        session_id="sess_dept_plan_completion",
        user_input="质量部三七计划书五月份计划有没有完成",
    )

    result = await tool._arun(
        payload={
            "month": "2026-05",
            "department": None,
            "doc_id": None,
            "include_weekly": True,
            "include_self_eval": True,
            "followup_days": 31,
            "limit": 100,
        },
        context=state.context,
    )

    assert result.success is True
    assert client.compare_calls[0]["department"] == "质量部"
    assert client.compare_calls[0]["month"] == "2026-05"
    assert client.compare_calls[0]["followup_days"] == 7
    output = result.output
    assert output["pairing_summary"]["judgement_owner"] == "backend_llm"
    assert output["dept_plan_followups"][0]["owner_weekly_report_refs"]
    assert output["owner_weekly_reports_pool"]
    assert output["dept_plan_followups"][0]["weekly_done_candidates"] == []
    assert output["dept_plan_followups"][0]["self_eval_candidates"]
    assert "三七计划完成核对压缩上下文" in output["dept_plan_completion_context_text"]
    assert result.metadata["dept_plan_followup_count"] == 1


@pytest.mark.asyncio
async def test_compare_dept_plan_completion_tool_keeps_explicit_long_followup() -> None:
    client = FakeDeptPlanBusinessClient()
    tool = CompareDeptPlanCompletionTool(client=client)
    state = LangGraphState.create(
        request_id="req_dept_plan_completion_followup",
        session_id="sess_dept_plan_completion_followup",
        user_input="质量部三七计划书五月份计划截至现在有没有完成",
    )

    result = await tool._arun(
        payload={
            "month": "2026-05",
            "department": None,
            "doc_id": None,
            "include_weekly": True,
            "include_self_eval": True,
            "followup_days": 7,
            "limit": 100,
        },
        context=state.context,
    )

    assert result.success is True
    assert client.compare_calls[0]["followup_days"] == 31


@pytest.mark.asyncio
async def test_compare_dept_plan_completion_tool_allows_strict_month_only() -> None:
    client = FakeDeptPlanBusinessClient()
    tool = CompareDeptPlanCompletionTool(client=client)
    state = LangGraphState.create(
        request_id="req_dept_plan_completion_month_only",
        session_id="sess_dept_plan_completion_month_only",
        user_input="质量部三七计划书五月份计划有没有完成，只看本月",
    )

    result = await tool._arun(
        payload={
            "month": "2026-05",
            "department": None,
            "doc_id": None,
            "include_weekly": True,
            "include_self_eval": True,
            "followup_days": 31,
            "limit": 100,
        },
        context=state.context,
    )

    assert result.success is True
    assert client.compare_calls[0]["followup_days"] == 0


@pytest.mark.asyncio
async def test_compare_dept_plan_completion_fetches_owner_full_reports_by_owner_and_date() -> None:
    client = FakeDeptPlanSourceBusinessClient()

    result = await client.compare_dept_plan_completion(
        month="2026-05",
        department="产品部",
        doc_id=None,
        include_weekly=True,
        include_self_eval=False,
        followup_days=7,
        limit=100,
    )

    assert client.owner_report_calls == [
        {
            "user_names": ["张三", "李四"],
            "start_date": "2026-05-01",
            "end_date": "2026-06-07",
            "limit": 300,
        }
    ]
    followup = result["dept_plan_followups"][0]
    assert followup["weekly_done_candidates"] == []
    assert followup["owner_weekly_report_refs"][0]["提交人"] == "张三"
    assert "周报原文" not in followup["owner_weekly_report_refs"][0]
    assert result["owner_weekly_reports_pool"][0]["提交人"] == "张三"
    assert "扫码枪已下单并完成验收" in result["owner_weekly_reports_pool"][0]["周报原文"]
    assert "继续跟进扫码枪使用反馈" in result["owner_weekly_reports_pool"][0]["周报原文"]
    assert followup["missing_owner_self_eval_users"] == []
    assert result["pairing_summary"]["plans_missing_owner_self_eval"] == 0
    assert result["pairing_summary"]["owner_weekly_report_count"] == 1
    assert result["pairing_summary"]["owner_weekly_report_ref_count"] == 1


def test_dept_plan_completion_evidence_builds_candidates_without_status() -> None:
    result = _build_dept_plan_completion_evidence(
        plans=[
            {
                "id": 1,
                "month": "2026-05",
                "department": "质量部",
                "plan_text": "完成供应商质量整改闭环",
                "owner_user": "张三",
                "due_date": "2026-05-31",
                "target": "第四周",
                "evidence_text": "第四周：完成供应商质量整改闭环（张三）",
            }
        ],
        weekly_completed=[
            {
                "id": 2,
                "department": "质量部",
                "user_name": "张三",
                "report_date": "2026-05-29",
                "item_type": "this_week_done",
                "item_text": "完成供应商质量整改闭环并归档",
                "evidence_text": "本周完成：完成供应商质量整改闭环并归档",
            },
            {
                "id": 22,
                "department": "质量部",
                "user_name": "李四",
                "report_date": "2026-05-29",
                "item_type": "this_week_done",
                "item_text": "协助完成供应商质量整改闭环资料复核",
                "evidence_text": "本周完成：协助完成供应商质量整改闭环资料复核",
            }
        ],
        owner_weekly_reports=[
            {
                "department": "质量部",
                "user_name": "张三",
                "report_date": "2026-05-29",
                "this_week_raw": "关键完成证据：完成供应商质量整改闭环并归档。",
                "next_week_raw": "继续跟进供应商质量维护。",
                "risk_and_help": "",
                "source_doc_id": "weekly_doc_1",
                "source_chunk_id": "weekly_chunk_1",
            }
        ],
        self_eval_items=[
            {
                "id": 3,
                "department": "质量部",
                "item_type": "achievement",
                "item_text": "完成供应商质量整改闭环",
                "evidence_text": "已完成供应商质量整改闭环",
            }
        ],
    )
    result["query_scope"] = {"month": "2026-05", "weekly_evidence_start": "2026-05-01", "weekly_evidence_end": "2026-06-30"}
    _attach_dept_plan_completion_context(result=result)

    followup = result["dept_plan_followups"][0]
    assert followup["owner_weekly_report_refs"][0]["提交人"] == "张三"
    assert "周报原文" not in followup["owner_weekly_report_refs"][0]
    assert result["owner_weekly_reports_pool"][0]["事项类型"] == "weekly_report"
    assert "关键完成证据：完成供应商质量整改闭环并归档。" in result["owner_weekly_reports_pool"][0]["周报原文"]
    assert "下周计划" in result["owner_weekly_reports_pool"][0]["周报原文"]
    assert followup["weekly_done_candidates"][0]["owner_match"] is False
    assert followup["weekly_done_candidates"][0]["candidate_source"] == "collaboration_weekly_report_hint"
    assert followup["weekly_match_audit"]["selected_count"] == 1
    assert followup["weekly_match_audit"]["owner_match_count"] == 0
    assert result["pairing_summary"]["owner_weekly_report_count"] == 1
    assert result["pairing_summary"]["owner_weekly_report_ref_count"] == 1
    assert followup["possible_weekly_evidence"] == []
    assert result["pairing_summary"]["possible_weekly_evidence_count"] == 0
    assert followup["self_eval_candidates"][0]["item_type"] == "achievement"
    assert "status" not in followup
    assert "三七计划完成核对压缩上下文" in result["dept_plan_completion_context_text"]
    assert "负责人完整周报" in result["dept_plan_completion_context_text"]


def test_dept_plan_completion_links_owner_employee_self_eval_records_directly() -> None:
    result = _build_dept_plan_completion_evidence(
        plans=[
            {
                "id": 1,
                "department": "质量部",
                "owner_user": "张三",
                "month": "2026-05",
                "due_date": "2026-05-31",
                "plan_text": "完成供应商质量整改闭环",
            }
        ],
        weekly_completed=[],
        owner_weekly_reports=[],
        collaboration_weekly_reports=[],
        owner_self_eval_items=[
            {
                "report_id": 10,
                "item_id": 101,
                "doc_id": "eval_doc_zhangsan",
                "month": "2026-05",
                "user_name": "张三",
                "department": "质量部",
                "position": "质量工程师",
                "report_sheet_name": "张三",
                "work_avg_completion_rate": "95%",
                "leader_rating_score": "A",
                "section": "工作任务",
                "item_type": "achievement",
                "item_text": "供应商质量整改闭环",
                "plan_text": "完成供应商质量整改闭环",
                "result_text": "已完成供应商质量整改闭环并归档",
                "completion_rate": "100%",
                "source_row": 12,
            },
            {
                "report_id": 11,
                "item_id": 201,
                "doc_id": "eval_doc_lisi",
                "month": "2026-05",
                "user_name": "李四",
                "department": "质量部",
                "item_type": "achievement",
                "item_text": "协助整改",
                "result_text": "协助供应商质量整改资料复核",
                "source_row": 8,
            },
        ],
    )
    result["query_scope"] = {"month": "2026-05", "weekly_evidence_start": "2026-05-01", "weekly_evidence_end": "2026-06-07"}
    _attach_dept_plan_completion_context(result=result)

    followup = result["dept_plan_followups"][0]
    assert followup["owner_self_eval_found"] is True
    assert followup["owner_self_eval_reports"][0]["user_name"] == "张三"
    assert len(followup["owner_self_eval_items"]) == 1
    assert followup["owner_self_eval_items"][0]["user_name"] == "张三"
    assert followup["owner_self_eval_items"][0]["result_text"] == "已完成供应商质量整改闭环并归档"
    assert followup["missing_owner_self_eval_users"] == []
    assert followup["self_eval_candidates"] == []
    assert result["pairing_summary"]["owner_self_eval_item_count"] == 1
    assert "负责人月度考核明细" in result["dept_plan_completion_context_text"]
    assert "自评候选" not in result["dept_plan_completion_context_text"]


def test_dept_plan_completion_splits_plus_owner_separator_for_owner_evidence() -> None:
    result = _build_dept_plan_completion_evidence(
        plans=[
            {
                "id": 1,
                "department": "行政部",
                "owner_user": "周斌+王浩",
                "month": "2026-06",
                "due_date": "2026-06-12",
                "plan_text": "在飞书中拉取周报和考勤",
            }
        ],
        weekly_completed=[],
        owner_weekly_reports=[
            {
                "id": 54,
                "report_date": "2026-06-12",
                "user_name": "王浩",
                "department": "九工机器/行政部",
                "this_week_raw": "飞书集成与LLM接入测试已完成，支持通过命令直接拉取考勤、汇报、用章统计数据。",
                "next_week_raw": "添加前端页面，接入飞书集成功能。",
                "risk_and_help": "",
                "source_doc_id": "每月月度周报_xlsx",
            }
        ],
        collaboration_weekly_reports=[],
        owner_self_eval_items=[
            {
                "report_id": 6,
                "item_id": 43,
                "doc_id": "王浩-2026年6月考核表_xlsx",
                "month": "2026-06",
                "user_name": "王浩",
                "department": "行政部",
                "item_type": "achievement",
                "item_text": "飞书拉取周报、考勤：通过命令直接能够拉取个人、部门、全员周报和考勤",
                "plan_text": "飞书拉取周报、考勤",
                "result_text": "通过命令直接能够拉取个人、部门、全员周报和考勤",
                "completion_rate": "90%",
            }
        ],
    )

    followup = result["dept_plan_followups"][0]
    assert followup["owner_names"] == ["周斌", "王浩"]
    assert followup["owner_weekly_report_refs"][0]["提交人"] == "王浩"
    assert followup["owner_self_eval_items"][0]["user_name"] == "王浩"
    assert followup["missing_owner_self_eval_users"] == ["周斌"]
    assert result["pairing_summary"]["owner_weekly_report_ref_count"] == 1
    assert result["pairing_summary"]["owner_self_eval_item_count"] == 1


def test_dept_plan_weekly_candidates_prioritize_owner_and_split_multiple_owners() -> None:
    plan = {
        "department": "产品部",
        "owner_user": "姚豪、刘雨康",
        "plan_text": "配合进行2.5锁扭机的调试相关工作",
    }
    weekly_completed = [
        {
            "department": "产品部",
            "user_name": "张三",
            "report_date": "2026-05-12",
            "item_type": "this_week_done",
            "item_text": "完成设备整理",
            "evidence_text": "完成设备整理",
        },
        {
            "department": "产品部",
            "user_name": "刘雨康",
            "report_date": "2026-05-13",
            "item_type": "this_week_done",
            "item_text": "2.5锁扭机调试跟进",
            "evidence_text": "2.5锁扭机调试跟进",
        },
        {
            "department": "产品部",
            "user_name": "姚豪",
            "report_date": "2026-05-14",
            "item_type": "this_week_done",
            "item_text": "继续跟进调试",
            "evidence_text": "继续跟进调试",
        },
        {
            "department": "产品部",
            "user_name": "王五",
            "report_date": "2026-05-15",
            "item_type": "this_week_done",
            "item_text": "配合进行2.5锁扭机的调试相关工作",
            "evidence_text": "配合进行2.5锁扭机的调试相关工作",
        },
    ]

    candidates = _dept_plan_weekly_candidates(plan=plan, weekly_completed=weekly_completed)

    assert len(candidates) >= 2
    assert candidates[0]["owner_match"] is True
    assert {candidate["user_name"] for candidate in candidates if candidate["owner_match"]} == {"刘雨康", "姚豪"}
    assert not any(candidate["user_name"] == "张三" for candidate in candidates)
    assert any(candidate["user_name"] == "王五" for candidate in candidates)
    assert any(candidate["candidate_confidence"] == "low" for candidate in candidates)
    assert any(candidate["candidate_source"] in {"owner_trace", "owner_keyword_trace"} for candidate in candidates)


def test_dept_plan_weekly_candidates_prioritize_monthly_keyword_evidence_over_later_owner_trace() -> None:
    plan = {
        "month": "2026-05",
        "department": "产品部",
        "owner_user": "姚豪",
        "plan_text": "锁扭机2.5伺服及模组的调试相关工作",
    }
    weekly_completed = [
        {
            "department": "九工机器/产品部",
            "user_name": "姚豪",
            "report_date": "2026-06-05",
            "item_type": "this_week_done",
            "item_text": "整理设备物料并打扫现场",
            "evidence_text": "整理设备物料并打扫现场",
        },
        {
            "department": "九工机器/产品部",
            "user_name": "姚豪",
            "report_date": "2026-05-09",
            "item_type": "this_week_done",
            "item_text": "伺服系统调试与限位设定",
            "evidence_text": "上周伺服调试中只完成了交换机构的原点标定，本周继续调试",
        },
    ]

    candidates = _dept_plan_weekly_candidates(plan=plan, weekly_completed=weekly_completed)

    assert candidates[0]["report_date"] == "2026-05-09"
    assert candidates[0]["candidate_source"] == "owner_keyword_trace"
    assert {"伺服", "调试"}.issubset(set(candidates[0]["overlap_keywords"]))


def test_dept_plan_department_compatibility_uses_business_aliases() -> None:
    assert _departments_compatible("软件专业", "九工机器/软件部") is True
    assert _departments_compatible("机械专业", "九工机器/机械设计部") is True
    assert _departments_compatible("生产制造部", "九工机器/制造部") is True
    assert _departments_compatible("总办", "九工机器/干部群") is True


def test_split_owner_users_filters_noise_and_normalizes_aliases() -> None:
    owners = _split_owner_users("朱一曼、节拍优化已做完、目前判断有无锁3D算法在兼容做、马立娜")

    assert owners == ["朱亦曼", "马丽娜"]
    assert _split_owner_users("厉害") == []


def test_split_owner_users_supports_plus_separator() -> None:
    assert _split_owner_users("周斌+王浩") == ["周斌", "王浩"]
    assert _split_owner_users("周斌＋王浩") == ["周斌", "王浩"]


def test_dept_plan_completion_normalizes_owner_display_names() -> None:
    result = _build_dept_plan_completion_evidence(
        plans=[
            {
                "id": 1,
                "month": "2026-06",
                "department": "机械专业",
                "plan_text": "小车新方案",
                "owner_user": "李奎，马立娜",
                "due_date": "2026-07-03",
                "target": "第三周（6.29-7.03）",
            }
        ],
        weekly_completed=[],
        owner_weekly_reports=[],
        collaboration_weekly_reports=[],
        owner_self_eval_items=[],
    )

    followup = result["dept_plan_followups"][0]
    assert followup["owner_user"] == "李奎、马丽娜"
    assert followup["owner_user_original"] == "李奎，马立娜"
    assert followup["owner_names"] == ["李奎", "马丽娜"]
    assert followup["owner_name_audit"]["alias_normalizations"] == [
        {"original": "马立娜", "canonical": "马丽娜"}
    ]


def test_dept_plan_completion_marks_invalid_owner_text_as_unrecognized() -> None:
    result = _build_dept_plan_completion_evidence(
        plans=[
            {
                "id": 1,
                "month": "2026-06",
                "department": "生产制造部",
                "plan_text": "镜像侧拍位数据采集",
                "owner_user": "厉害",
                "due_date": "2026-06-18",
                "target": "第一周（6.15-6.18）",
            }
        ],
        weekly_completed=[],
        owner_weekly_reports=[],
        collaboration_weekly_reports=[],
        owner_self_eval_items=[],
    )

    followup = result["dept_plan_followups"][0]
    assert followup["owner_user"] == "未识别负责人（原文：厉害）"
    assert followup["owner_user_original"] == "厉害"
    assert followup["owner_names"] == []
    assert followup["owner_name_audit"]["invalid_owner_text"] == "厉害"


def test_dept_plan_weekly_candidates_drop_recency_only_noise() -> None:
    plan = {
        "department": "产品部",
        "owner_user": "陈志妹",
        "plan_text": "UI功能测试",
    }
    weekly_completed = [
        {
            "department": "九工机器/产品部",
            "user_name": "刘宝莹",
            "report_date": "2026-06-05",
            "item_type": "this_week_done",
            "item_text": "技术宝典编撰",
            "evidence_text": "技术宝典编撰",
        },
        {
            "department": "九工机器/产品部",
            "user_name": "陈志妹",
            "report_date": "2026-05-16",
            "item_type": "this_week_done",
            "item_text": "完成 UI 功能测试并记录问题单",
            "evidence_text": "完成 UI 功能测试并记录问题单",
        },
    ]

    candidates = _dept_plan_weekly_candidates(plan=plan, weekly_completed=weekly_completed)

    assert [candidate["user_name"] for candidate in candidates] == ["陈志妹"]
    assert candidates[0]["owner_match"] is True
    assert candidates[0]["candidate_confidence"] == "high"


def test_dept_plan_weekly_candidates_match_department_alias_and_person_alias() -> None:
    plan = {
        "department": "软件专业",
        "owner_user": "朱一曼",
        "plan_text": "完成检索过滤策略优化",
    }
    weekly_completed = [
        {
            "department": "九工机器/软件部",
            "user_name": "朱亦曼",
            "report_date": "2026-05-16",
            "item_type": "this_week_done",
            "item_text": "完成检索过滤策略优化并上线",
            "evidence_text": "完成检索过滤策略优化并上线",
        },
    ]

    candidates = _dept_plan_weekly_candidates(plan=plan, weekly_completed=weekly_completed)

    assert candidates
    assert candidates[0]["owner_match"] is True
    assert candidates[0]["candidate_source"] == "owner_keyword_trace"


def test_dept_plan_weekly_match_result_audits_filtered_and_capped_rows(monkeypatch) -> None:
    monkeypatch.setenv("MYSQL_DEPT_PLAN_WEEKLY_CANDIDATE_MAX_ITEMS", "5")
    plan = {
        "department": "产品部",
        "owner_user": "陈志妹",
        "plan_text": "UI功能测试",
    }
    weekly_completed = [
        {
            "department": "九工机器/机械设计部",
            "user_name": "陈志妹",
            "report_date": "2026-05-12",
            "item_type": "this_week_done",
            "item_text": "完成 UI 功能测试",
            "evidence_text": "完成 UI 功能测试",
        },
        {
            "department": "九工机器/产品部",
            "user_name": "刘宝莹",
            "report_date": "2026-05-13",
            "item_type": "this_week_done",
            "item_text": "技术宝典编撰",
            "evidence_text": "技术宝典编撰",
        },
        *[
            {
                "department": "九工机器/产品部",
                "user_name": "陈志妹",
                "report_date": f"2026-05-{14 + index:02d}",
                "item_type": "this_week_done",
                "item_text": f"完成 UI 功能测试第{index}轮",
                "evidence_text": f"完成 UI 功能测试第{index}轮",
            }
            for index in range(6)
        ],
    ]

    result = _dept_plan_weekly_match_result(plan=plan, weekly_completed=weekly_completed)

    assert len(result["selected_candidates"]) == 7
    assert all(candidate["owner_match"] for candidate in result["selected_candidates"])
    assert result["audit"]["raw_weekly_count"] == 8
    assert result["audit"]["department_filtered_count"] == 7
    assert "department_mismatch" not in result["audit"]["filtered_counts"]
    assert result["audit"]["owner_cross_department_count"] == 1
    assert result["audit"]["owner_cross_department_keyword_count"] == 1
    assert result["audit"]["sample_owner_cross_department"][0]["user_name"] == "陈志妹"
    assert result["audit"]["filtered_counts"]["no_owner_or_keyword_support"] == 1
    assert result["audit"]["selected_before_cap_count"] == 7
    assert result["audit"]["selected_count"] == 7
    assert result["audit"]["owner_selected_count"] == 7
    assert result["audit"]["non_owner_selected_count"] == 0
    assert result["audit"]["selected_candidate_limit"] == 5
    assert result["audit"]["non_owner_candidate_limit"] == 0
    assert result["audit"]["owner_candidate_over_limit_count"] == 2
    assert result["audit"]["capped_candidate_count"] == 0
    assert result["possible_evidence"]
    assert {item["filter_reason"] for item in result["possible_evidence"]} == {"no_owner_or_keyword_support"}


def test_dept_plan_weekly_match_result_caps_non_owner_keyword_candidates(monkeypatch) -> None:
    monkeypatch.setenv("MYSQL_DEPT_PLAN_WEEKLY_CANDIDATE_MAX_ITEMS", "2")
    plan = {
        "department": "产品部",
        "owner_user": "张三",
        "plan_text": "UI功能测试",
    }
    weekly_completed = [
        {
            "department": "九工机器/产品部",
            "user_name": f"协作人{index}",
            "report_date": f"2026-05-{10 + index:02d}",
            "item_type": "this_week_done",
            "item_text": f"完成 UI 功能测试第{index}轮",
            "evidence_text": f"完成 UI 功能测试第{index}轮",
        }
        for index in range(4)
    ]

    result = _dept_plan_weekly_match_result(plan=plan, weekly_completed=weekly_completed)

    assert len(result["selected_candidates"]) == 2
    assert all(not candidate["owner_match"] for candidate in result["selected_candidates"])
    assert result["audit"]["owner_selected_count"] == 0
    assert result["audit"]["non_owner_selected_count"] == 2
    assert result["audit"]["non_owner_candidate_limit"] == 2
    assert result["audit"]["capped_candidate_count"] == 2
    assert {item["filter_reason"] for item in result["possible_evidence"]} == {"not_selected_due_to_candidate_cap"}


def test_dept_plan_weekly_candidates_rank_business_synonym_over_owner_generic_rows() -> None:
    plan = {
        "month": "2026-05",
        "department": "产品部",
        "owner_user": "刘宝莹",
        "plan_text": "处理电路图相关的修改",
    }
    weekly_completed = [
        {
            "department": "九工机器/产品部",
            "user_name": "刘宝莹",
            "report_date": "2026-05-29",
            "item_type": "this_week_done",
            "item_text": "完成技术宝典材料编码整理",
            "evidence_text": "完成技术宝典材料编码整理",
        },
        {
            "department": "九工机器/产品部",
            "user_name": "刘宝莹",
            "report_date": "2026-05-01",
            "item_type": "this_week_done",
            "item_text": "锁扭机S200伺服接线电气图纸修改",
            "evidence_text": "锁扭机S200伺服接线电气图纸修改",
        },
        {
            "department": "九工机器/产品部",
            "user_name": "刘宝莹",
            "report_date": "2026-05-22",
            "item_type": "this_week_done",
            "item_text": "电气图纸修改",
            "evidence_text": "电气图纸修改",
        },
    ]

    result = _dept_plan_weekly_match_result(plan=plan, weekly_completed=weekly_completed)

    selected_texts = [candidate["done_text"] for candidate in result["selected_candidates"]]

    assert selected_texts[:2] == ["电气图纸修改", "锁扭机S200伺服接线电气图纸修改"]
    assert selected_texts[-1] == "完成技术宝典材料编码整理"
    assert result["selected_candidates"][0]["business_keyword_support"] is True
    assert "电路图" in result["selected_candidates"][0]["overlap_keywords"]
    assert result["selected_candidates"][0]["candidate_confidence"] == "high"


def test_dept_plan_weekly_candidates_match_position_description_synonym() -> None:
    plan = {
        "month": "2026-05",
        "department": "产品部",
        "owner_user": "徐纯虎",
        "plan_text": "产品部岗位描述",
    }
    weekly_completed = [
        {
            "department": "九工机器/产品部",
            "user_name": "徐纯虎",
            "report_date": "2026-05-09",
            "item_type": "this_week_done",
            "item_text": "更新产品部电气工程师岗位说明",
            "evidence_text": "更新产品部电气工程师岗位说明",
        },
    ]

    result = _dept_plan_weekly_match_result(plan=plan, weekly_completed=weekly_completed)

    assert result["selected_candidates"]
    assert result["selected_candidates"][0]["business_keyword_support"] is True
    assert "岗位描述" in result["selected_candidates"][0]["overlap_keywords"]


def test_dept_plan_weekly_candidates_keep_owner_cross_department_keyword_evidence() -> None:
    plan = {
        "month": "2026-05",
        "department": "产品部",
        "owner_user": "陈志妹",
        "plan_text": "转测版本冒烟测试",
    }
    weekly_completed = [
        {
            "department": "九工机器/干部群",
            "user_name": "陈志妹",
            "report_date": "2026-05-09",
            "item_type": "this_week_done",
            "item_text": "本周主要进行了5.6转测版本的冒烟测试",
            "evidence_text": "本周主要进行了5.6转测版本的冒烟测试",
        },
        {
            "department": "九工机器/产品部",
            "user_name": "刘宝莹",
            "report_date": "2026-05-09",
            "item_type": "this_week_done",
            "item_text": "技术宝典编撰",
            "evidence_text": "技术宝典编撰",
        },
    ]

    result = _dept_plan_weekly_match_result(plan=plan, weekly_completed=weekly_completed)
    candidate = result["selected_candidates"][0]

    assert candidate["user_name"] == "陈志妹"
    assert candidate["owner_match"] is True
    assert candidate["department_match"] is False
    assert candidate["owner_cross_department"] is True
    assert candidate["candidate_source"] == "owner_cross_department_keyword_trace"
    assert {"转测版本", "冒烟测试"}.issubset(set(candidate["overlap_keywords"]))
    assert result["audit"]["owner_cross_department_selected_count"] == 1
    assert result["audit"]["filtered_counts"].get("department_mismatch", 0) == 0


def test_dept_plan_weekly_candidates_mark_owner_cross_department_without_keyword_as_low() -> None:
    plan = {
        "month": "2026-05",
        "department": "产品部",
        "owner_user": "陈志妹",
        "plan_text": "问题单回归测试",
    }
    weekly_completed = [
        {
            "department": "九工机器/干部群",
            "user_name": "陈志妹",
            "report_date": "2026-05-09",
            "item_type": "this_week_done",
            "item_text": "整理会议纪要并同步制度讨论结果",
            "evidence_text": "整理会议纪要并同步制度讨论结果",
        },
    ]

    candidates = _dept_plan_weekly_candidates(plan=plan, weekly_completed=weekly_completed)

    assert candidates[0]["owner_cross_department"] is True
    assert candidates[0]["candidate_source"] == "owner_cross_department_trace"
    assert candidates[0]["candidate_confidence"] == "low"
    assert candidates[0]["overlap_keywords"] == []


def test_dept_plan_weekly_candidates_do_not_cross_department_for_invalid_owner() -> None:
    plan = {
        "month": "2026-05",
        "department": "机械专业",
        "owner_user": "全体",
        "plan_text": "UG学习和内部培训",
    }
    weekly_completed = [
        {
            "department": "九工机器/干部群",
            "user_name": "翟志华",
            "report_date": "2026-05-09",
            "item_type": "this_week_done",
            "item_text": "整理会议纪要",
            "evidence_text": "整理会议纪要",
        },
    ]

    result = _dept_plan_weekly_match_result(plan=plan, weekly_completed=weekly_completed)

    assert result["selected_candidates"] == []
    assert result["audit"]["owner_names"] == []
    assert result["audit"]["owner_cross_department_count"] == 0
    assert result["audit"]["filtered_counts"]["department_mismatch"] == 1


def test_dept_plan_context_includes_match_audit_and_possible_evidence() -> None:
    result = _build_dept_plan_completion_evidence(
        plans=[
            {
                "id": 1,
                "month": "2026-05",
                "department": "产品部",
                "plan_text": "UI功能测试",
                "owner_user": "陈志妹",
                "due_date": "2026-05-31",
                "target": "第四周",
            }
        ],
        weekly_completed=[
            {
                "department": "九工机器/产品部",
                "user_name": "刘宝莹",
                "report_date": "2026-05-13",
                "item_type": "this_week_done",
                "item_text": "技术宝典编撰",
                "evidence_text": "技术宝典编撰",
            },
        ],
        self_eval_items=[],
    )
    result["query_scope"] = {"month": "2026-05", "weekly_evidence_start": "2026-05-01", "weekly_evidence_end": "2026-07-01"}
    _attach_dept_plan_completion_context(result=result)

    followup = result["dept_plan_followups"][0]
    assert followup["weekly_done_candidates"] == []
    assert followup["possible_weekly_evidence"][0]["filter_reason"] == "no_owner_or_keyword_support"
    assert result["pairing_summary"]["possible_weekly_evidence_count"] == 1
    assert result["pairing_summary"]["plans_with_possible_weekly_evidence"] == 1
    assert "周报匹配审计" in result["dept_plan_completion_context_text"]
    assert "周报备选复核" in result["dept_plan_completion_context_text"]
    assert "不能单独判已完成" in result["dept_plan_completion_context_text"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (None, "empty"),
        ("   ", "empty"),
        ("无", "no_blocker_statement"),
        ("暂无卡点", "no_blocker_statement"),
        ("目前无卡点，无需协调", "no_blocker_statement"),
        ("工作有序开展，暂无卡点。", "no_blocker_statement"),
        ("暂无卡点，无异常，不影响推进", "no_blocker_statement"),
        ("暂无卡点，未发现异常", "no_blocker_statement"),
        ("目前没有卡点，前面有问题点都有安排，在处理中。", "no_blocker_statement"),
        ("暂无卡点，但需要软件部协助联调", "actionable_blocker"),
        ("无3D相机导致识别受阻", "actionable_blocker"),
        ("无法完成相机标定", "actionable_blocker"),
        ("没有物料，安装暂停", "actionable_blocker"),
        ("接口联调依赖外部确认", "actionable_blocker"),
    ],
)
def test_risk_and_help_classification_distinguishes_no_blocker_statements(text: str | None, expected: str) -> None:
    assert _classify_risk_and_help(text) == expected


@pytest.mark.asyncio
async def test_compare_weekly_plan_done_traces_only_empty_risk_people() -> None:
    client = FakeTrackingBusinessClient()
    tool = CompareWeeklyPlanDoneTool(client=client)
    state = LangGraphState.create(
        request_id="req_empty_risk_trace",
        session_id="sess_empty_risk_trace",
        user_input="上周所有人汇报的卡点是什么",
    )
    state.context.set_task_result(
        "weekly_reports",
        {
            "items": [
                {"user_name": "张三", "department": "研发部", "risk_and_help": "接口联调依赖外部确认"},
                {"user_name": "李四", "department": "研发部", "risk_and_help": ""},
                {"user_name": "李四", "department": "研发部", "risk_and_help": None},
                {"user_name": "王五", "department": "测试部", "risk_and_help": "   "},
            ],
            "count": 4,
        },
    )

    result = await tool._arun(
        payload={
            "last_week_start": "2026-06-08",
            "last_week_end": "2026-06-14",
            "this_week_start": "2026-06-15",
            "this_week_end": "2026-06-21",
            "trace_only_empty_risk_from_output_key": "weekly_reports",
            "limit": 50,
        },
        context=state.context,
    )

    assert result.success is True
    assert [call["user_name"] for call in client.compare_calls] == ["李四", "王五"]
    assert client.compare_calls[0]["department"] == "研发部"
    assert client.compare_calls[1]["department"] == "测试部"
    output = result.output
    assert output["trace_scope"]["trace_filtered_by_risk_and_help"] is True
    assert [person["user_name"] for person in output["trace_scope"]["traced_people"]] == ["李四", "王五"]
    assert [person["user_name"] for person in output["trace_scope"]["skipped_people"]] == ["张三"]
    assert result.metadata["trace_user_count"] == 2
    assert len(output["plan_followups"]) == 2
    assert "weekly_blocker_context_text" in output
    assert "weekly_blocker_people" in output
    assert "final_report" not in output
    assert "final_report_type" not in output
    compact_context = output["weekly_blocker_context_text"]
    assert "张三 / 研发部" in compact_context
    assert "员工自填卡点" in compact_context
    assert "接口联调依赖外部确认" in compact_context
    assert "李四 / 研发部" in compact_context
    assert "王五 / 测试部" in compact_context
    assert "未填写卡点" in compact_context
    assert "risk_and_help" not in compact_context
    assert len(compact_context) < 6000


@pytest.mark.asyncio
async def test_compare_weekly_plan_done_short_department_matches_full_department_trace_scope() -> None:
    client = FakeTrackingBusinessClient()
    tool = CompareWeeklyPlanDoneTool(client=client)
    state = LangGraphState.create(
        request_id="req_short_department_trace",
        session_id="sess_short_department_trace",
        user_input="上周产品部谁没写卡点",
    )
    state.context.set_task_result(
        "weekly_reports",
        {
            "items": [
                {"user_name": "张三", "department": "九工机器/产品部", "risk_and_help": "需要软件协助"},
                {"user_name": "李四", "department": "九工机器/产品部", "risk_and_help": ""},
                {"user_name": "王五", "department": "九工机器/行政部", "risk_and_help": ""},
            ],
            "count": 3,
        },
    )

    result = await tool._arun(
        payload={
            "department": "产品部",
            "last_week_start": "2026-06-08",
            "last_week_end": "2026-06-14",
            "this_week_start": "2026-06-15",
            "this_week_end": "2026-06-21",
            "trace_only_empty_risk_from_output_key": "weekly_reports",
            "limit": 50,
        },
        context=state.context,
    )

    assert result.success is True
    assert [call["user_name"] for call in client.compare_calls] == ["李四"]
    assert client.compare_calls[0]["department"] == "九工机器/产品部"
    assert [person["user_name"] for person in result.output["trace_scope"]["traced_people"]] == ["李四"]
    assert [person["user_name"] for person in result.output["trace_scope"]["skipped_people"]] == ["张三"]


@pytest.mark.asyncio
async def test_compare_weekly_plan_done_infers_department_when_planner_omits_it() -> None:
    client = FakeTrackingBusinessClient()
    tool = CompareWeeklyPlanDoneTool(client=client)
    state = LangGraphState.create(
        request_id="req_infer_department_trace",
        session_id="sess_infer_department_trace",
        user_input="九工机器/产品部上周大家都卡在哪",
    )
    state.context.set_task_result(
        "weekly_reports",
        {
            "items": [
                {"user_name": "张三", "department": "九工机器/产品部", "risk_and_help": ""},
                {"user_name": "李四", "department": "九工机器/行政部", "risk_and_help": ""},
            ],
            "count": 2,
        },
    )

    result = await tool._arun(
        payload={
            "department": None,
            "last_week_start": "2026-06-08",
            "last_week_end": "2026-06-14",
            "this_week_start": "2026-06-15",
            "this_week_end": "2026-06-21",
            "trace_only_empty_risk_from_output_key": "weekly_reports",
            "limit": 50,
        },
        context=state.context,
    )

    assert result.success is True
    assert [call["user_name"] for call in client.compare_calls] == ["张三"]
    assert client.compare_calls[0]["department"] == "九工机器/产品部"
    assert [person["user_name"] for person in result.output["trace_scope"]["traced_people"]] == ["张三"]


@pytest.mark.asyncio
async def test_compare_weekly_plan_done_traces_no_blocker_statement_as_empty_risk() -> None:
    client = FakeTrackingBusinessClient()
    tool = CompareWeeklyPlanDoneTool(client=client)
    state = LangGraphState.create(
        request_id="req_skip_empty_risk_trace",
        session_id="sess_skip_empty_risk_trace",
        user_input="上周所有人汇报的卡点是什么",
    )
    state.context.set_task_result(
        "weekly_reports",
        {
            "items": [
                {"user_name": "张三", "department": "研发部", "risk_and_help": "接口联调依赖外部确认"},
                {"user_name": "李四", "department": "研发部", "risk_and_help": "暂无卡点"},
            ],
            "count": 2,
        },
    )

    result = await tool._arun(
        payload={
            "last_week_start": "2026-06-08",
            "last_week_end": "2026-06-14",
            "this_week_start": "2026-06-15",
            "this_week_end": "2026-06-21",
            "trace_only_empty_risk_from_output_key": "weekly_reports",
            "limit": 50,
        },
        context=state.context,
    )

    assert result.success is True
    assert [call["user_name"] for call in client.compare_calls] == ["李四"]
    assert len(result.output["plan_followups"]) == 1
    assert "trace_skipped" not in result.output["pairing_summary"]
    assert result.metadata["trace_user_count"] == 1
    assert result.metadata["trace_skipped_user_count"] == 1
    assert [person["user_name"] for person in result.output["trace_scope"]["traced_people"]] == ["李四"]
    assert result.output["trace_scope"]["traced_people"][0]["reason"] == "no_blocker_statement"
    assert [person["user_name"] for person in result.output["trace_scope"]["skipped_people"]] == ["张三"]
    compact_context = result.output["weekly_blocker_context_text"]
    assert "张三 / 研发部" in compact_context
    assert "李四 / 研发部" in compact_context
    assert "员工自填卡点" in compact_context
    assert "暂无卡点" not in compact_context
    assert "risk_and_help" not in compact_context
    assert "推断证据" in compact_context


@pytest.mark.asyncio
async def test_compare_weekly_plan_done_skips_trace_when_everyone_has_actionable_risk() -> None:
    client = FakeTrackingBusinessClient()
    tool = CompareWeeklyPlanDoneTool(client=client)
    state = LangGraphState.create(
        request_id="req_skip_actionable_risk_trace",
        session_id="sess_skip_actionable_risk_trace",
        user_input="上周所有人汇报的卡点是什么",
    )
    state.context.set_task_result(
        "weekly_reports",
        {
            "items": [
                {"user_name": "张三", "department": "研发部", "risk_and_help": "接口联调依赖外部确认"},
                {"user_name": "李四", "department": "研发部", "risk_and_help": "暂无卡点，但需要软件部协助联调"},
            ],
            "count": 2,
        },
    )

    result = await tool._arun(
        payload={
            "last_week_start": "2026-06-08",
            "last_week_end": "2026-06-14",
            "this_week_start": "2026-06-15",
            "this_week_end": "2026-06-21",
            "trace_only_empty_risk_from_output_key": "weekly_reports",
            "limit": 50,
        },
        context=state.context,
    )

    assert result.success is True
    assert client.compare_calls == []
    assert result.output["pairing_summary"]["trace_skipped"] is True
    assert result.metadata["trace_user_count"] == 0
    assert result.metadata["trace_skipped_user_count"] == 2
    compact_context = result.output["weekly_blocker_context_text"]
    assert "接口联调依赖外部确认" in compact_context
    assert "需要软件部协助联调" in compact_context
    assert "推断证据" not in compact_context


@pytest.mark.asyncio
async def test_classify_weekly_blockers_uses_llm_for_mixed_no_blocker_text() -> None:
    client = FakeWeeklyBlockerLLMClient(
        {
            "items": [
                {
                    "user_name": "张三",
                    "department": "研发部",
                    "raw_risk_and_help": "暂无卡点，但支架未到货，需要采购协调",
                    "classification": "mixed_current_blocker",
                    "has_effective_current_blocker": True,
                    "effective_blocker_text": "支架未到货，需要采购协调",
                    "needs_trace": False,
                    "reason": "否定套话后包含具体待协调事项",
                    "confidence": 0.91,
                },
                {
                    "user_name": "李四",
                    "department": "研发部",
                    "raw_risk_and_help": "暂无卡点",
                    "classification": "no_current_blocker",
                    "has_effective_current_blocker": False,
                    "effective_blocker_text": "",
                    "needs_trace": True,
                    "reason": "仅表示无当前卡点",
                    "confidence": 0.95,
                },
            ]
        }
    )
    tool = ClassifyWeeklyBlockersTool(client=client)
    state = LangGraphState.create(
        request_id="req_classify_mixed_blocker",
        session_id="sess_classify_mixed_blocker",
        user_input="上周所有人的卡点是什么",
    )
    state.context.set_task_result(
        "weekly_reports",
        {
            "items": [
                {"user_name": "张三", "department": "研发部", "risk_and_help": "暂无卡点，但支架未到货，需要采购协调"},
                {"user_name": "李四", "department": "研发部", "risk_and_help": "暂无卡点"},
            ],
            "count": 2,
        },
    )

    result = await tool._arun(payload={"weekly_reports_output_key": "weekly_reports"}, context=state.context)

    assert result.success is True
    output = result.output
    assert output["fallback_used"] is False
    by_name = {item["user_name"]: item for item in output["items"]}
    assert by_name["张三"]["classification"] == "mixed_current_blocker"
    assert by_name["张三"]["has_effective_current_blocker"] is True
    assert by_name["张三"]["needs_trace"] is False
    assert by_name["张三"]["effective_blocker_text"] == "支架未到货，需要采购协调"
    assert by_name["李四"]["classification"] == "no_current_blocker"
    assert by_name["李四"]["needs_trace"] is True
    assert client.requests


@pytest.mark.asyncio
async def test_classify_weekly_blockers_fallback_traces_all_people_on_llm_failure() -> None:
    client = FakeWeeklyBlockerLLMClient(fail=True)
    tool = ClassifyWeeklyBlockersTool(client=client)
    state = LangGraphState.create(
        request_id="req_classify_fallback",
        session_id="sess_classify_fallback",
        user_input="上周所有人的卡点是什么",
    )
    state.context.set_task_result(
        "weekly_reports",
        {
            "items": [
                {"user_name": "张三", "department": "研发部", "risk_and_help": "接口联调依赖外部确认"},
                {"user_name": "李四", "department": "研发部", "risk_and_help": ""},
            ],
            "count": 2,
        },
    )

    result = await tool._arun(payload={"weekly_reports_output_key": "weekly_reports"}, context=state.context)

    assert result.success is True
    assert result.output["fallback_used"] is True
    assert all(item["needs_trace"] is True for item in result.output["items"])
    assert {item["classification"] for item in result.output["items"]} == {"ambiguous", "empty"}
    assert all(not item["effective_blocker_text"] for item in result.output["items"])


@pytest.mark.asyncio
async def test_compare_weekly_plan_done_uses_classification_and_two_trace_windows() -> None:
    client = FakeHistoricalTrackingBusinessClient()
    tool = CompareWeeklyPlanDoneTool(client=client)
    state = LangGraphState.create(
        request_id="req_classified_two_week_trace",
        session_id="sess_classified_two_week_trace",
        user_input="上周所有人汇报的卡点是什么",
    )
    state.context.set_task_result(
        "weekly_blocker_classification",
        {
            "items": [
                {
                    "user_name": "张三",
                    "department": "研发部",
                    "classification": "mixed_current_blocker",
                    "has_effective_current_blocker": True,
                    "effective_blocker_text": "支架未到货，需要采购协调",
                    "needs_trace": False,
                    "reason": "有当前有效卡点",
                    "source_row_count": 1,
                },
                {
                    "user_name": "李四",
                    "department": "研发部",
                    "classification": "no_current_blocker",
                    "has_effective_current_blocker": False,
                    "effective_blocker_text": "",
                    "needs_trace": True,
                    "reason": "未确认当前卡点",
                    "source_row_count": 1,
                },
            ],
            "count": 2,
            "source_output_key": "weekly_reports",
        },
    )

    result = await tool._arun(
        payload={
            "last_week_start": "2026-06-08",
            "last_week_end": "2026-06-14",
            "this_week_start": "2026-06-15",
            "this_week_end": "2026-06-21",
            "weekly_blocker_classification_output_key": "weekly_blocker_classification",
            "trace_weeks": 2,
            "include_historical_blockers": True,
            "limit": 50,
        },
        context=state.context,
    )

    assert result.success is True
    assert [call["user_name"] for call in client.compare_calls] == ["李四", "李四"]
    assert client.compare_calls[0]["last_week_start"] == "2026-06-08"
    assert client.compare_calls[0]["last_week_end"] == "2026-06-14"
    assert client.compare_calls[0]["this_week_start"] == "2026-06-15"
    assert client.compare_calls[0]["this_week_end"] == "2026-06-21"
    assert client.compare_calls[1]["last_week_start"] == "2026-06-01"
    assert client.compare_calls[1]["last_week_end"] == "2026-06-07"
    assert client.compare_calls[1]["this_week_start"] == "2026-06-08"
    assert client.compare_calls[1]["this_week_end"] == "2026-06-21"
    output = result.output
    assert len(output["trace_windows"]) == 2
    assert [person["user_name"] for person in output["trace_scope"]["traced_people"]] == ["李四"]
    assert [person["user_name"] for person in output["trace_scope"]["skipped_people"]] == ["张三"]
    assert len(output["plan_followups"]) == 2
    assert output["plan_followups"][0]["trace_window"]["window_index"] == 1
    assert output["plan_followups"][1]["trace_window"]["window_index"] == 2
    assert output["historical_blocker_candidates"]
    assert output["historical_blocker_candidates"][0]["raw_risk_and_help"] == "供应商接口资料未提供，联调受阻"
    assert output["direct_blocker_people"][0]["blocker_text"] == "支架未到货，需要采购协调"
    assert result.metadata["trace_user_count"] == 1
    assert result.metadata["trace_skipped_user_count"] == 1


@pytest.mark.asyncio
async def test_judge_weekly_blocker_trace_builds_final_context() -> None:
    client = FakeWeeklyBlockerLLMClient(
        {
            "historical_blocker_followups": [
                {
                    "candidate_id": "historical_blocker_1",
                    "status": "resolved",
                    "reason": "后续周报写明已收到资料并完成联调",
                    "evidence": "已收到供应商接口资料并完成联调验证",
                    "confidence": 0.88,
                }
            ]
        }
    )
    tool = JudgeWeeklyBlockerTraceTool(client=client)
    state = LangGraphState.create(
        request_id="req_judge_blocker_trace",
        session_id="sess_judge_blocker_trace",
        user_input="上周所有人的卡点是什么",
    )
    state.context.set_task_result(
        "weekly_blocker_classification",
        {
            "items": [
                {
                    "user_name": "张三",
                    "department": "研发部",
                    "classification": "mixed_current_blocker",
                    "has_effective_current_blocker": True,
                    "effective_blocker_text": "支架未到货，需要采购协调",
                    "needs_trace": False,
                    "reason": "有当前有效卡点",
                },
                {
                    "user_name": "李四",
                    "department": "研发部",
                    "classification": "no_current_blocker",
                    "has_effective_current_blocker": False,
                    "effective_blocker_text": "",
                    "needs_trace": True,
                    "reason": "仅表示无当前卡点",
                },
            ],
            "count": 2,
        },
    )
    state.context.set_task_result(
        "weekly_plan_comparison",
        {
            "query_date_range": {
                "this_week_start": "2026-06-15",
                "this_week_end": "2026-06-21",
                "trace_windows": [
                    {
                        "window_index": 1,
                        "source_plan_week_start": "2026-06-08",
                        "source_plan_week_end": "2026-06-14",
                        "followup_start": "2026-06-15",
                        "followup_end": "2026-06-21",
                    }
                ],
            },
            "trace_scope": {
                "traced_people": [{"user_name": "李四", "department": "研发部"}],
                "skipped_people": [{"user_name": "张三", "department": "研发部"}],
            },
            "plan_followups": [
                {
                    "user_name": "李四",
                    "department": "研发部",
                    "plan_text": "推进供应商接口联调",
                    "done_items": [{"done_text": "已收到供应商接口资料并完成联调验证"}],
                    "trace_window": {"window_index": 1},
                }
            ],
            "historical_blocker_candidates": [
                {
                    "candidate_id": "historical_blocker_1",
                    "user_name": "李四",
                    "department": "研发部",
                    "report_date": "2026-06-08",
                    "raw_risk_and_help": "供应商接口资料未提供，联调受阻",
                    "followup_done_items": [
                        {
                            "report_date": "2026-06-18",
                            "item_type": "this_week_done",
                            "done_text": "已收到供应商接口资料并完成联调验证",
                        }
                    ],
                }
            ],
        },
    )

    result = await tool._arun(
        payload={
            "weekly_blocker_classification_output_key": "weekly_blocker_classification",
            "weekly_plan_comparison_output_key": "weekly_plan_comparison",
        },
        context=state.context,
    )

    assert result.success is True
    output = result.output
    assert output["historical_blocker_followups"][0]["status"] == "resolved"
    context_text = output["weekly_blocker_context_text"]
    assert "张三 / 研发部" in context_text
    assert "支架未到货，需要采购协调" in context_text
    assert "李四 / 研发部" in context_text
    assert "历史卡点追溯" in context_text
    assert "后续已有解决/闭环迹象" in context_text
    assert "risk_and_help" not in context_text



def test_weekly_plan_followup_evidence_pairs_next_report_date_only() -> None:
    plans = [
        {
            "id": 101,
            "user_name": "王浩",
            "department": "九工机器/行政部",
            "report_date": "2026-05-22",
            "item_type": "next_week_plan",
            "item_text": "开始 Runtime Persistence Layer 与 PostgreSQL Redis 数据边界建设",
            "evidence_text": "下周计划：开始 Runtime Persistence Layer 与 PostgreSQL Redis 数据边界建设",
        },
        {
            "id": 102,
            "user_name": "王浩",
            "department": "九工机器/行政部",
            "report_date": "2026-05-29",
            "item_type": "next_week_plan",
            "item_text": "优化分批次压缩总结机制",
            "evidence_text": "下周计划：优化分批次压缩总结机制",
        },
    ]
    completed = [
        {
            "id": 201,
            "user_name": "王浩",
            "department": "九工机器/行政部",
            "report_date": "2026-05-29",
            "item_type": "this_week_done",
            "item_text": "完成 Runtime Persistence Layer 设计，并明确 PostgreSQL Redis 数据边界",
            "evidence_text": "本周完成：完成 Runtime Persistence Layer 设计，并明确 PostgreSQL Redis 数据边界",
        },
        {
            "id": 202,
            "user_name": "王浩",
            "department": "九工机器/行政部",
            "report_date": "2026-06-05",
            "item_type": "this_week_done",
            "item_text": "分批次压缩机制升级",
            "evidence_text": "本周完成：分批次压缩机制升级；优化原有纯文本压缩逻辑",
        },
    ]

    result = _build_weekly_plan_followup_evidence(plans=plans, completed=completed)

    assert [item["completion_date"] for item in result["plan_followups"]] == ["2026-05-29", "2026-06-05"]
    assert [item["plan_id"] for item in result["plan_followups"]] == [101, 102]
    assert result["plan_followups"][0]["done_items"][0]["done_text"].startswith("完成 Runtime Persistence")
    assert result["plan_followups"][1]["candidate_done_items"][0]["done_text"] == "分批次压缩机制升级"
    assert result["pairing_summary"]["judgement_owner"] == "backend_llm"
    assert "plan_checks" not in result
    assert "final_report_type" not in result


def test_weekly_plan_followup_evidence_does_not_use_other_people_done_rows() -> None:
    plans = [
        {
            "id": 101,
            "user_name": "姚豪",
            "department": "产品部",
            "report_date": "2026-06-14",
            "item_type": "next_week_plan",
            "item_text": "对转存架原点校准工装进行安装测试",
            "evidence_text": "下周计划：对转存架原点校准工装进行安装测试",
        },
        {
            "id": 102,
            "user_name": "刘雨康",
            "department": "软件部",
            "report_date": "2026-06-14",
            "item_type": "next_week_plan",
            "item_text": "协助 QC 进行 UI 测试和 bug 处理",
            "evidence_text": "下周计划：协助 QC 进行 UI 测试和 bug 处理",
        },
    ]
    completed = [
        {
            "id": 201,
            "user_name": "刘雨康",
            "department": "软件部",
            "report_date": "2026-06-21",
            "item_type": "this_week_done",
            "item_text": "协助QC进行雷达、红绿灯、UI、ECS/CCS测试及bug处理等",
            "evidence_text": "本周完成：协助QC进行雷达、红绿灯、UI、ECS/CCS测试及bug处理等",
        }
    ]

    result = _build_weekly_plan_followup_evidence(plans=plans, completed=completed)

    by_user = {item["user_name"]: item for item in result["plan_followups"]}
    assert by_user["姚豪"]["completion_date"] is None
    assert by_user["姚豪"]["done_items"] == []
    assert by_user["刘雨康"]["completion_date"] == "2026-06-21"
    assert by_user["刘雨康"]["done_items"][0]["done_text"].startswith("协助QC")
    assert result["pairing_summary"]["plans_without_followup_records"] == 1


def test_weekly_plan_followup_evidence_accepts_short_and_full_department_for_same_person() -> None:
    plans = [
        {
            "id": 101,
            "user_name": "张三",
            "department": "产品部",
            "report_date": "2026-06-14",
            "item_type": "next_week_plan",
            "item_text": "完成扫码枪多类型扫码复测",
            "evidence_text": "下周计划：完成扫码枪多类型扫码复测",
        }
    ]
    completed = [
        {
            "id": 201,
            "user_name": "张三",
            "department": "九工机器/产品部",
            "report_date": "2026-06-21",
            "item_type": "this_week_done",
            "item_text": "完成扫码枪多类型扫码复测并记录问题",
            "evidence_text": "本周完成：完成扫码枪多类型扫码复测并记录问题",
        }
    ]

    result = _build_weekly_plan_followup_evidence(plans=plans, completed=completed)

    followup = result["plan_followups"][0]
    assert followup["completion_date"] == "2026-06-21"
    assert followup["done_items"][0]["department"] == "九工机器/产品部"


def test_judge_plan_followup_rejects_only_generic_or_object_keyword_overlap() -> None:
    judged = _judge_plan_followup(
        {
            "user_name": "李华",
            "department": "制造部",
            "plan_date": "2026-06-14",
            "completion_date": "2026-06-21",
            "plan_text": "推进2.5设备GI/GO信号程序改写调试",
            "done_items": [
                {
                    "done_text": "上周计划对2.5设备通道网络柜进行接线已经完成",
                    "report_date": "2026-06-21",
                }
            ],
            "candidate_done_items": [
                {
                    "done_text": "上周计划对2.5设备通道网络柜进行接线已经完成",
                    "report_date": "2026-06-21",
                    "overlap_keywords": ["2.5设备", ".5设备", "2.5设"],
                    "item_overlap_keywords": ["2.5设备", ".5设备", "2.5设"],
                    "coverage": 0.2,
                    "item_coverage": 0.2,
                    "longest_overlap": 5,
                    "item_longest_overlap": 5,
                }
            ],
        }
    )

    assert judged["status"] == "未完成"
    assert "未找到与该计划直接相关" in judged["reason"]


@pytest.mark.asyncio
async def test_compare_weekly_plan_done_queries_this_week_done_items() -> None:
    client = FakeCompareQueryBusinessClient()
    result = await client.compare_weekly_plan_done(
        user_name="张三",
        department="产品部",
        last_week_start="2026-06-08",
        last_week_end="2026-06-14",
        this_week_start="2026-06-15",
        this_week_end="2026-06-21",
        limit=50,
    )

    assert [call["item_type"] for call in client.query_calls] == [
        "next_week_plan",
        "this_week_work",
        "this_week_done",
    ]
    assert result["this_week_completed"][0]["item_type"] == "this_week_done"
    assert result["plan_followups"][0]["completion_date"] == "2026-06-21"


def test_previous_week_work_against_original_plan_normalizes_to_prior_plan_week() -> None:
    date_range = _normalize_compare_weekly_date_range(
        last_week_start="2026-06-15",
        last_week_end="2026-06-21",
        this_week_start="2026-06-22",
        this_week_end="2026-06-28",
        user_input="上周的工作有没有按原计划完成",
        runtime_timestamp=datetime(2026, 6, 23, 2, 0, 0, tzinfo=timezone.utc),
        timezone_name="Asia/Shanghai",
    )

    assert date_range["adjusted"] is True
    assert date_range["last_week_start"] == "2026-06-08"
    assert date_range["last_week_end"] == "2026-06-14"
    assert date_range["this_week_start"] == "2026-06-15"
    assert date_range["this_week_end"] == "2026-06-21"
    assert date_range["planner_original"]["last_week_start"] == "2026-06-15"


def test_weekly_plan_followup_evidence_keeps_missing_next_report_as_evidence_gap() -> None:
    plans = [
        {
            "user_name": "王浩",
            "department": "九工机器/行政部",
            "report_date": "2026-06-05",
            "item_type": "next_week_plan",
            "item_text": "新增 Agent 扫描 NAS 文件上传工具",
            "evidence_text": "下周计划：新增 Agent 扫描 NAS 文件上传工具",
        }
    ]

    result = _build_weekly_plan_followup_evidence(plans=plans, completed=[])

    followup = result["plan_followups"][0]
    assert followup["completion_date"] is None
    assert followup["done_items"] == []
    assert followup["candidate_done_items"] == []
    assert result["pairing_summary"]["plans_without_followup_records"] == 1


def test_weekly_plan_followup_evidence_candidate_items_are_not_status() -> None:
    plans = [
        {
            "user_name": "王浩",
            "department": "九工机器/行政部",
            "report_date": "2026-05-29",
            "item_type": "next_week_plan",
            "item_text": "优化检索过滤策略",
            "evidence_text": "下周计划：优化检索过滤策略",
        }
    ]
    completed = [
        {
            "user_name": "王浩",
            "department": "九工机器/行政部",
            "report_date": "2026-06-05",
            "item_type": "this_week_done",
            "item_text": "调整检索阶段的过滤策略，进一步提高准确率",
            "evidence_text": "本周完成：调整检索阶段的过滤策略，进一步提高准确率",
        }
    ]

    result = _build_weekly_plan_followup_evidence(plans=plans, completed=completed)

    candidate = result["plan_followups"][0]["candidate_done_items"][0]
    assert candidate["done_text"] == "调整检索阶段的过滤策略，进一步提高准确率"
    assert {"检索", "过滤", "策略"}.issubset(set(candidate["overlap_keywords"]))
    assert "status" not in candidate
    assert "score" not in candidate


def test_compare_weekly_plan_done_output_schema_exposes_llm_evidence_fields() -> None:
    schema = CompareWeeklyPlanDoneTool().get_output_schema()

    assert "weekly_pairs" in schema["properties"]
    assert "plan_followups" in schema["properties"]
    assert "pairing_summary" in schema["properties"]
    assert "plan_tracking_context_text" in schema["properties"]
    assert "final_report" in schema["properties"]
    assert "final_report_type" in schema["properties"]
    assert "plan_checks" not in schema["properties"]
    assert "summary" not in schema["properties"]


def test_weekly_plan_followup_evidence_adds_compact_plan_tracking_context() -> None:
    long_evidence = "证据" * 500
    plans = [
        {
            "id": 101,
            "user_name": "王浩",
            "department": "九工机器/行政部",
            "report_date": "2026-05-22",
            "item_type": "next_week_plan",
            "item_text": "优化检索过滤策略",
            "evidence_text": long_evidence,
        }
    ]
    completed = [
        {
            "id": 201,
            "user_name": "王浩",
            "department": "九工机器/行政部",
            "report_date": "2026-05-29",
            "item_type": "this_week_done",
            "item_text": "调整检索阶段的过滤策略，进一步提高准确率",
            "evidence_text": long_evidence,
        }
    ]

    result = _build_weekly_plan_followup_evidence(plans=plans, completed=completed)
    result["query_date_range"] = {
        "last_week_start": "2026-05-22",
        "last_week_end": "2026-05-22",
        "this_week_start": "2026-05-29",
        "this_week_end": "2026-05-29",
    }
    from app.tools.mysql_business_tools import _attach_plan_tracking_context

    _attach_plan_tracking_context(result=result)

    context_text = result["plan_tracking_context_text"]
    assert "计划追踪压缩上下文" in context_text
    assert "优化检索过滤策略" in context_text
    assert "调整检索阶段的过滤策略" in context_text
    assert len(result["plan_followups"][0]["plan_evidence_text"]) <= 403
    assert len(result["plan_followups"][0]["done_items"][0]["evidence_text"]) <= 403
    assert len(context_text) < 3000
    assert result["final_report_type"] == "weekly_plan_completion"
    assert "计划完成情况报告" in result["final_report"]
    assert "优化检索过滤策略" in result["final_report"]
    assert "调整检索阶段的过滤策略" in result["final_report"]


def test_compare_weekly_progress_detail_omits_large_evidence_payloads() -> None:
    result = {
        "query_date_range": {"last_week_start": "2026-05-01", "last_week_end": "2026-05-31"},
        "last_week_plans": [{"item_text": "计划", "evidence_text": "完整证据" * 1000}],
        "this_week_completed": [{"item_text": "完成", "evidence_text": "完成证据" * 1000}],
        "candidate_matches": [{"plan_text": "计划", "done_text": "完成"}],
        "weekly_pairs": [{"plan_date": "2026-05-08"}],
        "plan_followups": [{"plan_text": "计划", "done_items": [{"done_text": "完成"}]}],
        "trace_scope": {
            "trace_filtered_by_risk_and_help": True,
            "traced_people": [{"user_name": "李四"}],
            "skipped_people": [{"user_name": "张三"}],
        },
        "weekly_blocker_context_text": "卡点上下文",
        "plan_tracking_context_text": "计划上下文",
    }

    detail = _build_compare_weekly_progress_detail(result)

    assert detail["last_week_plan_count"] == 1
    assert detail["this_week_completed_count"] == 1
    assert detail["trace_user_count"] == 1
    assert detail["trace_skipped_user_count"] == 1
    assert detail["weekly_blocker_context_chars"] == len("卡点上下文")
    assert "last_week_plans" not in detail
    assert "this_week_completed" not in detail
    assert "plan_followups" not in detail
    assert "完整证据" not in str(detail)

def test_monthly_weekly_tracking_expands_first_week_planner_window() -> None:
    date_range = _normalize_compare_weekly_date_range(
        last_week_start="2026-05-01",
        last_week_end="2026-05-07",
        this_week_start="2026-05-08",
        this_week_end="2026-05-14",
        user_input="王浩上个月的每周的计划都完成了吗，还有哪些没有完成",
    )

    assert date_range["adjusted"] is True
    assert date_range["last_week_start"] == "2026-05-01"
    assert date_range["last_week_end"] == "2026-05-31"
    assert date_range["this_week_start"] == "2026-05-01"
    assert date_range["this_week_end"] >= "2026-06-30"
    assert date_range["planner_original"]["last_week_end"] == "2026-05-07"


def test_week_to_week_tracking_keeps_exact_planner_window() -> None:
    date_range = _normalize_compare_weekly_date_range(
        last_week_start="2026-06-08",
        last_week_end="2026-06-14",
        this_week_start="2026-06-15",
        this_week_end="2026-06-18",
        user_input="王浩上周的计划完成了吗",
    )

    assert date_range["adjusted"] is False
    assert date_range["last_week_start"] == "2026-06-08"
    assert date_range["last_week_end"] == "2026-06-14"
    assert date_range["this_week_start"] == "2026-06-15"
    assert date_range["this_week_end"] == "2026-06-18"
