from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.agents import SupervisorAgent
from app.context import LocalCheckpointStore, RuntimeCheckpointManager
from app.graph import GraphRuntimeDependencies, build_langgraph_runtime
from app.planner import LLMTaskPlanner
from app.router import TaskRouter
from app.router.capability import capability_from_tool
from app.tools.llm_client import LLMClient
from app.tools.llm_reason import LLMReasonTool
from app.tools.ab_case_search import ABCaseSearchTool
from app.tools.feishu_sync_to_nas_tool import FeishuSyncToNasTool
from app.tools.mysql_business_tools import (
    CompareDeptPlanCompletionTool,
    CompareWeeklyPlanDoneTool,
    MonthlyDepartmentAnalysisTool,
    QueryOplIssuesTool,
    QueryWeeklyReportsTool,
)
from app.tools.rag_batch_summarize import RAGBatchSummarizeTool
from app.tools.rag_search import RAGSearchTool
from app.tools.text_generate import TextGenerateTool
from app.tools.weekly_blocker_tools import ClassifyWeeklyBlockersTool, JudgeWeeklyBlockerTraceTool


class RuntimeComponents(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", validate_assignment=True)

    client: LLMClient = Field(...)
    repair_llm_client: LLMClient = Field(...)
    router: TaskRouter = Field(...)
    supervisor_agent: SupervisorAgent = Field(...)
    planner_agent: LLMTaskPlanner = Field(...)
    checkpoint_manager: RuntimeCheckpointManager | None = Field(default=None)


async def build_runtime_components(
    *,
    client: LLMClient,
    checkpoint_manager: RuntimeCheckpointManager | None = None,
) -> RuntimeComponents:
    router = TaskRouter()
    rag_tool = RAGSearchTool(client=client)
    ab_case_search_tool = ABCaseSearchTool()
    rag_batch_summarize_tool = RAGBatchSummarizeTool(client=client)
    reason_tool = LLMReasonTool(client=client)
    text_tool = TextGenerateTool(client=client)
    classify_weekly_blockers_tool = ClassifyWeeklyBlockersTool(client=client)
    judge_weekly_blocker_trace_tool = JudgeWeeklyBlockerTraceTool(client=client)
    feishu_sync_to_nas_tool = FeishuSyncToNasTool()
    query_weekly_reports_tool = QueryWeeklyReportsTool()
    compare_weekly_plan_done_tool = CompareWeeklyPlanDoneTool()
    monthly_department_analysis_tool = MonthlyDepartmentAnalysisTool()
    compare_dept_plan_completion_tool = CompareDeptPlanCompletionTool()
    query_opl_issues_tool = QueryOplIssuesTool()
    await router.register_tools(
        [
            (rag_tool, capability_from_tool(rag_tool)),
            (ab_case_search_tool, capability_from_tool(ab_case_search_tool)),
            (rag_batch_summarize_tool, capability_from_tool(rag_batch_summarize_tool)),
            (reason_tool, capability_from_tool(reason_tool)),
            (text_tool, capability_from_tool(text_tool)),
            (classify_weekly_blockers_tool, capability_from_tool(classify_weekly_blockers_tool)),
            (judge_weekly_blocker_trace_tool, capability_from_tool(judge_weekly_blocker_trace_tool)),
            (feishu_sync_to_nas_tool, capability_from_tool(feishu_sync_to_nas_tool)),
            (compare_dept_plan_completion_tool, capability_from_tool(compare_dept_plan_completion_tool)),
            (query_opl_issues_tool, capability_from_tool(query_opl_issues_tool)),
            (monthly_department_analysis_tool, capability_from_tool(monthly_department_analysis_tool)),
            (compare_weekly_plan_done_tool, capability_from_tool(compare_weekly_plan_done_tool)),
            (query_weekly_reports_tool, capability_from_tool(query_weekly_reports_tool)),
        ]
    )
    return RuntimeComponents(
        client=client,
        repair_llm_client=client,
        router=router,
        supervisor_agent=SupervisorAgent(client=client),
        planner_agent=LLMTaskPlanner(client=client),
        checkpoint_manager=checkpoint_manager
        or RuntimeCheckpointManager(
            store=LocalCheckpointStore("outputs/runtime_checkpoints"),
            enabled=True,
        ),
    )


def build_graph_runtime(components: RuntimeComponents):
    return build_langgraph_runtime(
        GraphRuntimeDependencies(
            router=components.router,
            supervisor_agent=components.supervisor_agent,
            planner_agent=components.planner_agent,
            repair_llm_client=components.repair_llm_client,
            checkpoint_manager=components.checkpoint_manager,
        )
    )
