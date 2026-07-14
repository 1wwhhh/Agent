"""LangGraph Runtime 的主流程编排图。"""

from __future__ import annotations

import logging
import re
import time
from datetime import date as date_type
from datetime import timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field

from app.agents import SupervisorAgent
from app.context import RuntimeCheckpointManager
from app.executor import TaskExecutor
from app.graph.feishu_preflight import preflight_feishu_sync_tasks
from app.llm.exceptions import LLMInvalidResponseError
from app.planner import LLMTaskPlanner, RepairPipeline, TaskParser
from app.prompts import ToolDefinition, build_supervisor_prompt, build_task_planner_prompt
from app.queue import TaskQueue
from app.router import PermissionContext, TaskRouter
from app.schemas.context import TokenUsage, ToolCallRecord
from app.schemas.graph import GraphPhase
from app.schemas.planner import TaskPlan
from app.schemas.task import TaskModel, TaskStatus, utc_now
from app.state import LangGraphState
from app.tools.base import BaseTool
from app.tools.llm_client import LLMClient
from app.utils import configure_runtime_logger, runtime_log, runtime_progress
from app.utils.time_context import build_runtime_time_context

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:  # pragma: no cover - 允许在未安装 LangGraph 时导入模块
    END = "__end__"
    START = "__start__"
    StateGraph = None


class RuntimeTaskTypeResolutionError(ValueError):
    """Raised when Runtime cannot safely resolve a planned task_type."""


# TODO: remove this compatibility alias after all Planner checkpoints rely on ToolCapability defaults.
TASK_TYPE_ALIASES = {
    "rag_batch_summarize_tool": "rag_batch_summary",
}


def _get_task_field(task: Any, field_name: str) -> Any:
    if isinstance(task, dict):
        return task.get(field_name)
    return getattr(task, field_name, None)


def _with_task_type(task: Any, task_type: str) -> Any:
    if isinstance(task, BaseModel):
        return task.model_copy(update={"task_type": task_type})
    if isinstance(task, dict):
        resolved = dict(task)
        resolved["task_type"] = task_type
        return resolved
    setattr(task, "task_type", task_type)
    return task


def _log_task_type_resolution(
    task: Any,
    *,
    planner_task_type: Any,
    resolved_task_type: str,
) -> None:
    runtime_log(
        layer="runtime_task_type_resolver",
        event="execute",
        data={
            "task_id": _get_task_field(task, "task_id") or "<unknown>",
            "tool": _get_task_field(task, "tool") or "<unknown>",
            "planner_task_type": planner_task_type,
            "resolved_task_type": resolved_task_type,
        },
        level=logging.DEBUG,
    )


def _format_task_type_resolution_error(
    task: Any,
    reason: str,
    *,
    supported_task_types: list[str] | None = None,
) -> RuntimeTaskTypeResolutionError:
    task_id = _get_task_field(task, "task_id") or "<unknown>"
    tool_name = _get_task_field(task, "tool") or "<unknown>"
    original_task_type = _get_task_field(task, "task_type")
    details = [
        f"task_id={task_id}",
        f"tool={tool_name}",
        f"task_type={original_task_type}",
        f"reason={reason}",
    ]
    if supported_task_types is not None:
        details.append(f"supported_task_types={supported_task_types}")
    return RuntimeTaskTypeResolutionError("Runtime task_type resolution failed: " + " | ".join(details))


def resolve_task_type(task: Any, router: TaskRouter) -> Any:
    """Resolve final task_type from Router ToolCapability instead of trusting Planner output."""
    tool_name = _get_task_field(task, "tool")
    if not isinstance(tool_name, str) or not tool_name.strip():
        raise _format_task_type_resolution_error(task, "missing tool")

    capability = router.get_tool_capability(tool_name)
    if capability is None:
        raise _format_task_type_resolution_error(task, "tool is not registered")

    supported_task_types = list(capability.supported_task_types)
    if not supported_task_types:
        raise _format_task_type_resolution_error(
            task,
            "tool capability has empty supported_task_types",
            supported_task_types=supported_task_types,
        )

    default_task_type = capability.default_task_type
    if default_task_type is not None and default_task_type not in supported_task_types:
        raise _format_task_type_resolution_error(
            task,
            "tool capability default_task_type is not in supported_task_types",
            supported_task_types=supported_task_types,
        )

    original_task_type = _get_task_field(task, "task_type")
    if len(supported_task_types) == 1:
        resolved_task_type = supported_task_types[0]
        if default_task_type is not None and default_task_type != resolved_task_type:
            raise _format_task_type_resolution_error(
                task,
                "single-task-type capability default_task_type does not match supported_task_types",
                supported_task_types=supported_task_types,
            )
        _log_task_type_resolution(
            task,
            planner_task_type=original_task_type,
            resolved_task_type=resolved_task_type,
        )
        return _with_task_type(task, resolved_task_type)

    if not isinstance(original_task_type, str) or not original_task_type.strip():
        raise _format_task_type_resolution_error(
            task,
            "multi-task-type tool requires an explicit task_type",
            supported_task_types=supported_task_types,
        )
    if original_task_type not in supported_task_types:
        raise _format_task_type_resolution_error(
            task,
            "task_type is not supported by selected tool",
            supported_task_types=supported_task_types,
        )
    _log_task_type_resolution(
        task,
        planner_task_type=original_task_type,
        resolved_task_type=original_task_type,
    )
    return task


def resolve_planned_tasks_task_type(tasks: list[Any], router: TaskRouter) -> list[Any]:
    return [resolve_task_type(task, router) for task in tasks]


def _normalize_planned_task_type(task: Any) -> Any:
    """Compatibility shim for old checkpoints/tests that only know the legacy alias."""
    task_type = _get_task_field(task, "task_type")
    resolved_task_type = TASK_TYPE_ALIASES.get(task_type)
    if resolved_task_type is None:
        return task
    return _with_task_type(task, resolved_task_type)


def _normalize_planned_task_types(tasks: list[Any]) -> list[Any]:
    return [_normalize_planned_task_type(task) for task in tasks]


class GraphRuntimeDependencies(BaseModel):
    """构建 LangGraph Runtime 所需的依赖集合。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", validate_assignment=True)

    router: TaskRouter = Field(..., description="已完成工具注册的任务路由器。")
    supervisor_agent: SupervisorAgent | None = Field(default=None, description="可选的 LLM Supervisor。")
    planner_tool: BaseTool | None = Field(default=None, description="供 Planner 节点直接调用的工具。")
    planner_agent: LLMTaskPlanner | None = Field(default=None, description="可选的结构化 Planner。")
    task_parser: TaskParser = Field(default_factory=TaskParser, description="负责解析 Planner 输出的解析器。")
    repair_llm_client: LLMClient | None = Field(..., description="Parser repair LLM client.")
    checkpoint_manager: RuntimeCheckpointManager | None = Field(default=None, description="可选的检查点管理器。")
    queue_max_concurrency: int = Field(default=4, gt=0, description="队列允许的最大并发数。")
    simple_word_threshold: int = Field(default=12, gt=0, description="简单任务启发式阈值。")
    default_simple_tool_name: str = Field(default="text_generate_tool", min_length=1)
    default_reason_tool_name: str = Field(default="llm_reason_tool", min_length=1)
    planner_result_override: str | None = Field(
        default=None,
        description="测试场景下可注入固定 Planner 输出。",
    )
    planner_model_name: str | None = Field(default=None, description="Planner 节点调用工具时使用的模型名。")
    planner_temperature: float = Field(default=0.2, ge=0.0, le=2.0)


def _looks_like_weekly_blocker_query(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False
    has_previous_week_scope = any(token in normalized for token in ("上周", "上一周", "上星期", "上个周"))
    has_blocker_intent = any(token in normalized for token in ("卡点", "风险", "求助", "阻塞", "卡在", "问题"))
    return has_previous_week_scope and has_blocker_intent


def _looks_like_plan_tracking_query(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False
    has_work_completion_intent = (
        any(token in normalized for token in ("上周", "上一周", "上星期", "上个周", "本周", "这个周"))
        and any(token in normalized for token in ("工作", "完成记录", "完成内容"))
        and any(token in normalized for token in ("完成", "没完成", "未完成"))
    )
    has_plan_intent = "计划" in normalized or has_work_completion_intent
    has_completion_intent = any(token in normalized for token in ("完成", "没完成", "未完成", "延期", "闭环", "落地"))
    has_time_scope = any(
        token in normalized
        for token in ("上周", "上一周", "上星期", "上个周", "本周", "这个周", "本月", "这个月", "上个月", "月度", "每周")
    ) or bool(re.search(r"\d{4}年\d{1,2}月|\d{1,2}月", normalized))
    return has_plan_intent and has_completion_intent and has_time_scope


def _looks_like_dept_plan_completion_query(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False
    has_dept_plan_scope = any(token in normalized for token in ("三七计划", "计划书", "部门计划", "月度计划"))
    has_completion_intent = any(
        token in normalized
        for token in ("完成", "没完成", "未完成", "完成率", "落地", "落实", "执行", "闭环")
    )
    return has_dept_plan_scope and has_completion_intent


def _looks_like_ab_case_query(text: str) -> bool:
    normalized = re.sub(r"\s+", "", str(text or "").strip().lower())
    if not normalized:
        return False
    has_case_scope = any(
        token in normalized
        for token in (
            "a/b案例",
            "ab案例",
            "a\\b案例",
            "a／b案例",
            "a/bcase",
            "abcase",
            "a\\bcase",
            "a案例",
            "b案例",
            "奖励案例",
            "惩罚案例",
            "处罚案例",
            "扣罚案例",
            "正向案例",
            "负向案例",
            "评分案例",
            "打分案例",
            "案例打分",
            "相似案例",
            "历史案例",
        )
    )
    has_case_action = any(
        token in normalized
        for token in (
            "检索",
            "搜索",
            "查找",
            "查询",
            "参考",
            "相似",
            "类似",
            "评分",
            "打分",
            "奖励",
            "惩罚",
            "处罚",
            "扣罚",
            "表扬",
            "加分",
            "扣分",
            "判断",
            "怎么判",
            "怎么评",
            "算a",
            "算b",
        )
    )
    return has_case_scope and has_case_action


def _looks_like_runtime_internal_disclosure_request(text: str) -> bool:
    normalized = re.sub(r"\s+", "", str(text or "").strip().lower())
    if not normalized:
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
        "sourcecode",
        "systemprompt",
        "internalprompt",
        "tooldefinition",
        "toolinterface",
        "implementation",
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
        "you",
        "your",
        "thisproject",
        "currentsystem",
        "thissystem",
        "thisruntime",
    )
    if any(pattern in normalized for pattern in strong_patterns) and any(
        subject in normalized for subject in runtime_subjects
    ):
        return True
    return any(
        pattern in normalized
        for pattern in (
            "底层源码",
            "底层代码",
            "内部代码",
            "系统提示词",
            "内部提示词",
        )
    )


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _ensure_min_input_int(task_input: dict[str, Any], key: str, minimum: int) -> bool:
    existing = _positive_int(task_input.get(key))
    if existing is not None and existing >= minimum:
        return False
    task_input[key] = minimum
    return True


def _sanitize_weekly_blocker_generation_prompt(value: Any) -> str:
    text = str(value or "").strip()
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


def _previous_week_range(state: LangGraphState) -> tuple[Any, Any]:
    timestamp = state.context.runtime.timestamp
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    timezone_name = str(state.context.runtime.metadata.get("client_timezone") or "UTC")
    try:
        local_tz = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        local_tz = timezone.utc
    local_date = timestamp.astimezone(local_tz).date()
    this_week_start = local_date - timedelta(days=local_date.weekday())
    target_start = this_week_start - timedelta(days=7)
    target_end = this_week_start - timedelta(days=1)
    return target_start, target_end


def _extract_month_from_text(text: str, *, state: LangGraphState) -> str:
    normalized = text.strip()
    full_match = re.search(r"(?P<year>\d{4})年(?P<month>\d{1,2})月", normalized)
    if full_match:
        return f"{int(full_match.group('year')):04d}-{int(full_match.group('month')):02d}"

    short_match = re.search(r"(?<!\d)(?P<month>\d{1,2})月", normalized)
    if short_match:
        timestamp = state.context.runtime.timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        timezone_name = str(state.context.runtime.metadata.get("client_timezone") or "UTC")
        try:
            local_tz = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError:
            local_tz = timezone.utc
        local_date = timestamp.astimezone(local_tz).date()
        return f"{local_date.year:04d}-{int(short_match.group('month')):02d}"

    chinese_match = re.search(r"(?P<month>十[一二]?|[一二三四五六七八九])月(?:份)?", normalized)
    if chinese_match:
        chinese_months = {
            "一": 1,
            "二": 2,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
            "十": 10,
            "十一": 11,
            "十二": 12,
        }
        timestamp = state.context.runtime.timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        timezone_name = str(state.context.runtime.metadata.get("client_timezone") or "UTC")
        try:
            local_tz = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError:
            local_tz = timezone.utc
        local_date = timestamp.astimezone(local_tz).date()
        return f"{local_date.year:04d}-{chinese_months[chinese_match.group('month')]:02d}"

    return state.context.runtime.timestamp.date().strftime("%Y-%m")


def _month_date_range(month: str) -> tuple[date_type, date_type]:
    year_text, month_text = month.split("-", 1)
    start = date_type(int(year_text), int(month_text), 1)
    if start.month == 12:
        next_month = date_type(start.year + 1, 1, 1)
    else:
        next_month = date_type(start.year, start.month + 1, 1)
    return start, next_month - timedelta(days=1)


def build_langgraph_runtime(dependencies: GraphRuntimeDependencies):
    """根据依赖构建完整 LangGraph Runtime。"""
    if StateGraph is None:
        raise RuntimeError("当前环境未安装 LangGraph，请先安装 `langgraph`。")

    builder = _RuntimeGraphBuilder(dependencies=dependencies)
    return builder.build()


class _RuntimeGraphBuilder:
    """Runtime 图构建与节点实现的内部封装。"""

    def __init__(self, *, dependencies: GraphRuntimeDependencies) -> None:
        self.dependencies = dependencies
        self.logger = configure_runtime_logger()

    def build(self):
        """注册节点与边，编译出 LangGraph 工作流。"""
        workflow = StateGraph(LangGraphState)
        workflow.add_node("supervisor", self.supervisor_node)
        workflow.add_node("planner", self.planner_node)
        workflow.add_node("simple_task", self.simple_task_node)
        workflow.add_node("parser", self.parser_node)
        workflow.add_node("queue", self.queue_node)
        workflow.add_node("executor", self.executor_node)
        workflow.add_node("aggregator", self.aggregator_node)

        workflow.add_edge(START, "supervisor")
        workflow.add_conditional_edges(
            "supervisor",
            self.route_after_supervisor,
            {
                "simple_task": "simple_task",
                "planner": "planner",
                "aggregator": "aggregator",
            },
        )
        workflow.add_conditional_edges(
            "planner",
            self.route_after_planner,
            {
                "parser": "parser",
                "aggregator": "aggregator",
            },
        )
        workflow.add_conditional_edges(
            "simple_task",
            self.route_after_simple_task,
            {
                "queue": "queue",
                "aggregator": "aggregator",
            },
        )
        workflow.add_conditional_edges(
            "parser",
            self.route_after_parser,
            {
                "queue": "queue",
                "aggregator": "aggregator",
            },
        )
        workflow.add_conditional_edges(
            "queue",
            self.route_after_queue,
            {
                "executor": "executor",
                "aggregator": "aggregator",
            },
        )
        workflow.add_edge("executor", "aggregator")
        workflow.add_edge("aggregator", END)

        return workflow.compile()

    async def supervisor_node(self, state: LangGraphState) -> LangGraphState:
        return await self._run_node("supervisor", state, GraphPhase.SUPERVISED, self._supervisor_impl)

    async def planner_node(self, state: LangGraphState) -> LangGraphState:
        return await self._run_node("planner", state, GraphPhase.PLANNED, self._planner_impl)

    async def simple_task_node(self, state: LangGraphState) -> LangGraphState:
        return await self._run_node("simple_task", state, GraphPhase.PLANNED, self._simple_task_impl)

    async def parser_node(self, state: LangGraphState) -> LangGraphState:
        return await self._run_node("parser", state, GraphPhase.PARSED, self._parser_impl)

    async def queue_node(self, state: LangGraphState) -> LangGraphState:
        return await self._run_node("queue", state, GraphPhase.QUEUED, self._queue_impl)

    async def executor_node(self, state: LangGraphState) -> LangGraphState:
        return await self._run_node("executor", state, GraphPhase.EXECUTING, self._executor_impl)

    async def aggregator_node(self, state: LangGraphState) -> LangGraphState:
        return await self._run_node("aggregator", state, GraphPhase.AGGREGATING, self._aggregator_impl)

    def route_after_supervisor(self, state: LangGraphState) -> str:
        """根据 Supervisor 结果决定下一跳。"""
        if state.phase == GraphPhase.FAILED:
            return "aggregator"
        return "simple_task" if state.supervisor_route == "SIMPLE_TASK" else "planner"

    def route_after_planner(self, state: LangGraphState) -> str:
        """Planner 失败时直接进入聚合，否则交给 Parser。"""
        return "aggregator" if state.phase == GraphPhase.FAILED else "parser"

    def route_after_simple_task(self, state: LangGraphState) -> str:
        """简单任务生成后进入 Queue。"""
        return "aggregator" if state.phase == GraphPhase.FAILED else "queue"

    def route_after_parser(self, state: LangGraphState) -> str:
        """Parser 成功后进入 Queue。"""
        if state.phase == GraphPhase.FAILED or state.metadata.get("need_clarification"):
            return "aggregator"
        return "queue"

    def route_after_queue(self, state: LangGraphState) -> str:
        """Queue 成功后进入 Executor。"""
        return "aggregator" if state.phase == GraphPhase.FAILED else "executor"

    async def _run_node(
        self,
        node_name: str,
        state: LangGraphState,
        phase: GraphPhase,
        handler,
    ) -> LangGraphState:
        """统一包装节点执行、计时、错误处理与结构化日志。"""
        previous_phase = state.phase
        effective_phase = state.phase if state.phase == GraphPhase.FAILED else phase
        state.set_current_node(node_name, phase=effective_phase)
        started_at = time.perf_counter()
        runtime_log(
            layer=node_name,
            event="start",
            data={
                "phase": effective_phase.value,
                "phase_transition": f"{previous_phase.value}->{effective_phase.value}",
            },
            logger=self.logger,
        )
        runtime_progress(step=node_name, status="开始")
        try:
            await handler(state)
        except Exception as exc:
            runtime_log(
                layer=node_name,
                event="error",
                data={"error": str(exc), "exception_type": type(exc).__name__},
                level=logging.ERROR,
                logger=self.logger,
            )
            state.set_failed(
                str(exc),
                details={"node": node_name, "exception_type": type(exc).__name__},
            )
        finally:
            duration_ms = round((time.perf_counter() - started_at) * 1000, 3)
            node_timings = state.metadata.setdefault("node_timings", {})
            node_timings[node_name] = {"duration_ms": duration_ms, "phase": state.phase.value}
            completed_nodes = state.metadata.setdefault("completed_nodes", [])
            if node_name not in completed_nodes:
                completed_nodes.append(node_name)
            await self._save_checkpoint(
                state,
                source_layer=node_name,
                event="end",
                metadata={"phase": state.phase.value, "duration_ms": duration_ms},
            )
            state.mark_node_completed(node_name)
            runtime_log(
                layer=node_name,
                event="end",
                data={
                    "phase": state.phase.value,
                    "phase_transition": f"{effective_phase.value}->{state.phase.value}",
                },
                latency_ms=duration_ms,
                logger=self.logger,
            )
            _prog_status = "失败" if state.phase == GraphPhase.FAILED else "完成"
            runtime_progress(step=node_name, status=_prog_status, detail=f"耗时 {duration_ms:.0f}ms")
        return state

    async def _supervisor_impl(self, state: LangGraphState) -> None:
        """执行任务入口判定。"""
        if state.metadata.get("resume_from_checkpoint") and state.supervisor_route:
            runtime_log(
                layer="supervisor",
                event="success",
                data={"task_type": state.supervisor_route, "mode": "checkpoint_resume"},
                logger=self.logger,
            )
            runtime_progress(step="supervisor", status="路由决定", detail="沿用已保存的处理方式")
            return

        override = state.metadata.get("force_route")
        if isinstance(override, str) and override.strip():
            route = override.strip().upper()
            if route not in {"SIMPLE_TASK", "COMPLEX_TASK"}:
                raise ValueError("force_route 只允许 SIMPLE_TASK 或 COMPLEX_TASK。")
            state.set_supervisor_route(route)
            runtime_log(
                layer="supervisor",
                event="success",
                data={"task_type": route, "mode": "override"},
                logger=self.logger,
            )
            runtime_progress(step="supervisor", status="路由决定", detail="使用指定处理方式")
            return

        if self.dependencies.supervisor_agent is not None:
            prompt_bundle = build_supervisor_prompt(
                user_input=state.context.runtime.user_input,
                context_summary=self._build_context_summary(state),
            )
            decision_result = await self.dependencies.supervisor_agent.classify(
                user_input=state.context.runtime.user_input,
                request_id=state.context.runtime.request_id,
                session_id=state.context.runtime.session_id,
                context_summary=self._build_context_summary(state),
            )
            state.context.set_shared_value(
                "supervisor_prompt",
                prompt_bundle.model_dump(mode="json"),
            )
            state.context.set_shared_value(
                "supervisor_decision",
                decision_result.decision.model_dump(mode="json"),
            )
            state.context.set_shared_value(
                "supervisor_trace",
                {
                    "trace_id": decision_result.trace_id,
                    "model_name": decision_result.model_name,
                    "model_version": decision_result.model_version,
                    "prompt_name": decision_result.prompt_name,
                    "prompt_version": decision_result.prompt_version,
                },
            )
            state.set_supervisor_route(decision_result.decision.route)
            runtime_log(
                layer="supervisor",
                event="success",
                data={
                    "task_type": decision_result.decision.route,
                    "model_name": decision_result.model_name,
                    "prompt_version": decision_result.prompt_version,
                },
                logger=self.logger,
            )
            runtime_progress(step="supervisor", status="路由决定", detail="已确定处理方式")
            return

        user_input = state.context.runtime.user_input.strip()
        normalized = user_input.lower()
        word_count = len(user_input.split())
        complex_signals = [
            " and ",
            " then ",
            " after ",
            " report",
            "plan",
            "workflow",
            "dag",
            "analyze",
            "research",
            "compare",
            "summarize",
            "ppt",
            "excel",
        ]
        is_complex = (
            _looks_like_ab_case_query(user_input)
            or word_count > self.dependencies.simple_word_threshold
            or any(signal in normalized for signal in complex_signals)
        )
        state.set_supervisor_route("COMPLEX_TASK" if is_complex else "SIMPLE_TASK")
        runtime_log(
            layer="supervisor",
            event="success",
            data={
                "task_type": state.supervisor_route,
                "word_count": word_count,
                "mode": "heuristic",
            },
            logger=self.logger,
        )
        runtime_progress(step="supervisor", status="路由决定", detail="已确定处理方式")

    async def _planner_impl(self, state: LangGraphState) -> None:
        """为复杂任务生成结构化 TaskPlan。"""
        if state.metadata.get("resume_from_checkpoint") and state.raw_plan_text:
            runtime_log(
                layer="planner",
                event="success",
                data={"mode": "checkpoint_resume", "raw_plan_length": len(state.raw_plan_text)},
                logger=self.logger,
            )
            return

        if self.dependencies.planner_agent is not None:
            self._attach_llm_call_recorder_to_client(client=self.dependencies.planner_agent.client, state=state)
            tool_catalog = await self._build_tool_catalog()
            context_summary = self._build_context_summary(state)
            try:
                planner_result = await self.dependencies.planner_agent.plan(
                    user_input=state.context.runtime.user_input,
                    request_id=state.context.runtime.request_id,
                    session_id=state.context.runtime.session_id,
                    tools=tool_catalog,
                    context_summary=context_summary,
                )
            except LLMInvalidResponseError as exc:
                fallback_result = self._build_planner_fallback_plan(state=state)
                if fallback_result is None:
                    raise
                fallback_type, fallback_plan = fallback_result
                prompt_bundle = build_task_planner_prompt(
                    user_input=state.context.runtime.user_input,
                    tools=tool_catalog,
                    context_summary=context_summary,
                )
                state.set_planner_prompt(prompt_bundle)
                state.set_raw_plan_text(fallback_plan.model_dump_json())
                state.metadata["planner_fallback"] = {
                    "reason": type(exc).__name__,
                    "message": str(exc),
                    "fallback_type": fallback_type,
                }
                runtime_log(
                    layer="planner",
                    event="success",
                    data={
                        "fallback_type": fallback_type,
                        "task_count": len(fallback_plan.tasks),
                        "error": str(exc),
                    },
                    logger=self.logger,
                )
                runtime_progress(step="planner", status="兜底规划完成", detail=f"生成 {len(fallback_plan.tasks)} 个任务")
                return
            prompt_bundle = build_task_planner_prompt(
                user_input=state.context.runtime.user_input,
                tools=tool_catalog,
                context_summary=context_summary,
            )
            state.set_planner_prompt(prompt_bundle)
            state.context.set_shared_value(
                "planner_trace",
                {
                    "trace_id": planner_result.trace_id,
                    "model_name": planner_result.model_name,
                    "model_version": planner_result.model_version,
                    "prompt_name": planner_result.prompt_name,
                    "prompt_version": planner_result.prompt_version,
                },
            )
            state.set_raw_plan_text(planner_result.task_plan.model_dump_json())
            runtime_log(
                layer="planner",
                event="success",
                data={
                    "task_count": len(planner_result.task_plan.tasks),
                    "model_name": planner_result.model_name,
                    "prompt_version": planner_result.prompt_version,
                },
                logger=self.logger,
            )
            runtime_log(
                layer="planner",
                event="execute",
                data={"task_graph": planner_result.task_plan.model_dump(mode="json")},
                level=logging.DEBUG,
                logger=self.logger,
            )
            runtime_progress(step="planner", status="规划完成", detail=f"生成 {len(planner_result.task_plan.tasks)} 个任务")
            return

        if self.dependencies.planner_result_override is not None:
            state.set_raw_plan_text(self.dependencies.planner_result_override)
            runtime_log(
                layer="planner",
                event="success",
                data={"mode": "override", "raw_plan_length": len(self.dependencies.planner_result_override)},
                logger=self.logger,
            )
            return

        if self.dependencies.planner_tool is None:
            raise RuntimeError("Planner 节点缺少可用的 planner_agent、planner_result_override 或 planner_tool。")

        tool_catalog = await self._build_tool_catalog()
        prompt_bundle = build_task_planner_prompt(
            user_input=state.context.runtime.user_input,
            tools=tool_catalog,
            context_summary=self._build_context_summary(state),
        )
        state.set_planner_prompt(prompt_bundle)

        planner_result = await self.dependencies.planner_tool.arun(
            payload={
                "prompt": prompt_bundle.user_prompt,
                "system_prompt": prompt_bundle.system_prompt,
                "model_name": self.dependencies.planner_model_name,
                "temperature": self.dependencies.planner_temperature,
            },
            context=state.context,
        )
        if not planner_result.success:
            raise RuntimeError(planner_result.error or "Planner 工具执行失败。")

        self._record_graph_tool_result(state, planner_result, task_id=None, status=TaskStatus.SUCCESS.value)
        raw_plan_text = self._extract_text_output(planner_result.output)
        state.set_raw_plan_text(raw_plan_text)
        runtime_log(
            layer="planner",
            event="success",
            data={
                "mode": "tool",
                "tool_name": planner_result.tool_name,
                "raw_plan_length": len(raw_plan_text),
            },
            logger=self.logger,
        )

    async def _simple_task_impl(self, state: LangGraphState) -> None:
        """把简单任务包装成单任务 TaskPlan。"""
        user_input = state.context.runtime.user_input.strip()
        if _looks_like_runtime_internal_disclosure_request(user_input):
            tool_name = self.dependencies.default_simple_tool_name
        else:
            tool_name = await self._resolve_simple_tool_name(user_input)
        task = TaskModel(
            task_id="task_1",
            task_name="complete_simple_request",
            description="使用单个可执行步骤完成简单用户请求。",
            tool=tool_name,
            input={"prompt": user_input},
            output_key="final_result",
            depends_on=[],
            priority=1,
            status=TaskStatus.PENDING,
            retry_count=0,
            max_retry=1,
            timeout=60,
            created_at=utc_now(),
        )
        resolved_tasks = self._resolve_runtime_task_types([task])
        plan = TaskPlan.model_validate(
            {
                "goal": user_input,
                "tasks": [resolved_task.model_dump(mode="json") for resolved_task in resolved_tasks],
            }
        )
        state.set_parsed_plan(plan)
        state.set_planned_tasks(resolved_tasks)
        runtime_log(
            layer="simple_task",
            event="success",
            data={"task_id": task.task_id, "tool": tool_name, "output_key": task.output_key},
            logger=self.logger,
        )
        runtime_progress(step="simple_task", status="处理方式", detail="已选择单步处理")

    async def _parser_impl(self, state: LangGraphState) -> None:
        """把 Planner 原始输出解析为运行时任务列表。"""
        if state.metadata.get("resume_from_checkpoint") and state.parsed_plan is not None and state.planned_tasks:
            resolved_tasks = self._resolve_state_planned_tasks(state)
            runtime_log(
                layer="parser",
                event="success",
                data={"task_count": len(resolved_tasks), "mode": "checkpoint_resume"},
                logger=self.logger,
            )
            return

        if not state.raw_plan_text:
            raise ValueError("Parser 节点未收到 Planner 原始输出。")
        self._attach_llm_call_recorder_to_client(client=self.dependencies.repair_llm_client, state=state)
        parser = self.dependencies.task_parser.with_tool_capabilities(await self._build_tool_capability_map())
        repair_pipeline = RepairPipeline(
            parser=parser,
            repair_llm_client=self.dependencies.repair_llm_client,
        )
        parser_result = await repair_pipeline.run(raw_planner_output=state.raw_plan_text, state=state)
        state.set_parsed_plan(parser_result.raw_plan)
        preflight_result = preflight_feishu_sync_tasks(parser_result.tasks)
        if preflight_result.warnings:
            state.metadata["feishu_sync_preflight_warnings"] = preflight_result.warnings
        if not preflight_result.ok:
            state.metadata["need_clarification"] = True
            state.metadata["clarification_response"] = preflight_result.response or {}
            state.set_planned_tasks([])
            runtime_log(
                layer="parser",
                event="clarification_required",
                data=preflight_result.response or {},
                logger=self.logger,
            )
            return

        normalized_tasks = []
        if isinstance(preflight_result.response, dict):
            raw_tasks = preflight_result.response.get("tasks", parser_result.tasks)
            normalized_tasks = list(raw_tasks) if isinstance(raw_tasks, list) else parser_result.tasks
        else:
            normalized_tasks = parser_result.tasks
        normalized_tasks = self._resolve_runtime_task_types(normalized_tasks)
        normalized_tasks = self._preserve_rag_user_queries(state, normalized_tasks)
        normalized_tasks = self._prefer_weekly_report_records_for_blocker_queries(state, normalized_tasks)
        normalized_tasks = self._compact_weekly_blocker_generation_context(state, normalized_tasks)
        normalized_tasks = self._compact_plan_tracking_generation_context(state, normalized_tasks)
        normalized_tasks = self._compact_dept_plan_completion_generation_context(state, normalized_tasks)
        state.set_planned_tasks(normalized_tasks)
        runtime_log(
            layer="parser",
            event="success",
            data={"task_count": len(normalized_tasks)},
            logger=self.logger,
        )
        runtime_progress(step="parser", status="解析完成", detail=f"共 {len(normalized_tasks)} 个任务")
        runtime_log(
            layer="parser",
            event="execute",
            data={"task_graph": state.parsed_plan.model_dump(mode="json") if state.parsed_plan else {}},
            level=logging.DEBUG,
            logger=self.logger,
        )

    def _resolve_runtime_task_types(self, tasks: list[Any]) -> list[TaskModel]:
        resolved_tasks = resolve_planned_tasks_task_type(list(tasks), self.dependencies.router)
        return [task if isinstance(task, TaskModel) else TaskModel.model_validate(task) for task in resolved_tasks]

    def _preserve_rag_user_queries(self, state: LangGraphState, tasks: list[TaskModel]) -> list[TaskModel]:
        user_input = state.context.runtime.user_input.strip()
        if not user_input:
            return tasks

        normalized: list[TaskModel] = []
        changed_task_ids: list[str] = []
        for task in tasks:
            if task.tool not in {"rag_search_tool", "rag_batch_summarize_tool", "ab_case_search_tool"}:
                normalized.append(task)
                continue
            task_input = dict(task.input or {})
            current_query = str(task_input.get("query") or "").strip()
            if current_query == user_input:
                normalized.append(task)
                continue
            task_input.setdefault("planner_query", current_query)
            task_input["query"] = user_input
            task_input["raw_user_query"] = user_input
            normalized.append(task.model_copy(update={"input": task_input}))
            changed_task_ids.append(task.task_id)

        if changed_task_ids:
            history = list(state.metadata.get("rag_query_preservation", []))
            history.append({"task_ids": changed_task_ids, "query": user_input})
            state.metadata["rag_query_preservation"] = history
            runtime_log(
                layer="rag_query_preservation",
                event="execute",
                data={"task_ids": changed_task_ids, "query": user_input},
                logger=self.logger,
            )
        return normalized


    def _compact_weekly_blocker_generation_context(self, state: LangGraphState, tasks: list[TaskModel]) -> list[TaskModel]:
        if not _looks_like_weekly_blocker_query(state.context.runtime.user_input):
            return tasks

        judge_tasks = {
            task.task_id: task
            for task in tasks
            if task.tool == "judge_weekly_blocker_trace"
        }
        compare_tasks = {
            task.task_id: task
            for task in tasks
            if task.tool == "compare_weekly_plan_done"
            and isinstance(task.input, dict)
            and (
                str(task.input.get("trace_only_empty_risk_from_output_key") or "").strip()
                or str(task.input.get("weekly_blocker_classification_output_key") or "").strip()
            )
        }
        if not judge_tasks and not compare_tasks:
            return tasks

        normalized: list[TaskModel] = []
        changed_task_ids: list[str] = []
        for task in tasks:
            if task.tool != "text_generate_tool":
                normalized.append(task)
                continue

            context_task = next((judge_tasks[task_id] for task_id in task.depends_on if task_id in judge_tasks), None)
            context_field = "weekly_blocker_context_text"
            if context_task is None:
                context_task = next((compare_tasks[task_id] for task_id in task.depends_on if task_id in compare_tasks), None)
            if context_task is None:
                normalized.append(task)
                continue

            desired_context = f"{{{{{context_task.output_key}.{context_field}}}}}"
            task_input = dict(task.input or {})
            changed = False
            if task_input.get("context") != desired_context:
                task_input["context"] = desired_context
                changed = True

            prompt = str(task_input.get("prompt") or "").strip()
            prompt = _sanitize_weekly_blocker_generation_prompt(prompt)
            reminder = (
                "上下文是压缩后的逐人卡点语义判断和追溯证据；请区分当前有效员工自填卡点、未确认当前卡点后的计划追溯证据、以及历史卡点是否已解决的判断。最终回答使用业务表述，不要出现数据库字段名、内部输出键或英文技术字段名。"
            )
            if reminder not in prompt:
                prompt = f"{prompt}\n{reminder}".strip() if prompt else reminder
            if task_input.get("prompt") != prompt:
                task_input["prompt"] = prompt
                changed = True

            normalized.append(task.model_copy(update={"input": task_input}) if changed else task)
            if changed:
                changed_task_ids.append(task.task_id)

        if changed_task_ids:
            history = list(state.metadata.get("weekly_blocker_context_rewrite", []))
            history.append(
                {
                    "task_ids": changed_task_ids,
                    "context": "weekly_blocker_context_text",
                    "reason": "avoid_full_weekly_report_context_timeout",
                }
            )
            state.metadata["weekly_blocker_context_rewrite"] = history
            runtime_log(
                layer="weekly_blocker_context_rewrite",
                event="execute",
                data=history[-1],
                logger=self.logger,
            )
        return normalized

    def _prefer_weekly_report_records_for_blocker_queries(self, state: LangGraphState, tasks: list[TaskModel]) -> list[TaskModel]:
        if not _looks_like_weekly_blocker_query(state.context.runtime.user_input):
            return tasks

        normalized: list[TaskModel] = []
        changed_task_ids: list[str] = []
        for task in tasks:
            if task.tool != "query_weekly_reports":
                normalized.append(task)
                continue
            task_input = dict(task.input or {})
            if task_input.get("record_level") == "reports":
                normalized.append(task)
                continue
            task_input["record_level"] = "reports"
            task_input["include_evidence_text"] = False
            normalized.append(task.model_copy(update={"input": task_input}))
            changed_task_ids.append(task.task_id)

        if changed_task_ids:
            history = list(state.metadata.get("weekly_blocker_report_level_rewrite", []))
            history.append(
                {
                    "task_ids": changed_task_ids,
                    "record_level": "reports",
                    "reason": "weekly_blocker_queries_should_read_risk_and_help_from_weekly_reports",
                }
            )
            state.metadata["weekly_blocker_report_level_rewrite"] = history
            runtime_log(
                layer="weekly_blocker_report_level_rewrite",
                event="execute",
                data=history[-1],
                logger=self.logger,
            )
        return normalized

    def _compact_plan_tracking_generation_context(self, state: LangGraphState, tasks: list[TaskModel]) -> list[TaskModel]:
        if not _looks_like_plan_tracking_query(state.context.runtime.user_input):
            return tasks

        compare_tasks = {
            task.task_id: task
            for task in tasks
            if task.tool == "compare_weekly_plan_done"
            and isinstance(task.input, dict)
            and not str(task.input.get("trace_only_empty_risk_from_output_key") or "").strip()
            and not str(task.input.get("weekly_blocker_classification_output_key") or "").strip()
        }
        if not compare_tasks:
            return tasks

        normalized: list[TaskModel] = []
        changed_task_ids: list[str] = []
        for task in tasks:
            if task.tool != "text_generate_tool":
                normalized.append(task)
                continue

            compare_task = next((compare_tasks[task_id] for task_id in task.depends_on if task_id in compare_tasks), None)
            if compare_task is None:
                normalized.append(task)
                continue

            desired_context = f"{{{{{compare_task.output_key}.plan_tracking_context_text}}}}"
            task_input = dict(task.input or {})
            changed = False
            if task_input.get("context") != desired_context:
                task_input["context"] = desired_context
                prompt = str(task_input.get("prompt") or "").strip()
                reminder = (
                    "上下文来自 plan_tracking_context_text 压缩字段；请逐条依据计划内容、后续完成项和候选完成项判断已完成、部分完成、未完成或证据不足。"
                )
                if reminder not in prompt:
                    task_input["prompt"] = f"{prompt}\n{reminder}".strip() if prompt else reminder
                changed = True

            changed = _ensure_min_input_int(task_input, "timeout_seconds", 360) or changed
            changed = _ensure_min_input_int(task_input, "tool_timeout_seconds", 360) or changed
            changed = _ensure_min_input_int(task_input, "executor_timeout_seconds", 360) or changed
            if "temperature" not in task_input:
                task_input["temperature"] = 0.2
                changed = True

            normalized.append(task.model_copy(update={"input": task_input}) if changed else task)
            if changed:
                changed_task_ids.append(task.task_id)

        if changed_task_ids:
            history = list(state.metadata.get("plan_tracking_context_rewrite", []))
            history.append(
                {
                    "task_ids": changed_task_ids,
                    "context": "plan_tracking_context_text",
                    "reason": "avoid_full_weekly_plan_comparison_context_and_generation_timeout",
                    "timeout_seconds": 360,
                    "tool_timeout_seconds": 360,
                    "executor_timeout_seconds": 360,
                }
            )
            state.metadata["plan_tracking_context_rewrite"] = history
            runtime_log(
                layer="plan_tracking_context_rewrite",
                event="execute",
                data=history[-1],
                logger=self.logger,
            )
        return normalized

    def _compact_dept_plan_completion_generation_context(self, state: LangGraphState, tasks: list[TaskModel]) -> list[TaskModel]:
        if not _looks_like_dept_plan_completion_query(state.context.runtime.user_input):
            return tasks

        compare_tasks = {
            task.task_id: task
            for task in tasks
            if task.tool == "compare_dept_plan_completion"
        }
        if not compare_tasks:
            return tasks

        normalized: list[TaskModel] = []
        changed_task_ids: list[str] = []
        for task in tasks:
            if task.tool != "text_generate_tool":
                normalized.append(task)
                continue

            compare_task = next((compare_tasks[task_id] for task_id in task.depends_on if task_id in compare_tasks), None)
            if compare_task is None:
                normalized.append(task)
                continue

            desired_context = f"{{{{{compare_task.output_key}.dept_plan_completion_context_text}}}}"
            task_input = dict(task.input or {})
            changed = False
            if task_input.get("context") != desired_context:
                task_input["context"] = desired_context
                prompt = str(task_input.get("prompt") or "").strip()
                reminder = (
                    "上下文来自 dept_plan_completion_context_text 压缩字段；请基于三七计划、周报证据、负责人本人月度考核记录和 OPL 问题闭环证据逐条判断已完成、部分完成、未完成或证据不足。OPL 未闭环问题只能作为风险/卡点证据，不要直接等同计划未完成。"
                )
                if reminder not in prompt:
                    task_input["prompt"] = f"{prompt}\n{reminder}".strip() if prompt else reminder
                changed = True

            changed = _ensure_min_input_int(task_input, "timeout_seconds", 360) or changed
            changed = _ensure_min_input_int(task_input, "tool_timeout_seconds", 360) or changed
            changed = _ensure_min_input_int(task_input, "executor_timeout_seconds", 360) or changed
            if "temperature" not in task_input:
                task_input["temperature"] = 0.2
                changed = True

            normalized.append(task.model_copy(update={"input": task_input}) if changed else task)
            if changed:
                changed_task_ids.append(task.task_id)

        if changed_task_ids:
            history = list(state.metadata.get("dept_plan_completion_context_rewrite", []))
            history.append(
                {
                    "task_ids": changed_task_ids,
                    "context": "dept_plan_completion_context_text",
                    "reason": "avoid_full_dept_plan_completion_context_and_generation_timeout",
                    "timeout_seconds": 360,
                    "tool_timeout_seconds": 360,
                    "executor_timeout_seconds": 360,
                }
            )
            state.metadata["dept_plan_completion_context_rewrite"] = history
            runtime_log(
                layer="dept_plan_completion_context_rewrite",
                event="execute",
                data=history[-1],
                logger=self.logger,
            )
        return normalized

    def _resolve_state_planned_tasks(self, state: LangGraphState) -> list[TaskModel]:
        resolved_tasks = self._resolve_runtime_task_types(state.planned_tasks)
        resolved_tasks = self._preserve_rag_user_queries(state, resolved_tasks)
        resolved_tasks = self._prefer_weekly_report_records_for_blocker_queries(state, resolved_tasks)
        resolved_tasks = self._compact_weekly_blocker_generation_context(state, resolved_tasks)
        resolved_tasks = self._compact_plan_tracking_generation_context(state, resolved_tasks)
        resolved_tasks = self._compact_dept_plan_completion_generation_context(state, resolved_tasks)
        state.set_planned_tasks(resolved_tasks)
        return resolved_tasks

    def _build_planner_fallback_plan(self, *, state: LangGraphState) -> tuple[str, TaskPlan] | None:
        ab_case_plan = self._build_ab_case_fallback_plan(state=state)
        if ab_case_plan is not None:
            return "ab_case_search_plan", ab_case_plan

        weekly_blocker_plan = self._build_weekly_blocker_fallback_plan(state=state)
        if weekly_blocker_plan is not None:
            return "weekly_blocker_mysql_plan", weekly_blocker_plan

        dept_plan = self._build_dept_plan_completion_fallback_plan(state=state)
        if dept_plan is not None:
            return "dept_plan_completion_mysql_plan", dept_plan

        return None

    def _build_ab_case_fallback_plan(self, *, state: LangGraphState) -> TaskPlan | None:
        user_input = state.context.runtime.user_input.strip()
        if not _looks_like_ab_case_query(user_input):
            return None

        created_at = state.context.runtime.timestamp.isoformat()
        return TaskPlan.model_validate(
            {
                "goal": user_input,
                "tasks": [
                    {
                        "task_id": "task_1",
                        "task_name": "search_ab_case_examples",
                        "description": "调用 A/B 案例专用检索接口，获取相似奖惩案例、相似度和完整案例字段。",
                        "task_type": "ab_case_search",
                        "tool": "ab_case_search_tool",
                        "input": {
                            "query": user_input,
                            "top_k": 8,
                        },
                        "output_key": "ab_case_results",
                        "depends_on": [],
                        "priority": 1,
                        "tags": ["ab_case", "case_search", "rag"],
                        "status": TaskStatus.PENDING,
                        "retry_count": 0,
                        "max_retry": 1,
                        "timeout": 60,
                        "created_at": created_at,
                    },
                    {
                        "task_id": "task_2",
                        "task_name": "generate_ab_case_answer",
                        "description": "根据相似 A/B 奖惩案例生成最终回答。",
                        "task_type": "text_generation",
                        "tool": "text_generate_tool",
                        "input": {
                            "prompt": (
                                "请基于 A/B 案例检索结果回答用户问题。A案例表示好事奖励，B案例表示坏事惩罚。"
                                "优先引用相似度高的案例，说明相似点、差异点、偏 A 奖励还是偏 B 惩罚，"
                                "以及可参考的奖惩/评分依据；如果相关性不足，"
                                "请明确说明未找到足够相似案例，不要强行套用。"
                            ),
                            "context": "{{ab_case_results.case_context_text}}",
                            "style": "structured",
                            "audience": "business_user",
                        },
                        "output_key": "final_result",
                        "depends_on": ["task_1"],
                        "priority": 2,
                        "tags": ["llm", "generation"],
                        "status": TaskStatus.PENDING,
                        "retry_count": 0,
                        "max_retry": 1,
                        "timeout": 120,
                        "created_at": created_at,
                    },
                ],
            }
        )

    def _build_weekly_blocker_fallback_plan(self, *, state: LangGraphState) -> TaskPlan | None:
        user_input = state.context.runtime.user_input.strip()
        if not _looks_like_weekly_blocker_query(user_input):
            return None

        target_start, target_end = _previous_week_range(state)
        trace_start = target_start - timedelta(days=7)
        trace_end = target_start - timedelta(days=1)
        created_at = state.context.runtime.timestamp.isoformat()
        return TaskPlan.model_validate(
            {
                "goal": user_input,
                "tasks": [
                    {
                        "task_id": "task_1",
                        "task_name": "query_last_week_weekly_reports",
                        "description": "查询目标周所有人的周报明细，并从 weekly_reports 主表带出员工自填卡点。",
                        "task_type": "query_weekly_reports",
                        "tool": "query_weekly_reports",
                        "input": {
                            "user_name": None,
                            "department": None,
                            "start_date": target_start.isoformat(),
                            "end_date": target_end.isoformat(),
                            "item_type": None,
                            "record_level": "reports",
                            "include_evidence_text": False,
                            "limit": 500,
                        },
                        "output_key": "weekly_reports",
                        "depends_on": [],
                        "priority": 1,
                        "tags": ["mysql", "business", "weekly_report"],
                        "status": TaskStatus.PENDING,
                        "retry_count": 0,
                        "max_retry": 1,
                        "timeout": 30,
                        "created_at": created_at,
                    },
                    {
                        "task_id": "task_2",
                        "task_name": "classify_target_week_blockers",
                        "description": "语义判断目标周员工自填卡点是否为当前有效卡点，并决定哪些人员需要追溯。",
                        "task_type": "classify_weekly_blockers",
                        "tool": "classify_weekly_blockers",
                        "input": {
                            "weekly_reports_output_key": "weekly_reports",
                            "limit": 500,
                        },
                        "output_key": "weekly_blocker_classification",
                        "depends_on": ["task_1"],
                        "priority": 1,
                        "tags": ["llm", "business", "weekly_report", "weekly_blocker"],
                        "status": TaskStatus.PENDING,
                        "retry_count": 0,
                        "max_retry": 1,
                        "timeout": 120,
                        "created_at": created_at,
                    },
                    {
                        "task_id": "task_3",
                        "task_name": "compare_weekly_plan_done_trace",
                        "description": "读取上游语义分类结果，仅对需要追溯的人员按两个独立周窗口追溯计划与完成记录，并收集历史卡点候选。",
                        "task_type": "compare_weekly_plan_done",
                        "tool": "compare_weekly_plan_done",
                        "input": {
                            "user_name": None,
                            "department": None,
                            "last_week_start": trace_start.isoformat(),
                            "last_week_end": trace_end.isoformat(),
                            "this_week_start": target_start.isoformat(),
                            "this_week_end": target_end.isoformat(),
                            "weekly_blocker_classification_output_key": "weekly_blocker_classification",
                            "trace_weeks": 2,
                            "include_historical_blockers": True,
                            "limit": 500,
                        },
                        "output_key": "weekly_plan_comparison",
                        "depends_on": ["task_1", "task_2"],
                        "priority": 1,
                        "tags": ["mysql", "business", "weekly_report"],
                        "status": TaskStatus.PENDING,
                        "retry_count": 0,
                        "max_retry": 1,
                        "timeout": 30,
                        "created_at": created_at,
                    },
                    {
                        "task_id": "task_4",
                        "task_name": "judge_weekly_blocker_trace",
                        "description": "判断历史卡点候选在后续周报中是否解决，并生成最终逐人卡点证据上下文。",
                        "task_type": "judge_weekly_blocker_trace",
                        "tool": "judge_weekly_blocker_trace",
                        "input": {
                            "weekly_blocker_classification_output_key": "weekly_blocker_classification",
                            "weekly_plan_comparison_output_key": "weekly_plan_comparison",
                        },
                        "output_key": "weekly_blocker_trace_judgement",
                        "depends_on": ["task_2", "task_3"],
                        "priority": 1,
                        "tags": ["llm", "business", "weekly_report", "weekly_blocker"],
                        "status": TaskStatus.PENDING,
                        "retry_count": 0,
                        "max_retry": 1,
                        "timeout": 120,
                        "created_at": created_at,
                    },
                    {
                        "task_id": "task_5",
                        "task_name": "generate_blockers_answer",
                        "description": "综合当前员工自填卡点、计划追溯证据和历史卡点闭环判断，生成逐人卡点汇总。",
                        "task_type": "text_generation",
                        "tool": "text_generate_tool",
                        "input": {
                            "prompt": "请回答用户关于上周所有人汇报卡点的问题。上下文已经是压缩后的逐人卡点语义判断和追溯证据：当前有效卡点以员工自填卡点为准，未确认当前卡点的人员包含两周计划追溯证据，历史卡点需要说明后续是否已有解决迹象。请区分员工自填、计划追溯推断、历史卡点追溯判断；最终回答使用业务表述，不要出现数据库字段名、内部输出键或英文技术字段名。",
                            "context": "{{weekly_blocker_trace_judgement.weekly_blocker_context_text}}",
                            "style": "structured",
                            "audience": "business_user",
                        },
                        "output_key": "final_result",
                        "depends_on": ["task_4"],
                        "priority": 2,
                        "tags": ["llm", "generation"],
                        "status": TaskStatus.PENDING,
                        "retry_count": 0,
                        "max_retry": 1,
                        "timeout": 120,
                        "created_at": created_at,
                    },
                ],
            }
        )

    def _build_dept_plan_completion_fallback_plan(self, *, state: LangGraphState) -> TaskPlan | None:
        user_input = state.context.runtime.user_input.strip()
        if not _looks_like_dept_plan_completion_query(user_input):
            return None

        month = _extract_month_from_text(user_input, state=state)
        month_start, month_end = _month_date_range(month)
        include_blocker_reports = any(
            token in user_input
            for token in ("卡点", "风险", "求助", "阻塞", "卡在", "卡着", "卡住", "没动", "问题一直")
        )
        created_at = state.context.runtime.timestamp.isoformat()
        tasks: list[dict[str, Any]] = [
            {
                "task_id": "task_1",
                "task_name": "compare_dept_plan_completion",
                "description": "查询三七计划书计划项，并直连负责人本人月度考核记录和周报证据。",
                "task_type": "compare_dept_plan_completion",
                "tool": "compare_dept_plan_completion",
                "input": {
                    "month": month,
                    "department": None,
                    "doc_id": None,
                    "include_weekly": True,
                    "include_self_eval": True,
                    "include_opl": True,
                    "followup_days": 7,
                    "limit": 2000,
                },
                "output_key": "dept_plan_completion",
                "depends_on": [],
                "priority": 1,
                "tags": ["mysql", "business", "dept_plan"],
                "status": TaskStatus.PENDING,
                "retry_count": 0,
                "max_retry": 1,
                "timeout": 60,
                "created_at": created_at,
            }
        ]
        final_depends_on = ["task_1"]
        prompt = (
            "请根据 compare_dept_plan_completion 的 dept_plan_completion_context_text 输出三七计划完成情况。"
            "逐条判断已完成、部分完成、未完成或证据不足，并说明依据；"
            "请综合负责人周报、负责人月度考核和 OPL 问题闭环证据，OPL 未闭环问题只能作为风险/卡点证据，不要直接等同计划未完成；"
            "请明确区分 Runtime 任务执行成功与业务计划项完成情况。"
        )
        if include_blocker_reports:
            tasks.append(
                {
                    "task_id": "task_2",
                    "task_name": "query_monthly_weekly_report_blockers",
                    "description": "查询计划月份周报主表中的员工自填卡点。",
                    "task_type": "query_weekly_reports",
                    "tool": "query_weekly_reports",
                    "input": {
                        "user_name": None,
                        "department": None,
                        "start_date": month_start.isoformat(),
                        "end_date": month_end.isoformat(),
                        "item_type": None,
                        "record_level": "reports",
                        "include_evidence_text": False,
                        "limit": 2000,
                    },
                    "output_key": "weekly_reports",
                    "depends_on": [],
                    "priority": 1,
                    "tags": ["mysql", "business", "weekly_report"],
                    "status": TaskStatus.PENDING,
                    "retry_count": 0,
                    "max_retry": 1,
                    "timeout": 60,
                    "created_at": created_at,
                }
            )
            final_depends_on.append("task_2")
            prompt = (
                f"{prompt}\n同时参考周报主表中的员工自填卡点回答卡点问题；"
                "如果所有人均未填写卡点，请明确说明没有员工自填卡点，"
                "再把三七计划中缺少有效完成证据的事项标为根据证据推断的卡住/需跟进项。"
                "最终回答使用业务表述，不要出现数据库字段名、内部输出键或英文技术字段名。"
            )

        tasks.append(
            {
                "task_id": "task_3" if include_blocker_reports else "task_2",
                "task_name": "generate_dept_plan_completion_answer",
                "description": "根据三七计划、周报和负责人本人月度考核记录，生成计划完成情况及卡住事项报告。",
                "task_type": "text_generation",
                "tool": "text_generate_tool",
                "input": {
                    "prompt": prompt,
                    "context": "{{dept_plan_completion.dept_plan_completion_context_text}}",
                    "style": "structured",
                    "audience": "business_user",
                    "timeout_seconds": 360,
                    "tool_timeout_seconds": 360,
                    "executor_timeout_seconds": 360,
                    "temperature": 0.2,
                },
                "output_key": "final_result",
                "depends_on": final_depends_on,
                "priority": 2,
                "tags": ["llm", "generation"],
                "status": TaskStatus.PENDING,
                "retry_count": 0,
                "max_retry": 1,
                "timeout": 120,
                "created_at": created_at,
            }
        )
        return TaskPlan.model_validate({"goal": user_input, "tasks": tasks})

    def _resolve_context_tasks(self, state: LangGraphState) -> list[TaskModel]:
        resolved_tasks = self._resolve_runtime_task_types(list(state.context.tasks.values()))
        resolved_tasks = self._preserve_rag_user_queries(state, resolved_tasks)
        resolved_tasks = self._prefer_weekly_report_records_for_blocker_queries(state, resolved_tasks)
        resolved_tasks = self._compact_weekly_blocker_generation_context(state, resolved_tasks)
        resolved_tasks = self._compact_plan_tracking_generation_context(state, resolved_tasks)
        resolved_tasks = self._compact_dept_plan_completion_generation_context(state, resolved_tasks)
        state.context.tasks = {task.task_id: task for task in resolved_tasks}
        return resolved_tasks

    async def _queue_impl(self, state: LangGraphState) -> None:
        """初始化 TaskQueue，并写入最新快照。"""
        resume_has_context_tasks = bool(state.metadata.get("resume_from_checkpoint") and state.context.tasks)
        if not state.planned_tasks and not resume_has_context_tasks:
            raise ValueError("Queue 节点未收到可调度任务。")
        if state.planned_tasks:
            self._resolve_state_planned_tasks(state)
        if resume_has_context_tasks:
            self._resolve_context_tasks(state)
        queue = TaskQueue(
            context=state.context,
            state=state.agent_state,
            max_concurrency=self.dependencies.queue_max_concurrency,
            checkpoint_saver=self._build_checkpoint_saver(state, "queue"),
        )
        if resume_has_context_tasks:
            snapshot = await queue.hydrate(state.context.tasks.values())
        else:
            snapshot = await queue.initialize(state.planned_tasks)
        state.set_queue_snapshot(snapshot)
        state.set_runtime_handle("queue", queue)
        runtime_log(
            layer="queue",
            event="success",
            data={
                "ready_tasks": list(snapshot.ready_task_ids),
                "blocked_tasks": list(snapshot.blocked_task_ids),
                "total_tasks": snapshot.total_tasks,
            },
            logger=self.logger,
        )
        runtime_progress(step="queue", status="队列就绪", detail=f"共 {snapshot.total_tasks} 个任务，待执行 {len(snapshot.ready_task_ids)} 个，等待依赖 {len(snapshot.blocked_task_ids)} 个")

    async def _executor_impl(self, state: LangGraphState) -> None:
        """执行队列中的全部任务，并更新状态。"""
        queue = state.get_runtime_handle("queue")
        if queue is None:
            resume_has_context_tasks = bool(state.metadata.get("resume_from_checkpoint") and state.context.tasks)
            if not state.planned_tasks and not resume_has_context_tasks:
                raise ValueError("Executor 节点缺少 Queue 和 planned_tasks。")
            if state.planned_tasks:
                self._resolve_state_planned_tasks(state)
            if resume_has_context_tasks:
                self._resolve_context_tasks(state)
            queue = TaskQueue(
                context=state.context,
                state=state.agent_state,
                max_concurrency=self.dependencies.queue_max_concurrency,
                checkpoint_saver=self._build_checkpoint_saver(state, "queue"),
            )
            if resume_has_context_tasks:
                snapshot = await queue.hydrate(state.context.tasks.values())
            else:
                snapshot = await queue.initialize(state.planned_tasks)
            state.set_queue_snapshot(snapshot)
            state.set_runtime_handle("queue", queue)
            runtime_log(
                layer="queue",
                event="success",
                data={
                    "ready_tasks": list(snapshot.ready_task_ids),
                    "blocked_tasks": list(snapshot.blocked_task_ids),
                    "total_tasks": snapshot.total_tasks,
                    "mode": "executor_bootstrap",
                },
                logger=self.logger,
            )

        self.dependencies.router.set_permission_context(self._build_permission_context(state))
        self.dependencies.router.set_routing_recorder(self._build_routing_recorder(state))
        self.dependencies.router.set_tool_load_provider(self._build_tool_load_provider(state))
        await self._attach_llm_call_recorder_to_router_tools(state)

        executor = TaskExecutor(
            context=state.context,
            queue=queue,
            router=self.dependencies.router,
            checkpoint_saver=self._build_checkpoint_saver(state, "executor"),
            result_callback=self._build_result_callback(state),
            failure_recorder=self._build_failure_recorder(state),
        )
        state.set_runtime_handle("executor", executor)
        results = await executor.execute_until_complete(
            batch_limit=state.metadata.get("executor_batch_limit"),
            stop_after_results=state.metadata.get("interrupt_after_task_count"),
        )

        state.set_queue_snapshot(await queue.get_snapshot())
        runtime_log(
            layer="executor",
            event="success",
            data={
                "task_ids": [result.task_id for result in results],
                "result_count": len(results),
                "failed_task_ids": list(state.agent_state.failed_task_ids),
            },
            logger=self.logger,
        )
        if await queue.has_terminal_failures():
            state.set_failed(
                "一个或多个任务在执行阶段失败。",
                details={"failed_task_ids": list(state.agent_state.failed_task_ids)},
            )

    async def _aggregator_impl(self, state: LangGraphState) -> None:
        """聚合最终输出与调试信息。"""
        clarification_response = state.metadata.get("clarification_response")
        has_tasks = bool(state.context.tasks)
        all_tasks_succeeded = has_tasks and not state.agent_state.pending_task_ids and not state.agent_state.failed_task_ids
        runtime_success = state.phase != GraphPhase.FAILED and (
            all_tasks_succeeded if has_tasks else not state.agent_state.failed_task_ids
        )
        payload = {
            "request_id": state.context.runtime.request_id,
            "session_id": state.context.runtime.session_id,
            "supervisor_route": state.supervisor_route,
            "success": runtime_success,
            "phase": state.phase.value,
            "final_output_ready": state.agent_state.final_output_ready,
            "task_summary": {
                "pending": list(state.agent_state.pending_task_ids),
                "completed": list(state.agent_state.completed_task_ids),
                "failed": list(state.agent_state.failed_task_ids),
            },
            "runtime_task_execution": self._build_runtime_task_execution_summary(state),
            "business_result": self._build_business_result_summary(state),
            "task_results": state.context.task_results,
            "execution_results": {
                task_id: result.model_dump(mode="json") for task_id, result in state.execution_results.items()
            },
            "errors": [error.model_dump(mode="json") for error in state.context.errors],
            "metadata": state.metadata,
        }
        if isinstance(clarification_response, dict):
            payload.update(clarification_response)
            payload["success"] = False
        state.set_final_response(payload)
        runtime_log(
            layer="aggregator",
            event="success",
            data={
                "final_keys": sorted(payload.keys()),
                "success": payload["success"],
                "failed_task_count": len(payload["task_summary"]["failed"]),
                "phase": payload["phase"],
                "phase_transition": f"{GraphPhase.AGGREGATING.value}->{payload['phase']}",
                "all_task_success": all_tasks_succeeded,
            },
            logger=self.logger,
        )

    def _build_runtime_task_execution_summary(self, state: LangGraphState) -> dict[str, Any]:
        tasks = list(state.context.tasks.values())
        total = len(tasks)
        completed = len([task for task in tasks if task.status == TaskStatus.SUCCESS])
        failed = len([task for task in tasks if task.status == TaskStatus.FAILED])
        cancelled = len([task for task in tasks if task.status == TaskStatus.CANCELLED])
        timed_out = len([task for task in tasks if task.status == TaskStatus.TIMEOUT])
        pending = max(0, total - completed - failed - cancelled - timed_out)
        return {
            "status": "SUCCESS" if total > 0 and completed == total else ("FAILED" if failed or timed_out or cancelled else "NO_TASKS"),
            "total_tasks": total,
            "completed_tasks": completed,
            "failed_tasks": failed,
            "cancelled_tasks": cancelled,
            "timeout_tasks": timed_out,
            "pending_tasks": pending,
            "message": (
                f"Runtime 执行完成：{completed}/{total} 个任务成功。"
                if total
                else "Runtime 未创建可执行任务。"
            ),
        }

    def _build_business_result_summary(self, state: LangGraphState) -> dict[str, Any]:
        has_dept_plan = any(task.tool == "compare_dept_plan_completion" for task in state.context.tasks.values())
        has_weekly_plan = any(task.tool == "compare_weekly_plan_done" for task in state.context.tasks.values())
        if has_dept_plan:
            source_output = next(
                (
                    output
                    for output in state.context.task_results.values()
                    if isinstance(output, dict) and isinstance(output.get("dept_plan_followups"), list)
                ),
                {},
            )
            pairing_summary = source_output.get("pairing_summary") if isinstance(source_output, dict) else {}
            return {
                "type": "dept_plan_completion_analysis",
                "status": "ANALYZED" if source_output else "UNKNOWN",
                "message": "业务结论表示三七计划项的完成/证据状态，不表示 Runtime 任务是否执行失败。",
                "source_summary": pairing_summary if isinstance(pairing_summary, dict) else {},
            }
        if has_weekly_plan:
            return {
                "type": "weekly_plan_completion_analysis",
                "status": "ANALYZED",
                "message": "业务结论表示周计划项的完成/证据状态，不表示 Runtime 任务是否执行失败。",
                "source_summary": {},
            }
        return {
            "type": "generic",
            "status": "NOT_APPLICABLE",
            "message": "无独立业务完成率判定。",
            "source_summary": {},
        }

    async def _build_tool_catalog(self) -> list[ToolDefinition]:
        """把已注册工具转换为 Planner Prompt 使用的目录结构。"""
        tool_names = await self.dependencies.router.list_tools(enabled_only=True)
        definitions: list[ToolDefinition] = []
        for tool_name in tool_names:
            tool = await self.dependencies.router.get_tool(tool_name)
            capability = self.dependencies.router.get_tool_capability(tool_name)
            definitions.append(
                ToolDefinition(
                    name=tool.name,
                    description=tool.description,
                    schema_version=tool.schema_version,
                    input_schema=tool.get_input_schema(),
                    output_schema=tool.get_output_schema(),
                    supported_task_types=capability.supported_task_types if capability is not None else [],
                    default_task_type=capability.default_task_type if capability is not None else None,
                    supported_tags=capability.supported_tags if capability is not None else [],
                )
            )
        return definitions

    async def _build_tool_capability_map(self) -> dict[str, Any]:
        tool_names = await self.dependencies.router.list_tools(enabled_only=True)
        capabilities: dict[str, Any] = {}
        for tool_name in tool_names:
            capability = self.dependencies.router.get_tool_capability(tool_name)
            if capability is not None:
                capabilities[tool_name] = capability
        return capabilities

    async def _resolve_simple_tool_name(self, user_input: str) -> str:
        """为简单任务选择默认工具。"""
        normalized = user_input.lower()
        prefers_reasoning = any(token in normalized for token in ["analyze", "why", "how", "compare", "reason"])
        preferred_name = (
            self.dependencies.default_reason_tool_name if prefers_reasoning else self.dependencies.default_simple_tool_name
        )

        enabled_tools = await self.dependencies.router.list_tools(enabled_only=True)
        if preferred_name in enabled_tools:
            return preferred_name
        if enabled_tools:
            return enabled_tools[0]
        raise RuntimeError("当前没有任何可用工具，无法执行简单任务。")

    def _build_context_summary(self, state: LangGraphState) -> str:
        """构造供 Supervisor/Planner 使用的简要上下文摘要。"""
        summary_parts = [
            f"Request ID: {state.context.runtime.request_id}",
            f"Session ID: {state.context.runtime.session_id}",
            build_runtime_time_context(state.context.runtime),
            f"Supervisor Route: {state.supervisor_route or 'UNKNOWN'}",
            f"Completed Tasks: {len(state.agent_state.completed_task_ids)}",
            f"Failed Tasks: {len(state.agent_state.failed_task_ids)}",
        ]
        return "\n".join(summary_parts)

    def _extract_text_output(self, payload: Any) -> str:
        """从 Planner 工具输出中提取文本内容。"""
        if isinstance(payload, dict):
            text = payload.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
        if isinstance(payload, str) and payload.strip():
            return payload.strip()
        raise ValueError("Planner 工具输出中缺少可解析的文本字段。")

    def _record_graph_tool_result(
        self,
        state: LangGraphState,
        tool_result,
        *,
        task_id: str | None,
        status: str,
    ) -> None:
        """记录图节点直接调用工具时的调用链与 Token 信息。"""
        state.context.add_tool_call(
            ToolCallRecord(
                tool_name=tool_result.tool_name,
                task_id=task_id,
                status=status,
                started_at=tool_result.started_at,
                finished_at=tool_result.finished_at,
                latency_ms=tool_result.latency_ms,
                metadata=tool_result.metadata,
            )
        )
        usage_payload = tool_result.metadata.get("usage")
        if usage_payload:
            usage = usage_payload if isinstance(usage_payload, TokenUsage) else TokenUsage.model_validate(usage_payload)
            state.context.add_token_usage(usage)

    async def _save_checkpoint(
        self,
        state: LangGraphState,
        *,
        source_layer: str,
        event: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """在启用检查点时保存当前图状态。"""
        manager = self.dependencies.checkpoint_manager
        if manager is None:
            return
        await manager.save_state(state, source_layer=source_layer, event=event, metadata=metadata)

    def _build_checkpoint_saver(self, state: LangGraphState, source_layer: str):
        """为 Queue / Executor 构造共享的检查点保存回调。"""
        manager = self.dependencies.checkpoint_manager
        if manager is None:
            return None

        async def _save(event: str, metadata: dict[str, Any] | None = None) -> None:
            await manager.save_state(state, source_layer=source_layer, event=event, metadata=metadata)

        return _save

    def _build_result_callback(self, state: LangGraphState):
        """把任务结果实时同步回 LangGraphState。"""
        async def _record(result) -> None:
            state.add_execution_result(result)
            if result.route_decision is not None:
                state.set_latest_route_decision(result.route_decision)

        return _record

    def _build_failure_recorder(self, state: LangGraphState):
        def _record_failure(record: dict[str, Any]) -> None:
            failures = list(state.metadata.get("task_failures", []))
            failures.append(record)
            state.metadata["task_failures"] = failures

        return _record_failure

    def _build_permission_context(self, state: LangGraphState) -> PermissionContext:
        runtime_metadata = state.context.runtime.metadata
        roles = runtime_metadata.get("roles", [])
        permissions = runtime_metadata.get("permissions", [])
        request_source = str(runtime_metadata.get("request_source", "runtime"))
        return PermissionContext(
            roles=[str(item) for item in roles] if isinstance(roles, list) else [],
            permissions=[str(item) for item in permissions] if isinstance(permissions, list) else [],
            request_source=request_source,
        )

    def _build_routing_recorder(self, state: LangGraphState):
        def _record_routing(record: dict[str, Any]) -> None:
            history = list(state.metadata.get("routing_history", []))
            history.append(record)
            state.metadata["routing_history"] = history

        return _record_routing

    def _build_tool_load_provider(self, state: LangGraphState):
        def _current_load(tool_name: str) -> int:
            matching = sum(
                1
                for task in state.context.tasks.values()
                if task.tool == tool_name and task.status in {TaskStatus.QUEUED, TaskStatus.RUNNING}
            )
            return max(0, matching - 1) if matching > 0 else 0

        return _current_load

    def _build_llm_call_recorder(self, state: LangGraphState):
        def _record_llm_call(record: dict[str, Any]) -> None:
            history = list(state.metadata.get("llm_calls", []))
            history.append(record)
            state.metadata["llm_calls"] = history

        return _record_llm_call

    def _attach_llm_call_recorder_to_client(self, *, client: Any, state: LangGraphState) -> None:
        if client is None or not hasattr(client, "set_llm_call_recorder"):
            return
        client.set_llm_call_recorder(self._build_llm_call_recorder(state))

    async def _attach_llm_call_recorder_to_router_tools(self, state: LangGraphState) -> None:
        tool_names = await self.dependencies.router.list_tools()
        for tool_name in tool_names:
            tool = await self.dependencies.router.get_tool(tool_name, require_enabled=False)
            self._attach_llm_call_recorder_to_client(client=getattr(tool, "client", None), state=state)
