"""LangGraph 运行时主状态对象。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.prompts.task_planner import PlannerPromptBundle
from app.schemas.context import AgentState, ContextStore, RuntimeContext
from app.schemas.executor import TaskExecutionResult
from app.schemas.graph import GraphPhase, GraphStateSnapshot
from app.schemas.planner import TaskPlan
from app.schemas.queue import QueueSnapshot
from app.schemas.router import ToolRouteDecision
from app.schemas.task import TaskModel


class LangGraphState(BaseModel):
    """在 LangGraph 节点之间传递的核心状态对象。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", validate_assignment=True)

    agent_state: AgentState = Field(..., description="底层共享运行时状态。")
    phase: GraphPhase = Field(default=GraphPhase.INITIALIZED, description="当前运行阶段。")
    current_node: str | None = Field(default=None, description="当前正在执行的图节点。")
    last_completed_node: str | None = Field(default=None, description="最近完成的图节点。")
    supervisor_route: str | None = Field(default=None, description="Supervisor 的路由结果。")
    planner_prompt: PlannerPromptBundle | None = Field(default=None, description="Planner 使用的 Prompt 包。")
    raw_plan_text: str | None = Field(default=None, description="Planner 原始输出文本。")
    parsed_plan: TaskPlan | None = Field(default=None, description="解析并校验后的任务计划。")
    planned_tasks: list[TaskModel] = Field(default_factory=list, description="准备进入调度的任务列表。")
    queue_snapshot: QueueSnapshot | None = Field(default=None, description="最近一次队列快照。")
    latest_route_decision: ToolRouteDecision | None = Field(default=None, description="最近一次路由决策。")
    execution_results: dict[str, TaskExecutionResult] = Field(
        default_factory=dict,
        description="按 task_id 存放的执行结果。",
    )
    final_response: Any | None = Field(default=None, description="最终聚合响应。")
    metadata: dict[str, Any] = Field(default_factory=dict, description="图级别元数据。")
    runtime_handles: dict[str, Any] = Field(
        default_factory=dict,
        description="仅在运行期间使用的不可序列化对象。",
        repr=False,
    )

    @classmethod
    def create(
        cls,
        *,
        request_id: str,
        session_id: str,
        user_input: str,
        runtime_metadata: dict[str, Any] | None = None,
        graph_metadata: dict[str, Any] | None = None,
    ) -> "LangGraphState":
        """根据入口请求信息构造初始图状态。"""
        runtime = RuntimeContext(
            request_id=request_id,
            session_id=session_id,
            user_input=user_input,
            metadata=runtime_metadata or {},
        )
        context = ContextStore(runtime=runtime)
        agent_state = AgentState(context=context)
        return cls(agent_state=agent_state, metadata=graph_metadata or {})

    @property
    def context(self) -> ContextStore:
        """便捷访问共享上下文。"""
        return self.agent_state.context

    def set_current_node(self, node_name: str, *, phase: GraphPhase | None = None) -> None:
        """更新当前正在执行的节点。"""
        self.current_node = node_name
        if phase is not None:
            self.phase = phase

    def mark_node_completed(self, node_name: str, *, phase: GraphPhase | None = None) -> None:
        """标记节点执行完成。"""
        self.last_completed_node = node_name
        self.current_node = None
        if phase is not None:
            self.phase = phase

    def set_supervisor_route(self, route: str) -> None:
        """保存 Supervisor 路由结果。"""
        self.supervisor_route = route
        self.context.set_shared_value("supervisor_route", route)

    def set_planner_prompt(self, prompt: PlannerPromptBundle) -> None:
        """保存 Planner Prompt，并把阶段推进到已规划。"""
        self.planner_prompt = prompt
        self.phase = GraphPhase.PLANNED
        self.context.set_shared_value("planner_prompt", prompt.model_dump(mode="json"))

    def set_raw_plan_text(self, raw_text: str) -> None:
        """保存 Planner 的原始输出文本。"""
        self.raw_plan_text = raw_text
        self.context.set_shared_value("raw_plan_text", raw_text)

    def set_parsed_plan(self, plan: TaskPlan) -> None:
        """保存解析后的任务计划。"""
        self.parsed_plan = plan
        self.phase = GraphPhase.PARSED
        self.context.set_shared_value("parsed_plan", plan.model_dump(mode="json"))

    def set_planned_tasks(self, tasks: list[TaskModel]) -> None:
        """保存待调度任务列表。"""
        self.planned_tasks = tasks
        self.context.set_shared_value(
            "planned_tasks",
            [task.model_dump(mode="json") for task in tasks],
        )

    def set_queue_snapshot(self, snapshot: QueueSnapshot) -> None:
        """保存最新队列快照。"""
        self.queue_snapshot = snapshot
        self.phase = GraphPhase.QUEUED
        self.context.set_shared_value("queue_snapshot", snapshot.model_dump(mode="json"))

    def set_latest_route_decision(self, decision: ToolRouteDecision) -> None:
        """保存最近一次工具路由决策。"""
        self.latest_route_decision = decision
        self.phase = GraphPhase.ROUTED
        self.context.set_shared_value("latest_route_decision", decision.model_dump(mode="json"))

    def add_execution_result(self, result: TaskExecutionResult) -> None:
        """追加单个任务执行结果。"""
        self.execution_results[result.task_id] = result
        self.phase = GraphPhase.EXECUTING
        self.context.set_shared_value(
            "execution_results",
            {task_id: item.model_dump(mode="json") for task_id, item in self.execution_results.items()},
        )

    def set_final_response(self, payload: Any) -> None:
        """保存最终聚合结果。"""
        self.final_response = payload
        if self.phase != GraphPhase.FAILED:
            self.phase = GraphPhase.COMPLETED
            self.agent_state.final_output_ready = True
        if isinstance(payload, dict):
            payload["phase"] = self.phase.value
            payload["final_output_ready"] = self.agent_state.final_output_ready
        self.context.final_output = payload
        self.context.set_shared_value("final_response", payload)

    def set_failed(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        """将图状态切换为失败，并记录失败原因。"""
        self.phase = GraphPhase.FAILED
        self.metadata["failure"] = {"message": message, "details": details or {}}

    def set_runtime_handle(self, key: str, value: Any) -> None:
        """挂载运行期间使用的对象句柄。"""
        self.runtime_handles[key] = value

    def get_runtime_handle(self, key: str) -> Any | None:
        """读取运行期间挂载的对象句柄。"""
        return self.runtime_handles.get(key)

    def to_snapshot(self) -> GraphStateSnapshot:
        """导出面向检查点与调试的状态快照。"""
        return GraphStateSnapshot(
            request_id=self.context.runtime.request_id,
            session_id=self.context.runtime.session_id,
            phase=self.phase,
            current_node=self.current_node,
            last_completed_node=self.last_completed_node,
            supervisor_route=self.supervisor_route,
            current_task_id=self.agent_state.current_task_id,
            pending_task_ids=list(self.agent_state.pending_task_ids),
            completed_task_ids=list(self.agent_state.completed_task_ids),
            failed_task_ids=list(self.agent_state.failed_task_ids),
            final_output_ready=self.agent_state.final_output_ready,
            final_output=self.context.final_output,
        )

    def serialize_for_checkpoint(self) -> dict[str, Any]:
        """导出适合 LangGraph 检查点持久化的 JSON 负载。"""
        snapshot = self.to_snapshot()
        return {
            "snapshot": snapshot.model_dump(mode="json"),
            "context": self.context.model_dump(mode="json"),
            "planner_prompt": self.planner_prompt.model_dump(mode="json") if self.planner_prompt else None,
            "raw_plan_text": self.raw_plan_text,
            "parsed_plan": self.parsed_plan.model_dump(mode="json") if self.parsed_plan else None,
            "planned_tasks": [task.model_dump(mode="json") for task in self.planned_tasks],
            "queue_snapshot": self.queue_snapshot.model_dump(mode="json") if self.queue_snapshot else None,
            "latest_route_decision": self.latest_route_decision.model_dump(mode="json") if self.latest_route_decision else None,
            "execution_results": {
                task_id: result.model_dump(mode="json") for task_id, result in self.execution_results.items()
            },
            "final_response": self.final_response,
            "metadata": self.metadata,
        }

    @classmethod
    def from_checkpoint_payload(cls, payload: dict[str, Any]) -> "LangGraphState":
        """根据检查点快照恢复 LangGraphState。"""
        snapshot = GraphStateSnapshot.model_validate(payload["snapshot"])
        context = ContextStore.model_validate(payload["context"])
        agent_state = AgentState(
            context=context,
            current_task_id=snapshot.current_task_id,
            pending_task_ids=list(snapshot.pending_task_ids),
            completed_task_ids=list(snapshot.completed_task_ids),
            failed_task_ids=list(snapshot.failed_task_ids),
            final_output_ready=snapshot.final_output_ready,
        )
        return cls(
            agent_state=agent_state,
            phase=snapshot.phase,
            current_node=snapshot.current_node,
            last_completed_node=snapshot.last_completed_node,
            supervisor_route=snapshot.supervisor_route,
            planner_prompt=(
                PlannerPromptBundle.model_validate(payload["planner_prompt"])
                if payload.get("planner_prompt") is not None
                else None
            ),
            raw_plan_text=payload.get("raw_plan_text"),
            parsed_plan=TaskPlan.model_validate(payload["parsed_plan"]) if payload.get("parsed_plan") is not None else None,
            planned_tasks=[TaskModel.model_validate(task) for task in payload.get("planned_tasks", [])],
            queue_snapshot=(
                QueueSnapshot.model_validate(payload["queue_snapshot"])
                if payload.get("queue_snapshot") is not None
                else None
            ),
            latest_route_decision=(
                ToolRouteDecision.model_validate(payload["latest_route_decision"])
                if payload.get("latest_route_decision") is not None
                else None
            ),
            execution_results={
                task_id: TaskExecutionResult.model_validate(item)
                for task_id, item in payload.get("execution_results", {}).items()
            },
            final_response=payload.get("final_response"),
            metadata=dict(payload.get("metadata", {})),
        )
