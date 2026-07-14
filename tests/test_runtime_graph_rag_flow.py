from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

import pytest
from pydantic import Field

from app.agents import SupervisorAgent
from app.graph.runtime_graph import (
    GraphRuntimeDependencies,
    RuntimeTaskTypeResolutionError,
    _RuntimeGraphBuilder,
    resolve_task_type,
)
from app.llm.exceptions import LLMInvalidResponseError
from app.planner import LLMTaskPlanner
from app.router import TaskRouter
from app.router.capability import ToolCapability, capability_from_tool
from app.schemas.context import ContextStore
from app.schemas.llm import LLMFunctionCall, LLMRequest, LLMResponse
from app.schemas.planner import TaskPlan
from app.schemas.task import TaskModel, TaskStatus
from app.state import LangGraphState
from app.tools.base import BaseTool
from app.tools.llm_client import LLMClient
from app.tools.llm_reason import LLMReasonTool
from app.tools.rag_batch_summarize import RAGBatchSummarizeTool
from app.tools.rag_search import RAGSearchTool
from app.tools.text_generate import TextGenerateTool


class ResolverOnlyTool(BaseTool):
    async def _arun(self, payload: dict[str, Any], context: ContextStore | None = None):
        return self.build_result(success=True, output={"ok": True})


class ExecutableRAGSearchTool(BaseTool):
    name: str = Field(default="rag_search_tool")
    description: str = Field(default="Deterministic RAG search test tool.")
    tags: list[str] = Field(default_factory=lambda: ["rag", "search", "knowledge_base"])

    def get_routing_capability(self) -> dict[str, Any]:
        capability = super().get_routing_capability()
        capability["supported_task_types"] = ["rag_search"]
        capability["default_task_type"] = "rag_search"
        capability["supported_tags"] = list(self.tags)
        return capability

    async def _arun(self, payload: dict[str, Any], context: ContextStore | None = None):
        query = str(payload.get("query") or payload.get("prompt") or "")
        return self.build_result(
            success=True,
            output={
                "query": query,
                "chunks": [{"chunk_id": "chunk_1", "text": "报销 SOP 需要提交 OA 单据。"}],
                "joined_context": "报销 SOP 需要提交 OA 单据。",
                "context_batches": [
                    {
                        "batch_id": "batch_1",
                        "joined_context": "报销 SOP 需要提交 OA 单据。",
                        "source_chunk_ids": ["chunk_1"],
                        "chunk_ids": ["chunk_1"],
                        "chars": 18,
                    }
                ],
                "summary": "检索到报销 SOP。",
                "low_relevance": False,
                "top_score": 0.9,
                "threshold": 0.5,
            },
        )


class ExecutableFinancialRAGSearchTool(BaseTool):
    name: str = Field(default="rag_search_tool")
    description: str = Field(default="Deterministic financial RAG search test tool.")
    tags: list[str] = Field(default_factory=lambda: ["rag", "search", "knowledge_base"])

    def get_routing_capability(self) -> dict[str, Any]:
        capability = super().get_routing_capability()
        capability["supported_task_types"] = ["rag_search"]
        capability["default_task_type"] = "rag_search"
        capability["supported_tags"] = list(self.tags)
        return capability

    async def _arun(self, payload: dict[str, Any], context: ContextStore | None = None):
        chunks = [
            {
                "chunk_id": f"fin-{index}",
                "context_text": "财务报表长上下文 " + "A" * 1500,
                "relative_path": "01财务报表/2022年财务报表.pdf",
                "score": 0.9 - index * 0.01,
            }
            for index in range(1, 7)
        ]
        return self.build_result(
            success=True,
            output={
                "query": str(payload.get("query") or ""),
                "chunks": chunks,
                "joined_context": "财务报表 利润表 营业收入 10,739,842.03",
                "context_batches": [
                    {
                        "batch_id": "batch_1",
                        "joined_context": (
                            "[1] source=01财务报表/2022年财务报表.pdf chunk_id=fin-1\n"
                            "财务报表 利润表 编制单位：九工机器（上海）有限公司 2022年12期 单位：元\n"
                            "营业收入 10,739,842.03\n"
                            "营业成本 5,146,519.97"
                        ),
                        "source_chunk_ids": ["src-fin-1", "src-fin-2"],
                        "chunk_ids": ["fin-1", "fin-2"],
                        "chars": 160,
                    },
                    {
                        "batch_id": "batch_2",
                        "joined_context": (
                            "[2] source=01财务报表/2022年财务报表.pdf chunk_id=fin-3\n"
                            "净利润 -5,418,627.21\n"
                            "现金净增加额 -25,037.71\n"
                            "期末现金余额 5,378.10"
                        ),
                        "source_chunk_ids": ["src-fin-3", "src-fin-4"],
                        "chunk_ids": ["fin-3", "fin-4"],
                        "chars": 120,
                    },
                ],
                "summary": "检索到财务报表。",
                "low_relevance": False,
                "top_score": 0.91,
                "threshold": 0.5,
            },
        )


class ExecutableRAGBatchSummarizeTool(BaseTool):
    name: str = Field(default="rag_batch_summarize_tool")
    description: str = Field(default="Deterministic RAG batch summary test tool.")
    tags: list[str] = Field(default_factory=lambda: ["rag", "summary", "llm"])

    def get_routing_capability(self) -> dict[str, Any]:
        capability = super().get_routing_capability()
        capability["supported_task_types"] = ["rag_batch_summary"]
        capability["default_task_type"] = "rag_batch_summary"
        capability["supported_tags"] = list(self.tags)
        return capability

    async def _arun(self, payload: dict[str, Any], context: ContextStore | None = None):
        if context is None or "rag_context" not in context.task_results:
            return self.build_result(success=False, error="missing rag_context")
        rag_context = context.task_results["rag_context"]
        return self.build_result(
            success=True,
            output={
                "text": "报销需要提交 OA 单据，并保留审批记录。",
                "summary": "报销需要提交 OA 单据。",
                "batch_summaries": [
                    {
                        "batch_id": "batch_1",
                        "text": "报销需要提交 OA 单据。",
                        "summary": "报销需要提交 OA 单据。",
                        "evidence_chunk_ids": ["chunk_1"],
                    }
                ],
                "source_chunk_ids": ["chunk_1"],
                "batch_count": len(rag_context.get("context_batches", [])) if isinstance(rag_context, dict) else 1,
            },
        )


class ExecutableTextGenerateTool(BaseTool):
    name: str = Field(default="text_generate_tool")
    description: str = Field(default="Deterministic text generation test tool.")
    tags: list[str] = Field(default_factory=lambda: ["llm", "generation", "text"])

    def get_routing_capability(self) -> dict[str, Any]:
        capability = super().get_routing_capability()
        capability["supported_task_types"] = ["text_generation"]
        capability["default_task_type"] = "text_generation"
        capability["supported_tags"] = list(self.tags)
        return capability

    async def _arun(self, payload: dict[str, Any], context: ContextStore | None = None):
        if context is None or "rag_summary" not in context.task_results:
            return self.build_result(success=False, error="missing rag_summary")
        return self.build_result(
            success=True,
            output={
                "text": "最终回答：报销需要提交 OA 单据，并保留审批记录。",
                "audience": payload.get("audience"),
                "style": payload.get("style"),
            },
        )


class GraphStructuredLLMClient(LLMClient):
    def __init__(
        self,
        *,
        rag_summary_task_type: str = "rag_batch_summary",
        text_task_tags: list[str] | None = None,
    ) -> None:
        super().__init__(timeout_seconds=30, model_name="graph-structured-client", model_version="test-v1")
        self.calls: list[dict[str, str]] = []
        self.rag_summary_task_type = rag_summary_task_type
        self.text_task_tags = text_task_tags or ["llm", "generation"]
        self.last_planner_payload: dict[str, object] | None = None

    async def _generate(self, request: LLMRequest) -> LLMResponse:
        function_name = request.tool_choice or (request.function_schemas[0].name if request.function_schemas else None)
        prompt = request.prompt or ""
        system_prompt = request.system_prompt or ""
        self.calls.append(
            {
                "function_name": function_name or "unknown",
                "prompt": prompt,
                "system_prompt": system_prompt,
            }
        )

        if request.metadata.get("component") == "parser_repair":
            return LLMResponse(
                text=self._repair_payload(),
                model_name=self.model_name,
                model_version=self.model_version,
                request_id=request.request_id,
                session_id=request.session_id,
                trace_id=request.trace_id,
                prompt_name=request.prompt_name,
                prompt_version=request.prompt_version,
                raw_response={"provider": "graph_structured_client"},
            )

        if function_name == "route_user_request":
            payload = self._supervisor_payload(prompt)
        elif function_name == "emit_task_plan":
            payload = self._planner_payload(prompt)
        else:
            raise AssertionError(f"unexpected function name: {function_name}")

        function_schema = request.function_schemas[0] if request.function_schemas else None
        return LLMResponse(
            text="structured graph response",
            model_name=self.model_name,
            model_version=self.model_version,
            request_id=request.request_id,
            session_id=request.session_id,
            trace_id=request.trace_id,
            prompt_name=request.prompt_name,
            prompt_version=request.prompt_version,
            function_call=LLMFunctionCall(
                tool_name=function_name or "unknown",
                arguments=payload,
                schema_name=function_schema.schema_name if function_schema is not None else None,
                schema_version=function_schema.schema_version if function_schema is not None else None,
            ),
            raw_response={"provider": "graph_structured_client"},
        )

    def _supervisor_payload(self, prompt: str) -> dict[str, object]:
        user_input = _extract_between(prompt, "User Input:\n", "\n\nRuntime Context Summary:")
        if _is_knowledge_request(user_input):
            return {
                "route": "COMPLEX_TASK",
                "complexity": "complex",
                "needs_planning": True,
                "reason": "knowledge retrieval required before answering",
            }
        return {
            "route": "SIMPLE_TASK",
            "complexity": "simple",
            "needs_planning": False,
            "reason": "can be completed directly",
        }

    def _planner_payload(self, prompt: str) -> dict[str, object]:
        user_goal = _extract_between(prompt, "User Goal:\n", "\n\nRuntime Context Summary:")
        created_at = _extract_created_at(prompt)
        if _is_knowledge_request(user_goal):
            payload = {
                "goal": user_goal,
                "tasks": [
                    {
                        "task_id": "task_1",
                        "task_name": "search_knowledge_base",
                        "description": "Search the company knowledge base for relevant material.",
                        "task_type": "rag_search",
                        "tool": "rag_search_tool",
                        "input": {"query": user_goal, "top_k": 10},
                        "output_key": "rag_context",
                        "depends_on": [],
                        "priority": 1,
                        "tags": ["rag", "search"],
                        "status": "PENDING",
                        "retry_count": 0,
                        "max_retry": 1,
                        "timeout": 30,
                        "created_at": created_at,
                    },
                    {
                        "task_id": "task_2",
                        "task_name": "summarize_rag_batches",
                        "description": "Summarize the retrieved RAG batches.",
                        "task_type": self.rag_summary_task_type,
                        "tool": "rag_batch_summarize_tool",
                        "input": {"query": user_goal, "rag_output_key": "rag_context"},
                        "output_key": "rag_summary",
                        "depends_on": ["task_1"],
                        "priority": 2,
                        "tags": ["rag", "summary"],
                        "status": "PENDING",
                        "retry_count": 0,
                        "max_retry": 1,
                        "timeout": 300,
                        "created_at": created_at,
                    },
                    {
                        "task_id": "task_3",
                        "task_name": "generate_answer_from_rag_summary",
                        "description": "Generate the final answer from summarized RAG evidence.",
                        "task_type": "text_generation",
                        "tool": "text_generate_tool",
                        "input": {
                            "prompt": "Answer from summarized evidence.",
                            "context": "{{rag_summary.text}}",
                            "rag_grounded": True,
                            "style": "clear",
                            "audience": "business_user",
                        },
                        "output_key": "final_result",
                        "depends_on": ["task_2"],
                        "priority": 3,
                        "tags": list(self.text_task_tags),
                        "status": "PENDING",
                        "retry_count": 0,
                        "max_retry": 1,
                        "timeout": 75,
                        "created_at": created_at,
                    },
                ],
            }
            self.last_planner_payload = deepcopy(payload)
            return payload
        payload = {
            "goal": user_goal,
            "tasks": [
                {
                    "task_id": "task_1",
                    "task_name": "complete_simple_request",
                    "description": "Complete the simple user request directly.",
                    "task_type": "text_generation",
                    "tool": "text_generate_tool",
                    "input": {"prompt": user_goal},
                    "output_key": "final_result",
                    "depends_on": [],
                    "priority": 1,
                    "tags": ["llm", "generation"],
                    "status": "PENDING",
                    "retry_count": 0,
                    "max_retry": 1,
                    "timeout": 60,
                    "created_at": created_at,
                }
            ],
        }
        self.last_planner_payload = deepcopy(payload)
        return payload

    def _repair_payload(self) -> str:
        if self.last_planner_payload is None:
            raise AssertionError("repair requested before planner payload was captured")
        repaired = deepcopy(self.last_planner_payload)
        for task in repaired.get("tasks", []):
            if isinstance(task, dict) and task.get("tool") == "text_generate_tool":
                task["tags"] = ["llm", "generation"]
        return json.dumps(repaired, ensure_ascii=False)


class BadInternalKnowledgePlannerClient(GraphStructuredLLMClient):
    def _planner_payload(self, prompt: str) -> dict[str, object]:
        user_goal = _extract_between(prompt, "User Goal:\n", "\n\nRuntime Context Summary:")
        created_at = _extract_created_at(prompt)
        if "来料质量检验" not in user_goal:
            return super()._planner_payload(prompt)
        payload = {
            "goal": user_goal,
            "tasks": [
                {
                    "task_id": "task_1",
                    "task_name": "quality_inspection_reasoning",
                    "description": "Incorrectly answer internal knowledge directly.",
                    "task_type": "reasoning",
                    "tool": "llm_reason_tool",
                    "input": {
                        "prompt": "请根据公司政策和流程，对来料质量检验进行详细说明。",
                        "model_name": "qwen",
                    },
                    "output_key": "inspection_reason",
                    "depends_on": [],
                    "priority": 1,
                    "tags": ["llm", "reasoning", "analysis"],
                    "status": "PENDING",
                    "retry_count": 0,
                    "max_retry": 1,
                    "timeout": 60,
                    "created_at": created_at,
                }
            ],
        }
        self.last_planner_payload = deepcopy(payload)
        return payload

    def _repair_payload(self) -> str:
        if self.last_planner_payload is None:
            raise AssertionError("repair requested before planner payload was captured")
        goal = str(self.last_planner_payload.get("goal") or "来料质量检验")
        created_at = str(self.last_planner_payload["tasks"][0]["created_at"])
        repaired = {
            "goal": goal,
            "tasks": [
                {
                    "task_id": "task_1",
                    "task_name": "search_knowledge_base",
                    "description": "Search the company knowledge base for relevant material.",
                    "task_type": "rag_search",
                    "tool": "rag_search_tool",
                    "input": {"query": goal, "top_k": 10},
                    "output_key": "rag_context",
                    "depends_on": [],
                    "priority": 1,
                    "tags": ["rag", "search"],
                    "status": "PENDING",
                    "retry_count": 0,
                    "max_retry": 1,
                    "timeout": 30,
                    "created_at": created_at,
                },
                {
                    "task_id": "task_2",
                    "task_name": "summarize_rag_batches",
                    "description": "Summarize the retrieved RAG batches.",
                    "task_type": "rag_batch_summary",
                    "tool": "rag_batch_summarize_tool",
                    "input": {"query": goal, "rag_output_key": "rag_context"},
                    "output_key": "rag_summary",
                    "depends_on": ["task_1"],
                    "priority": 2,
                    "tags": ["rag", "summary"],
                    "status": "PENDING",
                    "retry_count": 0,
                    "max_retry": 1,
                    "timeout": 300,
                    "created_at": created_at,
                },
                {
                    "task_id": "task_3",
                    "task_name": "generate_answer_from_rag_summary",
                    "description": "Generate the final answer from summarized RAG evidence.",
                    "task_type": "text_generation",
                    "tool": "text_generate_tool",
                    "input": {
                        "prompt": "Answer from summarized evidence.",
                        "context": "{{rag_summary.text}}",
                        "rag_grounded": True,
                        "style": "clear",
                        "audience": "business_user",
                    },
                    "output_key": "final_result",
                    "depends_on": ["task_2"],
                    "priority": 3,
                    "tags": ["llm", "generation"],
                    "status": "PENDING",
                    "retry_count": 0,
                    "max_retry": 1,
                    "timeout": 75,
                    "created_at": created_at,
                },
            ],
        }
        return json.dumps(repaired, ensure_ascii=False)


class RawFallbackGraphClient(GraphStructuredLLMClient):
    def __init__(self) -> None:
        super().__init__()
        self.text_generation_prompts: list[str] = []

    async def _generate(self, request: LLMRequest) -> LLMResponse:
        function_name = request.tool_choice or (request.function_schemas[0].name if request.function_schemas else None)
        if request.metadata.get("tool_name") == "rag_batch_summarize_tool":
            return LLMResponse(
                text="这不是 JSON，但原始财务数据在 batch_content 中。",
                model_name=self.model_name,
                model_version=self.model_version,
                request_id=request.request_id,
                session_id=request.session_id,
                trace_id=request.trace_id,
                prompt_name=request.prompt_name,
                prompt_version=request.prompt_version,
                raw_response={"provider": "raw_fallback_graph_client"},
            )
        if function_name == "emit_text_generation_output":
            self.text_generation_prompts.append(request.prompt or "")
            return LLMResponse(
                text="structured text generation response",
                model_name=self.model_name,
                model_version=self.model_version,
                request_id=request.request_id,
                session_id=request.session_id,
                trace_id=request.trace_id,
                prompt_name=request.prompt_name,
                prompt_version=request.prompt_version,
                function_call=LLMFunctionCall(
                    tool_name="emit_text_generation_output",
                    arguments={"text": "最终回答：已基于原始财务报表上下文回答。", "audience": "business_user", "style": "clear"},
                ),
                raw_response={"provider": "raw_fallback_graph_client"},
            )
        return await super()._generate(request)


class BrokenWeeklyBlockerPlannerClient(GraphStructuredLLMClient):
    async def _generate(self, request: LLMRequest) -> LLMResponse:
        function_name = request.tool_choice or (request.function_schemas[0].name if request.function_schemas else None)
        if function_name == "emit_task_plan":
            raise LLMInvalidResponseError(
                "Expecting ',' delimiter: line 1 column 1665 (char 1664)",
                provider="deepseek",
                model="deepseek-v4-flash",
                operation="planner",
            )
        return await super()._generate(request)


def _extract_between(text: str, start_marker: str, end_marker: str) -> str:
    start_index = text.index(start_marker) + len(start_marker)
    end_index = text.index(end_marker, start_index)
    return text[start_index:end_index].strip()


def _extract_created_at(prompt: str) -> str:
    marker = "15. Set task.created_at to this exact ISO 8601 UTC timestamp: "
    start_index = prompt.index(marker) + len(marker)
    end_index = prompt.index("\n", start_index)
    return prompt[start_index:end_index].strip()


def _is_knowledge_request(text: str) -> bool:
    normalized = text.lower()
    return any(
        token in normalized
        for token in [
            "知识",
            "申报",
            "报销",
            "oa",
            "erp",
            "wiki",
            "sop",
            "质量检验",
            "测试验收",
            "来料检验",
            "来料质量检验",
            "检验规范",
            "流程",
            "标准",
        ]
    )


async def _build_executable_rag_graph_builder(
    *,
    rag_summary_task_type: str = "rag_batch_summary",
    text_task_tags: list[str] | None = None,
) -> tuple[GraphStructuredLLMClient, _RuntimeGraphBuilder]:
    client = GraphStructuredLLMClient(rag_summary_task_type=rag_summary_task_type, text_task_tags=text_task_tags)
    router = TaskRouter()

    rag_tool = ExecutableRAGSearchTool()
    rag_batch_tool = ExecutableRAGBatchSummarizeTool()
    text_tool = ExecutableTextGenerateTool()
    reason_tool = LLMReasonTool(client=client)
    await router.register_tools(
        [
            (rag_tool, capability_from_tool(rag_tool)),
            (rag_batch_tool, capability_from_tool(rag_batch_tool)),
            (text_tool, capability_from_tool(text_tool)),
            (reason_tool, capability_from_tool(reason_tool)),
        ]
    )

    builder = _RuntimeGraphBuilder(
        dependencies=GraphRuntimeDependencies(
            router=router,
            supervisor_agent=SupervisorAgent(client=client),
            planner_agent=LLMTaskPlanner(client=client),
            repair_llm_client=client,
        )
    )
    return client, builder


async def _build_bad_internal_knowledge_graph_builder() -> tuple[BadInternalKnowledgePlannerClient, _RuntimeGraphBuilder]:
    client = BadInternalKnowledgePlannerClient()
    router = TaskRouter()

    rag_tool = ExecutableRAGSearchTool()
    rag_batch_tool = ExecutableRAGBatchSummarizeTool()
    text_tool = ExecutableTextGenerateTool()
    reason_tool = LLMReasonTool(client=client)
    await router.register_tools(
        [
            (rag_tool, capability_from_tool(rag_tool)),
            (rag_batch_tool, capability_from_tool(rag_batch_tool)),
            (text_tool, capability_from_tool(text_tool)),
            (reason_tool, capability_from_tool(reason_tool)),
        ]
    )

    builder = _RuntimeGraphBuilder(
        dependencies=GraphRuntimeDependencies(
            router=router,
            supervisor_agent=SupervisorAgent(client=client),
            planner_agent=LLMTaskPlanner(client=client),
            repair_llm_client=client,
        )
    )
    return client, builder


async def _build_raw_fallback_rag_graph_builder() -> tuple[RawFallbackGraphClient, _RuntimeGraphBuilder]:
    client = RawFallbackGraphClient()
    router = TaskRouter()

    rag_tool = ExecutableFinancialRAGSearchTool()
    rag_batch_tool = RAGBatchSummarizeTool(client=client)
    text_tool = TextGenerateTool(client=client)
    reason_tool = LLMReasonTool(client=client)
    await router.register_tools(
        [
            (rag_tool, capability_from_tool(rag_tool)),
            (rag_batch_tool, capability_from_tool(rag_batch_tool)),
            (text_tool, capability_from_tool(text_tool)),
            (reason_tool, capability_from_tool(reason_tool)),
        ]
    )

    builder = _RuntimeGraphBuilder(
        dependencies=GraphRuntimeDependencies(
            router=router,
            supervisor_agent=SupervisorAgent(client=client),
            planner_agent=LLMTaskPlanner(client=client),
            repair_llm_client=client,
        )
    )
    return client, builder


async def _build_weekly_blocker_fallback_graph_builder() -> _RuntimeGraphBuilder:
    client = BrokenWeeklyBlockerPlannerClient()
    router = TaskRouter()
    query_tool = ResolverOnlyTool(name="query_weekly_reports", description="Query weekly report rows.")
    classify_tool = ResolverOnlyTool(name="classify_weekly_blockers", description="Classify weekly blocker rows.")
    compare_tool = ResolverOnlyTool(name="compare_weekly_plan_done", description="Compare weekly plans and done items.")
    judge_tool = ResolverOnlyTool(name="judge_weekly_blocker_trace", description="Judge weekly blocker trace evidence.")
    text_tool = ResolverOnlyTool(name="text_generate_tool", description="Generate text from report evidence.")
    await router.register_tools(
        [
            (
                query_tool,
                ToolCapability(
                    tool_name=query_tool.name,
                    supported_task_types=["query_weekly_reports"],
                    supported_tags=["mysql", "business", "weekly_report"],
                ),
            ),
            (
                classify_tool,
                ToolCapability(
                    tool_name=classify_tool.name,
                    supported_task_types=["classify_weekly_blockers"],
                    supported_tags=["llm", "business", "weekly_report", "weekly_blocker"],
                ),
            ),
            (
                compare_tool,
                ToolCapability(
                    tool_name=compare_tool.name,
                    supported_task_types=["compare_weekly_plan_done"],
                    supported_tags=["mysql", "business", "weekly_report"],
                ),
            ),
            (
                judge_tool,
                ToolCapability(
                    tool_name=judge_tool.name,
                    supported_task_types=["judge_weekly_blocker_trace"],
                    supported_tags=["llm", "business", "weekly_report", "weekly_blocker"],
                ),
            ),
            (
                text_tool,
                ToolCapability(
                    tool_name=text_tool.name,
                    supported_task_types=["text_generation"],
                    supported_tags=["llm", "generation", "text"],
                ),
            ),
        ]
    )
    return _RuntimeGraphBuilder(
        dependencies=GraphRuntimeDependencies(
            router=router,
            planner_agent=LLMTaskPlanner(client=client),
            repair_llm_client=client,
        )
    )


async def _build_dept_plan_context_graph_builder() -> _RuntimeGraphBuilder:
    client = BrokenWeeklyBlockerPlannerClient()
    router = TaskRouter()
    query_tool = ResolverOnlyTool(name="query_weekly_reports", description="Query weekly report rows.")
    compare_tool = ResolverOnlyTool(name="compare_dept_plan_completion", description="Compare department plan completion.")
    text_tool = ResolverOnlyTool(name="text_generate_tool", description="Generate text from department plan evidence.")
    await router.register_tools(
        [
            (
                query_tool,
                ToolCapability(
                    tool_name=query_tool.name,
                    supported_task_types=["query_weekly_reports"],
                    supported_tags=["mysql", "business", "weekly_report"],
                ),
            ),
            (
                compare_tool,
                ToolCapability(
                    tool_name=compare_tool.name,
                    supported_task_types=["compare_dept_plan_completion"],
                    supported_tags=["mysql", "business", "dept_plan"],
                ),
            ),
            (
                text_tool,
                ToolCapability(
                    tool_name=text_tool.name,
                    supported_task_types=["text_generation"],
                    supported_tags=["llm", "generation", "text"],
                ),
            ),
        ]
    )
    return _RuntimeGraphBuilder(
        dependencies=GraphRuntimeDependencies(
            router=router,
            planner_agent=LLMTaskPlanner(client=client),
            repair_llm_client=client,
        )
    )


async def _build_graph_builder(
    *,
    rag_summary_task_type: str = "rag_batch_summary",
    text_task_tags: list[str] | None = None,
) -> tuple[GraphStructuredLLMClient, _RuntimeGraphBuilder]:
    client = GraphStructuredLLMClient(rag_summary_task_type=rag_summary_task_type, text_task_tags=text_task_tags)
    router = TaskRouter()

    rag_tool = RAGSearchTool()
    rag_batch_tool = RAGBatchSummarizeTool(client=client)
    reason_tool = LLMReasonTool(client=client)
    text_tool = TextGenerateTool(client=client)
    await router.register_tools(
        [
            (rag_tool, capability_from_tool(rag_tool)),
            (rag_batch_tool, capability_from_tool(rag_batch_tool)),
            (reason_tool, capability_from_tool(reason_tool)),
            (text_tool, capability_from_tool(text_tool)),
        ]
    )

    builder = _RuntimeGraphBuilder(
        dependencies=GraphRuntimeDependencies(
            router=router,
            supervisor_agent=SupervisorAgent(client=client),
            planner_agent=LLMTaskPlanner(client=client),
            repair_llm_client=client,
        )
    )
    return client, builder


def _build_summary_task(*, task_type: str | None) -> TaskModel:
    return TaskModel(
        task_id="task_2",
        task_name="summarize_rag_batches",
        description="Summarize retrieved RAG batches.",
        task_type=task_type,
        tool="rag_batch_summarize_tool",
        input={"query": "技术宝典", "rag_output_key": "rag_context"},
        output_key="rag_summary",
        depends_on=["task_1"],
        priority=2,
        tags=["rag", "summary"],
        timeout=300,
    )


async def test_single_task_type_resolver_overrides_tool_name_task_type_and_preserves_fields() -> None:
    _, builder = await _build_graph_builder()
    raw_task = {
        "task_id": "task_2",
        "tool": "rag_batch_summarize_tool",
        "task_type": "rag_batch_summarize_tool",
        "dependencies": ["legacy_dep"],
        "depends_on": ["task_1"],
        "input": {"query": "技术宝典", "rag_output_key": "rag_context"},
        "metadata": {"trace": "keep"},
    }

    resolved = resolve_task_type(raw_task, builder.dependencies.router)

    assert resolved is not raw_task
    assert resolved["task_type"] == "rag_batch_summary"
    for field_name in ["task_id", "tool", "dependencies", "depends_on", "input", "metadata"]:
        assert resolved[field_name] == raw_task[field_name]
    assert set(resolved) == set(raw_task)
    assert raw_task["task_type"] == "rag_batch_summarize_tool"


async def test_single_task_type_resolver_fills_missing_task_type() -> None:
    _, builder = await _build_graph_builder()
    task = _build_summary_task(task_type=None)

    resolved = resolve_task_type(task, builder.dependencies.router)

    assert isinstance(resolved, TaskModel)
    assert resolved.task_type == "rag_batch_summary"
    assert task.task_type is None
    assert resolved.task_id == task.task_id
    assert resolved.tool == task.tool
    assert resolved.depends_on == task.depends_on
    assert resolved.input == task.input


async def test_single_task_type_resolver_overrides_arbitrary_wrong_task_type() -> None:
    _, builder = await _build_graph_builder()
    task = _build_summary_task(task_type="not_a_real_task_type")

    resolved = resolve_task_type(task, builder.dependencies.router)

    assert isinstance(resolved, TaskModel)
    assert resolved.task_type == "rag_batch_summary"
    assert task.task_type == "not_a_real_task_type"


async def test_multi_task_type_resolver_allows_valid_explicit_task_type() -> None:
    router = TaskRouter()
    tool = ResolverOnlyTool(name="multi_task_tool", description="multi")
    await router.register_tool(
        tool,
        ToolCapability(tool_name=tool.name, supported_task_types=["analysis", "generation"]),
    )
    task = {"task_id": "task_multi", "tool": tool.name, "task_type": "analysis"}

    resolved = resolve_task_type(task, router)

    assert resolved is task
    assert resolved["task_type"] == "analysis"


async def test_multi_task_type_resolver_rejects_missing_task_type() -> None:
    router = TaskRouter()
    tool = ResolverOnlyTool(name="multi_task_tool", description="multi")
    await router.register_tool(
        tool,
        ToolCapability(tool_name=tool.name, supported_task_types=["analysis", "generation"]),
    )
    task = {"task_id": "task_multi", "tool": tool.name}

    with pytest.raises(RuntimeTaskTypeResolutionError) as exc_info:
        resolve_task_type(task, router)

    message = str(exc_info.value)
    assert "task_id=task_multi" in message
    assert "tool=multi_task_tool" in message
    assert "task_type=None" in message
    assert "multi-task-type tool requires an explicit task_type" in message


async def test_multi_task_type_resolver_rejects_illegal_task_type() -> None:
    router = TaskRouter()
    tool = ResolverOnlyTool(name="multi_task_tool", description="multi")
    await router.register_tool(
        tool,
        ToolCapability(tool_name=tool.name, supported_task_types=["analysis", "generation"]),
    )
    task = {"task_id": "task_multi", "tool": tool.name, "task_type": "summarize"}

    with pytest.raises(RuntimeTaskTypeResolutionError) as exc_info:
        resolve_task_type(task, router)

    message = str(exc_info.value)
    assert "task_id=task_multi" in message
    assert "tool=multi_task_tool" in message
    assert "task_type=summarize" in message
    assert "task_type is not supported by selected tool" in message


async def test_resolver_rejects_capability_default_outside_supported_task_types() -> None:
    router = TaskRouter()
    tool = ResolverOnlyTool(name="bad_default_tool", description="bad default")
    capability = ToolCapability(tool_name=tool.name, supported_task_types=["analysis", "generation"])
    object.__setattr__(capability, "default_task_type", "summary")
    await router.register_tool(tool, capability)
    task = {"task_id": "task_bad_default", "tool": tool.name, "task_type": "analysis"}

    with pytest.raises(RuntimeTaskTypeResolutionError) as exc_info:
        resolve_task_type(task, router)

    message = str(exc_info.value)
    assert "task_id=task_bad_default" in message
    assert "tool=bad_default_tool" in message
    assert "tool capability default_task_type is not in supported_task_types" in message


async def test_resolver_rejects_used_tool_with_empty_supported_task_types() -> None:
    router = TaskRouter()
    tool = ResolverOnlyTool(name="legacy_empty_tool", description="legacy")
    await router.register_tool(tool, ToolCapability(tool_name=tool.name, supported_task_types=[]))
    task = {"task_id": "task_legacy", "tool": tool.name, "task_type": "anything"}

    with pytest.raises(RuntimeTaskTypeResolutionError) as exc_info:
        resolve_task_type(task, router)

    message = str(exc_info.value)
    assert "task_id=task_legacy" in message
    assert "tool=legacy_empty_tool" in message
    assert "tool capability has empty supported_task_types" in message


async def test_parser_rewrites_weekly_blocker_text_context_to_compact_field() -> None:
    builder = await _build_weekly_blocker_fallback_graph_builder()
    created_at = "2026-06-22T07:57:38+00:00"
    raw_plan = {
        "goal": "上周所有人汇报的卡点是什么，卡在了哪些问题上",
        "tasks": [
            {
                "task_id": "task_1",
                "task_name": "query_last_week_weekly_reports",
                "description": "查询目标周所有人的周报明细。",
                "task_type": "query_weekly_reports",
                "tool": "query_weekly_reports",
                "input": {
                    "user_name": None,
                    "department": None,
                    "start_date": "2026-06-15",
                    "end_date": "2026-06-21",
                    "item_type": None,
                    "limit": 500,
                },
                "output_key": "weekly_reports",
                "depends_on": [],
                "priority": 1,
                "tags": ["mysql", "business", "weekly_report"],
                "status": "PENDING",
                "retry_count": 0,
                "max_retry": 1,
                "timeout": 30,
                "created_at": created_at,
            },
            {
                "task_id": "task_2",
                "task_name": "compare_weekly_plan_done_trace",
                "description": "追溯未填写卡点人员。",
                "task_type": "compare_weekly_plan_done",
                "tool": "compare_weekly_plan_done",
                "input": {
                    "user_name": None,
                    "department": None,
                    "last_week_start": "2026-06-08",
                    "last_week_end": "2026-06-14",
                    "this_week_start": "2026-06-15",
                    "this_week_end": "2026-06-21",
                    "trace_only_empty_risk_from_output_key": "weekly_reports",
                    "limit": 500,
                },
                "output_key": "weekly_plan_comparison",
                "depends_on": ["task_1"],
                "priority": 1,
                "tags": ["mysql", "business", "weekly_report"],
                "status": "PENDING",
                "retry_count": 0,
                "max_retry": 1,
                "timeout": 30,
                "created_at": created_at,
            },
            {
                "task_id": "task_3",
                "task_name": "generate_blockers_answer",
                "description": "生成卡点汇总。",
                "task_type": "text_generation",
                "tool": "text_generate_tool",
                "input": {
                    "prompt": "请根据以下数据回答用户问题。",
                    "context": "{{weekly_reports}}\n\n{{weekly_plan_comparison}}",
                    "style": "structured",
                    "audience": "business_user",
                },
                "output_key": "final_result",
                "depends_on": ["task_1", "task_2"],
                "priority": 2,
                "tags": ["llm", "generation"],
                "status": "PENDING",
                "retry_count": 0,
                "max_retry": 1,
                "timeout": 75,
                "created_at": created_at,
            },
        ],
    }
    state = LangGraphState.create(
        request_id="req_weekly_blocker_context_rewrite",
        session_id="sess_weekly_blocker_context_rewrite",
        user_input="上周所有人汇报的卡点是什么，卡在了哪些问题上",
        runtime_metadata={"client_timezone": "Asia/Shanghai"},
    )
    state.set_raw_plan_text(json.dumps(raw_plan, ensure_ascii=False))

    await builder.parser_node(state)

    text_task = next(task for task in state.planned_tasks if task.tool == "text_generate_tool")
    assert text_task.input["context"] == "{{weekly_plan_comparison.weekly_blocker_context_text}}"
    assert "{{weekly_reports}}" not in text_task.input["context"]
    assert "weekly_blocker_context_rewrite" in state.metadata
    assert state.metadata["weekly_blocker_context_rewrite"][0]["task_ids"] == ["task_3"]


async def test_parser_prefers_weekly_blocker_trace_judgement_context() -> None:
    builder = await _build_weekly_blocker_fallback_graph_builder()
    created_at = "2026-06-22T07:57:38+00:00"
    raw_plan = {
        "goal": "上周所有人汇报的卡点是什么，卡在了哪些问题上",
        "tasks": [
            {
                "task_id": "task_1",
                "task_name": "query_last_week_weekly_reports",
                "description": "查询目标周所有人的周报明细。",
                "task_type": "query_weekly_reports",
                "tool": "query_weekly_reports",
                "input": {
                    "user_name": None,
                    "department": None,
                    "start_date": "2026-06-15",
                    "end_date": "2026-06-21",
                    "item_type": None,
                    "record_level": "reports",
                    "include_evidence_text": False,
                    "limit": 500,
                },
                "output_key": "weekly_reports",
                "depends_on": [],
                "priority": 1,
                "tags": ["mysql", "business", "weekly_report"],
                "status": "PENDING",
                "retry_count": 0,
                "max_retry": 1,
                "timeout": 30,
                "created_at": created_at,
            },
            {
                "task_id": "task_2",
                "task_name": "classify_target_week_blockers",
                "description": "分类目标周卡点。",
                "task_type": "classify_weekly_blockers",
                "tool": "classify_weekly_blockers",
                "input": {"weekly_reports_output_key": "weekly_reports"},
                "output_key": "weekly_blocker_classification",
                "depends_on": ["task_1"],
                "priority": 1,
                "tags": ["llm", "business", "weekly_report", "weekly_blocker"],
                "status": "PENDING",
                "retry_count": 0,
                "max_retry": 1,
                "timeout": 120,
                "created_at": created_at,
            },
            {
                "task_id": "task_3",
                "task_name": "compare_weekly_plan_done_trace",
                "description": "按分类结果追溯。",
                "task_type": "compare_weekly_plan_done",
                "tool": "compare_weekly_plan_done",
                "input": {
                    "last_week_start": "2026-06-08",
                    "last_week_end": "2026-06-14",
                    "this_week_start": "2026-06-15",
                    "this_week_end": "2026-06-21",
                    "weekly_blocker_classification_output_key": "weekly_blocker_classification",
                    "trace_weeks": 2,
                    "include_historical_blockers": True,
                    "limit": 500,
                },
                "output_key": "weekly_plan_comparison",
                "depends_on": ["task_1", "task_2"],
                "priority": 1,
                "tags": ["mysql", "business", "weekly_report"],
                "status": "PENDING",
                "retry_count": 0,
                "max_retry": 1,
                "timeout": 30,
                "created_at": created_at,
            },
            {
                "task_id": "task_4",
                "task_name": "judge_weekly_blocker_trace",
                "description": "判断历史卡点追溯。",
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
                "status": "PENDING",
                "retry_count": 0,
                "max_retry": 1,
                "timeout": 120,
                "created_at": created_at,
            },
            {
                "task_id": "task_5",
                "task_name": "generate_blockers_answer",
                "description": "生成卡点汇总。",
                "task_type": "text_generation",
                "tool": "text_generate_tool",
                "input": {
                    "prompt": "请根据以下数据回答用户问题。",
                    "context": "{{weekly_reports}}\n\n{{weekly_plan_comparison}}",
                    "style": "structured",
                    "audience": "business_user",
                },
                "output_key": "final_result",
                "depends_on": ["task_4"],
                "priority": 2,
                "tags": ["llm", "generation"],
                "status": "PENDING",
                "retry_count": 0,
                "max_retry": 1,
                "timeout": 75,
                "created_at": created_at,
            },
        ],
    }
    state = LangGraphState.create(
        request_id="req_weekly_blocker_judge_context_rewrite",
        session_id="sess_weekly_blocker_judge_context_rewrite",
        user_input="上周所有人汇报的卡点是什么，卡在了哪些问题上",
        runtime_metadata={"client_timezone": "Asia/Shanghai"},
    )
    state.set_raw_plan_text(json.dumps(raw_plan, ensure_ascii=False))

    await builder.parser_node(state)

    text_task = next(task for task in state.planned_tasks if task.tool == "text_generate_tool")
    assert text_task.input["context"] == "{{weekly_blocker_trace_judgement.weekly_blocker_context_text}}"
    assert "{{weekly_plan_comparison}}" not in text_task.input["context"]


async def test_parser_rewrites_plan_tracking_text_context_to_compact_field() -> None:
    builder = await _build_weekly_blocker_fallback_graph_builder()
    created_at = "2026-06-22T07:57:38+00:00"
    raw_plan = {
        "goal": "6月份产品部每周计划有没有完成？没完成的列出来。",
        "tasks": [
            {
                "task_id": "task_1",
                "task_name": "compare_june_product_plan_done",
                "description": "查询产品部6月计划和后续完成。",
                "task_type": "compare_weekly_plan_done",
                "tool": "compare_weekly_plan_done",
                "input": {
                    "department": "产品部",
                    "last_week_start": "2026-06-01",
                    "last_week_end": "2026-06-30",
                    "this_week_start": "2026-06-01",
                    "this_week_end": "2026-07-05",
                    "limit": 500,
                },
                "output_key": "weekly_plan_comparison",
                "depends_on": [],
                "priority": 1,
                "tags": ["mysql", "business", "weekly_report"],
                "status": "PENDING",
                "retry_count": 0,
                "max_retry": 1,
                "timeout": 30,
                "created_at": created_at,
            },
            {
                "task_id": "task_2",
                "task_name": "generate_plan_tracking_answer",
                "description": "生成计划追踪答案。",
                "task_type": "text_generation",
                "tool": "text_generate_tool",
                "input": {
                    "prompt": "请判断每周计划是否完成。",
                    "context": "{{weekly_plan_comparison}}",
                    "style": "structured",
                    "audience": "business_user",
                },
                "output_key": "final_result",
                "depends_on": ["task_1"],
                "priority": 2,
                "tags": ["llm", "generation"],
                "status": "PENDING",
                "retry_count": 0,
                "max_retry": 1,
                "timeout": 75,
                "created_at": created_at,
            },
        ],
    }
    state = LangGraphState.create(
        request_id="req_plan_tracking_context_rewrite",
        session_id="sess_plan_tracking_context_rewrite",
        user_input="6月份产品部每周计划有没有完成？没完成的列出来。",
        runtime_metadata={"client_timezone": "Asia/Shanghai"},
    )
    state.set_raw_plan_text(json.dumps(raw_plan, ensure_ascii=False))

    await builder.parser_node(state)

    text_task = next(task for task in state.planned_tasks if task.tool == "text_generate_tool")
    assert text_task.input["context"] == "{{weekly_plan_comparison.plan_tracking_context_text}}"
    assert text_task.input["timeout_seconds"] == 360
    assert text_task.input["tool_timeout_seconds"] == 360
    assert text_task.input["executor_timeout_seconds"] == 360
    assert text_task.input["temperature"] == 0.2
    assert "plan_tracking_context_rewrite" in state.metadata
    assert state.metadata["plan_tracking_context_rewrite"][0]["task_ids"] == ["task_2"]
    assert state.metadata["plan_tracking_context_rewrite"][0]["timeout_seconds"] == 360


async def test_parser_rewrites_dept_plan_completion_text_context_to_compact_field() -> None:
    builder = await _build_dept_plan_context_graph_builder()
    created_at = "2026-06-22T07:57:38+00:00"
    raw_plan = {
        "goal": "三七计划书中各部门五月份的计划有没有完成",
        "tasks": [
            {
                "task_id": "task_1",
                "task_name": "compare_dept_plan_completion",
                "description": "查询三七计划书和候选完成证据。",
                "task_type": "compare_dept_plan_completion",
                "tool": "compare_dept_plan_completion",
                "input": {
                    "month": "2026-05",
                    "department": None,
                    "include_weekly": True,
                    "include_self_eval": True,
                    "limit": 1000,
                },
                "output_key": "dept_plan_completion",
                "depends_on": [],
                "priority": 1,
                "tags": ["mysql", "business", "dept_plan"],
                "status": "PENDING",
                "retry_count": 0,
                "max_retry": 1,
                "timeout": 30,
                "created_at": created_at,
            },
            {
                "task_id": "task_2",
                "task_name": "generate_dept_plan_answer",
                "description": "生成三七计划完成情况报告。",
                "task_type": "text_generation",
                "tool": "text_generate_tool",
                "input": {
                    "prompt": "请判断三七计划是否完成。",
                    "context": "{{dept_plan_completion}}",
                    "style": "structured",
                    "audience": "business_user",
                },
                "output_key": "final_result",
                "depends_on": ["task_1"],
                "priority": 2,
                "tags": ["llm", "generation"],
                "status": "PENDING",
                "retry_count": 0,
                "max_retry": 1,
                "timeout": 75,
                "created_at": created_at,
            },
        ],
    }
    state = LangGraphState.create(
        request_id="req_dept_plan_context_rewrite",
        session_id="sess_dept_plan_context_rewrite",
        user_input="三七计划书中各部门五月份的计划有没有完成",
        runtime_metadata={"client_timezone": "Asia/Shanghai"},
    )
    state.set_raw_plan_text(json.dumps(raw_plan, ensure_ascii=False))

    await builder.parser_node(state)

    text_task = next(task for task in state.planned_tasks if task.tool == "text_generate_tool")
    assert text_task.input["context"] == "{{dept_plan_completion.dept_plan_completion_context_text}}"
    assert text_task.input["timeout_seconds"] == 360
    assert text_task.input["tool_timeout_seconds"] == 360
    assert text_task.input["executor_timeout_seconds"] == 360
    assert text_task.input["temperature"] == 0.2
    assert "dept_plan_completion_context_rewrite" in state.metadata
    assert state.metadata["dept_plan_completion_context_rewrite"][0]["task_ids"] == ["task_2"]


async def test_parser_adds_plan_tracking_timeouts_when_context_is_already_compact() -> None:
    builder = await _build_weekly_blocker_fallback_graph_builder()
    created_at = "2026-06-22T07:57:38+00:00"
    raw_plan = {
        "goal": "6月份产品部每周计划有没有完成？没完成的列出来。",
        "tasks": [
            {
                "task_id": "task_1",
                "task_name": "compare_june_product_plan_done",
                "description": "查询产品部6月计划和后续完成。",
                "task_type": "compare_weekly_plan_done",
                "tool": "compare_weekly_plan_done",
                "input": {
                    "department": "产品部",
                    "last_week_start": "2026-06-01",
                    "last_week_end": "2026-06-30",
                    "this_week_start": "2026-06-01",
                    "this_week_end": "2026-07-05",
                    "limit": 500,
                },
                "output_key": "weekly_plan_comparison",
                "depends_on": [],
                "priority": 1,
                "tags": ["mysql", "business", "weekly_report"],
                "status": "PENDING",
                "retry_count": 0,
                "max_retry": 1,
                "timeout": 30,
                "created_at": created_at,
            },
            {
                "task_id": "task_2",
                "task_name": "generate_plan_tracking_answer",
                "description": "生成计划追踪答案。",
                "task_type": "text_generation",
                "tool": "text_generate_tool",
                "input": {
                    "prompt": "请判断每周计划是否完成。",
                    "context": "{{weekly_plan_comparison.plan_tracking_context_text}}",
                    "style": "structured",
                    "audience": "business_user",
                },
                "output_key": "final_result",
                "depends_on": ["task_1"],
                "priority": 2,
                "tags": ["llm", "generation"],
                "status": "PENDING",
                "retry_count": 0,
                "max_retry": 1,
                "timeout": 75,
                "created_at": created_at,
            },
        ],
    }
    state = LangGraphState.create(
        request_id="req_plan_tracking_timeout_rewrite",
        session_id="sess_plan_tracking_timeout_rewrite",
        user_input="6月份产品部每周计划有没有完成？没完成的列出来。",
        runtime_metadata={"client_timezone": "Asia/Shanghai"},
    )
    state.set_raw_plan_text(json.dumps(raw_plan, ensure_ascii=False))

    await builder.parser_node(state)

    text_task = next(task for task in state.planned_tasks if task.tool == "text_generate_tool")
    assert text_task.input["context"] == "{{weekly_plan_comparison.plan_tracking_context_text}}"
    assert text_task.input["timeout_seconds"] == 360
    assert text_task.input["tool_timeout_seconds"] == 360
    assert text_task.input["executor_timeout_seconds"] == 360
    assert text_task.input["temperature"] == 0.2


async def test_parser_rewrites_weekly_work_completion_query_as_plan_tracking() -> None:
    builder = await _build_weekly_blocker_fallback_graph_builder()
    created_at = "2026-06-23T02:34:30+00:00"
    raw_plan = {
        "goal": "上周的工作有没有完成呢",
        "tasks": [
            {
                "task_id": "task_1",
                "task_name": "compare_weekly_plan_done",
                "description": "查询上上周计划与上周完成记录。",
                "task_type": "compare_weekly_plan_done",
                "tool": "compare_weekly_plan_done",
                "input": {
                    "department": None,
                    "user_name": None,
                    "last_week_start": "2026-06-08",
                    "last_week_end": "2026-06-14",
                    "this_week_start": "2026-06-15",
                    "this_week_end": "2026-06-21",
                    "limit": 500,
                },
                "output_key": "weekly_plan_comparison",
                "depends_on": [],
                "priority": 1,
                "tags": ["mysql", "business", "weekly_report"],
                "status": "PENDING",
                "retry_count": 0,
                "max_retry": 1,
                "timeout": 30,
                "created_at": created_at,
            },
            {
                "task_id": "task_2",
                "task_name": "generate_weekly_plan_answer",
                "description": "生成上周工作完成情况。",
                "task_type": "text_generation",
                "tool": "text_generate_tool",
                "input": {
                    "prompt": "请输出上周工作完成情况。",
                    "context": "{{weekly_plan_comparison}}",
                    "style": "structured",
                    "audience": "business_user",
                },
                "output_key": "final_result",
                "depends_on": ["task_1"],
                "priority": 2,
                "tags": ["llm", "generation"],
                "status": "PENDING",
                "retry_count": 0,
                "max_retry": 1,
                "timeout": 75,
                "created_at": created_at,
            },
        ],
    }
    state = LangGraphState.create(
        request_id="req_weekly_work_completion_context_rewrite",
        session_id="sess_weekly_work_completion_context_rewrite",
        user_input="上周的工作有没有完成呢",
        runtime_metadata={"client_timezone": "Asia/Shanghai"},
    )
    state.set_raw_plan_text(json.dumps(raw_plan, ensure_ascii=False))

    await builder.parser_node(state)

    text_task = next(task for task in state.planned_tasks if task.tool == "text_generate_tool")
    assert text_task.input["context"] == "{{weekly_plan_comparison.plan_tracking_context_text}}"
    assert text_task.input["timeout_seconds"] == 360
    assert text_task.input["tool_timeout_seconds"] == 360
    assert text_task.input["executor_timeout_seconds"] == 360
    assert text_task.input["temperature"] == 0.2


async def test_planner_invalid_response_uses_weekly_blocker_fallback_plan() -> None:
    builder = await _build_weekly_blocker_fallback_graph_builder()
    state = LangGraphState.create(
        request_id="req_weekly_blocker_fallback",
        session_id="sess_weekly_blocker_fallback",
        user_input="上周所有人汇报的卡点是什么，卡在了哪些问题上",
        runtime_metadata={"client_timezone": "Asia/Shanghai"},
    )
    state.context.runtime.timestamp = datetime(2026, 6, 22, 7, 57, 38, tzinfo=timezone.utc)

    await builder.planner_node(state)

    assert builder.route_after_planner(state) == "parser"
    assert state.raw_plan_text is not None
    raw_plan = json.loads(state.raw_plan_text)
    assert [task["tool"] for task in raw_plan["tasks"]] == [
        "query_weekly_reports",
        "classify_weekly_blockers",
        "compare_weekly_plan_done",
        "judge_weekly_blocker_trace",
        "text_generate_tool",
    ]
    weekly_input = raw_plan["tasks"][0]["input"]
    assert weekly_input["start_date"] == "2026-06-15"
    assert weekly_input["end_date"] == "2026-06-21"
    assert weekly_input["user_name"] is None
    assert weekly_input["department"] is None
    assert weekly_input["item_type"] is None
    assert weekly_input["record_level"] == "reports"
    assert weekly_input["include_evidence_text"] is False
    classify_input = raw_plan["tasks"][1]["input"]
    assert classify_input["weekly_reports_output_key"] == "weekly_reports"
    assert raw_plan["tasks"][1]["depends_on"] == ["task_1"]
    compare_input = raw_plan["tasks"][2]["input"]
    assert compare_input["last_week_start"] == "2026-06-08"
    assert compare_input["last_week_end"] == "2026-06-14"
    assert compare_input["this_week_start"] == "2026-06-15"
    assert compare_input["this_week_end"] == "2026-06-21"
    assert compare_input["weekly_blocker_classification_output_key"] == "weekly_blocker_classification"
    assert compare_input["trace_weeks"] == 2
    assert compare_input["include_historical_blockers"] is True
    assert raw_plan["tasks"][2]["depends_on"] == ["task_1", "task_2"]
    assert raw_plan["tasks"][3]["depends_on"] == ["task_2", "task_3"]
    assert raw_plan["tasks"][4]["depends_on"] == ["task_4"]
    assert raw_plan["tasks"][4]["input"]["context"] == "{{weekly_blocker_trace_judgement.weekly_blocker_context_text}}"
    assert "{{weekly_reports}}" not in raw_plan["tasks"][4]["input"]["context"]
    assert state.metadata["planner_fallback"]["fallback_type"] == "weekly_blocker_mysql_plan"

    await builder.parser_node(state)

    assert [task.tool for task in state.planned_tasks] == [
        "query_weekly_reports",
        "classify_weekly_blockers",
        "compare_weekly_plan_done",
        "judge_weekly_blocker_trace",
        "text_generate_tool",
    ]
    assert state.planned_tasks[0].input["start_date"] == "2026-06-15"
    assert state.planned_tasks[1].input["weekly_reports_output_key"] == "weekly_reports"
    assert state.planned_tasks[2].input["last_week_start"] == "2026-06-08"
    assert state.planned_tasks[2].input["weekly_blocker_classification_output_key"] == "weekly_blocker_classification"
    assert state.planned_tasks[2].input["trace_weeks"] == 2
    assert state.planned_tasks[2].input["include_historical_blockers"] is True
    assert state.planned_tasks[2].depends_on == ["task_1", "task_2"]
    assert state.planned_tasks[3].depends_on == ["task_2", "task_3"]
    assert "risk_and_help" not in state.planned_tasks[4].input["prompt"]
    assert "员工自填卡点" in state.planned_tasks[4].input["prompt"]
    assert "历史卡点" in state.planned_tasks[4].input["prompt"]
    assert state.planned_tasks[4].input["context"] == "{{weekly_blocker_trace_judgement.weekly_blocker_context_text}}"


async def test_planner_invalid_response_uses_dept_plan_completion_fallback_plan() -> None:
    builder = await _build_dept_plan_context_graph_builder()
    state = LangGraphState.create(
        request_id="req_dept_plan_fallback",
        session_id="sess_dept_plan_fallback",
        user_input="三七计划书中各部门五月份的计划有没有按照计划完成，有哪些问题一直卡着没有动",
        runtime_metadata={"client_timezone": "Asia/Shanghai"},
    )
    state.context.runtime.timestamp = datetime(2026, 6, 24, 1, 57, 38, tzinfo=timezone.utc)

    await builder.planner_node(state)

    assert builder.route_after_planner(state) == "parser"
    assert state.raw_plan_text is not None
    raw_plan = json.loads(state.raw_plan_text)
    assert [task["tool"] for task in raw_plan["tasks"]] == [
        "compare_dept_plan_completion",
        "query_weekly_reports",
        "text_generate_tool",
    ]
    compare_input = raw_plan["tasks"][0]["input"]
    assert compare_input["month"] == "2026-05"
    assert compare_input["limit"] == 2000
    assert compare_input["include_weekly"] is True
    assert compare_input["include_self_eval"] is True
    assert compare_input["followup_days"] == 7
    weekly_input = raw_plan["tasks"][1]["input"]
    assert weekly_input["start_date"] == "2026-05-01"
    assert weekly_input["end_date"] == "2026-05-31"
    assert weekly_input["record_level"] == "reports"
    assert weekly_input["include_evidence_text"] is False
    assert weekly_input["limit"] == 2000
    final_task = raw_plan["tasks"][2]
    assert final_task["depends_on"] == ["task_1", "task_2"]
    assert final_task["input"]["context"] == "{{dept_plan_completion.dept_plan_completion_context_text}}"
    assert "risk_and_help" not in final_task["input"]["prompt"]
    assert "所有人均未填写卡点" in final_task["input"]["prompt"]
    assert state.metadata["planner_fallback"]["fallback_type"] == "dept_plan_completion_mysql_plan"

    await builder.parser_node(state)

    assert [task.tool for task in state.planned_tasks] == [
        "compare_dept_plan_completion",
        "query_weekly_reports",
        "text_generate_tool",
    ]
    assert state.planned_tasks[0].input["month"] == "2026-05"
    assert state.planned_tasks[1].input["record_level"] == "reports"
    assert state.planned_tasks[2].input["context"] == "{{dept_plan_completion.dept_plan_completion_context_text}}"
    assert state.planned_tasks[2].input["executor_timeout_seconds"] == 360


async def test_parser_repairs_unsupported_text_generate_tags_before_queue() -> None:
    client, builder = await _build_graph_builder(text_task_tags=["llm", "生成"])
    state = LangGraphState.create(
        request_id="req_runtime_graph_rag_tag_repair",
        session_id="sess_runtime_graph_rag_tag_repair",
        user_input="知识库里技术宝典中操作的注意事项有哪些",
    )

    await builder.planner_node(state)
    assert state.raw_plan_text and "生成" in state.raw_plan_text

    await builder.parser_node(state)

    assert builder.route_after_parser(state) == "queue"
    assert state.parsed_plan is not None
    assert len(state.planned_tasks) == 3
    assert [task.tool for task in state.planned_tasks] == [
        "rag_search_tool",
        "rag_batch_summarize_tool",
        "text_generate_tool",
    ]
    text_task = next(task for task in state.planned_tasks if task.tool == "text_generate_tool")
    assert text_task.tags == ["llm", "generation"]
    assert "生成" not in text_task.tags
    assert text_task.depends_on == ["task_2"]
    assert state.metadata["parser_repair_history"][0]["repair_type"] == "UNSUPPORTED_TAGS"
    assert state.metadata["parser_repair_history"][0]["success"] is True

    routed_text_tool, _ = await builder.dependencies.router.route_task(text_task)
    assert routed_text_tool.name == "text_generate_tool"


async def test_parser_normalizes_rag_batch_tool_name_task_type_before_planned_tasks() -> None:
    _, builder = await _build_graph_builder(rag_summary_task_type="rag_batch_summarize_tool")
    state = LangGraphState.create(
        request_id="req_runtime_graph_rag_alias",
        session_id="sess_runtime_graph_rag_alias",
        user_input="知识库里技术宝典中操作的注意事项有哪些",
    )

    await builder.planner_node(state)
    await builder.parser_node(state)

    rag_task = next(task for task in state.planned_tasks if task.tool == "rag_search_tool")
    summary_task = next(task for task in state.planned_tasks if task.tool == "rag_batch_summarize_tool")
    text_task = next(task for task in state.planned_tasks if task.tool == "text_generate_tool")
    assert summary_task.task_type == "rag_batch_summary"
    assert summary_task.task_id == "task_2"
    assert summary_task.tool == "rag_batch_summarize_tool"
    assert summary_task.depends_on == ["task_1"]
    assert summary_task.input == {"query": state.context.runtime.user_input, "rag_output_key": "rag_context"}
    assert state.context.shared_data["planned_tasks"][1]["task_type"] == "rag_batch_summary"

    routed_rag_tool, _ = await builder.dependencies.router.route_task(rag_task)
    routed_summary_tool, _ = await builder.dependencies.router.route_task(summary_task)
    routed_text_tool, _ = await builder.dependencies.router.route_task(text_task)
    assert routed_rag_tool.name == "rag_search_tool"
    assert routed_summary_tool.name == "rag_batch_summarize_tool"
    assert routed_text_tool.name == "text_generate_tool"


async def test_runtime_executes_rag_dag_with_resolved_rag_batch_task_type() -> None:
    _client, builder = await _build_executable_rag_graph_builder(
        rag_summary_task_type="rag_batch_summarize_tool"
    )
    state = LangGraphState.create(
        request_id="req_runtime_graph_rag_execute_alias",
        session_id="sess_runtime_graph_rag_execute_alias",
        user_input="知识库里报销 SOP 的注意事项有哪些",
    )

    await builder.supervisor_node(state)
    await builder.planner_node(state)
    await builder.parser_node(state)

    summary_task = next(task for task in state.planned_tasks if task.tool == "rag_batch_summarize_tool")
    assert summary_task.task_type == "rag_batch_summary"
    assert state.context.shared_data["planned_tasks"][1]["task_type"] == "rag_batch_summary"

    await builder.queue_node(state)

    queued_summary_task = state.context.tasks[summary_task.task_id]
    assert queued_summary_task.tool == "rag_batch_summarize_tool"
    assert queued_summary_task.task_type == "rag_batch_summary"

    await builder.executor_node(state)

    task_statuses = {
        task_id: task.status.value if hasattr(task.status, "value") else str(task.status)
        for task_id, task in state.context.tasks.items()
    }
    assert task_statuses == {"task_1": "SUCCESS", "task_2": "SUCCESS", "task_3": "SUCCESS"}
    assert state.context.task_results["rag_context"]["query"] == state.context.runtime.user_input
    assert state.context.task_results["rag_summary"]["text"]
    assert state.context.task_results["final_result"]["text"].startswith("最终回答")
    assert any(
        item["task_id"] == "task_2"
        and item["tool_name"] == "rag_batch_summarize_tool"
        and item["routing_result"] == "ALLOWED"
        for item in state.metadata.get("routing_history", [])
    )

    await builder.aggregator_node(state)

    assert state.final_response["success"] is True
    assert state.final_response["phase"] == "COMPLETED"
    assert state.final_response["runtime_task_execution"]["status"] == "SUCCESS"
    assert state.final_response["runtime_task_execution"]["completed_tasks"] == 3
    assert state.final_response["business_result"]["type"] == "generic"


async def test_runtime_rag_flow_passes_raw_context_to_text_generate_when_evidence_json_fails() -> None:
    client, builder = await _build_raw_fallback_rag_graph_builder()
    state = LangGraphState.create(
        request_id="req_runtime_graph_raw_context_fallback",
        session_id="sess_runtime_graph_raw_context_fallback",
        user_input="知识库里财务报表的数据有哪些",
    )

    await builder.supervisor_node(state)
    await builder.planner_node(state)
    await builder.parser_node(state)
    await builder.queue_node(state)
    await builder.executor_node(state)

    task_statuses = {
        task_id: task.status.value if hasattr(task.status, "value") else str(task.status)
        for task_id, task in state.context.tasks.items()
    }
    assert task_statuses == {"task_1": "SUCCESS", "task_2": "SUCCESS", "task_3": "SUCCESS"}
    rag_summary = state.context.task_results["rag_summary"]
    assert rag_summary["extraction_mode"] == "raw_context_fallback"
    assert "Raw Evidence Context" in rag_summary["text"]
    assert "营业收入 10,739,842.03" in rag_summary["text"]
    assert "净利润 -5,418,627.21" in rag_summary["text"]
    assert client.text_generation_prompts
    final_prompt = client.text_generation_prompts[-1]
    assert "Raw Evidence Context" in final_prompt
    assert "营业收入 10,739,842.03" in final_prompt
    assert "净利润 -5,418,627.21" in final_prompt
    assert "JSON 解析失败" in final_prompt
    assert state.context.task_results["final_result"]["text"].startswith("最终回答")

    await builder.aggregator_node(state)

    assert state.final_response["success"] is True
    assert state.final_response["phase"] == "COMPLETED"


async def test_aggregator_separates_runtime_success_from_dept_plan_business_status() -> None:
    builder = await _build_dept_plan_context_graph_builder()
    state = LangGraphState.create(
        request_id="req_dept_plan_business_status",
        session_id="sess_dept_plan_business_status",
        user_input="三七计划书中各部门五月份的计划有没有完成",
    )
    tasks = [
        TaskModel(
            task_id="task_1",
            task_name="compare_dept_plan_completion",
            description="Compare department plan completion.",
            task_type="compare_dept_plan_completion",
            tool="compare_dept_plan_completion",
            input={"month": "2026-05"},
            output_key="dept_plan_completion",
            depends_on=[],
            priority=1,
            tags=["mysql", "business", "dept_plan"],
            status=TaskStatus.SUCCESS,
            retry_count=0,
            max_retry=1,
            timeout=30,
            created_at=datetime(2026, 6, 24, 1, 0, tzinfo=timezone.utc),
        ),
        TaskModel(
            task_id="task_2",
            task_name="generate_dept_plan_completion_answer",
            description="Generate answer.",
            task_type="text_generation",
            tool="text_generate_tool",
            input={"context": "{{dept_plan_completion.dept_plan_completion_context_text}}"},
            output_key="final_result",
            depends_on=["task_1"],
            priority=2,
            tags=["llm", "generation"],
            status=TaskStatus.SUCCESS,
            retry_count=0,
            max_retry=1,
            timeout=120,
            created_at=datetime(2026, 6, 24, 1, 0, tzinfo=timezone.utc),
        ),
    ]
    state.context.tasks = {task.task_id: task for task in tasks}
    state.agent_state.completed_task_ids = {"task_1", "task_2"}
    state.agent_state.pending_task_ids = set()
    state.agent_state.failed_task_ids = set()
    state.agent_state.final_output_ready = True
    state.context.set_task_result(
        "dept_plan_completion",
        {
            "dept_plan_followups": [{"plan_id": 1}],
            "pairing_summary": {"total_plans": 1, "judgement_owner": "backend_llm"},
        },
    )
    state.context.set_task_result("final_result", {"text": "证据不足 1 条"})

    await builder.aggregator_node(state)

    assert state.final_response["success"] is True
    assert state.final_response["runtime_task_execution"]["status"] == "SUCCESS"
    assert state.final_response["runtime_task_execution"]["completed_tasks"] == 2
    assert state.final_response["business_result"]["type"] == "dept_plan_completion_analysis"
    assert "不表示 Runtime 任务是否执行失败" in state.final_response["business_result"]["message"]
    assert state.final_response["business_result"]["source_summary"]["total_plans"] == 1


async def test_checkpoint_resume_resolves_existing_planned_tasks_before_parser_returns() -> None:
    _, builder = await _build_graph_builder()
    state = LangGraphState.create(
        request_id="req_runtime_graph_checkpoint_alias",
        session_id="sess_runtime_graph_checkpoint_alias",
        user_input="知识库里技术宝典中操作的注意事项有哪些",
        graph_metadata={"resume_from_checkpoint": True},
    )
    stale_task = _build_summary_task(task_type="rag_batch_summarize_tool")
    state.set_parsed_plan(
        TaskPlan.model_validate(
            {
                "goal": state.context.runtime.user_input,
                "tasks": [stale_task.model_dump(mode="json")],
            }
        )
    )
    state.set_planned_tasks([stale_task])

    await builder.parser_node(state)

    assert state.planned_tasks[0].task_type == "rag_batch_summary"
    assert state.planned_tasks[0].task_id == stale_task.task_id
    assert state.planned_tasks[0].tool == stale_task.tool
    assert state.planned_tasks[0].input["query"] == state.context.runtime.user_input
    assert state.planned_tasks[0].input["planner_query"] == stale_task.input["query"]
    assert state.planned_tasks[0].input["raw_user_query"] == state.context.runtime.user_input
    assert state.context.shared_data["planned_tasks"][0]["task_type"] == "rag_batch_summary"


async def test_queue_resolves_planned_tasks_that_bypass_parser_before_initialize() -> None:
    _, builder = await _build_graph_builder()
    state = LangGraphState.create(
        request_id="req_runtime_graph_queue_fallback",
        session_id="sess_runtime_graph_queue_fallback",
        user_input="直接进入队列",
    )
    stale_task = _build_summary_task(task_type="totally_wrong_task_type").model_copy(update={"depends_on": []})
    state.set_planned_tasks([stale_task])

    await builder.queue_node(state)

    assert state.planned_tasks[0].task_type == "rag_batch_summary"
    assert state.context.shared_data["planned_tasks"][0]["task_type"] == "rag_batch_summary"
    assert state.context.tasks[stale_task.task_id].task_type == "rag_batch_summary"


async def test_queue_resolves_checkpoint_context_tasks_before_hydrate() -> None:
    _, builder = await _build_graph_builder()
    state = LangGraphState.create(
        request_id="req_runtime_graph_queue_hydrate",
        session_id="sess_runtime_graph_queue_hydrate",
        user_input="恢复队列",
        graph_metadata={"resume_from_checkpoint": True},
    )
    stale_task = _build_summary_task(task_type="rag_batch_summarize_tool").model_copy(update={"depends_on": []})
    state.set_planned_tasks([stale_task])
    state.context.tasks[stale_task.task_id] = stale_task

    await builder.queue_node(state)

    assert state.planned_tasks[0].task_type == "rag_batch_summary"
    assert state.context.tasks[stale_task.task_id].task_type == "rag_batch_summary"
    assert state.queue_snapshot is not None


async def test_parser_preserves_full_user_question_for_rag_task_queries() -> None:
    _, builder = await _build_graph_builder()
    user_input = "请根据九工机器南京研发中心机器人项目__2__xlsx，请分析每个人写最近的下周的计划是什么"
    state = LangGraphState.create(
        request_id="req_runtime_graph_rag_query_preservation",
        session_id="sess_runtime_graph_rag_query_preservation",
        user_input=user_input,
    )
    created_at = "2026-06-09T00:00:00+00:00"
    state.set_raw_plan_text(
        json.dumps(
            {
                "goal": user_input,
                "tasks": [
                    {
                        "task_id": "task_1",
                        "task_name": "search_knowledge_base",
                        "description": "Search the referenced Excel file.",
                        "task_type": "rag_search",
                        "tool": "rag_search_tool",
                        "input": {"query": "九工机器南京研发中心机器人项目__2__xlsx", "top_k": 10},
                        "output_key": "rag_context",
                        "depends_on": [],
                        "priority": 1,
                        "tags": ["rag", "search"],
                        "status": "PENDING",
                        "retry_count": 0,
                        "max_retry": 1,
                        "timeout": 30,
                        "created_at": created_at,
                    },
                    {
                        "task_id": "task_2",
                        "task_name": "summarize_rag_batches",
                        "description": "Summarize retrieved RAG batches.",
                        "task_type": "rag_batch_summary",
                        "tool": "rag_batch_summarize_tool",
                        "input": {"query": "九工机器南京研发中心机器人项目__2__xlsx", "rag_output_key": "rag_context"},
                        "output_key": "rag_summary",
                        "depends_on": ["task_1"],
                        "priority": 2,
                        "tags": ["rag", "summary"],
                        "status": "PENDING",
                        "retry_count": 0,
                        "max_retry": 1,
                        "timeout": 300,
                        "created_at": created_at,
                    },
                    {
                        "task_id": "task_3",
                        "task_name": "generate_answer_from_rag_summary",
                        "description": "Generate final answer.",
                        "task_type": "text_generation",
                        "tool": "text_generate_tool",
                        "input": {"prompt": "Answer from summarized evidence.", "context": "{{rag_summary.text}}", "rag_grounded": True},
                        "output_key": "final_result",
                        "depends_on": ["task_2"],
                        "priority": 3,
                        "tags": ["llm", "generation"],
                        "status": "PENDING",
                        "retry_count": 0,
                        "max_retry": 1,
                        "timeout": 75,
                        "created_at": created_at,
                    },
                ],
            },
            ensure_ascii=False,
        )
    )

    await builder.parser_node(state)

    rag_task = next(task for task in state.planned_tasks if task.tool == "rag_search_tool")
    summary_task = next(task for task in state.planned_tasks if task.tool == "rag_batch_summarize_tool")
    assert rag_task.input["query"] == user_input
    assert rag_task.input["raw_user_query"] == user_input
    assert rag_task.input["planner_query"] == "九工机器南京研发中心机器人项目__2__xlsx"
    assert summary_task.input["query"] == user_input
    assert summary_task.input["raw_user_query"] == user_input
    assert summary_task.input["planner_query"] == "九工机器南京研发中心机器人项目__2__xlsx"
    assert state.metadata["rag_query_preservation"][0]["task_ids"] == ["task_1", "task_2"]


async def test_runtime_graph_routes_knowledge_request_into_rag_dag() -> None:
    client, builder = await _build_graph_builder()
    state = LangGraphState.create(
        request_id="req_runtime_graph_rag",
        session_id="sess_runtime_graph_rag",
        user_input="今年公司都做了哪些申报",
    )

    await builder.supervisor_node(state)
    assert state.supervisor_route == "COMPLEX_TASK"
    assert state.context.shared_data["supervisor_decision"]["route"] == "COMPLEX_TASK"
    assert state.context.shared_data["supervisor_decision"]["needs_planning"] is True
    assert builder.route_after_supervisor(state) == "planner"

    await builder.planner_node(state)
    assert state.raw_plan_text
    assert state.planner_prompt is not None
    assert builder.route_after_planner(state) == "parser"

    await builder.parser_node(state)
    assert state.parsed_plan is not None
    assert state.planned_tasks
    assert builder.route_after_parser(state) == "queue"

    rag_task = next(task for task in state.planned_tasks if task.tool == "rag_search_tool")
    summary_task = next(task for task in state.planned_tasks if task.tool == "rag_batch_summarize_tool")
    text_task = next(task for task in state.planned_tasks if task.tool == "text_generate_tool")
    assert rag_task.output_key == "rag_context"
    assert summary_task.depends_on == [rag_task.task_id]
    assert summary_task.output_key == "rag_summary"
    assert text_task.depends_on == [summary_task.task_id]
    assert text_task.input["context"] == "{{rag_summary.text}}"
    assert text_task.input["rag_grounded"] is True

    await builder.queue_node(state)
    assert state.queue_snapshot is not None
    assert state.queue_snapshot.total_tasks == 3
    assert rag_task.task_id in state.queue_snapshot.ready_task_ids
    assert summary_task.task_id in state.queue_snapshot.blocked_task_ids
    assert text_task.task_id in state.queue_snapshot.blocked_task_ids
    assert state.context.tasks[rag_task.task_id].tool == "rag_search_tool"
    assert state.context.tasks[rag_task.task_id].task_type == "rag_search"
    assert state.context.tasks[summary_task.task_id].tool == "rag_batch_summarize_tool"
    assert state.context.tasks[summary_task.task_id].task_type == "rag_batch_summary"
    assert state.context.tasks[text_task.task_id].tool == "text_generate_tool"
    assert state.context.tasks[text_task.task_id].task_type == "text_generation"

    function_names = [item["function_name"] for item in client.calls]
    assert function_names == ["route_user_request", "emit_task_plan"]
    assert "rag_batch_summarize_tool" in client.calls[1]["prompt"]


async def test_runtime_graph_does_not_force_rag_for_casual_chat() -> None:
    client, builder = await _build_graph_builder()
    state = LangGraphState.create(
        request_id="req_runtime_graph_simple",
        session_id="sess_runtime_graph_simple",
        user_input="你好",
    )

    await builder.supervisor_node(state)
    assert state.supervisor_route == "SIMPLE_TASK"
    assert state.context.shared_data["supervisor_decision"]["route"] == "SIMPLE_TASK"
    assert builder.route_after_supervisor(state) == "simple_task"

    await builder.simple_task_node(state)
    assert state.parsed_plan is not None
    assert state.planned_tasks
    assert state.planned_tasks[0].task_type == "text_generation"
    assert not any(task.tool == "rag_search_tool" for task in state.planned_tasks)

    await builder.queue_node(state)
    assert state.queue_snapshot is not None
    assert state.queue_snapshot.total_tasks == 1
    assert not any(task.tool == "rag_search_tool" for task in state.context.tasks.values())
    assert state.context.tasks["task_1"].tool == "text_generate_tool"

    function_names = [item["function_name"] for item in client.calls]
    assert function_names == ["route_user_request"]
    assert "emit_task_plan" not in function_names



async def test_runtime_repairs_internal_knowledge_llm_reason_qwen_plan_before_executor() -> None:
    _client, builder = await _build_bad_internal_knowledge_graph_builder()
    state = LangGraphState.create(
        request_id="req_runtime_graph_internal_rag_repair",
        session_id="sess_runtime_graph_internal_rag_repair",
        user_input="来料质量检验",
    )

    await builder.supervisor_node(state)
    assert state.supervisor_route == "COMPLEX_TASK"

    await builder.planner_node(state)
    assert state.raw_plan_text
    assert "llm_reason_tool" in state.raw_plan_text
    assert '"model_name":"qwen"' in state.raw_plan_text or '"model_name": "qwen"' in state.raw_plan_text

    await builder.parser_node(state)

    assert builder.route_after_parser(state) == "queue"
    assert [task.tool for task in state.planned_tasks] == [
        "rag_search_tool",
        "rag_batch_summarize_tool",
        "text_generate_tool",
    ]
    assert all(task.input.get("model_name") != "qwen" for task in state.planned_tasks)
    assert state.metadata["parser_repair_history"][0]["repair_type"] == "INTERNAL_KNOWLEDGE_REQUIRES_RAG"
    assert state.metadata["parser_repair_history"][0]["success"] is True

    await builder.queue_node(state)
    await builder.executor_node(state)

    task_statuses = {
        task_id: task.status.value if hasattr(task.status, "value") else str(task.status)
        for task_id, task in state.context.tasks.items()
    }
    assert task_statuses == {"task_1": "SUCCESS", "task_2": "SUCCESS", "task_3": "SUCCESS"}
    assert not any(
        item.get("tool_name") == "llm_reason_tool"
        for item in state.metadata.get("routing_history", [])
    )
