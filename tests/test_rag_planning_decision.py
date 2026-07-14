from __future__ import annotations

import json
from dataclasses import dataclass

from app.agents import SupervisorAgent
from app.planner import LLMTaskPlanner
from app.prompts.task_planner import ToolDefinition
from app.schemas.llm import LLMFunctionCall, LLMRequest, LLMResponse
from app.tools.llm_client import LLMClient


def _build_tool_catalog() -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="rag_search_tool",
            description="Search company knowledge base through existing RAG /search API.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"},
                },
            },
            output_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "chunks": {"type": "array"},
                    "joined_context": {"type": "string"},
                    "summary": {"type": "string"},
                    "context_batches": {"type": "array"},
                },
            },
            supported_task_types=["rag_search"],
            default_task_type="rag_search",
            supported_tags=["rag", "search", "knowledge_base"],
        ),
        ToolDefinition(
            name="rag_batch_summarize_tool",
            description="Summarize RAG context batches for downstream generation.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "rag_output_key": {"type": "string"},
                },
            },
            output_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "summary": {"type": "string"},
                    "batch_summaries": {"type": "array"},
                },
            },
            supported_task_types=["rag_batch_summary"],
            default_task_type="rag_batch_summary",
            supported_tags=["rag", "summary", "llm"],
        ),
        ToolDefinition(
            name="text_generate_tool",
            description="Generate a final answer from prompt and optional context.",
            input_schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "context": {"type": "string"},
                    "style": {"type": "string"},
                    "audience": {"type": "string"},
                    "rag_grounded": {"type": "boolean"},
                },
            },
            output_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                },
            },
            supported_task_types=["text_generation"],
            default_task_type="text_generation",
            supported_tags=["llm", "generation", "text"],
        ),
        ToolDefinition(
            name="llm_reason_tool",
            description="Reason over general-purpose tasks that do not need retrieval.",
            input_schema={"type": "object", "properties": {"prompt": {"type": "string"}}},
            output_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            supported_task_types=["reasoning"],
            default_task_type="reasoning",
            supported_tags=["reasoning", "llm"],
        ),
    ]


@dataclass
class CapturedCall:
    function_name: str
    prompt: str
    system_prompt: str


class ScenarioStructuredLLMClient(LLMClient):
    def __init__(self) -> None:
        super().__init__(timeout_seconds=30, model_name="scenario-structured-client", model_version="test-v1")
        self.calls: list[CapturedCall] = []

    async def _generate(self, request: LLMRequest) -> LLMResponse:
        function_name = request.tool_choice or (request.function_schemas[0].name if request.function_schemas else None)
        prompt = request.prompt or ""
        system_prompt = request.system_prompt or ""
        self.calls.append(
            CapturedCall(
                function_name=function_name or "unknown",
                prompt=prompt,
                system_prompt=system_prompt,
            )
        )

        if function_name == "route_user_request":
            payload = self._supervisor_payload(prompt)
        elif function_name == "emit_task_plan":
            payload = self._planner_payload(prompt)
        else:
            raise AssertionError(f"unexpected function name: {function_name}")

        function_schema = request.function_schemas[0] if request.function_schemas else None
        return LLMResponse(
            text="structured response",
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
            raw_response={"provider": "scenario_structured_client"},
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
            "reason": "can be completed without retrieval or decomposition",
        }

    def _planner_payload(self, prompt: str) -> dict[str, object]:
        goal = _extract_between(prompt, "User Goal:\n", "\n\nRuntime Context Summary:")
        created_at = _extract_created_at(prompt)
        if _is_knowledge_request(goal):
            return {
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
        return {
            "goal": goal,
            "tasks": [
                {
                    "task_id": "task_1",
                    "task_name": "write_summary",
                    "description": "Write the requested content directly.",
                    "task_type": "text_generation",
                    "tool": "text_generate_tool",
                    "input": {
                        "prompt": goal,
                        "style": "clear",
                        "audience": "business_user",
                    },
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


def _extract_between(text: str, start_marker: str, end_marker: str) -> str:
    start_index = text.index(start_marker) + len(start_marker)
    end_index = text.index(end_marker, start_index)
    return text[start_index:end_index].strip()


def _extract_created_at(prompt: str) -> str:
    marker = "15. Set task.created_at to this exact ISO 8601 UTC timestamp: "
    start_index = prompt.index(marker) + len(marker)
    end_index = prompt.index("\n", start_index)
    return prompt[start_index:end_index].strip()


def _is_knowledge_request(user_input: str) -> bool:
    normalized = user_input.lower()
    return any(
        token in normalized
        for token in [
            "知识",
            "报销",
            "申报",
            "公司",
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
            "document",
            "policy",
            "contract",
        ]
    )


async def _classify(user_input: str) -> tuple[ScenarioStructuredLLMClient, object]:
    client = ScenarioStructuredLLMClient()
    agent = SupervisorAgent(client=client)
    result = await agent.classify(
        user_input=user_input,
        request_id="req_supervisor_test",
        session_id="sess_supervisor_test",
        context_summary="None",
    )
    return client, result


async def _plan(user_input: str) -> tuple[ScenarioStructuredLLMClient, object]:
    client = ScenarioStructuredLLMClient()
    planner = LLMTaskPlanner(client=client)
    result = await planner.plan(
        user_input=user_input,
        request_id="req_planner_test",
        session_id="sess_planner_test",
        tools=_build_tool_catalog(),
        context_summary="None",
    )
    return client, result


async def test_knowledge_request_enters_planner() -> None:
    client, result = await _classify("今年公司都做了哪些申报")

    assert result.decision.route == "COMPLEX_TASK"
    assert result.decision.complexity == "complex"
    assert result.decision.needs_planning is True
    assert result.decision.route != "SIMPLE_TASK"
    assert len(client.calls) == 1
    assert client.calls[0].function_name == "route_user_request"
    assert "Do not classify knowledge-base retrieval requests as SIMPLE_TASK." in client.calls[0].system_prompt
    assert "Choose COMPLEX_TASK when the runtime should search a knowledge base" in client.calls[0].prompt


async def test_planner_generates_rag_dag_for_knowledge_request() -> None:
    client, result = await _plan("今年公司都做了哪些申报")

    assert len(result.task_plan.tasks) == 3
    task_1 = result.task_plan.tasks[0]
    task_2 = result.task_plan.tasks[1]
    task_3 = result.task_plan.tasks[2]

    assert task_1.tool == "rag_search_tool"
    assert task_1.output_key == "rag_context"
    assert task_2.tool == "rag_batch_summarize_tool"
    assert task_2.depends_on == ["task_1"]
    assert task_2.output_key == "rag_summary"
    assert task_3.task_id == "task_3"
    assert task_3.tool == "text_generate_tool"
    assert task_3.depends_on == ["task_2"]
    assert task_3.input["context"] == "{{rag_summary.text}}"
    assert task_3.input["rag_grounded"] is True
    assert "生成" not in task_3.tags
    assert "generation" in task_3.tags
    assert len(client.calls) == 1
    assert client.calls[0].function_name == "emit_task_plan"
    assert "rag_search_tool" in client.calls[0].prompt
    assert "rag_batch_summarize_tool" in client.calls[0].prompt
    assert "{{rag_summary.text}}" in client.calls[0].prompt
    assert "rag_grounded" in client.calls[0].prompt
    assert "the final text_generate_tool input must include rag_grounded: true" in client.calls[0].prompt
    assert "rag_search_tool.input.query must be the exact original User Goal text" in client.calls[0].prompt
    assert "闲聊、通用写作、翻译，或不需要检索即可直接回答的问题，不要强制使用 rag_search_tool" in client.calls[0].prompt
    assert "you must use the fixed RAG chain rag_search_tool -> rag_batch_summarize_tool -> text_generate_tool" in client.calls[0].prompt
    assert "prefer rag_search_tool" not in client.calls[0].prompt.lower()
    assert "Never output model_name as \"qwen\"" in client.calls[0].prompt
    assert "Do not invent model_name values" in client.calls[0].prompt
    assert "task_type must not be a tool name" in client.calls[0].prompt
    assert "task_type must be selected from the selected tool's supported_task_types" in client.calls[0].prompt
    assert "tool = rag_batch_summarize_tool" in client.calls[0].prompt
    assert "task_type must be rag_batch_summary" in client.calls[0].prompt
    assert "never be rag_batch_summarize_tool" in client.calls[0].prompt
    assert "supported_tags" in client.calls[0].prompt
    assert "supported_tags: [\"llm\", \"generation\", \"text\"]" in client.calls[0].prompt
    assert "Every task.tags value must come from the selected tool capability supported_tags" in client.calls[0].prompt
    assert "task.tags must use English tags only; never use Chinese tags" in client.calls[0].prompt
    assert "For text_generate_tool, tags must be [\"llm\", \"generation\"]" in client.calls[0].prompt

    rag_example_text = client.calls[0].prompt.split("RAG DAG Example:\n", 1)[1].split(
        "\n\nRequired JSON Schema:", 1
    )[0]
    rag_example = json.loads(rag_example_text)
    task_3_example = rag_example["tasks"][2]
    assert task_3_example["task_id"] == "task_3"
    assert "id" not in task_3_example
    assert task_3_example["tool"] == "text_generate_tool"
    assert task_3_example["task_type"] == "text_generation"
    assert task_3_example["tags"] == ["llm", "generation"]
    assert task_3_example["input"]["rag_grounded"] is True
    assert task_3_example["input"]["context"] == "{{rag_summary.text}}"
    assert task_3_example["depends_on"] == ["task_2"]
    assert "生成" not in json.dumps(task_3_example, ensure_ascii=False)


async def test_casual_chat_is_not_forced_into_rag() -> None:
    client, result = await _classify("你好")

    assert result.decision.route == "SIMPLE_TASK"
    assert result.decision.complexity == "simple"
    assert result.decision.needs_planning is False
    assert len(client.calls) == 1
    assert client.calls[0].function_name == "route_user_request"


async def test_plain_writing_does_not_default_to_rag_retrieval() -> None:
    client, result = await _plan("写一段欢迎语")

    assert len(result.task_plan.tasks) == 1
    task_1 = result.task_plan.tasks[0]
    assert task_1.tool == "text_generate_tool"
    assert task_1.output_key == "final_result"
    assert task_1.depends_on == []
    assert "context" not in task_1.input
    assert all(task.tool != "rag_search_tool" for task in result.task_plan.tasks)
    assert len(client.calls) == 1
    assert client.calls[0].function_name == "emit_task_plan"



async def test_planner_generates_rag_dag_for_quality_inspection_request() -> None:
    client, result = await _plan("来料质量检验")

    assert [task.tool for task in result.task_plan.tasks] == [
        "rag_search_tool",
        "rag_batch_summarize_tool",
        "text_generate_tool",
    ]
    assert all("model_name" not in task.input for task in result.task_plan.tasks)
    assert "must use the fixed RAG chain" in client.calls[0].prompt
    assert "llm_reason_tool" in client.calls[0].prompt
    assert "Never output model_name as \"qwen\"" in client.calls[0].prompt
