from __future__ import annotations

from app.prompts import (
    LLM_REASON_PROMPT_NAME,
    RAG_ANSWER_PROMPT_NAME,
    RAG_BATCH_SUMMARY_PROMPT_NAME,
    RAG_EVIDENCE_EXTRACTION_PROMPT_NAME,
    RAG_SEARCH_INTENT_PROMPT_NAME,
    TEXT_GENERATE_PROMPT_NAME,
    build_default_prompt_registry,
    build_parser_repair_prompt,
    build_supervisor_prompt,
    build_task_planner_prompt,
)
from app.prompts.task_planner import ToolDefinition
from app.schemas.context import ContextStore, RuntimeContext
from app.schemas.llm import LLMRequest, LLMResponse
from app.schemas.parser import RepairType
from app.tools.llm_client import LLMClient
from app.tools.rag_batch_summarize import RAGBatchSummarizeTool


class _NoopLLMClient(LLMClient):
    async def _generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(text="{}", model_name="noop")


def _planner_tools() -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="rag_search_tool",
            description="Search company knowledge base through existing RAG /search API.",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            output_schema={"type": "object", "properties": {"chunks": {"type": "array"}}},
            supported_task_types=["rag_search"],
            supported_tags=["rag", "search"],
        ),
        ToolDefinition(
            name="rag_batch_summarize_tool",
            description="Summarize RAG context batches for downstream generation.",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            output_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            supported_task_types=["rag_batch_summary"],
            supported_tags=["rag", "summary", "llm"],
        ),
        ToolDefinition(
            name="text_generate_tool",
            description="Generate a final answer from prompt and optional context.",
            input_schema={"type": "object", "properties": {"prompt": {"type": "string"}}},
            output_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            supported_task_types=["text_generation"],
            supported_tags=["llm", "generation", "text"],
        ),
    ]


def test_registered_prompts_use_chinese_primary_language_policy() -> None:
    registry = build_default_prompt_registry()

    for prompt_name in [
        LLM_REASON_PROMPT_NAME,
        TEXT_GENERATE_PROMPT_NAME,
        RAG_SEARCH_INTENT_PROMPT_NAME,
        RAG_BATCH_SUMMARY_PROMPT_NAME,
        RAG_EVIDENCE_EXTRACTION_PROMPT_NAME,
        RAG_ANSWER_PROMPT_NAME,
        "supervisor_route_prompt",
        "task_planner_prompt",
    ]:
        template = registry.get(prompt_name)
        prompt_text = f"{template.system_template}\n{template.user_template}"
        assert "语言策略" in prompt_text
        assert "业务规则" in prompt_text
        assert "JSON key" in prompt_text
        assert "tool name" in prompt_text
        assert "enum value" in prompt_text


def test_runtime_structural_markers_and_identifiers_remain_english() -> None:
    supervisor = build_supervisor_prompt(user_input="请查询公司的报销制度", context_summary="None")
    assert "User Input:" in supervisor.user_prompt
    assert "Runtime Context Summary:" in supervisor.user_prompt
    assert "SIMPLE_TASK" in supervisor.system_prompt
    assert "COMPLEX_TASK" in supervisor.system_prompt
    assert "FeishuSyncToNasTool" in supervisor.system_prompt

    planner = build_task_planner_prompt(
        user_input="请根据公司知识库回答报销流程。",
        tools=_planner_tools(),
        context_summary="None",
        planning_timestamp="2026-05-25T00:00:00+00:00",
    )
    assert "User Goal:" in planner.user_prompt
    assert "Runtime Context Summary:" in planner.user_prompt
    assert "Required JSON Schema:" in planner.user_prompt
    assert "15. Set task.created_at to this exact ISO 8601 UTC timestamp: 2026-05-25T00:00:00+00:00" in planner.user_prompt
    assert "rag_search_tool" in planner.user_prompt
    assert "rag_batch_summarize_tool" in planner.user_prompt
    assert "text_generate_tool" in planner.user_prompt
    assert "{{rag_summary.text}}" in planner.user_prompt
    assert "task.tags must use English tags only" in planner.user_prompt


def test_parser_repair_prompt_uses_policy_but_preserves_schema_contract() -> None:
    prompt = build_parser_repair_prompt(
        raw_planner_output='{"goal":"x","tasks":[]}',
        parser_error_message="unsupported_tags=['生成']",
        repair_type=RepairType.UNSUPPORTED_TAGS,
    )

    assert "语言策略" in prompt.system_prompt
    assert "Return only a single strict JSON object." in prompt.system_prompt
    assert "Required TaskPlan JSON Schema:" in prompt.user_prompt
    assert '"goal": "..."' in prompt.user_prompt
    assert "Do not use Chinese tags" in prompt.user_prompt

