from __future__ import annotations

import importlib.util
import os

import httpx
import pytest

from app.prompts.supervisor import build_supervisor_prompt
from app.prompts.task_planner import ToolDefinition, build_task_planner_prompt
from app.schemas.model import available_model_providers
from app.utils import load_project_env
from scripts.validate_rag_runtime_e2e import _run_validation


def _build_planner_tools() -> list[ToolDefinition]:
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
        ),
        ToolDefinition(
            name="text_generate_tool",
            description="Generate a final user-facing answer from prompt and context.",
            input_schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "context": {"type": "string"},
                },
            },
            output_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                },
            },
        ),
    ]


def test_supervisor_prompt_guides_knowledge_requests_to_complex_task() -> None:
    prompt = build_supervisor_prompt(
        user_input="请帮我查询公司的报销制度和审批流程。",
        context_summary="None",
    )

    assert "COMPLEX_TASK" in prompt.system_prompt
    assert "knowledge base" in prompt.system_prompt
    assert "contracts" in prompt.system_prompt
    assert "SOPs" in prompt.system_prompt
    assert "Do not classify knowledge-base retrieval requests as SIMPLE_TASK." in prompt.system_prompt
    assert "Choose SIMPLE_TASK for casual chat" in prompt.user_prompt


def test_planner_prompt_includes_rag_guidance_and_example() -> None:
    prompt = build_task_planner_prompt(
        user_input="请根据公司知识库回答报销流程。",
        tools=_build_planner_tools(),
        context_summary="None",
        planning_timestamp="2026-05-25T00:00:00+00:00",
    )

    assert "rag_search_tool" in prompt.user_prompt
    assert "rag_batch_summarize_tool" in prompt.user_prompt
    assert "rag_context" in prompt.user_prompt
    assert "{{rag_summary.text}}" in prompt.user_prompt
    assert '"tool": "rag_search_tool"' in prompt.user_prompt
    assert '"output_key": "rag_context"' in prompt.user_prompt
    assert "source_type" in prompt.user_prompt
    assert "pdf, word, ppt, excel" in prompt.user_prompt
    assert "Do not use business labels such as company_docs" in prompt.user_prompt
    assert "闲聊、通用写作、翻译，或不需要检索即可直接回答的问题，不要强制使用 rag_search_tool" in prompt.user_prompt
    assert "rag_search_tool.input.query must be the exact original User Goal text" in prompt.user_prompt
    assert "retrieval-first plan that uses rag_search_tool before final generation" in prompt.system_prompt


async def _ensure_rag_runtime_environment() -> str:
    load_project_env()

    if importlib.util.find_spec("fastapi") is None:
        pytest.skip("fastapi is not installed; skipping RAG runtime E2E test")

    rag_base_url = os.getenv("RAG_BASE_URL", "").strip().rstrip("/")
    if not rag_base_url:
        pytest.skip("RAG_BASE_URL is not configured; skipping RAG runtime E2E test")

    if not available_model_providers():
        pytest.skip("No LLM API key is configured; skipping RAG runtime E2E test")

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{rag_base_url}/search",
                json={"query": "runtime rag healthcheck", "top_k": 1},
            )
            response.raise_for_status()
    except httpx.HTTPError as exc:
        pytest.skip(f"RAG service is unavailable: {exc}")

    return rag_base_url


@pytest.mark.asyncio
async def test_rag_runtime_e2e_flow() -> None:
    await _ensure_rag_runtime_environment()

    try:
        result = await _run_validation(query=os.getenv("RAG_E2E_QUERY", "测试问题"), top_k=3)
    except ValueError as exc:
        if "required to initialize" in str(exc):
            pytest.skip(f"LLM API key is not configured: {exc}")
        raise
    except httpx.HTTPError as exc:
        pytest.skip(f"Runtime network dependency is unavailable: {exc}")

    task_checks = result["task_execution_checks"]
    template_resolution = result["template_resolution"]
    prompt_checks = result["prompt_injection_checks"]
    llm_checks = result["llm_generation_checks"]

    assert result["runtime_registration_checks"]["rag_tool_import_present"] is True
    assert result["runtime_registration_checks"]["rag_tool_instantiation_present"] is True
    assert result["runtime_registration_checks"]["rag_tool_registration_present"] is True
    assert "rag_search_tool" in result["registered_tools"]
    assert task_checks["task_1_output_key"] == "rag_context"
    assert task_checks["task_2_depends_on"] == ["task_1"]
    assert task_checks["task_2_output_key"] == "rag_summary"
    assert task_checks["task_3_depends_on"] == ["task_2"]
    assert result["task_statuses"]["task_1"] == "SUCCESS"
    assert result["task_statuses"]["task_2"] == "SUCCESS"
    assert result["task_statuses"]["task_3"] == "SUCCESS"
    assert task_checks["task_3_executed"] is True
    assert task_checks["task_3_success"] is True
    assert task_checks["task_3_failed"] is False
    assert task_checks["task_3_tool_result_present"] is True
    assert task_checks["task_3_started_after_task_2_finished"] is True
    assert result["rag_context_present"] is True
    assert result["rag_context_chunk_count"] > 0
    assert template_resolution["resolved"] is True
    assert prompt_checks["resolved_payload_contains_placeholder"] is False
    assert prompt_checks["rendered_user_prompt_contains_placeholder"] is False
    assert prompt_checks["rendered_user_prompt_contains_summary_text"] is True
    assert llm_checks["llm_called"] is True
    assert llm_checks["final_result_matches_tool_output"] is True
    assert result["final_result_present"] is True
    assert bool(result["final_result"].get("text")) is True
    assert result["success"] is True
