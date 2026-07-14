"""Prompt 包。"""

from app.prompts.base import PromptTemplate, RenderedPrompt
from app.prompts.llm_tools import (
    LLM_REASON_PROMPT_NAME,
    LLM_REASON_PROMPT_VERSION,
    RAG_BATCH_SUMMARY_PROMPT_NAME,
    RAG_BATCH_SUMMARY_PROMPT_VERSION,
    RAG_SEARCH_INTENT_PROMPT_NAME,
    RAG_SEARCH_INTENT_PROMPT_VERSION,
    RAG_EVIDENCE_EXTRACTION_PROMPT_NAME,
    RAG_EVIDENCE_EXTRACTION_PROMPT_VERSION,
    RAG_ANSWER_PROMPT_NAME,
    RAG_ANSWER_PROMPT_VERSION,
    TEXT_GENERATE_PROMPT_NAME,
    TEXT_GENERATE_PROMPT_VERSION,
    get_llm_reason_prompt_template,
    get_rag_answer_prompt_template,
    get_rag_search_intent_prompt_template,
    get_rag_batch_summary_prompt_template,
    get_rag_evidence_extraction_prompt_template,
    get_text_generate_prompt_template,
)
from app.prompts.registry import PromptRegistry
from app.prompts.parser_repair import (
    PARSER_REPAIR_PROMPT_NAME,
    PARSER_REPAIR_PROMPT_VERSION,
    ParserRepairPromptBundle,
    build_parser_repair_prompt,
)
from app.prompts.supervisor import (
    SUPERVISOR_PROMPT_NAME,
    SUPERVISOR_PROMPT_VERSION,
    SupervisorPromptBundle,
    build_supervisor_prompt,
    get_supervisor_prompt_template,
)
from app.prompts.task_planner import (
    TASK_PLANNER_PROMPT_NAME,
    TASK_PLANNER_PROMPT_VERSION,
    PlannerPromptBundle,
    ToolDefinition,
    build_default_planner_prompt_registry,
    build_task_planner_prompt,
    get_task_planner_prompt_template,
)


def build_default_prompt_registry() -> PromptRegistry:
    registry = PromptRegistry()
    registry.register_many(
        [
            get_supervisor_prompt_template(),
            get_task_planner_prompt_template(),
            get_llm_reason_prompt_template(),
            get_rag_search_intent_prompt_template(),
            get_rag_batch_summary_prompt_template(),
            get_rag_evidence_extraction_prompt_template(),
            get_text_generate_prompt_template(),
            get_rag_answer_prompt_template(),
        ]
    )
    return registry


__all__ = [
    "LLM_REASON_PROMPT_NAME",
    "LLM_REASON_PROMPT_VERSION",
    "RAG_BATCH_SUMMARY_PROMPT_NAME",
    "RAG_BATCH_SUMMARY_PROMPT_VERSION",
    "RAG_SEARCH_INTENT_PROMPT_NAME",
    "RAG_SEARCH_INTENT_PROMPT_VERSION",
    "RAG_EVIDENCE_EXTRACTION_PROMPT_NAME",
    "RAG_EVIDENCE_EXTRACTION_PROMPT_VERSION",
    "RAG_ANSWER_PROMPT_NAME",
    "RAG_ANSWER_PROMPT_VERSION",
    "PARSER_REPAIR_PROMPT_NAME",
    "PARSER_REPAIR_PROMPT_VERSION",
    "PlannerPromptBundle",
    "ParserRepairPromptBundle",
    "PromptRegistry",
    "PromptTemplate",
    "RenderedPrompt",
    "SUPERVISOR_PROMPT_NAME",
    "SUPERVISOR_PROMPT_VERSION",
    "SupervisorPromptBundle",
    "TASK_PLANNER_PROMPT_NAME",
    "TASK_PLANNER_PROMPT_VERSION",
    "TEXT_GENERATE_PROMPT_NAME",
    "TEXT_GENERATE_PROMPT_VERSION",
    "ToolDefinition",
    "build_default_planner_prompt_registry",
    "build_default_prompt_registry",
    "build_parser_repair_prompt",
    "build_supervisor_prompt",
    "build_task_planner_prompt",
    "get_llm_reason_prompt_template",
    "get_rag_answer_prompt_template",
    "get_rag_search_intent_prompt_template",
    "get_rag_batch_summary_prompt_template",
    "get_rag_evidence_extraction_prompt_template",
    "get_supervisor_prompt_template",
    "get_task_planner_prompt_template",
    "get_text_generate_prompt_template",
]
