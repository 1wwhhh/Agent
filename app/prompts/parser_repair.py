from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field

from app.prompts.language_policy import chinese_primary_policy_block
from app.schemas.parser import RepairType
from app.schemas.planner import TaskPlan

PARSER_REPAIR_PROMPT_NAME = "parser_repair_prompt"
PARSER_REPAIR_PROMPT_VERSION = "v1"


class ParserRepairPromptBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    prompt_name: str = Field(..., min_length=1)
    prompt_version: str = Field(..., min_length=1)
    system_prompt: str = Field(..., min_length=1)
    user_prompt: str = Field(..., min_length=1)
    output_schema: dict[str, object] = Field(..., description="JSON schema enforced by the repair prompt.")


def build_parser_repair_prompt(
    *,
    raw_planner_output: str,
    parser_error_message: str,
    repair_type: RepairType,
) -> ParserRepairPromptBundle:
    schema_json = json.dumps(TaskPlan.model_json_schema(), ensure_ascii=True, indent=2)
    system_prompt = _build_system_prompt(repair_type)
    specific_guidance = _build_repair_specific_guidance(repair_type)
    user_prompt = "\n\n".join(
        [
            f"Repair Type:\n{repair_type.value}",
            f"Parser Error Message:\n{parser_error_message}",
            "Original Planner Output:",
            raw_planner_output,
            "Repair-Specific Rules:",
            specific_guidance,
            "Required TaskPlan JSON Schema:",
            schema_json,
            "Return exactly one JSON object matching this shape:",
            '{"goal": "...", "tasks": []}',
        ]
    )
    return ParserRepairPromptBundle(
        prompt_name=PARSER_REPAIR_PROMPT_NAME,
        prompt_version=PARSER_REPAIR_PROMPT_VERSION,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_schema=TaskPlan.model_json_schema(),
    )


def _build_system_prompt(repair_type: RepairType) -> str:
    base_rules = [
        chinese_primary_policy_block(role="你负责修复 Agent Runtime System 中损坏的 planner output。"),
        "请优先用中文理解修复规则，但输出必须保持 schema 要求的英文 JSON key 和枚举值。",
        "Return only a single strict JSON object.",
        "Do not return markdown.",
        "Do not return explanations.",
        "Do not return code fences.",
        "The output must satisfy the TaskPlan JSON schema exactly.",
        "The output object must include both 'goal' and 'tasks'.",
        "Every task.depends_on entry must reference an existing task_id in the same output.",
        "If there are no dependencies, use an empty array for depends_on.",
        "Only fix invalid fields required by the parser error.",
    ]
    if repair_type == RepairType.INTERNAL_KNOWLEDGE_REQUIRES_RAG:
        base_rules.extend(
            [
                "You may replace an incorrect single-node llm_reason_tool plan with the standard three-task RAG chain when the parser error requires RAG.",
                "Do not change the user goal.",
                "Do not add unrelated tasks.",
                "Do not bypass rag_search_tool.",
            ]
        )
    else:
        base_rules.append(
            "Do not change task count, dependencies, tools, task ids, or the RAG chain structure."
        )
    return "\n".join(base_rules)


def _build_repair_specific_guidance(repair_type: RepairType) -> str:
    if repair_type == RepairType.UNSUPPORTED_TAGS:
        return "\n".join(
            [
                "Planner output contains unsupported task tags.",
                "Fix rules:",
                "- Use only supported_tags from the tool capability.",
                "- Do not use Chinese tags.",
                "- Do not invent tags.",
                "- For text_generate_tool, use [\"llm\", \"generation\"].",
                "- Only fix invalid fields such as tags or schema field names.",
                "- Do not change task count.",
                "- Do not change dependencies.",
                "- Do not change tools or task ids.",
                "- Do not change the RAG chain structure.",
                "Please return the corrected plan JSON only.",
            ]
        )

    if repair_type == RepairType.INTERNAL_KNOWLEDGE_REQUIRES_RAG:
        return "\n".join(
            [
                "当前计划错误：内部知识库 / 公司流程 / 质量检验 / 测试验收 / 来料检验 / SOP / 检验规范类问题不能只使用 llm_reason_tool。",
                "必须修正为标准 RAG 链路：rag_search_tool -> rag_batch_summarize_tool -> text_generate_tool。",
                "允许从错误单节点 llm_reason_tool 修正为标准三段 RAG。",
                "Do not add unrelated tasks.",
                "Do not bypass RAG or rag_search_tool.",
                "Do not keep llm_reason_tool as the only answer chain.",
                "Do not keep illegal model_name fields such as model_name=\"qwen\".",
                "For the final text_generate_tool task, include rag_grounded: true and context: {{rag_summary.text}}.",
                "Please return the corrected plan JSON only.",
            ]
        )

    if repair_type == RepairType.UNSUPPORTED_MODEL_NAME:
        return "\n".join(
            [
                "当前计划包含未授权 model_name。",
                "不允许使用 model_name=\"qwen\"。",
                "除非用户明确指定合法模型，否则删除 model_name 字段。",
                "不要新增 model alias。",
                "不要保留非法 model_name。",
                "Do not map qwen to another model name.",
                "Only fix task.input.model_name; preserve task count, tools, task ids, output keys, and dependencies.",
                "Please return the corrected plan JSON only.",
            ]
        )

    return "Only repair the parser error while preserving the original task graph whenever possible."
