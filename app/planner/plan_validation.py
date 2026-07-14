from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from app.schemas.planner import TaskPlan

INTERNAL_KNOWLEDGE_RAG_KEYWORDS = (
    "公司制度",
    "内部知识库",
    "流程",
    "标准",
    "质量检验",
    "测试验收",
    "来料检验",
    "来料质量检验",
    "财务制度",
    "报销",
    "人事制度",
    "业务规范",
    "内部文档",
    "公司政策",
    "公司流程",
    "作业指导书",
    "sop",
    "检验规范",
)

REQUIRED_RAG_CHAIN_TOOLS = ("rag_search_tool", "rag_batch_summarize_tool", "text_generate_tool")


@dataclass(frozen=True)
class UnsupportedTagViolation:
    task_id: str
    tool: str
    unsupported_tags: list[str]
    supported_tags: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "tool": self.tool,
            "unsupported_tags": list(self.unsupported_tags),
            "supported_tags": list(self.supported_tags),
        }


@dataclass(frozen=True)
class InternalKnowledgeRAGViolation:
    goal: str
    matched_keywords: list[str]
    present_tools: list[str]
    missing_tools: list[str]
    expected_chain: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "matched_keywords": list(self.matched_keywords),
            "present_tools": list(self.present_tools),
            "missing_tools": list(self.missing_tools),
            "expected_chain": list(self.expected_chain),
        }


@dataclass(frozen=True)
class UnsupportedModelNameViolation:
    task_id: str
    tool: str
    model_name: str
    allowed_model_names: list[str]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "tool": self.tool,
            "model_name": self.model_name,
            "allowed_model_names": list(self.allowed_model_names),
            "reason": self.reason,
        }


ToolCapabilityMap = Mapping[str, Any]


def find_unsupported_tag_violations(
    *,
    plan: TaskPlan,
    tool_capabilities: ToolCapabilityMap | None,
) -> list[UnsupportedTagViolation]:
    if not tool_capabilities:
        return []

    violations: list[UnsupportedTagViolation] = []
    for task in plan.tasks:
        supported_tags = _get_supported_tags(tool_capabilities.get(task.tool))
        if supported_tags is None or not supported_tags:
            continue

        supported_tag_set = set(supported_tags)
        unsupported_tags = [tag for tag in task.tags if tag not in supported_tag_set]
        if not unsupported_tags:
            continue

        violations.append(
            UnsupportedTagViolation(
                task_id=task.task_id,
                tool=task.tool,
                unsupported_tags=unsupported_tags,
                supported_tags=supported_tags,
            )
        )

    return violations


def format_unsupported_tag_error(violation: UnsupportedTagViolation) -> str:
    return (
        "Planner generated unsupported tags before execution: task_id="
        f"{violation.task_id} | tool={violation.tool} | unsupported_tags={violation.unsupported_tags} "
        f"| supported_tags={violation.supported_tags} "
        "| reason=task.tags must be a subset of the tool capability supported_tags"
    )


def find_internal_knowledge_rag_violation(*, plan: TaskPlan) -> InternalKnowledgeRAGViolation | None:
    matched_keywords = get_internal_knowledge_rag_keywords(plan.goal)
    if not matched_keywords:
        return None

    present_tools = [task.tool for task in plan.tasks]
    present_tool_set = set(present_tools)
    missing_tools = [tool for tool in REQUIRED_RAG_CHAIN_TOOLS if tool not in present_tool_set]
    only_llm_reason = len(plan.tasks) == 1 and plan.tasks[0].tool == "llm_reason_tool"

    if not missing_tools and not only_llm_reason:
        return None

    return InternalKnowledgeRAGViolation(
        goal=plan.goal,
        matched_keywords=matched_keywords,
        present_tools=present_tools,
        missing_tools=missing_tools,
        expected_chain=list(REQUIRED_RAG_CHAIN_TOOLS),
    )


def get_internal_knowledge_rag_keywords(goal: str) -> list[str]:
    normalized_goal = goal.lower()
    return [keyword for keyword in INTERNAL_KNOWLEDGE_RAG_KEYWORDS if keyword.lower() in normalized_goal]


def format_internal_knowledge_rag_error(violation: InternalKnowledgeRAGViolation) -> str:
    return (
        "Planner generated invalid plan before execution: "
        f"reason=internal knowledge task requires RAG chain | goal={violation.goal!r} "
        f"| matched_keywords={violation.matched_keywords} | present_tools={violation.present_tools} "
        f"| missing_tools={violation.missing_tools} "
        "| expected_chain=rag_search_tool -> rag_batch_summarize_tool -> text_generate_tool"
    )


def find_unsupported_model_name_violations(
    *,
    plan: TaskPlan,
    allowed_model_names: set[str] | None = None,
) -> list[UnsupportedModelNameViolation]:
    allowed_names = {name.strip() for name in (allowed_model_names or set()) if name and name.strip()}
    sorted_allowed_names = sorted(allowed_names)
    violations: list[UnsupportedModelNameViolation] = []

    for task in plan.tasks:
        if not isinstance(task.input, dict):
            continue

        raw_model_name = task.input.get("model_name")
        if raw_model_name is None:
            continue

        model_name = str(raw_model_name).strip()
        if not model_name:
            continue

        if model_name == "qwen":
            violations.append(
                UnsupportedModelNameViolation(
                    task_id=task.task_id,
                    tool=task.tool,
                    model_name=model_name,
                    allowed_model_names=sorted_allowed_names,
                    reason='model_name "qwen" is explicitly forbidden; do not use provider aliases as model names',
                )
            )
            continue

        if not allowed_names:
            continue
        if model_name not in allowed_names:
            violations.append(
                UnsupportedModelNameViolation(
                    task_id=task.task_id,
                    tool=task.tool,
                    model_name=model_name,
                    allowed_model_names=sorted_allowed_names,
                    reason="model_name is not in allowed_model_names from runtime model configuration",
                )
            )

    return violations


def format_unsupported_model_name_error(violation: UnsupportedModelNameViolation) -> str:
    return (
        "Planner generated invalid plan before execution: reason=unsupported model_name | task_id="
        f"{violation.task_id} | tool={violation.tool} | invalid_model_name={violation.model_name} "
        f"| allowed_model_names={violation.allowed_model_names} | detail={violation.reason}"
    )


def load_allowed_model_names_from_config() -> set[str]:
    try:
        from app.schemas.model import ModelConfig, RuntimeLLMConfig
    except Exception:
        return set()

    allowed: set[str] = set()
    try:
        runtime_config = RuntimeLLMConfig.from_env()
    except Exception:
        runtime_config = None

    if runtime_config is not None:
        _add_model_name(allowed, runtime_config.primary.model_name)
        for fallback in runtime_config.fallbacks:
            _add_model_name(allowed, fallback.model_name)

    try:
        model_config = ModelConfig.from_env()
    except Exception:
        model_config = None

    if model_config is not None:
        _add_model_name(allowed, model_config.model_name)

    allowed.discard("qwen")
    return allowed


def plan_structure_signature(plan: TaskPlan) -> list[dict[str, Any]]:
    return [
        {
            "task_id": task.task_id,
            "tool": task.tool,
            "task_type": task.task_type,
            "output_key": task.output_key,
            "depends_on": list(task.depends_on),
        }
        for task in plan.tasks
    ]


def find_plan_structure_changes(
    *,
    original_structure: list[dict[str, Any]],
    repaired_plan: TaskPlan,
) -> list[str]:
    repaired_structure = plan_structure_signature(repaired_plan)
    changes: list[str] = []

    if len(original_structure) != len(repaired_structure):
        changes.append(f"task_count changed from {len(original_structure)} to {len(repaired_structure)}")
        return changes

    for index, (original_task, repaired_task) in enumerate(zip(original_structure, repaired_structure, strict=True)):
        for field_name in ("task_id", "tool", "task_type", "output_key", "depends_on"):
            if original_task.get(field_name) != repaired_task.get(field_name):
                changes.append(
                    f"task[{index}].{field_name} changed from "
                    f"{original_task.get(field_name)!r} to {repaired_task.get(field_name)!r}"
                )

    return changes


def _get_supported_tags(value: Any | None) -> list[str] | None:
    if value is None:
        return None

    raw_tags = value.get("supported_tags") if isinstance(value, dict) else getattr(value, "supported_tags", None)
    if raw_tags is None:
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_tags:
        tag = str(item).strip()
        if not tag:
            continue
        if tag in seen:
            continue
        normalized.append(tag)
        seen.add(tag)

    return normalized


def _add_model_name(allowed: set[str], model_name: str | None) -> None:
    normalized = str(model_name or "").strip()
    if normalized and normalized != "qwen":
        allowed.add(normalized)
