from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from app.planner.plan_validation import (
    ToolCapabilityMap,
    find_internal_knowledge_rag_violation,
    find_unsupported_model_name_violations,
    find_unsupported_tag_violations,
    format_internal_knowledge_rag_error,
    format_unsupported_model_name_error,
    format_unsupported_tag_error,
    load_allowed_model_names_from_config,
    plan_structure_signature,
)
from app.schemas.parser import ParserErrorDetail, TaskParserResult
from app.schemas.planner import TaskPlan
from app.schemas.task import TaskModel, TaskStatus


class TaskParserError(Exception):
    """Raised when planner output cannot be parsed into runtime tasks."""

    def __init__(self, error: ParserErrorDetail) -> None:
        super().__init__(error.message)
        self.error = error


class TaskParser:
    """Parse planner output into validated runtime task models."""

    def __init__(
        self,
        *,
        tool_capabilities: ToolCapabilityMap | None = None,
        allowed_model_names: set[str] | None = None,
    ) -> None:
        self.tool_capabilities = self._normalize_tool_capabilities(tool_capabilities)
        self.allowed_model_names = self._normalize_allowed_model_names(allowed_model_names)

    def with_tool_capabilities(self, tool_capabilities: ToolCapabilityMap | None) -> "TaskParser":
        return TaskParser(
            tool_capabilities=tool_capabilities,
            allowed_model_names=set(self.allowed_model_names),
        )

    def parse_text(self, raw_text: str) -> TaskParserResult:
        normalized_text = raw_text.strip()
        if not normalized_text:
            raise self._error(
                code="empty_response",
                stage="input_validation",
                message="planner output is empty",
                raw_text=raw_text,
            )

        payload = self._load_json_payload(normalized_text)
        plan = self._validate_plan(payload=payload, raw_text=normalized_text)
        tasks = self._build_runtime_tasks(plan=plan, raw_text=normalized_text)
        return TaskParserResult(raw_plan=plan, tasks=tasks)

    def parse_to_task_models(self, raw_text: str) -> list[TaskModel]:
        return self.parse_text(raw_text).tasks

    def _load_json_payload(self, raw_text: str) -> dict[str, Any]:
        candidates = [raw_text]
        extracted = self._extract_json_object(raw_text)
        if extracted and extracted != raw_text:
            candidates.append(extracted)

        last_error: json.JSONDecodeError | None = None
        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError as exc:
                last_error = exc
                continue

            if not isinstance(payload, dict):
                raise self._error(
                    code="json_root_not_object",
                    stage="json_loading",
                    message="planner output root must be a JSON object",
                    raw_text=raw_text,
                    details={"python_type": type(payload).__name__},
                )
            return payload

        raise self._error(
            code="invalid_json",
            stage="json_loading",
            message="planner output is not valid JSON",
            raw_text=raw_text,
            details={
                "line": getattr(last_error, "lineno", None),
                "column": getattr(last_error, "colno", None),
                "json_error": str(last_error) if last_error else None,
            },
        )

    def _validate_plan(self, payload: dict[str, Any], raw_text: str) -> TaskPlan:
        try:
            return TaskPlan.model_validate(payload)
        except ValidationError as exc:
            raise self._error(
                code="schema_validation_failed",
                stage="schema_validation",
                message="planner output does not satisfy the TaskPlan schema",
                raw_text=raw_text,
                details={"validation_errors": exc.errors(include_url=False)},
            ) from exc

    def _build_runtime_tasks(self, plan: TaskPlan, raw_text: str) -> list[TaskModel]:
        self._validate_plan_semantics(plan=plan, raw_text=raw_text)

        runtime_tasks: list[TaskModel] = []
        for planned_task in plan.tasks:
            try:
                runtime_tasks.append(TaskModel.model_validate(planned_task.model_dump()))
            except ValidationError as exc:
                raise self._error(
                    code="runtime_task_validation_failed",
                    stage="runtime_conversion",
                    message=f"task '{planned_task.task_id}' could not be converted into a runtime task",
                    raw_text=raw_text,
                    details={"validation_errors": exc.errors(include_url=False)},
                ) from exc

        return runtime_tasks

    def _validate_plan_semantics(self, plan: TaskPlan, raw_text: str) -> None:
        task_ids = {task.task_id for task in plan.tasks}
        output_keys: set[str] = set()

        for task in plan.tasks:
            if task.status != TaskStatus.PENDING:
                raise self._error(
                    code="invalid_initial_status",
                    stage="semantic_validation",
                    message=f"task '{task.task_id}' must use PENDING as its initial status",
                    raw_text=raw_text,
                    details={"task_id": task.task_id, "status": task.status},
                )

            if task.output_key in output_keys:
                raise self._error(
                    code="duplicate_output_key",
                    stage="semantic_validation",
                    message=f"duplicate output_key detected: '{task.output_key}'",
                    raw_text=raw_text,
                    details={"output_key": task.output_key},
                )
            output_keys.add(task.output_key)

            missing_dependencies = [dependency for dependency in task.depends_on if dependency not in task_ids]
            if missing_dependencies:
                raise self._error(
                    code="missing_dependency",
                    stage="semantic_validation",
                    message=f"task '{task.task_id}' references undefined dependencies",
                    raw_text=raw_text,
                    details={"task_id": task.task_id, "missing_dependencies": missing_dependencies},
                )

        cycle_path = self._find_cycle(plan)
        if cycle_path:
            raise self._error(
                code="cyclic_dependency",
                stage="semantic_validation",
                message="任务计划存在循环依赖 (task plan contains a cyclic dependency)",
                raw_text=raw_text,
                details={"cycle_path": cycle_path},
            )

        rag_violation = find_internal_knowledge_rag_violation(plan=plan)
        if rag_violation is not None:
            raise self._error(
                code="internal_knowledge_requires_rag",
                stage="semantic_validation",
                message=format_internal_knowledge_rag_error(rag_violation),
                raw_text=raw_text,
                details={
                    "violation": rag_violation.as_dict(),
                    "plan_structure": plan_structure_signature(plan),
                },
            )

        model_name_violations = find_unsupported_model_name_violations(
            plan=plan,
            allowed_model_names=self.allowed_model_names,
        )
        if model_name_violations:
            first_violation = model_name_violations[0]
            raise self._error(
                code="unsupported_model_name",
                stage="semantic_validation",
                message=format_unsupported_model_name_error(first_violation),
                raw_text=raw_text,
                details={
                    "violations": [violation.as_dict() for violation in model_name_violations],
                    "allowed_model_names": sorted(self.allowed_model_names),
                    "plan_structure": plan_structure_signature(plan),
                },
            )

        unsupported_tag_violations = find_unsupported_tag_violations(
            plan=plan,
            tool_capabilities=self.tool_capabilities,
        )
        if unsupported_tag_violations:
            first_violation = unsupported_tag_violations[0]
            raise self._error(
                code="unsupported_task_tags",
                stage="semantic_validation",
                message=format_unsupported_tag_error(first_violation),
                raw_text=raw_text,
                details={
                    "violations": [violation.as_dict() for violation in unsupported_tag_violations],
                    "plan_structure": plan_structure_signature(plan),
                },
            )

    def _find_cycle(self, plan: TaskPlan) -> list[str] | None:
        dependency_map = {task.task_id: task.depends_on for task in plan.tasks}
        visiting: set[str] = set()
        visited: set[str] = set()
        stack: list[str] = []

        def dfs(task_id: str) -> list[str] | None:
            if task_id in visiting:
                cycle_start = stack.index(task_id)
                return stack[cycle_start:] + [task_id]
            if task_id in visited:
                return None

            visiting.add(task_id)
            stack.append(task_id)

            for dependency in dependency_map[task_id]:
                cycle = dfs(dependency)
                if cycle:
                    return cycle

            stack.pop()
            visiting.remove(task_id)
            visited.add(task_id)
            return None

        for task_id in dependency_map:
            cycle = dfs(task_id)
            if cycle:
                return cycle

        return None

    def _extract_json_object(self, raw_text: str) -> str | None:
        fence_marker = "```"
        if fence_marker in raw_text:
            fenced = self._extract_code_fence(raw_text)
            if fenced:
                return fenced

        start = raw_text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escaped = False

        for index in range(start, len(raw_text)):
            char = raw_text[index]

            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
                continue

            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return raw_text[start : index + 1]

        return None

    def _extract_code_fence(self, raw_text: str) -> str | None:
        parts = raw_text.split("```")
        for part in parts:
            candidate = part.strip()
            if not candidate:
                continue
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                return candidate
        return None

    def _error(
        self,
        *,
        code: str,
        stage: str,
        message: str,
        raw_text: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> TaskParserError:
        return TaskParserError(
            ParserErrorDetail(
                code=code,
                stage=stage,
                message=message,
                raw_text=raw_text,
                details=details or {},
            )
        )

    def _normalize_tool_capabilities(
        self,
        tool_capabilities: ToolCapabilityMap | None,
    ) -> dict[str, Any]:
        if not tool_capabilities:
            return {}
        return dict(tool_capabilities)

    def _normalize_allowed_model_names(self, allowed_model_names: set[str] | None) -> set[str]:
        if allowed_model_names is None:
            try:
                return load_allowed_model_names_from_config()
            except Exception:
                return set()
        normalized = {str(item).strip() for item in allowed_model_names if str(item).strip()}
        normalized.discard("qwen")
        return normalized
