from __future__ import annotations

import json
import time
from uuid import uuid4

from app.planner.parser import TaskParser, TaskParserError
from app.planner.plan_validation import REQUIRED_RAG_CHAIN_TOOLS, find_plan_structure_changes, plan_structure_signature
from app.prompts import ParserRepairPromptBundle, build_parser_repair_prompt
from app.schemas.llm import LLMMessage, LLMRequest
from app.schemas.parser import ParserErrorDetail, RepairResult, RepairType, TaskParserResult
from app.state import LangGraphState
from app.tools.llm_client import LLMClient
from app.utils import runtime_log, runtime_progress

MAX_PARSE_RETRY = 3
STRICT_JSON_VIOLATION_REASON = "repair_output_not_strict_json"


class ParserRepairExhaustedError(Exception):
    def __init__(
        self,
        *,
        retry_count: int,
        last_repair_type: RepairType,
        last_error_message: str,
        original_error_message: str | None = None,
    ) -> None:
        self.retry_count = retry_count
        self.last_repair_type = last_repair_type
        self.last_error_message = last_error_message
        self.original_error_message = original_error_message
        message = (
            "parser repair exhausted after "
            f"{retry_count} retries; last_repair_type={last_repair_type.value}; "
            f"last_error_message={last_error_message}"
        )
        if original_error_message and original_error_message != last_error_message:
            message = f"{message}; original_error_message={original_error_message}"
        super().__init__(message)


class RepairPipeline:
    def __init__(
        self,
        *,
        parser: TaskParser,
        repair_llm_client: LLMClient | None,
        max_parse_retry: int = MAX_PARSE_RETRY,
    ) -> None:
        self.parser = parser
        self.repair_llm_client = repair_llm_client
        self.max_parse_retry = max(1, max_parse_retry)

    async def run(
        self,
        *,
        raw_planner_output: str,
        state: LangGraphState,
    ) -> TaskParserResult:
        try:
            return self.parser.parse_text(raw_planner_output)
        except TaskParserError as exc:
            current_error_message = exc.error.message
            current_repair_type = self._classify_error(exc)
            original_error_message = current_error_message
            original_plan_structure = self._extract_original_plan_structure(exc)

        if self.repair_llm_client is None:
            raise ParserRepairExhaustedError(
                retry_count=0,
                last_repair_type=current_repair_type,
                last_error_message="repair_llm_client is not configured",
                original_error_message=original_error_message,
            )

        for retry_count in range(1, self.max_parse_retry + 1):
            started_at = time.perf_counter()
            repaired_output: str | None = None
            attempt_repair_type = current_repair_type
            prompt_bundle = build_parser_repair_prompt(
                raw_planner_output=raw_planner_output,
                parser_error_message=current_error_message,
                repair_type=attempt_repair_type,
            )
            runtime_progress(
                step="parser:repair",
                status=f"repair attempt {retry_count}",
                detail=f"type={attempt_repair_type.value} | error={current_error_message[:120]}",
            )
            try:
                repaired_output = await self._call_repair_model(
                    state=state,
                    retry_count=retry_count,
                    repair_type=attempt_repair_type,
                    prompt_bundle=prompt_bundle,
                )
                output_contract_violation, violation_reason = self._evaluate_output_contract(repaired_output)
                if not repaired_output:
                    raise ValueError("parser repair model returned empty output")
                parser_result = self.parser.parse_text(repaired_output)
                self._validate_repair_structure_if_needed(
                    original_plan_structure=original_plan_structure,
                    parser_result=parser_result,
                    repaired_output=repaired_output,
                    repair_type=attempt_repair_type,
                )
            except TaskParserError as exc:
                latency_ms = (time.perf_counter() - started_at) * 1000
                current_error_message = exc.error.message
                current_repair_type = self._classify_error(exc)
                output_contract_violation, violation_reason = self._evaluate_output_contract(repaired_output)
                repair_result = RepairResult(
                    success=False,
                    repair_type=attempt_repair_type,
                    retry_count=retry_count,
                    repaired_output=repaired_output,
                    error_message=current_error_message,
                    output_contract_violation=output_contract_violation,
                    violation_reason=violation_reason,
                    latency_ms=latency_ms,
                )
                self._record_repair_history(state=state, repair_result=repair_result)
                runtime_log(
                    layer="parser_repair",
                    event="error",
                    data=repair_result.model_dump(mode="json"),
                )
                if retry_count >= self.max_parse_retry:
                    raise ParserRepairExhaustedError(
                        retry_count=retry_count,
                        last_repair_type=current_repair_type,
                        last_error_message=current_error_message,
                        original_error_message=original_error_message,
                    ) from exc
                continue
            except Exception as exc:
                latency_ms = (time.perf_counter() - started_at) * 1000
                current_error_message = str(exc)
                output_contract_violation, violation_reason = self._evaluate_output_contract(repaired_output)
                repair_result = RepairResult(
                    success=False,
                    repair_type=attempt_repair_type,
                    retry_count=retry_count,
                    repaired_output=repaired_output,
                    error_message=current_error_message,
                    output_contract_violation=output_contract_violation,
                    violation_reason=violation_reason,
                    latency_ms=latency_ms,
                )
                self._record_repair_history(state=state, repair_result=repair_result)
                runtime_log(
                    layer="parser_repair",
                    event="error",
                    data=repair_result.model_dump(mode="json"),
                )
                if retry_count >= self.max_parse_retry:
                    raise ParserRepairExhaustedError(
                        retry_count=retry_count,
                        last_repair_type=current_repair_type,
                        last_error_message=current_error_message,
                        original_error_message=original_error_message,
                    ) from exc
                continue

            latency_ms = (time.perf_counter() - started_at) * 1000
            repair_result = RepairResult(
                success=True,
                repair_type=attempt_repair_type,
                retry_count=retry_count,
                repaired_output=repaired_output,
                error_message=None,
                output_contract_violation=output_contract_violation,
                violation_reason=violation_reason,
                latency_ms=latency_ms,
            )
            self._record_repair_history(state=state, repair_result=repair_result)
            runtime_log(
                layer="parser_repair",
                event="success",
                data=repair_result.model_dump(mode="json"),
            )
            return parser_result

        raise ParserRepairExhaustedError(
            retry_count=self.max_parse_retry,
            last_repair_type=current_repair_type,
            last_error_message=current_error_message,
            original_error_message=original_error_message,
        )

    async def _call_repair_model(
        self,
        *,
        state: LangGraphState,
        retry_count: int,
        repair_type: RepairType,
        prompt_bundle: ParserRepairPromptBundle,
    ) -> str:
        request_id = state.context.runtime.request_id
        session_id = state.context.runtime.session_id
        request = LLMRequest(
            prompt=prompt_bundle.user_prompt,
            system_prompt=prompt_bundle.system_prompt,
            messages=[
                LLMMessage(role="system", content=prompt_bundle.system_prompt),
                LLMMessage(role="user", content=prompt_bundle.user_prompt),
            ],
            request_id=request_id,
            session_id=session_id,
            trace_id=f"{request_id}:parser_repair:{retry_count}:{uuid4().hex}",
            prompt_name=prompt_bundle.prompt_name,
            prompt_version=prompt_bundle.prompt_version,
            response_schema_name="TaskPlan",
            response_schema_version="v1",
            metadata={
                "component": "parser_repair",
                "operation": "repair",
                "repair_type": repair_type.value,
                "retry_count": retry_count,
            },
        )
        response = await self.repair_llm_client.generate(request)
        return response.text.strip()

    def _classify_error(self, error: TaskParserError) -> RepairType:
        code = error.error.code
        if code == "invalid_json":
            return RepairType.INVALID_JSON
        if code in {"missing_dependency", "cyclic_dependency"}:
            return RepairType.INVALID_DEPENDENCY
        if code in {"unsupported_task_tags", "repair_changed_plan_structure"}:
            return RepairType.UNSUPPORTED_TAGS
        if code == "internal_knowledge_requires_rag":
            return RepairType.INTERNAL_KNOWLEDGE_REQUIRES_RAG
        if code == "unsupported_model_name":
            return RepairType.UNSUPPORTED_MODEL_NAME
        if code == "schema_validation_failed":
            validation_errors = error.error.details.get("validation_errors", [])
            if any(
                str(item.get("type", "")).endswith("missing")
                for item in validation_errors
                if isinstance(item, dict)
            ):
                return RepairType.MISSING_FIELD
        return RepairType.INVALID_SCHEMA

    def _extract_original_plan_structure(self, error: TaskParserError) -> list[dict[str, object]] | None:
        if error.error.code not in {"unsupported_task_tags", "unsupported_model_name"}:
            return None
        raw_structure = error.error.details.get("plan_structure")
        if not isinstance(raw_structure, list):
            return None
        return [item for item in raw_structure if isinstance(item, dict)]

    def _validate_repair_structure_if_needed(
        self,
        *,
        original_plan_structure: list[dict[str, object]] | None,
        parser_result: TaskParserResult,
        repaired_output: str,
        repair_type: RepairType,
    ) -> None:
        if repair_type == RepairType.INTERNAL_KNOWLEDGE_REQUIRES_RAG:
            self._validate_internal_knowledge_rag_repair_structure(
                parser_result=parser_result,
                repaired_output=repaired_output,
            )
            return

        if original_plan_structure is None:
            return

        structure_changes = find_plan_structure_changes(
            original_structure=original_plan_structure,
            repaired_plan=parser_result.raw_plan,
        )
        if not structure_changes:
            return

        raise TaskParserError(
            ParserErrorDetail(
                code="repair_changed_plan_structure",
                stage="repair_validation",
                message=(
                    "parser repair changed task structure; repair may only fix invalid fields "
                    "such as tags or schema field names; "
                    f"structure_changes={structure_changes}"
                ),
                raw_text=repaired_output,
                details={
                    "structure_changes": structure_changes,
                    "original_plan_structure": original_plan_structure,
                    "repaired_plan_structure": plan_structure_signature(parser_result.raw_plan),
                },
            )
        )

    def _validate_internal_knowledge_rag_repair_structure(
        self,
        *,
        parser_result: TaskParserResult,
        repaired_output: str,
    ) -> None:
        repaired_tools = [task.tool for task in parser_result.raw_plan.tasks]
        expected_tools = list(REQUIRED_RAG_CHAIN_TOOLS)
        if repaired_tools == expected_tools:
            return

        raise TaskParserError(
            ParserErrorDetail(
                code="repair_changed_plan_structure",
                stage="repair_validation",
                message=(
                    "internal knowledge repair must produce the standard RAG chain only; "
                    f"expected_chain={expected_tools}; repaired_tools={repaired_tools}"
                ),
                raw_text=repaired_output,
                details={
                    "expected_chain": expected_tools,
                    "repaired_tools": repaired_tools,
                    "repaired_plan_structure": plan_structure_signature(parser_result.raw_plan),
                },
            )
        )

    def _evaluate_output_contract(self, repaired_output: str | None) -> tuple[bool, str | None]:
        if self._is_strict_json_object(repaired_output):
            return False, None
        return True, STRICT_JSON_VIOLATION_REASON

    def _is_strict_json_object(self, repaired_output: str | None) -> bool:
        if repaired_output is None:
            return False
        normalized_output = repaired_output.strip()
        if not normalized_output:
            return False
        try:
            payload = json.loads(normalized_output)
        except json.JSONDecodeError:
            return False
        return isinstance(payload, dict)

    def _record_repair_history(self, *, state: LangGraphState, repair_result: RepairResult) -> None:
        history = list(state.metadata.get("parser_repair_history", []))
        history.append(
            {
                "repair_type": repair_result.repair_type.value,
                "retry_count": repair_result.retry_count,
                "success": repair_result.success,
                "latency_ms": repair_result.latency_ms,
                "error_message": repair_result.error_message,
                "output_contract_violation": repair_result.output_contract_violation,
                "violation_reason": repair_result.violation_reason,
                "timestamp": repair_result.timestamp.isoformat(),
            }
        )
        state.metadata["parser_repair_history"] = history
        state.context.set_shared_value("parser_repair_history", history)
