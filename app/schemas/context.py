from __future__ import annotations

import json
import re
from datetime import datetime
from threading import RLock
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from app.schemas.task import TaskModel, TaskStatus, utc_now
from app.utils import configure_runtime_logger, runtime_log

_TEMPLATE_VARIABLE_PATTERN = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")
_NON_BUSINESS_RECORD_FIELDS = {"recorded_at"}


class TokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    model_name: str | None = Field(default=None, description="Model name for the usage record.")
    prompt_tokens: int = Field(default=0, ge=0, description="Prompt token count.")
    completion_tokens: int = Field(default=0, ge=0, description="Completion token count.")
    total_tokens: int = Field(default=0, ge=0, description="Total token count.")
    recorded_at: datetime = Field(default_factory=utc_now, description="Usage record timestamp.")


class ErrorRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    source: str = Field(..., min_length=1, description="Error source component.")
    message: str = Field(..., min_length=1, description="Human-readable error message.")
    task_id: str | None = Field(default=None, description="Related task id when available.")
    details: dict[str, Any] = Field(default_factory=dict, description="Structured error details.")
    created_at: datetime = Field(default_factory=utc_now, description="Error creation timestamp.")


class ToolCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    tool_name: str = Field(..., min_length=1, description="Tool name.")
    task_id: str | None = Field(default=None, description="Related task id when available.")
    idempotency_key: str | None = Field(default=None, description="Logical task idempotency key.")
    status: str = Field(..., min_length=1, description="Execution status for the tool call.")
    started_at: datetime = Field(default_factory=utc_now, description="Tool start timestamp.")
    finished_at: datetime | None = Field(default=None, description="Tool finish timestamp.")
    latency_ms: int | None = Field(default=None, ge=0, description="Tool latency in milliseconds.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Structured tool metadata.")


class ExecutionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    task_id: str = Field(..., min_length=1, description="Task id.")
    idempotency_key: str | None = Field(default=None, description="Logical task idempotency key.")
    status: TaskStatus = Field(..., description="Recorded task status.")
    attempt: int = Field(default=0, ge=0, description="Attempt index.")
    message: str | None = Field(default=None, description="Optional execution message.")
    recorded_at: datetime = Field(default_factory=utc_now, description="Execution record timestamp.")


class RuntimeContext(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    request_id: str = Field(..., min_length=1, description="Request id.")
    session_id: str = Field(..., min_length=1, description="Session id.")
    user_input: str = Field(..., min_length=1, description="Original user input.")
    timestamp: datetime = Field(default_factory=utc_now, description="Request creation timestamp.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional runtime metadata.")

    def set_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value


class IdempotencyRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    idempotency_key: str = Field(..., min_length=1, description="Logical idempotency key.")
    task_id: str = Field(..., min_length=1, description="Task id.")
    output_key: str = Field(..., min_length=1, description="Task output key.")
    tool_name: str | None = Field(default=None, description="Resolved tool name.")
    attempt_key: str | None = Field(default=None, description="Execution-attempt key.")
    final_status: TaskStatus = Field(..., description="Final status associated with the record.")
    success: bool = Field(..., description="Whether the record represents a successful result.")
    irreversible: bool = Field(default=False, description="Whether the task is irreversible.")
    output: Any | None = Field(default=None, description="Successful task output when available.")
    error_message: str | None = Field(default=None, description="Failure message when available.")
    recorded_at: datetime = Field(default_factory=utc_now, description="Record timestamp.")


class ContextStore(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", validate_assignment=True)

    runtime: RuntimeContext = Field(..., description="Runtime request context.")
    tasks: dict[str, TaskModel] = Field(default_factory=dict, description="Registered runtime tasks.")
    task_results: dict[str, Any] = Field(default_factory=dict, description="Successful task outputs by output_key.")
    shared_data: dict[str, Any] = Field(default_factory=dict, description="Shared runtime scratchpad.")
    execution_history: list[ExecutionRecord] = Field(default_factory=list, description="Task execution history.")
    errors: list[ErrorRecord] = Field(default_factory=list, description="Recorded runtime errors.")
    token_usage: list[TokenUsage] = Field(default_factory=list, description="Accumulated token usage records.")
    tool_call_chain: list[ToolCallRecord] = Field(default_factory=list, description="Normalized tool call records.")
    idempotency_records: dict[str, IdempotencyRecord] = Field(
        default_factory=dict,
        description="Logical task execution records by idempotency key.",
    )
    final_output: Any | None = Field(default=None, description="Final aggregated output.")
    _lock: RLock = PrivateAttr(default_factory=RLock)
    _logger = PrivateAttr(default_factory=configure_runtime_logger)

    def register_task(self, task: TaskModel) -> None:
        with self._lock:
            existing = self.tasks.get(task.task_id)
            if existing is not None and existing.model_dump(mode="json") != task.model_dump(mode="json"):
                self._log_conflict(
                    scope="tasks",
                    key=task.task_id,
                    existing=existing.model_dump(mode="json"),
                    incoming=task.model_dump(mode="json"),
                )
                raise ValueError(f"task '{task.task_id}' already exists with conflicting content.")
            self.tasks[task.task_id] = task

    def set_task_result(self, output_key: str, value: Any) -> None:
        with self._lock:
            if output_key in self.task_results and self.task_results[output_key] != value:
                self._log_conflict(
                    scope="task_results",
                    key=output_key,
                    existing=self.task_results[output_key],
                    incoming=value,
                )
                raise ValueError(f"task result output_key '{output_key}' already exists with conflicting content.")
            self.task_results[output_key] = value

    def set_shared_value(self, key: str, value: Any, *, allow_overwrite: bool = True) -> None:
        with self._lock:
            if not allow_overwrite and key in self.shared_data and self.shared_data[key] != value:
                self._log_conflict(
                    scope="shared_data",
                    key=key,
                    existing=self.shared_data[key],
                    incoming=value,
                )
                raise ValueError(f"shared_data '{key}' 已存在且内容冲突，不允许静默覆盖。")
            self.shared_data[key] = value

    def set_shared_mapping_value(
        self,
        key: str,
        item_key: str,
        value: Any,
        *,
        allow_overwrite: bool = False,
    ) -> None:
        with self._lock:
            existing = self.shared_data.get(key, {})
            payload = dict(existing) if isinstance(existing, dict) else {}
            if not allow_overwrite and item_key in payload and payload[item_key] != value:
                self._log_conflict(
                    scope=f"shared_data.{key}",
                    key=item_key,
                    existing=payload[item_key],
                    incoming=value,
                )
                raise ValueError(f"shared mapping '{key}' item '{item_key}' 已存在且内容冲突，不允许静默覆盖。")
            payload[item_key] = value
            self.shared_data[key] = payload

    def append_shared_list(self, key: str, value: Any) -> None:
        with self._lock:
            existing = self.shared_data.get(key, [])
            payload = list(existing) if isinstance(existing, list) else []
            payload.append(value)
            self.shared_data[key] = payload

    def increment_shared_counter(self, key: str, item_key: str) -> int:
        with self._lock:
            existing = self.shared_data.get(key, {})
            payload = dict(existing) if isinstance(existing, dict) else {}
            payload[item_key] = int(payload.get(item_key, 0)) + 1
            self.shared_data[key] = payload
            return int(payload[item_key])

    def add_execution_record(self, record: ExecutionRecord) -> None:
        with self._lock:
            self.execution_history.append(record)

    def add_error(self, error: ErrorRecord) -> None:
        with self._lock:
            self.errors.append(error)

    def add_token_usage(self, usage: TokenUsage) -> None:
        with self._lock:
            self.token_usage.append(usage)

    def add_tool_call(self, record: ToolCallRecord) -> None:
        with self._lock:
            self.tool_call_chain.append(record)

    def set_idempotency_record(self, record: IdempotencyRecord) -> None:
        with self._lock:
            existing = self.idempotency_records.get(record.idempotency_key)
            if existing is not None:
                existing_payload = self._normalize_record_payload(existing.model_dump(mode="json"))
                incoming_payload = self._normalize_record_payload(record.model_dump(mode="json"))
                if existing_payload == incoming_payload:
                    self.idempotency_records[record.idempotency_key] = record
                    return
                if existing.success and existing.irreversible:
                    self._log_conflict(
                        scope="idempotency_records",
                        key=record.idempotency_key,
                        existing=existing_payload,
                        incoming=incoming_payload,
                    )
                    raise ValueError(f"idempotency key '{record.idempotency_key}' already exists with conflicting content.")
            self.idempotency_records[record.idempotency_key] = record

    def get_idempotency_record(self, idempotency_key: str | None) -> IdempotencyRecord | None:
        if idempotency_key is None:
            return None
        with self._lock:
            return self.idempotency_records.get(idempotency_key)

    def ensure_task_runtime_values(self, task: TaskModel) -> TaskModel:
        if not task.idempotency_key:
            task.idempotency_key = f"{self.runtime.request_id}:{task.task_id}:{task.output_key}"
        return task

    def list_render_context_keys(self) -> list[str]:
        with self._lock:
            keys = {"runtime", "task_results", "shared_data", *self.task_results.keys(), *self.shared_data.keys()}
            return sorted(str(key) for key in keys)

    def resolve_template_value(self, path: str) -> Any:
        normalized_path = path.strip()
        if not normalized_path:
            raise KeyError("template path is empty")

        with self._lock:
            runtime_payload = self.runtime.model_dump(mode="json")
            roots: dict[str, Any] = {
                "runtime": runtime_payload,
                "task_results": self.task_results,
                "shared_data": self.shared_data,
                **self.shared_data,
                **self.task_results,
            }
            resolved, value = self._walk_template_path(roots, normalized_path.split("."))
            if not resolved:
                raise KeyError(normalized_path)
            return value

    def render_template_string(self, template: str) -> tuple[str, list[dict[str, Any]]]:
        replacements: list[dict[str, Any]] = []

        def _replace(match: re.Match[str]) -> str:
            normalized_path = match.group(1).strip()
            try:
                value = self.resolve_template_value(normalized_path)
            except KeyError:
                replacements.append({"path": normalized_path, "resolved": False, "value_preview": None})
                return match.group(0)

            rendered = self._stringify_template_value(value)
            replacements.append(
                {
                    "path": normalized_path,
                    "resolved": True,
                    "value_preview": rendered[:200],
                }
            )
            return rendered

        return _TEMPLATE_VARIABLE_PATTERN.sub(_replace, template), replacements

    def _walk_template_path(self, current: Any, segments: list[str]) -> tuple[bool, Any]:
        value = current
        for segment in segments:
            if isinstance(value, dict):
                if segment not in value:
                    return False, None
                value = value[segment]
                continue
            if isinstance(value, list):
                if not segment.isdigit():
                    return False, None
                index = int(segment)
                if index < 0 or index >= len(value):
                    return False, None
                value = value[index]
                continue
            return False, None
        return True, value

    def _stringify_template_value(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        if value is None:
            return "None"
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, indent=2)
        return str(value)

    def _normalize_record_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in payload.items() if key not in _NON_BUSINESS_RECORD_FIELDS}

    def _log_conflict(self, *, scope: str, key: str, existing: Any, incoming: Any) -> None:
        runtime_log(
            layer="context",
            event="error",
            data={
                "scope": scope,
                "key": key,
                "existing": existing,
                "incoming": incoming,
            },
            logger=self._logger,
        )


class AgentState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", validate_assignment=True)

    context: ContextStore = Field(..., description="Shared runtime context.")
    current_task_id: str | None = Field(default=None, description="Current running task id.")
    pending_task_ids: list[str] = Field(default_factory=list, description="Pending task ids.")
    completed_task_ids: list[str] = Field(default_factory=list, description="Completed task ids.")
    failed_task_ids: list[str] = Field(default_factory=list, description="Failed or cancelled task ids.")
    final_output_ready: bool = Field(default=False, description="Whether final output is ready.")
    last_error: ErrorRecord | None = Field(default=None, description="Latest runtime error.")

    def sync_task_status(self, task: TaskModel) -> None:
        task_id = task.task_id

        if task.status == TaskStatus.SUCCESS:
            if task_id not in self.completed_task_ids:
                self.completed_task_ids.append(task_id)
            if task_id in self.pending_task_ids:
                self.pending_task_ids.remove(task_id)
            if task_id in self.failed_task_ids:
                self.failed_task_ids.remove(task_id)
            return

        if task.status in {TaskStatus.FAILED, TaskStatus.TIMEOUT, TaskStatus.CANCELLED}:
            if task_id not in self.failed_task_ids:
                self.failed_task_ids.append(task_id)
            if task_id in self.pending_task_ids:
                self.pending_task_ids.remove(task_id)
            if task_id in self.completed_task_ids:
                self.completed_task_ids.remove(task_id)
            return

        if task_id not in self.pending_task_ids:
            self.pending_task_ids.append(task_id)
        if task_id in self.completed_task_ids:
            self.completed_task_ids.remove(task_id)
        if task_id in self.failed_task_ids:
            self.failed_task_ids.remove(task_id)
