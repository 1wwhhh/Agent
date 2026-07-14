from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import re
from typing import Any

from app.executor.exceptions import ExecutorCrashError, NonRetryableToolError, ToolTimeoutError
from app.executor.retry_policy import RetryPolicy
from app.queue.task_queue import TaskQueue
from app.router.task_router import TaskRouter, TaskRouterError
from app.schemas.context import ContextStore, ErrorRecord, IdempotencyRecord, TokenUsage, ToolCallRecord
from app.schemas.executor import ExecutorErrorDetail, TaskExecutionResult
from app.schemas.router import ToolRouteDecision
from app.schemas.task import TaskModel, TaskStatus, utc_now
from app.schemas.tool import ToolResult
from app.utils import configure_runtime_logger, runtime_log

FAIL_FAST_TOOL_NAME = "text_generate_tool"
FAIL_FAST_LLM_TIMEOUT_SECONDS = 45
FAIL_FAST_TOOL_TIMEOUT_SECONDS = 75
FAIL_FAST_EXECUTOR_TIMEOUT_SECONDS = 90

REASON_TOOL_NAME = "llm_reason_tool"
REASON_LLM_TIMEOUT_SECONDS = 45
REASON_TOOL_TIMEOUT_SECONDS = 75
REASON_EXECUTOR_TIMEOUT_SECONDS = 90

_TEMPLATE_VARIABLE_PATTERN = re.compile(r"{{\s*([^{}]+?)\s*}}")


class TaskExecutorError(Exception):
    def __init__(self, error: ExecutorErrorDetail) -> None:
        super().__init__(error.message)
        self.error = error


class TaskExecutor:
    def __init__(
        self,
        *,
        context: ContextStore,
        queue: TaskQueue,
        router: TaskRouter,
        checkpoint_saver: Callable[[str, dict[str, object] | None], Awaitable[None]] | None = None,
        result_callback: Callable[[TaskExecutionResult], Awaitable[None]] | None = None,
        failure_recorder: Callable[[dict[str, object]], None] | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.context = context
        self.queue = queue
        self.router = router
        self.checkpoint_saver = checkpoint_saver
        self.result_callback = result_callback
        self.failure_recorder = failure_recorder
        self.retry_policy = retry_policy or RetryPolicy()
        self.logger = configure_runtime_logger()

    async def execute_until_complete(
        self,
        *,
        batch_limit: int | None = None,
        stop_after_results: int | None = None,
    ) -> list[TaskExecutionResult]:
        results: list[TaskExecutionResult] = []
        while True:
            if await self.queue.is_complete():
                break

            ready_tasks = await self.queue.get_ready_tasks(limit=batch_limit)
            if not ready_tasks:
                if await self.queue.is_complete():
                    break
                continue

            batch_results = await self.execute_ready_tasks([task.task_id for task in ready_tasks])
            results.extend(batch_results)

            if stop_after_results is not None and len(results) >= stop_after_results:
                break

        return results

    async def execute_ready_tasks(self, task_ids: list[str]) -> list[TaskExecutionResult]:
        coroutines = [self.execute_task(task_id) for task_id in task_ids]
        raw_results = await asyncio.gather(*coroutines, return_exceptions=True)

        results: list[TaskExecutionResult] = []
        for task_id, item in zip(task_ids, raw_results, strict=False):
            if isinstance(item, TaskExecutionResult):
                results.append(item)
                continue
            results.append(await self._handle_unexpected_batch_exception(task_id=task_id, exc=item))
        return results

    async def execute_task(self, task_id: str) -> TaskExecutionResult:
        try:
            return await self._execute_task_impl(task_id)
        except Exception as exc:
            return await self._handle_unexpected_batch_exception(task_id=task_id, exc=exc)

    async def _execute_task_impl(self, task_id: str) -> TaskExecutionResult:
        task = self._get_task(task_id)

        restored = await self._try_restore_from_idempotency(task)
        if restored is not None:
            return restored

        await self.queue.mark_task_running(task_id)
        task = self._get_task(task_id)
        started_at = utc_now()
        attempt_key = self._build_attempt_key(task)
        routed_tool_name = task.tool
        route_decision: ToolRouteDecision | None = None
        tool_result: ToolResult | None = None
        timeout_seconds: int | None = None

        try:
            tool, route_decision = await self.router.route_task(task)
            routed_tool_name = tool.name
            payload = self._build_tool_payload(task)
            timeout_settings = self._resolve_timeout_settings(
                task=task,
                tool_timeout=getattr(tool, "timeout", task.timeout),
                payload=payload,
            )
            payload = {**payload, **timeout_settings}
            timeout_seconds = int(timeout_settings["executor_timeout_seconds"])

            tool_result = await asyncio.wait_for(
                tool.arun(payload=payload, context=self.context),
                timeout=timeout_seconds,
            )

            if not tool_result.success:
                return await self._handle_failure(
                    task=task,
                    attempt_key=attempt_key,
                    routed_tool_name=routed_tool_name,
                    route_decision=route_decision,
                    tool_result=tool_result,
                    failure_exception=None,
                    error_message=tool_result.error or "tool execution failed",
                    requested_status=TaskStatus.FAILED,
                    started_at=started_at,
                    timeout_seconds=timeout_seconds,
                )

            try:
                self._record_tool_call(
                    task=task,
                    tool_name=routed_tool_name,
                    status=TaskStatus.SUCCESS,
                    tool_result=tool_result,
                    route_decision=route_decision,
                    attempt_key=attempt_key,
                )
                self._record_token_usage(tool_result)
                self._store_success_result(task=task, tool_name=routed_tool_name, attempt_key=attempt_key, tool_result=tool_result)
            except Exception as exc:
                raise ExecutorCrashError("executor internal error", original_exception=exc) from exc

            await self.queue.mark_task_success(task.task_id)
            result = self._build_execution_result(
                task=task,
                attempt_key=attempt_key,
                routed_tool=routed_tool_name,
                success=True,
                final_status=TaskStatus.SUCCESS,
                retry_scheduled=False,
                error_message=None,
                route_decision=route_decision,
                tool_result=tool_result,
                started_at=started_at,
            )
            self._store_execution_result(result)
            await self._safe_emit_success_result(result)
            await self._safe_save_success_checkpoint(
                task=task,
                metadata={"task_id": task.task_id, "status": TaskStatus.SUCCESS.value},
            )
            return result
        except asyncio.TimeoutError as exc:
            return await self._handle_failure(
                task=task,
                attempt_key=attempt_key,
                routed_tool_name=routed_tool_name,
                route_decision=route_decision,
                tool_result=tool_result,
                failure_exception=ToolTimeoutError("task execution exceeded timeout", original_exception=exc),
                error_message=f"task execution exceeded timeout ({timeout_seconds or task.timeout}s)",
                requested_status=TaskStatus.TIMEOUT,
                started_at=started_at,
                timeout_seconds=timeout_seconds or task.timeout,
            )
        except TaskRouterError as exc:
            return await self._handle_failure(
                task=task,
                attempt_key=attempt_key,
                routed_tool_name=routed_tool_name,
                route_decision=route_decision,
                tool_result=tool_result,
                failure_exception=NonRetryableToolError(str(exc), original_exception=exc),
                error_message=str(exc),
                requested_status=TaskStatus.FAILED,
                started_at=started_at,
                timeout_seconds=timeout_seconds,
            )
        except ExecutorCrashError as exc:
            return await self._handle_failure(
                task=task,
                attempt_key=attempt_key,
                routed_tool_name=routed_tool_name,
                route_decision=route_decision,
                tool_result=tool_result,
                failure_exception=exc,
                error_message=str(exc),
                requested_status=TaskStatus.FAILED,
                started_at=started_at,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            return await self._handle_failure(
                task=task,
                attempt_key=attempt_key,
                routed_tool_name=routed_tool_name,
                route_decision=route_decision,
                tool_result=tool_result,
                failure_exception=exc,
                error_message=str(exc),
                requested_status=TaskStatus.FAILED,
                started_at=started_at,
                timeout_seconds=timeout_seconds,
            )

    async def _try_restore_from_idempotency(self, task: TaskModel) -> TaskExecutionResult | None:
        record = self.context.get_idempotency_record(task.idempotency_key)
        if record is None or not record.success or record.final_status != TaskStatus.SUCCESS or record.output is None:
            return None

        self.context.set_task_result(task.output_key, record.output)
        await self.queue.mark_task_success(task.task_id)

        result = TaskExecutionResult(
            task_id=task.task_id,
            output_key=task.output_key,
            idempotency_key=task.idempotency_key,
            attempt_key=record.attempt_key,
            success=True,
            final_status=TaskStatus.SUCCESS,
            attempt=task.retry_count,
            routed_tool=record.tool_name or task.tool,
            retry_scheduled=False,
            error_message=None,
            route_decision=None,
            tool_result=None,
            started_at=utc_now(),
            finished_at=utc_now(),
        )
        self._store_execution_result(result)
        await self._emit_execution_result(result)
        return result

    async def _handle_failure(
        self,
        *,
        task: TaskModel,
        attempt_key: str,
        routed_tool_name: str,
        route_decision: ToolRouteDecision | None,
        tool_result: ToolResult | None,
        failure_exception: Exception | None,
        error_message: str,
        requested_status: TaskStatus,
        started_at,
        timeout_seconds: int | None,
    ) -> TaskExecutionResult:
        classification = self.retry_policy.classify_failure(
            task=task,
            error_message=error_message,
            tool_result=tool_result,
            failure_exception=failure_exception,
            requested_status=requested_status,
        )

        final_status = classification.final_status
        retry_scheduled = classification.should_retry and final_status == TaskStatus.RETRY
        finished_at = utc_now()
        latency_ms = max(0, int((finished_at - started_at).total_seconds() * 1000))

        self._record_tool_call(
            task=task,
            tool_name=routed_tool_name,
            status=final_status,
            tool_result=tool_result,
            route_decision=route_decision,
            attempt_key=attempt_key,
            error_message=error_message,
            started_at=started_at,
            finished_at=finished_at,
            latency_ms=latency_ms,
        )

        if retry_scheduled:
            await self.queue.mark_task_for_retry(task.task_id)
        else:
            await self.queue.mark_task_failed(task.task_id, final_status=final_status, message=error_message)

        task = self._get_task(task.task_id)
        result = TaskExecutionResult(
            task_id=task.task_id,
            output_key=task.output_key,
            idempotency_key=task.idempotency_key,
            attempt_key=attempt_key,
            success=False,
            final_status=final_status,
            attempt=task.retry_count,
            routed_tool=routed_tool_name,
            retry_scheduled=retry_scheduled,
            error_message=error_message,
            route_decision=route_decision,
            tool_result=tool_result,
            started_at=started_at,
            finished_at=finished_at,
        )
        self._store_execution_result(result)
        self._record_failure_metadata(
            task=task,
            attempt_key=attempt_key,
            status=final_status,
            error_type=classification.error_type,
            error_message=error_message,
            latency_ms=latency_ms,
            timeout_seconds=timeout_seconds,
        )
        await self._emit_execution_result(result)
        await self._save_checkpoint(
            "retry" if retry_scheduled else ("timeout" if final_status == TaskStatus.TIMEOUT else "error"),
            {
                "task_id": task.task_id,
                "status": final_status.value,
                "retry_count": task.retry_count,
                "error_type": classification.error_type,
            },
        )
        return result

    async def _handle_unexpected_batch_exception(self, *, task_id: str, exc: Exception) -> TaskExecutionResult:
        task = self.context.tasks.get(task_id)
        if task is None:
            started_at = utc_now()
            finished_at = utc_now()
            return TaskExecutionResult(
                task_id=task_id,
                output_key="unknown",
                idempotency_key=None,
                attempt_key=None,
                success=False,
                final_status=TaskStatus.FAILED,
                attempt=0,
                routed_tool=None,
                retry_scheduled=False,
                error_message=str(exc),
                route_decision=None,
                tool_result=None,
                started_at=started_at,
                finished_at=finished_at,
            )

        error = ExecutorCrashError("executor internal error", original_exception=exc)
        if task.status in {TaskStatus.PENDING, TaskStatus.QUEUED, TaskStatus.RUNNING, TaskStatus.RETRY}:
            started_at = utc_now()
            return await self._handle_failure(
                task=task,
                attempt_key=self._build_attempt_key(task),
                routed_tool_name=task.tool,
                route_decision=None,
                tool_result=None,
                failure_exception=error,
                error_message=str(error),
                requested_status=TaskStatus.FAILED,
                started_at=started_at,
                timeout_seconds=None,
            )

        finished_at = utc_now()
        result = TaskExecutionResult(
            task_id=task.task_id,
            output_key=task.output_key,
            idempotency_key=task.idempotency_key,
            attempt_key=self._build_attempt_key(task),
            success=False,
            final_status=TaskStatus.FAILED,
            attempt=task.retry_count,
            routed_tool=task.tool,
            retry_scheduled=False,
            error_message=str(error),
            route_decision=None,
            tool_result=None,
            started_at=finished_at,
            finished_at=finished_at,
        )
        self._store_execution_result(result)
        self._record_failure_metadata(
            task=task,
            attempt_key=result.attempt_key,
            status=TaskStatus.FAILED,
            error_type="EXECUTOR_CRASH",
            error_message=str(error),
            latency_ms=0,
            timeout_seconds=None,
        )
        return result

    async def _emit_execution_result(self, result: TaskExecutionResult) -> None:
        if self.result_callback is not None:
            await self.result_callback(result)

    async def _safe_emit_success_result(self, result: TaskExecutionResult) -> None:
        try:
            await self._emit_execution_result(result)
        except Exception as exc:
            self._record_nonfatal_executor_error(
                error_code="post_success_hook_error",
                message=f"post-success result callback error: {exc}",
                task_id=result.task_id,
                details={
                    "hook": "result_callback",
                    "exception_type": type(exc).__name__,
                    "final_status": result.final_status.value,
                },
            )

    async def _safe_save_success_checkpoint(self, *, task: TaskModel, metadata: dict[str, object]) -> None:
        try:
            await self._save_checkpoint("success", metadata)
        except Exception as exc:
            self._record_nonfatal_executor_error(
                error_code="checkpoint_error",
                message=f"checkpoint save error: {exc}",
                task_id=task.task_id,
                details={
                    "hook": "checkpoint_saver",
                    "exception_type": type(exc).__name__,
                    "status": TaskStatus.SUCCESS.value,
                },
            )

    def _record_nonfatal_executor_error(
        self,
        *,
        error_code: str,
        message: str,
        task_id: str | None,
        details: dict[str, object] | None = None,
    ) -> None:
        payload = {"error_code": error_code, **(details or {})}
        self.context.add_error(
            ErrorRecord(
                source="task_executor",
                task_id=task_id,
                message=message,
                details=payload,
            )
        )
        self.context.append_shared_list(
            "executor_nonfatal_errors",
            {
                "error_code": error_code,
                "task_id": task_id,
                "message": message,
                "details": payload,
                "timestamp": utc_now().isoformat(),
            },
        )

    def _store_success_result(
        self,
        *,
        task: TaskModel,
        tool_name: str,
        attempt_key: str,
        tool_result: ToolResult,
    ) -> None:
        self.context.set_task_result(task.output_key, tool_result.output)
        if task.idempotency_key:
            self.context.set_idempotency_record(
                IdempotencyRecord(
                    idempotency_key=task.idempotency_key,
                    task_id=task.task_id,
                    output_key=task.output_key,
                    tool_name=tool_name,
                    attempt_key=attempt_key,
                    final_status=TaskStatus.SUCCESS,
                    success=True,
                    irreversible=task.irreversible,
                    output=tool_result.output,
                )
            )

    def _store_execution_result(self, result: TaskExecutionResult) -> None:
        payload = result.model_dump(mode="json")
        self.context.set_shared_mapping_value("task_execution_results", result.task_id, payload, allow_overwrite=True)
        self.context.append_shared_list("task_execution_results_history", payload)

    def _record_tool_call(
        self,
        *,
        task: TaskModel,
        tool_name: str,
        status: TaskStatus,
        tool_result: ToolResult | None,
        route_decision: ToolRouteDecision | None,
        attempt_key: str | None,
        error_message: str | None = None,
        started_at=None,
        finished_at=None,
        latency_ms: int | None = None,
    ) -> None:
        metadata: dict[str, Any] = {
            "input": self._build_tool_payload(task),
            "route_decision": route_decision.model_dump(mode="json") if route_decision is not None else None,
            "attempt_key": attempt_key,
        }
        if tool_result is not None:
            metadata.update(tool_result.metadata)
            metadata["output"] = tool_result.output
            if tool_result.error is not None:
                metadata["error"] = tool_result.error
        elif error_message is not None:
            metadata["error"] = error_message

        self.context.add_tool_call(
            ToolCallRecord(
                tool_name=tool_name,
                task_id=task.task_id,
                idempotency_key=task.idempotency_key,
                status=status.value,
                started_at=tool_result.started_at if tool_result is not None else (started_at or utc_now()),
                finished_at=tool_result.finished_at if tool_result is not None else finished_at,
                latency_ms=tool_result.latency_ms if tool_result is not None else latency_ms,
                metadata=metadata,
            )
        )

    def _record_token_usage(self, tool_result: ToolResult) -> None:
        usage_payload = tool_result.metadata.get("usage")
        if usage_payload is None:
            return
        usage = usage_payload if isinstance(usage_payload, TokenUsage) else TokenUsage.model_validate(usage_payload)
        self.context.add_token_usage(usage)

    def _record_failure_metadata(
        self,
        *,
        task: TaskModel,
        attempt_key: str | None,
        status: TaskStatus,
        error_type: str,
        error_message: str,
        latency_ms: int,
        timeout_seconds: int | None,
    ) -> None:
        if self.failure_recorder is None:
            return
        record: dict[str, object] = {
            "task_id": task.task_id,
            "tool": task.tool,
            "status": status.value,
            "error_type": error_type,
            "error_message": error_message,
            "retry_count": task.retry_count,
            "max_retry": task.max_retry,
            "latency_ms": latency_ms,
            "timestamp": utc_now().isoformat(),
            "attempt_key": attempt_key,
            "idempotency_key": task.idempotency_key,
        }
        if timeout_seconds is not None:
            record["timeout_seconds"] = timeout_seconds
            record["elapsed_ms"] = latency_ms
        try:
            self.failure_recorder(record)
        except Exception as exc:
            self.context.add_error(
                ErrorRecord(
                    source="task_executor",
                    task_id=task.task_id,
                    message=f"failure recorder error: {exc}",
                    details={"original_error_type": error_type},
                )
            )

    def _build_execution_result(
        self,
        *,
        task: TaskModel,
        attempt_key: str | None,
        routed_tool: str | None,
        success: bool,
        final_status: TaskStatus,
        retry_scheduled: bool,
        error_message: str | None,
        route_decision: ToolRouteDecision | None,
        tool_result: ToolResult | None,
        started_at,
    ) -> TaskExecutionResult:
        return TaskExecutionResult(
            task_id=task.task_id,
            output_key=task.output_key,
            idempotency_key=task.idempotency_key,
            attempt_key=attempt_key,
            success=success,
            final_status=final_status,
            attempt=task.retry_count,
            routed_tool=routed_tool,
            retry_scheduled=retry_scheduled,
            error_message=error_message,
            route_decision=route_decision,
            tool_result=tool_result,
            started_at=started_at,
            finished_at=utc_now(),
        )

    def _build_tool_payload(self, task: TaskModel) -> dict[str, Any]:
        payload = dict(task.input)
        dependency_outputs = self._collect_dependency_outputs(task)
        if dependency_outputs:
            existing_context = str(payload.get("context", "")).strip()
            if existing_context and self._context_references_dependency(existing_context, dependency_outputs):
                return payload

            dependency_context = self._format_dependency_context(dependency_outputs)
            payload["context"] = (
                f"{existing_context}\n\nDependency Context:\n{dependency_context}"
                if existing_context
                else dependency_context
            )
        return payload

    def _context_references_dependency(self, context_template: str, dependency_outputs: list[dict[str, Any]]) -> bool:
        output_keys = {
            str(item.get("output_key") or "").strip()
            for item in dependency_outputs
            if str(item.get("output_key") or "").strip()
        }
        if not output_keys:
            return False

        for match in _TEMPLATE_VARIABLE_PATTERN.finditer(context_template):
            path = match.group(1).strip()
            for output_key in output_keys:
                if (
                    path == output_key
                    or path.startswith(f"{output_key}.")
                    or path == f"task_results.{output_key}"
                    or path.startswith(f"task_results.{output_key}.")
                ):
                    return True
        return False

    def _collect_dependency_outputs(self, task: TaskModel) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        for dependency_id in task.depends_on:
            dependency_task = self.context.tasks.get(dependency_id)
            if dependency_task is None:
                continue
            collected.append(
                {
                    "task_id": dependency_id,
                    "output_key": dependency_task.output_key,
                    "output": self.context.task_results.get(dependency_task.output_key),
                }
            )
        return collected

    def _format_dependency_context(self, dependency_outputs: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for item in dependency_outputs:
            lines.append(
                f"- {item['task_id']} ({item['output_key']}): {self._preview_dependency_output(item.get('output'))}"
            )
        return "\n".join(lines)

    def _preview_dependency_output(self, value: Any) -> str:
        if value is None:
            return "None"
        if isinstance(value, dict):
            preferred = value.get("text") or value.get("summary")
            if preferred is not None:
                return str(preferred)
        return str(value)

    def _build_attempt_key(self, task: TaskModel) -> str:
        idempotency_key = task.idempotency_key or f"{self.context.runtime.request_id}:{task.task_id}:{task.output_key}"
        return f"{idempotency_key}:attempt:{task.retry_count + 1}"

    def _resolve_execution_timeout(self, *, task: TaskModel, tool_timeout: int, payload: dict[str, object]) -> int:
        settings = self._resolve_timeout_settings(task=task, tool_timeout=tool_timeout, payload=payload)
        return int(settings["executor_timeout_seconds"])

    def _resolve_timeout_settings(
        self,
        *,
        task: TaskModel,
        tool_timeout: int,
        payload: dict[str, object],
    ) -> dict[str, int]:
        explicit_timeout = self._coerce_positive_int(payload.get("timeout_seconds"))
        explicit_tool_timeout = self._coerce_positive_int(payload.get("tool_timeout_seconds"))
        explicit_executor_timeout = self._coerce_positive_int(payload.get("executor_timeout_seconds"))

        if task.tool not in {FAIL_FAST_TOOL_NAME, REASON_TOOL_NAME}:
            baseline_timeout = explicit_timeout or explicit_tool_timeout or explicit_executor_timeout
            if baseline_timeout <= 0:
                baseline_timeout = int(task.timeout if task.timeout > 0 and task.timeout not in {60, 120} else tool_timeout)
            return {
                "timeout_seconds": baseline_timeout,
                "tool_timeout_seconds": baseline_timeout,
                "executor_timeout_seconds": baseline_timeout,
            }

        llm_timeout_default = (
            FAIL_FAST_LLM_TIMEOUT_SECONDS if task.tool == FAIL_FAST_TOOL_NAME else REASON_LLM_TIMEOUT_SECONDS
        )
        tool_timeout_default = (
            FAIL_FAST_TOOL_TIMEOUT_SECONDS if task.tool == FAIL_FAST_TOOL_NAME else REASON_TOOL_TIMEOUT_SECONDS
        )
        executor_timeout_default = (
            FAIL_FAST_EXECUTOR_TIMEOUT_SECONDS
            if task.tool == FAIL_FAST_TOOL_NAME
            else REASON_EXECUTOR_TIMEOUT_SECONDS
        )

        llm_timeout = explicit_timeout or llm_timeout_default
        resolved_tool_timeout = explicit_tool_timeout or tool_timeout or tool_timeout_default
        resolved_executor_timeout = explicit_executor_timeout or max(
            executor_timeout_default,
            resolved_tool_timeout + 15,
            task.timeout if task.timeout > 0 and task.timeout not in {60, 120} else 0,
        )
        resolved_tool_timeout = max(resolved_tool_timeout, llm_timeout + 1)
        resolved_executor_timeout = max(resolved_executor_timeout, resolved_tool_timeout + 1, llm_timeout + 1)

        return {
            "timeout_seconds": llm_timeout,
            "tool_timeout_seconds": resolved_tool_timeout,
            "executor_timeout_seconds": resolved_executor_timeout,
        }

    def _coerce_positive_int(self, value: object) -> int:
        try:
            parsed = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0
        return parsed if parsed > 0 else 0

    def _get_task(self, task_id: str) -> TaskModel:
        task = self.context.tasks.get(task_id)
        if task is None:
            raise self._error(
                code="task_not_found",
                message=f"task '{task_id}' not found",
                details={"task_id": task_id},
            )
        return task

    def _error(self, *, code: str, message: str, details: dict[str, object] | None = None) -> TaskExecutorError:
        runtime_log(
            layer="executor",
            event="error",
            data={"code": code, "message": message, "details": details or {}},
            logger=self.logger,
        )
        return TaskExecutorError(ExecutorErrorDetail(code=code, message=message, details=details or {}))

    async def _save_checkpoint(self, event: str, metadata: dict[str, object] | None = None) -> None:
        if self.checkpoint_saver is None:
            return
        await self.checkpoint_saver(event, metadata)
