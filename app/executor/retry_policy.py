from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx

from app.executor.exceptions import (
    ExecutorCrashError,
    NonRetryableToolError,
    RetryableToolError,
    ToolTimeoutError,
)
from app.schemas.task import TaskModel, TaskStatus
from app.schemas.tool import ToolResult
from app.tools.llm_client import (
    FAIL_FAST_TIMEOUT_MARKER,
    LLMProviderRetryableError,
    LLMProviderTimeoutError,
    RETRYABLE_ERROR_MARKER,
)

REASON_TOOL_NAME = "llm_reason_tool"
REASON_MAX_RECOVERABLE_RETRIES = 1


@dataclass(frozen=True)
class RetryPolicyResult:
    error_type: str
    retryable: bool
    should_retry: bool
    backoff_seconds: int
    reason: str
    final_status: TaskStatus


class RetryPolicy:
    def __init__(self, *, reason_max_recoverable_retries: int = REASON_MAX_RECOVERABLE_RETRIES) -> None:
        self.reason_max_recoverable_retries = max(0, reason_max_recoverable_retries)

    def classify_failure(
        self,
        *,
        task: TaskModel,
        error_message: str,
        tool_result: ToolResult | None,
        failure_exception: Exception | None,
        requested_status: TaskStatus,
    ) -> RetryPolicyResult:
        normalized_exception = self._normalize_exception(
            error_message=error_message,
            tool_result=tool_result,
            failure_exception=failure_exception,
            requested_status=requested_status,
        )

        if isinstance(normalized_exception, ToolTimeoutError):
            return RetryPolicyResult(
                error_type="TOOL_TIMEOUT",
                retryable=True,
                should_retry=False,
                backoff_seconds=0,
                reason=normalized_exception.reason or "task execution exceeded timeout policy",
                final_status=TaskStatus.TIMEOUT,
            )

        if isinstance(normalized_exception, ExecutorCrashError):
            return RetryPolicyResult(
                error_type="EXECUTOR_CRASH",
                retryable=False,
                should_retry=False,
                backoff_seconds=0,
                reason="executor internal failure",
                final_status=TaskStatus.FAILED,
            )

        if isinstance(normalized_exception, RetryableToolError):
            should_retry = self._can_retry(task)
            return RetryPolicyResult(
                error_type="RETRYABLE_TOOL_ERROR",
                retryable=True,
                should_retry=should_retry,
                backoff_seconds=self._compute_backoff_seconds(task.retry_count) if should_retry else 0,
                reason=normalized_exception.reason or ("retry budget exhausted" if not should_retry else "retryable tool failure"),
                final_status=TaskStatus.RETRY if should_retry else TaskStatus.FAILED,
            )

        return RetryPolicyResult(
            error_type="NON_RETRYABLE_TOOL_ERROR",
            retryable=False,
            should_retry=False,
            backoff_seconds=0,
            reason=getattr(normalized_exception, "reason", None) or "non-retryable tool failure",
            final_status=TaskStatus.FAILED if requested_status != TaskStatus.TIMEOUT else TaskStatus.TIMEOUT,
        )

    def _normalize_exception(
        self,
        *,
        error_message: str,
        tool_result: ToolResult | None,
        failure_exception: Exception | None,
        requested_status: TaskStatus,
    ) -> Exception:
        if requested_status == TaskStatus.TIMEOUT:
            return ToolTimeoutError(error_message, reason="task timeout requested explicitly", original_exception=failure_exception)
        if isinstance(failure_exception, (ToolTimeoutError, RetryableToolError, NonRetryableToolError, ExecutorCrashError)):
            return failure_exception
        if self._is_timeout_failure(error_message=error_message, tool_result=tool_result, failure_exception=failure_exception):
            return ToolTimeoutError(error_message, reason="timeout marker detected", original_exception=failure_exception)
        if isinstance(failure_exception, LLMProviderRetryableError):
            return RetryableToolError(error_message, reason="provider retryable error", original_exception=failure_exception)
        if isinstance(
            failure_exception,
            (httpx.HTTPStatusError, httpx.ConnectError, httpx.RemoteProtocolError, httpx.NetworkError),
        ):
            return RetryableToolError(error_message, reason="transient network/provider failure", original_exception=failure_exception)
        if tool_result is not None and bool(tool_result.metadata.get("retryable_error")):
            return RetryableToolError(error_message, reason="tool metadata marked failure retryable")
        normalized_error = str(error_message or "").lower()
        if any(
            marker in normalized_error
            for marker in (
                RETRYABLE_ERROR_MARKER,
                "transient failure",
                "temporary provider unavailable",
                "temporarily unavailable",
                "provider unavailable",
                " 429 ",
                " 500 ",
                " 502 ",
                " 503 ",
                " 504 ",
            )
        ):
            return RetryableToolError(error_message, reason="retryable error marker matched")
        if isinstance(failure_exception, PermissionError):
            return NonRetryableToolError(error_message, reason="permission denied", original_exception=failure_exception)
        if any(
            marker in normalized_error
            for marker in (
                "invalid input",
                "invalid tool input",
                "schema violation",
                "unsupported tool",
                "tool disabled",
                "permission denied",
            )
        ):
            return NonRetryableToolError(error_message, reason="non-retryable error marker matched")
        return NonRetryableToolError(error_message, reason="default non-retryable classification", original_exception=failure_exception)

    def _is_timeout_failure(
        self,
        *,
        error_message: str,
        tool_result: ToolResult | None,
        failure_exception: Exception | None,
    ) -> bool:
        if tool_result is not None and bool(
            tool_result.metadata.get("fail_fast_timeout") or tool_result.metadata.get("timeout_fail_fast")
        ):
            return True
        if isinstance(
            failure_exception,
            (
                asyncio.TimeoutError,
                httpx.ReadTimeout,
                httpx.ConnectTimeout,
                httpx.TimeoutException,
                LLMProviderTimeoutError,
            ),
        ):
            return True
        normalized_error = str(error_message or "").lower()
        return FAIL_FAST_TIMEOUT_MARKER in normalized_error or "provider timeout" in normalized_error

    def _can_retry(self, task: TaskModel) -> bool:
        if not task.can_retry():
            return False
        if task.tool == REASON_TOOL_NAME and task.retry_count >= self.reason_max_recoverable_retries:
            return False
        return True

    def _compute_backoff_seconds(self, retry_count: int) -> int:
        return min(30, max(0, 2**max(0, retry_count)))
