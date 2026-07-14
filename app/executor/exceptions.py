from __future__ import annotations

from app.schemas.task import TaskStatus


class RetryableToolError(Exception):
    def __init__(
        self,
        message: str,
        *,
        reason: str | None = None,
        error_type: str | None = None,
        original_exception: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.error_type = error_type
        self.original_exception = original_exception


class NonRetryableToolError(Exception):
    def __init__(
        self,
        message: str,
        *,
        reason: str | None = None,
        error_type: str | None = None,
        original_exception: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.error_type = error_type
        self.original_exception = original_exception


class ToolTimeoutError(RetryableToolError):
    pass


class ExecutorCrashError(Exception):
    def __init__(self, message: str, *, original_exception: Exception | None = None) -> None:
        super().__init__(message)
        self.original_exception = original_exception


class InvalidTaskStateTransitionError(Exception):
    def __init__(self, *, old_status: TaskStatus, new_status: TaskStatus) -> None:
        self.old_status = old_status
        self.new_status = new_status
        super().__init__(f"invalid task state transition: {old_status.value} -> {new_status.value}")
