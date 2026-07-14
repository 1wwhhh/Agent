from __future__ import annotations

from app.executor.exceptions import InvalidTaskStateTransitionError
from app.schemas.task import TaskStatus

ALLOWED_TASK_STATUS_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.PENDING: {TaskStatus.QUEUED, TaskStatus.CANCELLED},
    TaskStatus.QUEUED: {TaskStatus.RUNNING, TaskStatus.SUCCESS, TaskStatus.CANCELLED},
    TaskStatus.RUNNING: {
        TaskStatus.SUCCESS,
        TaskStatus.FAILED,
        TaskStatus.RETRY,
        TaskStatus.TIMEOUT,
        TaskStatus.CANCELLED,
    },
    TaskStatus.RETRY: {TaskStatus.QUEUED, TaskStatus.RUNNING},
    TaskStatus.SUCCESS: set(),
    TaskStatus.FAILED: set(),
    TaskStatus.TIMEOUT: set(),
    TaskStatus.CANCELLED: set(),
}


def validate_transition(old_status: TaskStatus, new_status: TaskStatus) -> None:
    if new_status not in ALLOWED_TASK_STATUS_TRANSITIONS.get(old_status, set()):
        raise InvalidTaskStateTransitionError(old_status=old_status, new_status=new_status)
