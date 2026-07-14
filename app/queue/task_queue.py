"""DAG 感知的异步任务调度队列。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable

from app.executor.state_guard import validate_transition
from app.schemas.context import AgentState, ContextStore, ErrorRecord, ExecutionRecord
from app.schemas.queue import QueueErrorDetail, QueueSnapshot
from app.schemas.task import TaskModel, TaskStatus
from app.utils import configure_runtime_logger, runtime_log


class TaskQueueError(Exception):
    """当任务队列无法安全完成调度操作时抛出的异常。"""

    def __init__(self, error: QueueErrorDetail) -> None:
        super().__init__(error.message)
        self.error = error


class TaskQueue:
    """具备 DAG 依赖感知、死锁保护和检查点回调的异步任务队列。"""

    def __init__(
        self,
        *,
        context: ContextStore,
        state: AgentState | None = None,
        max_concurrency: int = 4,
        context_key: str = "task_queue",
        checkpoint_saver: Callable[[str, dict[str, object] | None], Awaitable[None]] | None = None,
    ) -> None:
        if max_concurrency <= 0:
            raise ValueError("max_concurrency 必须大于 0。")

        self.context = context
        self.state = state
        self.max_concurrency = max_concurrency
        self.context_key = context_key
        self.checkpoint_saver = checkpoint_saver
        self._lock = asyncio.Lock()
        self._task_order: dict[str, int] = {}
        self._dependents: dict[str, set[str]] = {}
        self.logger = configure_runtime_logger()

    async def initialize(self, tasks: Iterable[TaskModel]) -> QueueSnapshot:
        """初始化任务队列并校验任务图。"""
        async with self._lock:
            task_list = list(tasks)
            if not task_list:
                raise self._error(code="empty_task_list", message="任务队列初始化时不能为空。")

            self._validate_initial_tasks(task_list)
            self._task_order = {task.task_id: index for index, task in enumerate(task_list)}
            self._dependents = {task.task_id: set() for task in task_list}

            for task in task_list:
                self.context.ensure_task_runtime_values(task)
                self.context.register_task(task)
                for dependency in task.depends_on:
                    self._dependents[dependency].add(task.task_id)
                if self.state is not None:
                    self.state.sync_task_status(task)

            self._validate_dependency_graph(task_list)
            runtime_log(
                layer="queue",
                event="success",
                data={"task_count": len(task_list), "max_concurrency": self.max_concurrency},
                logger=self.logger,
            )
            snapshot = self._write_snapshot()

        await self._save_checkpoint(
            "success",
            {"task_count": snapshot.total_tasks, "ready_task_ids": list(snapshot.ready_task_ids)},
        )
        return snapshot

    async def hydrate(self, tasks: Iterable[TaskModel]) -> QueueSnapshot:
        """从已有任务状态恢复队列，而不重置任务生命周期。"""
        async with self._lock:
            task_list = list(tasks)
            if not task_list:
                raise self._error(code="empty_task_list", message="队列恢复时缺少任务定义。")

            self._task_order = {task.task_id: index for index, task in enumerate(task_list)}
            self._dependents = {task.task_id: set() for task in task_list}

            for task in task_list:
                self.context.ensure_task_runtime_values(task)
                self.context.tasks[task.task_id] = task
                for dependency in task.depends_on:
                    self._dependents[dependency].add(task.task_id)
                if self.state is not None:
                    self.state.sync_task_status(task)

            self._validate_dependency_graph(task_list)
            snapshot = self._write_snapshot()

        await self._save_checkpoint(
            "success",
            {
                "mode": "hydrate",
                "task_count": snapshot.total_tasks,
                "ready_task_ids": list(snapshot.ready_task_ids),
            },
        )
        return snapshot

    async def get_ready_tasks(self, limit: int | None = None) -> list[TaskModel]:
        """筛选已满足依赖的可执行任务，并推进到 QUEUED。"""
        async with self._lock:
            available_slots = self._available_slots()
            if available_slots <= 0:
                snapshot = self._write_snapshot()
                self._ensure_not_deadlocked(snapshot)
                return []

            requested_limit = available_slots if limit is None else min(limit, available_slots)
            if requested_limit <= 0:
                snapshot = self._write_snapshot()
                self._ensure_not_deadlocked(snapshot)
                return []

            candidates = [
                task
                for task in self.context.tasks.values()
                if task.status in {TaskStatus.PENDING, TaskStatus.RETRY} and self._dependencies_satisfied(task)
            ]
            candidates.sort(key=self._sort_key)
            selected = candidates[:requested_limit]

            for task in selected:
                self._transition_task_status(task, TaskStatus.QUEUED)
                self.context.add_execution_record(
                    ExecutionRecord(
                        task_id=task.task_id,
                        idempotency_key=task.idempotency_key,
                        status=TaskStatus.QUEUED,
                        attempt=task.retry_count,
                        message="任务已进入待执行队列。",
                    )
                )
                if self.state is not None:
                    self.state.sync_task_status(task)
                runtime_log(
                    layer="task",
                    event="execute",
                    data={
                        "task_id": task.task_id,
                        "status": TaskStatus.QUEUED.value,
                        "depends_on": list(task.depends_on),
                        "idempotency_key": task.idempotency_key,
                    },
                    logger=self.logger,
                )

            snapshot = self._write_snapshot()
            self._ensure_not_deadlocked(snapshot)
            runtime_log(
                layer="queue",
                event="execute",
                data={
                    "ready_tasks": [task.task_id for task in selected],
                    "blocked_tasks": list(snapshot.blocked_task_ids),
                    "available_slots": snapshot.available_slots,
                },
                logger=self.logger,
            )

        await self._save_checkpoint(
            "execute",
            {
                "ready_task_ids": [task.task_id for task in selected],
                "blocked_task_ids": list(snapshot.blocked_task_ids),
            },
        )
        return selected

    async def mark_task_running(self, task_id: str) -> TaskModel:
        """把一个已入队任务标记为 RUNNING。"""
        async with self._lock:
            task = self._get_task(task_id)
            if task.status != TaskStatus.QUEUED:
                raise self._error(
                    code="invalid_running_transition",
                    message=f"任务 '{task_id}' 必须先进入 QUEUED 才能转为 RUNNING。",
                    details={"task_id": task_id, "status": task.status},
                )

            self._transition_task_status(task, TaskStatus.RUNNING)
            self.context.add_execution_record(
                ExecutionRecord(
                    task_id=task.task_id,
                    idempotency_key=task.idempotency_key,
                    status=TaskStatus.RUNNING,
                    attempt=task.retry_count,
                    message="任务开始执行。",
                )
            )
            if self.state is not None:
                self.state.current_task_id = task_id
                self.state.sync_task_status(task)

            self._write_snapshot()
            runtime_log(
                layer="task",
                event="execute",
                data={
                    "task_id": task.task_id,
                    "status": TaskStatus.RUNNING.value,
                    "depends_on": list(task.depends_on),
                    "idempotency_key": task.idempotency_key,
                },
                logger=self.logger,
            )

        await self._save_checkpoint("execute", {"task_id": task.task_id, "status": TaskStatus.RUNNING.value})
        return task

    async def mark_task_success(self, task_id: str) -> TaskModel:
        """把一个执行中的任务标记为 SUCCESS。"""
        async with self._lock:
            task = self._get_task(task_id)
            if task.status not in {TaskStatus.RUNNING, TaskStatus.QUEUED}:
                raise self._error(
                    code="invalid_success_transition",
                    message=f"任务 '{task_id}' 当前状态不允许转为 SUCCESS。",
                    details={"task_id": task_id, "status": task.status},
                )

            self._transition_task_status(task, TaskStatus.SUCCESS)
            self.context.add_execution_record(
                ExecutionRecord(
                    task_id=task.task_id,
                    idempotency_key=task.idempotency_key,
                    status=TaskStatus.SUCCESS,
                    attempt=task.retry_count,
                    message="任务执行成功。",
                )
            )
            if self.state is not None:
                if self.state.current_task_id == task_id:
                    self.state.current_task_id = None
                self.state.sync_task_status(task)

            self._write_snapshot()
            runtime_log(
                layer="task",
                event="success",
                data={
                    "task_id": task.task_id,
                    "status": TaskStatus.SUCCESS.value,
                    "depends_on": list(task.depends_on),
                    "idempotency_key": task.idempotency_key,
                },
                logger=self.logger,
            )

        await self._save_checkpoint("success", {"task_id": task.task_id, "status": TaskStatus.SUCCESS.value})
        return task

    async def mark_task_failed(
        self,
        task_id: str,
        *,
        final_status: TaskStatus = TaskStatus.FAILED,
        message: str | None = None,
    ) -> TaskModel:
        """把任务标记为 FAILED、TIMEOUT 或 CANCELLED 等终态失败。"""
        if final_status not in {TaskStatus.FAILED, TaskStatus.TIMEOUT, TaskStatus.CANCELLED}:
            raise self._error(
                code="invalid_failure_status",
                message="final_status 必须是 FAILED、TIMEOUT 或 CANCELLED。",
                details={"final_status": final_status},
            )

        async with self._lock:
            task = self._get_task(task_id)
            if task.status not in {TaskStatus.RUNNING, TaskStatus.QUEUED, TaskStatus.PENDING, TaskStatus.RETRY}:
                raise self._error(
                    code="invalid_failure_transition",
                    message=f"任务 '{task_id}' 当前状态不允许转入终态失败。",
                    details={"task_id": task_id, "status": task.status},
                )

            self._transition_task_status(task, final_status)
            error_message = message or f"任务进入终态：{final_status.value}"
            self.context.add_execution_record(
                ExecutionRecord(
                    task_id=task.task_id,
                    idempotency_key=task.idempotency_key,
                    status=final_status,
                    attempt=task.retry_count,
                    message=error_message,
                )
            )
            self.context.add_error(
                ErrorRecord(
                    source="task_queue",
                    task_id=task.task_id,
                    message=error_message,
                    details={"final_status": final_status.value},
                )
            )
            if self.state is not None:
                if self.state.current_task_id == task_id:
                    self.state.current_task_id = None
                self.state.last_error = self.context.errors[-1]
                self.state.sync_task_status(task)

            self._cancel_blocked_dependents(
                failed_task_id=task_id,
                reason=f"依赖任务 '{task_id}' 以 {final_status.value} 结束",
            )
            self._write_snapshot()
            runtime_log(
                layer="task",
                event="timeout" if final_status == TaskStatus.TIMEOUT else "error",
                data={
                    "task_id": task.task_id,
                    "status": final_status.value,
                    "depends_on": list(task.depends_on),
                    "idempotency_key": task.idempotency_key,
                },
                logger=self.logger,
            )

        await self._save_checkpoint(
            "timeout" if final_status == TaskStatus.TIMEOUT else "error",
            {"task_id": task.task_id, "status": final_status.value, "error": error_message},
        )
        return task

    async def mark_task_for_retry(self, task_id: str) -> TaskModel:
        """增加重试次数后，把任务推进到 RETRY。"""
        async with self._lock:
            task = self._get_task(task_id)
            current_status = task.status if isinstance(task.status, TaskStatus) else TaskStatus(str(task.status))
            validate_transition(current_status, TaskStatus.RETRY)
            if False:
                raise self._error(
                    code="invalid_retry_transition",
                    message=f"任务 '{task_id}' 当前状态不允许进入重试。",
                    details={"task_id": task_id, "status": task.status},
                )
            if not task.can_retry():
                raise self._error(
                    code="retry_exhausted",
                    message=f"任务 '{task_id}' 的重试次数已耗尽。",
                    details={"task_id": task_id, "retry_count": task.retry_count, "max_retry": task.max_retry},
                )

            task.mark_retry()
            task.status = TaskStatus.RETRY
            self.context.add_execution_record(
                ExecutionRecord(
                    task_id=task.task_id,
                    idempotency_key=task.idempotency_key,
                    status=TaskStatus.RETRY,
                    attempt=task.retry_count,
                    message="任务进入重试队列。",
                )
            )
            if self.state is not None:
                if self.state.current_task_id == task_id:
                    self.state.current_task_id = None
                self.state.sync_task_status(task)

            self._write_snapshot()
            runtime_log(
                layer="task",
                event="retry",
                data={
                    "task_id": task.task_id,
                    "status": TaskStatus.RETRY.value,
                    "retry_count": task.retry_count,
                    "max_retry": task.max_retry,
                    "depends_on": list(task.depends_on),
                    "idempotency_key": task.idempotency_key,
                },
                logger=self.logger,
            )

        await self._save_checkpoint(
            "retry",
            {"task_id": task.task_id, "status": TaskStatus.RETRY.value, "retry_count": task.retry_count},
        )
        return task

    async def get_snapshot(self) -> QueueSnapshot:
        """在不修改任务状态的前提下返回当前快照。"""
        async with self._lock:
            return self._write_snapshot()

    async def is_complete(self) -> bool:
        """当所有任务都进入终态时返回 True。"""
        async with self._lock:
            non_terminal = {
                TaskStatus.PENDING,
                TaskStatus.QUEUED,
                TaskStatus.RUNNING,
                TaskStatus.RETRY,
            }
            is_complete = all(task.status not in non_terminal for task in self.context.tasks.values())
            if self.state is not None:
                self.state.final_output_ready = is_complete and not self._has_terminal_failures()
            snapshot = self._write_snapshot()
            self._ensure_not_deadlocked(snapshot)
            return is_complete

    async def has_terminal_failures(self) -> bool:
        """判断当前是否存在终态失败任务。"""
        async with self._lock:
            return self._has_terminal_failures()

    def _validate_initial_tasks(self, tasks: list[TaskModel]) -> None:
        """校验初始化任务列表是否合法。"""
        task_ids = [task.task_id for task in tasks]
        if len(task_ids) != len(set(task_ids)):
            raise self._error(code="duplicate_task_id", message="任务列表中存在重复 task_id。")

        for task in tasks:
            if task.status not in {TaskStatus.PENDING, TaskStatus.RETRY}:
                raise self._error(
                    code="invalid_initial_task_status",
                    message=f"任务 '{task.task_id}' 进入队列时必须是 PENDING 或 RETRY。",
                    details={"task_id": task.task_id, "status": task.status},
                )

    def _validate_dependency_graph(self, tasks: list[TaskModel]) -> None:
        """校验依赖存在性、孤儿任务和环路。"""
        task_ids = {task.task_id for task in tasks}
        for task in tasks:
            missing_dependencies = [dependency for dependency in task.depends_on if dependency not in task_ids]
            if missing_dependencies:
                raise self._error(
                    code="missing_dependency",
                    message=f"任务 '{task.task_id}' 依赖了未定义任务。",
                    details={"task_id": task.task_id, "missing_dependencies": missing_dependencies},
                )

        orphan_task_ids = self._find_orphan_tasks(tasks)
        if orphan_task_ids:
            raise self._error(
                code="orphan_task_detected",
                message="任务图中存在孤儿任务。",
                details={"orphan_task_ids": orphan_task_ids},
            )

        cycle_path = self._find_cycle(tasks)
        if cycle_path:
            raise self._error(
                code="cyclic_dependency",
                message="任务依赖图中存在循环依赖。",
                details={"cycle_path": cycle_path},
            )

    def _find_cycle(self, tasks: list[TaskModel]) -> list[str] | None:
        """深度优先搜索检测依赖环。"""
        dependency_map = {task.task_id: task.depends_on for task in tasks}
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

    def _find_orphan_tasks(self, tasks: list[TaskModel]) -> list[str]:
        """检测无法从任意根节点到达的任务。"""
        if not tasks:
            return []

        indegree = {task.task_id: len(task.depends_on) for task in tasks}
        dependents: dict[str, list[str]] = {task.task_id: [] for task in tasks}
        for task in tasks:
            for dependency in task.depends_on:
                dependents[dependency].append(task.task_id)

        roots = [task_id for task_id, degree in indegree.items() if degree == 0]
        if not roots:
            return []

        visited: set[str] = set()
        stack = list(roots)
        while stack:
            task_id = stack.pop()
            if task_id in visited:
                continue
            visited.add(task_id)
            stack.extend(dependents[task_id])

        return sorted([task.task_id for task in tasks if task.task_id not in visited])

    def _dependencies_satisfied(self, task: TaskModel) -> bool:
        """判断任务依赖是否全部成功。"""
        return all(self.context.tasks[dependency].status == TaskStatus.SUCCESS for dependency in task.depends_on)

    def _sort_key(self, task: TaskModel) -> tuple[int, object, int]:
        """队列排序规则。"""
        return (task.priority, task.created_at, self._task_order[task.task_id])

    def _available_slots(self) -> int:
        """计算当前剩余并发槽位。"""
        active_count = sum(
            1
            for task in self.context.tasks.values()
            if task.status in {TaskStatus.QUEUED, TaskStatus.RUNNING}
        )
        return max(0, self.max_concurrency - active_count)

    def _has_terminal_failures(self) -> bool:
        """判断是否存在终态失败任务。"""
        return any(
            task.status in {TaskStatus.FAILED, TaskStatus.TIMEOUT, TaskStatus.CANCELLED}
            for task in self.context.tasks.values()
        )

    def _get_task(self, task_id: str) -> TaskModel:
        """按 ID 读取任务。"""
        task = self.context.tasks.get(task_id)
        if task is None:
            raise self._error(
                code="task_not_found",
                message=f"任务 '{task_id}' 未注册到队列中。",
                details={"task_id": task_id},
            )
        return task

    def _build_snapshot(self) -> QueueSnapshot:
        """构造当前队列快照。"""
        ready_task_ids = [
            task.task_id
            for task in sorted(self.context.tasks.values(), key=self._sort_key)
            if task.status in {TaskStatus.PENDING, TaskStatus.RETRY} and self._dependencies_satisfied(task)
        ]
        blocked_task_ids = [
            task.task_id
            for task in sorted(self.context.tasks.values(), key=self._sort_key)
            if task.status in {TaskStatus.PENDING, TaskStatus.RETRY} and not self._dependencies_satisfied(task)
        ]
        queued_task_ids = [task.task_id for task in self.context.tasks.values() if task.status == TaskStatus.QUEUED]
        running_task_ids = [task.task_id for task in self.context.tasks.values() if task.status == TaskStatus.RUNNING]
        retry_task_ids = [task.task_id for task in self.context.tasks.values() if task.status == TaskStatus.RETRY]
        completed_task_ids = [task.task_id for task in self.context.tasks.values() if task.status == TaskStatus.SUCCESS]
        failed_task_ids = [
            task.task_id
            for task in self.context.tasks.values()
            if task.status in {TaskStatus.FAILED, TaskStatus.TIMEOUT, TaskStatus.CANCELLED}
        ]

        return QueueSnapshot(
            total_tasks=len(self.context.tasks),
            max_concurrency=self.max_concurrency,
            available_slots=self._available_slots(),
            ready_task_ids=ready_task_ids,
            blocked_task_ids=blocked_task_ids,
            queued_task_ids=queued_task_ids,
            running_task_ids=running_task_ids,
            retry_task_ids=retry_task_ids,
            completed_task_ids=completed_task_ids,
            failed_task_ids=failed_task_ids,
        )

    def _cancel_blocked_dependents(self, *, failed_task_id: str, reason: str) -> None:
        """递归取消所有受失败依赖影响的下游任务。"""
        for dependent_id in sorted(self._dependents.get(failed_task_id, set()), key=self._task_order.get):
            dependent_task = self.context.tasks[dependent_id]
            if dependent_task.status in {
                TaskStatus.SUCCESS,
                TaskStatus.FAILED,
                TaskStatus.TIMEOUT,
                TaskStatus.CANCELLED,
            }:
                continue

            self._transition_task_status(dependent_task, TaskStatus.CANCELLED)
            cancel_message = f"任务被取消，原因：{reason}"
            self.context.add_execution_record(
                ExecutionRecord(
                    task_id=dependent_task.task_id,
                    idempotency_key=dependent_task.idempotency_key,
                    status=TaskStatus.CANCELLED,
                    attempt=dependent_task.retry_count,
                    message=cancel_message,
                )
            )
            self.context.add_error(
                ErrorRecord(
                    source="task_queue",
                    task_id=dependent_task.task_id,
                    message=cancel_message,
                    details={"blocked_by": failed_task_id},
                )
            )
            if self.state is not None:
                self.state.last_error = self.context.errors[-1]
                self.state.sync_task_status(dependent_task)
            runtime_log(
                layer="task",
                event="error",
                data={
                    "task_id": dependent_task.task_id,
                    "status": TaskStatus.CANCELLED.value,
                    "depends_on": list(dependent_task.depends_on),
                    "blocked_by": failed_task_id,
                    "idempotency_key": dependent_task.idempotency_key,
                },
                logger=self.logger,
            )

            self._cancel_blocked_dependents(
                failed_task_id=dependent_task.task_id,
                reason=f"上游任务 '{dependent_task.task_id}' 已被取消",
            )

    def _write_snapshot(self) -> QueueSnapshot:
        """写回队列快照到共享上下文。"""
        snapshot = self._build_snapshot()
        self.context.set_shared_value(self.context_key, snapshot.model_dump(mode="json"))
        return snapshot

    def _transition_task_status(self, task: TaskModel, new_status: TaskStatus) -> None:
        current_status = task.status if isinstance(task.status, TaskStatus) else TaskStatus(str(task.status))
        validate_transition(current_status, new_status)
        task.status = new_status

    def _ensure_not_deadlocked(self, snapshot: QueueSnapshot) -> None:
        """检测永久阻塞的 DAG 状态。"""
        has_active_tasks = bool(snapshot.queued_task_ids or snapshot.running_task_ids)
        has_ready_tasks = bool(snapshot.ready_task_ids)
        has_blocked_tasks = bool(snapshot.blocked_task_ids or snapshot.retry_task_ids)
        if not has_ready_tasks and not has_active_tasks and has_blocked_tasks:
            blocked_reasons = {
                task_id: [
                    dependency
                    for dependency in self.context.tasks[task_id].depends_on
                    if self.context.tasks[dependency].status != TaskStatus.SUCCESS
                ]
                for task_id in snapshot.blocked_task_ids
            }
            raise self._error(
                code="permanently_blocked_dag",
                message="DAG 已永久阻塞，当前没有可执行任务。",
                details={"blocked_reasons": blocked_reasons, "snapshot": snapshot.model_dump(mode="json")},
            )

    def _error(self, *, code: str, message: str, details: dict[str, object] | None = None) -> TaskQueueError:
        """构造统一队列错误，并输出结构化日志。"""
        payload = details or {}
        runtime_log(
            layer="queue",
            event="error",
            data={"code": code, "message": message, "details": payload},
            logger=self.logger,
        )
        return TaskQueueError(QueueErrorDetail(code=code, message=message, details=payload))

    async def _save_checkpoint(self, event: str, metadata: dict[str, object] | None = None) -> None:
        """在启用检查点时保存当前状态。"""
        if self.checkpoint_saver is None:
            return
        await self.checkpoint_saver(event, metadata)
