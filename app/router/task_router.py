"""Asynchronous task-to-tool routing with capability and permission checks."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from pydantic import ValidationError

from app.executor.exceptions import NonRetryableToolError, RetryableToolError
from app.router.capability import ToolCapability
from app.router.permissions import PermissionContext
from app.schemas.router import RouterErrorDetail, ToolRouteCandidate, ToolRouteDecision
from app.schemas.task import TaskModel, utc_now
from app.tools.base import BaseTool
from app.utils import configure_runtime_logger, runtime_log


class TaskRouterError(Exception):
    def __init__(self, error: RouterErrorDetail) -> None:
        super().__init__(error.message)
        self.error = error


class RouterConfigurationError(Exception):
    """Raised when router registration/configuration is invalid."""


@dataclass
class _RegisteredTool:
    tool: BaseTool
    capability: ToolCapability


class TaskRouter:
    def __init__(
        self,
        *,
        permission_context: PermissionContext | None = None,
        routing_recorder: Callable[[dict[str, object]], None] | None = None,
        tool_load_provider: Callable[[str], int] | None = None,
    ) -> None:
        self._lock = asyncio.Lock()
        self._registry: dict[str, _RegisteredTool] = {}
        self._registry_order: list[str] = []
        self._permission_context = permission_context or PermissionContext()
        self._routing_recorder = routing_recorder
        self._tool_load_provider = tool_load_provider
        self.logger = configure_runtime_logger()

    def set_permission_context(self, permission_context: PermissionContext) -> None:
        self._permission_context = permission_context

    def set_routing_recorder(self, routing_recorder: Callable[[dict[str, object]], None] | None) -> None:
        self._routing_recorder = routing_recorder

    def set_tool_load_provider(self, tool_load_provider: Callable[[str], int] | None) -> None:
        self._tool_load_provider = tool_load_provider

    async def register_tool(self, tool: BaseTool, capability: ToolCapability | dict[str, object] | None = None) -> None:
        async with self._lock:
            registered = self._build_registry_entry(tool=tool, capability=capability)
            if tool.name not in self._registry:
                self._registry_order.append(tool.name)
            self._registry[tool.name] = registered

    async def register_tools(
        self,
        tools: Iterable[tuple[BaseTool, ToolCapability] | tuple[BaseTool, dict[str, object]]],
    ) -> None:
        async with self._lock:
            for item in tools:
                if not isinstance(item, tuple) or len(item) != 2:
                    raise RouterConfigurationError(
                        "register_tools() requires explicit (tool, capability) tuples for every tool"
                    )
                tool, capability = item
                registered = self._build_registry_entry(tool=tool, capability=capability)
                if tool.name not in self._registry:
                    self._registry_order.append(tool.name)
                self._registry[tool.name] = registered

    async def unregister_tool(self, tool_name: str) -> None:
        async with self._lock:
            if tool_name not in self._registry:
                raise self._error(
                    code="tool_not_found",
                    message=f"tool '{tool_name}' is not registered",
                    details={"tool_name": tool_name},
                )
            del self._registry[tool_name]
            self._registry_order = [name for name in self._registry_order if name != tool_name]

    async def get_tool(self, tool_name: str, *, require_enabled: bool = True) -> BaseTool:
        async with self._lock:
            registered = self._registry.get(tool_name)
            if registered is None:
                raise self._error(
                    code="tool_not_found",
                    message=f"tool '{tool_name}' is not registered",
                    details={"tool_name": tool_name},
                )
            if require_enabled and not self._is_registered_tool_enabled(registered):
                raise self._error(
                    code="tool_disabled",
                    message=f"tool '{tool_name}' is disabled",
                    details={"tool_name": tool_name},
                )
            return registered.tool

    async def list_tools(self, *, enabled_only: bool = False) -> list[str]:
        async with self._lock:
            return [
                name
                for name in self._registry_order
                if name in self._registry and (not enabled_only or self._is_registered_tool_enabled(self._registry[name]))
            ]

    def get_tool_capability(self, tool_name: str) -> ToolCapability | None:
        registered = self._registry.get(tool_name)
        if registered is None:
            return None
        return registered.capability.model_copy(deep=True)

    async def route_task(self, task: TaskModel) -> tuple[BaseTool, ToolRouteDecision]:
        async with self._lock:
            if not self._registry:
                self._record_routing_result(
                    task=task,
                    tool_name=task.tool,
                    routing_result="DENIED",
                    reason="unsupported tool",
                    capability_match=False,
                    permission_match=False,
                )
                raise NonRetryableToolError(
                    f"no supported tool available for task '{task.task_id}'",
                    reason="unsupported tool",
                    error_type="UNSUPPORTED_TOOL",
                )

            route_mode = self._normalize_route_mode(task)
            candidate_names = self._collect_candidate_names(task, route_mode=route_mode)
            candidates = self._build_candidates(task, candidate_names=candidate_names, route_mode=route_mode)
            selected = next((candidate for candidate in candidates if candidate.available and candidate.enabled), None)

            if selected is None:
                self._record_routing_result(
                    task=task,
                    tool_name=task.tool,
                    routing_result="DENIED",
                    reason="unsupported tool",
                    capability_match=False,
                    permission_match=False,
                )
                raise NonRetryableToolError(
                    f"no supported tool available for task '{task.task_id}'",
                    reason="unsupported tool",
                    error_type="UNSUPPORTED_TOOL",
                )

            registered = self._registry[selected.tool_name]
            self._validate_capability(task=task, registered=registered)
            self._validate_permissions(task=task, registered=registered)
            self._validate_concurrency(task=task, registered=registered)

            decision = ToolRouteDecision(
                task_id=task.task_id,
                requested_tool=task.tool,
                selected_tool=selected.tool_name,
                routing_mode=self._resolve_routing_mode(
                    task=task,
                    route_mode=route_mode,
                    selected_tool=selected.tool_name,
                ),
                reason=selected.reason,
                candidate_tools=candidates,
                failover_chain=[
                    candidate.tool_name
                    for candidate in candidates
                    if candidate.tool_name != selected.tool_name and not (candidate.available and candidate.enabled)
                ],
            )
            self._record_routing_result(
                task=task,
                tool_name=selected.tool_name,
                routing_result="ALLOWED",
                reason=selected.reason,
                capability_match=True,
                permission_match=True,
            )
            return registered.tool, decision

    def _build_registry_entry(
        self,
        *,
        tool: BaseTool,
        capability: ToolCapability | dict[str, object] | None,
    ) -> _RegisteredTool:
        if capability is None:
            raise RouterConfigurationError(
                f"tool '{tool.name}' must be registered with an explicit ToolCapability"
            )
        try:
            resolved_capability = (
                capability
                if isinstance(capability, ToolCapability)
                else ToolCapability.model_validate(capability)
            )
        except ValidationError as exc:
            raise RouterConfigurationError(
                f"tool '{tool.name}' has invalid capability configuration: {exc}"
            ) from exc
        if resolved_capability.tool_name != tool.name:
            raise RouterConfigurationError("tool capability tool_name must match the registered tool name")
        return _RegisteredTool(tool=tool, capability=resolved_capability)

    def _normalize_route_mode(self, task: TaskModel) -> str:
        raw_mode = str(task.input.get("route_mode", "static")).strip().lower()
        if task.tool == "auto" and raw_mode == "static":
            return "dynamic"
        if raw_mode not in {"static", "dynamic"}:
            raise self._error(
                code="invalid_route_mode",
                message=f"task '{task.task_id}' uses unsupported route_mode '{raw_mode}'",
                details={"task_id": task.task_id, "route_mode": raw_mode},
            )
        return raw_mode

    def _collect_candidate_names(self, task: TaskModel, *, route_mode: str) -> list[str]:
        explicit_candidates = self._normalize_name_list(task.input.get("tool_candidates", []))
        required_tags = self._normalize_name_list(task.input.get("required_tags", []))
        candidate_names: list[str] = []

        if task.tool != "auto":
            candidate_names.append(task.tool)
        candidate_names.extend(explicit_candidates)

        if route_mode == "dynamic" or required_tags:
            for tool_name in self._registry_order:
                registered = self._registry[tool_name]
                if required_tags and not all(tag in registered.tool.tags for tag in required_tags):
                    continue
                candidate_names.append(tool_name)

        return self._dedupe_preserve_order(candidate_names)

    def _build_candidates(
        self,
        task: TaskModel,
        *,
        candidate_names: list[str],
        route_mode: str,
    ) -> list[ToolRouteCandidate]:
        if not candidate_names:
            raise self._error(
                code="no_candidate_tools",
                message=f"task '{task.task_id}' did not produce any candidate tools",
                details={"task_id": task.task_id, "route_mode": route_mode},
            )

        explicit_candidates = self._normalize_name_list(task.input.get("tool_candidates", []))
        required_tags = self._normalize_name_list(task.input.get("required_tags", []))
        candidates: list[ToolRouteCandidate] = []

        for index, tool_name in enumerate(candidate_names):
            registered = self._registry.get(tool_name)
            available = registered is not None
            enabled = self._is_registered_tool_enabled(registered) if registered is not None else False
            score = max(0, 100 - index)
            reason_parts: list[str] = []

            if tool_name == task.tool:
                score += 100
                reason_parts.append("matched requested tool")
            elif tool_name in explicit_candidates:
                score += 60
                reason_parts.append("matched explicit candidate")

            if registered is not None and required_tags:
                matched_tags = [tag for tag in required_tags if tag in registered.tool.tags]
                if matched_tags:
                    score += len(matched_tags) * 20
                    reason_parts.append(f"matched routing tags: {', '.join(matched_tags)}")

            if route_mode == "dynamic" and registered is not None:
                score += 10
                reason_parts.append("dynamic routing candidate")

            if registered is None:
                reason_parts.append("tool not found")
            elif not enabled:
                reason_parts.append("tool disabled")
            else:
                reason_parts.append("tool available")

            candidates.append(
                ToolRouteCandidate(
                    tool_name=tool_name,
                    available=available,
                    enabled=enabled,
                    score=score,
                    reason="; ".join(reason_parts),
                )
            )

        candidates.sort(
            key=lambda candidate: (
                candidate.available and candidate.enabled,
                candidate.score,
                -self._registry_index(candidate.tool_name),
            ),
            reverse=True,
        )
        return candidates

    def _validate_capability(self, *, task: TaskModel, registered: _RegisteredTool) -> None:
        capability = registered.capability
        if not capability.enabled or not registered.tool.enabled:
            self._record_routing_result(
                task=task,
                tool_name=registered.tool.name,
                routing_result="DENIED",
                reason="tool disabled",
                capability_match=False,
                permission_match=False,
            )
            raise NonRetryableToolError(
                f"tool '{registered.tool.name}' is disabled",
                reason="tool disabled",
                error_type="UNSUPPORTED_TOOL",
            )

        if capability.supported_task_types and task.task_type is not None and task.task_type not in capability.supported_task_types:
            self._record_routing_result(
                task=task,
                tool_name=registered.tool.name,
                routing_result="DENIED",
                reason="unsupported task type",
                capability_match=False,
                permission_match=True,
            )
            raise NonRetryableToolError(
                f"tool '{registered.tool.name}' does not support task_type '{task.task_type}'",
                reason="unsupported task type",
                error_type="CAPABILITY_MISMATCH",
            )

        if capability.supported_tags and task.tags and not all(tag in capability.supported_tags for tag in task.tags):
            self._record_routing_result(
                task=task,
                tool_name=registered.tool.name,
                routing_result="DENIED",
                reason="tags mismatch",
                capability_match=False,
                permission_match=True,
            )
            raise NonRetryableToolError(
                f"tool '{registered.tool.name}' does not support required tags",
                reason="tags mismatch",
                error_type="CAPABILITY_MISMATCH",
            )

        if task.max_retry > 0 and not capability.supports_retry:
            self._record_routing_result(
                task=task,
                tool_name=registered.tool.name,
                routing_result="DENIED",
                reason="retry not supported",
                capability_match=False,
                permission_match=True,
            )
            raise NonRetryableToolError(
                f"tool '{registered.tool.name}' does not support retry",
                reason="retry not supported",
                error_type="CAPABILITY_MISMATCH",
            )

        if task.timeout > 0 and not capability.supports_timeout:
            self._record_routing_result(
                task=task,
                tool_name=registered.tool.name,
                routing_result="DENIED",
                reason="timeout not supported",
                capability_match=False,
                permission_match=True,
            )
            raise NonRetryableToolError(
                f"tool '{registered.tool.name}' does not support timeout",
                reason="timeout not supported",
                error_type="CAPABILITY_MISMATCH",
            )

    def _validate_permissions(self, *, task: TaskModel, registered: _RegisteredTool) -> None:
        capability = registered.capability
        permission_context = self._permission_context

        if capability.allowed_roles and not set(capability.allowed_roles).intersection(permission_context.roles):
            self._record_routing_result(
                task=task,
                tool_name=registered.tool.name,
                routing_result="DENIED",
                reason="role not allowed",
                capability_match=True,
                permission_match=False,
            )
            raise NonRetryableToolError(
                f"permission denied for tool '{registered.tool.name}'",
                reason="role not allowed",
                error_type="PERMISSION_DENIED",
            )

        if capability.required_permissions and not all(
            permission in permission_context.permissions for permission in capability.required_permissions
        ):
            self._record_routing_result(
                task=task,
                tool_name=registered.tool.name,
                routing_result="DENIED",
                reason="missing required permission",
                capability_match=True,
                permission_match=False,
            )
            raise NonRetryableToolError(
                f"permission denied for tool '{registered.tool.name}'",
                reason="missing required permission",
                error_type="PERMISSION_DENIED",
            )

    def _validate_concurrency(self, *, task: TaskModel, registered: _RegisteredTool) -> None:
        if self._tool_load_provider is None:
            current_load = 0
        else:
            try:
                current_load = self._tool_load_provider(registered.tool.name)
            except Exception as exc:
                self._record_routing_result(
                    task=task,
                    tool_name=registered.tool.name,
                    routing_result="DENIED",
                    reason="tool load unavailable",
                    capability_match=True,
                    permission_match=True,
                )
                raise RetryableToolError(
                    f"tool load unavailable for '{registered.tool.name}': {exc}",
                    reason=f"tool load unavailable: {exc}",
                    error_type="TOOL_LOAD_UNAVAILABLE",
                    original_exception=exc,
                ) from exc
        if current_load >= registered.capability.max_concurrency:
            self._record_routing_result(
                task=task,
                tool_name=registered.tool.name,
                routing_result="DENIED",
                reason="tool concurrency limit reached",
                capability_match=True,
                permission_match=True,
            )
            raise RetryableToolError(
                f"tool '{registered.tool.name}' reached concurrency limit",
                reason="tool concurrency limit reached",
                error_type="TOOL_CONCURRENCY_LIMIT",
            )

    def _resolve_routing_mode(self, *, task: TaskModel, route_mode: str, selected_tool: str) -> str:
        if selected_tool == task.tool and task.tool != "auto":
            return "static"
        if selected_tool != task.tool and task.tool != "auto":
            return "failover"
        return route_mode

    def _normalize_name_list(self, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            items = [value]
        elif isinstance(value, list):
            items = value
        else:
            raise self._error(
                code="invalid_routing_hint",
                message="routing hints must be a string or list of strings",
                details={"python_type": type(value).__name__},
            )

        normalized: list[str] = []
        for item in items:
            if not isinstance(item, str):
                raise self._error(
                    code="invalid_routing_hint",
                    message="routing hints must contain only strings",
                    details={"python_type": type(item).__name__},
                )
            item_value = item.strip()
            if item_value:
                normalized.append(item_value)
        return normalized

    def _dedupe_preserve_order(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for item in items:
            if item not in seen:
                deduped.append(item)
                seen.add(item)
        return deduped

    def _registry_index(self, tool_name: str) -> int:
        try:
            return self._registry_order.index(tool_name)
        except ValueError:
            return len(self._registry_order)

    def _record_routing_result(
        self,
        *,
        task: TaskModel,
        tool_name: str,
        routing_result: str,
        reason: str,
        capability_match: bool,
        permission_match: bool,
    ) -> None:
        if self._routing_recorder is None:
            return
        try:
            self._routing_recorder(
                {
                    "task_id": task.task_id,
                    "tool_name": tool_name,
                    "routing_result": routing_result,
                    "reason": reason,
                    "capability_match": capability_match,
                    "permission_match": permission_match,
                    "timestamp": utc_now().isoformat(),
                }
            )
        except Exception as exc:
            runtime_log(
                layer="router",
                event="error",
                data={
                    "message": "routing_recorder failed",
                    "task_id": task.task_id,
                    "tool_name": tool_name,
                    "routing_result": routing_result,
                    "exception_type": type(exc).__name__,
                    "error": str(exc),
                },
                logger=self.logger,
            )

    def _is_registered_tool_enabled(self, registered: _RegisteredTool | None) -> bool:
        return bool(registered is not None and registered.tool.enabled and registered.capability.enabled)

    def _error(self, *, code: str, message: str, details: dict[str, object] | None = None) -> TaskRouterError:
        return TaskRouterError(RouterErrorDetail(code=code, message=message, details=details or {}))
