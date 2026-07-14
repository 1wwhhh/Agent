from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from app.executor.exceptions import NonRetryableToolError, RetryableToolError
from app.router import RouterConfigurationError
from app.router.capability import ToolCapability, capability_from_tool
from app.router.permissions import PermissionContext
from app.router.task_router import TaskRouter
from app.schemas.context import ContextStore, RuntimeContext
from app.schemas.task import TaskModel, TaskStatus
from app.tools.base import BaseTool
from tests.support.runtime_runner import run_runtime
from tests.support.test_case_generator import TestCaseGenerator
from tests.support.tools import RuntimeTestTool


class CountingTool(BaseTool):
    calls: int = 0

    async def _arun(self, payload: dict[str, Any], context: ContextStore | None = None):
        self.calls += 1
        return self.build_result(success=True, output={"ok": True})


def _build_task(
    *,
    task_id: str = "task_1",
    tool: str = "text_generate_tool",
    task_type: str | None = None,
    tags: list[str] | None = None,
    max_retry: int = 1,
    timeout: int = 60,
    input_payload: dict[str, Any] | None = None,
) -> TaskModel:
    return TaskModel(
        task_id=task_id,
        task_name=task_id,
        description=task_id,
        task_type=task_type,
        tool=tool,
        input=input_payload or {"prompt": "hello"},
        output_key=f"{task_id}_output",
        max_retry=max_retry,
        timeout=timeout,
        tags=tags or [],
    )


def _build_capability(
    *,
    tool_name: str,
    enabled: bool = True,
    supported_task_types: list[str] | None = None,
    default_task_type: str | None = None,
    supported_tags: list[str] | None = None,
    required_permissions: list[str] | None = None,
    allowed_roles: list[str] | None = None,
    max_concurrency: int = 64,
    supports_retry: bool = True,
    supports_timeout: bool = True,
) -> ToolCapability:
    return ToolCapability(
        tool_name=tool_name,
        enabled=enabled,
        supported_task_types=supported_task_types or [],
        default_task_type=default_task_type,
        supported_tags=supported_tags or [],
        required_permissions=required_permissions or [],
        allowed_roles=allowed_roles or [],
        max_concurrency=max_concurrency,
        supports_retry=supports_retry,
        supports_timeout=supports_timeout,
    )


@pytest.mark.asyncio
async def test_enabled_tool_routes_normally() -> None:
    router = TaskRouter()
    tool = CountingTool(name="text_generate_tool", description="text")
    await router.register_tool(tool, _build_capability(tool_name=tool.name))

    task = _build_task()
    selected_tool, decision = await router.route_task(task)

    assert selected_tool.name == "text_generate_tool"
    assert decision.selected_tool == "text_generate_tool"


@pytest.mark.asyncio
async def test_disabled_tool_raises_unsupported_tool() -> None:
    router = TaskRouter()
    tool = CountingTool(name="text_generate_tool", description="text", enabled=False)
    await router.register_tool(tool, _build_capability(tool_name=tool.name, enabled=False))

    with pytest.raises(NonRetryableToolError) as exc_info:
        await router.route_task(_build_task())

    assert exc_info.value.error_type == "UNSUPPORTED_TOOL"


@pytest.mark.asyncio
async def test_unsupported_task_type_raises_capability_mismatch() -> None:
    router = TaskRouter()
    tool = CountingTool(name="text_generate_tool", description="text")
    await router.register_tool(
        tool,
        _build_capability(tool_name=tool.name, supported_task_types=["analysis"]),
    )

    with pytest.raises(NonRetryableToolError) as exc_info:
        await router.route_task(_build_task(task_type="generation"))

    assert exc_info.value.error_type == "CAPABILITY_MISMATCH"


@pytest.mark.asyncio
async def test_permission_denied_for_missing_role() -> None:
    router = TaskRouter(permission_context=PermissionContext(roles=["viewer"], permissions=[]))
    tool = CountingTool(name="text_generate_tool", description="text")
    await router.register_tool(
        tool,
        _build_capability(tool_name=tool.name, allowed_roles=["admin"]),
    )

    with pytest.raises(NonRetryableToolError) as exc_info:
        await router.route_task(_build_task())

    assert exc_info.value.error_type == "PERMISSION_DENIED"


@pytest.mark.asyncio
async def test_missing_required_permission_raises_permission_denied() -> None:
    router = TaskRouter(permission_context=PermissionContext(roles=["admin"], permissions=["read"]))
    tool = CountingTool(name="text_generate_tool", description="text")
    await router.register_tool(
        tool,
        _build_capability(tool_name=tool.name, required_permissions=["read", "write"]),
    )

    with pytest.raises(NonRetryableToolError) as exc_info:
        await router.route_task(_build_task())

    assert exc_info.value.error_type == "PERMISSION_DENIED"


@pytest.mark.asyncio
async def test_unsupported_tool_name_raises_unsupported_tool() -> None:
    router = TaskRouter()

    with pytest.raises(NonRetryableToolError) as exc_info:
        await router.route_task(_build_task(tool="missing_tool"))

    assert exc_info.value.error_type == "UNSUPPORTED_TOOL"


@pytest.mark.asyncio
async def test_tags_mismatch_raises_capability_mismatch() -> None:
    router = TaskRouter()
    tool = CountingTool(name="text_generate_tool", description="text")
    await router.register_tool(
        tool,
        _build_capability(tool_name=tool.name, supported_tags=["report"]),
    )

    with pytest.raises(NonRetryableToolError) as exc_info:
        await router.route_task(_build_task(tags=["ppt"]))

    assert exc_info.value.error_type == "CAPABILITY_MISMATCH"

@pytest.mark.asyncio
async def test_text_generate_tool_allows_supported_generation_tags() -> None:
    router = TaskRouter()
    tool = CountingTool(name="text_generate_tool", description="text")
    await router.register_tool(
        tool,
        _build_capability(
            tool_name=tool.name,
            supported_task_types=["text_generation"],
            supported_tags=["llm", "generation", "text"],
        ),
    )

    selected_tool, decision = await router.route_task(
        _build_task(task_type="text_generation", tags=["llm", "generation"])
    )

    assert selected_tool.name == "text_generate_tool"
    assert decision.selected_tool == "text_generate_tool"


@pytest.mark.asyncio
async def test_text_generate_tool_rejects_chinese_generation_tag() -> None:
    router = TaskRouter()
    tool = CountingTool(name="text_generate_tool", description="text")
    await router.register_tool(
        tool,
        _build_capability(
            tool_name=tool.name,
            supported_task_types=["text_generation"],
            supported_tags=["llm", "generation", "text"],
        ),
    )

    with pytest.raises(NonRetryableToolError) as exc_info:
        await router.route_task(_build_task(task_type="text_generation", tags=["llm", "生成"]))

    assert exc_info.value.error_type == "CAPABILITY_MISMATCH"
    assert "does not support required tags" in str(exc_info.value)


@pytest.mark.asyncio
async def test_concurrency_limit_raises_retryable_tool_error() -> None:
    router = TaskRouter(tool_load_provider=lambda _tool_name: 1)
    tool = CountingTool(name="text_generate_tool", description="text")
    await router.register_tool(
        tool,
        _build_capability(tool_name=tool.name, max_concurrency=1),
    )

    with pytest.raises(RetryableToolError) as exc_info:
        await router.route_task(_build_task())

    assert exc_info.value.error_type == "TOOL_CONCURRENCY_LIMIT"


@pytest.mark.asyncio
async def test_routing_history_recorder_writes_metadata() -> None:
    metadata: dict[str, Any] = {"routing_history": []}

    def _recorder(record: dict[str, object]) -> None:
        history = list(metadata.get("routing_history", []))
        history.append(record)
        metadata["routing_history"] = history

    router = TaskRouter(routing_recorder=_recorder)
    tool = CountingTool(name="text_generate_tool", description="text")
    await router.register_tool(tool, _build_capability(tool_name=tool.name))
    await router.route_task(_build_task())

    history = metadata["routing_history"]
    assert len(history) == 1
    assert history[0]["routing_result"] == "ALLOWED"


@pytest.mark.asyncio
async def test_router_does_not_modify_task_status() -> None:
    router = TaskRouter()
    tool = CountingTool(name="text_generate_tool", description="text")
    await router.register_tool(tool, _build_capability(tool_name=tool.name))
    task = _build_task()
    original_status = task.status

    await router.route_task(task)

    assert task.status == original_status == TaskStatus.PENDING


@pytest.mark.asyncio
async def test_router_does_not_execute_tool() -> None:
    router = TaskRouter()
    tool = CountingTool(name="text_generate_tool", description="text")
    await router.register_tool(tool, _build_capability(tool_name=tool.name))

    await router.route_task(_build_task())

    assert tool.calls == 0


@pytest.mark.asyncio
async def test_permission_context_is_not_read_from_task_input() -> None:
    router = TaskRouter(permission_context=PermissionContext())
    tool = CountingTool(name="text_generate_tool", description="text")
    await router.register_tool(
        tool,
        _build_capability(tool_name=tool.name, required_permissions=["admin:write"]),
    )

    task = _build_task(
        input_payload={
            "prompt": "hello",
            "permissions": ["admin:write"],
            "roles": ["admin"],
        }
    )

    with pytest.raises(NonRetryableToolError) as exc_info:
        await router.route_task(task)

    assert exc_info.value.error_type == "PERMISSION_DENIED"


@pytest.mark.asyncio
async def test_tool_load_provider_defaults_to_zero_when_missing() -> None:
    router = TaskRouter()
    tool = CountingTool(name="text_generate_tool", description="text")
    await router.register_tool(
        tool,
        _build_capability(tool_name=tool.name, max_concurrency=1),
    )

    selected_tool, _decision = await router.route_task(_build_task())

    assert selected_tool.name == "text_generate_tool"


@pytest.mark.asyncio
async def test_default_llm_tools_capability_allows_normal_routing() -> None:
    router = TaskRouter()
    reason_tool = RuntimeTestTool(name="llm_reason_tool", description="reason", tags=["reasoning", "llm"])
    text_tool = RuntimeTestTool(name="text_generate_tool", description="text", tags=["text", "generation", "llm"])
    await router.register_tools(
        [
            (reason_tool, capability_from_tool(reason_tool)),
            (text_tool, capability_from_tool(text_tool)),
        ]
    )

    reason_tool, reason_decision = await router.route_task(
        _build_task(tool="llm_reason_tool", task_id="task_reason")
    )
    text_tool, text_decision = await router.route_task(
        _build_task(tool="text_generate_tool", task_id="task_text")
    )

    assert reason_tool.name == "llm_reason_tool"
    assert text_tool.name == "text_generate_tool"
    assert reason_decision.selected_tool == "llm_reason_tool"
    assert text_decision.selected_tool == "text_generate_tool"


@pytest.mark.asyncio
async def test_runtime_graph_writes_routing_history_to_metadata() -> None:
    result = await run_runtime(TestCaseGenerator.simple_success_case())

    routing_history = result.final_output["metadata"]["routing_history"]
    assert routing_history
    assert routing_history[0]["routing_result"] == "ALLOWED"


@pytest.mark.asyncio
async def test_register_tool_requires_explicit_capability() -> None:
    router = TaskRouter()
    tool = CountingTool(name="text_generate_tool", description="text")

    with pytest.raises(RouterConfigurationError):
        await router.register_tool(tool)


@pytest.mark.asyncio
async def test_register_tools_requires_explicit_capability_tuple() -> None:
    router = TaskRouter()
    tool = CountingTool(name="text_generate_tool", description="text")

    with pytest.raises(RouterConfigurationError):
        await router.register_tools([tool])  # type: ignore[list-item]


@pytest.mark.asyncio
async def test_runtime_default_tools_are_registered_with_explicit_capabilities() -> None:
    tool = CountingTool(name="text_generate_tool", description="text", tags=["text"])
    capability = capability_from_tool(tool)
    assert capability.tool_name == tool.name
    assert capability.supported_tags == []
    router = TaskRouter()
    await router.register_tool(tool, capability)
    selected_tool, decision = await router.route_task(_build_task(tool=tool.name))
    assert selected_tool.name == tool.name
    assert decision.selected_tool == tool.name


@pytest.mark.asyncio
async def test_router_configuration_error_is_registration_only() -> None:
    router = TaskRouter()
    tool = CountingTool(name="text_generate_tool", description="text")
    with pytest.raises(RouterConfigurationError):
        await router.register_tool(tool, {"tool_name": "other_tool"})

    valid_tool = CountingTool(name="valid_tool", description="valid")
    await router.register_tool(valid_tool, _build_capability(tool_name=valid_tool.name, enabled=False))
    with pytest.raises(NonRetryableToolError):
        await router.route_task(_build_task(tool="valid_tool"))


@pytest.mark.asyncio
async def test_routing_recorder_failure_does_not_interrupt_allowed_route() -> None:
    router = TaskRouter(routing_recorder=lambda _record: (_ for _ in ()).throw(RuntimeError("recorder exploded")))
    tool = CountingTool(name="text_generate_tool", description="text")
    await router.register_tool(tool, _build_capability(tool_name=tool.name))

    selected_tool, decision = await router.route_task(_build_task())

    assert selected_tool.name == tool.name
    assert decision.selected_tool == tool.name
    assert decision.reason


@pytest.mark.asyncio
async def test_tool_load_provider_error_becomes_retryable_tool_error() -> None:
    router = TaskRouter(tool_load_provider=lambda _tool_name: (_ for _ in ()).throw(RuntimeError("load unavailable")))
    tool = CountingTool(name="text_generate_tool", description="text")
    await router.register_tool(tool, _build_capability(tool_name=tool.name, max_concurrency=2))
    task = _build_task()
    original_status = task.status

    with pytest.raises(RetryableToolError) as exc_info:
        await router.route_task(task)

    assert exc_info.value.error_type == "TOOL_LOAD_UNAVAILABLE"
    assert "load unavailable" in str(exc_info.value)
    assert task.status == original_status == TaskStatus.PENDING


@pytest.mark.asyncio
async def test_empty_supported_task_types_means_no_restriction() -> None:
    router = TaskRouter()
    tool = CountingTool(name="text_generate_tool", description="text")
    await router.register_tool(tool, _build_capability(tool_name=tool.name, supported_task_types=[]))

    selected_tool, _decision = await router.route_task(_build_task(task_type="any-task-type"))
    assert selected_tool.name == tool.name


@pytest.mark.asyncio
async def test_empty_supported_tags_means_no_restriction() -> None:
    router = TaskRouter()
    tool = CountingTool(name="text_generate_tool", description="text")
    await router.register_tool(tool, _build_capability(tool_name=tool.name, supported_tags=[]))

    selected_tool, _decision = await router.route_task(_build_task(tags=["ppt", "report"]))
    assert selected_tool.name == tool.name


@pytest.mark.asyncio
async def test_empty_allowed_roles_means_no_role_restriction() -> None:
    router = TaskRouter(permission_context=PermissionContext(roles=[], permissions=[]))
    tool = CountingTool(name="text_generate_tool", description="text")
    await router.register_tool(tool, _build_capability(tool_name=tool.name, allowed_roles=[]))

    selected_tool, _decision = await router.route_task(_build_task())
    assert selected_tool.name == tool.name


@pytest.mark.asyncio
async def test_empty_required_permissions_means_no_permission_requirement() -> None:
    router = TaskRouter(permission_context=PermissionContext(roles=["viewer"], permissions=[]))
    tool = CountingTool(name="text_generate_tool", description="text")
    await router.register_tool(tool, _build_capability(tool_name=tool.name, required_permissions=[]))

    selected_tool, _decision = await router.route_task(_build_task())
    assert selected_tool.name == tool.name


def test_tool_capability_sets_default_task_type_for_single_supported_type() -> None:
    capability = ToolCapability(tool_name="text_generate_tool", supported_task_types=["text_generation"])

    assert capability.default_task_type == "text_generation"


def test_tool_capability_rejects_default_task_type_outside_supported_task_types() -> None:
    with pytest.raises(ValidationError):
        ToolCapability(
            tool_name="multi_tool",
            supported_task_types=["analysis", "generation"],
            default_task_type="summary",
        )


def test_tool_capability_rejects_different_default_for_single_supported_type() -> None:
    with pytest.raises(ValidationError):
        ToolCapability(
            tool_name="text_generate_tool",
            supported_task_types=["text_generation"],
            default_task_type="reasoning",
        )


def test_tool_capability_rejects_zero_max_concurrency() -> None:
    with pytest.raises(ValidationError):
        ToolCapability(tool_name="text_generate_tool", max_concurrency=0)


def test_tool_capability_rejects_negative_max_concurrency() -> None:
    with pytest.raises(ValidationError):
        ToolCapability(tool_name="text_generate_tool", max_concurrency=-1)
