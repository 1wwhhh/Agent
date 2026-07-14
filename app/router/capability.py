from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:
    from app.tools.base import BaseTool


class ToolCapability(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    tool_name: str = Field(..., min_length=1)
    enabled: bool = Field(default=True)
    supported_task_types: list[str] = Field(default_factory=list)
    default_task_type: str | None = Field(default=None)
    supported_tags: list[str] = Field(default_factory=list)
    required_permissions: list[str] = Field(default_factory=list)
    allowed_roles: list[str] = Field(default_factory=list)
    max_concurrency: int = Field(default=64, ge=1)
    supports_streaming: bool = Field(default=False)
    supports_retry: bool = Field(default=True)
    supports_timeout: bool = Field(default=True)

    @field_validator(
        "supported_task_types",
        "supported_tags",
        "required_permissions",
        "allowed_roles",
    )
    @classmethod
    def _normalize_unique_values(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for item in values:
            value = item.strip()
            if not value:
                continue
            if value not in seen:
                normalized.append(value)
                seen.add(value)
        return normalized

    @model_validator(mode="after")
    def validate_default_task_type(self) -> "ToolCapability":
        default_task_type = self.default_task_type.strip() if self.default_task_type is not None else None
        if default_task_type == "":
            default_task_type = None

        if len(self.supported_task_types) == 1:
            only_supported = self.supported_task_types[0]
            if default_task_type is None:
                object.__setattr__(self, "default_task_type", only_supported)
                return self
            if default_task_type != only_supported:
                raise ValueError("default_task_type must equal the only supported_task_type")

        if default_task_type is not None and default_task_type not in self.supported_task_types:
            raise ValueError("default_task_type must be one of supported_task_types")

        object.__setattr__(self, "default_task_type", default_task_type)
        return self


def capability_from_tool(tool: "BaseTool") -> ToolCapability:
    """Build an explicit ToolCapability from a tool helper payload."""
    return ToolCapability.model_validate(tool.get_routing_capability())
