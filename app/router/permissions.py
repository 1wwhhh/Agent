from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PermissionContext(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    roles: list[str] = Field(default_factory=list)
    permissions: list[str] = Field(default_factory=list)
    request_source: str = Field(default="runtime", min_length=1)

    @field_validator("roles", "permissions")
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
