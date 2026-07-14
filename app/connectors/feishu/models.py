from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FileItem(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    name: str = Field(..., min_length=1)
    token: str = Field(..., min_length=1)
    type: str = Field(..., min_length=1)
    url: str | None = Field(default=None)
    parent_token: str | None = Field(default=None)


class DestinationRule(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    source_prefix: str = Field(..., min_length=1)
    target_root: str = Field(..., min_length=1)


class SyncResult(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    success: bool
    downloaded: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    skipped: int = Field(default=0, ge=0)
    nas_dir: str
    errors: list[str] = Field(default_factory=list)
