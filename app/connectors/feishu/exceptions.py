from __future__ import annotations

from typing import Any


class FeishuError(Exception):
    """Base class for Feishu connector failures."""


class FeishuConfigError(FeishuError):
    """Raised when Feishu configuration is missing or invalid."""


class FeishuAuthError(FeishuError):
    """Raised when tenant access token acquisition fails."""


class FeishuTransportError(FeishuError):
    """Raised when transport-level retries are exhausted."""


class FeishuDownloadError(FeishuError):
    """Raised when a Feishu download cannot be completed."""


class FeishuApiError(FeishuError):
    def __init__(
        self,
        message: str,
        *,
        path: str,
        status_code: int | None = None,
        code: int | None = None,
        response_body: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.path = path
        self.status_code = status_code
        self.code = code
        self.response_body = response_body


class FeishuExportError(FeishuError):
    """Raised when a Feishu export task fails or times out."""


class FeishuSyncError(FeishuError):
    """Raised when folder synchronization fails before individual file handling."""
