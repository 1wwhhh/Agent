from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.connectors.feishu.client import FeishuClient
from app.connectors.feishu.exceptions import FeishuApiError, FeishuDownloadError
from app.connectors.feishu.models import FileItem

SUPPORTED_DIRECT_DOWNLOAD_TYPES = {"file", "pdf"}
SUPPORTED_EXPORT_TYPES = {"docx", "sheet", "bitable"}
SUPPORTED_KNOWN_TYPES = SUPPORTED_DIRECT_DOWNLOAD_TYPES | SUPPORTED_EXPORT_TYPES | {"folder", "doc", "shortcut_info"}


class _FolderListResponseData(BaseModel):
    model_config = ConfigDict(extra="allow", validate_assignment=True, str_strip_whitespace=True)

    items: list[dict[str, Any]] = Field(default_factory=list)
    files: list[dict[str, Any]] = Field(default_factory=list)
    has_more: bool = Field(default=False)
    page_token: str | None = Field(default=None)


class _FolderListResponse(BaseModel):
    model_config = ConfigDict(extra="allow", validate_assignment=True, str_strip_whitespace=True)

    code: int = Field(default=0)
    msg: str | None = Field(default=None)
    data: _FolderListResponseData = Field(default_factory=_FolderListResponseData)


class FeishuDriveClient:
    def __init__(self, client: FeishuClient | None = None) -> None:
        self.client = client or FeishuClient()

    async def list_folder_files(self, folder_token: str) -> list[FileItem]:
        page_token: str | None = None
        items: list[FileItem] = []
        seen_page_tokens: set[str] = set()

        while True:
            params: dict[str, Any] = {
                "folder_token": folder_token,
                "page_size": 200,
            }
            if page_token:
                params["page_token"] = page_token

            response = await self.client.get("drive/v1/files", params=params)
            parsed = _FolderListResponse.model_validate(response)
            raw_items = parsed.data.files or parsed.data.items
            for raw_item in raw_items:
                items.append(self._normalize_item(raw_item))

            if not parsed.data.has_more:
                break

            next_page_token = (parsed.data.page_token or "").strip()
            if not next_page_token:
                break
            if next_page_token in seen_page_tokens:
                raise FeishuApiError(
                    "folder pagination returned a repeated page_token",
                    path="/drive/v1/files",
                    response_body=response,
                )
            seen_page_tokens.add(next_page_token)
            page_token = next_page_token

        return items

    async def download_file(self, file_token: str, save_path: str) -> None:
        try:
            await self.client.download_to_path(f"drive/v1/files/{file_token}/download", save_path)
        except FeishuDownloadError:
            raise

    def _normalize_item(self, raw_item: dict[str, Any]) -> FileItem:
        name = str(raw_item.get("name") or raw_item.get("file_name") or raw_item.get("token") or "").strip()
        token = str(raw_item.get("token") or raw_item.get("file_token") or "").strip()
        raw_type = str(raw_item.get("obj_type") or raw_item.get("type") or "").strip().lower()
        url = raw_item.get("url")
        parent_token = raw_item.get("parent_token")

        if not name:
            name = token or "untitled"
        if not token:
            raise FeishuApiError(
                "folder item is missing token",
                path="/drive/v1/files",
                response_body=raw_item,
            )

        normalized_type = self._normalize_type(raw_type=raw_type, name=name, raw_item=raw_item)
        return FileItem(
            name=name,
            token=token,
            type=normalized_type,
            url=str(url).strip() if url else None,
            parent_token=str(parent_token).strip() if parent_token else None,
        )

    def _normalize_type(self, *, raw_type: str, name: str, raw_item: dict[str, Any]) -> str:
        if raw_item.get("shortcut_info") is not None or raw_type == "shortcut":
            return "shortcut_info"
        if raw_type in SUPPORTED_KNOWN_TYPES:
            if raw_type == "file" and name.lower().endswith(".pdf"):
                return "pdf"
            return raw_type
        if name.lower().endswith(".pdf"):
            return "pdf"
        if raw_type:
            return raw_type
        return "unknown"
