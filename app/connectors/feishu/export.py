from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.connectors.feishu.client import FeishuClient
from app.connectors.feishu.exceptions import FeishuApiError, FeishuExportError

SUPPORTED_EXPORT_TARGETS = {
    "docx": {"file_extension": "docx"},
    "sheet": {"file_extension": "xlsx"},
    "bitable": {"file_extension": "xlsx"},
}


class _ExportTaskCreateData(BaseModel):
    model_config = ConfigDict(extra="allow", validate_assignment=True, str_strip_whitespace=True)

    ticket: str | None = Field(default=None)


class _ExportTaskCreateResponse(BaseModel):
    model_config = ConfigDict(extra="allow", validate_assignment=True, str_strip_whitespace=True)

    code: int = Field(default=0)
    msg: str | None = Field(default=None)
    data: _ExportTaskCreateData = Field(default_factory=_ExportTaskCreateData)


class _ExportTaskResult(BaseModel):
    model_config = ConfigDict(extra="allow", validate_assignment=True, str_strip_whitespace=True)

    ticket: str | None = Field(default=None)
    type: str | None = Field(default=None)
    job_status: int | None = Field(default=None)
    job_error_msg: str | None = Field(default=None)
    token: str | None = Field(default=None)
    file_token: str | None = Field(default=None)
    file_extension: str | None = Field(default=None)
    file_name: str | None = Field(default=None)


class _ExportTaskStatusData(BaseModel):
    model_config = ConfigDict(extra="allow", validate_assignment=True, str_strip_whitespace=True)

    result: _ExportTaskResult | None = Field(default=None)


class _ExportTaskStatusResponse(BaseModel):
    model_config = ConfigDict(extra="allow", validate_assignment=True, str_strip_whitespace=True)

    code: int = Field(default=0)
    msg: str | None = Field(default=None)
    data: _ExportTaskStatusData = Field(default_factory=_ExportTaskStatusData)


class FeishuExportClient:
    def __init__(
        self,
        client: FeishuClient | None = None,
        *,
        poll_interval_seconds: float = 1.5,
        max_poll_attempts: int = 60,
    ) -> None:
        self.client = client or FeishuClient()
        self.poll_interval_seconds = max(0.1, float(poll_interval_seconds))
        self.max_poll_attempts = max(1, int(max_poll_attempts))

    async def export_and_download(self, file_token: str, file_type: str, save_path: str) -> None:
        target = self._resolve_export_target(file_type)
        ticket = await self._create_export_task(file_token=file_token, file_type=file_type, file_extension=target["file_extension"])
        result = await self._poll_export_task(ticket=ticket, token=file_token)
        export_file_token = (result.file_token or result.token or "").strip()
        if not export_file_token:
            raise FeishuExportError("export task completed without a downloadable file token")
        await self.client.download_to_path(
            f"drive/v1/export_tasks/file/{export_file_token}/download",
            save_path,
        )

    async def _create_export_task(self, *, file_token: str, file_type: str, file_extension: str) -> str:
        response = await self.client.post(
            "drive/v1/export_tasks",
            json_body={
                "file_extension": file_extension,
                "token": file_token,
                "type": file_type,
            },
        )
        parsed = _ExportTaskCreateResponse.model_validate(response)
        ticket = (parsed.data.ticket or "").strip()
        if not ticket:
            raise FeishuApiError(
                "export task response did not include a ticket",
                path="/drive/v1/export_tasks",
                response_body=response,
            )
        return ticket

    async def _poll_export_task(self, *, ticket: str, token: str) -> _ExportTaskResult:
        last_result: _ExportTaskResult | None = None
        for _attempt in range(1, self.max_poll_attempts + 1):
            response = await self.client.get(f"drive/v1/export_tasks/{ticket}", params={"token": token})
            parsed = _ExportTaskStatusResponse.model_validate(response)
            result = parsed.data.result
            if result is None:
                await asyncio.sleep(self.poll_interval_seconds)
                continue

            last_result = result
            job_status = result.job_status
            job_error_msg = (result.job_error_msg or "").strip().lower()
            export_file_token = (result.file_token or result.token or "").strip()

            if job_status == 0 and export_file_token:
                return result

            if job_status is not None and job_status < 0:
                raise FeishuExportError(f"export task failed with status {job_status}: {result.job_error_msg}")

            if job_error_msg and job_error_msg not in {"success", "processing", "pending", "running"}:
                raise FeishuExportError(f"export task failed: {result.job_error_msg}")

            await asyncio.sleep(self.poll_interval_seconds)

        if last_result is None:
            raise FeishuExportError(f"export task timed out for ticket {ticket}")
        raise FeishuExportError(
            f"export task timed out for ticket {ticket} with last status {last_result.job_status}"
        )

    def _resolve_export_target(self, file_type: str) -> dict[str, str]:
        normalized = str(file_type or "").strip().lower()
        target = SUPPORTED_EXPORT_TARGETS.get(normalized)
        if target is None:
            raise FeishuExportError(f"unsupported export file type: {file_type!r}")
        return target
