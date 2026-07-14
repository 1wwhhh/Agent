from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import Field

from app.connectors.feishu.downloader import FeishuSyncDownloader
from app.connectors.feishu.exceptions import FeishuConfigError
from app.connectors.feishu.models import DestinationRule, SyncResult
from app.connectors.feishu.utils import validate_feishu_folder_url_value, validate_feishu_nas_dir_value
from app.schemas.context import ContextStore
from app.schemas.tool import ToolResult
from app.tools.base import BaseTool

FEISHU_FOLDER_URL_ENV = "FEISHU_FOLDER_URL"
FEISHU_SYNC_NAS_DIR_ENV = "FEISHU_SYNC_NAS_DIR"
FEISHU_SYNC_DESTINATION_RULES_ENV = "FEISHU_SYNC_DESTINATION_RULES"
FEISHU_SYNC_DESTINATION_RULES_FILE_ENV = "FEISHU_SYNC_DESTINATION_RULES_FILE"


class FeishuSyncToNasTool(BaseTool):
    name: str = Field(default="FeishuSyncToNasTool")
    description: str = Field(default="将飞书共享文件夹中的文件下载/导出并保存到 NAS 指定目录。")
    timeout: int = Field(default=3600, gt=0)
    tags: list[str] = Field(default_factory=lambda: ["connector", "feishu", "nas", "heavy"])
    downloader: FeishuSyncDownloader | None = Field(default=None, exclude=True)

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "folder_url": {
                    "type": "string",
                    "description": f"Feishu shared folder URL. If omitted, {FEISHU_FOLDER_URL_ENV} is used.",
                },
                "nas_dir": {
                    "type": "string",
                    "description": f"Allowed local NAS mount path under /mnt/ or /data/. If omitted, {FEISHU_SYNC_NAS_DIR_ENV} is used.",
                },
                "recursive": {"type": "boolean", "default": True},
                "overwrite": {"type": "boolean", "default": False},
            },
            "additionalProperties": False,
        }

    def get_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "downloaded": {"type": "integer"},
                "failed": {"type": "integer"},
                "skipped": {"type": "integer"},
                "nas_dir": {"type": "string"},
                "errors": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["success", "downloaded", "failed", "skipped", "nas_dir", "errors"],
            "additionalProperties": False,
        }

    def get_routing_capability(self) -> dict[str, Any]:
        capability = super().get_routing_capability()
        capability["supported_task_types"] = ["feishu_sync_to_nas"]
        capability["default_task_type"] = "feishu_sync_to_nas"
        capability["supported_tags"] = ["connector", "feishu", "nas", "heavy"]
        capability["max_concurrency"] = 2
        capability["supports_retry"] = False
        return capability

    async def _arun(self, payload: dict[str, Any], context: ContextStore | None = None) -> ToolResult:
        try:
            folder_url = self._resolve_string(payload, "folder_url", FEISHU_FOLDER_URL_ENV)
            nas_dir = self._resolve_string(payload, "nas_dir", FEISHU_SYNC_NAS_DIR_ENV)
            validate_feishu_folder_url_value(folder_url)
            validate_feishu_nas_dir_value(nas_dir)
            destination_rules = self._resolve_destination_rules(payload)
            recursive = self._as_bool(payload.get("recursive", True), default=True)
            overwrite = self._as_bool(payload.get("overwrite", False), default=False)
        except Exception as exc:
            return self.build_result(
                success=False,
                error=f"invalid FeishuSyncToNasTool parameters: {exc}",
                metadata={"payload_keys": sorted(payload.keys())},
            )

        try:
            result = await self._get_downloader().sync_folder_to_nas(
                folder_url=folder_url,
                nas_dir=nas_dir,
                recursive=recursive,
                overwrite=overwrite,
                destination_rules=destination_rules,
            )
        except FeishuConfigError as exc:
            return self.build_result(success=False, error=str(exc), metadata={"nas_dir": nas_dir})
        except Exception as exc:
            return self.build_result(
                success=False,
                error=f"Feishu sync to NAS failed: {exc}",
                metadata={"nas_dir": nas_dir, "exception_type": type(exc).__name__},
            )

        return self.build_result(
            success=result.success,
            output=self._result_output(result),
            error=None if result.success else "Feishu sync to NAS completed with failures",
            metadata={
                "nas_dir": result.nas_dir,
                "downloaded": result.downloaded,
                "failed": result.failed,
                "destination_rule_count": len(destination_rules),
            },
        )

    def _get_downloader(self) -> FeishuSyncDownloader:
        if self.downloader is None:
            self.downloader = FeishuSyncDownloader()
        return self.downloader

    def _resolve_string(self, payload: dict[str, Any], key: str, env_name: str) -> str:
        value = payload.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()

        env_value = os.getenv(env_name, "").strip()
        if env_value:
            return env_value

        raise ValueError(f"{key} is required or set {env_name}")

    def _resolve_destination_rules(self, payload: dict[str, Any]) -> list[DestinationRule]:
        raw_rules = payload.get("destination_rules")
        if raw_rules is None:
            raw_rules = self._load_destination_rules_from_env()
        if raw_rules in (None, ""):
            return []

        if isinstance(raw_rules, dict):
            raw_rules = raw_rules.get("destination_rules") or raw_rules.get("rules")
        if not isinstance(raw_rules, list):
            raise ValueError("destination rules must be a JSON array")

        return [DestinationRule.model_validate(rule) for rule in raw_rules]

    def _load_destination_rules_from_env(self) -> Any:
        rules_file = os.getenv(FEISHU_SYNC_DESTINATION_RULES_FILE_ENV, "").strip()
        if rules_file:
            path = Path(rules_file).expanduser()
            if not path.is_absolute():
                path = Path(__file__).resolve().parents[2] / path
            return json.loads(path.read_text(encoding="utf-8"))

        rules_json = os.getenv(FEISHU_SYNC_DESTINATION_RULES_ENV, "").strip()
        if rules_json:
            return json.loads(rules_json)
        return None

    def _as_bool(self, value: Any, *, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "y"}:
                return True
            if normalized in {"false", "0", "no", "n"}:
                return False
        return bool(value)

    def _result_output(self, result: SyncResult) -> dict[str, Any]:
        return {
            "success": result.success,
            "downloaded": result.downloaded,
            "failed": result.failed,
            "skipped": result.skipped,
            "nas_dir": result.nas_dir,
            "errors": result.errors,
        }
