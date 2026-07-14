from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from app.connectors.feishu.utils import (
    FeishuUrlError,
    NasPathError,
    validate_feishu_folder_url_value,
    validate_feishu_nas_dir_value,
)
from app.schemas.task import TaskModel

FEISHU_SYNC_TOOL_NAME = "FeishuSyncToNasTool"
FEISHU_SYNC_TASK_TYPE = "feishu_sync_to_nas"
FEISHU_SYNC_TAGS = ["connector", "feishu", "nas", "heavy"]
FEISHU_FOLDER_URL_ENV = "FEISHU_FOLDER_URL"
FEISHU_SYNC_NAS_DIR_ENV = "FEISHU_SYNC_NAS_DIR"
_COLLAPSED_FEISHU_SYNC_TAG = "connector/feishu/nas/heavy"


@dataclass(frozen=True)
class FeishuSyncPreflightResult:
    ok: bool
    task: TaskModel | None = None
    response: dict[str, Any] | None = None
    warnings: list[str] | None = None


def is_feishu_sync_task(task: TaskModel) -> bool:
    return task.tool == FEISHU_SYNC_TOOL_NAME or task.task_type == FEISHU_SYNC_TASK_TYPE


def preflight_feishu_sync_task(task: TaskModel) -> FeishuSyncPreflightResult:
    if not is_feishu_sync_task(task):
        return FeishuSyncPreflightResult(ok=True, task=task, warnings=[])

    warnings: list[str] = []
    normalized_task = _normalize_feishu_sync_task(task, warnings=warnings)
    normalized_task = _apply_feishu_sync_env_defaults(normalized_task, warnings=warnings)
    missing_fields, invalid_reasons = _validate_feishu_sync_inputs(normalized_task)
    if missing_fields or invalid_reasons:
        return FeishuSyncPreflightResult(
            ok=False,
            response=_build_clarification_response(
                missing_fields=missing_fields,
                invalid_reasons=invalid_reasons,
            ),
            warnings=warnings,
        )

    return FeishuSyncPreflightResult(ok=True, task=normalized_task, warnings=warnings)


def preflight_feishu_sync_tasks(tasks: list[TaskModel]) -> FeishuSyncPreflightResult:
    normalized_tasks: list[TaskModel] = []
    warnings: list[str] = []

    for task in tasks:
        result = preflight_feishu_sync_task(task)
        warnings.extend(result.warnings or [])
        if not result.ok:
            return FeishuSyncPreflightResult(ok=False, response=result.response, warnings=warnings)
        normalized_tasks.append(result.task or task)

    return FeishuSyncPreflightResult(
        ok=True,
        task=None,
        response={"tasks": normalized_tasks},
        warnings=warnings,
    )


def _normalize_feishu_sync_task(task: TaskModel, *, warnings: list[str]) -> TaskModel:
    update: dict[str, Any] = {}
    if task.task_type != FEISHU_SYNC_TASK_TYPE:
        update["task_type"] = FEISHU_SYNC_TASK_TYPE
        warnings.append(f"{task.task_id}: normalized task_type to {FEISHU_SYNC_TASK_TYPE}")

    if task.tags != FEISHU_SYNC_TAGS:
        if task.tags == [_COLLAPSED_FEISHU_SYNC_TAG] or task.tags == [",".join(FEISHU_SYNC_TAGS)]:
            warnings.append(f"{task.task_id}: normalized collapsed Feishu sync tags")
        elif task.tags:
            warnings.append(f"{task.task_id}: normalized Feishu sync tags")
        update["tags"] = FEISHU_SYNC_TAGS

    if task.max_retry != 0:
        update["max_retry"] = 0
        warnings.append(f"{task.task_id}: normalized max_retry to 0")

    if task.tool != FEISHU_SYNC_TOOL_NAME:
        update["tool"] = FEISHU_SYNC_TOOL_NAME
        warnings.append(f"{task.task_id}: normalized tool to {FEISHU_SYNC_TOOL_NAME}")

    return task.model_copy(update=update) if update else task


def _apply_feishu_sync_env_defaults(task: TaskModel, *, warnings: list[str]) -> TaskModel:
    payload = dict(task.input) if isinstance(task.input, dict) else {}
    resolved_input = dict(payload)
    updated = False

    if not _string_value(payload.get("folder_url")):
        env_folder_url = _string_value(os.getenv(FEISHU_FOLDER_URL_ENV))
        if env_folder_url:
            resolved_input["folder_url"] = env_folder_url
            updated = True
            warnings.append(f"{task.task_id}: filled folder_url from {FEISHU_FOLDER_URL_ENV}")

    if not _string_value(payload.get("nas_dir")):
        env_nas_dir = _string_value(os.getenv(FEISHU_SYNC_NAS_DIR_ENV))
        if env_nas_dir:
            resolved_input["nas_dir"] = env_nas_dir
            updated = True
            warnings.append(f"{task.task_id}: filled nas_dir from {FEISHU_SYNC_NAS_DIR_ENV}")

    if not updated:
        return task
    return task.model_copy(update={"input": resolved_input})


def _validate_feishu_sync_inputs(task: TaskModel) -> tuple[list[str], dict[str, str]]:
    payload = task.input if isinstance(task.input, dict) else {}
    folder_url = _string_value(payload.get("folder_url"))
    nas_dir = _string_value(payload.get("nas_dir"))
    missing_fields: list[str] = []
    invalid_reasons: dict[str, str] = {}

    if not folder_url:
        missing_fields.append("folder_url")
    else:
        try:
            validate_feishu_folder_url_value(folder_url)
        except FeishuUrlError as exc:
            invalid_reasons["folder_url"] = str(exc)

    if not nas_dir:
        missing_fields.append("nas_dir")
    else:
        try:
            validate_feishu_nas_dir_value(nas_dir)
        except NasPathError as exc:
            invalid_reasons["nas_dir"] = str(exc)

    return missing_fields, invalid_reasons


def _build_clarification_response(*, missing_fields: list[str], invalid_reasons: dict[str, str]) -> dict[str, Any]:
    fields = [*missing_fields, *[field for field in invalid_reasons if field not in missing_fields]]
    if fields == ["folder_url"]:
        message = "Missing real Feishu shared folder link folder_url."
    elif fields == ["nas_dir"]:
        message = "Missing NAS target directory nas_dir."
    else:
        message = "A real Feishu folder_url and NAS nas_dir are required before sync can run."

    return {
        "success": False,
        "need_clarification": True,
        "message": message,
        "missing_fields": fields,
        "invalid_reasons": invalid_reasons,
    }


def _string_value(value: Any) -> str:
    return str(value or "").strip()
