from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from app.connectors.feishu.drive import FeishuDriveClient
from app.connectors.feishu.exceptions import FeishuSyncError
from app.connectors.feishu.export import FeishuExportClient
from app.connectors.feishu.models import DestinationRule, FileItem, SyncResult
from app.connectors.feishu.utils import extract_folder_token, sanitize_filename, validate_nas_dir
from app.connectors.storage.nas import NasStorage

DIRECT_DOWNLOAD_TYPES = {"file", "pdf"}
EXPORT_TYPES = {"docx", "sheet", "bitable"}
UNSUPPORTED_WITH_CLEAR_REASON = {
    "shortcut_info": "shortcut file is not supported",
    "doc": "legacy doc is not supported",
}
DEFAULT_STATE_FILENAME = ".feishu_sync_state.json"


class _DestinationRoute:
    def __init__(self, *, source_prefix: str, target_root: str) -> None:
        self.source_parts = tuple(_normalize_remote_path_parts(source_prefix))
        self.target_root = target_root


class DestinationRouter:
    def __init__(self, *, default_root: str, rules: list[DestinationRule]) -> None:
        self.default_root = default_root
        self.routes = sorted(
            [
                _DestinationRoute(source_prefix=rule.source_prefix, target_root=rule.target_root)
                for rule in rules
            ],
            key=lambda route: len(route.source_parts),
            reverse=True,
        )

    def resolve_dir(self, remote_dir_path: str) -> str:
        remote_parts = tuple(_normalize_remote_path_parts(remote_dir_path))
        for route in self.routes:
            if self._matches(remote_parts, route.source_parts):
                return _join_local_path(route.target_root, remote_parts[len(route.source_parts) :])
        return _join_local_path(self.default_root, remote_parts)

    def _matches(self, remote_parts: tuple[str, ...], source_parts: tuple[str, ...]) -> bool:
        return len(source_parts) <= len(remote_parts) and remote_parts[: len(source_parts)] == source_parts


def _normalize_remote_path_parts(remote_path: str) -> list[str]:
    raw_path = str(remote_path or "").replace("\\", "/").strip("/")
    parts: list[str] = []
    for raw_part in raw_path.split("/"):
        cleaned_part = raw_part.strip()
        if cleaned_part:
            parts.append(sanitize_filename(cleaned_part))
    return parts


def _join_remote_path(parent_path: str, child_name: str) -> str:
    child_part = sanitize_filename(child_name)
    parent_parts = _normalize_remote_path_parts(parent_path)
    return "/".join([*parent_parts, child_part])


def _join_local_path(root: str, relative_parts: tuple[str, ...]) -> str:
    path = Path(root)
    for part in relative_parts:
        path = path / sanitize_filename(part)
    return str(path)


class FeishuSyncStateStore:
    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path
        self._data: dict[str, object] = {"version": 1, "folders": {}}
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return

        if self.state_path.exists():
            try:
                raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise FeishuSyncError(f"failed to load sync state from '{self.state_path}': {exc}") from exc
            self._data = self._normalize_state(raw)

        self._loaded = True

    def should_skip(self, folder_token: str, file_token: str) -> bool:
        self.load()
        folders = self._folders()
        folder_state = folders.get(folder_token)
        if not isinstance(folder_state, dict):
            return False
        files = folder_state.get("files")
        if not isinstance(files, dict):
            return False
        return file_token in files

    def mark_synced(
        self,
        *,
        folder_token: str,
        file_token: str,
        name: str,
        file_type: str,
        local_path: str,
    ) -> None:
        self.load()
        folders = self._folders()
        folder_state = folders.setdefault(folder_token, {})
        if not isinstance(folder_state, dict):
            raise FeishuSyncError("sync state folder entry is corrupted")
        files = folder_state.setdefault("files", {})
        if not isinstance(files, dict):
            raise FeishuSyncError("sync state file index is corrupted")
        files[file_token] = {
            "name": name,
            "type": file_type,
            "local_path": local_path,
            "synced_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save()

    def _normalize_state(self, raw: object) -> dict[str, object]:
        if not isinstance(raw, dict):
            raise FeishuSyncError("sync state file must contain a JSON object")

        folders = raw.get("folders", {})
        if not isinstance(folders, dict):
            raise FeishuSyncError("sync state file is missing a valid 'folders' object")

        normalized_folders: dict[str, object] = {}
        for folder_token, folder_state in folders.items():
            if not isinstance(folder_token, str) or not folder_token.strip():
                continue
            if not isinstance(folder_state, dict):
                continue
            files = folder_state.get("files", {})
            if not isinstance(files, dict):
                continue
            normalized_files: dict[str, object] = {}
            for file_token, record in files.items():
                if not isinstance(file_token, str) or not file_token.strip():
                    continue
                if isinstance(record, dict):
                    normalized_files[file_token] = record
            normalized_folders[folder_token] = {"files": normalized_files}

        version = raw.get("version", 1)
        return {"version": version, "folders": normalized_folders}

    def _folders(self) -> dict[str, object]:
        folders = self._data.get("folders")
        if not isinstance(folders, dict):
            folders = {}
            self._data["folders"] = folders
        return folders

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.state_path.with_name(f"{self.state_path.name}.part")
        payload = json.dumps(self._data, ensure_ascii=False, indent=2)
        temp_path.write_text(payload, encoding="utf-8")
        temp_path.replace(self.state_path)


class FeishuSyncDownloader:
    def __init__(
        self,
        *,
        drive_client: FeishuDriveClient | None = None,
        export_client: FeishuExportClient | None = None,
        storage: NasStorage | None = None,
        state_path: str | Path | None = None,
    ) -> None:
        self.drive_client = drive_client or FeishuDriveClient()
        self.export_client = export_client or FeishuExportClient(client=self.drive_client.client)
        self.storage = storage or NasStorage()
        self.state_path = Path(state_path).expanduser() if state_path is not None else None

    async def sync_folder_to_nas(
        self,
        folder_url: str,
        nas_dir: str,
        recursive: bool = True,
        overwrite: bool = False,
        destination_rules: list[DestinationRule | dict[str, str]] | None = None,
    ) -> SyncResult:
        validate_nas_dir(nas_dir)
        folder_token = extract_folder_token(folder_url)
        self.storage.ensure_dir(nas_dir)
        destination_router = self._build_destination_router(nas_dir, destination_rules)
        state_store = self._get_state_store(nas_dir)

        counters = {"downloaded": 0, "failed": 0, "skipped": 0}
        errors: list[str] = []
        await self._sync_folder_token(
            root_folder_token=folder_token,
            folder_token=folder_token,
            remote_dir_path="",
            local_dir=destination_router.resolve_dir(""),
            recursive=recursive,
            overwrite=overwrite,
            destination_router=destination_router,
            state_store=state_store,
            counters=counters,
            errors=errors,
        )
        return SyncResult(
            success=counters["failed"] == 0,
            downloaded=counters["downloaded"],
            failed=counters["failed"],
            skipped=counters["skipped"],
            nas_dir=nas_dir,
            errors=errors,
        )

    async def _sync_folder_token(
        self,
        *,
        root_folder_token: str,
        folder_token: str,
        remote_dir_path: str,
        local_dir: str,
        recursive: bool,
        overwrite: bool,
        destination_router: DestinationRouter,
        state_store: FeishuSyncStateStore,
        counters: dict[str, int],
        errors: list[str],
    ) -> None:
        try:
            items = await self.drive_client.list_folder_files(folder_token)
        except Exception as exc:
            counters["failed"] += 1
            errors.append(f"folder:{folder_token}: failed to list folder: {exc}")
            return

        for item in items:
            await self._sync_item(
                root_folder_token=root_folder_token,
                item=item,
                remote_dir_path=remote_dir_path,
                local_dir=local_dir,
                recursive=recursive,
                overwrite=overwrite,
                destination_router=destination_router,
                state_store=state_store,
                counters=counters,
                errors=errors,
            )

    async def _sync_item(
        self,
        *,
        root_folder_token: str,
        item: FileItem,
        remote_dir_path: str,
        local_dir: str,
        recursive: bool,
        overwrite: bool,
        destination_router: DestinationRouter,
        state_store: FeishuSyncStateStore,
        counters: dict[str, int],
        errors: list[str],
    ) -> None:
        item_type = item.type.lower()
        if item_type == "folder":
            child_remote_dir_path = _join_remote_path(remote_dir_path, item.name)
            child_dir = destination_router.resolve_dir(child_remote_dir_path)
            self.storage.ensure_dir(child_dir)
            if not recursive:
                counters["skipped"] += 1
                errors.append(self._format_error(item, "folder skipped because recursive=false"))
                return
            await self._sync_folder_token(
                root_folder_token=root_folder_token,
                folder_token=item.token,
                remote_dir_path=child_remote_dir_path,
                local_dir=child_dir,
                recursive=recursive,
                overwrite=overwrite,
                destination_router=destination_router,
                state_store=state_store,
                counters=counters,
                errors=errors,
            )
            return

        if item_type in UNSUPPORTED_WITH_CLEAR_REASON:
            counters["skipped"] += 1
            errors.append(self._format_error(item, UNSUPPORTED_WITH_CLEAR_REASON[item_type]))
            return

        if item_type not in DIRECT_DOWNLOAD_TYPES and item_type not in EXPORT_TYPES:
            counters["skipped"] += 1
            errors.append(self._format_error(item, f"unsupported type: {item.type}"))
            return

        try:
            if not overwrite and state_store.should_skip(root_folder_token, item.token):
                counters["skipped"] += 1
                return
            filename = self._resolve_filename(item)
            save_path = self.storage.resolve_save_path(local_dir, filename, overwrite)
            if item_type in DIRECT_DOWNLOAD_TYPES:
                await self.drive_client.download_file(item.token, save_path)
            else:
                await self.export_client.export_and_download(item.token, item_type, save_path)
            state_store.mark_synced(
                folder_token=root_folder_token,
                file_token=item.token,
                name=item.name,
                file_type=item.type,
                local_path=save_path,
            )
            counters["downloaded"] += 1
        except Exception as exc:
            counters["failed"] += 1
            errors.append(self._format_error(item, str(exc)))

    def _build_destination_router(
        self,
        nas_dir: str,
        destination_rules: list[DestinationRule | dict[str, str]] | None,
    ) -> DestinationRouter:
        rules: list[DestinationRule] = []
        for raw_rule in destination_rules or []:
            rule = raw_rule if isinstance(raw_rule, DestinationRule) else DestinationRule.model_validate(raw_rule)
            validate_nas_dir(rule.target_root)
            self.storage.ensure_dir(rule.target_root)
            rules.append(rule)
        return DestinationRouter(default_root=nas_dir, rules=rules)

    def _resolve_filename(self, item: FileItem) -> str:
        item_type = item.type.lower()
        filename = sanitize_filename(item.name)
        if item_type == "docx" and not filename.lower().endswith(".docx"):
            return f"{filename}.docx"
        if item_type in {"sheet", "bitable"} and not filename.lower().endswith(".xlsx"):
            return f"{filename}.xlsx"
        return filename

    def _format_error(self, item: FileItem, reason: str) -> str:
        return f"{item.name} ({item.type}): {reason}"

    def _get_state_store(self, nas_dir: str) -> FeishuSyncStateStore:
        state_path = self.state_path or (Path(nas_dir) / DEFAULT_STATE_FILENAME)
        return FeishuSyncStateStore(state_path=state_path)
