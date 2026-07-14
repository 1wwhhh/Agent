from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

from app.connectors.feishu.downloader import FeishuSyncDownloader
from app.connectors.feishu.models import DestinationRule, FileItem


class RecordingDriveClient:
    def __init__(self) -> None:
        self.downloads: list[tuple[str, str]] = []
        self.items_by_folder = {
            "root_token": [
                FileItem(name="A", token="folder_a", type="folder"),
                FileItem(name="B", token="folder_b", type="folder"),
                FileItem(name="root.txt", token="root_file", type="file"),
            ],
            "folder_a": [
                FileItem(name="合同", token="folder_contracts", type="folder"),
                FileItem(name="a-local.txt", token="a_file", type="file"),
            ],
            "folder_contracts": [
                FileItem(name="contract.txt", token="contract_file", type="file"),
            ],
            "folder_b": [
                FileItem(name="deep", token="folder_b_deep", type="folder"),
                FileItem(name="b.txt", token="b_file", type="file"),
            ],
            "folder_b_deep": [
                FileItem(name="deep.txt", token="deep_file", type="file"),
            ],
        }

    async def list_folder_files(self, folder_token: str) -> list[FileItem]:
        return self.items_by_folder[folder_token]

    async def download_file(self, file_token: str, save_path: str) -> None:
        self.downloads.append((file_token, save_path))


class UnusedExportClient:
    async def export_and_download(self, file_token: str, item_type: str, save_path: str) -> None:
        raise AssertionError("export should not be used in this test")


class RecordingStorage:
    def __init__(self) -> None:
        self.dirs: list[str] = []

    def ensure_dir(self, path: str) -> None:
        self.dirs.append(path)

    def resolve_save_path(self, dir_path: str, filename: str, overwrite: bool) -> str:
        return str(Path(dir_path) / filename)


def test_destination_rules_route_nested_feishu_folders_to_fixed_targets() -> None:
    workdir = Path("outputs") / "test_feishu_sync_destination_routing" / uuid4().hex
    workdir.mkdir(parents=True, exist_ok=True)
    drive_client = RecordingDriveClient()
    storage = RecordingStorage()
    downloader = FeishuSyncDownloader(
        drive_client=drive_client,
        export_client=UnusedExportClient(),
        storage=storage,
        state_path=workdir / "state.json",
    )

    result = asyncio.run(
        downloader.sync_folder_to_nas(
            folder_url="https://example.feishu.cn/drive/folder/root_token",
            nas_dir="/data/default",
            destination_rules=[
                DestinationRule(source_prefix="A", target_root="/data/a-root"),
                DestinationRule(source_prefix="A/合同", target_root="/data/contracts"),
                DestinationRule(source_prefix="B", target_root="/data/team-b"),
            ],
        )
    )

    assert result.success is True
    assert result.downloaded == 5
    assert dict(drive_client.downloads) == {
        "root_file": "/data/default/root.txt",
        "a_file": "/data/a-root/a-local.txt",
        "contract_file": "/data/contracts/contract.txt",
        "b_file": "/data/team-b/b.txt",
        "deep_file": "/data/team-b/deep/deep.txt",
    }
    assert "/data/contracts" in storage.dirs
    assert "/data/team-b/deep" in storage.dirs
