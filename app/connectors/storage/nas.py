from __future__ import annotations

from pathlib import Path

from app.connectors.feishu.utils import sanitize_filename


class NasStorageError(OSError):
    """Raised when NAS directory or path handling fails."""


class NasStorage:
    def ensure_dir(self, path: str) -> None:
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise NasStorageError(f"failed to create NAS directory '{path}': {exc}") from exc

    def resolve_save_path(self, dir_path: str, filename: str, overwrite: bool) -> str:
        safe_filename = sanitize_filename(filename)
        directory = Path(dir_path)
        self.ensure_dir(str(directory))

        candidate = directory / safe_filename
        if overwrite or not candidate.exists():
            return str(candidate)

        stem = candidate.stem
        suffix = candidate.suffix
        index = 1
        while True:
            next_candidate = directory / f"{stem}_{index}{suffix}"
            if not next_candidate.exists():
                return str(next_candidate)
            index += 1
