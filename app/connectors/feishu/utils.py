from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse


class FeishuUrlError(ValueError):
    """Raised when a Feishu URL cannot be parsed safely."""


class NasPathError(ValueError):
    """Raised when a NAS path is outside the configured allowlist."""


_FOLDER_TOKEN_PATTERN = re.compile(r"^/drive/folder/([^/?#]+)")
_ILLEGAL_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_NAMES = {".", ".."}
_ALLOWED_NAS_PREFIXES = ("/mnt/", "/data/")
_FORBIDDEN_NAS_PATHS = {"/", "/etc", "/usr", "/var", "/root", "/home", "/tmp"}
_MAX_FILENAME_LENGTH = 180
FEISHU_FOLDER_URL_PLACEHOLDERS = {
    "FEISHU_FOLDER_URL",
    "$FEISHU_FOLDER_URL",
    "${FEISHU_FOLDER_URL}",
}
FEISHU_SYNC_NAS_DIR_PLACEHOLDERS = {
    "FEISHU_SYNC_NAS_DIR",
    "$FEISHU_SYNC_NAS_DIR",
    "${FEISHU_SYNC_NAS_DIR}",
}


def extract_folder_token(folder_url: str) -> str:
    raw_url = str(folder_url or "").strip()
    if not raw_url:
        raise FeishuUrlError("folder_url is required")

    parsed = urlparse(raw_url)
    host = parsed.netloc.lower()
    if parsed.scheme not in {"http", "https"} or not (
        host.endswith(".feishu.cn") or host.endswith(".larksuite.com")
    ):
        raise FeishuUrlError("folder_url must be a Feishu or LarkSuite folder link")

    match = _FOLDER_TOKEN_PATTERN.match(parsed.path)
    if match is None:
        raise FeishuUrlError("folder_url must match /drive/folder/<folder_token>")

    token = match.group(1).strip()
    if not token:
        raise FeishuUrlError("folder_url does not contain a folder token")
    return token


def is_feishu_folder_url_placeholder(value: str | None) -> bool:
    return str(value or "").strip() in FEISHU_FOLDER_URL_PLACEHOLDERS


def is_feishu_nas_dir_placeholder(value: str | None) -> bool:
    return str(value or "").strip() in FEISHU_SYNC_NAS_DIR_PLACEHOLDERS


def validate_feishu_folder_url_value(folder_url: str) -> None:
    raw_url = str(folder_url or "").strip()
    if not raw_url:
        raise FeishuUrlError("folder_url is required")
    if is_feishu_folder_url_placeholder(raw_url):
        raise FeishuUrlError("folder_url must be a real Feishu folder link, not an environment variable placeholder")
    extract_folder_token(raw_url)


def validate_feishu_nas_dir_value(nas_dir: str) -> None:
    raw_path = str(nas_dir or "").strip()
    if not raw_path:
        raise NasPathError("nas_dir is required")
    if is_feishu_nas_dir_placeholder(raw_path):
        raise NasPathError("nas_dir must be a real NAS path, not an environment variable placeholder")
    validate_nas_dir(raw_path)


def sanitize_filename(name: str) -> str:
    value = str(name or "").strip()
    value = _ILLEGAL_FILENAME_CHARS.sub("_", value)
    value = re.sub(r"\s+", " ", value).strip(" .")
    if not value or value in _RESERVED_NAMES:
        value = "untitled"

    if len(value) > _MAX_FILENAME_LENGTH:
        suffix = ""
        if "." in value and not value.startswith("."):
            stem, ext = value.rsplit(".", 1)
            suffix = f".{ext[:20]}"
            value = stem[: max(1, _MAX_FILENAME_LENGTH - len(suffix))] + suffix
        else:
            value = value[:_MAX_FILENAME_LENGTH]
    return value or "untitled"


def validate_nas_dir(nas_dir: str) -> None:
    raw_path = str(nas_dir or "").strip()
    if not raw_path:
        raise NasPathError("nas_dir is required")

    path_parts = Path(raw_path).parts
    if any(part == ".." for part in path_parts):
        raise NasPathError("nas_dir must not contain parent-directory traversal")

    raw_posix = raw_path.replace("\\", "/")
    raw_is_absolute_posix = raw_posix.startswith("/")
    normalized_raw = "/" + raw_posix.lstrip("/")
    normalized_raw = re.sub(r"/+", "/", normalized_raw).rstrip("/") or "/"
    if raw_is_absolute_posix and normalized_raw in _FORBIDDEN_NAS_PATHS:
        raise NasPathError(f"nas_dir '{nas_dir}' is not allowed")

    try:
        resolved_posix = Path(raw_path).expanduser().resolve(strict=False).as_posix()
    except Exception as exc:
        raise NasPathError(f"nas_dir '{nas_dir}' cannot be resolved: {exc}") from exc

    normalized_candidates = {resolved_posix if resolved_posix.endswith("/") else f"{resolved_posix}/"}
    if raw_is_absolute_posix:
        normalized_candidates.add(normalized_raw if normalized_raw.endswith("/") else f"{normalized_raw}/")
    if any(candidate.startswith(_ALLOWED_NAS_PREFIXES) for candidate in normalized_candidates):
        return

    raise NasPathError("nas_dir must be under /mnt/ or /data/")
