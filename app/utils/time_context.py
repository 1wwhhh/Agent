from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.schemas.context import RuntimeContext


_TIMEZONE_METADATA_KEYS = ("client_timezone", "timezone", "tz")


def build_runtime_time_context(runtime: RuntimeContext | None) -> str:
    """Render the current request time as model-visible runtime context."""
    if runtime is None:
        return ""

    current_utc = _as_utc(runtime.timestamp)
    timezone_name = _metadata_timezone_name(runtime.metadata)
    local_timezone, local_timezone_name = _resolve_timezone(timezone_name)
    current_local = current_utc.astimezone(local_timezone)

    return "\n".join(
        [
            "Runtime Time Context:",
            f"- Current UTC datetime: {current_utc.isoformat()}",
            f"- Client timezone: {local_timezone_name}",
            f"- Current client-local datetime: {current_local.isoformat()}",
            f"- Current client-local date: {current_local.date().isoformat()}",
            (
                "- Current client-local date in Chinese: "
                f"{current_local.year}年{current_local.month}月{current_local.day}日"
            ),
            (
                "- Guidance: when the user asks about 今天、今日、现在、today, "
                "or the current date/time, use the client-local date/time above as authoritative "
                "unless the user specifies another timezone."
            ),
        ]
    )


def merge_runtime_time_context(context_block: Any, runtime: RuntimeContext | None) -> str:
    """Append runtime time context to an existing prompt context block."""
    base_context = _string_context(context_block)
    time_context = build_runtime_time_context(runtime)
    if not time_context:
        return base_context
    if base_context == "None":
        return time_context
    return f"{base_context}\n\n{time_context}"


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _metadata_timezone_name(metadata: dict[str, Any]) -> str:
    for key in _TIMEZONE_METADATA_KEYS:
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return "UTC"


def _resolve_timezone(timezone_name: str) -> tuple[ZoneInfo | timezone, str]:
    if timezone_name.upper() == "UTC":
        return timezone.utc, "UTC"
    try:
        return ZoneInfo(timezone_name), timezone_name
    except (ZoneInfoNotFoundError, ValueError):
        return timezone.utc, "UTC"


def _string_context(context_block: Any) -> str:
    if context_block is None:
        return "None"
    context_text = str(context_block).strip()
    return context_text or "None"
