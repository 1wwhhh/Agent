from __future__ import annotations

import contextvars
import json
import logging
import re
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

REQUEST_ID_CONTEXT: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agent_runtime_request_id",
    default=None,
)
LOGGER_NAME = "agent_runtime"
PROGRESS_LOGGER_NAME = "agent_runtime_progress"
ALLOWED_EVENTS = frozenset(
    {"start", "end", "execute", "success", "error", "retry", "timeout", "route", "clarification_required"}
)


def configure_runtime_logger() -> logging.Logger:
    """返回 Runtime 统一日志记录器，并确保日志级别和传播行为稳定一致。"""
    logger = logging.getLogger(LOGGER_NAME)
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.DEBUG)
    logger.propagate = True
    return logger


def configure_runtime_progress_logger() -> logging.Logger:
    """Return a console logger dedicated to human-readable runtime progress."""
    logger = logging.getLogger(PROGRESS_LOGGER_NAME)
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)
    logger.propagate = False

    has_handler = any(getattr(handler, "_agent_runtime_progress", False) for handler in logger.handlers)
    if not has_handler:
        handler = logging.StreamHandler(sys.stdout)
        handler._agent_runtime_progress = True  # type: ignore[attr-defined]
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    return logger


def bind_request_id(request_id: str) -> contextvars.Token[str | None]:
    """把 request_id 绑定到异步上下文中，使下游日志自动继承该标识。"""
    return REQUEST_ID_CONTEXT.set(request_id)


def clear_request_id(token: contextvars.Token[str | None]) -> None:
    """恢复上一个 request_id 绑定状态。"""
    REQUEST_ID_CONTEXT.reset(token)


def get_request_id() -> str | None:
    return REQUEST_ID_CONTEXT.get()


def runtime_log(
    *,
    layer: str,
    event: str,
    data: Mapping[str, Any] | None = None,
    latency_ms: float | None = None,
    level: int = logging.INFO,
    logger: logging.Logger | None = None,
) -> None:
    """输出结构化 Runtime 日志，并自动注入 request_id。"""
    if event not in ALLOWED_EVENTS:
        raise ValueError(f"unsupported runtime event: {event}")

    payload = {
        "request_id": get_request_id(),
        "layer": layer,
        "event": event,
        "data": dict(data or {}),
        "latency_ms": round(latency_ms, 3) if latency_ms is not None else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    target_logger = logger or configure_runtime_logger()
    target_logger.log(level, json.dumps(payload, ensure_ascii=True, default=_json_default))


def runtime_progress(
    *,
    step: str,
    status: str,
    detail: str | None = None,
    request_id: str | None = None,
    session_id: str | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Emit a concise, terminal-friendly progress message for the runtime."""
    target_logger = logger or configure_runtime_progress_logger()
    resolved_request_id = request_id or get_request_id() or "-"
    safe_step = _sanitize_progress_text(step, is_detail=False)
    safe_status = _sanitize_progress_text(status, is_detail=False)
    safe_detail = _sanitize_progress_text(detail, is_detail=True) if detail else None

    parts = [
        "[runtime]",
        f"request={resolved_request_id}",
    ]
    if session_id:
        parts.append(f"session={session_id}")
    parts.append(f"step={safe_step}")
    parts.append(f"status={safe_status}")
    if safe_detail:
        parts.append(f"detail={safe_detail}")
    target_logger.info(" | ".join(parts))


def _sanitize_progress_text(value: str | None, *, is_detail: bool) -> str:
    text = str(value or "")
    if not text:
        return ""
    if is_detail and any(
        marker in text
        for marker in (
            "Task Prompt:",
            "Additional Context:",
            "Runtime Time Context:",
            "Return the final answer using the required structured function call",
        )
    ):
        return "模型提示与上下文已隐藏"

    replacements = {
        "text_generate_tool": "文本生成",
        "llm_reason_tool": "推理生成",
        "emit_text_generation_output": "结构化输出",
        "emit_reasoning_output": "结构化输出",
        "Runtime Time Context": "时间上下文",
        "Task Prompt": "用户任务",
        "Additional Context": "附加上下文",
        "Function Calling": "工具调用",
        "JSON Schema": "结构化校验",
        "SIMPLE_TASK": "单步处理",
        "COMPLEX_TASK": "规划处理",
        "supervisor:决策详情": "请求判断",
        "supervisor": "请求判断",
        "simple_task": "单步处理",
        "planner": "任务规划",
        "parser": "任务解析",
        "queue": "任务队列",
        "executor": "任务执行",
        "aggregator": "结果聚合",
        "llm": "模型请求",
    }
    for raw, safe in replacements.items():
        text = text.replace(raw, safe)
    if is_detail and ("工具=" in text or "输出键=" in text):
        text = re.sub(r"工具=[^|]+(?:\|\s*输出键=[^|]+)?", "处理组件已选择", text)
        text = re.sub(r"输出键=[^|]+", "输出已设置", text)
    text = text.replace("LLM", "模型")
    text = re.sub(r"provider=[^|]+(?:\|\s*model=[^|]+)?", "模型请求已发送", text)
    text = re.sub(r"model=[^|]+", "模型已选择", text)
    return text


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, set):
        return sorted(value)
    return str(value)
