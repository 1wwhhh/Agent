from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import Any

from app.utils import configure_runtime_logger

LOGGER = configure_runtime_logger()


async def safe_observe(
    name: str,
    fn: Callable[..., Any],
    *args: Any,
    logger: logging.Logger | None = None,
    default: Any = None,
    **kwargs: Any,
) -> Any:
    try:
        result = fn(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result
    except Exception as exc:
        target_logger = logger or LOGGER
        target_logger.warning("observability operation failed: %s (%s)", name, exc, exc_info=exc)
        return default
