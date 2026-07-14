from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any


DEFAULT_MAX_STRING_CHARS = 1200
DEFAULT_MAX_LIST_ITEMS = 20
DEFAULT_MAX_DICT_KEYS = 80
HEAVY_LIST_KEYS = {
    "items",
    "dept_plans",
    "weekly_items",
    "self_eval_items",
    "dept_plan_followups",
    "plan_followups",
    "last_week_plans",
    "this_week_completed",
    "candidate_matches",
    "weekly_pairs",
    "execution_history",
    "tool_calls",
}
HEAVY_TEXT_KEYS = {
    "dept_plan_completion_context_text",
    "plan_tracking_context_text",
    "weekly_blocker_context_text",
    "evidence_text",
    "raw_evidence_contexts",
    "raw_model_outputs",
}


def compact_observability_payload(
    value: Any,
    *,
    max_string_chars: int = DEFAULT_MAX_STRING_CHARS,
    max_list_items: int = DEFAULT_MAX_LIST_ITEMS,
    max_dict_keys: int = DEFAULT_MAX_DICT_KEYS,
) -> Any:
    return _compact(
        value,
        max_string_chars=max_string_chars,
        max_list_items=max_list_items,
        max_dict_keys=max_dict_keys,
        path=(),
    )


def _compact(
    value: Any,
    *,
    max_string_chars: int,
    max_list_items: int,
    max_dict_keys: int,
    path: tuple[str, ...],
) -> Any:
    if isinstance(value, str):
        return _compact_string(value, max_string_chars=max_string_chars)
    if isinstance(value, Mapping):
        return _compact_mapping(
            value,
            max_string_chars=max_string_chars,
            max_list_items=max_list_items,
            max_dict_keys=max_dict_keys,
            path=path,
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _compact_sequence(
            value,
            max_string_chars=max_string_chars,
            max_list_items=max_list_items,
            max_dict_keys=max_dict_keys,
            path=path,
        )
    return deepcopy(value)


def _compact_mapping(
    value: Mapping[str, Any],
    *,
    max_string_chars: int,
    max_list_items: int,
    max_dict_keys: int,
    path: tuple[str, ...],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for index, (key, item) in enumerate(value.items()):
        if index >= max_dict_keys:
            output["_truncated_keys"] = len(value) - max_dict_keys
            break
        key_text = str(key)
        if key_text in HEAVY_LIST_KEYS and isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            output[key_text] = _summarize_sequence(item, max_string_chars=max_string_chars)
            continue
        if key_text in HEAVY_TEXT_KEYS and isinstance(item, str):
            output[key_text] = _compact_string(item, max_string_chars=max_string_chars)
            continue
        output[key_text] = _compact(
            item,
            max_string_chars=max_string_chars,
            max_list_items=max_list_items,
            max_dict_keys=max_dict_keys,
            path=(*path, key_text),
        )
    return output


def _compact_sequence(
    value: Sequence[Any],
    *,
    max_string_chars: int,
    max_list_items: int,
    max_dict_keys: int,
    path: tuple[str, ...],
) -> list[Any]:
    compacted = [
        _compact(
            item,
            max_string_chars=max_string_chars,
            max_list_items=max_list_items,
            max_dict_keys=max_dict_keys,
            path=(*path, "[]"),
        )
        for item in list(value)[:max_list_items]
    ]
    if len(value) > max_list_items:
        compacted.append({"_truncated_items": len(value) - max_list_items})
    return compacted


def _compact_string(value: str, *, max_string_chars: int) -> str | dict[str, Any]:
    if len(value) <= max_string_chars:
        return value
    return {
        "_preview": value[:max_string_chars],
        "_truncated_chars": len(value) - max_string_chars,
        "_original_chars": len(value),
    }


def _summarize_sequence(value: Sequence[Any], *, max_string_chars: int) -> dict[str, Any]:
    return {
        "_observability_compacted": True,
        "_type": "list",
        "_count": len(value),
        "_preview": compact_observability_payload(list(value)[:3], max_string_chars=max_string_chars, max_list_items=3),
    }
