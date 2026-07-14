from __future__ import annotations

from app.state import LangGraphState


def coerce_langgraph_state(payload: LangGraphState | dict) -> LangGraphState:
    """把图执行输出规范化为 LangGraphState 实例。"""
    if isinstance(payload, LangGraphState):
        return payload
    return LangGraphState.model_validate(payload)
