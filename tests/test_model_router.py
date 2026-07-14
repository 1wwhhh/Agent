from __future__ import annotations

import pytest

from app.adapters import ModelRouter
from app.schemas.model import ModelConfig, ModelProvider, RuntimeLLMConfig
from app.tools.llm_client import LLMClient

pytest.importorskip("fastapi")

from app.api.runtime import build_runtime_components


def _build_runtime_config() -> RuntimeLLMConfig:
    return RuntimeLLMConfig(
        primary=ModelConfig(
            provider=ModelProvider.DEEPSEEK,
            api_key="deepseek-test-key",
            base_url="https://api.deepseek.com",
            model_name="deepseek-v4-flash",
            timeout_seconds=30,
            max_retries=2,
        ),
        fallbacks=[
            ModelConfig(
                provider=ModelProvider.QWEN,
                api_key="qwen-test-key",
                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                model_name="qwen-plus",
                timeout_seconds=30,
                max_retries=1,
            )
        ],
    )


def test_model_router_builds_single_runtime_visible_llm_client():
    client = ModelRouter().build_client(_build_runtime_config())

    assert isinstance(client, LLMClient)
    assert [provider.provider_name for provider in client._provider_states] == ["deepseek", "qwen"]


@pytest.mark.asyncio
async def test_runtime_components_are_model_agnostic_and_register_tools():
    client = ModelRouter().build_client(_build_runtime_config())
    components = await build_runtime_components(client=client)

    try:
        assert isinstance(components.client, LLMClient)
        assert components.repair_llm_client is components.client
        assert components.supervisor_agent.client is components.client
        assert components.planner_agent.client is components.client
        assert await components.router.list_tools(enabled_only=True) == [
            "rag_search_tool",
            "ab_case_search_tool",
            "rag_batch_summarize_tool",
            "llm_reason_tool",
            "text_generate_tool",
            "classify_weekly_blockers",
            "judge_weekly_blocker_trace",
            "FeishuSyncToNasTool",
            "compare_dept_plan_completion",
            "query_opl_issues",
            "monthly_department_analysis",
            "compare_weekly_plan_done",
            "query_weekly_reports",
        ]
    finally:
        await components.client.aclose()
