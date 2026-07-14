from __future__ import annotations

import pytest

from app.executor.task_executor import TaskExecutor
from app.queue.task_queue import TaskQueue
from app.router.task_router import TaskRouter
from app.schemas.context import ContextStore, RuntimeContext
from app.schemas.llm import LLMFunctionCall, LLMRequest, LLMResponse
from app.schemas.task import TaskModel, TaskStatus
from app.tools.llm_client import LLMClient
from app.tools.text_generate import TextGenerateTool


class StubStructuredLLMClient(LLMClient):
    def __init__(self) -> None:
        super().__init__(timeout_seconds=30, model_name="stub-llm", model_version="test-v1")

    async def _generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            text='{"tool_name":"emit_text_generation_output","arguments":{"text":"ok","audience":"通用受众","style":"正式报告"}}',
            model_name=self.model_name,
            model_version=self.model_version,
            request_id=request.request_id,
            session_id=request.session_id,
            trace_id=request.trace_id,
            prompt_name=request.prompt_name,
            prompt_version=request.prompt_version,
            function_call=LLMFunctionCall(
                tool_name="emit_text_generation_output",
                arguments={"text": "ok", "audience": "通用受众", "style": "正式报告"},
            ),
            raw_response={"provider": "stub"},
        )


def _build_context() -> ContextStore:
    return ContextStore(
        runtime=RuntimeContext(
            request_id="req_test",
            session_id="sess_test",
            user_input="test",
        )
    )


@pytest.mark.asyncio
async def test_llm_tool_renders_output_key_template_from_context():
    context = _build_context()
    context.set_task_result(
        "analysis_result",
        {
            "text": "# 人工智能（AI）发展趋势分析",
            "summary": "summary",
            "key_points": [],
        },
    )
    tool = TextGenerateTool(client=StubStructuredLLMClient())

    result = await tool.arun(
        payload={
            "prompt": "请生成报告。\n\nAdditional Context:\n{{analysis_result.text}}",
            "style": "正式报告",
            "audience": "通用受众",
        },
        context=context,
    )

    assert result.success is True
    rendered_prompt = result.metadata["request"]["prompt"]
    assert "{{analysis_result.text}}" not in rendered_prompt
    assert "# 人工智能（AI）发展趋势分析" in rendered_prompt


def test_executor_injects_dependency_output_into_downstream_context():
    context = _build_context()
    queue = TaskQueue(context=context)
    router = TaskRouter()
    executor = TaskExecutor(context=context, queue=queue, router=router)

    upstream_task = TaskModel(
        task_id="topic_analysis",
        task_name="topic_analysis",
        description="analyze topic",
        tool="llm_reason_tool",
        input={"prompt": "analyze"},
        output_key="analysis_result",
        status=TaskStatus.SUCCESS,
    )
    downstream_task = TaskModel(
        task_id="report_generation",
        task_name="report_generation",
        description="generate report",
        tool="text_generate_tool",
        input={"prompt": "请生成正式报告"},
        output_key="final_report",
        depends_on=["topic_analysis"],
        status=TaskStatus.PENDING,
    )

    context.register_task(upstream_task)
    context.register_task(downstream_task)
    context.set_task_result(
        "analysis_result",
        {
            "text": "# 人工智能（AI）发展趋势分析",
            "summary": "summary",
            "key_points": [],
        },
    )

    payload = executor._build_tool_payload(downstream_task)

    assert "context" in payload
    assert "# 人工智能（AI）发展趋势分析" in str(payload["context"])
