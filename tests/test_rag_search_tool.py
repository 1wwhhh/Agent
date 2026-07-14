from __future__ import annotations

import pytest

from app.prompts.task_planner import ToolDefinition, build_task_planner_prompt
from app.schemas.context import ContextStore, RuntimeContext
from app.schemas.llm import LLMFunctionCall, LLMRequest, LLMResponse
from app.tools.llm_client import LLMClient
from app.tools.rag_search import RAGSearchTool


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


class _RecordingAsyncClient:
    last_request_json: dict[str, object] | None = None

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self) -> "_RecordingAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, url: str, json: dict[str, object]) -> _FakeResponse:
        type(self).last_request_json = dict(json)
        return _FakeResponse(
            {
                "query": json["query"],
                "results": [
                    {
                        "doc_id": "doc-1",
                        "context_text": "Reimbursement must be submitted in OA.",
                        "score": 0.91,
                        "source_type": json.get("source_type"),
                    }
                ],
                "total": 1,
                "low_relevance": False,
                "top_score": 0.91,
                "threshold": 0.5,
            }
        )


class _SearchIntentLLMClient(LLMClient):
    def __init__(self, arguments: dict[str, object] | None = None, *, error: Exception | None = None) -> None:
        super().__init__()
        self.arguments = arguments or {}
        self.error = error
        self.requests: list[LLMRequest] = []

    async def _generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return LLMResponse(
            text="intent",
            model_name="stub-intent-model",
            function_call=LLMFunctionCall(
                tool_name="extract_rag_search_intent",
                arguments=self.arguments,
            ),
        )


class _LongContextAsyncClient(_RecordingAsyncClient):
    async def post(self, url: str, json: dict[str, object]) -> _FakeResponse:
        type(self).last_request_json = dict(json)
        return _FakeResponse(
            {
                "query": json["query"],
                "results": [
                    {
                        "doc_id": "doc-1",
                        "chunk_id": "chunk-1",
                        "context_text": "A" * 3500,
                        "score": 0.9,
                    },
                    {
                        "doc_id": "doc-2",
                        "chunk_id": "chunk-2",
                        "context_text": "B" * 3500,
                        "score": 0.85,
                    },
                ],
                "total": 2,
                "low_relevance": False,
                "top_score": 0.9,
                "threshold": 0.5,
            }
        )


def _assert_search_only_plan_tracking_request(result) -> None:
    assert _RecordingAsyncClient.last_request_json is not None
    assert "fetch_all" not in _RecordingAsyncClient.last_request_json
    assert "fetch_mode" not in _RecordingAsyncClient.last_request_json
    assert result.metadata["fetch_all_requested"] is False
    assert result.metadata["fetch_mode"] == "search"
    assert result.output is not None
    assert result.output["fetch_all_requested"] is False
    assert result.output["fetch_mode"] == "search"


def _build_tools() -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="rag_search_tool",
            description="Search company knowledge base through existing RAG /search API.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"},
                    "source_type": {"type": "string"},
                },
            },
            output_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "chunks": {"type": "array"},
                    "joined_context": {"type": "string"},
                    "summary": {"type": "string"},
                    "context_batches": {"type": "array"},
                },
            },
        )
    ]


@pytest.mark.asyncio
async def test_rag_search_tool_ignores_invalid_source_type(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.tools.rag_search.httpx.AsyncClient", _RecordingAsyncClient)
    tool = RAGSearchTool()

    result = await tool.arun(
        {
            "query": "公司报销流程是什么？",
            "top_k": 10,
            "source_type": "company_docs",
            "rag_base_url": "http://example.com",
        }
    )

    assert result.success is True
    assert _RecordingAsyncClient.last_request_json is not None
    assert _RecordingAsyncClient.last_request_json["source_type"] is None
    assert result.metadata["ignored_source_type"] == "company_docs"
    assert result.output is not None
    assert result.output["joined_context"]
    assert result.output["joined_context_is_preview"] is True
    assert result.output["summary"]
    assert result.output["context_batches"]


@pytest.mark.asyncio
async def test_rag_search_tool_preserves_valid_source_type(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.tools.rag_search.httpx.AsyncClient", _RecordingAsyncClient)
    tool = RAGSearchTool()

    result = await tool.arun(
        {
            "query": "公司报销流程是什么？",
            "top_k": 10,
            "source_type": "pdf",
            "rag_base_url": "http://example.com",
        }
    )

    assert result.success is True
    assert _RecordingAsyncClient.last_request_json is not None
    assert _RecordingAsyncClient.last_request_json["source_type"] == "pdf"
    assert "ignored_source_type" not in result.metadata


@pytest.mark.asyncio
async def test_rag_search_tool_extracts_folder_absolute_path_and_posts_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.tools.rag_search.httpx.AsyncClient", _RecordingAsyncClient)
    tool = RAGSearchTool()

    result = await tool.arun(
        {
            "query": "请分析这个文件夹\\9goo-nas\\部门\\考评记录2026年5月，三七计划计划中有没有落地成功",
            "top_k": 10,
            "rag_base_url": "http://example.com",
        }
    )

    assert result.success is True
    assert _RecordingAsyncClient.last_request_json is not None
    assert _RecordingAsyncClient.last_request_json["query"] == "三七计划计划中有没有落地成功"
    assert _RecordingAsyncClient.last_request_json["absolute_path"] == "\\9goo-nas\\部门\\考评记录2026年5月"
    assert "relative_path" not in _RecordingAsyncClient.last_request_json
    assert _RecordingAsyncClient.last_request_json["doc_id"] is None
    assert _RecordingAsyncClient.last_request_json["source_type"] is None
    _assert_search_only_plan_tracking_request(result)
    assert result.output is not None
    assert result.output["absolute_path"] == "\\9goo-nas\\部门\\考评记录2026年5月"
    assert result.output["parsed_search_query"]["doc_id"] is None
    assert result.output["parsed_search_query"]["source_type"] is None


@pytest.mark.asyncio
async def test_rag_search_tool_ignores_legacy_full_fetch_flags_for_plan_tracking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.tools.rag_search.httpx.AsyncClient", _RecordingAsyncClient)
    tool = RAGSearchTool()

    result = await tool.arun(
        {
            "query": "王浩上个月的每周计划都完成了吗，还有哪些没有完成",
            "top_k": 10,
            "source_type": "excel",
            "doc_id": "周报.xlsx",
            "fetch_all": True,
            "fetch_mode": "full_document",
            "rag_base_url": "http://example.com",
        }
    )

    assert result.success is True
    assert _RecordingAsyncClient.last_request_json is not None
    assert _RecordingAsyncClient.last_request_json["query"] == "王浩上个月的每周计划都完成了吗还有哪些没有完成"
    assert _RecordingAsyncClient.last_request_json["source_type"] == "excel"
    assert _RecordingAsyncClient.last_request_json["doc_id"] == "周报.xlsx"
    _assert_search_only_plan_tracking_request(result)


@pytest.mark.asyncio
async def test_rag_search_tool_prefers_bare_absolute_path_over_llm_doc_guess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.tools.rag_search.httpx.AsyncClient", _RecordingAsyncClient)
    client = _SearchIntentLLMClient(
        {
            "query": "考评记录2026年5月",
            "source_type": "word",
            "doc_id": "考评记录2026年5月",
            "relative_path": None,
            "absolute_path": None,
            "confidence": "high",
        }
    )
    tool = RAGSearchTool(client=client)

    result = await tool.arun(
        {
            "query": "\\9goo-nas\\部门\\考评记录2026年5月",
            "top_k": 10,
            "rag_base_url": "http://example.com",
        }
    )

    assert result.success is True
    assert _RecordingAsyncClient.last_request_json is not None
    assert _RecordingAsyncClient.last_request_json["query"] == "考评记录2026年5月"
    assert _RecordingAsyncClient.last_request_json["absolute_path"] == "\\9goo-nas\\部门\\考评记录2026年5月"
    assert "relative_path" not in _RecordingAsyncClient.last_request_json
    assert _RecordingAsyncClient.last_request_json["doc_id"] is None
    assert _RecordingAsyncClient.last_request_json["source_type"] is None
    assert _RecordingAsyncClient.last_request_json["fetch_all"] is True
    assert _RecordingAsyncClient.last_request_json["fetch_mode"] == "full_document"
    assert result.metadata["llm_search_intent"]["doc_id"] == "考评记录2026年5月"
    assert result.output is not None
    assert result.output["absolute_path"] == "\\9goo-nas\\部门\\考评记录2026年5月"
    assert result.output["fetch_all_requested"] is True


@pytest.mark.asyncio
async def test_rag_search_tool_uses_llm_search_intent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.tools.rag_search.httpx.AsyncClient", _RecordingAsyncClient)
    client = _SearchIntentLLMClient(
        {
            "query": "安全内容",
            "source_type": "word",
            "doc_id": "技术宝典.docx",
            "confidence": "high",
        }
    )
    tool = RAGSearchTool(client=client)

    result = await tool.arun(
        {
            "query": "请查询技术宝典.docx里的安全内容",
            "top_k": 10,
            "rag_base_url": "http://example.com",
        }
    )

    assert result.success is True
    assert client.requests
    assert client.requests[0].prompt_name == "rag_search_intent_prompt"
    assert _RecordingAsyncClient.last_request_json == {
        "query": "安全内容",
        "top_k": 10,
        "source_type": "word",
        "doc_id": "技术宝典.docx",
        "weight_m3": 0.5,
        "weight_zh": 0.4,
        "weight_sparse": 0.2,
    }
    assert result.metadata["search_intent_mode"] == "llm"
    assert result.metadata["llm_search_intent"]["doc_id"] == "技术宝典.docx"
    assert result.output is not None
    assert result.output["search_intent_mode"] == "llm"


@pytest.mark.asyncio
async def test_rag_search_tool_corrects_excel_doc_title_query_and_requests_full_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.tools.rag_search.httpx.AsyncClient", _RecordingAsyncClient)
    doc_id = "机器人项目 (2026_05_01-2026_05_29).xlsx"
    client = _SearchIntentLLMClient(
        {
            "query": "机器人项目",
            "source_type": "excel",
            "doc_id": doc_id,
            "confidence": "medium",
        }
    )
    tool = RAGSearchTool(client=client)

    result = await tool.arun(
        {
            "query": f"请找到 {doc_id}，对比每个人每个日期下的工作内容和计划是否有冲突",
            "top_k": 10,
            "rag_base_url": "http://example.com",
        }
    )

    assert result.success is True
    assert _RecordingAsyncClient.last_request_json == {
        "query": "对比每个人每个日期下的工作内容和计划是否有冲突",
        "top_k": 2000,
        "source_type": "excel",
        "doc_id": doc_id,
        "weight_m3": 0.5,
        "weight_zh": 0.4,
        "weight_sparse": 0.2,
        "fetch_all": True,
        "fetch_mode": "full_document",
    }
    assert result.metadata["query_adjustment"] == "doc_title_query_replaced"
    assert result.metadata["fetch_all_requested"] is True
    assert result.metadata["fetch_mode"] == "full_document"
    assert result.output is not None
    assert result.output["fetch_all_requested"] is True




@pytest.mark.asyncio
async def test_rag_search_tool_replaces_excel_completion_intent_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.tools.rag_search.httpx.AsyncClient", _RecordingAsyncClient)
    doc_id = "机器人项目 (2026_05_01-2026_05_29).xlsx"
    client = _SearchIntentLLMClient(
        {
            "query": "对比每个人每个日期下的工作内容和计划是否有冲突",
            "source_type": "excel",
            "doc_id": doc_id,
            "confidence": "high",
        }
    )
    tool = RAGSearchTool(client=client)

    result = await tool.arun(
        {
            "query": f"请找到 {doc_id}，请分析每个人写的下周的计划中，是否在下一周本周的工作中完成，有没有好几个人同时完成",
            "top_k": 10,
            "rag_base_url": "http://example.com",
        }
    )

    assert result.success is True
    assert _RecordingAsyncClient.last_request_json is not None
    assert _RecordingAsyncClient.last_request_json["query"] == (
        "请分析每个人写的下周的计划中，是否在下一周本周的工作中完成，有没有好几个人同时完成"
    )
    assert "冲突" not in _RecordingAsyncClient.last_request_json["query"]
    assert _RecordingAsyncClient.last_request_json["source_type"] == "excel"
    assert _RecordingAsyncClient.last_request_json["doc_id"] == doc_id
    _assert_search_only_plan_tracking_request(result)
    assert result.metadata["llm_search_intent"]["query"] == (
        "请分析每个人写的下周的计划中，是否在下一周本周的工作中完成，有没有好几个人同时完成"
    )


@pytest.mark.asyncio
async def test_rag_search_tool_replaces_excel_invented_overlap_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.tools.rag_search.httpx.AsyncClient", _RecordingAsyncClient)
    doc_id = "机器人项目 (2026_05_01-2026_05_29).xlsx"
    client = _SearchIntentLLMClient(
        {
            "query": "请分析每个人写的下周的计划中，是否在下一周本周的工作中完成，有没有好几个人同时完成",
            "source_type": "excel",
            "doc_id": doc_id,
            "confidence": "high",
        }
    )
    tool = RAGSearchTool(client=client)

    result = await tool.arun(
        {
            "query": f"请找到 {doc_id}，请分析每个人写的下周的计划中，是否在下一周本周的工作中完成",
            "top_k": 10,
            "rag_base_url": "http://example.com",
        }
    )

    assert result.success is True
    assert _RecordingAsyncClient.last_request_json is not None
    assert _RecordingAsyncClient.last_request_json["query"] == (
        "请分析每个人写的下周的计划中，是否在下一周本周的工作中完成"
    )
    assert "好几个人" not in _RecordingAsyncClient.last_request_json["query"]
    assert "同时完成" not in _RecordingAsyncClient.last_request_json["query"]
    assert _RecordingAsyncClient.last_request_json["source_type"] == "excel"
    assert _RecordingAsyncClient.last_request_json["doc_id"] == doc_id
    _assert_search_only_plan_tracking_request(result)
    assert result.metadata["llm_search_intent"]["query"] == (
        "请分析每个人写的下周的计划中，是否在下一周本周的工作中完成"
    )


@pytest.mark.asyncio
async def test_rag_search_tool_uses_rule_doc_id_when_llm_puts_filename_in_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.tools.rag_search.httpx.AsyncClient", _RecordingAsyncClient)
    doc_id = "机器人项目 (2026_05_01-2026_05_29).xlsx"
    client = _SearchIntentLLMClient(
        {
            "query": doc_id,
            "source_type": "excel",
            "doc_id": None,
            "confidence": "medium",
        }
    )
    tool = RAGSearchTool(client=client)

    result = await tool.arun(
        {
            "query": f"请找到 {doc_id}，请分析每个人写的下周的计划中，是否在下一周本周的工作中完成",
            "top_k": 10,
            "rag_base_url": "http://example.com",
        }
    )

    assert result.success is True
    assert _RecordingAsyncClient.last_request_json is not None
    assert _RecordingAsyncClient.last_request_json["query"] == (
        "请分析每个人写的下周的计划中，是否在下一周本周的工作中完成"
    )
    assert _RecordingAsyncClient.last_request_json["source_type"] == "excel"
    assert _RecordingAsyncClient.last_request_json["doc_id"] == doc_id
    _assert_search_only_plan_tracking_request(result)
    assert result.metadata["query_adjustment"] == "doc_title_query_replaced"
    assert result.metadata["parsed_search_query"]["doc_id"] == doc_id
    assert result.metadata["llm_search_intent"]["doc_id"] is None


@pytest.mark.asyncio
async def test_rag_search_tool_uses_runtime_user_input_when_planner_passes_only_filename(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.tools.rag_search.httpx.AsyncClient", _RecordingAsyncClient)
    doc_id = "机器人项目 (2026_05_01-2026_05_29).xlsx"
    user_input = f"请找到 {doc_id}，请分析每个人写的下周的计划中，是否在下一周本周的工作中完成"
    context = ContextStore(
        runtime=RuntimeContext(
            request_id="req_rag_search_filename_only",
            session_id="sess_rag_search_filename_only",
            user_input=user_input,
        )
    )
    client = _SearchIntentLLMClient(
        {
            "query": doc_id,
            "source_type": "excel",
            "doc_id": None,
            "confidence": "medium",
        }
    )
    tool = RAGSearchTool(client=client)

    result = await tool.arun(
        {
            "query": doc_id,
            "top_k": 10,
            "rag_base_url": "http://example.com",
        },
        context=context,
    )

    assert result.success is True
    assert _RecordingAsyncClient.last_request_json is not None
    assert _RecordingAsyncClient.last_request_json["query"] == (
        "请分析每个人写的下周的计划中，是否在下一周本周的工作中完成"
    )
    assert _RecordingAsyncClient.last_request_json["source_type"] == "excel"
    assert _RecordingAsyncClient.last_request_json["doc_id"] == doc_id
    _assert_search_only_plan_tracking_request(result)
    assert result.metadata["raw_query_source"] == "runtime_user_input"
    assert result.metadata["tool_query"] == doc_id
    assert result.metadata["raw_query"] == user_input
    assert result.metadata["parsed_search_query"]["doc_id"] == doc_id

@pytest.mark.asyncio
async def test_rag_search_tool_prefers_rule_query_when_llm_compresses_weekly_plan_intent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.tools.rag_search.httpx.AsyncClient", _RecordingAsyncClient)
    doc_id = "九工机器南京研发中心机器人项目__2__xlsx"
    raw_query = (
        f"请根据{doc_id},分析每个人写的下周的计划中，"
        "是否在下一周本周的工作中完成，哪些完成了哪些没完成"
    )
    client = _SearchIntentLLMClient(
        {
            "query": "下周计划 完成情况",
            "source_type": "excel",
            "doc_id": doc_id,
            "confidence": "high",
        }
    )
    tool = RAGSearchTool(client=client)

    result = await tool.arun(
        {
            "query": raw_query,
            "top_k": 10,
            "rag_base_url": "http://example.com",
        }
    )

    assert result.success is True
    assert _RecordingAsyncClient.last_request_json is not None
    assert _RecordingAsyncClient.last_request_json["query"] == (
        "分析每个人写的下周的计划中，是否在下一周本周的工作中完成，哪些完成了哪些没完成"
    )
    assert _RecordingAsyncClient.last_request_json["source_type"] == "excel"
    assert _RecordingAsyncClient.last_request_json["doc_id"] == doc_id
    _assert_search_only_plan_tracking_request(result)
    assert result.metadata["query_adjustment"] == "excel_rule_query_preferred_over_llm"
    assert result.metadata["llm_search_intent"]["query"] == "下周计划 完成情况"
    assert result.metadata["parsed_search_query"]["doc_id"] == doc_id


@pytest.mark.asyncio
async def test_rag_search_tool_uses_runtime_user_input_when_planner_passes_task_without_doc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.tools.rag_search.httpx.AsyncClient", _RecordingAsyncClient)
    doc_id = "九工机器南京研发中心机器人项目__2__xlsx"
    tool_query = "分析每个人写的下周的计划中，是否在下一周本周的工作中完成，哪些完成了哪些没完成"
    user_input = f"请根据{doc_id},{tool_query}"
    context = ContextStore(
        runtime=RuntimeContext(
            request_id="req_rag_search_task_only",
            session_id="sess_rag_search_task_only",
            user_input=user_input,
        )
    )
    client = _SearchIntentLLMClient(
        {
            "query": "下周计划 完成情况",
            "source_type": "excel",
            "doc_id": doc_id,
            "confidence": "high",
        }
    )
    tool = RAGSearchTool(client=client)

    result = await tool.arun(
        {
            "query": tool_query,
            "top_k": 10,
            "rag_base_url": "http://example.com",
        },
        context=context,
    )

    assert result.success is True
    assert _RecordingAsyncClient.last_request_json is not None
    assert _RecordingAsyncClient.last_request_json["query"] == tool_query
    assert _RecordingAsyncClient.last_request_json["source_type"] == "excel"
    assert _RecordingAsyncClient.last_request_json["doc_id"] == doc_id
    _assert_search_only_plan_tracking_request(result)
    assert result.metadata["raw_query_source"] == "runtime_user_input"
    assert result.metadata["tool_query"] == tool_query
    assert result.metadata["raw_query"] == user_input


@pytest.mark.asyncio
async def test_rag_search_tool_falls_back_to_rule_parser_when_llm_intent_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.tools.rag_search.httpx.AsyncClient", _RecordingAsyncClient)
    client = _SearchIntentLLMClient(error=RuntimeError("intent failed"))
    tool = RAGSearchTool(client=client)

    result = await tool.arun(
        {
            "query": "技术宝典中的安全内容",
            "top_k": 10,
            "rag_base_url": "http://example.com",
        }
    )

    assert result.success is True
    assert _RecordingAsyncClient.last_request_json is not None
    assert _RecordingAsyncClient.last_request_json["query"] == "安全"
    assert _RecordingAsyncClient.last_request_json["doc_id"] == "技术宝典"
    assert result.metadata["search_intent_mode"] == "rules_fallback"
    assert "llm_search_intent_error" in result.metadata


@pytest.mark.asyncio
async def test_rag_search_tool_parses_natural_language_query(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.tools.rag_search.httpx.AsyncClient", _RecordingAsyncClient)
    tool = RAGSearchTool()

    result = await tool.arun(
        {
            "query": "技术宝典中的安全内容",
            "top_k": 10,
            "rag_base_url": "http://example.com",
        }
    )

    assert result.success is True
    assert _RecordingAsyncClient.last_request_json is not None
    assert _RecordingAsyncClient.last_request_json == {
        "query": "安全",
        "top_k": 10,
        "source_type": None,
        "doc_id": "技术宝典",
        "weight_m3": 0.5,
        "weight_zh": 0.4,
        "weight_sparse": 0.2,
    }
    assert result.metadata["raw_query"] == "技术宝典中的安全内容"
    assert result.metadata["request_query"] == "安全"
    assert result.metadata["parsed_search_query"] == {
        "query": "安全",
        "scope_keyword": "技术宝典",
        "source_type": None,
        "doc_id": None,
        "relative_path": None,
        "absolute_path": None,
    }
    assert result.output is not None
    assert result.output["raw_query"] == "技术宝典中的安全内容"
    assert result.output["parsed_search_query"] == result.metadata["parsed_search_query"]


@pytest.mark.asyncio
async def test_rag_search_tool_does_not_treat_patent_subject_question_as_doc_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.tools.rag_search.httpx.AsyncClient", _RecordingAsyncClient)
    client = _SearchIntentLLMClient(
        {
            "query": "面的内容都",
            "source_type": None,
            "doc_id": "集装箱锁扭机夹具的专利是怎么写的",
            "relative_path": None,
            "absolute_path": None,
            "confidence": "medium",
        }
    )
    tool = RAGSearchTool(client=client)

    result = await tool.arun(
        {
            "query": "集装箱锁扭机夹具的专利是怎么写的，里面的内容都有什么",
            "top_k": 10,
            "rag_base_url": "http://example.com",
        }
    )

    assert result.success is True
    assert _RecordingAsyncClient.last_request_json is not None
    assert _RecordingAsyncClient.last_request_json["doc_id"] is None
    assert _RecordingAsyncClient.last_request_json["query"] == (
        "集装箱锁扭机夹具专利 内容 技术方案 权利要求 摘要 说明书"
    )
    assert "面的内容都" not in _RecordingAsyncClient.last_request_json["query"]
    assert result.metadata["query_adjustment"] == "rule_query_preferred_over_llm"


@pytest.mark.asyncio
async def test_rag_search_tool_keeps_explicit_patent_pdf_doc_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.tools.rag_search.httpx.AsyncClient", _RecordingAsyncClient)
    tool = RAGSearchTool()

    result = await tool.arun(
        {
            "query": "一种集装箱锁钮拆装用夹具及其使用方法.pdf里面的内容是什么",
            "top_k": 10,
            "rag_base_url": "http://example.com",
        }
    )

    assert result.success is True
    assert _RecordingAsyncClient.last_request_json is not None
    assert _RecordingAsyncClient.last_request_json["query"] == "内容是什么"
    assert _RecordingAsyncClient.last_request_json["source_type"] == "pdf"
    assert _RecordingAsyncClient.last_request_json["doc_id"] == "一种集装箱锁钮拆装用夹具及其使用方法.pdf"


@pytest.mark.asyncio
async def test_rag_search_tool_uses_short_scope_keyword_as_doc_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.tools.rag_search.httpx.AsyncClient", _RecordingAsyncClient)
    tool = RAGSearchTool()

    result = await tool.arun(
        {
            "query": "技术宝典中的安全内容",
            "top_k": 10,
            "rag_base_url": "http://example.com",
        }
    )

    assert result.success is True
    assert _RecordingAsyncClient.last_request_json is not None
    assert _RecordingAsyncClient.last_request_json["query"] == "安全"
    assert _RecordingAsyncClient.last_request_json["doc_id"] == "技术宝典"


@pytest.mark.asyncio
async def test_rag_search_tool_uses_parsed_source_type_and_doc_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.tools.rag_search.httpx.AsyncClient", _RecordingAsyncClient)
    tool = RAGSearchTool()

    result = await tool.arun(
        {
            "query": "查询 doc_id=abc123 中的安全内容",
            "rag_base_url": "http://example.com",
        }
    )

    assert result.success is True
    assert _RecordingAsyncClient.last_request_json is not None
    assert _RecordingAsyncClient.last_request_json["query"] == "安全"
    assert _RecordingAsyncClient.last_request_json["doc_id"] == "abc123"


@pytest.mark.asyncio
async def test_rag_search_tool_prefers_explicit_doc_id_over_scope_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.tools.rag_search.httpx.AsyncClient", _RecordingAsyncClient)
    tool = RAGSearchTool()

    result = await tool.arun(
        {
            "query": "技术宝典中的安全内容",
            "doc_id": "doc-explicit",
            "rag_base_url": "http://example.com",
        }
    )

    assert result.success is True
    assert _RecordingAsyncClient.last_request_json is not None
    assert _RecordingAsyncClient.last_request_json["query"] == "安全"
    assert _RecordingAsyncClient.last_request_json["doc_id"] == "doc-explicit"


@pytest.mark.asyncio
@pytest.mark.parametrize("source_type", ["", None])
async def test_rag_search_tool_accepts_empty_source_type(
    monkeypatch: pytest.MonkeyPatch,
    source_type: str | None,
) -> None:
    monkeypatch.setattr("app.tools.rag_search.httpx.AsyncClient", _RecordingAsyncClient)
    tool = RAGSearchTool()

    result = await tool.arun(
        {
            "query": "公司报销流程是什么？",
            "top_k": 10,
            "source_type": source_type,
            "rag_base_url": "http://example.com",
        }
    )

    assert result.success is True
    assert _RecordingAsyncClient.last_request_json is not None
    assert _RecordingAsyncClient.last_request_json["source_type"] is None
    assert "ignored_source_type" not in result.metadata


@pytest.mark.asyncio
async def test_rag_search_tool_batches_long_context_without_dropping_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.tools.rag_search.httpx.AsyncClient", _LongContextAsyncClient)
    monkeypatch.setenv("RAG_MAX_CONTEXT_CHUNKS", "1")
    monkeypatch.setenv("RAG_JOINED_CONTEXT_MAX_CHARS", "2000")
    tool = RAGSearchTool()

    result = await tool.arun(
        {
            "query": "申报",
            "top_k": 10,
            "rag_base_url": "http://example.com",
        }
    )

    assert result.success is True
    assert result.output is not None
    assert len(result.output["chunks"]) == 2
    assert len(result.output["context_batches"]) >= 2
    assert result.output["joined_context_is_preview"] is True
    assert all(batch["chars"] <= tool.joined_context_max_chars for batch in result.output["context_batches"])


def test_task_planner_prompt_mentions_source_type_constraints() -> None:
    prompt = build_task_planner_prompt(
        user_input="请根据公司知识库回答报销流程是什么",
        tools=_build_tools(),
        context_summary="None",
        planning_timestamp="2026-05-25T00:00:00+00:00",
    )

    assert "source_type" in prompt.user_prompt
    assert "pdf" in prompt.user_prompt
    assert "word" in prompt.user_prompt
    assert "ppt" in prompt.user_prompt
    assert "excel" in prompt.user_prompt
    assert "Do not use company_docs" in prompt.user_prompt
    assert "rag_batch_summarize_tool" in prompt.user_prompt
    assert "{{rag_summary.text}}" in prompt.user_prompt
