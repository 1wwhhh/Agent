from __future__ import annotations

import json

import pytest

from app.schemas.context import ContextStore, RuntimeContext
from app.schemas.llm import LLMRequest, LLMResponse
from app.tools.llm_client import LLMClient
from app.tools.rag_batch_summarize import (
    JSON_FALLBACK_WITH_RAW_CONTEXT_MISSING_INFORMATION,
    JSON_FALLBACK_WITH_RAW_CONTEXT_MODE,
    RAGBatchSummarizeTool,
    RAW_CONTEXT_FALLBACK_MISSING_INFORMATION,
    RAW_CONTEXT_FALLBACK_MODE,
)


class StubEvidenceExtractionClient(LLMClient):
    def __init__(self, responses: dict[str, str] | None = None) -> None:
        super().__init__(timeout_seconds=30, model_name="stub-evidence-extraction-client")
        self.responses = responses or {}
        self.requests: list[LLMRequest] = []
        self.batch_inputs: list[dict[str, object]] = []

    async def _generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        request_payload = _extract_batch_input(request.messages[1].content)
        self.batch_inputs.append(request_payload)
        batch_id = str(request_payload["batch_id"])
        return LLMResponse(
            text=self.responses.get(batch_id, ""),
            model_name=self.model_name,
            prompt_name=request.prompt_name,
            prompt_version=request.prompt_version,
        )


def _extract_batch_input(prompt: str) -> dict[str, object]:
    start = prompt.index("{")
    end = prompt.rindex("}") + 1
    return json.loads(prompt[start:end])


def _build_context(*, short: bool) -> ContextStore:
    context = ContextStore(
        runtime=RuntimeContext(
            request_id="req_rag_batch_summary",
            session_id="sess_rag_batch_summary",
            user_input="今年公司都做了哪些申报",
        )
    )
    if short:
        chunks = [
            {
                "chunk_id": "chunk-a",
                "context_text": "2024-05-01 起，申报材料必须在 30 天内提交。",
                "relative_path": "制度A.pdf",
                "score": 0.93,
            },
            {
                "id": "row-2",
                "text": "金额大于 10000 元时，需要部门负责人审批。",
                "doc_id": "报销制度",
                "final_score": 0.88,
            },
            {
                "source_chunk_id": "src-3",
                "content": "流程顺序为：申请、复核、审批、归档。",
                "source": "流程手册",
                "rerank_score": 0.81,
            },
        ]
    else:
        chunks = [
            {
                "chunk_id": f"chunk-{index}",
                "context_text": f"chunk-{index} 长上下文内容 " + "A" * 1600,
                "relative_path": f"doc-{index}.pdf",
                "score": 0.9 - index * 0.01,
            }
            for index in range(1, 7)
        ]

    context.task_results["rag_context"] = {
        "summary": "RAG 检索完成。",
        "chunks": chunks,
        "context_batches": [
            {
                "batch_id": "batch_1",
                "chunk_ids": ["chunk-1", "chunk-2"],
                "source_chunk_ids": ["chunk-1", "chunk-2"],
                "joined_context": "[chunk_id=chunk-1] 2024-05-01 后必须 30 天内提交。\n[chunk_id=chunk-2] 金额大于 10000 元需要审批。",
                "source_count": 2,
                "chars": 90,
            },
            {
                "batch_id": "batch_2",
                "chunk_ids": ["chunk-3", "chunk-4"],
                "source_chunk_ids": ["chunk-3", "chunk-4"],
                "joined_context": "[chunk_id=chunk-3] 流程顺序为申请、复核、审批、归档。",
                "source_count": 2,
                "chars": 30,
            },
        ],
        "low_relevance": False,
        "top_score": 0.91,
        "threshold": 0.5,
    }
    return context


def _build_financial_context() -> ContextStore:
    context = ContextStore(
        runtime=RuntimeContext(
            request_id="req_rag_batch_financial",
            session_id="sess_rag_batch_financial",
            user_input="财务报表的数据有哪些",
        )
    )
    chunks = [
        {
            "chunk_id": f"fin-{index}",
            "context_text": "财务报表长上下文 " + "A" * 1500,
            "relative_path": "01财务报表/2022年财务报表.pdf",
            "score": 0.91 - index * 0.01,
        }
        for index in range(1, 7)
    ]
    context.task_results["rag_context"] = {
        "summary": "RAG 检索完成。",
        "chunks": chunks,
        "context_batches": [
            {
                "batch_id": "batch_1",
                "chunk_ids": ["fin-1", "fin-2"],
                "source_chunk_ids": ["src-fin-1", "src-fin-2"],
                "joined_context": (
                    "[1] source=01财务报表/2022年财务报表.pdf chunk_id=fin-1\n"
                    "财务报表 利润表 编制单位：九工机器（上海）有限公司 2022年12期 单位：元\n"
                    "营业收入 10,739,842.03\n"
                    "营业成本 5,146,519.97\n"
                    "管理费用 11,000,097.69"
                ),
                "source_count": 2,
                "chars": 180,
            },
            {
                "batch_id": "batch_2",
                "chunk_ids": ["fin-3", "fin-4"],
                "source_chunk_ids": ["src-fin-3", "src-fin-4"],
                "joined_context": (
                    "[2] source=01财务报表/2022年财务报表.pdf chunk_id=fin-3\n"
                    "净利润 -5,418,627.21\n"
                    "现金净增加额 -25,037.71\n"
                    "期末现金余额 5,378.10"
                ),
                "source_count": 2,
                "chars": 120,
            },
        ],
        "low_relevance": False,
        "top_score": 0.91,
        "threshold": 0.5,
    }
    return context


@pytest.mark.asyncio
async def test_rag_batch_summarize_short_context_passthrough_does_not_call_llm() -> None:
    client = StubEvidenceExtractionClient()
    tool = RAGBatchSummarizeTool(client=client)
    context = _build_context(short=True)

    result = await tool.arun({"query": "今年公司都做了哪些申报", "rag_output_key": "rag_context"}, context=context)

    assert result.success is True
    assert client.requests == []
    assert result.output is not None
    assert result.output["extraction_mode"] == "direct_context_passthrough"
    assert result.output["metadata"]["llm_called"] is False
    assert result.metadata["llm_called"] is False
    assert result.output["metadata"]["chunk_count"] == 3
    expected_chars = sum(
        len(chunk.get("context_text") or chunk.get("text") or "")
        for chunk in context.task_results["rag_context"]["chunks"]
    )
    assert result.output["metadata"]["total_context_chars"] == expected_chars
    assert result.output["evidence_chunk_ids"] == ["chunk-a", "row-2", "src-3"]
    assert result.output["irrelevant_chunk_ids"] == []
    assert result.output["missing_information"] == []
    assert result.output["confidence"] == "medium"
    assert result.output["batch_summaries"][0]["batch_id"] == "direct_context_passthrough"
    assert "2024-05-01" in result.output["text"]
    assert "30 天" in result.output["text"]
    assert "10000" in result.output["text"]
    assert "流程顺序" in result.output["text"]


@pytest.mark.asyncio
async def test_rag_batch_summarize_long_context_extracts_evidence_and_filters_ids() -> None:
    client = StubEvidenceExtractionClient(
        responses={
            "batch_1": json.dumps(
                {
                    "answer_facts": ["2024-05-01 后必须 30 天内提交。"],
                    "key_points": ["金额大于 10000 元需要审批。"],
                    "evidence_chunk_ids": ["chunk-1", "chunk-2", "invented-evidence"],
                    "irrelevant_chunk_ids": ["chunk-1", "chunk-2", "invented-irrelevant"],
                    "missing_information": [],
                    "confidence": "high",
                },
                ensure_ascii=False,
            ),
            "batch_2": json.dumps(
                {
                    "answer_facts": ["流程顺序为申请、复核、审批、归档。"],
                    "key_points": [],
                    "evidence_chunk_ids": ["chunk-3"],
                    "irrelevant_chunk_ids": ["chunk-4"],
                    "missing_information": ["未检索到责任人说明。"],
                    "confidence": "medium",
                },
                ensure_ascii=False,
            ),
        }
    )
    tool = RAGBatchSummarizeTool(client=client)
    context = _build_context(short=False)

    result = await tool.arun({"query": "今年公司都做了哪些申报", "rag_output_key": "rag_context"}, context=context)

    assert result.success is True
    assert len(client.batch_inputs) == 2
    first_input = client.batch_inputs[0]
    assert set(first_input) == {"query", "batch_id", "chunk_ids", "source_chunk_ids", "batch_content"}
    assert first_input["query"] == "今年公司都做了哪些申报"
    assert first_input["batch_id"] == "batch_1"
    assert first_input["chunk_ids"] == ["chunk-1", "chunk-2"]
    assert first_input["source_chunk_ids"] == ["chunk-1", "chunk-2"]
    assert result.output is not None
    assert result.output["extraction_mode"] == "evidence_extraction"
    assert "invented-evidence" not in result.output["evidence_chunk_ids"]
    assert "invented-irrelevant" not in result.output["irrelevant_chunk_ids"]
    assert "chunk-1" in result.output["evidence_chunk_ids"]
    assert "chunk-2" in result.output["evidence_chunk_ids"]
    assert "chunk-1" not in result.output["irrelevant_chunk_ids"]
    assert "chunk-2" not in result.output["irrelevant_chunk_ids"]
    assert "chunk-4" in result.output["irrelevant_chunk_ids"]
    assert result.output["confidence"] == "medium"
    assert "Evidence Extraction" in result.output["text"]
    assert "2024-05-01" in result.output["text"]
    assert "未检索到责任人说明" in result.output["text"]


@pytest.mark.asyncio
async def test_rag_batch_summarize_generic_evidence_rebatches_single_chunk_batches_by_eight() -> None:
    client = StubEvidenceExtractionClient(
        responses={
            "batch_1": json.dumps(
                {
                    "answer_facts": ["前 8 个 chunk 已合批分析。"],
                    "key_points": [],
                    "evidence_chunk_ids": ["chunk-1", "chunk-8"],
                    "irrelevant_chunk_ids": [],
                    "missing_information": [],
                    "confidence": "high",
                },
                ensure_ascii=False,
            ),
            "batch_2": json.dumps(
                {
                    "answer_facts": ["剩余 2 个 chunk 已合批分析。"],
                    "key_points": [],
                    "evidence_chunk_ids": ["chunk-9", "chunk-10"],
                    "irrelevant_chunk_ids": [],
                    "missing_information": [],
                    "confidence": "high",
                },
                ensure_ascii=False,
            ),
        }
    )
    tool = RAGBatchSummarizeTool(client=client)
    context = ContextStore(
        runtime=RuntimeContext(
            request_id="req_generic_rebatch",
            session_id="sess_generic_rebatch",
            user_input="分析这些专利资料",
        )
    )
    chunks = [
        {
            "chunk_id": f"chunk-{index}",
            "context_text": f"第 {index} 个专利资料片段 " + "A" * 900,
            "relative_path": f"patent-{index}.pdf",
        }
        for index in range(1, 11)
    ]
    context.task_results["rag_context"] = {
        "summary": "RAG 检索完成。",
        "chunks": chunks,
        "context_batches": [
            {
                "batch_id": f"batch_{index}",
                "chunk_ids": [f"chunk-{index}"],
                "source_chunk_ids": [f"chunk-{index}"],
                "joined_context": f"[chunk_id=chunk-{index}] 第 {index} 个专利资料片段",
                "source_count": 1,
                "chars": 40,
            }
            for index in range(1, 11)
        ],
        "low_relevance": False,
        "top_score": 0.9,
        "threshold": 0.5,
    }

    result = await tool.arun({"query": "分析这些专利资料", "rag_output_key": "rag_context"}, context=context)

    assert result.success is True
    assert len(client.batch_inputs) == 2
    assert client.batch_inputs[0]["batch_id"] == "batch_1"
    assert client.batch_inputs[0]["chunk_ids"] == [f"chunk-{index}" for index in range(1, 9)]
    assert client.batch_inputs[1]["batch_id"] == "batch_2"
    assert client.batch_inputs[1]["chunk_ids"] == ["chunk-9", "chunk-10"]


@pytest.mark.asyncio
async def test_rag_batch_summarize_json_parse_failure_preserves_raw_financial_context() -> None:
    client = StubEvidenceExtractionClient(responses={"batch_1": "这不是 JSON", "batch_2": "也不是 JSON"})
    tool = RAGBatchSummarizeTool(client=client)
    context = _build_financial_context()

    result = await tool.arun({"query": "财务报表的数据有哪些", "rag_output_key": "rag_context"}, context=context)

    assert result.success is True
    assert len(client.requests) == 2
    assert result.output is not None
    assert result.output["confidence"] == "low"
    assert result.output["extraction_mode"] == RAW_CONTEXT_FALLBACK_MODE
    assert result.output["metadata"]["extraction_mode"] == RAW_CONTEXT_FALLBACK_MODE
    assert RAW_CONTEXT_FALLBACK_MISSING_INFORMATION in result.output["missing_information"]
    assert JSON_FALLBACK_WITH_RAW_CONTEXT_MISSING_INFORMATION in result.output["missing_information"]
    assert all(
        item["parse_mode"] == JSON_FALLBACK_WITH_RAW_CONTEXT_MODE
        for item in result.output["batch_summaries"]
    )
    assert result.output["successful_batch_count"] == 2
    assert result.output["failed_batch_count"] == 0
    assert "Raw Evidence Context" in result.output["text"]
    assert "营业收入 10,739,842.03" in result.output["text"]
    assert "营业成本 5,146,519.97" in result.output["text"]
    assert "净利润 -5,418,627.21" in result.output["text"]
    assert "现金净增加额 -25,037.71" in result.output["text"]
    assert "期末现金余额 5,378.10" in result.output["text"]
    assert "fin-1" in result.output["evidence_chunk_ids"]
    assert "src-fin-1" in result.output["evidence_chunk_ids"]
    assert "fin-3" in result.output["evidence_chunk_ids"]
    first_batch = result.output["batch_summaries"][0]
    assert first_batch["raw_model_output"] == "这不是 JSON"
    assert "营业收入 10,739,842.03" in first_batch["raw_evidence_context"]
    assert result.output["raw_model_outputs"] == ["这不是 JSON", "也不是 JSON"]
    assert any("营业收入 10,739,842.03" in item for item in result.output["raw_evidence_contexts"])


@pytest.mark.asyncio
async def test_rag_batch_summarize_partial_fallback_keeps_facts_and_raw_context() -> None:
    client = StubEvidenceExtractionClient(
        responses={
            "batch_1": json.dumps(
                {
                    "answer_facts": ["营业收入 10,739,842.03。"],
                    "key_points": ["营业成本 5,146,519.97。"],
                    "evidence_chunk_ids": ["fin-1", "invented-id"],
                    "irrelevant_chunk_ids": [],
                    "missing_information": [],
                    "confidence": "high",
                },
                ensure_ascii=False,
            ),
            "batch_2": "模型说：净利润和现金流在上下文里，但这不是 JSON",
        }
    )
    tool = RAGBatchSummarizeTool(client=client)
    context = _build_financial_context()

    result = await tool.arun({"query": "财务报表的数据有哪些", "rag_output_key": "rag_context"}, context=context)

    assert result.success is True
    assert result.output is not None
    assert result.output["extraction_mode"] == "evidence_extraction"
    assert result.output["batch_summaries"][0]["parse_mode"] == "json"
    assert result.output["batch_summaries"][1]["parse_mode"] == JSON_FALLBACK_WITH_RAW_CONTEXT_MODE
    assert "营业收入 10,739,842.03" in result.output["text"]
    assert "营业成本 5,146,519.97" in result.output["text"]
    assert "净利润 -5,418,627.21" in result.output["text"]
    assert "invented-id" not in result.output["evidence_chunk_ids"]
    assert "fin-1" in result.output["evidence_chunk_ids"]
    assert "fin-3" in result.output["evidence_chunk_ids"]
    assert "src-fin-3" in result.output["evidence_chunk_ids"]


@pytest.mark.asyncio
async def test_rag_batch_summarize_tolerates_json_code_blocks_and_embedded_objects() -> None:
    client = StubEvidenceExtractionClient(
        responses={
            "batch_1": (
                "下面是结果：\n```json\n"
                + json.dumps(
                    {
                        "answer_facts": ["2024-05-01 后必须 30 天内提交。"],
                        "evidence_chunk_ids": ["chunk-1"],
                        "confidence": "high",
                    },
                    ensure_ascii=False,
                )
                + "\n```"
            ),
            "batch_2": (
                "说明文字 "
                + json.dumps(
                    {
                        "key_points": ["流程顺序为申请、复核、审批、归档。"],
                        "evidence_chunk_ids": ["chunk-3", "invented-id"],
                        "confidence": "medium",
                    },
                    ensure_ascii=False,
                )
                + " 结束"
            ),
        }
    )
    tool = RAGBatchSummarizeTool(client=client)
    context = _build_context(short=False)

    result = await tool.arun({"query": "今年公司都做了哪些申报", "rag_output_key": "rag_context"}, context=context)

    assert result.success is True
    assert result.output is not None
    assert all(item["parse_mode"] == "json" for item in result.output["batch_summaries"])
    assert result.output["raw_evidence_contexts"] == []
    assert "2024-05-01" in result.output["text"]
    assert "流程顺序为申请、复核、审批、归档" in result.output["text"]
    assert "chunk-1" in result.output["evidence_chunk_ids"]
    assert "chunk-3" in result.output["evidence_chunk_ids"]
    assert "invented-id" not in result.output["evidence_chunk_ids"]
