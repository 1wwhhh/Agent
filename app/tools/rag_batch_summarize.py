from __future__ import annotations

import json
import re
import time
from typing import Any

from pydantic import Field

from app.llm.structured import parse_json_output
from app.prompts import (
    RAG_EVIDENCE_EXTRACTION_PROMPT_NAME,
    RAG_EVIDENCE_EXTRACTION_PROMPT_VERSION,
    PromptRegistry,
    build_default_prompt_registry,
)
from app.schemas.context import ContextStore
from app.schemas.llm import LLMMessage, LLMRequest
from app.schemas.tool import ToolResult
from app.schemas.tool_outputs import RagEvidenceExtractionToolOutput
from app.tools.base import BaseTool
from app.tools.llm_client import LLMClient
from app.utils import runtime_log, runtime_progress

DIRECT_CONTEXT_CHUNK_LIMIT = 5
DIRECT_CONTEXT_CHAR_LIMIT = 8000
DIRECT_CONTEXT_MODE = "direct_context_passthrough"
EVIDENCE_EXTRACTION_MODE = "evidence_extraction"
GENERIC_EVIDENCE_BATCH_CHAR_LIMIT = 12000
GENERIC_EVIDENCE_BATCH_CHUNK_LIMIT = 8
RAW_CONTEXT_FALLBACK_MODE = "raw_context_fallback"
JSON_FALLBACK_WITH_RAW_CONTEXT_MODE = "json_fallback_with_raw_context"
JSON_FALLBACK_MISSING_INFORMATION = "模型未返回有效证据提取 JSON"
JSON_FALLBACK_WITH_RAW_CONTEXT_MISSING_INFORMATION = (
    "模型未返回有效证据提取 JSON，已保留原始检索证据供最终回答使用"
)
RAW_CONTEXT_FALLBACK_MISSING_INFORMATION = "证据抽取 JSON 解析失败，已改用原始检索内容作为最终回答上下文"
CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}
BATCH_PROGRESS_TEXT_LIMIT = 5000


class RAGBatchSummarizeTool(BaseTool):
    name: str = Field(default="rag_batch_summarize_tool")
    description: str = Field(default="Extract query-focused RAG evidence for downstream generation.")
    timeout: int = Field(default=1800, gt=0)
    tags: list[str] = Field(default_factory=lambda: ["rag", "summary", "llm"])
    client: LLMClient = Field(...)
    prompt_registry: PromptRegistry = Field(default_factory=build_default_prompt_registry)

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "rag_output_key": {"type": ["string", "null"]},
            },
            "required": ["query"],
            "additionalProperties": True,
        }

    def get_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "summary": {"type": "string"},
                "batch_summaries": {"type": "array"},
                "answer_facts": {"type": "array"},
                "key_points": {"type": "array"},
                "evidence_chunk_ids": {"type": "array"},
                "irrelevant_chunk_ids": {"type": "array"},
                "missing_information": {"type": "array"},
                "records": {"type": "array"},
                "confidence": {"type": "string"},
                "extraction_mode": {"type": "string"},
                "metadata": {"type": "object"},
                "source_chunk_ids": {"type": "array"},
                "batch_count": {"type": "integer"},
                "low_relevance": {"type": "boolean"},
                "top_score": {"type": "number"},
                "threshold": {"type": "number"},
                "raw_evidence_contexts": {"type": "array"},
                "raw_model_outputs": {"type": "array"},
            },
            "required": ["text", "summary", "batch_summaries"],
            "additionalProperties": True,
        }

    def get_routing_capability(self) -> dict[str, Any]:
        capability = super().get_routing_capability()
        capability["supported_task_types"] = ["rag_batch_summary"]
        capability["default_task_type"] = "rag_batch_summary"
        capability["supported_tags"] = ["rag", "summary", "llm"]
        return capability

    async def _arun(self, payload: dict[str, Any], context: ContextStore | None = None) -> ToolResult:
        if context is None:
            return self.build_result(
                success=False,
                error="rag_batch_summarize_tool requires context",
                metadata={"payload": payload},
            )

        query = str(payload.get("query") or "").strip()
        if not query:
            return self.build_result(
                success=False,
                error="rag_batch_summarize_tool requires query",
                metadata={"payload": payload},
            )

        rag_output_key = str(payload.get("rag_output_key") or "rag_context").strip() or "rag_context"
        rag_output = context.task_results.get(rag_output_key)
        if not isinstance(rag_output, dict):
            return self.build_result(
                success=False,
                error=f"rag output '{rag_output_key}' not found",
                metadata={"payload": payload},
            )

        context_batches = rag_output.get("context_batches")
        if not isinstance(context_batches, list) or not context_batches:
            return self.build_result(
                success=False,
                error="rag output does not contain context_batches",
                metadata={"payload": payload},
            )

        chunks_raw = rag_output.get("chunks")
        chunks = [item for item in chunks_raw if isinstance(item, dict)] if isinstance(chunks_raw, list) else []
        chunk_count = len(chunks)
        total_context_chars = self._total_context_chars(chunks)
        selected_batches = context_batches
        low_relevance = bool(rag_output.get("low_relevance", False))
        top_score = self._safe_float(rag_output.get("top_score"), default=0.0)
        threshold = self._safe_float(rag_output.get("threshold"), default=0.0)
        total_batches = len(selected_batches)

        runtime_progress(
            step="rag_batch_summarize",
            status="start",
            detail=(
                f"tool={self.name} rag_output_key={rag_output_key} batch_count={total_batches} "
                f"chunk_count={chunk_count} total_context_chars={total_context_chars} "
                f"low_relevance={low_relevance} top_score={top_score} threshold={threshold} "
                f"extraction_mode={EVIDENCE_EXTRACTION_MODE}"
            ),
            request_id=context.runtime.request_id,
            session_id=context.runtime.session_id,
        )
        runtime_log(
            layer="rag_batch_summarize",
            event="start",
            data={
                "request_id": context.runtime.request_id,
                "tool_name": self.name,
                "rag_output_key": rag_output_key,
                "batch_count": total_batches,
                "chunk_count": chunk_count,
                "total_context_chars": total_context_chars,
                "low_relevance": low_relevance,
                "top_score": top_score,
                "threshold": threshold,
                "extraction_mode": EVIDENCE_EXTRACTION_MODE,
            },
        )

        if chunk_count <= DIRECT_CONTEXT_CHUNK_LIMIT or total_context_chars <= DIRECT_CONTEXT_CHAR_LIMIT:
            return self._build_direct_context_result(
                payload=payload,
                chunks=chunks,
                context_batches=selected_batches,
                chunk_count=chunk_count,
                total_context_chars=total_context_chars,
                low_relevance=low_relevance,
                top_score=top_score,
                threshold=threshold,
            )

        return await self._extract_long_context_evidence(
            payload=payload,
            context=context,
            query=query,
            context_batches=selected_batches,
            chunks=chunks,
            chunk_count=chunk_count,
            total_context_chars=total_context_chars,
            low_relevance=low_relevance,
            top_score=top_score,
            threshold=threshold,
        )

    def _build_direct_context_result(
        self,
        *,
        payload: dict[str, Any],
        chunks: list[dict[str, Any]],
        context_batches: list[Any],
        chunk_count: int,
        total_context_chars: int,
        low_relevance: bool,
        top_score: float,
        threshold: float,
    ) -> ToolResult:
        evidence_context = self._format_evidence_context(chunks)
        evidence_chunk_ids = self._collect_chunk_ids(chunks)
        output_metadata = {
            "llm_called": False,
            "chunk_count": chunk_count,
            "total_context_chars": total_context_chars,
            "extraction_mode": DIRECT_CONTEXT_MODE,
        }
        batch_summaries = [
            {
                "batch_id": DIRECT_CONTEXT_MODE,
                "text": evidence_context,
                "summary": evidence_context,
                "answer_facts": [],
                "key_points": [],
                "evidence_chunk_ids": evidence_chunk_ids,
                "irrelevant_chunk_ids": [],
                "missing_information": [],
                "records": [],
                "confidence": "medium",
                "extraction_mode": DIRECT_CONTEXT_MODE,
                "parse_mode": DIRECT_CONTEXT_MODE,
            }
        ]
        metadata = {
            "payload": payload,
            "failed_batches": [],
            "batch_count": len(context_batches),
            "successful_batch_count": 1,
            "failed_batch_count": 0,
            "source_chunk_ids": evidence_chunk_ids,
            "low_relevance": low_relevance,
            "top_score": top_score,
            "threshold": threshold,
            **output_metadata,
        }
        runtime_log(
            layer="rag_batch_summarize",
            event="success",
            data={"stage": DIRECT_CONTEXT_MODE, **output_metadata},
        )
        return self.build_result(
            success=True,
            output={
                "text": evidence_context,
                "summary": evidence_context,
                "batch_summaries": batch_summaries,
                "answer_facts": [],
                "key_points": [],
                "evidence_chunk_ids": evidence_chunk_ids,
                "irrelevant_chunk_ids": [],
                "missing_information": [],
                "records": [],
                "confidence": "medium",
                "extraction_mode": DIRECT_CONTEXT_MODE,
                "metadata": output_metadata,
                "source_chunk_ids": evidence_chunk_ids,
                "batch_count": len(context_batches),
                "successful_batch_count": 1,
                "failed_batch_count": 0,
                "low_relevance": low_relevance,
                "top_score": top_score,
                "threshold": threshold,
            },
            metadata=metadata,
        )

    async def _extract_long_context_evidence(
        self,
        *,
        payload: dict[str, Any],
        context: ContextStore,
        query: str,
        context_batches: list[Any],
        chunks: list[dict[str, Any]],
        chunk_count: int,
        total_context_chars: int,
        low_relevance: bool,
        top_score: float,
        threshold: float,
    ) -> ToolResult:
        context_batches = self._generic_evidence_context_batches(context_batches=context_batches, chunks=chunks)
        batch_summaries: list[dict[str, Any]] = []
        failed_batches: list[dict[str, Any]] = []
        answer_facts: list[str] = []
        key_points: list[str] = []
        evidence_chunk_ids: list[str] = []
        irrelevant_chunk_ids: list[str] = []
        missing_information: list[str] = []
        records: list[dict[str, Any]] = []
        confidence_values: list[str] = []
        source_chunk_ids: set[str] = set()

        for index, batch in enumerate(context_batches, start=1):
            if not isinstance(batch, dict):
                continue

            batch_id = str(batch.get("batch_id") or f"batch_{len(batch_summaries) + 1}")
            batch_content = str(batch.get("joined_context") or "")
            if not batch_content.strip():
                continue

            chunk_ids = self._string_list(batch.get("chunk_ids"))
            source_ids = self._string_list(batch.get("source_chunk_ids"))
            source_chunk_ids.update(source_ids)
            known_ids = set(chunk_ids) | set(source_ids)
            batch_input = {
                "query": query,
                "batch_id": batch_id,
                "chunk_ids": chunk_ids,
                "source_chunk_ids": source_ids,
                "batch_content": batch_content,
            }
            request = self._build_evidence_request(
                payload=payload,
                context=context,
                batch_input=batch_input,
            )
            prompt_chars = self._message_chars(request.messages)
            batch_chars = int(batch.get("chars") or len(batch_content))

            runtime_progress(
                step="rag_batch_summarize",
                status="batch_start",
                detail=(
                    f"batch={batch_id} index={index}/{len(context_batches)} chars={batch_chars} "
                    f"chunk_ids={len(chunk_ids)} prompt_chars={prompt_chars} stage=before_generate"
                ),
                request_id=context.runtime.request_id,
                session_id=context.runtime.session_id,
            )
            runtime_log(
                layer="rag_batch_summarize",
                event="execute",
                data={
                    "batch_id": batch_id,
                    "batch_index": index,
                    "batch_count": len(context_batches),
                    "chars": batch_chars,
                    "chunk_ids_count": len(chunk_ids),
                    "prompt_chars": prompt_chars,
                    "stage": "before_generate",
                    "extraction_mode": EVIDENCE_EXTRACTION_MODE,
                },
            )

            try:
                generate_started_at = time.perf_counter()
                response = await self.client.generate(request)
                generate_latency_ms = int(max(0.0, (time.perf_counter() - generate_started_at) * 1000))
                response_text = (response.text or "").strip()
                response_chars = len(response_text)
                if not response_text:
                    raw_output = self._json_parse_fallback_with_raw_context(
                        batch_id=batch_id,
                        batch_content=batch_content,
                        chunk_ids=chunk_ids,
                        source_ids=source_ids,
                        raw_model_output=response_text,
                    )
                    parse_mode = JSON_FALLBACK_WITH_RAW_CONTEXT_MODE
                else:
                    try:
                        raw_output = self._parse_evidence_response(response_text)
                        parse_mode = "json"
                    except Exception:
                        raw_output = self._json_parse_fallback_with_raw_context(
                            batch_id=batch_id,
                            batch_content=batch_content,
                            chunk_ids=chunk_ids,
                            source_ids=source_ids,
                            raw_model_output=response_text,
                        )
                        parse_mode = JSON_FALLBACK_WITH_RAW_CONTEXT_MODE

                extraction = self._normalize_extraction(raw_output, known_ids=known_ids)
                batch_text = self._format_batch_extraction_text(extraction, parse_mode=parse_mode)
                batch_summary = {
                    "batch_id": batch_id,
                    "text": batch_text,
                    "summary": batch_text,
                    "answer_facts": extraction["answer_facts"],
                    "key_points": extraction["key_points"],
                    "evidence_chunk_ids": extraction["evidence_chunk_ids"],
                    "irrelevant_chunk_ids": extraction["irrelevant_chunk_ids"],
                    "missing_information": extraction["missing_information"],
                    "records": extraction["records"],
                    "confidence": extraction["confidence"],
                    "extraction_mode": EVIDENCE_EXTRACTION_MODE,
                    "parse_mode": parse_mode,
                    "chunk_ids": chunk_ids,
                    "source_chunk_ids": source_ids,
                }
                if extraction["raw_model_output"]:
                    batch_summary["raw_model_output"] = extraction["raw_model_output"]
                if extraction["raw_evidence_context"]:
                    batch_summary["raw_evidence_context"] = extraction["raw_evidence_context"]
                batch_summaries.append(batch_summary)
                runtime_progress(
                    step="rag_batch_summarize",
                    status="batch_result",
                    detail=self._batch_result_progress_detail(
                        batch_id=batch_id,
                        index=index,
                        total_batches=len(context_batches),
                        batch_chars=batch_chars,
                        prompt_chars=prompt_chars,
                        response_chars=response_chars,
                        latency_ms=generate_latency_ms,
                        parse_mode=parse_mode,
                        extraction=extraction,
                        batch_text=batch_text,
                    ),
                    request_id=context.runtime.request_id,
                    session_id=context.runtime.session_id,
                )
                self._extend_unique(answer_facts, extraction["answer_facts"])
                self._extend_unique(key_points, extraction["key_points"])
                self._extend_unique(evidence_chunk_ids, extraction["evidence_chunk_ids"])
                self._extend_unique(irrelevant_chunk_ids, extraction["irrelevant_chunk_ids"])
                self._extend_unique(missing_information, extraction["missing_information"])
                self._extend_unique_records(records, extraction["records"])
                confidence_values.append(extraction["confidence"])
                runtime_log(
                    layer="rag_batch_summarize",
                    event="success",
                    data={
                        "batch_id": batch_id,
                        "latency_ms": generate_latency_ms,
                        "response_chars": response_chars,
                        "prompt_chars": prompt_chars,
                        "parse_mode": parse_mode,
                        "stage": "evidence_extraction_success",
                    },
                )
            except Exception as exc:
                failed_batches.append({"batch_id": batch_id, "error": str(exc)})
                runtime_progress(
                    step="rag_batch_summarize",
                    status="batch_error",
                    detail=self._batch_error_progress_detail(
                        batch_id=batch_id,
                        index=index,
                        total_batches=len(context_batches),
                        prompt_chars=prompt_chars,
                        exception=exc,
                    ),
                    request_id=context.runtime.request_id,
                    session_id=context.runtime.session_id,
                )
                runtime_log(
                    layer="rag_batch_summarize",
                    event="error",
                    data={
                        "batch_id": batch_id,
                        "stage": "generate",
                        "exception_type": type(exc).__name__,
                        "error_message": str(exc),
                        "prompt_chars": prompt_chars,
                    },
                )

        if not batch_summaries:
            return self.build_result(
                success=False,
                error="all rag evidence extraction batches failed",
                metadata={"payload": payload, "failed_batches": failed_batches},
            )

        irrelevant_chunk_ids = [item for item in irrelevant_chunk_ids if item not in set(evidence_chunk_ids)]
        fallback_batch_summaries = [
            item for item in batch_summaries if item.get("parse_mode") == JSON_FALLBACK_WITH_RAW_CONTEXT_MODE
        ]
        raw_evidence_contexts = self._string_list(
            [item.get("raw_evidence_context") for item in fallback_batch_summaries]
        )
        raw_model_outputs = self._string_list([item.get("raw_model_output") for item in fallback_batch_summaries])
        all_batches_fallback = bool(batch_summaries) and len(fallback_batch_summaries) == len(batch_summaries)
        output_extraction_mode = (
            RAW_CONTEXT_FALLBACK_MODE
            if all_batches_fallback and raw_evidence_contexts
            else EVIDENCE_EXTRACTION_MODE
        )
        confidence = (
            "low"
            if output_extraction_mode == RAW_CONTEXT_FALLBACK_MODE
            else self._merge_confidence(confidence_values)
        )
        if output_extraction_mode == RAW_CONTEXT_FALLBACK_MODE:
            self._extend_unique(missing_information, [RAW_CONTEXT_FALLBACK_MISSING_INFORMATION])
        if low_relevance:
            self._extend_unique(missing_information, ["当前知识库未检索到足够相关的资料。"])
        parse_statuses = [
            f"{item.get('batch_id')}: {item.get('parse_mode')}"
            for item in batch_summaries
            if item.get("batch_id") and item.get("parse_mode")
        ]
        important_notices = []
        if raw_evidence_contexts:
            important_notices = [
                "Evidence extraction JSON parsing failed, but raw retrieved evidence is preserved below.",
                "JSON 解析失败不等于没有检索到数据。",
            ]
        merged_text = self._format_extraction_summary(
            answer_facts=answer_facts,
            key_points=key_points,
            evidence_chunk_ids=evidence_chunk_ids,
            irrelevant_chunk_ids=irrelevant_chunk_ids,
            missing_information=missing_information,
            confidence=confidence,
            parse_statuses=parse_statuses,
            important_notices=important_notices,
            raw_evidence_contexts=raw_evidence_contexts,
            records=records if records else None,
        )
        output_metadata = {
            "llm_called": True,
            "chunk_count": chunk_count,
            "total_context_chars": total_context_chars,
            "extraction_mode": output_extraction_mode,
            "all_batches_fallback": all_batches_fallback,
            "requested_extraction_mode": EVIDENCE_EXTRACTION_MODE,
            "record_count": len(records),
            "summary_text_chars": len(merged_text),
            "summary_text_mode": EVIDENCE_EXTRACTION_MODE,
        }
        metadata = {
            "payload": payload,
            "failed_batches": failed_batches,
            "batch_count": len(context_batches),
            "successful_batch_count": len(batch_summaries),
            "failed_batch_count": len(failed_batches),
            "source_chunk_ids": sorted(source_chunk_ids),
            "low_relevance": low_relevance,
            "top_score": top_score,
            "threshold": threshold,
            **output_metadata,
        }
        runtime_progress(
            step="rag_batch_summarize",
            status="completed",
            detail=f"success_batches={len(batch_summaries)} failed_batches={len(failed_batches)} stage=build_result",
            request_id=context.runtime.request_id,
            session_id=context.runtime.session_id,
        )
        runtime_log(
            layer="rag_batch_summarize",
            event="end",
            data={
                "batch_count": len(context_batches),
                "success_batches": len(batch_summaries),
                "failed_batches": len(failed_batches),
                "stage": "build_result",
                **output_metadata,
            },
        )
        return self.build_result(
            success=True,
            output={
                "text": merged_text,
                "summary": merged_text,
                "batch_summaries": batch_summaries,
                "answer_facts": answer_facts,
                "key_points": key_points,
                "evidence_chunk_ids": evidence_chunk_ids,
                "irrelevant_chunk_ids": irrelevant_chunk_ids,
                "missing_information": missing_information,
                "records": records,
                "confidence": confidence,
                "extraction_mode": output_extraction_mode,
                "metadata": output_metadata,
                "source_chunk_ids": sorted(source_chunk_ids),
                "batch_count": len(context_batches),
                "successful_batch_count": len(batch_summaries),
                "failed_batch_count": len(failed_batches),
                "raw_evidence_contexts": raw_evidence_contexts,
                "raw_model_outputs": raw_model_outputs,
                "low_relevance": low_relevance,
                "top_score": top_score,
                "threshold": threshold,
            },
            metadata=metadata,
        )

    def _build_context_batches_from_chunks(
        self,
        chunks: list[dict[str, Any]],
        *,
        batch_prefix: str,
        char_limit: int,
        chunk_limit: int,
    ) -> list[dict[str, Any]]:
        batches: list[dict[str, Any]] = []
        current_parts: list[str] = []
        current_chunk_ids: list[str] = []
        current_chars = 0
        for index, chunk in enumerate(chunks, start=1):
            chunk_id = self._chunk_id(chunk) or f"chunk_{index}"
            source = self._chunk_source(chunk)
            header = f"[chunk_id={chunk_id}]"
            if source:
                header = f"[chunk_id={chunk_id} source={source}]"
            segment = f"{header}\n{self._chunk_text(chunk)}".strip()
            if not segment:
                continue
            if current_parts and (len(current_chunk_ids) >= chunk_limit or current_chars + len(segment) + 2 > char_limit):
                batches.append(
                    {
                        "batch_id": f"{batch_prefix}_{len(batches) + 1}",
                        "chunk_ids": current_chunk_ids,
                        "source_chunk_ids": current_chunk_ids,
                        "joined_context": "\n\n".join(current_parts),
                        "source_count": len(current_chunk_ids),
                        "chars": current_chars,
                    }
                )
                current_parts = []
                current_chunk_ids = []
                current_chars = 0
            current_parts.append(segment)
            current_chunk_ids.append(chunk_id)
            current_chars += len(segment) + 2
        if current_parts:
            batches.append(
                {
                    "batch_id": f"{batch_prefix}_{len(batches) + 1}",
                    "chunk_ids": current_chunk_ids,
                    "source_chunk_ids": current_chunk_ids,
                    "joined_context": "\n\n".join(current_parts),
                    "source_count": len(current_chunk_ids),
                    "chars": current_chars,
                }
            )
        return batches

    def _should_rebatch_generic_evidence_batches(
        self,
        *,
        context_batches: list[Any],
        chunks: list[dict[str, Any]],
    ) -> bool:
        if len(chunks) <= 1:
            return False
        if not context_batches:
            return False
        dict_batches = [batch for batch in context_batches if isinstance(batch, dict)]
        if len(dict_batches) != len(context_batches):
            return False
        chunk_batch_count = 0
        for batch in dict_batches:
            chunk_ids = self._string_list(batch.get("chunk_ids")) or self._string_list(batch.get("source_chunk_ids"))
            if len(chunk_ids) != 1:
                return False
            chunk_batch_count += 1
        return chunk_batch_count == len(chunks)

    def _generic_evidence_context_batches(
        self,
        *,
        context_batches: list[Any],
        chunks: list[dict[str, Any]],
    ) -> list[Any]:
        if not self._should_rebatch_generic_evidence_batches(context_batches=context_batches, chunks=chunks):
            return context_batches
        return self._build_context_batches_from_chunks(
            chunks,
            batch_prefix="batch",
            char_limit=GENERIC_EVIDENCE_BATCH_CHAR_LIMIT,
            chunk_limit=GENERIC_EVIDENCE_BATCH_CHUNK_LIMIT,
        )

    def _build_evidence_request(
        self,
        *,
        payload: dict[str, Any],
        context: ContextStore,
        batch_input: dict[str, Any],
    ) -> LLMRequest:
        rendered_prompt = self.prompt_registry.render(
            RAG_EVIDENCE_EXTRACTION_PROMPT_NAME,
            version=RAG_EVIDENCE_EXTRACTION_PROMPT_VERSION,
            variables={"batch_input_json": json.dumps(batch_input, ensure_ascii=False, indent=2)},
        )
        return LLMRequest(
            prompt=rendered_prompt.user_prompt,
            system_prompt=rendered_prompt.system_prompt,
            messages=[
                LLMMessage(role="system", content=rendered_prompt.system_prompt),
                LLMMessage(role="user", content=rendered_prompt.user_prompt),
            ],
            model_name=payload.get("model_name"),
            temperature=float(payload.get("temperature", 0.2)),
            timeout_seconds=int(payload.get("timeout_seconds", self.timeout)),
            request_id=context.runtime.request_id,
            session_id=context.runtime.session_id,
            prompt_name=rendered_prompt.name,
            prompt_version=rendered_prompt.version,
            metadata={"operation": "tool", "tool_name": self.name, "extraction_mode": EVIDENCE_EXTRACTION_MODE},
        )

    def _parse_evidence_response(self, response_text: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for candidate in self._json_response_candidates(response_text):
            try:
                parsed = parse_json_output(
                    raw_output=candidate,
                    output_model=RagEvidenceExtractionToolOutput,
                )
                return parsed.output.model_dump(mode="json")
            except Exception as exc:
                last_error = exc

            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError as exc:
                last_error = exc
                continue
            if isinstance(payload, dict):
                return payload

        if last_error is not None:
            raise last_error
        raise ValueError("model output does not contain a JSON object")

    def _json_response_candidates(self, response_text: str) -> list[str]:
        candidates: list[str] = []
        stripped = response_text.strip()
        if stripped:
            candidates.append(stripped)

        for match in re.findall(r"```(?:json)?\s*(.*?)```", response_text, flags=re.IGNORECASE | re.DOTALL):
            candidate = match.strip()
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        embedded = self._extract_first_json_object(response_text)
        if embedded and embedded not in candidates:
            candidates.append(embedded)

        return candidates

    def _extract_first_json_object(self, text: str) -> str | None:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                payload, end_index = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return text[index : index + end_index]
        return None

    def _json_parse_fallback(self) -> dict[str, Any]:
        return {
            "answer_facts": [],
            "key_points": [],
            "evidence_chunk_ids": [],
            "irrelevant_chunk_ids": [],
            "missing_information": [JSON_FALLBACK_MISSING_INFORMATION],
            "records": [],
            "confidence": "low",
        }

    def _json_parse_fallback_with_raw_context(
        self,
        *,
        batch_id: str,
        batch_content: str,
        chunk_ids: list[str],
        source_ids: list[str],
        raw_model_output: str,
    ) -> dict[str, Any]:
        evidence_ids: list[str] = []
        self._extend_unique(evidence_ids, chunk_ids)
        self._extend_unique(evidence_ids, source_ids)
        return {
            "answer_facts": [],
            "key_points": [],
            "evidence_chunk_ids": evidence_ids,
            "irrelevant_chunk_ids": [],
            "missing_information": [JSON_FALLBACK_WITH_RAW_CONTEXT_MISSING_INFORMATION],
            "records": [],
            "confidence": "low",
            "parse_mode": JSON_FALLBACK_WITH_RAW_CONTEXT_MODE,
            "raw_model_output": raw_model_output,
            "raw_evidence_context": self._format_raw_evidence_context(
                batch_id=batch_id,
                batch_content=batch_content,
                chunk_ids=chunk_ids,
                source_ids=source_ids,
            ),
        }

    def _normalize_extraction(self, raw_output: dict[str, Any], *, known_ids: set[str]) -> dict[str, Any]:
        evidence_ids = self._filter_known_ids(self._string_list(raw_output.get("evidence_chunk_ids")), known_ids)
        irrelevant_ids = self._filter_known_ids(self._string_list(raw_output.get("irrelevant_chunk_ids")), known_ids)
        evidence_set = set(evidence_ids)
        irrelevant_ids = [item for item in irrelevant_ids if item not in evidence_set]
        confidence = str(raw_output.get("confidence") or "low").strip().lower()
        if confidence not in CONFIDENCE_ORDER:
            confidence = "low"
        return {
            "answer_facts": self._string_list(raw_output.get("answer_facts")),
            "key_points": self._string_list(raw_output.get("key_points")),
            "evidence_chunk_ids": evidence_ids,
            "irrelevant_chunk_ids": irrelevant_ids,
            "missing_information": self._string_list(raw_output.get("missing_information")),
            "records": self._record_list(raw_output.get("records"), known_ids=known_ids),
            "confidence": confidence,
            "raw_model_output": str(raw_output.get("raw_model_output") or "").strip(),
            "raw_evidence_context": str(raw_output.get("raw_evidence_context") or "").strip(),
        }

    def _format_evidence_context(self, chunks: list[dict[str, Any]]) -> str:
        parts = ["Evidence Context:"]
        for index, chunk in enumerate(chunks, start=1):
            chunk_id = self._chunk_id(chunk) or f"chunk_{index}"
            text = self._chunk_text(chunk)
            if not text:
                continue
            source = self._chunk_source(chunk)
            score = self._chunk_score(chunk)
            header_parts = [f"chunk_id={chunk_id}"]
            if source:
                header_parts.append(f"source={source}")
            if score is not None:
                header_parts.append(f"score={score}")
            parts.append(f"[{ ' '.join(header_parts) }]\n{text}")
        if len(parts) == 1:
            parts.append("No retrieved chunks available.")
        return "\n\n".join(parts)

    def _format_batch_extraction_text(
        self,
        extraction: dict[str, Any],
        *,
        parse_mode: str,
    ) -> str:
        important_notices = []
        if extraction.get("raw_evidence_context"):
            important_notices = [
                "Evidence extraction JSON parsing failed, but raw retrieved evidence is preserved below.",
                "JSON 解析失败不等于没有检索到数据。",
            ]
        return self._format_extraction_summary(
            answer_facts=extraction["answer_facts"],
            key_points=extraction["key_points"],
            evidence_chunk_ids=extraction["evidence_chunk_ids"],
            irrelevant_chunk_ids=extraction["irrelevant_chunk_ids"],
            missing_information=extraction["missing_information"],
            confidence=extraction["confidence"],
            parse_statuses=[parse_mode] if parse_mode else None,
            important_notices=important_notices,
            raw_evidence_contexts=(
                [extraction["raw_evidence_context"]] if extraction.get("raw_evidence_context") else None
            ),
            records=extraction["records"] if extraction["records"] else None,
        )

    def _batch_result_progress_detail(
        self,
        *,
        batch_id: str,
        index: int,
        total_batches: int,
        batch_chars: int,
        prompt_chars: int,
        response_chars: int,
        latency_ms: int,
        parse_mode: str,
        extraction: dict[str, Any],
        batch_text: str,
    ) -> str:
        return json.dumps(
            {
                "batch": batch_id,
                "index": f"{index}/{total_batches}",
                "stage": "after_generate",
                "chars": batch_chars,
                "prompt_chars": prompt_chars,
                "response_chars": response_chars,
                "latency_ms": latency_ms,
                "parse_mode": parse_mode,
                "confidence": extraction["confidence"],
                "answer_facts": extraction["answer_facts"],
                "key_points": extraction["key_points"],
                "evidence_chunk_ids": extraction["evidence_chunk_ids"],
                "irrelevant_chunk_ids": extraction["irrelevant_chunk_ids"],
                "missing_information": extraction["missing_information"],
                "records": extraction["records"],
                "summary": self._truncate_progress_text(batch_text),
            },
            ensure_ascii=False,
        )

    def _batch_error_progress_detail(
        self,
        *,
        batch_id: str,
        index: int,
        total_batches: int,
        prompt_chars: int,
        exception: Exception,
    ) -> str:
        return json.dumps(
            {
                "batch": batch_id,
                "index": f"{index}/{total_batches}",
                "stage": "after_generate",
                "prompt_chars": prompt_chars,
                "exception_type": type(exception).__name__,
                "error_message": self._truncate_progress_text(str(exception)),
            },
            ensure_ascii=False,
        )

    def _truncate_progress_text(self, text: str, *, limit: int = BATCH_PROGRESS_TEXT_LIMIT) -> str:
        if len(text) <= limit:
            return text
        omitted = len(text) - limit
        return f"{text[:limit]}... [truncated {omitted} chars]"

    def _format_extraction_summary(
        self,
        *,
        answer_facts: list[str],
        key_points: list[str],
        evidence_chunk_ids: list[str],
        irrelevant_chunk_ids: list[str],
        missing_information: list[str],
        confidence: str,
        parse_statuses: list[str] | None = None,
        important_notices: list[str] | None = None,
        raw_evidence_contexts: list[str] | None = None,
        records: list[dict[str, Any]] | None = None,
    ) -> str:
        sections = ["Evidence Extraction:"]
        if parse_statuses:
            sections.append(self._format_list_section("Parse Status", parse_statuses))
        if important_notices:
            sections.append(self._format_list_section("Important Notice", important_notices))
        sections.append(self._format_list_section("Answer Facts", answer_facts))
        sections.append(self._format_list_section("Key Points", key_points))
        if records is not None:
            sections.append(self._format_records_section("Records", records))
        sections.append(self._format_list_section("Evidence Chunk IDs", evidence_chunk_ids))
        sections.append(self._format_list_section("Irrelevant Chunk IDs", irrelevant_chunk_ids))
        if raw_evidence_contexts:
            sections.append(self._format_raw_evidence_contexts_section(raw_evidence_contexts))
        sections.append(self._format_list_section("Missing Information", missing_information))
        sections.append(f"Confidence:\n{confidence}")
        return "\n\n".join(sections)

    def _format_records_section(self, title: str, records: list[dict[str, Any]]) -> str:
        if not records:
            return f"{title}:\n- None"
        return f"{title}:\n" + "\n".join(
            f"- {json.dumps(record, ensure_ascii=False, sort_keys=True)}" for record in records
        )

    def _format_raw_evidence_context(
        self,
        *,
        batch_id: str,
        batch_content: str,
        chunk_ids: list[str],
        source_ids: list[str],
    ) -> str:
        header_parts = [f"batch_id={batch_id}"]
        if chunk_ids:
            header_parts.append(f"chunk_ids={','.join(chunk_ids)}")
        if source_ids:
            header_parts.append(f"source_chunk_ids={','.join(source_ids)}")
        header = " ".join(header_parts)
        return f"Raw Evidence Context:\n[{header}]\n{batch_content.strip()}"

    def _format_raw_evidence_contexts_section(self, contexts: list[str]) -> str:
        bodies = [self._raw_evidence_context_body(context) for context in contexts if context.strip()]
        if not bodies:
            return "Raw Evidence Context:\n- None"
        return "Raw Evidence Context:\n" + "\n\n".join(bodies)

    def _raw_evidence_context_body(self, context: str) -> str:
        text = context.strip()
        prefix = "Raw Evidence Context:"
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
        return text

    def _format_list_section(self, title: str, values: list[str]) -> str:
        if not values:
            return f"{title}:\n- None"
        return f"{title}:\n" + "\n".join(f"- {item}" for item in values)

    def _collect_chunk_ids(self, chunks: list[dict[str, Any]]) -> list[str]:
        collected: list[str] = []
        for chunk in chunks:
            chunk_id = self._chunk_id(chunk)
            if chunk_id and chunk_id not in collected:
                collected.append(chunk_id)
        return collected

    def _total_context_chars(self, chunks: list[dict[str, Any]]) -> int:
        return sum(len(str(chunk.get("context_text") or chunk.get("text") or "")) for chunk in chunks)

    def _chunk_id(self, chunk: dict[str, Any]) -> str | None:
        for key in ("chunk_id", "id", "source_chunk_id"):
            value = chunk.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return None

    def _chunk_text(self, chunk: dict[str, Any]) -> str:
        return str(chunk.get("context_text") or chunk.get("text") or chunk.get("content") or "").strip()

    def _chunk_source(self, chunk: dict[str, Any]) -> str | None:
        for key in ("relative_path", "absolute_path", "doc_id", "source"):
            value = chunk.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return None

    def _chunk_score(self, chunk: dict[str, Any]) -> Any | None:
        for key in ("score", "rerank_score", "final_score"):
            value = chunk.get(key)
            if value is not None and str(value).strip():
                return value
        return None

    def _filter_known_ids(self, values: list[str], known_ids: set[str]) -> list[str]:
        if not known_ids:
            return []
        return [value for value in values if value in known_ids]

    def _string_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        items = value if isinstance(value, list) else [value]
        result: list[str] = []
        for item in items:
            if item is None:
                continue
            text = str(item).strip()
            if text and text not in result:
                result.append(text)
        return result

    def _extend_unique(self, target: list[str], values: list[str]) -> None:
        for value in values:
            if value not in target:
                target.append(value)

    def _record_list(self, value: Any, *, known_ids: set[str]) -> list[dict[str, Any]]:
        if value is None:
            return []
        items = value if isinstance(value, list) else [value]
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            record: dict[str, Any] = {}
            for key, raw_value in item.items():
                field = str(key).strip()
                if not field:
                    continue
                if field == "source_chunk_ids":
                    source_ids = self._filter_known_ids(self._string_list(raw_value), known_ids)
                    if source_ids:
                        record[field] = source_ids
                    continue
                if field == "source_chunk_id":
                    source_chunk_id = self._record_scalar_text(raw_value)
                    if source_chunk_id and (not known_ids or source_chunk_id in known_ids):
                        record[field] = source_chunk_id
                    continue
                text = self._record_scalar_text(raw_value)
                if text:
                    record[field] = text

            source_chunk_id = self._record_scalar_text(record.get("source_chunk_id"))
            source_ids = self._string_list(record.get("source_chunk_ids"))
            if source_chunk_id and (not known_ids or source_chunk_id in known_ids) and source_chunk_id not in source_ids:
                source_ids.insert(0, source_chunk_id)
            if source_ids:
                record["source_chunk_ids"] = source_ids
            if not record:
                continue
            identity = self._record_identity(record)
            if identity in seen:
                continue
            seen.add(identity)
            records.append(record)
        return records

    def _record_scalar_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            return "；".join(self._string_list(value))
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        return str(value).strip()

    def _record_identity(self, record: dict[str, Any]) -> str:
        return json.dumps(record, ensure_ascii=False, sort_keys=True)

    def _extend_unique_records(self, target: list[dict[str, Any]], values: list[dict[str, Any]]) -> None:
        seen = {self._record_identity(record) for record in target}
        for record in values:
            identity = self._record_identity(record)
            if identity in seen:
                continue
            seen.add(identity)
            target.append(record)

    def _merge_confidence(self, values: list[str]) -> str:
        if not values:
            return "low"
        normalized = [value if value in CONFIDENCE_ORDER else "low" for value in values]
        return min(normalized, key=lambda item: CONFIDENCE_ORDER[item])

    def _safe_float(self, value: Any, *, default: float) -> float:
        if value is None or value == "":
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _message_chars(self, messages: list[LLMMessage]) -> int:
        return sum(len(message.content or "") for message in messages)
