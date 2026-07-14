from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ReasoningToolOutput(BaseModel):
    """写入运行时上下文的结构化推理结果。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    text: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    key_points: list[str] = Field(default_factory=list)


class TextGenerateToolOutput(BaseModel):
    """写入运行时上下文的结构化文本生成结果。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    text: str = Field(..., min_length=1)
    audience: str | None = Field(default=None)
    style: str | None = Field(default=None)


class RagSearchIntentToolOutput(BaseModel):
    """Structured LLM output for RAG search intent extraction."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    query: str = Field(..., min_length=1)
    source_type: Literal["pdf", "word", "ppt", "excel"] | None = Field(default=None)
    doc_id: str | None = Field(default=None)
    relative_path: str | None = Field(default=None)
    absolute_path: str | None = Field(default=None)
    confidence: Literal["high", "medium", "low"] = Field(default="medium")


class RagBatchSummaryToolOutput(BaseModel):
    """Structured batch summary output for map-reduce RAG."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    text: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    key_points: list[str] = Field(default_factory=list)
    evidence_chunk_ids: list[str] = Field(default_factory=list)


class RagEvidenceExtractionToolOutput(BaseModel):
    """Query-focused evidence extraction output for RAG batches."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    answer_facts: list[str] = Field(default_factory=list)
    key_points: list[str] = Field(default_factory=list)
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    irrelevant_chunk_ids: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    records: list[dict[str, Any]] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = Field(default="low")
    raw_model_output: str | None = Field(default=None)
    raw_evidence_context: str | None = Field(default=None)


class ToolCallArguments(BaseModel):
    """由大模型产出的标准化函数调用载荷。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    payload: dict[str, Any] = Field(default_factory=dict)
