from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

_REQUEST_PREFIX_RE = re.compile(
    r"^(?:请\s*)?(?:帮我\s*)?(?:查询|检索|查找|搜索|找到|根据|基于|依据|按照)(?:一下)?\s*",
    re.IGNORECASE,
)
_LEADING_FILLER_RE = re.compile(r"^(?:关于|有关|针对|围绕|就)\s*", re.IGNORECASE)
_LEADING_RELATION_RE = re.compile(r"^(?:里面的|中的|里的|里面|中|里)\s*", re.IGNORECASE)
_EXPLICIT_FILE_DOC_ID_RE = re.compile(
    r"(?P<doc_id>[^,，。;；!?！？\n\r]*?(?:\.(?:xlsx?|docx?|pdf|pptx?)|__(?:xlsx?|docx?|pdf|pptx?)))",
    re.IGNORECASE,
)
_DOC_ID_LEADING_PREFIX_RE = re.compile(
    r"^(?:请\s*)?(?:帮我\s*)?"
    r"(?:查询|检索|查找|搜索|找到|找|打开|读取|比较|对比|判断|统计|汇总|提取|分析|核对|识别|筛选|排序|列出|归纳|根据|基于|依据|按照)\s*",
    re.IGNORECASE,
)
_DOC_ID_TRAILING_CONNECTORS = ("里面的", "中的", "里的", "里面", "中", "里", "的")
_SOURCE_TYPE_BY_EXTENSION = {
    "doc": "word",
    "docx": "word",
    "pdf": "pdf",
    "ppt": "ppt",
    "pptx": "ppt",
    "xls": "excel",
    "xlsx": "excel",
}

_PATH_SCOPE_RE = re.compile(
    r"(?P<prefix>"
    r"(?:请\s*)?(?:帮我\s*)?"
    r"(?:(?:分析|查询|检索|搜索|查找|读取|查看|打开|看一下|根据|基于|依据|按照)\s*)?"
    r"(?:这个|该)?(?:文件夹|目录|路径)(?:路径)?\s*[:：]?\s*"
    r")"
    r"(?P<path>(?:[A-Za-z]:[\\/]|[\\/]{1,2}|[^,，。;；!?！？\n\r]*[\\/])[^,，。;；!?！？\n\r]*)",
    re.IGNORECASE,
)
_BARE_ABSOLUTE_PATH_RE = re.compile(
    r"^\s*"
    r"(?:(?:请\s*)?(?:帮我\s*)?"
    r"(?:(?:分析|查询|检索|搜索|查找|读取|查看|打开|看一下|根据|基于|依据|按照)\s*)?)?"
    r"(?P<path>(?:[A-Za-z]:[\\/]|[\\/]{1,2})[^,，。;；!?！？\n\r]*)"
    r"(?:[,，。;；!?！？]\s*(?P<query>.+))?"
    r"\s*$",
    re.IGNORECASE,
)
_BARE_RELATIVE_PATH_WITH_QUERY_RE = re.compile(
    r"^\s*"
    r"(?:(?:请\s*)?(?:帮我\s*)?"
    r"(?:(?:分析|查询|检索|搜索|查找|读取|查看|打开|看一下|根据|基于|依据|按照)\s*)?)?"
    r"(?P<path>[^,，。;；!?！？\n\r]*[\\/][^,，。;；!?！？\n\r]*)"
    r"[,，。;；!?！？]\s*(?P<query>.+?)"
    r"\s*$",
    re.IGNORECASE,
)
_TRAILING_PATH_CONNECTORS = ("目录下", "文件夹下", "下面的", "里面的", "中的", "里的", "下面", "里面", "下", "中", "里", "的")

_DOC_ID_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?<![A-Za-z0-9_])(?:doc[_\s-]*id|文档id)\s*(?P<doc_id>[A-Za-z0-9._-]+)",
        re.IGNORECASE,
    ),
)

_SOURCE_TYPE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "word",
        re.compile(
            r"(?<![A-Za-z0-9_])(?:word(?:\s*文档)?|docx?(?:\s*文档)?)(?![A-Za-z0-9_])",
            re.IGNORECASE,
        ),
    ),
    (
        "pdf",
        re.compile(
            r"(?<![A-Za-z0-9_])pdf(?:\s*文档)?(?![A-Za-z0-9_])",
            re.IGNORECASE,
        ),
    ),
    (
        "ppt",
        re.compile(
            r"(?<![A-Za-z0-9_])(?:pptx?(?:\s*文档)?|幻灯片(?:\s*文档)?|演示文稿(?:\s*文档)?)(?![A-Za-z0-9_])",
            re.IGNORECASE,
        ),
    ),
    (
        "excel",
        re.compile(
            r"(?<![A-Za-z0-9_])(?:excel(?:\s*文档)?|xls(?:x)?(?:\s*文档)?|表格(?:\s*文档)?)(?![A-Za-z0-9_])",
            re.IGNORECASE,
        ),
    ),
)

_DIRECT_SOURCE_QUERY_RE = re.compile(
    r"^\s*"
    r"(?:(?:请\s*)?(?:帮我\s*)?(?:查询|检索|查找|搜索)\s*)?"
    r"(?:在\s*)?"
    r"(?P<source_phrase>(?:word(?:\s*文档)?|docx?(?:\s*文档)?|pdf(?:\s*文档)?|pptx?(?:\s*文档)?|幻灯片(?:\s*文档)?|演示文稿(?:\s*文档)?|excel(?:\s*文档)?|xls(?:x)?(?:\s*文档)?|表格(?:\s*文档)?))"
    r"\s*(?:里面的|中的|里的|里面|中|里|的)\s*"
    r"(?P<query>.+?)"
    r"\s*$",
    re.IGNORECASE,
)

_CONTENT_SCOPE_RE = re.compile(
    r"^"
    r"(?P<scope>.+?)"
    r"(?:里面|中|里)"
    r"(?:的)?"
    r"(?:都)?"
    r"(?:包含了哪些内容|包含哪些内容|都包含了哪些内容|包含什么|都包含什么|有哪些内容|哪些内容|有什么内容|有什么)"
    r"$",
)

_SCOPE_QUERY_A_RE = re.compile(
    r"^在"
    r"(?P<scope>.+?)"
    r"(?:里面|中|里)"
    r"\s*"
    r"(?:关于|有关|针对|围绕|就)?"
    r"(?P<query>.+)$",
)

_SCOPE_QUERY_B_RE = re.compile(
    r"^(?P<scope>.+?)"
    r"(?:里面|中|里)"
    r"(?:的)?"
    r"(?P<query>.+)$",
)

_BASE_QUERY_SUFFIX_RE = (
    re.compile(r"都包含了哪些内容$"),
    re.compile(r"包含了哪些内容$"),
    re.compile(r"包含哪些内容$"),
    re.compile(r"里面包含什么$"),
    re.compile(r"里包含什么$"),
    re.compile(r"内容是什么$"),
    re.compile(r"有什么内容$"),
    re.compile(r"有什么$"),
    re.compile(r"是什么$"),
    re.compile(r"包含什么$"),
    re.compile(r"有哪些内容$"),
    re.compile(r"哪些内容$"),
)
_CONTENT_ONLY_QUERY_RE = re.compile(
    r"^(?:的)?(?:全部|所有|主要|具体)?(?:内容|信息|资料)"
    r"(?:都)?"
    r"(?:是什么|有什么|有哪些|包含什么|包含哪些内容|包含了哪些内容|都包含了哪些内容)?$"
)
_WRITE_CONTENT_QUERY_RE = re.compile(
    r"^(?P<subject>.+?)"
    r"(?:是)?(?:怎么写的|怎么写|如何写|如何撰写|怎么撰写|写法是什么|写什么)"
    r"(?:\s*(?:里面的|中的|里的|里面|中|里|的)\s*)?"
    r"(?:内容|信息|资料)?"
    r"(?:都)?"
    r"(?:是什么|有什么|有哪些|包含什么|包含哪些内容|包含了哪些内容|都包含了哪些内容)?$"
)
_SUBJECT_CONTENT_QUERY_RE = re.compile(
    r"^(?P<subject>.+?)"
    r"(?:里面的|中的|里的|里面|中|里)"
    r"(?:的)?"
    r"(?:内容|信息|资料)?"
    r"(?:都)?"
    r"(?:是什么|有什么|有哪些|包含什么|包含哪些内容|包含了哪些内容|都包含了哪些内容|有哪些内容|哪些内容)$"
)
_NATURAL_SCOPE_MARKERS = (
    "怎么",
    "如何",
    "什么",
    "哪些",
    "是否",
    "有没有",
    "完成",
    "落地",
    "成功",
    "失败",
    "情况",
    "状态",
    "进度",
    "判断",
    "分析",
)
_DESCRIPTIVE_DOCUMENT_KIND_MARKERS = (
    "专利",
    "权利要求",
    "说明书",
    "技术方案",
    "技术交底书",
    "交底书",
)
_DOCUMENT_CONTENT_QUESTION_MARKERS = (
    "内容",
    "信息",
    "资料",
    "包含",
    "有什么",
    "有哪些",
    "怎么写",
    "如何写",
    "权利要求",
    "技术方案",
    "摘要",
)
_PATENT_QUERY_EXPANSION_TERMS = ("技术方案", "权利要求", "摘要", "说明书")


class ParsedSearchQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    query: str = Field(..., min_length=1)
    scope_keyword: str | None = Field(default=None)
    source_type: str | None = Field(default=None)
    doc_id: str | None = Field(default=None)
    relative_path: str | None = Field(default=None)
    absolute_path: str | None = Field(default=None)


def parse_search_query(user_input: str) -> ParsedSearchQuery:
    """Parse a natural-language search question into structured search parameters."""
    doc_reference_text = _normalize_doc_reference_input(user_input)
    relative_path, absolute_path, path_removed_text = _extract_path_scope(doc_reference_text)
    explicit_doc_id, doc_reference_removed_text = _extract_explicit_file_doc_id(path_removed_text)
    normalized = _normalize_input(doc_reference_removed_text if explicit_doc_id else path_removed_text)
    if not normalized:
        raise ValueError("user_input is empty")

    working_text = normalized
    parsed_doc_id, working_text = _extract_doc_id(working_text)
    doc_id = explicit_doc_id or parsed_doc_id

    direct_source_query = _extract_direct_source_query(working_text)
    if direct_source_query is not None:
        source_type, raw_query = direct_source_query
        source_type = source_type or _infer_source_type_from_doc_id(doc_id)
        query = _clean_query(raw_query, allow_content_suffix_strip=True)
        if not query:
            raise ValueError("parsed query is empty")
        return ParsedSearchQuery(
            query=query,
            scope_keyword=None,
            source_type=source_type,
            doc_id=doc_id,
            relative_path=relative_path,
            absolute_path=absolute_path,
        )

    source_type, working_text = _extract_source_type(working_text)
    source_type = source_type or _infer_source_type_from_doc_id(doc_id)
    working_text = _strip_request_prefix(working_text)

    scope_keyword = None
    query = None
    if not doc_id and not (relative_path or absolute_path):
        scope_keyword, query = _extract_scope_query(working_text)
    if scope_keyword is not None:
        return ParsedSearchQuery(
            query=query,
            scope_keyword=scope_keyword,
            source_type=source_type,
            doc_id=doc_id,
            relative_path=relative_path,
            absolute_path=absolute_path,
        )

    query = _clean_query(working_text, allow_content_suffix_strip=bool(doc_id or source_type))
    if not query:
        raise ValueError("parsed query is empty")

    return ParsedSearchQuery(
        query=query,
        scope_keyword=None,
        source_type=source_type,
        doc_id=doc_id,
        relative_path=relative_path,
        absolute_path=absolute_path,
    )


def build_search_payload(
    parsed: ParsedSearchQuery,
    *,
    top_k: int = 5,
    weight_m3: float | None = None,
    weight_zh: float | None = None,
    weight_sparse: float | None = None,
    doc_created_at_from: int | str | None = None,
    doc_created_at_to: int | str | None = None,
) -> dict[str, object]:
    """Build the existing /search payload from parsed query fields."""
    payload: dict[str, object] = {
        "query": parsed.query,
        "top_k": top_k,
    }

    if parsed.source_type:
        payload["source_type"] = parsed.source_type

    doc_id = parsed.doc_id or parsed.scope_keyword
    if doc_id:
        payload["doc_id"] = doc_id

    if parsed.relative_path:
        payload["relative_path"] = parsed.relative_path

    if parsed.absolute_path:
        payload["absolute_path"] = parsed.absolute_path

    optional_fields = {
        "weight_m3": weight_m3,
        "weight_zh": weight_zh,
        "weight_sparse": weight_sparse,
        "doc_created_at_from": doc_created_at_from,
        "doc_created_at_to": doc_created_at_to,
    }
    for field_name, value in optional_fields.items():
        if value is not None:
            payload[field_name] = value

    return payload


def _normalize_doc_reference_input(user_input: str) -> str:
    if not isinstance(user_input, str):
        raise TypeError("user_input must be a string")

    normalized = unicodedata.normalize("NFKC", user_input).strip()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _normalize_input(user_input: str) -> str:
    if not isinstance(user_input, str):
        raise TypeError("user_input must be a string")

    normalized = unicodedata.normalize("NFKC", user_input).strip()
    normalized = normalized.translate(
        str.maketrans(
            {
                "，": " ",
                ",": " ",
                "。": " ",
                ".": " ",
                "！": " ",
                "!": " ",
                "？": " ",
                "?": " ",
                "；": " ",
                ";": " ",
                "：": " ",
                ":": " ",
                "、": " ",
                "=": " ",
                "（": " ",
                "）": " ",
                "(": " ",
                ")": " ",
                "【": " ",
                "】": " ",
                "[": " ",
                "]": " ",
                "《": " ",
                "》": " ",
                "<": " ",
                ">": " ",
                "“": " ",
                "”": " ",
                '"': " ",
                "‘": " ",
                "’": " ",
                "\u3000": " ",
            }
        )
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _extract_path_scope(text: str) -> tuple[str | None, str | None, str]:
    match = _PATH_SCOPE_RE.search(text)
    if match is not None:
        return _build_path_scope_result(
            text=text,
            raw_path=match.group("path"),
            start=match.start(),
            end=match.end(),
        )

    bare_match = _BARE_ABSOLUTE_PATH_RE.match(text)
    if bare_match is not None:
        path = _clean_path_candidate(bare_match.group("path"))
        if path and _is_absolute_path(path):
            remaining = _squash_whitespace(bare_match.group("query") or "")
            if not remaining:
                remaining = _default_query_from_path(path)
            return None, path, remaining

    bare_relative_match = _BARE_RELATIVE_PATH_WITH_QUERY_RE.match(text)
    if bare_relative_match is None:
        return None, None, text

    path = _clean_path_candidate(bare_relative_match.group("path"))
    if not path or not _looks_like_path(path) or _is_absolute_path(path):
        return None, None, text

    remaining = _squash_whitespace(bare_relative_match.group("query") or "")
    if not remaining:
        remaining = _default_query_from_path(path)
    if _looks_like_nas_root_path(path):
        return None, path, remaining
    return path, None, remaining


def _build_path_scope_result(*, text: str, raw_path: str, start: int, end: int) -> tuple[str | None, str | None, str]:
    path = _clean_path_candidate(raw_path)
    if not path or not _looks_like_path(path):
        return None, None, text

    remaining = f"{text[:start]} {text[end:]}"
    remaining = _squash_whitespace(remaining).strip(" ,，。;；:：!?！？")
    if not remaining:
        remaining = _default_query_from_path(path)
    if _is_absolute_path(path) or _looks_like_nas_root_path(path):
        return None, path, remaining
    return path, None, remaining


def _default_query_from_path(path: str) -> str:
    candidate = path.strip().rstrip("\\/")
    parts = [part for part in re.split(r"[\\/]+", candidate) if part]
    return parts[-1] if parts else candidate


def _clean_path_candidate(value: str) -> str:
    candidate = unicodedata.normalize("NFKC", value).strip()
    while candidate:
        stripped = _strip_request_prefix(candidate)
        stripped = _DOC_ID_LEADING_PREFIX_RE.sub("", stripped, count=1).strip()
        if stripped == candidate:
            break
        candidate = stripped
    candidate = candidate.strip(" \t\r\n'\"“”‘’` ,，。;；:：!?！？")
    changed = True
    while changed and candidate:
        changed = False
        for connector in _TRAILING_PATH_CONNECTORS:
            if candidate.endswith(connector):
                candidate = candidate[: -len(connector)].strip()
                changed = True
                break
    return candidate.strip(" \t\r\n'\"“”‘’` ,，。;；:：!?！？")


def _looks_like_path(value: str) -> bool:
    return bool(value) and ("\\" in value or "/" in value)


def _is_absolute_path(value: str) -> bool:
    return bool(re.match(r"^(?:[A-Za-z]:[\\/]|[\\/])", value))


def _looks_like_nas_root_path(value: str) -> bool:
    candidate = value.strip().strip("\\/")
    if not candidate:
        return False
    root = re.split(r"[\\/]+", candidate, maxsplit=1)[0].lower()
    return root == "9goo-nas"


def _extract_doc_id(text: str) -> tuple[str | None, str]:
    for pattern in _DOC_ID_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        doc_id = match.group("doc_id").strip()
        if not doc_id:
            continue
        remaining = pattern.sub(" ", text, count=1)
        return doc_id, _squash_whitespace(remaining)
    return None, text


def _extract_explicit_file_doc_id(text: str) -> tuple[str | None, str]:
    for match in _EXPLICIT_FILE_DOC_ID_RE.finditer(text):
        raw_doc_id = match.group("doc_id")
        doc_id = _clean_explicit_file_doc_id(raw_doc_id)
        if not doc_id:
            continue
        relative_start = raw_doc_id.rfind(doc_id)
        if relative_start < 0:
            continue
        start = match.start("doc_id") + relative_start
        end = start + len(doc_id)
        remaining = _remove_doc_reference_at(text, start=start, end=end)
        return doc_id, _squash_whitespace(remaining)
    return None, text


def _clean_explicit_file_doc_id(raw_doc_id: str) -> str:
    candidate = raw_doc_id.strip().strip(" ,，。;；:：!?！？")
    changed = True
    while changed and candidate:
        changed = False
        stripped = _DOC_ID_LEADING_PREFIX_RE.sub("", candidate, count=1).strip()
        if stripped != candidate:
            candidate = stripped
            changed = True
    return candidate.strip().strip(" ,，。;；:：!?！？")


def _remove_doc_reference_at(text: str, *, start: int, end: int) -> str:
    before = text[:start]
    after = text[end:]
    stripped_after = after.lstrip()
    leading_whitespace = after[: len(after) - len(stripped_after)]
    for connector in _DOC_ID_TRAILING_CONNECTORS:
        if stripped_after.startswith(connector):
            after = leading_whitespace + stripped_after[len(connector) :]
            break
    return f"{before} {after}"


def _infer_source_type_from_doc_id(doc_id: str | None) -> str | None:
    if not doc_id:
        return None
    match = re.search(r"(?:\.|__)([A-Za-z0-9]+)\s*$", doc_id)
    if match is None:
        return None
    return _SOURCE_TYPE_BY_EXTENSION.get(match.group(1).lower())


def _extract_direct_source_query(text: str) -> tuple[str, str] | None:
    match = _DIRECT_SOURCE_QUERY_RE.match(text)
    if match is None:
        return None

    source_type = _canonicalize_source_type_phrase(match.group("source_phrase"))
    query = _squash_whitespace(match.group("query"))
    if not query:
        return None
    return source_type, query


def _extract_source_type(text: str) -> tuple[str | None, str]:
    match_info: tuple[int, int, str, str] | None = None

    for source_type, pattern in _SOURCE_TYPE_RULES:
        match = pattern.search(text)
        if match is None:
            continue

        start, end = match.span()
        phrase = match.group(0)
        if match_info is None:
            match_info = (start, end, source_type, phrase)
            continue

        best_start, best_end, _, _ = match_info
        if start < best_start or (start == best_start and end - start > best_end - best_start):
            match_info = (start, end, source_type, phrase)

    if match_info is None:
        return None, text

    _, _, source_type, _ = match_info
    pattern = dict(_SOURCE_TYPE_RULES)[source_type]
    stripped = pattern.sub(" ", text)
    return source_type, _squash_whitespace(stripped)


def _extract_scope_query(text: str) -> tuple[str | None, str | None]:
    content_match = _CONTENT_SCOPE_RE.match(text)
    if content_match is not None:
        scope_keyword = _squash_whitespace(content_match.group("scope"))
        if scope_keyword:
            if _scope_should_remain_plain_query(scope_keyword, raw_text=text):
                return None, None
            return scope_keyword, f"{scope_keyword} 包含内容"

    scope_match = _SCOPE_QUERY_A_RE.match(text)
    if scope_match is not None:
        scope_keyword = _squash_whitespace(scope_match.group("scope"))
        query = _clean_query(scope_match.group("query"), allow_content_suffix_strip=True)
        if scope_keyword and query:
            if _scope_should_remain_plain_query(scope_keyword, raw_text=text, query=query):
                return None, None
            return scope_keyword, query

    scope_match = _SCOPE_QUERY_B_RE.match(text)
    if scope_match is not None:
        scope_keyword = _squash_whitespace(scope_match.group("scope"))
        query = _clean_query(scope_match.group("query"), allow_content_suffix_strip=True)
        if scope_keyword and query:
            if _scope_should_remain_plain_query(scope_keyword, raw_text=text, query=query):
                return None, None
            return scope_keyword, query

    return None, None


def _clean_query(text: str, *, allow_content_suffix_strip: bool = False) -> str:
    candidate = _squash_whitespace(text)
    if not candidate:
        return ""

    candidate = _strip_request_prefix(candidate)
    candidate = _strip_leading_fillers(candidate)
    candidate = _strip_leading_relation_words(candidate)
    normalized_natural_query = _normalize_natural_content_query(candidate)
    if normalized_natural_query:
        return normalized_natural_query
    if _is_content_only_query(candidate):
        return "内容"

    changed = True
    while changed and candidate:
        changed = False
        for pattern in _BASE_QUERY_SUFFIX_RE:
            stripped = pattern.sub("", candidate).strip()
            if stripped != candidate:
                if not stripped and _is_content_only_query(candidate):
                    return "内容"
                candidate = _squash_whitespace(stripped)
                candidate = _strip_leading_fillers(candidate)
                candidate = _strip_leading_relation_words(candidate)
                changed = True
                break

    if allow_content_suffix_strip:
        while candidate:
            stripped = re.sub(r"(?:的)?内容$", "", candidate).strip()
            if stripped == candidate:
                break
            candidate = _squash_whitespace(stripped)
            candidate = _strip_leading_fillers(candidate)
            candidate = _strip_leading_relation_words(candidate)

        while candidate.endswith("的"):
            candidate = _squash_whitespace(candidate[:-1])

    return _compact_cjk_spaces(_squash_whitespace(candidate))


def _normalize_natural_content_query(text: str) -> str | None:
    candidate = _squash_whitespace(text)
    if not candidate:
        return None

    write_match = _WRITE_CONTENT_QUERY_RE.match(candidate)
    if write_match is not None:
        subject = _clean_natural_query_subject(write_match.group("subject"))
        if subject:
            return _expand_natural_content_query(subject)

    content_match = _SUBJECT_CONTENT_QUERY_RE.match(candidate)
    if content_match is not None:
        subject = _clean_natural_query_subject(content_match.group("subject"))
        if subject:
            return _expand_natural_content_query(subject, default_terms=("包含内容",))

    return None


def _clean_natural_query_subject(text: str) -> str:
    subject = _squash_whitespace(text).strip(" 的")
    subject = re.sub(r"的(专利|权利要求|说明书|技术方案|技术交底书|交底书)$", r" \1", subject)
    return _compact_cjk_spaces(_squash_whitespace(subject))


def _expand_natural_content_query(subject: str, *, default_terms: tuple[str, ...] = ("内容",)) -> str:
    terms: list[str] = []
    if "专利" in subject:
        terms.extend(_PATENT_QUERY_EXPANSION_TERMS)
    elif any(marker in subject for marker in ("技术交底书", "交底书")):
        terms.extend(("技术方案", "内容"))
    else:
        terms.extend(default_terms)

    if not any(term in (subject + " ".join(terms)) for term in default_terms):
        terms.extend(default_terms)

    deduped_terms: list[str] = []
    for term in (*default_terms, *terms):
        if term not in subject and term not in deduped_terms:
            deduped_terms.append(term)
    return _squash_whitespace(" ".join([subject, *deduped_terms]))


def _is_content_only_query(text: str) -> bool:
    return bool(_CONTENT_ONLY_QUERY_RE.match(_squash_whitespace(text)))


def _scope_should_remain_plain_query(
    scope_keyword: str,
    *,
    raw_text: str,
    query: str | None = None,
) -> bool:
    scope_text = _squash_whitespace(scope_keyword)
    raw = _squash_whitespace(raw_text)
    query_text = _squash_whitespace(query or "")
    if not scope_text:
        return False
    if any(marker in scope_text for marker in _NATURAL_SCOPE_MARKERS):
        return True
    if any(kind in scope_text for kind in _DESCRIPTIVE_DOCUMENT_KIND_MARKERS):
        haystack = f"{raw} {query_text}"
        return any(marker in haystack for marker in _DOCUMENT_CONTENT_QUESTION_MARKERS)
    return False


def _strip_request_prefix(text: str) -> str:
    candidate = _REQUEST_PREFIX_RE.sub("", text, count=1)
    return _squash_whitespace(candidate)


def _strip_leading_fillers(text: str) -> str:
    candidate = text
    while candidate:
        stripped = _LEADING_FILLER_RE.sub("", candidate, count=1)
        if stripped == candidate:
            break
        candidate = _squash_whitespace(stripped)
    return candidate


def _strip_leading_relation_words(text: str) -> str:
    candidate = text
    while candidate:
        stripped = _LEADING_RELATION_RE.sub("", candidate, count=1)
        if stripped == candidate:
            break
        candidate = _squash_whitespace(stripped)
    return candidate


def _canonicalize_source_type_phrase(source_phrase: str) -> str:
    phrase = source_phrase.strip()
    lower = phrase.lower()

    if re.search(r"(?<![A-Za-z0-9_])(?:word|docx?)(?![A-Za-z0-9_])", lower) or "word文档" in lower:
        return "word"
    if re.search(r"(?<![A-Za-z0-9_])pdf(?![A-Za-z0-9_])", lower):
        return "pdf"
    if (
        re.search(r"(?<![A-Za-z0-9_])(?:pptx?|ppt)(?![A-Za-z0-9_])", lower)
        or "幻灯片" in phrase
        or "演示文稿" in phrase
    ):
        return "ppt"
    if (
        re.search(r"(?<![A-Za-z0-9_])(?:excel|xlsx?|xls)(?![A-Za-z0-9_])", lower)
        or "表格" in phrase
    ):
        return "excel"

    raise ValueError(f"unsupported source type phrase: {source_phrase}")


def _squash_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _compact_cjk_spaces(text: str) -> str:
    return re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse a natural-language search question into structured fields.")
    parser.add_argument("user_input", help="Natural-language search question.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        parsed = parse_search_query(args.user_input)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    print(json.dumps(parsed.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
