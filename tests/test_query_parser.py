from __future__ import annotations

import json
import subprocess
import sys

import pytest

from app.rag.query_parser import parse_search_query


@pytest.mark.parametrize(
    ("user_input", "expected"),
    [
        (
            "在技术宝典中，关于安全的内容是什么",
            {
                "query": "安全",
                "scope_keyword": "技术宝典",
                "source_type": None,
                "doc_id": None,
            },
        ),
        (
            "技术宝典中的安全内容",
            {
                "query": "安全",
                "scope_keyword": "技术宝典",
                "source_type": None,
                "doc_id": None,
            },
        ),
        (
            "AGV调度系统专利里面都包含了哪些内容",
            {
                "query": "AGV调度系统专利 包含内容 技术方案 权利要求 摘要 说明书",
                "scope_keyword": None,
                "source_type": None,
                "doc_id": None,
            },
        ),
        (
            "集装箱锁扭机夹具的专利是怎么写的，里面的内容都有什么",
            {
                "query": "集装箱锁扭机夹具专利 内容 技术方案 权利要求 摘要 说明书",
                "scope_keyword": None,
                "source_type": None,
                "doc_id": None,
            },
        ),
        (
            "集装箱锁扭机夹具专利里面的内容都有什么",
            {
                "query": "集装箱锁扭机夹具专利 包含内容 技术方案 权利要求 摘要 说明书",
                "scope_keyword": None,
                "source_type": None,
                "doc_id": None,
            },
        ),
        (
            "集装箱锁扭机夹具的专利权利要求是什么",
            {
                "query": "集装箱锁扭机夹具的专利权利要求",
                "scope_keyword": None,
                "source_type": None,
                "doc_id": None,
            },
        ),
        (
            "集装箱锁扭机夹具技术交底书怎么写",
            {
                "query": "集装箱锁扭机夹具技术交底书 内容 技术方案",
                "scope_keyword": None,
                "source_type": None,
                "doc_id": None,
            },
        ),
        (
            "一种集装箱锁钮拆装用夹具及其使用方法.pdf里面的内容是什么",
            {
                "query": "内容",
                "scope_keyword": None,
                "source_type": "pdf",
                "doc_id": "一种集装箱锁钮拆装用夹具及其使用方法.pdf",
            },
        ),
        (
            "查询 word 文档中的 AGV 和 IGV 区别",
            {
                "query": "AGV 和 IGV 区别",
                "scope_keyword": None,
                "source_type": "word",
                "doc_id": None,
            },
        ),
        (
            "查询 PDF 中的安全协议",
            {
                "query": "安全协议",
                "scope_keyword": None,
                "source_type": "pdf",
                "doc_id": None,
            },
        ),
        (
            "AGV 和 IGV 区别是什么",
            {
                "query": "AGV 和 IGV 区别",
                "scope_keyword": None,
                "source_type": None,
                "doc_id": None,
            },
        ),
        (
            "查询 doc_id=abc123 中的安全内容",
            {
                "query": "安全",
                "scope_keyword": None,
                "source_type": None,
                "doc_id": "abc123",
            },
        ),
        (
            "请找到 机器人项目 (2026_05_01-2026_05_29).xlsx，请分析每个人写的下周的计划中，是否在下一周本周的工作中完成",
            {
                "query": "请分析每个人写的下周的计划中是否在下一周本周的工作中完成",
                "scope_keyword": None,
                "source_type": "excel",
                "doc_id": "机器人项目 (2026_05_01-2026_05_29).xlsx",
            },
        ),
        (
            "请根据九工机器南京研发中心机器人项目__2__xlsx,分析每个人写的下周的计划中，是否在下一周本周的工作中完成，哪些完成了哪些没完成",
            {
                "query": "分析每个人写的下周的计划中是否在下一周本周的工作中完成哪些完成了哪些没完成",
                "scope_keyword": None,
                "source_type": "excel",
                "doc_id": "九工机器南京研发中心机器人项目__2__xlsx",
            },
        ),
        (
            "比较周报.xlsx中每个人本周工作和下周计划",
            {
                "query": "比较每个人本周工作和下周计划",
                "scope_keyword": None,
                "source_type": "excel",
                "doc_id": "周报.xlsx",
            },
        ),
        (
            "请分析这个文件夹\\9goo-nas\\部门\\考评记录2026年5月，三七计划计划中有没有落地成功",
            {
                "query": "三七计划计划中有没有落地成功",
                "scope_keyword": None,
                "source_type": None,
                "doc_id": None,
                "relative_path": None,
                "absolute_path": "\\9goo-nas\\部门\\考评记录2026年5月",
            },
        ),
        (
            "\\9goo-nas\\部门\\考评记录2026年5月",
            {
                "query": "考评记录2026年5月",
                "scope_keyword": None,
                "source_type": None,
                "doc_id": None,
                "relative_path": None,
                "absolute_path": "\\9goo-nas\\部门\\考评记录2026年5月",
            },
        ),
        (
            "请根据分析9goo-nas/部门/考评记录2026年5月，三七计划中有没有落地成功",
            {
                "query": "三七计划中有没有落地成功",
                "scope_keyword": None,
                "source_type": None,
                "doc_id": None,
                "relative_path": None,
                "absolute_path": "9goo-nas/部门/考评记录2026年5月",
            },
        ),
    ],
)
def test_parse_search_query_examples(user_input: str, expected: dict[str, object]) -> None:
    parsed = parse_search_query(user_input)
    expected_payload = {"relative_path": None, "absolute_path": None, **expected}
    assert parsed.model_dump(mode="json") == expected_payload


def test_parse_search_query_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        parse_search_query("   ")


def test_query_parser_cli_outputs_json() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "app.rag.query_parser", "在技术宝典中，关于安全的内容是什么"],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    assert payload == {
        "query": "安全",
        "scope_keyword": "技术宝典",
        "source_type": None,
        "doc_id": None,
        "relative_path": None,
        "absolute_path": None,
    }
