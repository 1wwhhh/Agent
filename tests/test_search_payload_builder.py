from __future__ import annotations

from app.rag.query_parser import ParsedSearchQuery, build_search_payload


def test_scope_keyword_maps_to_doc_id() -> None:
    parsed = ParsedSearchQuery.model_validate(
        {
            "query": "安全",
            "scope_keyword": "技术宝典",
            "source_type": None,
            "doc_id": None,
        }
    )

    payload = build_search_payload(parsed)

    assert payload == {
        "query": "安全",
        "top_k": 5,
        "doc_id": "技术宝典",
    }
    assert "scope_keyword" not in payload


def test_explicit_doc_id_wins_over_scope_keyword() -> None:
    parsed = ParsedSearchQuery.model_validate(
        {
            "query": "安全",
            "scope_keyword": "技术宝典",
            "source_type": "word",
            "doc_id": "_技术宝典_260518_docx",
        }
    )

    payload = build_search_payload(parsed)

    assert payload == {
        "query": "安全",
        "top_k": 5,
        "source_type": "word",
        "doc_id": "_技术宝典_260518_docx",
    }
    assert "scope_keyword" not in payload


def test_source_type_is_passed_through() -> None:
    parsed = ParsedSearchQuery.model_validate(
        {
            "query": "AGV 和 IGV 区别",
            "source_type": "word",
        }
    )

    payload = build_search_payload(parsed)

    assert payload == {
        "query": "AGV 和 IGV 区别",
        "top_k": 5,
        "source_type": "word",
    }


def test_doc_id_is_omitted_when_scope_and_doc_id_are_missing() -> None:
    parsed = ParsedSearchQuery.model_validate(
        {
            "query": "AGV 和 IGV 区别",
        }
    )

    payload = build_search_payload(parsed)

    assert payload == {
        "query": "AGV 和 IGV 区别",
        "top_k": 5,
    }
    assert "doc_id" not in payload
    assert "scope_keyword" not in payload


def test_optional_weights_and_time_filters_are_passed_through() -> None:
    parsed = ParsedSearchQuery.model_validate(
        {
            "query": "安全",
            "scope_keyword": "技术宝典",
        }
    )

    payload = build_search_payload(
        parsed,
        top_k=10,
        weight_m3=0.6,
        weight_zh=0.4,
        weight_sparse=0.2,
        doc_created_at_from=1716163200,
        doc_created_at_to=1718841599,
    )

    assert payload == {
        "query": "安全",
        "top_k": 10,
        "doc_id": "技术宝典",
        "weight_m3": 0.6,
        "weight_zh": 0.4,
        "weight_sparse": 0.2,
        "doc_created_at_from": 1716163200,
        "doc_created_at_to": 1718841599,
    }
    assert "scope_keyword" not in payload

def test_absolute_path_is_passed_through() -> None:
    parsed = ParsedSearchQuery.model_validate(
        {
            "query": "三七计划计划中有没有落地成功",
            "absolute_path": "\\9goo-nas\\部门\\考评记录2026年5月",
        }
    )

    payload = build_search_payload(parsed)

    assert payload == {
        "query": "三七计划计划中有没有落地成功",
        "top_k": 5,
        "absolute_path": "\\9goo-nas\\部门\\考评记录2026年5月",
    }
    assert "doc_id" not in payload
    assert "source_type" not in payload

