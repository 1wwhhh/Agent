from __future__ import annotations

__all__ = ["ParsedSearchQuery", "parse_search_query"]


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from app.rag.query_parser import ParsedSearchQuery, parse_search_query

    return {
        "ParsedSearchQuery": ParsedSearchQuery,
        "parse_search_query": parse_search_query,
    }[name]
