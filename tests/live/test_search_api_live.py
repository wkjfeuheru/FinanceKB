"""两条检索 API 的真实 Milvus 验证（默认跳过）。

执行::

    FINANCEKB_LIVE=1 pytest tests/live/test_search_api_live.py -m live
"""

from __future__ import annotations

import os

import pytest

from src.api.schemas.search import SearchFilters, SearchRequest
from src.core.config import POLICY_COLLECTION_NAME, REPORT_COLLECTION_NAME
from src.retrieve import search_service

pytestmark = pytest.mark.live

if os.getenv("FINANCEKB_LIVE") != "1":
    pytest.skip("设置 FINANCEKB_LIVE=1 后才连接真实 Milvus", allow_module_level=True)


def test_research_scope_returns_report_hits() -> None:
    response = search_service.search_research(
        SearchRequest(query="新能源汽车", page=1, page_size=5)
    )
    assert response.collection == REPORT_COLLECTION_NAME
    assert response.as_of.endswith("Z")
    for item in response.items:
        assert item.title
        assert item.ingested_at is None or item.ingested_at.endswith("Z")


def test_policy_scope_is_empty_but_healthy() -> None:
    response = search_service.search_policy(SearchRequest(query="补贴政策"))
    assert response.collection == POLICY_COLLECTION_NAME
    assert response.items == []
    assert response.has_more is False


def test_filters_are_accepted_by_real_collection() -> None:
    response = search_service.search_research(
        SearchRequest(
            query="行业展望",
            page=1,
            page_size=3,
            filters=SearchFilters(date_from="2020-01-01"),
        )
    )
    assert response.collection == REPORT_COLLECTION_NAME
