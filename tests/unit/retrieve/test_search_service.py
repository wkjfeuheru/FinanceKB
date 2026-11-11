from __future__ import annotations

from typing import Any

import pytest

from src.api.schemas.search import SearchFilters, SearchRequest
from src.core.config import POLICY_COLLECTION_NAME, REPORT_COLLECTION_NAME
from src.retrieve import search_service


class FakeKB:
    def __init__(self, rows: int = 2, pool: int = 24) -> None:
        self.rows = rows
        self.pool = pool
        self.calls: list[dict[str, Any]] = []

    def has_rows(self) -> bool:
        return self.rows > 0

    def hybrid_search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append({"query": query, **kwargs})
        k = min(int(kwargs["k"]), self.pool)
        return [
            {
                "id": f"chunk-{index}",
                "source": "中信证券-新能源汽车-2025.pdf",
                "title": "新能源汽车行业2025年展望",
                "content": f"内容 {index}",
                "score": 0.8,
                "rerank_score": 0.9,
                "chunk": index,
                "block_type": "text",
                "start_page": 3,
                "end_page": 4,
                "heading_path": "行业展望/需求",
                "date": "2025-06-01",
                "report_type": "行业",
                "broker": "中信证券",
                "ingested_at": 1_700_000_000,
            }
            for index in range(k)
        ]


@pytest.fixture
def fake_bases(monkeypatch: pytest.MonkeyPatch) -> dict[str, FakeKB]:
    bases = {
        REPORT_COLLECTION_NAME: FakeKB(rows=2),
        POLICY_COLLECTION_NAME: FakeKB(rows=0),
    }
    monkeypatch.setattr(search_service, "get_knowledge_base", lambda name: bases[name])
    return bases


def test_research_binds_report_collection(fake_bases: dict[str, FakeKB]) -> None:
    response = search_service.search_research(SearchRequest(query="新能源汽车"))
    assert response.collection == REPORT_COLLECTION_NAME
    assert response.items
    assert len(fake_bases[POLICY_COLLECTION_NAME].calls) == 0


def test_policy_binds_policy_collection(fake_bases: dict[str, FakeKB]) -> None:
    response = search_service.search_policy(SearchRequest(query="新能源汽车"))
    assert response.collection == POLICY_COLLECTION_NAME
    assert response.items == []
    assert response.has_more is False
    # 空集合必须短路：不触发嵌入/重排检索
    assert fake_bases[POLICY_COLLECTION_NAME].calls == []


def test_candidate_k_is_doubled_page_and_capped(
    fake_bases: dict[str, FakeKB], monkeypatch: pytest.MonkeyPatch
) -> None:
    kb = fake_bases[REPORT_COLLECTION_NAME]
    search_service.run_search("research", SearchRequest(query="新能源", page=2, page_size=8))
    assert kb.calls[-1]["k"] == 32

    monkeypatch.setattr(search_service, "MAX_CANDIDATE_CHUNKS", 10)
    search_service.run_search("research", SearchRequest(query="新能源", page=1, page_size=50))
    assert kb.calls[-1]["k"] == 10


def test_filters_are_converted_to_milvus_expression(fake_bases: dict[str, FakeKB]) -> None:
    kb = fake_bases[REPORT_COLLECTION_NAME]
    request = SearchRequest(
        query="新能源",
        filters=SearchFilters(
            broker=["中信证券"],
            report_type=["行业"],
            date_from="2025-01-01",
            date_to="2025-12-31",
        ),
    )
    search_service.search_research(request)
    assert kb.calls[-1]["filters"] == {
        "broker": ["中信证券"],
        "report_type": ["行业"],
        "date": {"gte": "2025-01-01", "lte": "2025-12-31"},
    }


def test_pagination_window_and_has_more(fake_bases: dict[str, FakeKB]) -> None:
    response = search_service.search_research(
        SearchRequest(query="新能源", page=1, page_size=8)
    )
    assert response.returned == 8
    assert response.has_more is True
    assert response.next_page == 2

    response = search_service.search_research(
        SearchRequest(query="新能源", page=3, page_size=8)
    )
    assert response.returned == 8
    assert response.has_more is False
    assert response.next_page is None


def test_ingested_at_epoch_is_iso_utc(fake_bases: dict[str, FakeKB]) -> None:
    response = search_service.search_research(SearchRequest(query="新能源"))
    assert response.items[0].ingested_at == "2023-11-14T22:13:20Z"
    assert response.as_of.endswith("Z")
