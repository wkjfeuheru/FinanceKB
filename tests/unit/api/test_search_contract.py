"""检索 API 请求合同：未知字段与非法值必须 422，而不是静默忽略。"""

import pytest
from fastapi import FastAPI
from pydantic import ValidationError
from starlette.testclient import TestClient

from src.api import dependencies
from src.api.routes.search import router
from src.api.schemas.search import SearchFilters, SearchRequest, SearchResponse
from src.core.exceptions import VectorStoreError
from src.retrieve import search_service


def test_unknown_filter_key_is_rejected():
    for key in ("category", "industry", "collection", "tenant_id"):
        with pytest.raises(ValidationError):
            SearchFilters(**{key: "x"})


def test_collection_cannot_be_specified_on_request():
    with pytest.raises(ValidationError):
        SearchRequest(query="新能源", collection="finance_kb")


def test_report_type_must_be_known():
    with pytest.raises(ValidationError):
        SearchFilters(report_type=["点评"])


def test_date_format_and_order_are_validated():
    with pytest.raises(ValidationError):
        SearchFilters(date_from="2025/01/01")
    with pytest.raises(ValidationError):
        SearchFilters(date_from="2025-12-31", date_to="2025-01-01")


@pytest.mark.parametrize("page,page_size", [(0, 8), (1, 0), (1, 51), (21, 10)])
def test_pagination_bounds(page, page_size):
    with pytest.raises(ValidationError):
        SearchRequest(query="新能源", page=page, page_size=page_size)


def test_query_length_bounds():
    with pytest.raises(ValidationError):
        SearchRequest(query="   ")
    with pytest.raises(ValidationError):
        SearchRequest(query="x" * 501)


def test_filters_map_to_retriever_shape():
    filters = SearchFilters(
        broker=["中信证券"], report_type=["行业"],
        date_from="2025-01-01", date_to="2025-12-31",
    )

    assert filters.to_milvus() == {
        "broker": ["中信证券"],
        "report_type": ["行业"],
        "date": {"gte": "2025-01-01", "lte": "2025-12-31"},
    }


def test_empty_filters_produce_no_expression():
    assert SearchFilters().to_milvus() == {}


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/search")
    return app


def _response(collection: str) -> SearchResponse:
    return SearchResponse(
        collection=collection,
        as_of="2026-11-20T08:00:00Z",
        page=1,
        page_size=8,
        returned=0,
        has_more=False,
        next_page=None,
        items=[],
    )


def test_routes_require_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dependencies, "DATA_API_TOKENS", ("secret",))
    with TestClient(_app()) as client:
        assert client.post("/api/v1/search/research", json={"query": "x"}).status_code == 401
        response = client.post(
            "/api/v1/search/research",
            json={"query": "x"},
            headers={"Authorization": "Bearer wrong"},
        )
        assert response.status_code == 401


def test_routes_return_fixed_collections(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dependencies, "DATA_API_TOKENS", ("secret",))
    monkeypatch.setattr(
        search_service, "search_research", lambda req: _response("finance_report")
    )
    monkeypatch.setattr(
        search_service, "search_policy", lambda req: _response("finance_policy")
    )
    headers = {"Authorization": "Bearer secret"}
    with TestClient(_app()) as client:
        research = client.post(
            "/api/v1/search/research", json={"query": "新能源"}, headers=headers
        )
        policy = client.post(
            "/api/v1/search/policy", json={"query": "新能源"}, headers=headers
        )
    assert research.status_code == 200
    assert research.json()["collection"] == "finance_report"
    assert research.json()["as_of"] == "2026-11-20T08:00:00Z"
    assert policy.status_code == 200
    assert policy.json()["items"] == []


def test_routes_do_not_accept_collection_or_unknown_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dependencies, "DATA_API_TOKENS", ("secret",))
    headers = {"Authorization": "Bearer secret"}
    with TestClient(_app()) as client:
        assert client.post(
            "/api/v1/search/research",
            json={"query": "x", "collection": "finance_kb"},
            headers=headers,
        ).status_code == 422
        assert client.post(
            "/api/v1/search/research",
            json={"query": "x", "filters": {"industry": ["银行"]}},
            headers=headers,
        ).status_code == 422


def test_vector_store_failure_becomes_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dependencies, "DATA_API_TOKENS", ("secret",))

    def _boom(req):
        raise VectorStoreError("milvus down")

    monkeypatch.setattr(search_service, "search_research", _boom)
    with TestClient(_app()) as client:
        response = client.post(
            "/api/v1/search/research",
            json={"query": "新能源"},
            headers={"Authorization": "Bearer secret"},
        )
    assert response.status_code == 503
    assert "milvus down" not in response.text
