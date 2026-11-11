from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from src.api.schemas.search import SearchResponse
from src.mcp.common.server import StrictToolsMCP
from src.mcp.knowledge import client as knowledge_client
from src.mcp.knowledge import server as knowledge_server


def _response(collection: str = "finance_report", items: int = 1) -> SearchResponse:
    return SearchResponse(
        collection=collection,
        as_of="2026-11-20T08:00:00Z",
        page=1,
        page_size=8,
        returned=items,
        has_more=False,
        next_page=None,
        items=[
            {
                "id": f"chunk-{index}",
                "source": "报告.pdf",
                "title": "报告",
                "content": "内容",
                "score": 0.8,
                "rerank_score": 0.9,
                "chunk": index,
                "block_type": "text",
                "start_page": 1,
                "end_page": 2,
                "heading_path": "行业展望",
                "date": "2025-06-01",
                "report_type": "行业",
                "broker": "中信证券",
                "ingested_at": "2026-10-31T12:34:56Z",
            }
            for index in range(items)
        ],
    )


class FakeClient:
    def __init__(
        self,
        response: SearchResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response or _response()
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def search(self, scope: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((scope, payload))
        if self.error is not None:
            raise self.error
        return self.response.model_dump(mode="json")


def _server() -> FastMCP:
    mcp = StrictToolsMCP("test")
    knowledge_server.register_tools(mcp)
    return mcp


@pytest.mark.asyncio
async def test_search_research_proxies_to_research_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeClient()
    monkeypatch.setattr(knowledge_server, "get_client", lambda: fake)
    content, result = await _server().call_tool(
        "search_research",
        {"query": "新能源汽车", "page": 2, "page_size": 5},
    )
    scope, payload = fake.calls[0]
    assert scope == "research"
    assert payload["query"] == "新能源汽车"
    assert payload["page"] == 2
    assert payload["page_size"] == 5
    assert payload["filters"] == {
        "broker": [],
        "report_type": [],
        "date_from": None,
        "date_to": None,
    }
    assert result["collection"] == "finance_report"
    assert result["returned"] == 1
    assert json.loads(content[0].text) == result


@pytest.mark.asyncio
async def test_search_policy_proxies_to_policy_scope_and_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeClient(_response(collection="finance_policy", items=0))
    monkeypatch.setattr(knowledge_server, "get_client", lambda: fake)
    _, result = await _server().call_tool("search_policy", {"query": "补贴"})
    assert fake.calls[0][0] == "policy"
    assert result["collection"] == "finance_policy"
    assert result["items"] == []
    assert result["has_more"] is False


@pytest.mark.asyncio
async def test_search_policy_maps_date_filters_to_search_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeClient(_response(collection="finance_policy", items=0))
    monkeypatch.setattr(knowledge_server, "get_client", lambda: fake)
    await _server().call_tool(
        "search_policy",
        {"query": "补贴", "filters": {"date_from": "2025-01-01", "date_to": "2025-12-31"}},
    )
    filters = fake.calls[0][1]["filters"]
    assert filters["broker"] == []
    assert filters["report_type"] == []
    assert filters["date_from"] == "2025-01-01"
    assert filters["date_to"] == "2025-12-31"


@pytest.mark.asyncio
async def test_search_policy_rejects_research_only_filter_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeClient()
    monkeypatch.setattr(knowledge_server, "get_client", lambda: fake)
    with pytest.raises(ToolError):
        await _server().call_tool(
            "search_policy", {"query": "补贴", "filters": {"broker": ["中信证券"]}}
        )
    assert fake.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,message",
    [
        (401, "知识检索服务内部鉴权失败，请联系管理员"),
        (422, "检索参数无效，请检查 query / page / page_size / filters"),
        (503, "知识检索服务暂时不可用，请稍后重试"),
        (504, "知识检索服务暂时不可用，请稍后重试"),
    ],
)
async def test_upstream_errors_are_mapped_and_sanitized(
    monkeypatch: pytest.MonkeyPatch, status: int, message: str
) -> None:
    error = knowledge_client.KnowledgeAPIError(status, "internal-token-abc")
    monkeypatch.setattr(knowledge_server, "get_client", lambda: FakeClient(error=error))
    with pytest.raises(ToolError) as excinfo:
        await _server().call_tool("search_research", {"query": "新能源"})
    assert json.loads(str(excinfo.value))["error"]["message"] == message
    assert "internal-token-abc" not in str(excinfo.value)


@pytest.mark.asyncio
async def test_invalid_arguments_raise_tool_error_without_calling_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeClient()
    monkeypatch.setattr(knowledge_server, "get_client", lambda: fake)
    with pytest.raises(ToolError):
        await _server().call_tool("search_research", {"query": "新能源", "page_size": 500})
    assert fake.calls == []


def _patch_transport(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        knowledge_client.httpx,
        "AsyncClient",
        lambda timeout: real_client(transport=transport, timeout=timeout),
    )


@pytest.mark.asyncio
async def test_client_posts_to_scope_path_with_internal_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=_response().model_dump(mode="json"))

    _patch_transport(monkeypatch, handler)
    client = knowledge_client.KnowledgeAPIClient(base_url="http://app:8000", token="internal-token")
    payload = await client.search("policy", {"query": "补贴"})
    assert seen["url"] == "http://app:8000/api/v1/search/policy"
    assert seen["auth"] == "Bearer internal-token"
    assert payload["collection"] == "finance_report"


@pytest.mark.asyncio
@pytest.mark.parametrize("status,expected", [(401, 401), (422, 422), (503, 503)])
async def test_client_maps_http_status(
    monkeypatch: pytest.MonkeyPatch, status: int, expected: int
) -> None:
    _patch_transport(monkeypatch, lambda request: httpx.Response(status, json={}))
    client = knowledge_client.KnowledgeAPIClient(base_url="http://app:8000", token="t")
    with pytest.raises(knowledge_client.KnowledgeAPIError) as excinfo:
        await client.search("research", {"query": "x"})
    assert excinfo.value.status_code == expected


@pytest.mark.asyncio
async def test_client_maps_timeout_to_504(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    _patch_transport(monkeypatch, handler)
    client = knowledge_client.KnowledgeAPIClient(base_url="http://app:8000", token="t")
    with pytest.raises(knowledge_client.KnowledgeAPIError) as excinfo:
        await client.search("research", {"query": "x"})
    assert excinfo.value.status_code == 504


def test_client_timeout_defaults_to_120_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KNOWLEDGE_API_TIMEOUT_SECONDS", raising=False)
    client = knowledge_client.KnowledgeAPIClient(base_url="http://app:8000", token="t")
    assert client._timeout == 120.0


def test_client_timeout_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOWLEDGE_API_TIMEOUT_SECONDS", "180")
    client = knowledge_client.KnowledgeAPIClient(base_url="http://app:8000", token="t")
    assert client._timeout == 180.0
