from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import httpx
import pytest
import uvicorn
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import McpError
from mcp.types import INVALID_PARAMS

from mcp import ClientSession
from src.api.schemas.search import SearchResponse
from src.mcp.common import server as common_server
from src.mcp.fundamentals import server as fundamentals
from src.mcp.knowledge import server as knowledge
from src.mcp.macro import server as macro
from src.mcp.market import server as market
from src.mcp.news import server as news
from src.queries.market_types import Instrument

TOKEN = "integration-token"
SERVERS = {
    "market": market,
    "fundamentals": fundamentals,
    "macro": macro,
    "news": news,
    "knowledge": knowledge,
}


@contextmanager
def _running(app) -> Iterator[str]:
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("MCP test server failed to start")
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def _search_response() -> SearchResponse:
    return SearchResponse(
        collection="finance_report",
        as_of="2026-11-20T08:00:00Z",
        page=1,
        page_size=8,
        returned=1,
        has_more=False,
        next_page=None,
        items=[
            {
                "id": "chunk-1",
                "source": "报告.pdf",
                "title": "新能源汽车展望",
                "content": "需求保持增长",
                "score": 0.8,
                "rerank_score": 0.9,
                "chunk": 1,
                "block_type": "text",
                "start_page": 1,
                "end_page": 2,
                "heading_path": "行业展望",
                "date": "2025-06-01",
                "report_type": "行业",
                "broker": "中信证券",
                "ingested_at": "2026-10-31T12:34:56Z",
            }
        ],
    )


class FakeKnowledgeClient:
    async def search(self, scope: str, payload: dict) -> dict:
        return _search_response().model_dump(mode="json")


@pytest.mark.asyncio
async def test_unknown_structured_tool_is_a_jsonrpc_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(common_server, "MCP_API_TOKENS", (TOKEN,))
    with _running(macro.create_app()) as base:
        async with httpx.AsyncClient(headers={"Authorization": f"Bearer {TOKEN}"}) as http:
            async with streamable_http_client(f"{base}/mcp/macro", http_client=http) as (
                read,
                write,
                _,
            ):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    with pytest.raises(McpError) as exc:
                        await session.call_tool("get_financial_statements", {})
                    assert exc.value.error.code == INVALID_PARAMS
                    invalid = await session.call_tool("get_macro_indicator", {"indicator": "PPI"})
                    assert invalid.isError


@pytest.mark.asyncio
async def test_all_mcp_services_speak_streamable_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(common_server, "MCP_API_TOKENS", (TOKEN,))
    monkeypatch.setattr(knowledge, "get_client", lambda: FakeKnowledgeClient())
    monkeypatch.setattr(
        market.queries,
        "lookup_instruments",
        lambda _: [
            Instrument(
                symbol="600519.SH",
                name="贵州茅台",
                exchange="SSE",
                instrument_type="stock",
                source="infoway",
            )
        ],
    )

    apps = {name: module.create_app() for name, module in SERVERS.items()}
    with contextlib.ExitStack() as stack:
        urls = {name: stack.enter_context(_running(app)) for name, app in apps.items()}
        async with httpx.AsyncClient(timeout=15) as bare:
            for name, base in urls.items():
                assert (await bare.get(f"{base}/healthz")).status_code == 200
                unauthenticated = await bare.post(f"{base}/mcp/{name}")
                assert unauthenticated.status_code == 401, name

        for name, base in urls.items():
            async with httpx.AsyncClient(
                headers={"Authorization": f"Bearer {TOKEN}"}, timeout=15
            ) as http_client:
                url = f"{base}/mcp/{name}"
                async with streamable_http_client(url, http_client=http_client) as (
                    read,
                    write,
                    _,
                ):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        tools = await session.list_tools()
                        assert tools.tools, name
                        if name == "market":
                            result = await session.call_tool(
                                "resolve_symbol", {"keyword": "贵州茅台"}
                            )
                            assert result.structuredContent["available"] is True
                            assert result.structuredContent["items"][0]["symbol"] == "600519.SH"
                            assert result.structuredContent["domain"] == "market"
                        elif name == "knowledge":
                            result = await session.call_tool(
                                "search_research", {"query": "新能源汽车"}
                            )
                            assert result.structuredContent["collection"] == "finance_report"
                            assert result.structuredContent["items"]
