from __future__ import annotations

import contextlib
import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal

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
from src.mcp.futures import server as futures
from src.mcp.knowledge import server as knowledge
from src.mcp.macro import server as macro
from src.mcp.market import server as market
from src.mcp.news import server as news
from src.queries import fundamentals_sources, futures_sources
from src.queries.market_types import Instrument

TOKEN = "integration-token"
SERVERS = {
    "market": market,
    "fundamentals": fundamentals,
    "macro": macro,
    "news": news,
    "knowledge": knowledge,
    "futures": futures,
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


@pytest.mark.asyncio
async def test_segment_revenue_streamable_http_discovery_call_and_validation(monkeypatch):
    monkeypatch.setattr(common_server, "MCP_API_TOKENS", (TOKEN,))
    stock_calls, source_calls = [], []
    monkeypatch.setattr(fundamentals.queries, "require_stock", stock_calls.append)
    monkeypatch.setattr(
        fundamentals_sources, "call_infoway", lambda *_: pytest.fail("unexpected Infoway request")
    )
    monkeypatch.setattr(
        httpx, "get", lambda *a, **kw: pytest.fail("unexpected external HTTP request")
    )

    def fake_source(url, params):
        source_calls.append((url, params))
        return {"zygcfx": [{
            "SECUCODE": "300750.SZ", "SECURITY_CODE": "300750",
            "REPORT_DATE": "2026-06-30 00:00:00",
            "MAINOP_TYPE": "2", "ITEM_NAME": "储能电池系统",
            "MAIN_BUSINESS_INCOME": "53260967000", "MAIN_BUSINESS_COST": "40500853000",
            "MAIN_BUSINESS_RPOFIT": "12760114000", "GROSS_RPOFIT_RATIO": "0.239577",
            "MBI_RATIO": "0.250000001",
        }]}

    monkeypatch.setattr(fundamentals_sources, "_eastmoney", fake_source)
    with _running(fundamentals.create_app()) as base:
        async with httpx.AsyncClient(timeout=15) as bare:
            assert (await bare.post(f"{base}/mcp/fundamentals")).status_code == 401
        async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {TOKEN}"}, timeout=15
        ) as http:
            async with streamable_http_client(f"{base}/mcp/fundamentals", http_client=http) as (
                read, write, _,
            ):
                async with ClientSession(read, write) as session:
                    initialized = await session.initialize()
                    assert initialized.capabilities.tools is not None
                    listed = await session.list_tools()
                    assert {tool.name for tool in listed.tools} == {
                        "get_financial_statements", "get_valuation_metrics", "get_company_profile",
                        "get_dividend_history", "get_segment_revenue",
                    }
                    tool = next(t for t in listed.tools if t.name == "get_segment_revenue")
                    assert tool.inputSchema["additionalProperties"] is False
                    assert tool.annotations.readOnlyHint is True and tool.outputSchema
                    result = await session.call_tool(
                        "get_segment_revenue",
                        {"symbol": "300750.SZ", "report_date": "2026-06-30", "dimension": "product"},
                    )
                    assert not result.isError
                    envelope = result.structuredContent
                    assert envelope["available"] is True and envelope["reason"] == ""
                    assert envelope["domain"] == "fundamentals"
                    (row,) = envelope["items"]
                    assert row["report_date"] == "2026-06-30" and row["dimension"] == "product"
                    assert row["segment_name"] == "储能电池系统"
                    assert row["revenue"] == "53260967000" and row["cost"] == "40500853000"
                    assert row["gross_profit"] == "12760114000" and row["gross_profit_basis"] == "source"
                    assert isinstance(row["gross_margin_pct"], str)
                    assert isinstance(row["revenue_share_pct"], str)
                    assert Decimal(row["gross_margin_pct"]) == Decimal("23.9577")
                    assert Decimal(row["revenue_share_pct"]) == Decimal("25.0000001")
                    assert row["currency"] is None and row["unit"] == "元" and row["source"] == "eastmoney"
                    for invalid in ({"report_date": "2026-02-29"}, {"dimension": "all"}, {"limit": 1}):
                        error = await session.call_tool(
                            "get_segment_revenue", {"symbol": "300750.SZ", **invalid}
                        )
                        assert error.isError
                        details = json.loads(error.content[0].text)["error"]
                        assert details["code"] == "invalid_argument" and details["retryable"] is False
    assert stock_calls == ["300750.SZ"]
    assert source_calls == [(
        "https://emweb.securities.eastmoney.com/PC_HSF10/BusinessAnalysis/PageAjax",
        {"code": "SZ300750"},
    )]


@pytest.mark.asyncio
async def test_commodity_price_streamable_http_discovery_call_and_validation(monkeypatch):
    monkeypatch.setattr(common_server, "MCP_API_TOKENS", (TOKEN,))
    source_calls = []
    monkeypatch.setattr(
        httpx, "get", lambda *a, **kw: pytest.fail("unexpected external HTTP request")
    )

    def fake_get(url, *, source, timeout=8.0, headers=None, params=None, **kwargs):
        source_calls.append((url, dict(params)))
        if params == {"msgid": "225"}:
            body = [
                {"mktid": "225", "mktname": "广期所", "mktshort": "GFEX",
                 "rtime": "2026-11-11 17:43:28", "vcode": "si", "vname": "工业硅", "vtype": "1"},
                {"mktid": "225", "mktname": "广期所", "mktshort": "GFEX",
                 "rtime": "2026-11-11 17:43:28", "vcode": "lc", "vname": "碳酸锂", "vtype": "2"},
                {"mktid": "225", "mktname": "广期所", "mktshort": "GFEX",
                 "rtime": "2026-11-11 17:43:28", "vcode": "ps", "vname": "多晶硅", "vtype": "3"},
            ]
        elif params == {"msgid": "225_2"}:
            body = [
                {"code": "lc2705", "linetype": 0, "mktid": 225, "mktname": "广期所",
                 "mktshort": "GFEX", "name": "碳酸锂2705", "operno": 213279,
                 "rtime": "2026-11-12 08:39:30", "scale": 0, "typeid": 2,
                 "uptime": "2026-11-12 08:39:26.560780000",
                 "vcode": "lc", "vname": "碳酸锂", "vtype": "2"},
                {"code": "lcm", "linetype": 0, "mktid": 225, "mktname": "广期所",
                 "mktshort": "GFEX", "name": "碳酸锂主连", "operno": 213279,
                 "rtime": "2026-11-12 08:39:30", "scale": 0, "typeid": 2,
                 "uptime": "2026-11-12 08:39:26.562541000",
                 "vcode": "lc", "vname": "碳酸锂", "vtype": "2"},
                {"code": "lcs", "linetype": 0, "mktid": 225, "mktname": "广期所",
                 "mktshort": "GFEX", "name": "碳酸锂次主连", "operno": 213279,
                 "rtime": "2026-11-12 08:39:30", "scale": 0, "typeid": 2,
                 "uptime": "2026-11-12 08:39:26.565745000",
                 "vcode": "lc", "vname": "碳酸锂", "vtype": "2"},
            ]
        else:
            body = {"rc": 0, "rt": 4, "svr": 183120281, "lt": 1, "full": 1, "dlmkts": "",
                    "dsc": "0", "data": {"f43": 88880.0, "f57": "lcm", "f58": "碳酸锂主连",
                                         "f59": 0, "f86": 1794464936}}
        return httpx.Response(
            200,
            request=httpx.Request("GET", url, params=params),
            content=json.dumps(body).encode(),
        )

    monkeypatch.setattr(futures_sources, "get", fake_get)
    with _running(futures.create_app()) as base:
        async with httpx.AsyncClient(timeout=15) as bare:
            assert (await bare.post(f"{base}/mcp/futures")).status_code == 401
        async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {TOKEN}"}, timeout=15
        ) as http:
            async with streamable_http_client(f"{base}/mcp/futures", http_client=http) as (
                read, write, _,
            ):
                async with ClientSession(read, write) as session:
                    initialized = await session.initialize()
                    assert initialized.capabilities.tools is not None
                    listed = await session.list_tools()
                    assert {tool.name for tool in listed.tools} == {"get_commodity_price"}
                    tool = listed.tools[0]
                    assert tool.inputSchema["additionalProperties"] is False
                    assert tool.annotations.readOnlyHint is True and tool.outputSchema
                    result = await session.call_tool(
                        "get_commodity_price", {"symbol": "225.LCM"}
                    )
                    assert not result.isError
                    envelope = result.structuredContent
                    assert envelope["available"] is True and envelope["reason"] == ""
                    assert envelope["domain"] == "futures" and envelope["as_of"]
                    (row,) = envelope["items"]
                    assert row["symbol"] == "225.lcm" and row["contract_code"] == "lcm"
                    assert row["contract_name"] == "碳酸锂主连"
                    assert row["commodity_code"] == "lc" and row["commodity_name"] == "碳酸锂"
                    assert row["exchange"] == "GFEX"
                    assert row["contract_type"] == "main_continuous"
                    assert row["price"] == "88880.0"
                    assert row["currency"] is None and row["price_unit"] is None
                    assert row["source_time"] == "2026-11-12T14:28:56+08:00"
                    assert row["source"] == "eastmoney"
                    for invalid in ("225.lcs", "220.ifm", "600519.SH"):
                        error = await session.call_tool(
                            "get_commodity_price", {"symbol": invalid}
                        )
                        assert error.isError
                        details = json.loads(error.content[0].text)["error"]
                        assert details["code"] == "invalid_argument"
                        assert details["retryable"] is False
    assert source_calls == [
        ("https://futsse-static.eastmoney.com/redis", {"msgid": "225"}),
        ("https://futsse-static.eastmoney.com/redis", {"msgid": "225_2"}),
        ("https://push2.eastmoney.com/api/qt/stock/get",
         {"secid": "225.lcm", "fltt": "2", "fields": "f43,f57,f58,f59,f86"}),
    ]
