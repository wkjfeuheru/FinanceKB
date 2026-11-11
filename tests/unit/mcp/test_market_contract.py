from __future__ import annotations

import json
from datetime import date

import httpx
import pytest
from mcp.server.fastmcp import FastMCP
from starlette.testclient import TestClient

from src.mcp.common import server as common_server
from src.mcp.market import server as market_server
from src.queries import market, market_sources
from src.queries.market_types import Bar, Instrument


@pytest.fixture
def catalog(monkeypatch):
    records = [
        {
            "symbol": "000001.SH",
            "name": "上证指数",
            "exchange": "SSE",
            "instrument_type": "index",
            "source": "eastmoney",
        },
        {
            "symbol": "000001.SZ",
            "name": "平安银行",
            "exchange": "SZSE",
            "instrument_type": "stock",
            "source": "eastmoney",
        },
        {
            "symbol": "000300.CSI",
            "name": "沪深300",
            "exchange": "CSI",
            "instrument_type": "index",
            "source": "eastmoney",
        },
    ]
    by_symbol = {row["symbol"]: row for row in records}
    bar = {
        "symbol": "000300.CSI",
        "instrument_type": "index",
        "source": "eastmoney",
        "interval": "1d",
        "time": "2026-11-06T00:00:00+08:00",
        "trade_date": "2026-11-06",
        "open": 4017.94,
        "high": 4069.07,
        "low": 4013.79,
        "close": 4054.28,
        "volume": 192152012,
        "volume_unit": "lot",
        "turnover": 478840399941.30,
        "price_unit": "point",
    }

    def bars(symbol, start, end, interval):
        assert (symbol.symbol, start.isoformat(), end.isoformat(), interval) == (
            "000300.CSI",
            "2026-11-02",
            "2026-11-06",
            "1d",
        )
        return [Bar.model_validate(bar)]

    monkeypatch.setattr(
        market_sources,
        "_eastmoney_stock_lookup",
        lambda keyword: [Instrument.model_validate(by_symbol["000001.SZ"])],
    )
    monkeypatch.setattr(market, "fetch_bars", bars)
    return by_symbol


def rpc(client, name, params, headers, request_id):
    response = client.post(
        "/mcp/market",
        headers=headers,
        json={"jsonrpc": "2.0", "id": request_id, "method": name, "params": params},
    )
    assert response.status_code == 200
    return response, json.loads(
        next(line[6:] for line in response.text.splitlines() if line.startswith("data: "))
    )


def test_market_http_publishes_typed_schemas_and_rejects_unknown_fields(monkeypatch, catalog):
    monkeypatch.setattr(common_server, "MCP_API_TOKENS", ("offline-market-test",))
    headers = {
        "Authorization": "Bearer offline-market-test",
        "Accept": "application/json, text/event-stream",
    }
    with TestClient(market_server.create_app()) as client:
        response, _ = rpc(
            client,
            "initialize",
            {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "market-contract-test", "version": "1"},
            },
            headers,
            1,
        )
        headers["Mcp-Session-Id"] = response.headers["mcp-session-id"]
        headers["MCP-Protocol-Version"] = "2025-11-25"
        response = client.post(
            "/mcp/market",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            },
        )
        assert response.status_code == 202
        _, discovered = rpc(client, "tools/list", {}, headers, 2)
        tools = {tool["name"]: tool for tool in discovered["result"]["tools"]}
        assert set(tools) == {"resolve_symbol", "get_ohlcv", "get_realtime_quote"}
        resolve = tools["resolve_symbol"]
        assert resolve["inputSchema"]["additionalProperties"] is False
        assert resolve["annotations"]["readOnlyHint"] is True
        assert resolve["outputSchema"]["properties"]["items"]["items"]["$ref"]
        _, invalid = rpc(
            client,
            "tools/call",
            {
                "name": "resolve_symbol",
                "arguments": {"keyword": "000001", "vendor": "other"},
            },
            headers,
            3,
        )
        assert invalid["result"]["isError"] is True
        _, wrong_type = rpc(
            client,
            "tools/call",
            {
                "name": "resolve_symbol",
                "arguments": {"keyword": 1},
            },
            headers,
            4,
        )
        assert wrong_type["result"]["isError"] is True
        _, invalid_date = rpc(
            client,
            "tools/call",
            {
                "name": "get_ohlcv",
                "arguments": {
                    "symbol": "000300.CSI",
                    "start_date": "2026-02-30",
                    "end_date": "2026-11-06",
                },
            },
            headers,
            5,
        )
        assert invalid_date["result"]["isError"] is True


@pytest.mark.asyncio
async def test_resolve_returns_index_and_stock_candidates_for_bare_code(catalog):
    mcp = FastMCP("market-contract-test")
    market_server.register_tools(mcp)
    _, result = await mcp.call_tool("resolve_symbol", {"keyword": "000001"})
    assert result["available"] is True
    assert [(row["symbol"], row["instrument_type"]) for row in result["items"]] == [
        ("000001.SH", "index"),
        ("000001.SZ", "stock"),
    ]


@pytest.mark.asyncio
async def test_ohlcv_reads_index_bar_with_point_and_volume_units(catalog):
    mcp = FastMCP("market-contract-test")
    market_server.register_tools(mcp)
    content, result = await mcp.call_tool(
        "get_ohlcv",
        {
            "symbol": "000300.CSI",
            "start_date": "2026-11-02",
            "end_date": "2026-11-06",
        },
    )
    assert result["available"] is True
    assert result["items"][0]["close"] == 4054.28
    assert result["items"][0]["price_unit"] == "point"
    assert result["items"][0]["volume_unit"] == "lot"
    assert json.loads(content[0].text) == result


@pytest.mark.asyncio
async def test_realtime_quote_returns_index_point_and_stock_price_in_request_order(
    catalog, monkeypatch
):
    def quotes(instruments):
        return [
            {
                "symbol": row["symbol"],
                "instrument_type": row["instrument_type"],
                "source": "sina",
                "source_time": "2026-11-09T15:43:39+08:00",
                "value": 4056.1081 if row["instrument_type"] == "index" else 11.89,
                "value_unit": "point" if row["instrument_type"] == "index" else "CNY",
            }
            for row in reversed(instruments)
        ]

    monkeypatch.setattr(market, "fetch_quotes", quotes, raising=False)
    mcp = FastMCP("market-contract-test")
    market_server.register_tools(mcp)
    _, result = await mcp.call_tool("get_realtime_quote", {"symbols": ["000300.CSI", "000001.SZ"]})
    assert result["available"] is True
    assert [(row["symbol"], row["value_unit"]) for row in result["items"]] == [
        ("000300.CSI", "point"),
        ("000001.SZ", "CNY"),
    ]


def test_eastmoney_suggest_resolves_sz_stock(monkeypatch):
    seen = {}

    def fake_get(url, *, timeout, params=None, headers=None, **kwargs):
        seen["url"] = url
        seen["params"] = params
        return httpx.Response(
            200,
            json={
                "QuotationCodeTable": {
                    "Status": 0,
                    "Data": [
                        {
                            "Code": "000001",
                            "MktNum": "0",
                            "Name": "平安银行",
                            "Classify": "AStock",
                        }
                    ],
                }
            },
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    instruments = market_sources._eastmoney_stock_lookup("000001")
    assert seen["url"] == "https://searchapi.eastmoney.com/api/suggest/get"
    assert seen["params"]["input"] == "000001"
    assert [(row.symbol, row.exchange) for row in instruments] == [("000001.SZ", "SZSE")]


def test_eastmoney_cn_bar_normalizes_lots_to_shares(monkeypatch):
    def fake_get(url, *, timeout, params=None, headers=None, **kwargs):
        assert "kline/get" in url
        assert params["secid"] == "1.600519"
        return httpx.Response(
            200,
            json={
                "rc": 0,
                "data": {
                    "code": "600519",
                    "market": 1,
                    "klines": ["2026-11-06 00:00,1440.8,1453,1460.5,1436.8,42848,6199528289.00"],
                },
            },
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    instrument = Instrument(
        symbol="600519.SH",
        name="贵州茅台",
        exchange="SSE",
        instrument_type="stock",
        source="eastmoney",
    )
    rows = market_sources.fetch_bars(instrument, date(2026, 11, 6), date(2026, 11, 6), "1d")
    assert len(rows) == 1
    assert rows[0].volume == 4_284_800
    assert rows[0].volume_unit == "share"
