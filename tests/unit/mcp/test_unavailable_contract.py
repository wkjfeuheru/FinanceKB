from __future__ import annotations

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from src.mcp.common.server import StrictToolsMCP
from src.mcp.macro import server as macro
from src.mcp.market import server as market
from src.mcp.news import server as news
from src.queries.content_http import ContentSourceError

CASES = [
    (
        macro,
        "get_macro_indicator",
        {"indicator": "CPI"},
    ),
    (
        macro,
        "get_economic_calendar",
        {"start_date": "2026-11-10", "end_date": "2026-11-10"},
    ),
    (
        news,
        "search_news",
        {"keyword": "新能源汽车"},
    ),
    (
        news,
        "get_filings",
        {"symbol": "600519.SH"},
    ),
    (
        news,
        "get_earnings_calendar",
        {},
    ),
]


def _server(module) -> FastMCP:
    mcp = StrictToolsMCP("test")
    module.register_tools(mcp)
    return mcp


@pytest.mark.asyncio
@pytest.mark.parametrize("module,tool,arguments", CASES)
async def test_source_failures_surface_as_tool_errors(
    module, tool: str, arguments: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def missing(*args):
        raise ContentSourceError("来源请求失败")

    monkeypatch.setattr(module.queries, "fetch_content", missing)
    if module is news:
        monkeypatch.setattr(news.queries, "require_eastmoney_stock", lambda _: None)
    with pytest.raises(ToolError, match="来源请求失败"):
        await _server(module).call_tool(tool, arguments)


@pytest.mark.asyncio
async def test_unpublished_calendar_year_returns_unavailable() -> None:
    _, structured = await _server(macro).call_tool(
        "get_economic_calendar", {"start_date": "2027-01-01", "end_date": "2027-01-01"}
    )
    assert structured["available"] is False
    assert structured["domain"] == "macro"
    assert structured["reason"] == "economic_calendar_unavailable_for_year:2027"
    assert structured["items"] == []
    assert structured["as_of"].endswith("Z")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "module, tool, args",
    [
        (macro, "get_macro_indicator", {"indicator": "PPI"}),
        (macro, "get_economic_calendar", {"start_date": "2026-02-30"}),
        (news, "search_news", {"keyword": "   "}),
        (news, "search_news", {"keyword": "半导体", "limit": "10"}),
        (news, "get_filings", {"symbol": "600519"}),
        (news, "get_earnings_calendar", {"start_date": "2026-10-31", "end_date": "2026-10-01"}),
    ],
)
async def test_macro_and_news_reject_invalid_arguments(module, tool, args) -> None:
    with pytest.raises(ToolError):
        await _server(module).call_tool(tool, args)


@pytest.mark.asyncio
@pytest.mark.parametrize("module", [macro, news])
async def test_tools_publish_typed_read_only_contracts(module) -> None:
    for tool in await _server(module).list_tools():
        assert tool.annotations.readOnlyHint is True
        assert tool.inputSchema["additionalProperties"] is False
        assert tool.outputSchema["properties"]["items"]["items"]["$ref"]


@pytest.mark.asyncio
async def test_invalid_arguments_are_rejected_not_faked() -> None:
    with pytest.raises(ToolError):
        await _server(market).call_tool("resolve_symbol", {"keyword": "   "})
    with pytest.raises(ToolError):
        await _server(market).call_tool(
            "get_ohlcv",
            {
                "symbol": "600519.SH",
                "start_date": "2025-01-01",
                "end_date": "2025-12-31",
                "interval": "2h",
            },
        )
