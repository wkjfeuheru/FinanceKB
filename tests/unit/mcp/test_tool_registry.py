from __future__ import annotations

import pytest
from mcp.server.fastmcp import FastMCP

from src.mcp.fundamentals import server as fundamentals
from src.mcp.knowledge import server as knowledge_server
from src.mcp.macro import server as macro
from src.mcp.market import server as market
from src.mcp.news import server as news

EXPECTED_TOOLS = {
    "market": {"resolve_symbol", "get_ohlcv", "get_realtime_quote"},
    "fundamentals": {
        "get_financial_statements",
        "get_valuation_metrics",
        "get_company_profile",
        "get_dividend_history",
    },
    "macro": {"get_macro_indicator", "get_economic_calendar"},
    "news": {"search_news", "get_filings", "get_earnings_calendar"},
    "knowledge": {"search_research", "search_policy"},
}

SERVERS = {
    "market": market,
    "fundamentals": fundamentals,
    "macro": macro,
    "news": news,
    "knowledge": knowledge_server,
}


def _tool_names(module) -> set[str]:
    mcp = FastMCP("test")
    module.register_tools(mcp)
    return {tool.name for tool in mcp._tool_manager.list_tools()}


@pytest.mark.parametrize("domain", sorted(SERVERS))
def test_each_server_registers_only_its_own_tools(domain: str) -> None:
    names = _tool_names(SERVERS[domain])
    assert names == EXPECTED_TOOLS[domain]
    for other, other_tools in EXPECTED_TOOLS.items():
        if other != domain:
            assert names.isdisjoint(other_tools)


def test_streamable_http_paths_are_domain_scoped() -> None:
    for domain, module in SERVERS.items():
        assert module.STREAMABLE_HTTP_PATH == f"/mcp/{domain}"
