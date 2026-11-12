from __future__ import annotations

import pytest
from mcp.server.fastmcp import FastMCP

from src.mcp.common.server import StrictToolsMCP
from src.mcp.fundamentals import server as fundamentals
from src.mcp.futures import server as futures_server
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
        "get_segment_revenue",
    },
    "macro": {"get_macro_indicator", "get_economic_calendar"},
    "news": {"search_news", "get_filings", "get_earnings_calendar"},
    "knowledge": {"search_research", "search_policy"},
    "futures": {"get_commodity_price"},
}

SERVERS = {
    "market": market,
    "fundamentals": fundamentals,
    "macro": macro,
    "news": news,
    "knowledge": knowledge_server,
    "futures": futures_server,
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


@pytest.mark.asyncio
async def test_public_discovery_has_six_servers_and_sixteen_tools():
    discovered = {}
    for domain, module in SERVERS.items():
        server = StrictToolsMCP(f"discovery-{domain}")
        module.register_tools(server)
        discovered[domain] = {tool.name for tool in await server.list_tools()}
    assert discovered == EXPECTED_TOOLS
    assert len(discovered) == 6
    assert len(discovered["fundamentals"]) == 5
    assert sum(map(len, discovered.values())) == 16
    assert len(set.union(*discovered.values())) == 16


@pytest.mark.asyncio
async def test_segment_revenue_publishes_strict_described_typed_schema():
    server = StrictToolsMCP("segment-schema")
    fundamentals.register_tools(server)
    tool = next(tool for tool in await server.list_tools() if tool.name == "get_segment_revenue")
    assert tool.description.strip()
    assert tool.annotations.readOnlyHint is True
    inputs = tool.inputSchema
    assert inputs["additionalProperties"] is False
    assert inputs["required"] == ["symbol"]
    assert set(inputs["properties"]) == {"symbol", "report_date", "dimension"}
    assert all(field.get("description", "").strip() for field in inputs["properties"].values())
    assert inputs["properties"]["symbol"]["type"] == "string"
    assert inputs["properties"]["report_date"]["default"] is None
    assert {entry["type"] for entry in inputs["properties"]["report_date"]["anyOf"]} == {"string", "null"}
    assert inputs["properties"]["dimension"]["default"] == "product"
    assert set(inputs["properties"]["dimension"]["enum"]) == {"product", "industry", "region"}

    output = tool.outputSchema
    assert output and output["type"] == "object"
    assert {"available", "reason", "domain", "as_of", "items"} <= output["properties"].keys()
    items = output["properties"]["items"]
    assert items["type"] == "array"
    model = output["$defs"][items["items"]["$ref"].rsplit("/", 1)[-1]]
    fields = model["properties"]
    assert set(fields) == {
        "symbol", "report_date", "dimension", "segment_name", "revenue", "cost",
        "gross_profit", "gross_profit_basis", "gross_margin_pct", "revenue_share_pct",
        "currency", "unit", "source",
    }
    assert all(field.get("description", "").strip() for field in fields.values())
    assert fields["report_date"]["format"] == "date"
    assert set(fields["dimension"]["enum"]) == {"product", "industry", "region"}
    for name in ("revenue", "cost", "gross_profit", "gross_margin_pct", "revenue_share_pct"):
        # FastMCP publishes a validation schema; call_tool tests check wire strings.
        types = {entry["type"] for entry in fields[name]["anyOf"]}
        assert {"string", "null"} <= types <= {"string", "number", "null"}, name


@pytest.mark.asyncio
async def test_commodity_price_publishes_strict_described_typed_schema():
    server = StrictToolsMCP("futures-schema")
    futures_server.register_tools(server)
    tool = next(tool for tool in await server.list_tools() if tool.name == "get_commodity_price")
    assert tool.description.strip()
    assert tool.annotations.readOnlyHint is True
    inputs = tool.inputSchema
    assert inputs["additionalProperties"] is False
    assert inputs["required"] == ["symbol"]
    assert set(inputs["properties"]) == {"symbol"}
    assert inputs["properties"]["symbol"]["description"].strip()
    assert inputs["properties"]["symbol"]["type"] == "string"
    assert inputs["properties"]["symbol"]["pattern"].startswith("^(113|114|115|142|225)")
    assert inputs["properties"]["symbol"]["maxLength"] == 16

    output = tool.outputSchema
    assert output and output["type"] == "object"
    assert {"available", "reason", "domain", "as_of", "items"} <= output["properties"].keys()
    items = output["properties"]["items"]
    assert items["type"] == "array"
    model = output["$defs"][items["items"]["$ref"].rsplit("/", 1)[-1]]
    fields = model["properties"]
    assert set(fields) == {
        "symbol", "contract_code", "contract_name", "commodity_code", "commodity_name",
        "exchange", "contract_type", "price", "currency", "price_unit", "source_time", "source",
    }
    assert all(field.get("description", "").strip() for field in fields.values())
    assert set(fields["exchange"]["enum"]) == {"SHFE", "DCE", "CZCE", "INE", "GFEX"}
    assert set(fields["contract_type"]["enum"]) == {"main_continuous", "delivery"}
    assert fields["source"]["const"] == "eastmoney"
    # FastMCP publishes a validation schema; call_tool tests check wire strings.
    types = {entry["type"] for entry in fields["price"]["anyOf"]}
    assert {"string", "number"} <= types <= {"string", "number", "null"}
    for name in ("currency", "price_unit", "source_time"):
        optional = {entry["type"] for entry in fields[name]["anyOf"]}
        assert {"string", "null"} <= optional <= {"string", "number", "null"}, name
