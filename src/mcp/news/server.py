"""mcp-news：东财新闻、公司公告与财报披露日程。"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from src.mcp.common.server import StrictToolsMCP, build_mcp_app
from src.queries import news as queries

NAME = "mcp-news"
STREAMABLE_HTTP_PATH = "/mcp/news"
INSTRUCTIONS = "按需查询东财新闻、沪深个股公告及财报披露日程。个股公告使用标准股票代码。"


def register_tools(mcp: FastMCP) -> None:
    for tool in (queries.search_news, queries.get_filings, queries.get_earnings_calendar):
        mcp.add_tool(tool, annotations=ToolAnnotations(readOnlyHint=True))


def create_app():
    return build_mcp_app(
        NAME,
        STREAMABLE_HTTP_PATH,
        register_tools,
        instructions=INSTRUCTIONS,
        mcp_class=StrictToolsMCP,
    )


__all__ = ["NAME", "STREAMABLE_HTTP_PATH", "create_app", "register_tools"]
