"""mcp-macro：宏观指标与经济日历。"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from src.mcp.common.server import StrictToolsMCP, build_mcp_app
from src.queries import macro as queries

NAME = "mcp-macro"
STREAMABLE_HTTP_PATH = "/mcp/macro"
INSTRUCTIONS = "按需查询中国 CPI、GDP 及国家统计局主要统计信息计划发布日程。"


def register_tools(mcp: FastMCP) -> None:
    for tool in (queries.get_macro_indicator, queries.get_economic_calendar):
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
