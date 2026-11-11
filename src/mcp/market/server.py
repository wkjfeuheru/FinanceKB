"""mcp-market：沪深股票与宽基指数行情。"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from src.mcp.common.server import StrictToolsMCP, build_mcp_app
from src.queries import market as queries

NAME = "mcp-market"
STREAMABLE_HTTP_PATH = "/mcp/market"
INSTRUCTIONS = (
    "查询沪深股票和首批宽基指数。先用 resolve_symbol 确认标准代码，再查历史 K 线或最新快照。"
    "行情时间与服务响应时间分别返回。"
)


def register_tools(mcp: FastMCP) -> None:
    for tool in (queries.resolve_symbol, queries.get_ohlcv, queries.get_realtime_quote):
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
