"""mcp-fundamentals：沪深股票财务、估值、公司资料、分红和主营构成。"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from src.mcp.common.server import StrictToolsMCP, build_mcp_app
from src.queries import fundamentals as queries

NAME = "mcp-fundamentals"
STREAMABLE_HTTP_PATH = "/mcp/fundamentals"
INSTRUCTIONS = (
    "按需查询沪深股票基本面。使用 resolve_symbol 得到的标准股票代码。"
    "财报和主营构成按报告期，估值按最近已收盘交易日，分红按公告日期查询。"
    "主营构成按产品、行业或地区分别查询，跨维度不可相加；源未单列储能时不得估算。"
)


def register_tools(mcp: FastMCP) -> None:
    for tool in (
        queries.get_financial_statements,
        queries.get_valuation_metrics,
        queries.get_company_profile,
        queries.get_dividend_history,
        queries.get_segment_revenue,
    ):
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
