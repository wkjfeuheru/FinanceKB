"""mcp-futures：东财商品期货最新报价。"""

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from src.mcp.common.server import StrictToolsMCP, build_mcp_app
from src.queries import futures as queries

NAME = "mcp-futures"
STREAMABLE_HTTP_PATH = "/mcp/futures"
INSTRUCTIONS = (
    "查询东财商品期货最新报价，使用东财合约标识，如225.lcm碳酸锂主连、"
    "225.lc2705月份合约、113.cum沪铜主连。"
    "期货不等于现货，主连不等于交割合约，最新价不等于结算价；"
    "源更新时间不等于交易日，报价可能延迟。币种或计价单位为null时不得自行推断。"
)


def register_tools(mcp: FastMCP) -> None:
    mcp.add_tool(queries.get_commodity_price, annotations=ToolAnnotations(readOnlyHint=True))


def create_app():
    return build_mcp_app(
        NAME,
        STREAMABLE_HTTP_PATH,
        register_tools,
        instructions=INSTRUCTIONS,
        mcp_class=StrictToolsMCP,
    )


__all__ = ["NAME", "STREAMABLE_HTTP_PATH", "create_app", "register_tools"]
