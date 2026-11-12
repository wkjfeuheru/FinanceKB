"""按需查询东财商品期货最新报价。"""

import asyncio

from src.queries.base import QueryEnvelope, utc_now_iso
from src.queries.futures_sources import fetch_commodity_price
from src.queries.futures_types import CommodityPrice, FuturesSymbol

DOMAIN = "futures"


async def get_commodity_price(symbol: FuturesSymbol) -> QueryEnvelope[CommodityPrice]:
    """查询东财商品期货主连或月份合约的最新报价，返回合约身份、价格和源更新时间。仅支持上期所、大商所、郑商所、上期能源、广期所当前目录；不含现货、金融期货、期权、次主连或历史K线。主连不是交割合约，最新价不是结算价；可能延迟或停留在旧时点。未核验的币种、计价单位为null，缺价返回available=false；无此合约或源异常为工具错误。"""
    quote = await asyncio.to_thread(fetch_commodity_price, symbol)
    return QueryEnvelope[CommodityPrice](
        available=quote is not None,
        domain=DOMAIN,
        reason="" if quote is not None else f"commodity_price_unavailable:{symbol}",
        as_of=utc_now_iso(),
        items=[quote] if quote is not None else [],
    )


__all__ = ["DOMAIN", "get_commodity_price"]
