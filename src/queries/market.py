"""沪深股票与首批宽基指数行情查询。"""

from __future__ import annotations

import asyncio
from typing import Annotated

from pydantic import Field

from src.queries.market_sources import (
    fetch_bars,
    fetch_quotes,
    lookup_instruments,
    require_instrument,
)
from src.queries.market_types import (
    MAX_BARS,
    Bar,
    DateArgument,
    Instrument,
    Interval,
    Quote,
    Symbol,
    date_range,
)

from .base import QueryEnvelope, utc_now_iso

DOMAIN = "market"


async def resolve_symbol(
    keyword: Annotated[
        str,
        Field(
            min_length=1,
            max_length=64,
            description="股票或指数的简称、裸代码或标准代码。",
        ),
    ],
) -> QueryEnvelope[Instrument]:
    """精确匹配沪深股票及上证指数、深证成指、沪深300的简称或代码，返回标准代码、名称、交易所、类型和来源。歧义返回候选；股票经东财搜索接口实时解析。"""
    query = "".join(keyword.split()).casefold()
    if not query:
        raise ValueError("keyword 不能为空")
    items = await asyncio.to_thread(lookup_instruments, query)
    return QueryEnvelope[Instrument](
        available=True, domain=DOMAIN, as_of=utc_now_iso(), items=items
    )


async def get_ohlcv(
    symbol: Symbol,
    start_date: DateArgument,
    end_date: DateArgument,
    interval: Annotated[
        Interval,
        Field(
            description="K 线周期，指数支持 1d、1w、1M，股票支持全部枚举周期。",
        ),
    ] = "1d",
) -> QueryEnvelope[Bar]:
    """按日期查询最多5000根OHLCV及来源时间、成交额和单位，K线统一来自东财。股票不复权、价格为元、成交量为股，指数为点和手。历史边界：日线可回溯多年，分钟级仅限近期（1分钟约5个交易日，5/15/30/60分钟约最近7周），覆盖不足或超量报错。"""
    start, end = date_range(start_date, end_date)
    instrument = await asyncio.to_thread(require_instrument, symbol)
    records = await asyncio.to_thread(fetch_bars, instrument, start, end, interval)
    if len(records) > MAX_BARS:
        raise ValueError(f"单次最多返回 {MAX_BARS} 根 K 线，请缩小日期范围")
    return QueryEnvelope[Bar](
        available=bool(records),
        domain=DOMAIN,
        reason="" if records else "market_history_unavailable_in_requested_period",
        as_of=utc_now_iso(),
        items=[Bar.model_validate(row) for row in records],
    )


async def get_realtime_quote(
    symbols: Annotated[
        list[Symbol],
        Field(
            min_length=1,
            max_length=20,
            json_schema_extra={"uniqueItems": True},
            description="1 至 20 个互不重复的股票或指数标准代码。",
        ),
    ],
) -> QueryEnvelope[Quote]:
    """查询股票价格或指数点位及可用涨跌幅，附来源和时间；快照统一来自新浪。收盘后时间可停留在交易时段。任一标的缺报价时整体available=false且items为空。"""
    if len(set(symbols)) != len(symbols):
        raise ValueError("symbols 不得包含重复标的")
    instruments = await asyncio.to_thread(
        lambda: [require_instrument(symbol).model_dump() for symbol in symbols]
    )
    quotes = await asyncio.to_thread(fetch_quotes, instruments)
    by_symbol = {row["symbol"]: Quote.model_validate(row) for row in quotes}
    complete = all(symbol in by_symbol for symbol in symbols)
    return QueryEnvelope[Quote](
        available=complete,
        domain=DOMAIN,
        reason="" if complete else "market_quote_unavailable",
        as_of=utc_now_iso(),
        items=[by_symbol[symbol] for symbol in symbols] if complete else [],
    )


__all__ = ["DOMAIN", "get_ohlcv", "get_realtime_quote", "resolve_symbol"]
