"""按需查询沪深股票基本面。"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from typing import Annotated

from pydantic import BaseModel, Field

from src.queries.fundamentals_sources import (
    fetch_company_profile,
    fetch_dividend_history,
    fetch_financial_statements,
    fetch_trading_calendar,
    fetch_valuation_metrics,
)
from src.queries.fundamentals_types import (
    CompanyProfile,
    DividendEvent,
    FinancialStatement,
    Period,
    StockSymbol,
    ValuationMetric,
    shanghai_now,
)
from src.queries.market_sources import require_stock

from .base import QueryEnvelope, utc_now_iso

DOMAIN = "fundamentals"


def _result(model: type[BaseModel], items: list, reason: str = "") -> QueryEnvelope:
    return QueryEnvelope[model](
        available=not reason,
        domain=DOMAIN,
        reason=reason,
        as_of=utc_now_iso(),
        items=items,
    )


async def get_financial_statements(
    symbol: StockSymbol,
    period: Annotated[
        Period,
        Field(
            description="annual 只查年报。quarterly 查各季末（含年末），利润和现金流为年初至期末累计。"
        ),
    ] = "annual",
    years: Annotated[int, Field(ge=1, le=10, description="查询完整年度或季度窗口的年数。annual以本年1月1日、quarterly以本季度首日为不含端点的终点，再向前years年。")] = 3,
) -> QueryEnvelope[FinancialStatement]:
    """查询东财利润表、资产负债表和现金流量表，返回报告期、公告日、科目及单位。季报利润和现金流为年初至期末累计，资产负债为期末时点；Decimal数值以十进制字符串返回。"""
    await asyncio.to_thread(require_stock, symbol)
    today = shanghai_now().date()
    month = 1 if period == "annual" else (today.month - 1) // 3 * 3 + 1
    start, end = date(today.year - years, month, 1), date(today.year, month, 1)
    rows = await asyncio.to_thread(fetch_financial_statements, symbol, period, start, end)
    items = sorted(
        (row for row in rows if start <= row.period_date < end),
        key=lambda row: (row.period_date, row.statement),
        reverse=True,
    )
    reason = "" if items else "financial_statements_unavailable_in_requested_period"
    return _result(FinancialStatement, items, reason)


async def get_valuation_metrics(symbol: StockSymbol) -> QueryEnvelope[ValuationMetric]:
    """查询最近已收盘交易日的东财PE(TTM)、PB(MRQ)、PS(TTM)，附交易日和定义。北京时间15点后才考虑当天；目标交易日指标未更新时返回available=false。"""
    await asyncio.to_thread(require_stock, symbol)
    now = shanghai_now()
    cutoff = now.date() if now.hour >= 15 else now.date() - timedelta(days=1)
    calendar = await asyncio.to_thread(fetch_trading_calendar, cutoff - timedelta(days=21), cutoff)
    target = max((day for day in calendar.trade_days if day <= cutoff), default=None)
    if target is None:
        return _result(ValuationMetric, [], "valuation_no_closed_trading_date")
    rows = await asyncio.to_thread(fetch_valuation_metrics, symbol)
    items = [row for row in rows if row.trade_date == target]
    reason = "" if items else f"valuation_unavailable_for_trade_date:{target.isoformat()}"
    return _result(ValuationMetric, items, reason)


async def get_company_profile(symbol: StockSymbol) -> QueryEnvelope[CompanyProfile]:
    """查询公司概况、董监高、十大股东、十大流通股东和实际控制人，附各自来源与可用报告期。各组报告期可不同，缺失字段为null或空列表。"""
    await asyncio.to_thread(require_stock, symbol)
    company = await asyncio.to_thread(fetch_company_profile, symbol)
    return _result(CompanyProfile, [company])


async def get_dividend_history(
    symbol: StockSymbol,
    limit: Annotated[int, Field(ge=1, le=100, description="最多返回的分红事件数。")] = 20,
) -> QueryEnvelope[DividendEvent]:
    """按公告日倒序查询分红方案，返回方案状态、每股税前人民币金额、送转、报告期及可用公告、登记、除息、发放日期。"""
    await asyncio.to_thread(require_stock, symbol)
    rows = await asyncio.to_thread(fetch_dividend_history, symbol)
    items = sorted(rows, key=lambda row: row.announcement_date or date.min, reverse=True)
    return _result(DividendEvent, items[:limit])


__all__ = [
    "DOMAIN",
    "get_company_profile",
    "get_dividend_history",
    "get_financial_statements",
    "get_valuation_metrics",
]
