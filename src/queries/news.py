"""按需查询东财新闻、公司公告和业绩日历。"""

import asyncio
from typing import Annotated

from pydantic import Field

from .base import QueryEnvelope
from .content import OptionalDate, content_result, fetch_content, query_window
from .content_types import (
    EarningsMatch,
    Filing,
    NewsArticle,
    NewsKeyword,
    news_keyword,
)
from .fundamentals_types import StockSymbol, shanghai_now
from .market_sources import require_eastmoney_stock
from .news_sources import fetch_earnings_calendar, fetch_filings, fetch_news

DOMAIN = "news"
Limit = Annotated[int, Field(ge=1, le=100, description="最多返回的记录数，范围 1–100。")]


async def search_news(
    keyword: NewsKeyword,
    start_date: OptionalDate = "",
    end_date: OptionalDate = "",
    limit: Annotated[int, Field(ge=1, le=100, description="返回条数上限，默认10。兼容1–100，来源首批最多10条，日期过滤后可能更少。")] = 10,
) -> QueryEnvelope[NewsArticle]:
    """搜索东财公司或主题新闻，在来源首批最多10条中按日期筛选，默认最近7天，按发布时间倒序返回标题、时间、媒体、链接与来源。只给end_date=2026-11-10时，起点补为2026-11-04。"""
    keyword = news_keyword(keyword)
    start, end = query_window(start_date, end_date, shanghai_now().date(), days=7)
    rows = await fetch_content(fetch_news, keyword)
    items = sorted(
        (row for row in rows if start <= row.published_at.date() <= end),
        key=lambda row: (row.published_at, row.article_id),
        reverse=True,
    )
    return content_result(NewsArticle, DOMAIN, items[:limit])


async def get_filings(symbol: StockSymbol, limit: Limit = 10) -> QueryEnvelope[Filing]:
    """查询单只沪深股票的东财公告，按披露日倒序，返回类型、可识别的报告期和详情链接。"""
    await asyncio.to_thread(require_eastmoney_stock, symbol)
    rows = await fetch_content(fetch_filings, symbol)
    items = sorted(
        rows,
        key=lambda row: (row.disclosure_date, row.filing_id),
        reverse=True,
    )
    return content_result(Filing, DOMAIN, items[:limit])


async def get_earnings_calendar(
    start_date: OptionalDate = "",
    end_date: OptionalDate = "",
) -> QueryEnvelope[EarningsMatch]:
    """按当前预约日或实际披露日命中（或关系）查询沪深财报日程，默认今天起30天。返回首次预约、变更、当前预约、实际披露日和命中类型。每种日期查询最多20页，超量须缩小范围。只给start_date=2026-11-01时，终点补为2026-11-30。"""
    start, end = query_window(start_date, end_date, shanghai_now().date(), days=30, future=True)
    rows = await fetch_content(fetch_earnings_calendar, start, end)
    matches = []
    for row in rows:
        dates = {
            "current_appointment": row.current_appointment_date,
            "actual_disclosure": row.actual_disclosure_date,
        }
        matched = {
            kind: day for kind, day in dates.items() if day is not None and start <= day <= end
        }
        if matched:
            item = EarningsMatch(**row.model_dump(), matched_date_types=list(matched))
            matches.append((min(matched.values()), item))
    matches.sort(key=lambda pair: (pair[0], pair[1].symbol, pair[1].report_date))
    return content_result(EarningsMatch, DOMAIN, [item for _, item in matches])


__all__ = ["DOMAIN", "get_earnings_calendar", "get_filings", "search_news"]
